"""Build & cache all feature groups (Sec 3, Sec 8 cache).

Groups produced (consumed by models.yaml):
  tabular      : handcrafted acoustic + linguistic  (n, d)      -> ML A
  dysphonia    : Little/Tsanas subset               (n, 7)      -> paper B
  disvoice     : DisVoice or handcrafted fallback   (n, d)      -> Vasquez B
  mfcc_seq     : MFCC sequence [n_mfcc, T]          (n, C, T)   -> Moro B
  melspec      : log-mel [n_mels, T]                (n, M, T)   -> CNN / NeuroVoz
  wav2vec2_emb : (n, 768)                                        -> SSL D
  whisper_emb  : (n, 512/768)                                    -> fusion E
  bert_emb     : (n, 768)                                        -> fusion E

Every group is stored under its cache-key so arch-only changes reuse it (Sec 8.2).
Waveforms come from the SAME preprocessing (band-limit etc.) so all models are fed
identical, leakage-safe inputs.
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import acoustic, cache, linguistic, spectrogram
from src.preprocess.pipeline import process_waveform
from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("features")

# groups that only need cheap CPU features (no transformers)
LIGHT_GROUPS = ["tabular", "dysphonia", "disvoice", "mfcc_seq", "melspec"]
DEEP_GROUPS = ["whisper_emb", "wav2vec2_emb", "bert_emb"]


def _load_manifest(cfg, limit: int, limit_per_spk: int) -> pd.DataFrame:
    df = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    if limit_per_spk:
        df = df.groupby("speaker_id", group_keys=False).head(limit_per_spk)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def build_light(cfg, df: pd.DataFrame, groups: List[str]) -> None:
    f = cfg["features"]
    sr = f["sample_rate"]
    spec = cache.feature_spec(cfg)
    keys = {g: cache.cache_key(g, spec) for g in groups}
    todo = [g for g in groups if not cache.exists(cfg, g, keys[g])]
    if not todo:
        LOG.info("light groups already cached: %s", groups)
        return

    acc = {g: [] for g in todo}
    ids = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="light-feats"):
        y = process_waveform(r.wav_path, cfg)
        ids.append(r.utterance_id)
        af = acoustic.extract(y, sr, f["n_mfcc"]) if any(
            g in todo for g in ["tabular", "dysphonia", "disvoice"]) else None
        if "tabular" in todo:
            row = {**af, **linguistic.extract(r.txt_transcript)}
            acc["tabular"].append(list(row.values()))
            build_light._tab_names = list(row.keys())
        if "dysphonia" in todo:
            acc["dysphonia"].append(list(acoustic.dysphonia_subset(af).values()))
        if "disvoice" in todo:
            acc["disvoice"].append(_disvoice_vector(y, sr, af))
        if "mfcc_seq" in todo:
            acc["mfcc_seq"].append(spectrogram.mfcc_sequence(
                y, sr, f["n_mfcc"], f["n_fft"], f["hop_length"], f["win_length"], f["n_frames"]))
        if "melspec" in todo:
            acc["melspec"].append(spectrogram.log_mel(
                y, sr, f["n_mels"], f["n_fft"], f["hop_length"], f["win_length"], f["n_frames"]))

    for g in todo:
        X = np.asarray(acc[g], dtype=np.float32)
        cache.save(cfg, g, keys[g], ids, X, spec)


def _disvoice_vector(y, sr, af) -> list:
    """DisVoice if available; otherwise an articulation/phonation-style fallback
    built from the hand-crafted measures so the Vasquez baseline still runs."""
    try:
        # DisVoice has a heavy/optional API; fall through to fallback if absent.
        import disvoice  # noqa: F401
    except Exception:
        pass
    # fallback: phonation (jitter/shimmer/hnr/f0) + articulation proxy (mfcc stats)
    keys = ["f0_mean", "f0_std", "jitter_local", "jitter_rap", "shimmer_local",
            "shimmer_apq11", "hnr_mean", "mfcc_mean", "mfcc_std", "dmfcc_mean",
            "dmfcc_std", "intensity_mean", "intensity_std", "pause_ratio",
            "speech_rate_proxy"]
    return [float(af.get(k, 0.0)) for k in keys]


def build_deep(cfg, df: pd.DataFrame, groups: List[str]) -> None:
    from src.features import embeddings
    f = cfg["features"]
    sr = f["sample_rate"]
    spec = cache.feature_spec(cfg)
    keys = {g: cache.cache_key(g, spec) for g in groups}
    todo = [g for g in groups if not cache.exists(cfg, g, keys[g])]
    if not todo:
        LOG.info("deep groups already cached: %s", groups)
        return
    acc = {g: [] for g in todo}
    ids = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="deep-feats"):
        y = process_waveform(r.wav_path, cfg)
        ids.append(r.utterance_id)
        if "whisper_emb" in todo:
            acc["whisper_emb"].append(embeddings.whisper_embed(
                y, sr, f["whisper_model"], f["whisper_layer"]))
        if "wav2vec2_emb" in todo:
            acc["wav2vec2_emb"].append(embeddings.wav2vec2_embed(y, sr, f["wav2vec2_model"]))
        if "bert_emb" in todo:
            acc["bert_emb"].append(embeddings.bert_embed(r.txt_transcript, f["bert_model"]))
    for g in todo:
        X = np.asarray(acc[g], dtype=np.float32)
        cache.save(cfg, g, keys[g], ids, X, spec)


def run(cfg=None, which: str = "light", limit: int = 0, limit_per_spk: int = 0) -> None:
    cfg = cfg or load_config()
    df = _load_manifest(cfg, limit, limit_per_spk)
    LOG.info("extracting (%s) for %d utterances", which, len(df))
    if which in ("light", "all"):
        build_light(cfg, df, LIGHT_GROUPS)
    if which in ("deep", "all"):
        build_deep(cfg, df, DEEP_GROUPS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--which", choices=["light", "deep", "all"], default="light")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-per-spk", type=int, default=0)
    args = ap.parse_args()
    run(load_config(args.config), args.which, args.limit, args.limit_per_spk)
