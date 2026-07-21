"""Scan Data/ and build a canonical manifest.csv.  [C5]

Real layout (verified):
  PD (label=1): denoised-speech-dataset/{DL,emma,Faces,LW,Tessi}
      - DL, LW, Tessi, emma  = ONE speaker each (emma has task subfolders IC*/WP*)
      - Faces                = a COLLECTION: each subfolder (BG_au, JC_au, ...) is a
                               separate speaker.  -> 6 speakers.
      => 10 PD speakers total, matching the dataset description.
  HC (label=0): cleaned-HC-speech-dataset/{AT,BT,...,TT}  = 12 speakers, one per folder.

Speaker derivation is deliberate: getting it wrong would leak a speaker across
train/test folds (Sec 5.1).  Every wav reads its TRUE sample-rate/duration/channels
from the header (never inferred from the filename).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import soundfile as sf

from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("manifest")

# Sources whose immediate subfolders are separate speakers (a collection source).
COLLECTION_SOURCES = {"Faces"}


def _pd_variant_dir(cfg) -> str:
    return {"original": "original-speech-dataset",
            "denoised": "denoised-speech-dataset"}[cfg["paths"]["pd_variant"]]


def _hc_variant_dir(cfg) -> str:
    return {"original": "original-HC-speech-dataset",
            "cleaned": "cleaned-HC-speech-dataset"}[cfg["paths"]["hc_variant"]]


def _read_transcript(wav: Path) -> str:
    txt = wav.with_suffix(".txt")
    if txt.exists():
        try:
            return txt.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""
    return ""


def _speaker_id(label: int, source_group: str, rel_parts: List[str]) -> str:
    """rel_parts = path components under the dataset root, excluding filename."""
    if label == 0:  # HC: top folder is the speaker
        return f"HC_{source_group}"
    # PD
    if source_group in COLLECTION_SOURCES and len(rel_parts) >= 2:
        return f"{source_group}_{rel_parts[1]}"      # e.g. Faces_BG_au
    return f"PD_{source_group}"                       # DL/LW/Tessi/emma -> one speaker


def _scan(dataset_root: Path, label: int, variant: str) -> List[dict]:
    rows = []
    if not dataset_root.exists():
        LOG.warning("dataset root missing: %s", dataset_root)
        return rows
    for wav in sorted(dataset_root.rglob("*.wav")):
        rel = wav.relative_to(dataset_root)
        parts = list(rel.parts)                       # [source, (sub), ..., file.wav]
        source_group = parts[0]
        speaker = _speaker_id(label, source_group, parts[:-1])
        try:
            info = sf.info(str(wav))
            sr, dur, ch = info.samplerate, float(info.duration), info.channels
        except Exception as e:                         # corrupt file -> skip, log
            LOG.warning("unreadable %s (%s)", wav, e)
            continue
        rows.append(dict(
            utterance_id=f"{speaker}__{wav.stem}",
            speaker_id=speaker,
            label=label,
            source_group=source_group,
            wav_path=str(wav.resolve()),
            txt_transcript=_read_transcript(wav),
            orig_sample_rate=sr,
            orig_duration_s=round(dur, 4),
            orig_num_channels=ch,
            dataset_variant=variant,
        ))
    return rows


def build(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    data_root = resolve(cfg["paths"]["data_root"])
    pd_dir = data_root / _pd_variant_dir(cfg)
    hc_dir = data_root / _hc_variant_dir(cfg)

    rows = _scan(pd_dir, 1, cfg["paths"]["pd_variant"]) + \
           _scan(hc_dir, 0, cfg["paths"]["hc_variant"])
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No wavs found -- check paths.data_root in config.")

    # de-dup utterance_id defensively
    if df["utterance_id"].duplicated().any():
        dupes = df["utterance_id"].duplicated(keep=False).sum()
        LOG.warning("%d duplicated utterance_ids -> disambiguating with index", dupes)
        df["utterance_id"] = df["utterance_id"] + "__" + df.groupby("utterance_id").cumcount().astype(str)

    out = resolve(cfg["paths"]["manifest"])
    ensure_dir(out.parent)
    df.to_csv(out, index=False, encoding="utf-8")

    _summary(df)
    LOG.info("manifest written: %s (%d rows)", out, len(df))
    return df


def _summary(df: pd.DataFrame) -> None:
    LOG.info("utterances: %d | speakers: %d | PD spk: %d | HC spk: %d",
             len(df), df.speaker_id.nunique(),
             df[df.label == 1].speaker_id.nunique(),
             df[df.label == 0].speaker_id.nunique())
    LOG.info("class balance (utt): PD=%d HC=%d",
             int((df.label == 1).sum()), int((df.label == 0).sum()))
    # emma dominance check (Sec 1)
    top = df[df.label == 1].speaker_id.value_counts()
    if len(top):
        frac = top.iloc[0] / (df.label == 1).sum()
        LOG.info("largest PD speaker: %s = %.0f%% of PD utterances",
                 top.index[0], 100 * frac)
    # confound signal: SR distribution per class
    LOG.info("sample-rate by class:\n%s",
             df.groupby("label").orig_sample_rate.value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    build(load_config(args.config))
