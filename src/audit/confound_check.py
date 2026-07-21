"""Confound audit loop (Sec 2, [C5]).

Question: can a classifier separate PD/HC using ONLY nuisance signal
(sample-rate / recording-pipeline fingerprint) rather than pathology?

Three probes, all under SPEAKER-GROUPED CV (leakage-safe):
  1. metadata-only     : LR on {orig_sample_rate, orig_duration_s, orig_num_channels}
                         -> demonstrates the confound EXISTS (expected ~100%).
  2. spectral @resample : LR on cheap spectral summary of mono/16k audio (NO band-limit)
  3. spectral @bandlimit: same spectral summary AFTER low-pass band-limiting

Acceptance: probe (3) accuracy must fall to ~chance (0.5 +/- tol).  If it stays
high, the recording-pipeline confound survives band-limiting -> RED warning, and
downstream model results cannot be trusted over these baselines.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.preprocess.pipeline import band_limit, load_mono, normalize_loudness
from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("confound")


def _spectral_summary(y: np.ndarray, sr: int) -> dict:
    """Cheap spectral fingerprint features (mean/std over frames)."""
    import librosa
    if y.size < sr // 10:
        y = np.pad(y, (0, sr // 10))
    cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rol = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    out = {}
    for name, v in [("cen", cen), ("rol", rol), ("bw", bw), ("flat", flat), ("zcr", zcr)]:
        out[f"{name}_mean"] = float(np.mean(v))
        out[f"{name}_std"] = float(np.std(v))
    # high-band energy ratio (>6 kHz) -- direct SR fingerprint
    S = np.abs(librosa.stft(y, n_fft=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    hi = S[freqs > 6000].sum()
    out["hi_ratio"] = float(hi / (S.sum() + 1e-9))
    return out


def _build_feature_table(df: pd.DataFrame, cfg: dict, limit_per_spk: int = 0) -> pd.DataFrame:
    sr = cfg["preprocess"]["target_sample_rate"]
    bl = cfg["preprocess"]["band_limit"]
    rows = []
    if limit_per_spk:
        df = df.groupby("speaker_id", group_keys=False).head(limit_per_spk)
    for _, r in tqdm(df.iterrows(), total=len(df), desc="confound-feats"):
        try:
            y = load_mono(r.wav_path, sr)                      # resample only
            fa = _spectral_summary(y, sr)
            yb = band_limit(y, sr, bl["cutoff_hz"], bl["order"])
            yb = normalize_loudness(yb, sr, "rms", -23.0, 0.05)
            fb = _spectral_summary(yb, sr)
        except Exception as e:
            LOG.warning("skip %s (%s)", r.utterance_id, e)
            continue
        row = dict(utterance_id=r.utterance_id, speaker_id=r.speaker_id, label=r.label,
                   orig_sample_rate=r.orig_sample_rate, orig_duration_s=r.orig_duration_s,
                   orig_num_channels=r.orig_num_channels)
        row.update({f"res_{k}": v for k, v in fa.items()})
        row.update({f"bl_{k}": v for k, v in fb.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _grouped_acc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42) -> float:
    n_splits = min(5, len(np.unique(groups)),
                   int(np.min(np.bincount(y))))  # cannot exceed minority-class count
    n_splits = max(2, n_splits)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue  # degenerate fold (class-pure speakers) -> skip
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else float("nan")


def run(cfg=None, limit_per_spk: int = 0) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    tbl = _build_feature_table(df, cfg, limit_per_spk)

    y = tbl.label.values.astype(int)
    g = tbl.speaker_id.values

    meta_cols = cfg["confound"]["metadata_cols"]
    res_cols = [c for c in tbl.columns if c.startswith("res_")]
    bl_cols = [c for c in tbl.columns if c.startswith("bl_")]

    probes = {
        "1_metadata_only": tbl[meta_cols].values,
        "2_spectral_resample_only": tbl[res_cols].values,
        "3_spectral_band_limited": tbl[bl_cols].values,
    }
    results = []
    for name, X in probes.items():
        acc = _grouped_acc(X, y, g, cfg["seed"])
        results.append(dict(probe=name, n=len(y), accuracy=round(acc, 4)))
        LOG.info("%-26s acc=%.3f", name, acc)

    res_df = pd.DataFrame(results)
    tol = cfg["confound"]["chance_tol"]
    bl_acc = res_df.loc[res_df.probe == "3_spectral_band_limited", "accuracy"].iloc[0]
    clean = abs(bl_acc - 0.5) <= tol
    res_df["chance_tol"] = tol
    res_df["band_limited_clean"] = clean

    out = ensure_dir(cfg["paths"]["results_dir"]) / "confound_audit.csv"
    res_df.to_csv(out, index=False)
    tbl.to_csv(resolve(cfg["paths"]["results_dir"]) / "confound_features.csv", index=False)

    if clean:
        LOG.info("PASS: band-limited spectral acc=%.3f ~ chance. Confound reduced. [C5]", bl_acc)
    else:
        LOG.warning("RED: band-limited spectral acc=%.3f still >> chance (tol=%.2f). "
                    "Recording-pipeline confound survives -- trust model results only "
                    "if they beat THIS baseline.", bl_acc, tol)
    LOG.info("written: %s", out)
    return res_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--limit-per-spk", type=int, default=0,
                    help="subsample utterances per speaker for a fast audit (0=all)")
    args = ap.parse_args()
    run(load_config(args.config), args.limit_per_spk)
