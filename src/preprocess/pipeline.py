"""Uniform audio preprocessing (Sec 2).  [C5]

Steps (each toggleable via config so the confound audit can sweep them):
  1. downmix mono + resample to target SR
  2. band-limit (low-pass BOTH classes) -> erase sample-rate fingerprint  <-- key step
  3. loudness normalization (LUFS via pyloudnorm, RMS fallback)
  4. VAD trim (energy via librosa, webrtc optional) + fixed length pad/center-crop
  5. transcript cleaning (keeps fillers)

Graceful degradation: pyloudnorm / webrtcvad / torchaudio are optional; librosa
fallbacks keep the pipeline runnable everywhere.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("preprocess")

try:
    import pyloudnorm as pyln
    _HAVE_PYLN = True
except Exception:
    _HAVE_PYLN = False

try:
    import webrtcvad
    _HAVE_WEBRTC = True
except Exception:
    _HAVE_WEBRTC = False


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
# soxr_hq is high-quality and needs no resampy; fall back if soxr is unavailable.
try:
    import soxr  # noqa: F401
    _RES_TYPE = "soxr_hq"
except Exception:
    _RES_TYPE = "polyphase"


def load_mono(path: str, target_sr: int) -> np.ndarray:
    """Load, downmix to mono, resample.  Uses soundfile+librosa (no torchaudio)."""
    y, sr = sf.read(path, always_2d=True)          # (n, ch)
    y = y.mean(axis=1).astype(np.float32)          # downmix
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type=_RES_TYPE)
    return y.astype(np.float32)


def band_limit(y: np.ndarray, sr: int, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase low-pass. Applied to BOTH classes to remove the SR fingerprint."""
    nyq = sr / 2.0
    wc = min(cutoff_hz, nyq * 0.99) / nyq
    sos = butter(order, wc, btype="low", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def normalize_loudness(y: np.ndarray, sr: int, method: str,
                       target_lufs: float, target_rms: float) -> np.ndarray:
    if method == "lufs" and _HAVE_PYLN:
        meter = pyln.Meter(sr)
        try:
            loud = meter.integrated_loudness(y)
            if np.isfinite(loud):
                return pyln.normalize.loudness(y, loud, target_lufs).astype(np.float32)
        except Exception:
            pass
    # RMS fallback
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    return (y * (target_rms / rms)).astype(np.float32)


def _energy_vad_trim(y: np.ndarray, top_db: int) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if yt.size > 0 else y


def _webrtc_vad_trim(y: np.ndarray, sr: int, frame_ms: int, aggr: int) -> np.ndarray:
    vad = webrtcvad.Vad(aggr)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
    n = int(sr * frame_ms / 1000)
    step = n * 2  # bytes per frame (int16)
    voiced = []
    for i in range(0, len(pcm) - step, step):
        frame = pcm[i:i + step]
        try:
            if vad.is_speech(frame, sr):
                voiced.append(i // 2)
        except Exception:
            break
    if not voiced:
        return y
    return y[voiced[0]: voiced[-1] + n]


def vad_trim(y: np.ndarray, sr: int, cfg_vad: dict) -> np.ndarray:
    if cfg_vad["method"] == "webrtc" and _HAVE_WEBRTC and sr in (8000, 16000, 32000, 48000):
        return _webrtc_vad_trim(y, sr, cfg_vad["frame_ms"], cfg_vad["aggressiveness"])
    return _energy_vad_trim(y, cfg_vad["top_db"])


def fix_length(y: np.ndarray, sr: int, max_dur: float, mode: str) -> np.ndarray:
    target = int(round(max_dur * sr))
    if len(y) == target:
        return y
    if len(y) > target:                            # center / end crop
        if mode == "center":
            start = (len(y) - target) // 2
            return y[start:start + target]
        return y[:target]
    pad = target - len(y)                          # pad
    if mode == "center":
        left = pad // 2
        return np.pad(y, (left, pad - left))
    return np.pad(y, (0, pad))


# --------------------------------------------------------------------------- #
# Transcript cleaning (Sec 2.5) -- keep fillers, no stopword removal
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def clean_transcript(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9'\s]", " ", t)            # keep apostrophes; drop punctuation
    return _WS.sub(" ", t).strip()


# --------------------------------------------------------------------------- #
# Full pipeline for a single waveform
# --------------------------------------------------------------------------- #
def process_waveform(path: str, cfg: dict) -> np.ndarray:
    p = cfg["preprocess"]
    sr = p["target_sample_rate"]
    y = load_mono(path, sr)
    if p["band_limit"]["enabled"]:
        y = band_limit(y, sr, p["band_limit"]["cutoff_hz"], p["band_limit"]["order"])
    if p["loudness"]["enabled"]:
        y = normalize_loudness(y, sr, p["loudness"]["method"],
                               p["loudness"]["target_lufs"], p["loudness"]["target_rms"])
    if p["vad"]["enabled"]:
        y = vad_trim(y, sr, p["vad"])
    if p["fixed_length"]["enabled"]:
        y = fix_length(y, sr, p["fixed_length"]["max_duration_s"], p["fixed_length"]["pad_mode"])
    return y.astype(np.float32)


def run(cfg=None, limit: int = 0) -> None:
    """Preprocess every utterance in the manifest, cache .npy to disk."""
    cfg = cfg or load_config()
    df = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    if limit:
        df = df.head(limit)
    out_dir = ensure_dir(cfg["paths"]["preprocessed_dir"])
    sr = cfg["preprocess"]["target_sample_rate"]
    LOG.info("preprocessing %d utterances -> %s (pyloudnorm=%s webrtc=%s)",
             len(df), out_dir, _HAVE_PYLN, _HAVE_WEBRTC)
    for _, r in tqdm(df.iterrows(), total=len(df), desc="preprocess"):
        dst = out_dir / f"{r.utterance_id}.npy"
        if dst.exists():
            continue
        try:
            y = process_waveform(r.wav_path, cfg)
            np.save(dst, y)
        except Exception as e:
            LOG.warning("failed %s (%s)", r.utterance_id, e)
    LOG.info("done. sample rate=%d, fixed len=%.1fs", sr,
             cfg["preprocess"]["fixed_length"]["max_duration_s"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(load_config(args.config), limit=args.limit)
