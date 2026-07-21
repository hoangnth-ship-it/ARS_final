"""Hand-crafted acoustic features (Sec 3.1) + dysphonia subset for paper baselines.

parselmouth/Praat -> F0, jitter, shimmer, HNR (classic dysphonia measures).
librosa            -> MFCC(+delta) stats, speech rate proxy, pause ratio, intensity.

Returns an ordered dict per utterance.  Names are stable so the tabular matrix is
column-aligned across utterances.
"""
from __future__ import annotations

from collections import OrderedDict

import librosa
import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
    _HAVE_PM = True
except Exception:
    _HAVE_PM = False


def _praat_dysphonia(y: np.ndarray, sr: int) -> OrderedDict:
    out = OrderedDict()
    if not _HAVE_PM:
        for k in ["f0_mean", "f0_std", "jitter_local", "jitter_rap", "shimmer_local",
                  "shimmer_apq11", "hnr_mean"]:
            out[k] = 0.0
        return out
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0 = pitch.selected_array["frequency"]
        f0v = f0[f0 > 0]
        out["f0_mean"] = float(np.mean(f0v)) if f0v.size else 0.0
        out["f0_std"] = float(np.std(f0v)) if f0v.size else 0.0
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        out["jitter_local"] = float(call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3))
        out["jitter_rap"] = float(call(pp, "Get jitter (rap)", 0, 0, 1e-4, 0.02, 1.3))
        out["shimmer_local"] = float(call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6))
        out["shimmer_apq11"] = float(call([snd, pp], "Get shimmer (apq11)", 0, 0, 1e-4, 0.02, 1.3, 1.6))
        harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        hv = harm.values[harm.values != -200]
        out["hnr_mean"] = float(np.mean(hv)) if hv.size else 0.0
    except Exception:
        for k in ["f0_mean", "f0_std", "jitter_local", "jitter_rap", "shimmer_local",
                  "shimmer_apq11", "hnr_mean"]:
            out.setdefault(k, 0.0)
    return OrderedDict((k, (v if np.isfinite(v) else 0.0)) for k, v in out.items())


def _librosa_features(y: np.ndarray, sr: int, n_mfcc: int = 40) -> OrderedDict:
    out = OrderedDict()
    # intensity
    rms = librosa.feature.rms(y=y)[0]
    out["intensity_mean"] = float(np.mean(rms))
    out["intensity_std"] = float(np.std(rms))
    # pause ratio + speech-rate proxy (via energy VAD)
    intervals = librosa.effects.split(y, top_db=30)
    voiced = int(sum(e - s for s, e in intervals))
    total = max(len(y), 1)
    out["pause_ratio"] = float(1.0 - voiced / total)
    dur_s = total / sr
    out["speech_rate_proxy"] = float(len(intervals) / dur_s) if dur_s > 0 else 0.0
    # MFCC + delta stats
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    for name, M in [("mfcc", mfcc), ("dmfcc", d1)]:
        out[f"{name}_mean"] = float(np.mean(M))
        out[f"{name}_std"] = float(np.std(M))
        out[f"{name}_skew"] = float(_skew(M))
        out[f"{name}_kurt"] = float(_kurt(M))
    # per-coeff MFCC means (compact but informative)
    for i in range(min(13, n_mfcc)):
        out[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))
    return OrderedDict((k, (v if np.isfinite(v) else 0.0)) for k, v in out.items())


def _skew(M):
    x = M.ravel(); m = x.mean(); s = x.std() + 1e-9
    return np.mean(((x - m) / s) ** 3)


def _kurt(M):
    x = M.ravel(); m = x.mean(); s = x.std() + 1e-9
    return np.mean(((x - m) / s) ** 4) - 3.0


def extract(y: np.ndarray, sr: int, n_mfcc: int = 40) -> OrderedDict:
    """Full hand-crafted acoustic vector."""
    out = OrderedDict()
    out.update(_praat_dysphonia(y, sr))
    out.update(_librosa_features(y, sr, n_mfcc))
    return out


def dysphonia_subset(feat: OrderedDict) -> OrderedDict:
    """Little(2009)/Tsanas(2012)-style dysphonia measures subset."""
    keys = ["f0_mean", "f0_std", "jitter_local", "jitter_rap", "shimmer_local",
            "shimmer_apq11", "hnr_mean"]
    return OrderedDict((k, feat.get(k, 0.0)) for k in keys)
