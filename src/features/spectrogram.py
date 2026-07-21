"""Log-Mel spectrogram (Sec 3.5) for CNN / NeuroVoz baselines.

Returns a fixed [n_mels, n_frames] array (padded/cropped in time), per-utterance
log-power mel.  Input-size parameters (n_mels/n_fft/hop/n_frames) are cache-keyed.
"""
from __future__ import annotations

import librosa
import numpy as np


def log_mel(y: np.ndarray, sr: int, n_mels: int, n_fft: int, hop_length: int,
            win_length: int, n_frames: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop_length, win_length=win_length,
                                       power=2.0)
    logS = librosa.power_to_db(S, ref=np.max).astype(np.float32)   # (n_mels, T)
    T = logS.shape[1]
    if T < n_frames:
        logS = np.pad(logS, ((0, 0), (0, n_frames - T)), mode="constant",
                      constant_values=logS.min())
    elif T > n_frames:
        start = (T - n_frames) // 2
        logS = logS[:, start:start + n_frames]
    # per-utterance standardize (global stats leak nothing across utterances)
    logS = (logS - logS.mean()) / (logS.std() + 1e-6)
    return logS.astype(np.float32)


def mfcc_sequence(y: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int,
                  win_length: int, n_frames: int) -> np.ndarray:
    """MFCC sequence [n_mfcc, n_frames] for Moro-2019 GMM baseline."""
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                             hop_length=hop_length, win_length=win_length).astype(np.float32)
    T = m.shape[1]
    if T < n_frames:
        m = np.pad(m, ((0, 0), (0, n_frames - T)))
    elif T > n_frames:
        s = (T - n_frames) // 2
        m = m[:, s:s + n_frames]
    return m
