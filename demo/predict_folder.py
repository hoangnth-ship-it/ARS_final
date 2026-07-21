"""Demo: predict PD/HC for every .wav in a folder (Sec 9 deliverable).

Trains a quick LogisticRegression on the cached tabular features (or loads a saved
bundle if present), then scores new files through the SAME preprocessing + feature
extraction, so inference matches training exactly.

Usage:
  python demo/predict_folder.py --folder path/to/wavs
For a real-time mic demo, record a few seconds to a wav and pass its folder; the same
`process_waveform` + `acoustic/linguistic` path is used (a notebook wrapper can loop
on microphone chunks).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import acoustic, cache, linguistic
from src.preprocess.pipeline import process_waveform
from src.utils.common import get_logger, load_config

LOG = get_logger("demo")


def _train_reference(cfg):
    """Train a reference classifier on cached tabular features (all speakers)."""
    import pandas as pd
    from src.utils.common import resolve
    spec = cache.feature_spec(cfg)
    ids, X = cache.load(cfg, "tabular", cache.cache_key("tabular", spec))
    man = pd.read_csv(resolve(cfg["paths"]["manifest"])).set_index("utterance_id").loc[ids]
    y = man.label.values
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X, y)
    return clf


def _vector(y, sr, cfg):
    af = acoustic.extract(y, sr, cfg["features"]["n_mfcc"])
    lf = linguistic.extract("")   # no transcript at inference time
    return np.array(list(af.values()) + list(lf.values()), dtype=np.float32)


def run(folder: str, cfg=None):
    cfg = cfg or load_config()
    sr = cfg["features"]["sample_rate"]
    clf = _train_reference(cfg)
    files = sorted(Path(folder).rglob("*.wav"))
    if not files:
        LOG.warning("no wav files in %s", folder); return
    LOG.info("scoring %d files", len(files))
    for f in files:
        y = process_waveform(str(f), cfg)
        v = _vector(y, sr, cfg).reshape(1, -1)
        p = float(clf.predict_proba(v)[0, 1])
        print(f"{f.name:40s}  P(PD)={p:.3f}  -> {'PD' if p >= 0.5 else 'HC'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    run(args.folder, load_config(args.config))
