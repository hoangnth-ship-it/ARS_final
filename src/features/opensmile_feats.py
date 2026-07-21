"""openSMILE eGeMAPS / ComParE features (Sec 2.2, [C2]).

Standalone, OPTIONAL: only runs if the `opensmile` package is installed. Produces an
`egemaps` cached feature group usable as a drop-in tabular feature set for any ML/paper
model (set feature_group: egemaps in configs/models.yaml). No-op with a warning if the
package is missing, so the rest of the pipeline is unaffected.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import cache
from src.preprocess.pipeline import process_waveform
from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("opensmile")


def available() -> bool:
    try:
        import opensmile  # noqa: F401
        return True
    except Exception:
        return False


def run(cfg=None, feature_set: str = "eGeMAPSv02", limit_per_spk: int = 0) -> None:
    cfg = cfg or load_config()
    if not available():
        LOG.warning("opensmile not installed -- skipping eGeMAPS extraction "
                    "(pip install opensmile). Hand-crafted acoustic features cover "
                    "the equivalent role in the default pipeline.")
        return
    import opensmile
    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, feature_set),
        feature_level=opensmile.FeatureLevel.Functionals)
    df = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    if limit_per_spk:
        df = df.groupby("speaker_id", group_keys=False).head(limit_per_spk)
    sr = cfg["features"]["sample_rate"]
    ids, rows = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="egemaps"):
        y = process_waveform(r.wav_path, cfg)
        feats = smile.process_signal(y, sr)
        rows.append(feats.values.ravel().astype(np.float32))
        ids.append(r.utterance_id)
    X = np.asarray(rows, dtype=np.float32)
    spec = cache.feature_spec(cfg)
    cache.save(cfg, "egemaps", cache.cache_key("egemaps", spec), ids, X, spec)
    LOG.info("eGeMAPS (%s) cached: %s", feature_set, X.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--feature-set", default="eGeMAPSv02")
    ap.add_argument("--limit-per-spk", type=int, default=0)
    args = ap.parse_args()
    run(load_config(args.config), args.feature_set, args.limit_per_spk)
