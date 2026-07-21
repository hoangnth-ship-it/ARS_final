"""Feature cache with input-size cache-keys (Sec 8.2).  [C1 fairness / [C5]]

The cache key is a hash of ONLY the inputs that change the extracted feature
(sample_rate, n_mels, n_fft, hop_length, win_length, n_frames, band_limit...).
- change an INPUT-size parameter -> new key -> re-extract, store, share.
- change only the ARCHITECTURE -> same key -> reuse cached features.

A cached group is stored as an .npz holding X (n, ...) plus aligned utterance_id
and a JSON sidecar recording the exact spec, so any number is reproducible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("cache")

# which config fields feed the cache key, per feature group
_KEYED_FIELDS = ["sample_rate", "n_mels", "n_fft", "hop_length", "win_length",
                 "n_frames", "n_mfcc"]


def cache_key(group: str, spec: Dict) -> str:
    keyed = {k: spec[k] for k in _KEYED_FIELDS if k in spec}
    keyed["group"] = group
    keyed["band_limit"] = spec.get("band_limit", True)
    keyed["cutoff_hz"] = spec.get("cutoff_hz")
    blob = json.dumps(keyed, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _paths(cfg, group: str, key: str) -> Tuple[Path, Path]:
    d = ensure_dir(cfg["paths"]["cache_dir"])
    return d / f"{group}__{key}.npz", d / f"{group}__{key}.json"


def save(cfg, group: str, key: str, ids: List[str], X: np.ndarray, spec: Dict) -> None:
    npz, sidecar = _paths(cfg, group, key)
    np.savez_compressed(npz, X=X.astype(np.float32), ids=np.array(ids))
    sidecar.write_text(json.dumps({"group": group, "key": key,
                                   "spec": spec, "shape": list(X.shape)}, indent=2))
    LOG.info("cached %s [%s] shape=%s", group, key, X.shape)


def load(cfg, group: str, key: str):
    npz, _ = _paths(cfg, group, key)
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    return list(d["ids"]), d["X"]


def exists(cfg, group: str, key: str) -> bool:
    return _paths(cfg, group, key)[0].exists()


def feature_spec(cfg) -> Dict:
    """Assemble the cache-key spec from the feature + preprocess config."""
    f = cfg["features"]
    p = cfg["preprocess"]
    return dict(sample_rate=f["sample_rate"], n_mels=f["n_mels"], n_fft=f["n_fft"],
                hop_length=f["hop_length"], win_length=f["win_length"],
                n_frames=f["n_frames"], n_mfcc=f["n_mfcc"],
                band_limit=p["band_limit"]["enabled"], cutoff_hz=p["band_limit"]["cutoff_hz"],
                max_duration_s=p["fixed_length"]["max_duration_s"])
