"""Acceptance tests for preprocessing, protocol and cache (Sec 2/5/8)."""
import numpy as np
import pytest

from src.utils.common import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_preprocess_shapes_and_loudness(cfg):
    from src.preprocess.pipeline import process_waveform
    import pandas as pd
    from src.utils.common import resolve
    man = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    sr = cfg["preprocess"]["target_sample_rate"]
    y = process_waveform(man.iloc[0].wav_path, cfg)
    expected = int(cfg["preprocess"]["fixed_length"]["max_duration_s"] * sr)
    assert y.shape[0] == expected                 # fixed length
    assert np.isfinite(y).all()
    assert 0.01 < np.sqrt((y ** 2).mean()) < 1.0  # loudness normalized


def test_cache_key_input_vs_arch(cfg):
    from src.features import cache
    spec = cache.feature_spec(cfg)
    k1 = cache.cache_key("melspec", spec)
    # arch-only change (nothing in spec) -> same key
    assert cache.cache_key("melspec", spec) == k1
    # input change (n_mels) -> different key
    spec2 = dict(spec); spec2["n_mels"] = spec["n_mels"] + 8
    assert cache.cache_key("melspec", spec2) != k1


def test_folds_are_leakage_free(cfg):
    from src.eval import protocol
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    cfg = dict(cfg); cfg["eval"] = dict(cfg["eval"]); cfg["eval"]["cv"] = "loso"
    for tr, te in protocol.make_folds(y, groups, cfg):
        assert not (set(groups[tr]) & set(groups[te]))
