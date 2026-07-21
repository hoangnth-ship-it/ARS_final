"""Acceptance tests for the manifest (Sec 1)."""
import pandas as pd
import pytest

from src.utils.common import load_config, resolve
from src.data import build_manifest


@pytest.fixture(scope="module")
def df():
    cfg = load_config()
    path = resolve(cfg["paths"]["manifest"])
    if not path.exists():
        build_manifest.build(cfg)
    return pd.read_csv(path)


def test_counts(df):
    assert len(df) == 1091
    assert df.speaker_id.nunique() == 22


def test_no_speaker_crosses_labels(df):
    # a speaker must belong to exactly one class (leakage guard, Sec 5.1)
    per = df.groupby("speaker_id").label.nunique()
    assert (per == 1).all()


def test_class_speaker_split(df):
    assert df[df.label == 1].speaker_id.nunique() == 10   # PD
    assert df[df.label == 0].speaker_id.nunique() == 12   # HC


def test_all_wavs_exist(df):
    from pathlib import Path
    missing = [p for p in df.wav_path if not Path(p).exists()]
    assert not missing, f"{len(missing)} wav paths missing"


def test_sample_rate_confound_present(df):
    # sanity: the confound the whole project is about
    assert (df[df.label == 1].orig_sample_rate == 16000).mean() > 0.99
    assert (df[df.label == 0].orig_sample_rate == 44100).mean() > 0.99
