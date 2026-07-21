"""Deep families: CNN (C), SSL (D), NeuroVoz (B), Fusion (E).  Sec 4.

All share eval.runner + models.torch_train, so protocol/metrics match the ML
baselines exactly.  Whisper/BERT/Wav2Vec2 are frozen precomputed embeddings, so
these heads are light enough to train on CPU (folds/epochs configurable).
"""
from __future__ import annotations

import argparse

import yaml

from src.eval import runner
from src.models import torch_train
from src.models.nets import EmbeddingMLP, FusionNet, MelCNN
from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("deep")

RESNET_LIKE_WIDTHS = [16, 32, 64, 128]


def _cnn_fit_predict(cfg):
    n_mels = cfg["features"]["n_mels"]
    drop = cfg["train"]["dropout"]

    def fp(Xtr, ytr, Xte, meta):
        factory = lambda: MelCNN(n_mels, RESNET_LIKE_WIDTHS, pooling="attention",
                                 padding="same", dropout=drop)
        return torch_train.train_eval(factory, Xtr, ytr, Xte, meta)
    return fp


def _emb_fit_predict(cfg):
    def fp(Xtr, ytr, Xte, meta):
        in_dim = Xtr.shape[1]
        factory = lambda: EmbeddingMLP(in_dim, dropout=cfg["train"]["dropout"])
        return torch_train.train_eval(factory, Xtr, ytr, Xte, meta)
    return fp


def _fusion_fit_predict(cfg, sections, mode):
    def fp(Xtr, ytr, Xte, meta):
        factory = lambda: FusionNet(sections, mode=mode, dropout=cfg["train"]["dropout"])
        return torch_train.train_eval(factory, Xtr, ytr, Xte, meta)
    return fp


def _ensure_cnn_emb(cfg, group: str):
    """Extract frozen-backbone spectrogram embeddings on demand if not cached."""
    from src.features import cache, spec_cnn_emb
    key = cache.cache_key(group, cache.feature_spec(cfg))
    if not cache.exists(cfg, group, key):
        LOG.info("cnn_emb group '%s' missing -> extracting", group)
        spec_cnn_emb.run(cfg, only=[group])


def _load_models():
    with open(resolve("configs/models.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def run(cfg=None, only=None) -> None:
    cfg = cfg or load_config()
    for m in _load_models():
        name, fam = m["name"], m["family"]
        if only and name not in only:
            continue
        try:
            if fam == "cnn_emb":                       # frozen ImageNet backbone + MLP head
                _ensure_cnn_emb(cfg, m["feature_group"])
                fp = _emb_fit_predict(cfg)
                summary = runner.evaluate(cfg, name, m["feature_group"], fp)
            elif fam in ("cnn",) or name == "neurovoz2024":
                fp = _cnn_fit_predict(cfg)
                summary = runner.evaluate(cfg, name, "melspec", fp)
            elif fam == "ssl":
                fp = _emb_fit_predict(cfg)
                summary = runner.evaluate(cfg, name, m["feature_group"], fp)
            elif fam == "fusion":
                ids, X, y, g, sections = runner.load_multi(
                    cfg, ["whisper_emb", "bert_emb", "tabular"])
                fp = _fusion_fit_predict(cfg, sections, m["fusion"])
                summary = runner.evaluate(cfg, name, "multi", fp, data=(ids, X, y, g))
            else:
                continue
            runner.append_summary(cfg, summary)
        except FileNotFoundError as e:
            LOG.warning("skip %s: %s", name, e)
        except Exception as e:
            LOG.exception("error on %s: %s", name, e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.only)
