"""Run ML (A) + paper (B) baselines under the shared protocol (Sec 4).  [C1][C2]

Deep families (CNN/SSL/fusion) live in run_deep.py; this driver covers everything
that is a sklearn-style fit_predict on a cached feature group.
"""
from __future__ import annotations

import argparse

import yaml

from src.eval import runner
from src.models import ml_baselines
from src.models.paper_baselines import REGISTRY as PAPER
from src.utils.common import get_logger, load_config, resolve

LOG = get_logger("baselines")


def _load_models(cfg):
    with open(resolve("configs/models.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def run(cfg=None, only=None) -> None:
    cfg = cfg or load_config()
    models = _load_models(cfg)
    for m in models:
        name, fam = m["name"], m["family"]
        if only and name not in only:
            continue
        try:
            if fam == "ml":
                fp = ml_baselines.make_fit_predict(m["estimator"], cfg["eval"]["class_weight"])
            elif fam == "paper" and name in PAPER:
                fp = PAPER[name]
            else:
                continue  # cnn/ssl/fusion/neurovoz handled elsewhere
            summary = runner.evaluate(cfg, name, m["feature_group"], fp)
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
