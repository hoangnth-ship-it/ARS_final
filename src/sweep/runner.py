"""Feature/architecture size ablation sweep (Sec 8/9).  [C1 extended]

For each variant in configs/sweep.yaml:
  1. build an input spec -> cache-key.  Variants sharing an input cache-key REUSE
     the extracted mel features (only re-extract when input size changes).  [Sec 8.2]
  2. build a MelCNN from the arch spec (config-driven, not hardcoded).      [Sec 8.1]
  3. evaluate under the shared protocol -> quality metrics.                 [Sec 8.3]
  4. profile params / FLOPs / latency / memory / output size.              [Sec 8.3/8.4]

Outputs a summary table (CSV) with cache-key + spec per row for 1-command
reproduction, plus data used by figures for Pareto fronts.               [Sec 8.6/8.7]
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
import yaml

from src.eval import runner as evalrunner
from src.features import cache, extract
from src.models import torch_train
from src.models.nets import MelCNN
from src.sweep import profiler
from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("sweep")


def _apply_input(cfg, inp: dict):
    c = copy.deepcopy(cfg)
    for k, v in inp.items():
        c["features"][k] = v
    return c


def _ensure_melspec(cfg_v, df):
    """Extract melspec for this input group if its cache-key is new (Sec 8.2)."""
    spec = cache.feature_spec(cfg_v)
    key = cache.cache_key("melspec", spec)
    if cache.exists(cfg_v, "melspec", key):
        LOG.info("reuse melspec cache [%s] (n_mels=%s n_frames=%s)",
                 key, spec["n_mels"], spec["n_frames"])
    else:
        LOG.info("extract melspec [%s] (n_mels=%s n_frames=%s)",
                 key, spec["n_mels"], spec["n_frames"])
        extract.build_light(cfg_v, df, ["melspec"])
    return key


def run(cfg=None, limit_per_spk=6, epochs=6, k_folds=3, seeds=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    with open(resolve("configs/sweep.yaml"), encoding="utf-8") as f:
        sweep = yaml.safe_load(f)["sweep"]
    seeds = seeds or sweep["shared"].get("seeds", cfg["seeds"])[:1]

    man = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    if limit_per_spk:
        man = man.groupby("speaker_id", group_keys=False).head(limit_per_spk).reset_index(drop=True)

    rows = []
    for v in sweep["variants"]:
        name, inp, arch = v["name"], v["input"], v["arch"]
        cfg_v = _apply_input(cfg, inp)
        cfg_v["eval"]["cv"] = "sgkf"; cfg_v["eval"]["k_folds"] = k_folds
        cfg_v["train"]["epochs"] = epochs
        key = _ensure_melspec(cfg_v, man)

        n_mels = cfg_v["features"]["n_mels"]; n_frames = cfg_v["features"]["n_frames"]

        def factory(a=arch, nm=n_mels):
            return MelCNN(nm, a["widths"], pooling=a["pooling"],
                          padding=a["padding"], dropout=a.get("dropout", 0.3))

        def fp(Xtr, ytr, Xte, meta, fac=factory):
            return torch_train.train_eval(fac, Xtr, ytr, Xte, meta)

        import time
        t0 = time.perf_counter()
        summary = evalrunner.evaluate(cfg_v, f"sweep_{name}", "melspec", fp, seeds=seeds)
        train_time = time.perf_counter() - t0

        # profiling on one model instance
        model = factory()
        params = profiler.count_params(model)
        flops = profiler.count_flops(model, (n_mels, n_frames))
        latency = profiler.inference_latency_ms(model, (n_mels, n_frames))
        out_dim = model.cin

        rows.append(dict(
            variant=name, cache_key=key, n_mels=n_mels, n_frames=n_frames,
            pooling=arch["pooling"], padding=arch["padding"], blocks=arch["blocks"],
            accuracy=summary.get("accuracy_mean"), auc=summary.get("auc_mean"),
            f1=summary.get("f1_mean"), sensitivity=summary.get("sensitivity_mean"),
            specificity=summary.get("specificity_mean"),
            subj_accuracy=summary.get("subj_accuracy_mean"),
            params=params, macs=flops, latency_ms=latency, out_feature_dim=out_dim,
            train_time_s=round(train_time, 2), device=profiler.device_name(),
            input_spec=json.dumps(inp), arch_spec=json.dumps(arch),
        ))
        LOG.info("%-24s acc=%.3f params=%d macs=%.2e lat=%.2fms",
                 name, rows[-1]["accuracy"] or float("nan"), params, flops, latency)

    df = pd.DataFrame(rows)
    out = ensure_dir(cfg["paths"]["results_dir"]) / "sweep_results.csv"
    df.to_csv(out, index=False)
    LOG.info("sweep table -> %s", out)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--limit-per-spk", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--k-folds", type=int, default=3)
    args = ap.parse_args()
    run(load_config(args.config), args.limit_per_spk, args.epochs, args.k_folds)
