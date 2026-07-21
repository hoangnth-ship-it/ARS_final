"""Statistical model comparison (Sec 6).  [C1]

  - Pairwise McNemar on pooled utterances + Holm-Bonferroni correction.
    CAVEAT (printed & saved): utterances from one speaker are NOT independent, so
    McNemar is anti-conservative here.
  - Friedman omnibus on per-fold accuracy + post-hoc Nemenyi + critical-difference
    data.  With few folds power is low (also flagged).

Consumes the per-utterance prediction files written by eval.runner.
"""
from __future__ import annotations

import argparse
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, friedmanchisquare, rankdata

from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("stats")


def _load_preds(cfg) -> pd.DataFrame:
    pu = resolve(cfg["paths"]["results_dir"]) / "per_utterance"
    frames = [pd.read_csv(f) for f in glob.glob(str(pu / "preds_*.csv"))]
    if not frames:
        raise FileNotFoundError("no per-utterance predictions; run models first.")
    df = pd.concat(frames, ignore_index=True)
    # keep only the canonical model comparison: exclude ablation/sweep helper runs
    df = df[~df.model.str.startswith(("abl_", "sweep_"))]
    # use a single reference seed for a clean paired table
    seed = sorted(df.seed.unique())[0]
    return df[df.seed == seed]


def mcnemar_pair(a_correct: np.ndarray, b_correct: np.ndarray):
    """Return (statistic, p) for McNemar with continuity correction."""
    b = int(((a_correct == 1) & (b_correct == 0)).sum())   # a right, b wrong
    c = int(((a_correct == 0) & (b_correct == 1)).sum())   # a wrong, b right
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(chi2.sf(stat, df=1))
    return float(stat), p, b, c


def holm_bonferroni(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        prev = max(prev, min(val, 1.0))
        adj[idx] = prev
    return adj


def run_mcnemar(cfg, df: pd.DataFrame) -> pd.DataFrame:
    models = sorted(df.model.unique())
    # align on common utterance_ids
    piv = df.pivot_table(index="utterance_id", columns="model",
                         values="y_pred", aggfunc="first")
    truth = df.groupby("utterance_id").y_true.first()
    piv = piv.dropna()
    truth = truth.loc[piv.index]
    correct = {m: (piv[m].values == truth.values).astype(int) for m in models if m in piv}
    rows = []
    pairs = list(combinations([m for m in models if m in correct], 2))
    for a, b in pairs:
        stat, p, nb, nc = mcnemar_pair(correct[a], correct[b])
        rows.append(dict(model_a=a, model_b=b, b=nb, c=nc, statistic=stat, p_value=p))
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm_bonferroni(res.p_value.values)
        res["significant_holm"] = res.p_holm < 0.05
    res.attrs["caveat"] = ("Utterances from the same speaker are not independent; "
                           "McNemar is anti-conservative here (Sec 6.2).")
    return res


def run_friedman(cfg, fold_dir: Path) -> dict:
    """Friedman on per-fold accuracy (models x folds) + Nemenyi CD."""
    frames = {}
    for f in glob.glob(str(fold_dir / "folds_*.csv")):
        name = Path(f).stem.replace("folds_", "")
        if name.startswith(("abl_", "sweep_")):   # canonical comparison only
            continue
        d = pd.read_csv(f)
        d = d[d.seed == sorted(d.seed.unique())[0]]
        frames[name] = d.set_index("fold").accuracy
    if len(frames) < 3:
        return dict(error="need >=3 models for Friedman", n_models=len(frames))
    mat = pd.DataFrame(frames).dropna()
    if len(mat) < 2:
        return dict(error="need >=2 folds", n_folds=len(mat))
    stat, p = friedmanchisquare(*[mat[c].values for c in mat.columns])
    # average ranks (lower acc -> higher rank number); rank so best = rank 1
    ranks = mat.apply(lambda r: rankdata(-r.values), axis=1)
    avg_rank = pd.DataFrame(list(ranks), columns=mat.columns).mean().to_dict()
    k, n = mat.shape[1], mat.shape[0]
    q_alpha = 3.314  # Nemenyi critical value, alpha=0.05, k up to ~15 (approx)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
    return dict(statistic=float(stat), p_value=float(p), n_folds=int(n),
                n_models=int(k), avg_ranks=avg_rank, critical_difference=float(cd),
                low_power=(n < 10))


def run(cfg=None) -> None:
    cfg = cfg or load_config()
    rdir = ensure_dir(cfg["paths"]["results_dir"])
    df = _load_preds(cfg)

    mc = run_mcnemar(cfg, df)
    mc.to_csv(rdir / "mcnemar.csv", index=False)
    (Path(rdir) / "mcnemar_caveat.txt").write_text(mc.attrs.get("caveat", ""))
    LOG.info("McNemar: %d pairs, %d significant after Holm", len(mc),
             int(mc.significant_holm.sum()) if "significant_holm" in mc else 0)

    fr = run_friedman(cfg, Path(rdir))
    import json
    (Path(rdir) / "friedman.json").write_text(json.dumps(fr, indent=2))
    if "error" not in fr:
        LOG.info("Friedman chi2=%.3f p=%.4g CD=%.3f (folds=%d, low_power=%s)",
                 fr["statistic"], fr["p_value"], fr["critical_difference"],
                 fr["n_folds"], fr["low_power"])
    else:
        LOG.warning("Friedman: %s", fr["error"])
    LOG.info("stats written to %s", rdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    run(load_config(args.config))
