"""Generic evaluation runner (Sec 4/5).  [C1] fair comparison.

Every model is evaluated through this one function:
  - same folds (subject-wise), same train-only scaler, same metrics.
  - >=3 seeds; per-utterance predictions saved (for McNemar/Friedman, Sec 6).

`fit_predict_fn(Xtr, ytr, Xte, meta) -> yprob_te` abstracts the model family, so ML,
paper, CNN, SSL and fusion models all reuse this loop.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.eval import metrics, protocol
from src.features import cache
from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("runner")


def load_group(cfg, group: str):
    """Load a cached feature group and align it to manifest labels/speakers."""
    spec = cache.feature_spec(cfg)
    key = cache.cache_key(group, spec)
    loaded = cache.load(cfg, group, key)
    if loaded is None:
        raise FileNotFoundError(f"feature group '{group}' (key {key}) not cached. "
                                f"Run features first.")
    ids, X = loaded
    man = pd.read_csv(resolve(cfg["paths"]["manifest"])).set_index("utterance_id")
    man = man.loc[ids]
    y = man.label.values.astype(int)
    groups = man.speaker_id.values
    if cfg["eval"].get("drop_emma", False):
        keep = groups != "PD_emma"
        X, y, groups, ids = X[keep], y[keep], groups[keep], list(np.array(ids)[keep])
    return np.asarray(ids), X, y, groups


def load_multi(cfg, groups: List[str]):
    """Load several cached groups, align by utterance_id, concat -> (ids,X,y,g,sections).

    Used by the fusion model (whisper_emb + bert_emb + tabular).
    """
    spec = cache.feature_spec(cfg)
    loaded = {}
    for g in groups:
        key = cache.cache_key(g, spec)
        r = cache.load(cfg, g, key)
        if r is None:
            raise FileNotFoundError(f"group '{g}' (key {key}) not cached.")
        ids, X = r
        loaded[g] = {i: x for i, x in zip(ids, X.reshape(len(ids), -1))}
    common = [i for i in loaded[groups[0]] if all(i in loaded[g] for g in groups)]
    common.sort()
    parts = [np.stack([loaded[g][i] for i in common]) for g in groups]
    sections = [p.shape[1] for p in parts]
    X = np.concatenate(parts, axis=1).astype(np.float32)
    man = pd.read_csv(resolve(cfg["paths"]["manifest"])).set_index("utterance_id").loc[common]
    y = man.label.values.astype(int)
    grp = man.speaker_id.values
    if cfg["eval"].get("drop_emma", False):
        keep = grp != "PD_emma"
        X, y, grp, common = X[keep], y[keep], grp[keep], list(np.array(common)[keep])
    return np.asarray(common), X, y, grp, sections


def evaluate(cfg, model_name: str, group: str,
             fit_predict_fn: Callable, seeds: List[int] = None, data=None) -> Dict:
    if data is not None:
        ids, X, y, groups = data
    else:
        ids, X, y, groups = load_group(cfg, group)
    seeds = seeds or cfg["seeds"]
    fold_rows, per_utt = [], []
    pooled_rows, pooled_subj_rows = [], []   # per-seed pooled metrics (valid AUC under LOSO)

    for seed in seeds:
        from src.utils.common import set_seed
        set_seed(seed)
        folds = protocol.make_folds(y, groups, cfg)
        seed_yt, seed_yp, seed_spk = [], [], []
        for fi, (tr, te) in enumerate(folds):
            meta = dict(cfg=cfg, seed=seed, y=y, groups=groups, tr=tr, te=te,
                        ids=ids, group=group)
            yprob = np.asarray(fit_predict_fn(X[tr], y[tr], X[te], meta), dtype=float)
            # per-fold accuracy is valid even for single-speaker folds (for Friedman)
            m = dict(accuracy=float((yprob.round() == y[te]).mean()),
                     model=model_name, seed=seed, fold=fi, n_test=len(te))
            fold_rows.append(m)
            seed_yt.extend(y[te]); seed_yp.extend(yprob); seed_spk.extend(groups[te])
            for uid, yt, yp, spk in zip(ids[te], y[te], yprob, groups[te]):
                per_utt.append(dict(model=model_name, seed=seed, fold=fi,
                                    utterance_id=uid, speaker_id=spk,
                                    y_true=int(yt), y_prob=float(yp),
                                    y_pred=int(yp >= 0.5)))
        # pooled over all out-of-fold predictions for this seed -> valid AUC/sens/spec
        pm = metrics.binary_metrics(seed_yt, seed_yp); pm["seed"] = seed
        pooled_rows.append(pm)
        psm = metrics.subject_metrics(seed_spk, seed_yt, seed_yp); psm["seed"] = seed
        pooled_subj_rows.append(psm)

    utt_summary = metrics.aggregate_folds(pooled_rows)
    subj_summary = {f"subj_{k}": v for k, v in metrics.aggregate_folds(pooled_subj_rows).items()}
    foldacc = np.array([r["accuracy"] for r in fold_rows], dtype=float)
    summary = dict(model=model_name, feature_group=group,
                   n_utt=len(ids), cv=cfg["eval"]["cv"], seeds=str(seeds),
                   fold_accuracy_mean=float(np.nanmean(foldacc)),
                   fold_accuracy_std=float(np.nanstd(foldacc)),
                   **utt_summary, **subj_summary)
    _persist(cfg, model_name, fold_rows, per_utt)
    LOG.info("%-16s acc=%.3f auc=%.3f f1=%.3f sens=%.3f spec=%.3f (subj acc=%.3f)",
             model_name, summary.get("accuracy_mean", float("nan")),
             summary.get("auc_mean", float("nan")),
             summary.get("f1_mean", float("nan")),
             summary.get("sensitivity_mean", float("nan")),
             summary.get("specificity_mean", float("nan")),
             summary.get("subj_accuracy_mean", float("nan")))
    return summary


def _persist(cfg, model_name, fold_rows, per_utt):
    rdir = ensure_dir(cfg["paths"]["results_dir"])
    pu = ensure_dir(resolve(cfg["paths"]["results_dir"]) / "per_utterance")
    pd.DataFrame(fold_rows).to_csv(rdir / f"folds_{model_name}.csv", index=False)
    pd.DataFrame(per_utt).to_csv(pu / f"preds_{model_name}.csv", index=False)


def append_summary(cfg, summary: Dict) -> None:
    """Append/update a row in the master comparison table."""
    path = resolve(cfg["paths"]["results_dir"]) / "model_comparison.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = df[df.get("model", pd.Series(dtype=str)) != summary["model"]] if len(df) else df
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    df.to_csv(path, index=False)
