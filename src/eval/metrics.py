"""Uniform metrics (Sec 4).  Computed identically for every model.

Reported at UTTERANCE level and SUBJECT level (probabilities averaged per speaker).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def _safe_auc(y, p) -> float:
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        return float("nan")


def binary_metrics(y_true, y_prob, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall / sensitivity
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        sensitivity=sens,
        specificity=spec,
        precision=precision_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=_safe_auc(y_true, y_prob),
    )


def subject_metrics(speaker_ids, y_true, y_prob, thr: float = 0.5) -> Dict[str, float]:
    """Aggregate to subject level by averaging probability per speaker."""
    import pandas as pd
    d = pd.DataFrame(dict(spk=speaker_ids, y=np.asarray(y_true).astype(int),
                          p=np.asarray(y_prob, dtype=float)))
    g = d.groupby("spk").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    return binary_metrics(g.y.values, g.p.values, thr)


def aggregate_folds(rows) -> Dict[str, float]:
    """mean +/- std across folds for each metric."""
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    for c in ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]:
        if c in df:
            out[f"{c}_mean"] = float(np.nanmean(df[c]))
            out[f"{c}_std"] = float(np.nanstd(df[c]))
    return out
