"""Subject-wise evaluation protocol (Sec 5).  [C5] leakage-safe.

Provides the fold generator used by EVERY model, so differences reflect the model,
not the protocol:
  - LOSO  : Leave-One-Subject-Out (default; robust for n=22, emma-dominated)
  - SGKF  : StratifiedGroupKFold(k)

Also provides a speaker-wise inner validation split for early stopping, and asserts
train / test speaker sets are disjoint.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(y: np.ndarray, groups: np.ndarray, cfg) -> List[Tuple[np.ndarray, np.ndarray]]:
    mode = cfg["eval"]["cv"]
    if mode == "loso":
        folds = _loso(groups)
    elif mode == "sgkf":
        folds = _sgkf(y, groups, cfg["eval"]["k_folds"], cfg["seed"])
    else:
        raise ValueError(f"unknown cv: {mode}")
    for tr, te in folds:                      # leakage assertion (Sec 5.1)
        assert not (set(groups[tr]) & set(groups[te])), "speaker leaked across split!"
    return folds


def _loso(groups: np.ndarray):
    folds = []
    for spk in sorted(np.unique(groups)):
        te = np.where(groups == spk)[0]
        tr = np.where(groups != spk)[0]
        folds.append((tr, te))
    return folds


def _sgkf(y, groups, k, seed):
    k = min(k, len(np.unique(groups)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def inner_val_split(tr_idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out whole speakers from the train fold for early-stopping (Sec 5.2).

    Guarantees both classes appear in the val set when possible.
    """
    rng = np.random.RandomState(seed)
    tr_spk = np.array(sorted(set(groups[tr_idx])))
    spk_label = {s: y[tr_idx][groups[tr_idx] == s][0] for s in tr_spk}
    val_spk = set()
    for lab in (0, 1):
        cands = [s for s in tr_spk if spk_label[s] == lab]
        if not cands:
            continue
        n = max(1, int(round(len(cands) * frac)))
        val_spk.update(rng.choice(cands, size=min(n, len(cands)), replace=False))
    val_mask = np.isin(groups, list(val_spk))
    inner_val = tr_idx[val_mask[tr_idx]]
    inner_tr = tr_idx[~val_mask[tr_idx]]
    if len(inner_val) == 0 or len(inner_tr) == 0:      # fallback: random 80/20
        rng.shuffle(tr_idx)
        cut = int(len(tr_idx) * (1 - frac))
        return tr_idx[:cut], tr_idx[cut:]
    return inner_tr, inner_val
