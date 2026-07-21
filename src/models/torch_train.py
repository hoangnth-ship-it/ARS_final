"""Shared PyTorch training loop (Sec 5).  [C5]

- speaker-wise inner validation for early stopping on val AUC
- class-weighted BCE
- AdamW with two LR groups (head vs pretrained) -- here all heads, single group,
  but the hook is exposed for future fine-tuning.
Returns test-set probabilities so it plugs straight into eval.runner.
"""
from __future__ import annotations

import numpy as np

from src.eval import protocol
from src.utils.common import get_logger, pick_device

LOG = get_logger("torch_train")


def train_eval(model_factory, Xtr, ytr, Xte, meta, epochs=None, batch_size=None,
               lr=None, weight_decay=None, patience=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = meta["cfg"]
    tcfg = cfg["train"]
    device = pick_device(tcfg["device"])
    epochs = epochs or tcfg["epochs"]
    batch_size = batch_size or tcfg["batch_size"]
    lr = lr or tcfg["lr_head"]
    weight_decay = weight_decay or tcfg["weight_decay"]
    patience = patience or tcfg["patience"]

    # inner speaker-wise val split from the current train fold (Sec 5.2)
    y_all, groups_all, tr_idx = meta["y"], meta["groups"], meta["tr"]
    inner_tr_g, inner_val_g = protocol.inner_val_split(
        tr_idx, y_all, groups_all, cfg["eval"]["inner_val_frac"], meta["seed"])
    # map global indices -> local positions within the train fold
    pos = {gi: i for i, gi in enumerate(tr_idx)}
    itr = np.array([pos[i] for i in inner_tr_g])
    ival = np.array([pos[i] for i in inner_val_g])

    def to_t(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32)

    Xtr_t = to_t(Xtr)
    ytr_t = to_t(ytr)
    Xte_t = to_t(Xte).to(device)

    model = model_factory().to(device)
    # class weights
    n1 = max(1, int(ytr[itr].sum())); n0 = max(1, len(itr) - n1)
    pos_weight = torch.tensor([n0 / n1], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(Xtr_t[itr], ytr_t[itr])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    Xval = Xtr_t[ival].to(device); yval = ytr[ival]

    best_auc, best_state, wait = -1.0, None, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb).squeeze(-1)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xval).squeeze(-1)).cpu().numpy()
        try:
            auc = roc_auc_score(yval, vp) if len(np.unique(yval)) > 1 else 0.5
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_state, wait = auc, {k: v.detach().clone()
                                               for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xte_t).squeeze(-1)).cpu().numpy()
    return prob.astype(float)
