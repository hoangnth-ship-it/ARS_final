"""Generate result figures (Sec 7/8.6).  All figures regenerate from result CSVs;
no numbers are hardcoded.  Architecture diagrams are emitted as Mermaid (.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("figures")


def _rd(cfg, name):
    p = resolve(cfg["paths"]["results_dir"]) / name
    return pd.read_csv(p) if p.exists() else None


def fig_sample_rate(cfg, outdir):
    man = pd.read_csv(resolve(cfg["paths"]["manifest"]))
    ct = man.groupby(["label", "orig_sample_rate"]).size().unstack(fill_value=0)
    ax = ct.T.plot(kind="bar", figsize=(6, 4))
    ax.set_xlabel("original sample rate (Hz)"); ax.set_ylabel("#utterances")
    ax.set_title("Confound: sample-rate distribution by class")
    ax.legend(["HC (0)", "PD (1)"])
    plt.tight_layout(); plt.savefig(outdir / "confound_sample_rate.png", dpi=130); plt.close()


def fig_confound_audit(cfg, outdir):
    df = _rd(cfg, "confound_audit.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(df.probe, df.accuracy, color=["#c0392b", "#e67e22", "#27ae60"])
    ax.axhline(0.5, ls="--", c="k", label="chance")
    ax.set_ylim(0, 1.05); ax.set_ylabel("grouped-CV accuracy")
    ax.set_title("Confound audit: metadata / spectral probes")
    ax.set_xticklabels(df.probe, rotation=20, ha="right", fontsize=8)
    ax.legend(); plt.tight_layout()
    plt.savefig(outdir / "confound_audit.png", dpi=130); plt.close()


def fig_model_comparison(cfg, outdir):
    df = _rd(cfg, "model_comparison.csv")
    if df is None or df.empty:
        return
    df = df.sort_values("auc_mean", ascending=False)
    x = np.arange(len(df)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.8), 4.5))
    ax.bar(x - w / 2, df.accuracy_mean, w, label="accuracy (pooled)")
    ax.bar(x + w / 2, df.auc_mean, w, label="AUC")
    ax.set_xticks(x); ax.set_xticklabels(df.model, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison (subject-wise CV)")
    ax.legend(); plt.tight_layout()
    plt.savefig(outdir / "model_comparison.png", dpi=130); plt.close()


def fig_roc(cfg, outdir):
    from sklearn.metrics import roc_curve
    pu = resolve(cfg["paths"]["results_dir"]) / "per_utterance"
    files = sorted(pu.glob("preds_*.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for f in files[:12]:
        d = pd.read_csv(f)
        d = d[d.seed == sorted(d.seed.unique())[0]]
        if d.y_true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(d.y_true, d.y_prob)
        ax.plot(fpr, tpr, lw=1, label=f.stem.replace("preds_", "")[:16])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (pooled utterances)")
    ax.legend(fontsize=6, loc="lower right"); plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=130); plt.close()


def fig_confusion(cfg, outdir):
    from sklearn.metrics import confusion_matrix
    pu = resolve(cfg["paths"]["results_dir"]) / "per_utterance"
    files = sorted(pu.glob("preds_*.csv"))
    if not files:
        return
    f = files[0]
    d = pd.read_csv(f); d = d[d.seed == sorted(d.seed.unique())[0]]
    cm = confusion_matrix(d.y_true, d.y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["HC", "PD"]); ax.set_yticklabels(["HC", "PD"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion: {f.stem.replace('preds_', '')}")
    plt.colorbar(im, fraction=0.046); plt.tight_layout()
    plt.savefig(outdir / "confusion.png", dpi=130); plt.close()


def fig_pareto(cfg, outdir):
    df = _rd(cfg, "sweep_results.csv")
    if df is None or df.empty:
        return
    for xcol, xlabel, fname in [("params", "#parameters", "pareto_params"),
                                ("latency_ms", "latency (ms/utt)", "pareto_latency")]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(df[xcol], df.accuracy, s=60)
        for _, r in df.iterrows():
            ax.annotate(r.variant, (r[xcol], r.accuracy), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(xlabel); ax.set_ylabel("accuracy")
        ax.set_title(f"Pareto: accuracy vs {xlabel}")
        plt.tight_layout(); plt.savefig(outdir / f"{fname}.png", dpi=130); plt.close()


def fig_sweep_effects(cfg, outdir):
    df = _rd(cfg, "sweep_results.csv")
    if df is None or df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df.n_mels * df.n_frames, df.accuracy)
    axes[0].set_xlabel("input size (n_mels x n_frames)"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("accuracy vs input size")
    order = df.groupby("pooling").accuracy.mean()
    axes[1].bar(order.index, order.values)
    axes[1].set_title("accuracy vs pooling"); axes[1].set_ylabel("mean accuracy")
    plt.tight_layout(); plt.savefig(outdir / "sweep_effects.png", dpi=130); plt.close()


def run(cfg=None) -> None:
    cfg = cfg or load_config()
    outdir = ensure_dir(cfg["paths"]["figures_dir"])
    for fn in [fig_sample_rate, fig_confound_audit, fig_model_comparison,
               fig_roc, fig_confusion, fig_pareto, fig_sweep_effects]:
        try:
            fn(cfg, outdir)
        except Exception as e:
            LOG.warning("%s failed: %s", fn.__name__, e)
    _write_architecture_diagrams(outdir)
    LOG.info("figures -> %s", outdir)


def _write_architecture_diagrams(outdir):
    """Sec 7 architecture diagrams as Mermaid (render in GitHub/mermaid.live)."""
    (Path(outdir) / "architecture_diagrams.md").write_text(_MERMAID, encoding="utf-8")


_MERMAID = """# Architecture diagrams (Sec 7)  [C4]

## 7.1 Overall pipeline
```mermaid
flowchart LR
  A[raw wav<br/>PD 16k / HC 44.1k] --> B[preprocess<br/>mono, 16k, band-limit 7.5k,<br/>LUFS, VAD, fix-len]
  B --> C1[log-Mel 80xT]
  B --> C2[waveform]
  B --> C3[transcript]
  C1 --> D1[Whisper encoder<br/>frozen -> attn-pool 512]
  C2 --> D2[Wav2Vec2<br/>frozen -> pool 768]
  C3 --> D3[BERT<br/>frozen -> CLS 768]
  B --> D4[14 hand-crafted<br/>jitter/shimmer/HNR/MFCC...]
  D1 --> F[Fusion: concat / cross-attn / gated]
  D3 --> F
  D4 --> F
  F --> H[MLP head + sigmoid] --> P[P(PD)]
```

## 7.2 Whisper encoder block
```mermaid
flowchart LR
  M[log-Mel 80xT] --> C[2x Conv1d + GELU] --> E[Transformer encoder blocks]
  E --> Hh[hidden 1500x512] --> AP[attention pooling] --> V[512-d vector]
```

## 7.3 Fusion strategies
```mermaid
flowchart TB
  subgraph concat
    a1[whisper]-->cc[concat]; a2[bert]-->cc; a3[handcrafted]-->cc; cc-->h1[head]
  end
  subgraph cross_attention
    b1[whisper]-->xa[MultiheadAttention]; b2[bert]-->xa; b3[handcrafted]-->xa; xa-->h2[head]
  end
  subgraph gated
    c1[whisper]-->g[softmax gate]; c2[bert]-->g; c3[handcrafted]-->g; g-->h3[head]
  end
```

## 7.5 Subject-wise CV (leakage-safe)
```mermaid
flowchart LR
  S[22 speakers] --> K{LOSO / StratifiedGroupKFold}
  K --> TR[train speakers]
  K --> TE[test speaker(s)]
  TR --> IV[inner speaker-wise val<br/>early stopping]
  TR -.scaler fit train-only.-> TE
  note[assert train ∩ test speakers = ∅]
```
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    run(load_config(args.config))
