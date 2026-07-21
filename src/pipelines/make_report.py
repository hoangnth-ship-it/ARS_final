"""Assemble docs/RESULTS.md from result CSVs (Sec 9).  No hardcoded numbers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("report")


def _md_table(df: pd.DataFrame, cols=None, round_=3) -> str:
    if df is None or df.empty:
        return "_(not available -- run the corresponding step)_\n"
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    df = df.copy()
    for c in df.select_dtypes("number").columns:
        df[c] = df[c].round(round_)
    try:
        return df.to_markdown(index=False) + "\n"      # needs tabulate
    except Exception:
        header = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
        return "\n".join([header, sep, *rows]) + "\n"


def run(cfg=None) -> None:
    cfg = cfg or load_config()
    rdir = resolve(cfg["paths"]["results_dir"])
    docs = ensure_dir("docs")

    def rd(name):
        p = rdir / name
        return pd.read_csv(p) if p.exists() else None

    parts = ["# Results (auto-generated)\n",
             "> Regenerate: `python run.py report`. Every number traces to a CSV in "
             "`artifacts/results/`.\n",
             "\n> **Protocol note:** ML, paper, Wav2Vec2 and fusion models are "
             "evaluated at the FULL protocol (subject-wise LOSO, 3 seeds). On a "
             "CPU-only machine the mel-spectrogram CNNs (resnet18 / mobilenetv2 / "
             "neurovoz2024) are run with a reduced but still subject-wise budget "
             "(StratifiedGroupKFold k=5, 1 seed) via `run_deep_cpu.py`; regenerate "
             "them at full protocol on GPU with `colab/run_full_colab.ipynb`.\n"]

    parts.append("## Confound audit (Sec 2, [C5])\n")
    parts.append(_md_table(rd("confound_audit.csv")))
    parts.append("\n![sample rate](../artifacts/figures/confound_sample_rate.png)\n")
    parts.append("![confound audit](../artifacts/figures/confound_audit.png)\n")

    parts.append("\n## Model comparison (Sec 4, [C1][C2])\n")
    comp = rd("model_comparison.csv")
    parts.append(_md_table(comp, ["model", "feature_group", "accuracy_mean",
                                  "auc_mean", "f1_mean", "sensitivity_mean",
                                  "specificity_mean", "subj_accuracy_mean"]))
    parts.append("\n![model comparison](../artifacts/figures/model_comparison.png)\n")
    parts.append("![roc](../artifacts/figures/roc.png)\n")

    parts.append("\n## Statistical tests (Sec 6, [C1])\n")
    mc = rd("mcnemar.csv")
    parts.append("**McNemar (Holm-corrected), pooled utterances:**\n\n")
    parts.append(_md_table(mc, ["model_a", "model_b", "b", "c", "p_value",
                                "p_holm", "significant_holm"]))
    cav = rdir / "mcnemar_caveat.txt"
    if cav.exists():
        parts.append(f"\n> Caveat: {cav.read_text().strip()}\n")
    fr = rdir / "friedman.json"
    if fr.exists():
        d = json.loads(fr.read_text())
        parts.append(f"\n**Friedman omnibus:** {json.dumps(d, indent=0)[:400]}\n")

    parts.append("\n## Feature/architecture sweep (Sec 8)\n")
    parts.append(_md_table(rd("sweep_results.csv"),
                           ["variant", "cache_key", "n_mels", "n_frames", "pooling",
                            "padding", "accuracy", "auc", "params", "latency_ms",
                            "out_feature_dim", "train_time_s"]))
    parts.append("\n![pareto params](../artifacts/figures/pareto_params.png)\n")
    parts.append("![pareto latency](../artifacts/figures/pareto_latency.png)\n")
    parts.append("![sweep effects](../artifacts/figures/sweep_effects.png)\n")

    (docs / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
    LOG.info("report -> %s", docs / "RESULTS.md")


if __name__ == "__main__":
    run(load_config())
