"""Modality / branch ablation of the fusion model (Sec 5 ablation).  [C1]

Isolates the contribution of each branch of the proposed Whisper+BERT+hand-crafted
model by training the SAME light MLP head on each subset of cached embeddings,
under the identical LOSO protocol (3 seeds, train-only scaler):

  audio_only        : whisper_emb
  text_only         : bert_emb
  handcrafted_only  : tabular
  audio+text        : whisper_emb + bert_emb
  audio+handcrafted : whisper_emb + tabular
  full              : whisper_emb + bert_emb + tabular   (= fusion_concat features)

Writes artifacts/results/ablation_modality.csv.
"""
from __future__ import annotations

import pandas as pd

from src.eval import runner
from src.pipelines.run_deep import _emb_fit_predict
from src.utils.common import ensure_dir, get_logger, load_config, resolve

LOG = get_logger("ablation")

CONFIGS = [
    ("audio_only",        ["whisper_emb"]),
    ("text_only",         ["bert_emb"]),
    ("handcrafted_only",  ["tabular"]),
    ("audio+text",        ["whisper_emb", "bert_emb"]),
    ("audio+handcrafted", ["whisper_emb", "tabular"]),
    ("full",              ["whisper_emb", "bert_emb", "tabular"]),
]


def run(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    fp = _emb_fit_predict(cfg)
    rows = []
    for name, groups in CONFIGS:
        try:
            if len(groups) == 1:
                ids, X, y, g = runner.load_group(cfg, groups[0])
            else:
                ids, X, y, g, _ = runner.load_multi(cfg, groups)
            summary = runner.evaluate(cfg, f"abl_{name}", "+".join(groups), fp,
                                      data=(ids, X, y, g))
            rows.append(dict(
                config=name, features="+".join(groups),
                accuracy=summary.get("accuracy_mean"), acc_std=summary.get("accuracy_std"),
                auc=summary.get("auc_mean"), f1=summary.get("f1_mean"),
                sensitivity=summary.get("sensitivity_mean"),
                specificity=summary.get("specificity_mean"),
                subj_accuracy=summary.get("subj_accuracy_mean")))
        except Exception as e:
            LOG.exception("ablation %s failed: %s", name, e)
    df = pd.DataFrame(rows)
    out = ensure_dir(cfg["paths"]["results_dir"]) / "ablation_modality.csv"
    df.to_csv(out, index=False)
    LOG.info("ablation table -> %s\n%s", out, df.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    run(load_config())
