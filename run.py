"""One-command driver (Sec 9 deliverable). Cross-platform (no `make` needed).

Usage:
  python run.py manifest        # build manifest.csv (+ prints summary)
  python run.py confound [--fast]   # confound audit loop  [C5]
  python run.py features [--which light|deep|all] [--limit-per-spk N]
  python run.py baselines       # ML (A) + paper (B) models   [C1][C2]
  python run.py deep [--only ...]   # CNN / SSL / fusion (C/D/E)
  python run.py stats           # McNemar + Friedman          [C1]
  python run.py sweep           # feature/arch size ablation  [Sec 8]
  python run.py figures         # all figures                 [C4]
  python run.py report          # docs/RESULTS.md
  python run.py all [--fast]    # full pipeline end-to-end

--fast subsamples (per-speaker) and shortens training so the whole pipeline runs
on CPU in minutes for a smoke check; drop it for the full paper run (GPU advised).
"""
from __future__ import annotations

import argparse

from src.utils.common import get_logger, load_config

LOG = get_logger("run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "confound", "features", "baselines",
                                    "deep", "stats", "sweep", "figures", "report", "all"])
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--which", default="all")
    ap.add_argument("--limit-per-spk", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    lps = args.limit_per_spk or (6 if args.fast else 0)

    def do_manifest():
        from src.data import build_manifest
        build_manifest.build(cfg)

    def do_confound():
        from src.audit import confound_check
        confound_check.run(cfg, limit_per_spk=(8 if args.fast else 0))

    def do_features():
        from src.features import extract
        extract.run(cfg, which=args.which, limit_per_spk=lps)
        # frozen ImageNet-backbone spectrogram embeddings (resnet18_emb / mobilenetv2_emb)
        # needed by the CNN family; depends on the melspec cache produced above.
        if args.which in ("light", "all"):
            from src.features import spec_cnn_emb
            spec_cnn_emb.run(cfg)

    def do_baselines():
        from src.pipelines import run_baselines
        run_baselines.run(cfg, args.only)

    def do_deep():
        if args.fast:
            cfg["eval"]["cv"] = "sgkf"; cfg["eval"]["k_folds"] = 3
            cfg["train"]["epochs"] = 6; cfg["seeds"] = cfg["seeds"][:1]
        from src.pipelines import run_deep
        run_deep.run(cfg, args.only)

    def do_stats():
        from src.stats import tests
        tests.run(cfg)

    def do_sweep():
        from src.sweep import runner as sweeprunner
        if args.fast:
            sweeprunner.run(cfg, limit_per_spk=4, epochs=4, k_folds=2)
        else:
            # full-ish sweep (all utterances) -- feasible on GPU
            sweeprunner.run(cfg, limit_per_spk=0, epochs=15, k_folds=5)

    def do_figures():
        from src.figures import make_figures
        make_figures.run(cfg)

    def do_report():
        from src.pipelines import make_report
        make_report.run(cfg)

    table = dict(manifest=do_manifest, confound=do_confound, features=do_features,
                 baselines=do_baselines, deep=do_deep, stats=do_stats,
                 sweep=do_sweep, figures=do_figures, report=do_report)

    if args.cmd == "all":
        for step in ["manifest", "confound", "features", "baselines", "deep",
                     "stats", "sweep", "figures", "report"]:
            LOG.info("=== %s ===", step)
            table[step]()
    else:
        table[args.cmd]()


if __name__ == "__main__":
    main()
