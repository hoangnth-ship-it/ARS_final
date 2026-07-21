"""Chạy nốt phần còn rút gọn ở FULL protocol, ngay trên máy (CPU).

Chỉ chạy lại 2 thứ (15 model khác đã có số full trong artifacts/results/):
  1) neurovoz2024  -> FULL: LOSO 22-fold, 3 seed, 40 epoch (dùng config.yaml)
  2) sweep         -> FULL: toàn bộ 1091 câu, k=5, 15 epoch
Sau đó cập nhật lại stats + figures + report trên toàn bộ 16 model.

CẢNH BÁO: trên CPU việc này rất lâu (~8-10 giờ). Nên chạy trong terminal riêng:
    python run_remainder_full.py
để nó không phụ thuộc phiên làm việc nào.
"""
from __future__ import annotations

import time

from src.utils.common import get_logger, load_config

LOG = get_logger("remainder")


def _hms(s):
    return time.strftime("%H:%M:%S", time.gmtime(s))


def main():
    cfg = load_config()
    t0 = time.perf_counter()

    LOG.info("=== [1/4] neurovoz2024 FULL (LOSO, 3 seed, 40 epoch) — phần lâu nhất ===")
    from src.pipelines import run_deep
    run_deep.run(cfg, only=["neurovoz2024"])
    LOG.info("neurovoz done @ %s", _hms(time.perf_counter() - t0))

    LOG.info("=== [2/4] sweep FULL (1091 utt, k=5, 15 epoch) ===")
    from src.sweep import runner as sweeprunner
    sweeprunner.run(cfg, limit_per_spk=0, epochs=15, k_folds=5)
    LOG.info("sweep done @ %s", _hms(time.perf_counter() - t0))

    LOG.info("=== [3/4] stats (McNemar + Friedman) trên 16 model ===")
    from src.stats import tests
    tests.run(cfg)

    LOG.info("=== [4/4] figures + report ===")
    from src.figures import make_figures
    make_figures.run(cfg)
    from src.pipelines import make_report
    make_report.run(cfg)

    LOG.info("=== HOÀN TẤT sau %s ===", _hms(time.perf_counter() - t0))


if __name__ == "__main__":
    main()
