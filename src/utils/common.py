"""Shared utilities: config loading, seeding, logging, small IO helpers.

Every module imports from here so behaviour (seed, paths, logging) is uniform.
[C5] reproducibility.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Project root = two levels up from this file (src/utils/common.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | os.PathLike = "configs/config.yaml") -> Dict[str, Any]:
    """Load a YAML config, resolving relative paths against the project root."""
    cfg_path = (PROJECT_ROOT / path) if not os.path.isabs(path) else Path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (+ cuda) for reproducibility. [C5]"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # torch optional for pure-ML paths
        pass


# --------------------------------------------------------------------------- #
# Logging (Sec 0.5 -- no scattered print)
# --------------------------------------------------------------------------- #
_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "pd", logs_dir: str = "artifacts/logs") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        d = ensure_dir(logs_dir)
        fh = logging.FileHandler(d / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    _LOGGERS[name] = logger
    return logger


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
