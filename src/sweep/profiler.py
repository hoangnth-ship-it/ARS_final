"""Cost/perf profiling (Sec 8.3/8.4).

Measures per model variant:
  - #parameters, FLOPs/MACs (thop if available, else params-only)
  - training time, inference latency (ms/utt, warm-up + median)
  - peak memory (cuda if available, else tracemalloc)
  - output feature size (embedding dim)
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Tuple

import numpy as np


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_flops(model, input_shape: Tuple[int, ...]) -> float:
    """MACs for one forward pass; returns NaN if thop unavailable."""
    try:
        import torch
        from thop import profile
        x = torch.randn(1, *input_shape)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs)
    except Exception:
        return float("nan")


def inference_latency_ms(model, input_shape: Tuple[int, ...], n_warmup=5, n_iter=30) -> float:
    import torch
    model.eval()
    x = torch.randn(1, *input_shape)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def peak_memory_mb(fn) -> Tuple[float, object]:
    """Run fn(), return (peak_mb, result). CUDA peak if available else tracemalloc."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            res = fn()
            return torch.cuda.max_memory_allocated() / 1e6, res
    except Exception:
        pass
    tracemalloc.start()
    res = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6, res


def device_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
