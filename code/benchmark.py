"""Reproducible latency measurement helpers."""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Measure ``fn`` in milliseconds after warmup, reporting tail latencies."""
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    values = np.asarray(times)
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "mean": float(values.mean()), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Benchmark one forward pass; CUDA is synchronized on both sides of timing."""
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    if batch_size <= 0 or img_size <= 0:
        raise ValueError("batch_size and img_size must be positive")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if dtype == "fp16" and selected.type != "cuda":
        raise ValueError("fp16 benchmarking is supported only on CUDA")
    model = model.to(selected).eval()
    if dtype == "fp16":
        model = model.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=selected,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if selected.type == "cuda" else None
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=selected.type,
                                                     enabled=dtype == "amp" and selected.type == "cuda"):
            model(x)
    metrics = bench(forward, warmup, iters, sync)
    metrics.update({"gpu": torch.cuda.get_device_name(selected) if selected.type == "cuda" else "CPU",
                    "dtype": dtype, "batch": batch_size, "img_size": img_size,
                    "images_per_s": batch_size / (metrics["p50"] / 1000), "torch": torch.__version__})
    return metrics


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure actual K-view cost and include the corresponding one-view report."""
    if k_views <= 0:
        raise ValueError("k_views must be positive")
    base = latency_report(model, **kw)
    device = torch.device(kw.get("device", "cuda"))
    batch, size = int(kw["batch_size"]), int(kw["img_size"])
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    amp_enabled = kw.get("dtype", "fp32") == "amp" and device.type == "cuda"
    def forward_tta():
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp_enabled):
            for _ in range(k_views):
                model(x)
    measured = bench(forward_tta, kw.get("warmup", 10), kw.get("iters", 100), sync)
    measured.update({"k_views": k_views, "one_view_p50": base["p50"],
                     "expected_p50": base["p50"] * k_views,
                     "images_per_s": batch / (measured["p50"] / 1000)})
    return measured
