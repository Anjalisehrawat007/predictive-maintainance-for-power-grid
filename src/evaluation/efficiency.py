"""Measured (never estimated) efficiency metrics."""
from __future__ import annotations

import io
import pickle
import time

import numpy as np
import torch

from src.models.recurrent import count_parameters


def model_size_kb(model) -> float:
    buf = io.BytesIO()
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), buf)
    else:
        pickle.dump(model, buf)
    return buf.getbuffer().nbytes / 1024


@torch.no_grad()
def torch_latency(model: torch.nn.Module, seq_len: int, n_features: int, batch: int = 1,
                  device: str = "cpu", threads: int = 1, n_iter: int = 200, warmup: int = 20) -> dict:
    """Median / p95 wall-clock latency (ms) of one forward pass."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.randn(batch, seq_len, n_features, device=dev)
    times = []
    for i in range(n_iter + warmup):
        t = time.perf_counter()
        model(x)
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(prev)
    model.to("cpu")
    return {"latency_ms_median": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95)),
            "throughput_per_s": batch * 1000 / float(np.median(times))}


def sklearn_latency(model, n_features: int, n_iter: int = 100) -> dict:
    x = np.random.randn(1, n_features).astype(np.float32)
    for _ in range(5):
        model.predict_proba(x)
    times = []
    for _ in range(n_iter):
        t = time.perf_counter()
        model.predict_proba(x)
        times.append((time.perf_counter() - t) * 1000)
    return {"latency_ms_median": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95))}


def torch_efficiency(model: torch.nn.Module, seq_len: int, n_features: int) -> dict:
    lat = torch_latency(model, seq_len, n_features, batch=1, threads=1)
    thr = torch_latency(model, seq_len, n_features, batch=512, threads=4, n_iter=30, warmup=5)
    return {"parameters": count_parameters(model), "model_size_kb": model_size_kb(model),
            "inference_latency_ms": lat["latency_ms_median"], "latency_p95_ms": lat["latency_ms_p95"],
            "throughput_per_s": thr["throughput_per_s"]}
