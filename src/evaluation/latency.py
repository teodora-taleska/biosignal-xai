"""
Inference latency benchmarking utilities.

Measures wall-clock time per sample (and per batch) for any nn.Module.
Used by the final evaluation notebook and the Streamlit app's performance panel.

The benchmark follows standard practice:
  1. Warm-up runs (not measured): fills CUDA caches, JIT compiles kernels
  2. Timed runs: median latency (more robust than mean against outliers)

Usage
-----
from src.evaluation.latency import benchmark_latency

report = benchmark_latency(model, input_shape=(12, 1000), batch_sizes=[1, 8, 32])
"""

from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np


def _make_dummy(batch_size: int, input_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Random float tensor on device, shape (batch_size, *input_shape)."""
    return torch.randn(batch_size, *input_shape, device=device)


def measure_latency(
    model:       nn.Module,
    input_shape: Tuple[int, ...],
    batch_size:  int = 1,
    n_warmup:    int = 10,
    n_runs:      int = 100,
    device:      torch.device = None,
) -> dict:
    """
    Measure inference latency for a single batch size.

    Parameters
    ----------
    model        : nn.Module — must already be in eval mode
    input_shape  : shape *excluding* batch, e.g. (12, 1000) for ECG
    batch_size   : samples per batch
    n_warmup     : warm-up forward passes (not measured)
    n_runs       : measured forward passes
    device       : defaults to CUDA if available, else CPU

    Returns
    -------
    dict with:
        batch_size           : int
        device               : str ('cuda' or 'cpu')
        latency_ms_mean      : mean latency in ms per batch
        latency_ms_median    : median latency in ms per batch
        latency_ms_p95       : 95th-percentile latency in ms per batch
        latency_ms_per_sample: median / batch_size
        throughput_per_sec   : batch_size / (median_ms / 1000)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device).eval()
    dummy = _make_dummy(batch_size, input_shape, device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Synchronise GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _  = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    times   = np.array(times)
    median  = float(np.median(times))
    return {
        'batch_size':            batch_size,
        'device':                str(device),
        'latency_ms_mean':       float(np.mean(times)),
        'latency_ms_median':     median,
        'latency_ms_p95':        float(np.percentile(times, 95)),
        'latency_ms_per_sample': round(median / batch_size, 3),
        'throughput_per_sec':    round(batch_size / (median / 1000), 1),
    }


def benchmark_latency(
    model:        nn.Module,
    input_shape:  Tuple[int, ...] = (12, 1000),
    batch_sizes:  List[int] = None,
    n_warmup:     int = 10,
    n_runs:       int = 50,
    device:       torch.device = None,
) -> dict:
    """
    Benchmark latency across multiple batch sizes.

    Parameters
    ----------
    model       : nn.Module in eval mode
    input_shape : (leads, time_steps), default (12, 1000)
    batch_sizes : list of batch sizes to test; default [1, 4, 16, 32, 64]
    n_warmup    : warm-up passes per batch size
    n_runs      : measured passes per batch size
    device      : defaults to CUDA if available

    Returns
    -------
    dict with:
        model_params    : total parameter count
        results         : list of per-batch-size dicts (from measure_latency)
        summary_table   : printable string
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 32, 64]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_params = sum(p.numel() for p in model.parameters())

    results = []
    for bs in batch_sizes:
        try:
            r = measure_latency(model, input_shape, batch_size=bs,
                                n_warmup=n_warmup, n_runs=n_runs, device=device)
            results.append(r)
            print(f"  bs={bs:3d} | median {r['latency_ms_median']:6.1f} ms | "
                  f"{r['latency_ms_per_sample']:.2f} ms/sample | "
                  f"{r['throughput_per_sec']:.0f} samples/s")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  bs={bs}: OOM — skipped")
                torch.cuda.empty_cache()
            else:
                raise

    # Build summary table string
    header = f"{'bs':>4}  {'median ms':>10}  {'ms/sample':>10}  {'samples/s':>10}"
    rows   = [header, '-' * len(header)]
    for r in results:
        rows.append(f"{r['batch_size']:>4}  {r['latency_ms_median']:>10.1f}  "
                    f"{r['latency_ms_per_sample']:>10.2f}  "
                    f"{r['throughput_per_sec']:>10.0f}")

    return {
        'model_params':  total_params,
        'device':        str(device),
        'input_shape':   input_shape,
        'results':       results,
        'summary_table': '\n'.join(rows),
    }
