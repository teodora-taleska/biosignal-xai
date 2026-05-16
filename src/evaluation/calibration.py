"""
Calibration utilities for multi-label ECG classification.

A well-calibrated model should satisfy: when it predicts P(y=1) = 0.7,
roughly 70 % of those samples should actually be positive.

Functions
---------
expected_calibration_error : scalar ECE (lower is better)
reliability_data           : bin-level data for plotting reliability diagrams
reliability_diagram        : matplotlib figure (optional, used in notebooks)

Reference: Guo et al. "On Calibration of Modern Neural Networks." ICML 2017.
"""

from __future__ import annotations

import numpy as np
import torch

from src.utils.config import CFG

SUPERCLASSES = CFG['data']['superclasses']


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def reliability_data(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute per-bin accuracy and confidence for a reliability diagram.

    For multi-label problems we treat each (sample, class) pair as an
    independent binary prediction — same as the paper's evaluation framework.

    Parameters
    ----------
    probs  : (N, C) sigmoid probabilities
    labels : (N, C) multi-hot integer ground truth
    n_bins : number of equal-width confidence bins

    Returns
    -------
    dict with keys:
        bin_centers  : (n_bins,) midpoint of each confidence bin
        bin_acc      : (n_bins,) fraction of positives in each bin
        bin_conf     : (n_bins,) mean predicted confidence in each bin
        bin_count    : (n_bins,) number of (sample, class) pairs in each bin
        per_class    : dict  class→ same structure for each superclass
    """
    probs  = _to_numpy(probs).ravel()    # flatten all (sample, class) pairs
    labels = _to_numpy(labels).ravel().astype(int)

    bins       = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_acc    = np.zeros(n_bins)
    bin_conf   = np.zeros(n_bins)
    bin_count  = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc[i]   = labels[mask].mean()
        bin_conf[i]  = probs[mask].mean()
        bin_count[i] = int(mask.sum())

    return {
        'bin_centers': bin_centers.tolist(),
        'bin_acc':     bin_acc.tolist(),
        'bin_conf':    bin_conf.tolist(),
        'bin_count':   bin_count.tolist(),
    }


def expected_calibration_error(
    probs:  np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) — weighted mean |accuracy - confidence|.

    ECE ≈ 0  → perfectly calibrated
    ECE ≈ 1  → completely miscalibrated

    Computed over all (sample, class) pairs (multi-label convention).
    """
    probs  = _to_numpy(probs).ravel()
    labels = _to_numpy(labels).ravel().astype(int)
    N      = len(probs)

    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        n    = mask.sum()
        if n == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (n / N) * abs(acc - conf)

    return float(ece)


def per_class_ece(
    probs:  np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    n_bins: int = 10,
) -> dict:
    """ECE computed separately for each of the 5 superclasses."""
    probs  = _to_numpy(probs)
    labels = _to_numpy(labels).astype(int)
    return {
        sc: expected_calibration_error(probs[:, i], labels[:, i], n_bins)
        for i, sc in enumerate(SUPERCLASSES)
    }


def calibration_summary(
    probs:  np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    n_bins: int = 10,
) -> dict:
    """
    Full calibration report — used in the final evaluation notebook and Streamlit app.

    Returns
    -------
    dict with:
        ece           : overall ECE (scalar)
        per_class_ece : {superclass → ECE}
        reliability   : reliability_data() output for plotting
    """
    probs  = _to_numpy(probs)
    labels = _to_numpy(labels).astype(int)
    return {
        'ece':           expected_calibration_error(probs, labels, n_bins),
        'per_class_ece': per_class_ece(probs, labels, n_bins),
        'reliability':   reliability_data(probs, labels, n_bins),
    }
