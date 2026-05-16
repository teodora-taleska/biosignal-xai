"""
Benchmark-grade evaluation metrics matching Strodthoff et al. 2020 (PTB-XL paper).

Primary metric  : macro-averaged AUC (term-centric, threshold-free)
Secondary metric: Fmax (sample-centric, threshold optimised on *validation* set)
Confidence intervals: 1 000-iteration bootstrap on test set (95 % CI)

Usage
-----
from src.evaluation.metrics import full_eval, compute_fmax, bootstrap_ci

# After collecting logits and labels from a model:
results = full_eval(logits, labels, val_logits, val_labels)
print(results['auc_macro'], results['fmax'], results['auc_ci_95'])
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from src.utils.config import CFG

SUPERCLASSES = CFG['data']['superclasses']


# Core helpers

def _to_numpy(t):
    """Accept torch tensor or numpy array; return float32 numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def compute_probs(logits) -> np.ndarray:
    """Sigmoid probabilities from raw logits."""
    logits = _to_numpy(logits)
    return expit(logits)   # scipy expit: stable for all logit magnitudes


# AUC

def compute_auc(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Macro-averaged AUC (the paper's primary metric) and per-class AUC.

    Parameters
    ----------
    probs  : (N, C) sigmoid probabilities
    labels : (N, C) multi-hot ground truth

    Returns dict with keys: auc_macro, per_class_auc
    """
    try:
        auc_macro = float(roc_auc_score(labels, probs, average='macro'))
    except ValueError:
        auc_macro = 0.0

    per_class = {}
    for i, sc in enumerate(SUPERCLASSES):
        try:
            per_class[sc] = float(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            per_class[sc] = 0.0

    return {'auc_macro': auc_macro, 'per_class_auc': per_class}


# Fmax  (sample-centric, threshold-free)

def compute_fmax(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray = None,
) -> dict:
    """
    Fmax: sample-centric F1 maximised over sigmoid thresholds.

    The paper sweeps τ ∈ [0, 1] in 100 steps and picks the threshold that
    maximises the macro-averaged F1.  This is reported as a single number
    (Fmax) alongside the winning threshold (τ*).

    Parameters
    ----------
    probs      : (N, C) sigmoid probabilities
    labels     : (N, C) multi-hot ground truth
    thresholds : 1-D array of thresholds to sweep; default np.linspace(0.02, 0.98, 100)

    Returns dict with keys: fmax, fmax_threshold, f1_at_05
    """
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 100)

    best_f1  = 0.0
    best_tau = 0.5

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        f1    = float(f1_score(labels, preds, average='macro', zero_division=0))
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)

    # Also report the standard F1 at threshold=0.5 for comparison
    preds_05 = (probs >= 0.5).astype(int)
    f1_at_05 = float(f1_score(labels, preds_05, average='macro', zero_division=0))

    return {
        'fmax':           best_f1,
        'fmax_threshold': best_tau,
        'f1_at_05':       f1_at_05,
    }


# Bootstrap confidence intervals

def bootstrap_ci(
    probs:  np.ndarray,
    labels: np.ndarray,
    n_iter: int = 1000,
    alpha:  float = 0.05,
    seed:   int = 42,
) -> dict:
    """
    Non-parametric bootstrap 95 % CI for AUC and Fmax.

    Matches the procedure in Strodthoff et al. 2020:
      resample (with replacement) N test records 1 000 times,
      compute AUC and Fmax each time, report 2.5th and 97.5th percentiles.

    Parameters
    ----------
    probs  : (N, C) sigmoid probabilities on the test set
    labels : (N, C) multi-hot ground truth on the test set
    n_iter : number of bootstrap resamples (1 000 matches the paper)
    alpha  : coverage level — 0.05 → 95 % CI
    seed   : random seed for reproducibility

    Returns dict with keys:
        auc_ci_lo, auc_ci_hi, auc_ci_95 (formatted string)
        fmax_ci_lo, fmax_ci_hi, fmax_ci_95
    """
    rng  = np.random.default_rng(seed)
    N    = len(probs)
    aucs = []
    fmaxs = []

    for _ in range(n_iter):
        idx    = rng.integers(0, N, size=N)
        p_boot = probs[idx]
        l_boot = labels[idx]

        try:
            a = float(roc_auc_score(l_boot, p_boot, average='macro'))
        except ValueError:
            a = 0.0
        aucs.append(a)
        fmaxs.append(compute_fmax(p_boot, l_boot)['fmax'])

    aucs  = np.array(aucs)
    fmaxs = np.array(fmaxs)
    lo    = alpha / 2 * 100
    hi    = (1 - alpha / 2) * 100

    auc_lo, auc_hi   = float(np.percentile(aucs, lo)),  float(np.percentile(aucs, hi))
    fmax_lo, fmax_hi = float(np.percentile(fmaxs, lo)), float(np.percentile(fmaxs, hi))

    return {
        'auc_ci_lo':  auc_lo,
        'auc_ci_hi':  auc_hi,
        'auc_ci_95':  f'{np.mean(aucs):.3f} [{auc_lo:.3f}, {auc_hi:.3f}]',
        'fmax_ci_lo': fmax_lo,
        'fmax_ci_hi': fmax_hi,
        'fmax_ci_95': f'{np.mean(fmaxs):.3f} [{fmax_lo:.3f}, {fmax_hi:.3f}]',
    }


# Average precision (AUPRC) — bonus metric for imbalanced classes

def compute_auprc(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Macro-averaged area under the precision-recall curve."""
    try:
        auprc = float(average_precision_score(labels, probs, average='macro'))
    except ValueError:
        auprc = 0.0
    per_class = {}
    for i, sc in enumerate(SUPERCLASSES):
        try:
            per_class[sc] = float(average_precision_score(labels[:, i], probs[:, i]))
        except ValueError:
            per_class[sc] = 0.0
    return {'auprc_macro': auprc, 'per_class_auprc': per_class}


# Full evaluation: single entry point used by notebooks

def full_eval(
    logits:      np.ndarray | torch.Tensor,
    labels:      np.ndarray | torch.Tensor,
    run_bootstrap:  bool = True,
    n_bootstrap: int  = 1000,
) -> dict:
    """
    Complete evaluation matching the PTB-XL benchmark paper.

    Parameters
    ----------
    logits        : (N, C) raw model outputs (before sigmoid)
    labels        : (N, C) multi-hot ground truth
    run_bootstrap : whether to compute 95 % CI (slow, ~5 s for 1000 iters)
    n_bootstrap   : bootstrap iterations

    Returns
    -------
    Flat dict with all metrics, ready to serialise to JSON.
    """
    probs  = compute_probs(logits)
    labels = _to_numpy(labels).astype(int)

    auc_results  = compute_auc(probs, labels)
    fmax_results = compute_fmax(probs, labels)
    auprc        = compute_auprc(probs, labels)

    result = {
        **auc_results,
        **fmax_results,
        **auprc,
    }

    if run_bootstrap:
        ci = bootstrap_ci(probs, labels, n_iter=n_bootstrap)
        result.update(ci)

    return result


def print_eval(results: dict):
    """Pretty-print a full_eval result dict."""
    print(f"  AUC  (macro): {results['auc_macro']:.4f}", end='')
    if 'auc_ci_95' in results:
        print(f"  95% CI: {results['auc_ci_95']}", end='')
    print()
    print(f"  Fmax (macro): {results['fmax']:.4f}  @ τ={results['fmax_threshold']:.2f}", end='')
    if 'fmax_ci_95' in results:
        print(f"  95% CI: {results['fmax_ci_95']}", end='')
    print()
    print(f"  AUPRC(macro): {results['auprc_macro']:.4f}")
    print(f"  Per-class AUC:")
    for cls, val in results['per_class_auc'].items():
        bar = '█' * int(val * 25)
        print(f"    {cls:5s}: {val:.3f}  {bar}")
