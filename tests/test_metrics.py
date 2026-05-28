"""Unit tests for src/evaluation/metrics.py."""

import numpy as np
import torch
import pytest

from src.evaluation.metrics import (
    compute_probs,
    compute_auc,
    compute_fmax,
    compute_auprc,
)

N = 50   # number of fake records


def _perfect_probs_and_labels():
    """Return (probs, labels) where predictions are perfect (AUC = 1, Fmax = 1)."""
    rng    = np.random.default_rng(1)
    labels = (rng.random((N, 5)) > 0.5).astype(int)
    # Probabilities: 0.95 where label=1, 0.05 where label=0
    probs  = np.where(labels == 1, 0.95, 0.05).astype(np.float32)
    return probs, labels


def _random_probs_and_labels():
    rng    = np.random.default_rng(2)
    labels = (rng.random((N, 5)) > 0.5).astype(int)
    probs  = rng.random((N, 5)).astype(np.float32)
    return probs, labels


# ── compute_probs ─────────────────────────────────────────────────────────────

def test_compute_probs_range():
    """Sigmoid output must be in (0, 1) for any finite logit."""
    logits = np.array([[-10.0, -1.0, 0.0, 1.0, 10.0]])
    probs  = compute_probs(logits)
    assert probs.min() > 0.0
    assert probs.max() < 1.0


def test_compute_probs_zero_logit():
    """Logit = 0 maps to probability = 0.5."""
    probs = compute_probs(np.zeros((1, 5)))
    assert np.allclose(probs, 0.5, atol=1e-6)


def test_compute_probs_accepts_tensor():
    """compute_probs should handle a torch.Tensor input without error."""
    logits = torch.zeros(4, 5)
    probs  = compute_probs(logits)
    assert probs.shape == (4, 5)


def test_compute_probs_shape_preserved():
    logits = np.random.randn(N, 5).astype(np.float32)
    probs  = compute_probs(logits)
    assert probs.shape == (N, 5)


# ── compute_auc ───────────────────────────────────────────────────────────────

def test_compute_auc_perfect():
    """Perfect predictions must yield AUC macro ≈ 1.0."""
    probs, labels = _perfect_probs_and_labels()
    result = compute_auc(probs, labels)
    assert result['auc_macro'] > 0.99


def test_compute_auc_keys():
    """Return dict must contain 'auc_macro' and 'per_class_auc'."""
    probs, labels = _random_probs_and_labels()
    result = compute_auc(probs, labels)
    assert 'auc_macro' in result
    assert 'per_class_auc' in result


def test_compute_auc_per_class_has_all_superclasses():
    """per_class_auc must have an entry for each of the 5 superclasses."""
    probs, labels = _random_probs_and_labels()
    result = compute_auc(probs, labels)
    assert len(result['per_class_auc']) == 5


def test_compute_auc_range():
    """AUC macro must be in [0, 1]."""
    probs, labels = _random_probs_and_labels()
    result = compute_auc(probs, labels)
    assert 0.0 <= result['auc_macro'] <= 1.0


# ── compute_fmax ──────────────────────────────────────────────────────────────

def test_compute_fmax_keys():
    """Return dict must contain fmax, fmax_threshold, and f1_at_05."""
    probs, labels = _random_probs_and_labels()
    result = compute_fmax(probs, labels)
    assert {'fmax', 'fmax_threshold', 'f1_at_05'} <= result.keys()


def test_compute_fmax_perfect():
    """Perfect predictions must yield Fmax = 1.0."""
    probs, labels = _perfect_probs_and_labels()
    result = compute_fmax(probs, labels)
    assert result['fmax'] > 0.99


def test_compute_fmax_threshold_in_range():
    """The winning threshold must be within the sweep range [0, 1]."""
    probs, labels = _random_probs_and_labels()
    result = compute_fmax(probs, labels)
    assert 0.0 <= result['fmax_threshold'] <= 1.0


def test_compute_fmax_ge_f1_at_05():
    """Fmax must be >= F1 at threshold=0.5 by definition (it's the maximum)."""
    probs, labels = _random_probs_and_labels()
    result = compute_fmax(probs, labels)
    assert result['fmax'] >= result['f1_at_05'] - 1e-6


# ── compute_auprc ─────────────────────────────────────────────────────────────

def test_compute_auprc_keys():
    probs, labels = _random_probs_and_labels()
    result = compute_auprc(probs, labels)
    assert 'auprc_macro' in result
    assert 'per_class_auprc' in result


def test_compute_auprc_range():
    """AUPRC must be in [0, 1]."""
    probs, labels = _random_probs_and_labels()
    result = compute_auprc(probs, labels)
    assert 0.0 <= result['auprc_macro'] <= 1.0
