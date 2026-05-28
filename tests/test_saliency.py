"""Unit tests for src/explainability/saliency.py."""

import numpy as np
import torch
import torch.nn as nn
import pytest

from src.explainability.saliency import compute_saliency, top_salient_leads

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SIGNAL = np.random.default_rng(42).standard_normal((12, 1000)).astype(np.float32)


# ── Minimal differentiable model for compute_saliency ────────────────────────

class _TinyECGModel(nn.Module):
    """Minimal 1-layer model: pools over time then projects to 5 classes."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 5)

    def forward(self, x):
        # x: (B, 12, 1000) → pool over time → (B, 12)
        return self.fc(x.mean(dim=2))


# ── compute_saliency ──────────────────────────────────────────────────────────

def test_compute_saliency_output_shape():
    """Saliency map must be (12, 1000) — one value per lead per time-step."""
    model = _TinyECGModel().eval()
    sal   = compute_saliency(model, SIGNAL, target_class_idx=0, device=torch.device('cpu'))
    assert sal.shape == (12, 1000)


def test_compute_saliency_non_negative():
    """Saliency values are absolute gradients, so must all be >= 0."""
    model = _TinyECGModel().eval()
    sal   = compute_saliency(model, SIGNAL, target_class_idx=0, device=torch.device('cpu'))
    assert sal.min() >= 0.0


def test_compute_saliency_is_float32():
    model = _TinyECGModel().eval()
    sal   = compute_saliency(model, SIGNAL, target_class_idx=0, device=torch.device('cpu'))
    assert sal.dtype == np.float32


def test_compute_saliency_all_classes():
    """Saliency should run without error for each of the 5 class indices."""
    model = _TinyECGModel().eval()
    for cls in range(5):
        sal = compute_saliency(model, SIGNAL, target_class_idx=cls, device=torch.device('cpu'))
        assert sal.shape == (12, 1000)


def test_compute_saliency_fallback_on_broken_model():
    """A model that raises during forward should trigger the uniform fallback (all ones)."""
    class _BrokenModel(nn.Module):
        def forward(self, x):
            raise RuntimeError('simulated failure')

    sal = compute_saliency(_BrokenModel(), SIGNAL, target_class_idx=0, device=torch.device('cpu'))
    assert sal.shape == (12, 1000)
    assert np.all(sal == 1.0)


# ── top_salient_leads ─────────────────────────────────────────────────────────

def test_top_salient_leads_default_count():
    """Default top_k=3 should return exactly 3 lead names."""
    sal    = np.random.default_rng(0).random((12, 1000)).astype(np.float32)
    result = top_salient_leads(sal, LEAD_NAMES)
    assert len(result) == 3


def test_top_salient_leads_custom_k():
    """top_k parameter is respected."""
    sal    = np.random.default_rng(0).random((12, 1000)).astype(np.float32)
    result = top_salient_leads(sal, LEAD_NAMES, top_k=5)
    assert len(result) == 5


def test_top_salient_leads_names_from_lead_list():
    """All returned names must come from the provided lead_names list."""
    sal    = np.random.default_rng(0).random((12, 1000)).astype(np.float32)
    result = top_salient_leads(sal, LEAD_NAMES, top_k=3)
    for name in result:
        assert name in LEAD_NAMES


def test_top_salient_leads_most_salient_is_first():
    """The lead with the highest mean saliency must appear at index 0."""
    sal = np.zeros((12, 1000), dtype=np.float32)
    sal[7] = 1.0   # V2 is the brightest lead
    result = top_salient_leads(sal, LEAD_NAMES, top_k=3)
    assert result[0] == 'V2'


def test_top_salient_leads_no_duplicates():
    """Each returned lead name must be unique."""
    sal    = np.random.default_rng(0).random((12, 1000)).astype(np.float32)
    result = top_salient_leads(sal, LEAD_NAMES, top_k=6)
    assert len(result) == len(set(result))
