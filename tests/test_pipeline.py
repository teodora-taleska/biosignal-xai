"""Unit tests for src/inference/pipeline.py — ECGInferencePipeline and HeartBERTPipeline."""

import numpy as np
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from src.inference.pipeline import ECGInferencePipeline, HeartBERTPipeline

SUPERCLASSES  = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
EXPECTED_KEYS = {
    'predicted_classes', 'class_probabilities', 'confidence_score',
    'uncertainty', 'raw_logits', 'uncertainty_level',
}


# ── Minimal models ────────────────────────────────────────────────────────────

class _DeterministicModel(nn.Module):
    """Returns fixed logits — no uncertainty head."""
    def __init__(self, logits):
        super().__init__()
        # nn.Parameter so .to(device) works and the model has at least one param
        self._logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32),
                                    requires_grad=False)

    def forward(self, x):
        return self._logits.unsqueeze(0).expand(x.shape[0], -1)


class _UncertaintyModel(nn.Module):
    """Returns (mean, log_var) — mimics AleatoricWrapper."""
    def __init__(self, logits, log_var_val=-2.0):
        super().__init__()
        self._logits  = nn.Parameter(torch.tensor(logits, dtype=torch.float32),
                                     requires_grad=False)
        self._log_var = nn.Parameter(torch.tensor([log_var_val] * len(logits),
                                                   dtype=torch.float32),
                                     requires_grad=False)

    def forward(self, x):
        B = x.shape[0]
        return (
            self._logits.unsqueeze(0).expand(B, -1),
            self._log_var.unsqueeze(0).expand(B, -1),
        )


# ── ECGInferencePipeline ──────────────────────────────────────────────────────

def test_pipeline_predict_output_keys():
    """Result dict must contain all expected keys."""
    model    = _DeterministicModel([2.0, -1.0, -1.0, -1.0, -1.0])
    pipeline = ECGInferencePipeline(model, device=torch.device('cpu'), has_uncertainty=False)
    result   = pipeline.predict(np.zeros((12, 1000), dtype=np.float32))
    assert EXPECTED_KEYS <= result.keys()


def test_pipeline_predict_high_logit_wins():
    """The class with the highest logit should appear in predicted_classes."""
    model    = _DeterministicModel([-1.0, 5.0, -1.0, -1.0, -1.0])  # MI wins
    pipeline = ECGInferencePipeline(model, device=torch.device('cpu'), has_uncertainty=False)
    result   = pipeline.predict(np.zeros((12, 1000), dtype=np.float32))
    assert 'MI' in result['predicted_classes']


def test_pipeline_predict_accepts_transposed_input():
    """(1000, 12) input must be transposed internally without error."""
    model    = _DeterministicModel([1.0, 0.0, 0.0, 0.0, 0.0])
    pipeline = ECGInferencePipeline(model, device=torch.device('cpu'), has_uncertainty=False)
    result   = pipeline.predict(np.zeros((1000, 12), dtype=np.float32))
    assert 'predicted_classes' in result


def test_pipeline_predict_argmax_fallback():
    """When all logits are strongly negative (all probs < 0.5), argmax is used as fallback."""
    model    = _DeterministicModel([-5.0, -4.0, -5.0, -5.0, -5.0])  # MI is argmax
    pipeline = ECGInferencePipeline(model, device=torch.device('cpu'), has_uncertainty=False)
    result   = pipeline.predict(np.zeros((12, 1000), dtype=np.float32))
    # Must predict exactly one class (the argmax fallback)
    assert len(result['predicted_classes']) == 1
    assert result['predicted_classes'][0] == 'MI'


def test_pipeline_predict_class_probabilities_sum_le_5():
    """Five independent sigmoid outputs: sum is in (0, 5], not necessarily 1."""
    model    = _DeterministicModel([0.0, 0.0, 0.0, 0.0, 0.0])
    pipeline = ECGInferencePipeline(model, device=torch.device('cpu'), has_uncertainty=False)
    result   = pipeline.predict(np.zeros((12, 1000), dtype=np.float32))
    total    = sum(result['class_probabilities'].values())
    assert 0.0 < total <= 5.0


def test_pipeline_uncertainty_model_returns_float():
    """With an uncertainty model, 'uncertainty' in the result must be a float."""
    model    = _UncertaintyModel([2.0, -1.0, -1.0, -1.0, -1.0])
    pipeline = ECGInferencePipeline(model, device=torch.device('cpu'), has_uncertainty=True)
    result   = pipeline.predict(np.zeros((12, 1000), dtype=np.float32))
    assert isinstance(result['uncertainty'], float)


def test_pipeline_deterministic_model_uncertainty_is_none():
    """Without an uncertainty head, 'uncertainty' must be None."""
    model    = _DeterministicModel([1.0, 0.0, 0.0, 0.0, 0.0])
    pipeline = ECGInferencePipeline(model, device=torch.device('cpu'), has_uncertainty=False)
    result   = pipeline.predict(np.zeros((12, 1000), dtype=np.float32))
    assert result['uncertainty'] is None


# ── ECGInferencePipeline._uncertainty_level ───────────────────────────────────

def test_uncertainty_level_none():
    assert 'N/A' in ECGInferencePipeline._uncertainty_level(None)


def test_uncertainty_level_low():
    # uncertainty_threshold = 0.5; low is < 0.25
    label = ECGInferencePipeline._uncertainty_level(0.1)
    assert 'low' in label


def test_uncertainty_level_high():
    label = ECGInferencePipeline._uncertainty_level(1.0)
    assert 'high' in label


# ── HeartBERTPipeline ─────────────────────────────────────────────────────────

def _mock_heartbert_classifier(logits):
    """Create a mock HeartBERTClassifier that returns fixed logits."""
    clf = MagicMock()
    clf.predict_logits = MagicMock(
        return_value=np.array([logits], dtype=np.float32)
    )
    clf.device = torch.device('cpu')
    return clf


def test_heartbert_pipeline_predict_keys():
    """Result dict must contain all expected keys."""
    pipeline = HeartBERTPipeline(_mock_heartbert_classifier([2.0, -1.0, -1.0, -1.0, -1.0]))
    result   = pipeline.predict(np.zeros(1000, dtype=np.float32))
    assert EXPECTED_KEYS <= result.keys()


def test_heartbert_pipeline_extracts_lead_ii():
    """Given a (12, 1000) signal, only Lead II (row 1) should be passed to the classifier."""
    captured = {}

    clf = MagicMock()
    clf.device = torch.device('cpu')

    def _capture(x):
        captured['x'] = x.copy()
        return np.array([[1.0, -1.0, -1.0, -1.0, -1.0]], dtype=np.float32)

    clf.predict_logits = _capture

    signal     = np.zeros((12, 1000), dtype=np.float32)
    signal[1]  = 5.0   # Lead II = 5, everything else = 0

    HeartBERTPipeline(clf).predict(signal)
    assert np.all(captured['x'][0] == 5.0), "Lead II not extracted correctly"


def test_heartbert_pipeline_uncertainty_always_none():
    """HeartBERT has no uncertainty head — 'uncertainty' must always be None."""
    pipeline = HeartBERTPipeline(_mock_heartbert_classifier([1.0, -1.0, -1.0, -1.0, -1.0]))
    result   = pipeline.predict(np.zeros(1000, dtype=np.float32))
    assert result['uncertainty'] is None


def test_heartbert_pipeline_accepts_transposed_signal():
    """(1000, 12) input must be handled the same as (12, 1000)."""
    pipeline = HeartBERTPipeline(_mock_heartbert_classifier([1.0, -1.0, -1.0, -1.0, -1.0]))
    result   = pipeline.predict(np.zeros((1000, 12), dtype=np.float32))
    assert 'predicted_classes' in result
