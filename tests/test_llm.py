"""Unit tests for src/explainability/llm.py — pure prompt-building logic only.

Qwen2 model loading and generation are not tested here (require GPU / large download).
"""

import numpy as np
import pytest

from src.explainability.llm import build_ecg_prompt

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def _base_result(predicted=('NORM',), confidence=0.88, uncertainty=0.12):
    """Minimal result dict that matches the ECGInferencePipeline output contract."""
    return {
        'predicted_classes':   list(predicted),
        'class_probabilities': {c: 0.88 if c == predicted[0] else 0.05 for c in SUPERCLASSES},
        'confidence_score':    confidence,
        'uncertainty':         uncertainty,
        'raw_logits':          [2.0, -1.0, -1.0, -1.0, -1.0],
        'uncertainty_level':   'low - signal was clean, result is trustworthy',
    }


# ── build_ecg_prompt ──────────────────────────────────────────────────────────

def test_build_ecg_prompt_returns_string():
    """build_ecg_prompt must return a non-empty string."""
    prompt = build_ecg_prompt(_base_result(), saliency=None, lead_names=LEAD_NAMES)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_ecg_prompt_contains_predicted_class():
    """The predicted class must appear somewhere in the prompt."""
    prompt = build_ecg_prompt(_base_result(predicted=('MI',)), saliency=None, lead_names=LEAD_NAMES)
    assert 'MI' in prompt


def test_build_ecg_prompt_no_saliency_says_not_computed():
    """When saliency=None the prompt must state 'not computed' for salient leads."""
    prompt = build_ecg_prompt(_base_result(), saliency=None, lead_names=LEAD_NAMES)
    assert 'not computed' in prompt


def test_build_ecg_prompt_with_saliency_includes_lead_name():
    """When saliency is provided, at least one lead name from the top-3 must appear."""
    sal = np.zeros((12, 1000), dtype=np.float32)
    sal[1] = 1.0   # Lead II is the most salient
    prompt = build_ecg_prompt(_base_result(), saliency=sal, lead_names=LEAD_NAMES)
    assert 'II' in prompt


def test_build_ecg_prompt_uncertainty_none_renders_na():
    """uncertainty=None must produce 'N/A' in the prompt (not crash)."""
    result = _base_result()
    result['uncertainty'] = None
    prompt = build_ecg_prompt(result, saliency=None, lead_names=LEAD_NAMES)
    assert 'N/A' in prompt


def test_build_ecg_prompt_with_true_classes():
    """When true_classes is provided it must appear in the prompt."""
    prompt = build_ecg_prompt(
        _base_result(), saliency=None, lead_names=LEAD_NAMES, true_classes=['NORM']
    )
    assert 'NORM' in prompt


def test_build_ecg_prompt_unknown_true_classes():
    """When true_classes is not provided the prompt must contain 'unknown'."""
    prompt = build_ecg_prompt(
        _base_result(), saliency=None, lead_names=LEAD_NAMES, true_classes=None
    )
    assert 'unknown' in prompt


def test_build_ecg_prompt_with_uncertainty_per_class():
    """When uncertainty_per_class is present, ± notation must appear in the prompt."""
    result = _base_result()
    result['uncertainty_per_class'] = {c: 0.02 for c in SUPERCLASSES}
    prompt = build_ecg_prompt(result, saliency=None, lead_names=LEAD_NAMES)
    assert '+/-' in prompt
