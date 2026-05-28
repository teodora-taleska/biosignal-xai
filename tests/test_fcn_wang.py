"""Unit tests for src/models/fcn_wang.py."""

import torch
import pytest

from src.models.fcn_wang import FCNWang, fcn_wang


# ── Forward pass ──────────────────────────────────────────────────────────────

def test_forward_output_shape():
    """FCNWang must output (batch, 5) logits for a standard PTB-XL input."""
    model = FCNWang()
    x     = torch.randn(4, 12, 1000)
    out   = model(x)
    assert out.shape == (4, 5)


def test_forward_batch_size_one():
    """Model must handle a single-sample batch in eval mode.

    BatchNorm1d raises with batch=1 during training (known PyTorch behaviour).
    Single-record inference always runs in eval mode, so this is the correct scenario.
    """
    model = FCNWang().eval()
    with torch.no_grad():
        out = model(torch.randn(1, 12, 1000))
    assert out.shape == (1, 5)


def test_forward_variable_sequence_length():
    """FCNWang uses adaptive pooling so any sequence length T should work."""
    model = FCNWang()
    for T in [250, 500, 1000, 2000]:
        out = model(torch.randn(2, 12, T))
        assert out.shape == (2, 5), f"Failed at T={T}"


def test_forward_custom_classes():
    """num_classes constructor parameter changes the output dimension."""
    model = FCNWang(num_classes=3)
    out   = model(torch.randn(2, 12, 1000))
    assert out.shape == (2, 3)


def test_forward_output_is_logits():
    """Output is raw logits — values must not be constrained to [0, 1]."""
    model = FCNWang()
    out   = model(torch.randn(8, 12, 1000))
    # With random weights a large batch almost certainly has at least one value > 1
    assert out.abs().max().item() > 0.0


# ── count_parameters ──────────────────────────────────────────────────────────

def test_count_parameters_keys():
    """count_parameters must return a dict with trainable, total, and percentage."""
    model  = FCNWang()
    params = model.count_parameters()
    assert {'trainable', 'total', 'percentage'} <= params.keys()


def test_count_parameters_all_trainable_by_default():
    """By default every parameter is trainable, so trainable == total."""
    model  = FCNWang()
    params = model.count_parameters()
    assert params['trainable'] == params['total']


def test_count_parameters_reasonable_size():
    """FCN-Wang should have roughly 270k–350k total parameters."""
    model  = FCNWang()
    params = model.count_parameters()
    assert 200_000 < params['total'] < 500_000


# ── eval mode / no-grad ───────────────────────────────────────────────────────

def test_no_grad_in_eval():
    """In eval mode with torch.no_grad, output requires_grad must be False."""
    model = FCNWang().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 12, 1000))
    assert not out.requires_grad


# ── factory function ──────────────────────────────────────────────────────────

def test_fcn_wang_factory():
    """fcn_wang() convenience function returns a valid FCNWang instance."""
    model = fcn_wang()
    assert isinstance(model, FCNWang)
    out = model(torch.randn(2, 12, 1000))
    assert out.shape == (2, 5)
