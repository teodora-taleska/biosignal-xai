"""Unit tests for src/preprocessing/label_utils.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.label_utils import _to_label_vec, _scp_to_superclasses

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


# ── _to_label_vec ─────────────────────────────────────────────────────────────

def test_to_label_vec_length():
    """Output vector is always length 5 regardless of input."""
    assert len(_to_label_vec(['NORM'])) == 5
    assert len(_to_label_vec([])) == 5


def test_to_label_vec_norm():
    """NORM maps to index 0."""
    vec = _to_label_vec(['NORM'])
    assert vec[0] == 1.0
    assert sum(vec) == 1.0


def test_to_label_vec_multi_label():
    """Two active classes produce two 1s in the correct positions."""
    vec = _to_label_vec(['MI', 'HYP'])
    assert vec[SUPERCLASSES.index('MI')] == 1.0
    assert vec[SUPERCLASSES.index('HYP')] == 1.0
    assert sum(vec) == 2.0


def test_to_label_vec_all_classes():
    """All five superclasses active → all-ones vector."""
    vec = _to_label_vec(SUPERCLASSES)
    assert all(v == 1.0 for v in vec)


def test_to_label_vec_empty():
    """No active classes → all-zeros vector."""
    vec = _to_label_vec([])
    assert all(v == 0.0 for v in vec)


def test_to_label_vec_unknown_class_ignored():
    """Unknown class strings are silently ignored."""
    vec = _to_label_vec(['NORM', 'UNKNOWN_CLASS'])
    assert vec[0] == 1.0
    assert sum(vec) == 1.0


# ── _scp_to_superclasses ──────────────────────────────────────────────────────

def _make_scp_df(rows: dict) -> pd.DataFrame:
    """Build a minimal scp_statements DataFrame indexed by SCP code."""
    df = pd.DataFrame.from_dict(rows, orient='index', columns=['diagnostic_class'])
    df.index.name = None
    return df


def test_scp_to_superclasses_maps_norm():
    """A single NORM code with likelihood > 0 should map to ['NORM']."""
    scp_df = _make_scp_df({'NORM': {'diagnostic_class': 'NORM'}})
    result = _scp_to_superclasses({'NORM': 100.0}, scp_df)
    assert result == ['NORM']


def test_scp_to_superclasses_zero_likelihood_excluded():
    """Codes with likelihood = 0 must be excluded even if they match a superclass."""
    scp_df = _make_scp_df({'MI': {'diagnostic_class': 'MI'}})
    result = _scp_to_superclasses({'MI': 0.0}, scp_df)
    assert result == []


def test_scp_to_superclasses_unknown_code_ignored():
    """SCP codes absent from the reference DataFrame are silently skipped."""
    scp_df = _make_scp_df({'NORM': {'diagnostic_class': 'NORM'}})
    result = _scp_to_superclasses({'NORM': 100.0, 'GHOST': 50.0}, scp_df)
    assert result == ['NORM']


def test_scp_to_superclasses_returns_sorted():
    """Output list must be sorted (consistent ordering for label vectors)."""
    scp_df = _make_scp_df({
        'MI':   {'diagnostic_class': 'MI'},
        'NORM': {'diagnostic_class': 'NORM'},
    })
    result = _scp_to_superclasses({'MI': 100.0, 'NORM': 50.0}, scp_df)
    assert result == sorted(result)


def test_scp_to_superclasses_nan_diagnostic_class():
    """Codes whose diagnostic_class is NaN in the reference are skipped."""
    scp_df = pd.DataFrame(
        {'diagnostic_class': [float('nan')]},
        index=['SR'],
    )
    result = _scp_to_superclasses({'SR': 100.0}, scp_df)
    assert result == []
