"""Unit tests for src/preprocessing/preprocess.py."""

import numpy as np
import pytest

from src.preprocessing.preprocess import (
    bandpass_filter,
    normalize_signal,
    normalize_minmax,
    normalize_robust,
    create_windows,
    preprocess_record,
)

# 1000 samples × 12 leads — matches a real PTB-XL record at 100 Hz
SIGNAL = np.random.default_rng(0).standard_normal((1000, 12)).astype(np.float32)


# ── bandpass_filter ───────────────────────────────────────────────────────────

def test_bandpass_filter_preserves_shape():
    """Output shape must exactly match input shape."""
    out = bandpass_filter(SIGNAL)
    assert out.shape == SIGNAL.shape


def test_bandpass_filter_output_is_float():
    """Filtered output should be a floating-point array."""
    out = bandpass_filter(SIGNAL)
    assert np.issubdtype(out.dtype, np.floating)


def test_bandpass_filter_attenuates_dc():
    """A DC-offset signal (constant across time) should be strongly reduced by the 0.5 Hz high-pass."""
    dc = np.ones((1000, 12), dtype=np.float32) * 5.0
    out = bandpass_filter(dc)
    assert np.abs(out).max() < 0.1


# ── normalize_signal (z-score) ────────────────────────────────────────────────

def test_normalize_signal_zero_mean():
    """Each lead should have mean ≈ 0 after z-score normalisation."""
    out = normalize_signal(SIGNAL)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-5)


def test_normalize_signal_unit_std():
    """Each lead should have std ≈ 1 after z-score normalisation."""
    out = normalize_signal(SIGNAL)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-5)


def test_normalize_signal_flat_lead_no_crash():
    """A completely flat lead (std = 0) must not raise a divide-by-zero error."""
    flat = SIGNAL.copy()
    flat[:, 0] = 0.0
    out = normalize_signal(flat)
    assert np.all(np.isfinite(out))


def test_normalize_signal_preserves_shape():
    out = normalize_signal(SIGNAL)
    assert out.shape == SIGNAL.shape


# ── normalize_minmax ──────────────────────────────────────────────────────────

def test_normalize_minmax_default_range():
    """With the default target_range=(-1, 1) every value must land in [-1, 1]."""
    out = normalize_minmax(SIGNAL)
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_normalize_minmax_custom_range():
    """Output should be clipped to an arbitrary target range."""
    out = normalize_minmax(SIGNAL, target_range=(0.0, 1.0))
    assert out.min() >= -1e-6
    assert out.max() <= 1.0 + 1e-6


def test_normalize_minmax_flat_lead_no_crash():
    """Flat lead (range = 0) must not crash."""
    flat = SIGNAL.copy()
    flat[:, 3] = 2.5
    out = normalize_minmax(flat)
    assert np.all(np.isfinite(out))


# ── normalize_robust ──────────────────────────────────────────────────────────

def test_normalize_robust_preserves_shape():
    out = normalize_robust(SIGNAL)
    assert out.shape == SIGNAL.shape


def test_normalize_robust_finite():
    """No NaNs or infs, even for well-behaved Gaussian input."""
    out = normalize_robust(SIGNAL)
    assert np.all(np.isfinite(out))


def test_normalize_robust_flat_lead_no_crash():
    """Flat lead (MAD = 0) must not produce NaNs."""
    flat = SIGNAL.copy()
    flat[:, 5] = 0.0
    out = normalize_robust(flat)
    assert np.all(np.isfinite(out))


# ── create_windows ────────────────────────────────────────────────────────────

def test_create_windows_count():
    """1000-sample signal with window=250 stride=125 should produce exactly 7 windows."""
    windows = create_windows(SIGNAL, window_size=250, stride=125)
    assert len(windows) == 7


def test_create_windows_shape():
    """Each window must be (window_size, 12)."""
    windows = create_windows(SIGNAL, window_size=250, stride=125)
    for w in windows:
        assert w.shape == (250, 12)


def test_create_windows_no_overflow():
    """No window should reference samples beyond the signal length."""
    windows = create_windows(SIGNAL, window_size=250, stride=125)
    assert len(windows) > 0  # sanity
    # last window ends exactly at or before 1000
    assert len(windows) * 125 + 250 - 125 <= 1000 + 125


# ── preprocess_record ─────────────────────────────────────────────────────────

def test_preprocess_record_returns_list():
    """Full pipeline should return a non-empty list."""
    windows = preprocess_record(SIGNAL)
    assert isinstance(windows, list)
    assert len(windows) > 0


def test_preprocess_record_window_dtype():
    """Windows from the full pipeline must be float32."""
    windows = preprocess_record(SIGNAL)
    assert windows[0].dtype == np.float32
