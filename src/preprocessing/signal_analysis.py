"""
Signal-quality helpers used by 02_preprocessing_pipeline and by the
Streamlit app's signal-QA panel.

All functions accept ECG arrays shaped (T, leads) - the same convention as
wfdb.rdsamp() output and the rest of src/preprocessing/preprocess.py.
"""

import numpy as np
from scipy.signal import welch

from src.utils.config import CFG


def welch_psd(signal, fs=None, nperseg=256):
    """
    Welch power spectral density per lead.

    Returns
    -------
    freqs : (F,) ndarray  - frequency bins in Hz
    psd   : (F, leads) ndarray  - power spectral density per lead
    """
    fs = fs or CFG['data']['sampling_rate']
    nperseg = min(nperseg, signal.shape[0])
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, axis=0)
    return freqs, psd


def band_energy(signal, fs=None, band=(0.5, 40.0), nperseg=256):
    """
    Total spectral energy in a frequency band, averaged across leads.
    Useful for quantifying how much baseline wander or high-frequency
    noise the filter removed.
    """
    freqs, psd = welch_psd(signal, fs=fs, nperseg=nperseg)
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    df = freqs[1] - freqs[0]
    return float(psd[mask].sum(axis=0).mean() * df)


def baseline_wander_energy(signal, fs=None, nperseg=256):
    """
    Energy in the < 0.5 Hz band (baseline drift from breathing,
    electrode movement, etc).
    """
    fs = fs or CFG['data']['sampling_rate']
    return band_energy(signal, fs=fs, band=(0.0, 0.5), nperseg=nperseg)


def highfreq_noise_energy(signal, fs=None, nperseg=256):
    """
    Energy in the > 40 Hz band (muscle activity, 50/60 Hz powerline,
    high-frequency electronic noise).
    """
    fs = fs or CFG['data']['sampling_rate']
    nyq = fs / 2.0
    return band_energy(signal, fs=fs, band=(40.0, nyq), nperseg=nperseg)


def ecg_band_energy(signal, fs=None, nperseg=256):
    """
    Energy in the diagnostic ECG band [0.5, 40] Hz - what the bandpass
    filter is meant to preserve.
    """
    return band_energy(signal, fs=fs, band=(0.5, 40.0), nperseg=nperseg)


def signal_quality_summary(signal, fs=None):
    """
    Per-record summary stats useful for QA dashboards.
    Returns a flat dict ready to drop into JSON / a DataFrame row.
    """
    fs = fs or CFG['data']['sampling_rate']
    return {
        'amp_min':            float(signal.min()),
        'amp_max':            float(signal.max()),
        'amp_mean':           float(signal.mean()),
        'amp_std':            float(signal.std()),
        'per_lead_std_min':   float(signal.std(axis=0).min()),
        'per_lead_std_max':   float(signal.std(axis=0).max()),
        'per_lead_std_ratio': float(signal.std(axis=0).max() / max(signal.std(axis=0).min(), 1e-8)),
        'baseline_energy':    baseline_wander_energy(signal, fs=fs),
        'ecg_band_energy':    ecg_band_energy(signal, fs=fs),
        'highfreq_energy':    highfreq_noise_energy(signal, fs=fs),
    }
