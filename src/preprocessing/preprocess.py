import numpy as np
from scipy.signal import butter, filtfilt

from src.utils.config import CFG

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """
    Remove baseline wander (below 0.5 Hz) and high-frequency noise (above 40 Hz).
    signal: numpy array (1000, 12) — time steps × leads
    """
    nyq = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # filtfilt = zero-phase filter (no time shift introduced)
    return filtfilt(b, a, signal, axis=0)


def normalize_signal(signal):
    """
    Z-score normalization per lead independently.
    Each of the 12 leads gets zero mean and unit variance.
    Prevents leads with higher voltage dominating the model.
    """
    mean = signal.mean(axis=0, keepdims=True)   # shape (1, 12)
    std  = signal.std(axis=0,  keepdims=True)   # shape (1, 12)
    std  = np.where(std < 1e-8, 1e-8, std)      # avoid division by zero
    return (signal - mean) / std


def normalize_minmax(signal, target_range=(-1.0, 1.0)):
    """
    Min-max scale each lead independently into target_range.
    Preserves waveform morphology but is sensitive to amplitude outliers
    (a single spike pulls the whole lead's scale).
    """
    lo, hi = target_range
    s_min = signal.min(axis=0, keepdims=True)
    s_max = signal.max(axis=0, keepdims=True)
    span  = s_max - s_min
    span  = np.where(span < 1e-8, 1e-8, span)
    return (signal - s_min) / span * (hi - lo) + lo


def normalize_robust(signal):
    """
    Per-lead robust z-score using median and MAD instead of mean/std.
    Less affected by motion artefacts or stray spikes than `normalize_signal`,
    but slightly more expensive.

    MAD is scaled by 1.4826 so that for Gaussian data the output matches
    standard z-score.
    """
    median = np.median(signal, axis=0, keepdims=True)
    mad    = np.median(np.abs(signal - median), axis=0, keepdims=True)
    mad    = np.where(mad < 1e-8, 1e-8, mad)
    return (signal - median) / (1.4826 * mad)


def create_windows(signal, window_size=CFG['data']['window_size'], stride=CFG['data']['stride']):
    """
    Slice a (1000, 12) signal into overlapping windows.

    window_size=250 → 2.5 seconds at 100Hz
    stride=125      → 50% overlap between windows
    Returns: list of arrays, each shape (250, 12)

    Why windowing?
    - Gives the model more training samples per patient
    - Transformer handles fixed-length sequences
    - Anomalies may only appear in part of the 10-second recording
    """
    windows = []
    start = 0
    while start + window_size <= signal.shape[0]:
        windows.append(signal[start : start + window_size])
        start += stride
    return windows  # typically 7 windows per 10-second record


def preprocess_record(signal):
    """
    Full pipeline for one ECG record.
    Input:  raw signal (1000, 12) from wfdb.rdsamp()
    Output: list of cleaned windows, each (250, 12)
    """
    signal = bandpass_filter(signal)     # remove noise
    signal = normalize_signal(signal)    # scale amplitudes
    signal = signal.astype(np.float32)  # save memory (float32 not float64)
    windows = create_windows(signal)     # split into chunks
    return windows
