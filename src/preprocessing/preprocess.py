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