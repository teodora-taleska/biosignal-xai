"""
ECGDatasetAblation: flexible dataset for preprocessing ablation studies.

Unlike ECGDataset (hardwired to bandpass + z-score + 2.5-s windows),
this class accepts any combination of:
  - filter function (or None for raw)
  - normalisation function (or None for raw)
  - window size and stride
  - window aggregation mode for inference

Used in notebook 02_preprocessing_ablation to systematically compare
preprocessing choices on XResNet1D-101.

Predefined config registry
--------------------------
    from src.preprocessing.dataset_ablation import ABLATION_CONFIGS, ECGDatasetAblation

    cfg = ABLATION_CONFIGS['bandpass_zscore_250']
    ds  = ECGDatasetAblation(train_df, data_path, **cfg)
"""

from __future__ import annotations

from typing import Callable, Optional, List

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset

from src.preprocessing.preprocess import (
    bandpass_filter,
    normalize_signal,
    normalize_minmax,
    normalize_robust,
)
from src.utils.config import CFG

_DEFAULT_FS = CFG['data']['sampling_rate']   # 100 Hz → 1000 samples for 10 s



# Flexible dataset


class ECGDatasetAblation(Dataset):
    """
    Configurable ECG dataset for preprocessing ablation experiments.

    Parameters
    ----------
    dataframe   : filtered ptbxl_database DataFrame with 'label_vec' column
    data_path   : path to the PTB-XL root folder (containing records100/)
    filter_fn   : applied first; None → raw signal (no filtering)
    norm_fn     : applied after filtering; None → no normalisation
    window_size : samples per window; 1000 = full record (no windowing)
    stride      : hop size between windows; ignored when window_size == 1000

    Output shapes
    -------------
    x : (12, window_size) float32  — leads × time
    y : (5,) float32               — multi-hot label

    Notes
    -----
    When window_size == record_length (1000 @ 100 Hz), a single window
    covering the full record is returned — equivalent to ECGDatasetFull.
    The XResNet1D-101 adaptive-pooling head handles any sequence length.
    """

    def __init__(
        self,
        dataframe,
        data_path:   str,
        filter_fn:   Optional[Callable] = bandpass_filter,
        norm_fn:     Optional[Callable] = normalize_signal,
        window_size: int  = 250,
        stride:      int  = 125,
    ):
        self.df          = dataframe.reset_index(drop=True)
        self.data_path   = data_path
        self.filter_fn   = filter_fn
        self.norm_fn     = norm_fn
        self.window_size = window_size
        self.stride      = stride if window_size < 1000 else 1000  # full record → 1 window

        # Build flat index of (record_idx, window_idx) pairs
        self.index: List[tuple] = []
        for i in range(len(self.df)):
            n = self._count_windows()
            for w in range(n):
                self.index.append((i, w))

    def _count_windows(self) -> int:
        """Number of windows for a signal of length 1000."""
        return max(1, (1000 - self.window_size) // self.stride + 1)

    def _preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Apply filter_fn then norm_fn to a (1000, 12) array."""
        sig = raw.copy()
        if self.filter_fn is not None:
            sig = self.filter_fn(sig)
        if self.norm_fn is not None:
            sig = self.norm_fn(sig)
        return sig.astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        rec_idx, win_idx = self.index[idx]
        row = self.df.iloc[rec_idx]

        raw, _ = wfdb.rdsamp(self.data_path + row['filename_lr'])
        sig    = self._preprocess(raw)                       # (1000, 12)

        start  = win_idx * self.stride
        window = sig[start : start + self.window_size]      # (W, 12)
        x = torch.tensor(window.T, dtype=torch.float32)     # (12, W)
        y = torch.tensor(row['label_vec'], dtype=torch.float32)

        return x, y



# Record-level aggregation for inference


def aggregate_window_predictions(
    probs:   torch.Tensor,
    index:   list,
    n_records: int,
    mode:    str = 'mean',
) -> torch.Tensor:
    """
    Aggregate per-window sigmoid probabilities to per-record predictions.

    Parameters
    ----------
    probs     : (N_windows, C) sigmoid probabilities
    index     : dataset.index — list of (record_idx, window_idx) tuples
    n_records : total number of records
    mode      : 'mean'  — average across windows (default)
                'max'   — element-wise maximum across windows
                          (equivalent to predict-any-window-positive)

    Returns
    -------
    (n_records, C) float tensor
    """
    C         = probs.shape[1]
    acc       = torch.zeros(n_records, C)
    counts    = torch.zeros(n_records, 1)

    for i, (rec_idx, _) in enumerate(index):
        acc[rec_idx] += probs[i]
        counts[rec_idx] += 1

    if mode == 'mean':
        return acc / counts.clamp(min=1)
    elif mode == 'max':
        rec_max = torch.zeros(n_records, C)
        for i, (rec_idx, _) in enumerate(index):
            rec_max[rec_idx] = torch.maximum(rec_max[rec_idx], probs[i])
        return rec_max
    else:
        raise ValueError(f"Unknown aggregation mode: {mode!r}. Use 'mean' or 'max'.")



# Pre-built ablation configs (used directly in notebook cells)


ABLATION_CONFIGS = {
    # ---------- filter ablation (z-score norm, 2.5-s windows) ----------
    'raw_zscore_250': dict(
        filter_fn=None, norm_fn=normalize_signal,
        window_size=250, stride=125,
    ),
    'bandpass_zscore_250': dict(
        filter_fn=bandpass_filter, norm_fn=normalize_signal,
        window_size=250, stride=125,
    ),

    # ---------- norm ablation (bandpass filter, 2.5-s windows) ----------
    'bandpass_none_250': dict(
        filter_fn=bandpass_filter, norm_fn=None,
        window_size=250, stride=125,
    ),
    'bandpass_minmax_250': dict(
        filter_fn=bandpass_filter, norm_fn=normalize_minmax,
        window_size=250, stride=125,
    ),
    'bandpass_robust_250': dict(
        filter_fn=bandpass_filter, norm_fn=normalize_robust,
        window_size=250, stride=125,
    ),

    # ---------- window-size ablation (bandpass + z-score) ----------
    'bandpass_zscore_500': dict(
        filter_fn=bandpass_filter, norm_fn=normalize_signal,
        window_size=500, stride=250,
    ),
    'bandpass_zscore_1000': dict(
        filter_fn=bandpass_filter, norm_fn=normalize_signal,
        window_size=1000, stride=1000,
    ),

    # ---------- raw baseline (paper's approach — no preprocessing) ----------
    'raw_none_1000': dict(
        filter_fn=None, norm_fn=None,
        window_size=1000, stride=1000,
    ),
}

# Human-readable labels for plots
ABLATION_LABELS = {
    'raw_zscore_250':      'Raw + Z-score, 2.5 s',
    'bandpass_zscore_250': 'Bandpass + Z-score, 2.5 s  ★',
    'bandpass_none_250':   'Bandpass only, 2.5 s',
    'bandpass_minmax_250': 'Bandpass + Min-max, 2.5 s',
    'bandpass_robust_250': 'Bandpass + Robust, 2.5 s',
    'bandpass_zscore_500': 'Bandpass + Z-score, 5 s',
    'bandpass_zscore_1000':'Bandpass + Z-score, 10 s',
    'raw_none_1000':       'Raw (paper baseline), 10 s',
}
