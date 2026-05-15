import torch
from torch.utils.data import Dataset
import numpy as np
import wfdb

from src.preprocessing.preprocess import bandpass_filter, normalize_signal


class ECGDatasetFull(Dataset):
    """
    Dataset for HuBERT-ECG and LeadwiseTransformer: returns FULL 10-second records.
    No windowing. One record = one training sample.

    Different from ECGDataset which returns 2.5-sec windows.
    Use this for HuBERT-ECG and LeadwiseTransformer experiments.

    Output shapes:
        x: (12, 1000)  ← 12 leads, 1000 time steps
        y: (5,)        ← multi-hot label vector
    """

    def __init__(self, dataframe, data_path):
        self.df        = dataframe.reset_index()
        self.data_path = data_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load full 10-second record
        raw, _ = wfdb.rdsamp(self.data_path + row['filename_lr'])

        # Preprocess: filter and normalize only, no windowing
        raw = bandpass_filter(raw)
        raw = normalize_signal(raw)
        raw = raw.astype(np.float32)     # (1000, 12)

        # Transpose → channels first
        x = torch.tensor(raw.T)          # (12, 1000)
        y = torch.tensor(row['label_vec'], dtype=torch.float32)  # (5,)

        return x, y
