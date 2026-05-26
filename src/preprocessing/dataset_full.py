import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset


class ECGDatasetFull(Dataset):
    """
    Full 10-second 12-lead ECG dataset. No windowing or preprocessing.

    Returns
    -------
    x : (12, 1000) float32 tensor  — leads × time samples
    y : (5,)       float32 tensor  — multi-hot superclass label
    """

    def __init__(self, df, data_path: str):
        self.df        = df.reset_index(drop=True)
        self.data_path = data_path.rstrip("/")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        signal, _ = wfdb.rdsamp(f"{self.data_path}/{row['filename_lr']}")
        x = torch.tensor(signal.T.astype(np.float32))                   # (12, 1000)
        y = torch.tensor(np.array(row["label_vec"], dtype=np.float32))  # (5,)
        return x, y
