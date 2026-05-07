import torch
from torch.utils.data import Dataset
import numpy as np
import wfdb

from src.preprocessing.preprocess import preprocess_record
from src.preprocessing.label_utils import SUPERCLASSES
from src.utils.config import CFG


class ECGDataset(Dataset):
    """
    Loads ECG records on-the-fly, preprocesses them, and returns
    (window_tensor, label_tensor) pairs ready for the DataLoader.

    Uses windowing — so one 10-second record becomes ~7 training samples.
    All windows from the same record share the same label.
    """

    def __init__(self, dataframe, data_path, preload=False):
        """
        dataframe:  filtered rows of ptbxl_database.csv with 'label_vec' column
        data_path:  path to ptb-xl folder (the one containing records100/)
        preload:    if True, loads all signals into RAM at init (faster training,
                    needs ~4GB RAM). If False, loads per batch (slower but safe).
        """
        self.df        = dataframe.reset_index()
        self.data_path = data_path
        self.preload   = preload

        # Build a flat index: (record_idx, window_idx) for every window
        # We need this because one record → multiple windows
        self.index = []

        if preload:
            print("Preloading all signals into RAM...")
            self.signals = []
            for i, row in self.df.iterrows():
                windows = self._load_and_process(row['filename_lr'])
                self.signals.append(windows)
                for w_idx in range(len(windows)):
                    self.index.append((i, w_idx))
            print(f"Done. Total windows: {len(self.index)}")
        else:
            # Just build the index by loading signals once to count windows
            for i, row in self.df.iterrows():
                raw, _ = wfdb.rdsamp(self.data_path + row['filename_lr'])
                n_windows = (raw.shape[0] - CFG['data']['window_size']) // CFG['data']['stride'] + 1
                for w_idx in range(n_windows):
                    self.index.append((i, w_idx))

    def _load_and_process(self, filename_lr):
        """Load one record from disk and return its windows."""
        raw, _ = wfdb.rdsamp(self.data_path + filename_lr)
        return preprocess_record(raw)  # list of (250, 12) arrays

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        record_idx, window_idx = self.index[idx]
        row = self.df.iloc[record_idx]

        # Get signal window
        if self.preload:
            window = self.signals[record_idx][window_idx]
        else:
            windows = self._load_and_process(row['filename_lr'])
            window  = windows[window_idx]

        # Shape: (250, 12) → transpose to (12, 250) for Conv1d/Transformer
        # 12 = channels (leads), 250 = sequence length
        x = torch.tensor(window.T, dtype=torch.float32)

        # Label: multi-hot vector (5,)
        y = torch.tensor(row['label_vec'], dtype=torch.float32)

        return x, y