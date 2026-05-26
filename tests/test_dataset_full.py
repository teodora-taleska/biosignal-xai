import numpy as np
import pandas as pd
import torch
from unittest.mock import patch


def _make_df(n=4):
    rows = []
    for i in range(n):
        rows.append({
            "filename_lr": f"records100/00000/{i:05d}_lr",
            "strat_fold": 1,
            "label_vec": [1.0, 0.0, 0.0, 0.0, 0.0],
        })
    return pd.DataFrame(rows)


def _fake_rdsamp(path):
    signal = np.random.randn(1000, 12).astype(np.float32)
    meta   = {"fs": 100, "sig_name": ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]}
    return signal, meta


def test_dataset_length():
    from src.preprocessing.dataset_full import ECGDatasetFull
    df = _make_df(4)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds = ECGDatasetFull(df, "data/")
        assert len(ds) == 4


def test_dataset_signal_shape():
    from src.preprocessing.dataset_full import ECGDatasetFull
    df = _make_df(2)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds  = ECGDatasetFull(df, "data/")
        x, y = ds[0]
        assert x.shape == (12, 1000), f"Expected (12, 1000), got {x.shape}"
        assert x.dtype == torch.float32


def test_dataset_label_shape():
    from src.preprocessing.dataset_full import ECGDatasetFull
    df = _make_df(2)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds  = ECGDatasetFull(df, "data/")
        x, y = ds[0]
        assert y.shape == (5,), f"Expected (5,), got {y.shape}"
        assert y.dtype == torch.float32


def test_dataset_label_values():
    from src.preprocessing.dataset_full import ECGDatasetFull
    df = _make_df(1)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds  = ECGDatasetFull(df, "data/")
        _, y = ds[0]
        assert y[0].item() == 1.0   # NORM = 1
        assert y[1].item() == 0.0   # MI   = 0
