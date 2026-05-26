"""Temporary test runner — deleted after use."""
import sys
sys.path.insert(0, r'D:\GitHub\biosignal-xai\.claude\worktrees\dreamy-sutherland-b22df1')

import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

PASS = []
FAIL = []

def ok(name):  PASS.append(name); print(f"  PASS  {name}")
def fail(name, e): FAIL.append(name); print(f"  FAIL  {name}: {e}")

# ── helpers ───────────────────────────────────────────────────────────────────

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
    return signal, {"fs": 100}

# ── ECGDatasetFull ────────────────────────────────────────────────────────────

from src.preprocessing.dataset_full import ECGDatasetFull

try:
    df = _make_df(4)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds = ECGDatasetFull(df, "data/")
        assert len(ds) == 4
    ok("test_dataset_length")
except Exception as e: fail("test_dataset_length", e)

try:
    df = _make_df(2)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds = ECGDatasetFull(df, "data/")
        x, y = ds[0]
        assert x.shape == (12, 1000), x.shape
        assert x.dtype == torch.float32
    ok("test_dataset_signal_shape")
except Exception as e: fail("test_dataset_signal_shape", e)

try:
    df = _make_df(2)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds = ECGDatasetFull(df, "data/")
        x, y = ds[0]
        assert y.shape == (5,), y.shape
        assert y.dtype == torch.float32
    ok("test_dataset_label_shape")
except Exception as e: fail("test_dataset_label_shape", e)

try:
    df = _make_df(1)
    with patch("src.preprocessing.dataset_full.wfdb.rdsamp", side_effect=_fake_rdsamp):
        ds = ECGDatasetFull(df, "data/")
        _, y = ds[0]
        assert y[0].item() == 1.0
        assert y[1].item() == 0.0
    ok("test_dataset_label_values")
except Exception as e: fail("test_dataset_label_values", e)

# ── HeartBERT helpers ─────────────────────────────────────────────────────────

from src.models.heartbert import _ecg_to_text

try:
    sig = np.linspace(-1.0, 1.0, 200)
    result = _ecg_to_text(sig, n_bins=20)
    tokens = result.split(" ")
    assert len(tokens) == 200, len(tokens)
    ok("test_ecg_to_text_length")
except Exception as e: fail("test_ecg_to_text_length", e)

try:
    sig = np.linspace(-1.0, 1.0, 50)
    result = _ecg_to_text(sig, n_bins=20)
    for token in result.split(" "):
        assert token in "ABCDEFGHIJKLMNOPQRST", token
    ok("test_ecg_to_text_alphabet")
except Exception as e: fail("test_ecg_to_text_alphabet", e)

try:
    import torch.nn as nn
    tiny = nn.Linear(8, 5)
    from src.models.heartbert import HeartBERTClassifier
    hb = HeartBERTClassifier(num_labels=5)
    hb.model = tiny
    result = hb.count_parameters()
    assert "total" in result
    assert "trainable" in result
    assert "percentage" in result
    ok("test_heartbert_count_parameters_keys")
except Exception as e: fail("test_heartbert_count_parameters_keys", e)

# ── ECG-PT helpers ────────────────────────────────────────────────────────────

from src.models.ecgpt import _ECGPatchTokenizer

try:
    tok = _ECGPatchTokenizer(patch_size=36, vocab_size=256)
    X   = np.random.randn(4, 1000).astype(np.float32)
    ids = tok.tokenize_batch(X)
    assert ids.shape == (4, 1000 // 36), ids.shape
    assert ids.dtype == torch.long
    ok("test_patch_tokenizer_shape")
except Exception as e: fail("test_patch_tokenizer_shape", e)

try:
    tok = _ECGPatchTokenizer(patch_size=36, vocab_size=256)
    X   = np.random.randn(2, 1000).astype(np.float32)
    ids = tok.tokenize_batch(X)
    assert ids.min().item() >= 0
    assert ids.max().item() <= 255
    ok("test_patch_tokenizer_range")
except Exception as e: fail("test_patch_tokenizer_range", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)
