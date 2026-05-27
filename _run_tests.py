"""Temporary test runner — deleted after use."""
import sys
sys.path.insert(0, r'D:\GitHub\biosignal-xai')

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

# ── load_adapter / predict_logits ─────────────────────────────────────────────

try:
    from src.models.heartbert import HeartBERTClassifier
    hb = HeartBERTClassifier()
    try:
        hb.load_adapter('some/path')
        fail("test_heartbert_load_adapter_requires_load", "no AssertionError raised")
    except AssertionError:
        ok("test_heartbert_load_adapter_requires_load")
except Exception as e:
    fail("test_heartbert_load_adapter_requires_load", e)

try:
    from src.models.ecgpt import ECGPTClassifier
    clf = ECGPTClassifier()
    try:
        clf.load_adapter('some/path')
        fail("test_ecgpt_load_adapter_requires_load", "no AssertionError raised")
    except AssertionError:
        ok("test_ecgpt_load_adapter_requires_load")
except Exception as e:
    fail("test_ecgpt_load_adapter_requires_load", e)

try:
    from src.models.hubert_ecg import HuBERTECGClassifier
    clf = HuBERTECGClassifier()
    try:
        clf.load_adapter('some/path')
        fail("test_hubert_load_adapter_requires_load", "no AssertionError raised")
    except AssertionError:
        ok("test_hubert_load_adapter_requires_load")
except Exception as e:
    fail("test_hubert_load_adapter_requires_load", e)

try:
    from unittest.mock import MagicMock
    from src.models.heartbert import HeartBERTClassifier
    hb = HeartBERTClassifier(num_labels=5)
    mock_tok_out = {
        'input_ids':      torch.zeros(2, 8, dtype=torch.long),
        'attention_mask': torch.ones(2, 8, dtype=torch.long),
    }
    hb.tokenizer = MagicMock(return_value=mock_tok_out)
    mock_out = MagicMock()
    mock_out.logits = torch.randn(2, 5) * 5.0
    hb.model = MagicMock(return_value=mock_out)
    hb.model.eval = MagicMock()
    X = np.random.randn(2, 1000).astype(np.float32)
    result = hb.predict_logits(X)
    assert result.shape == (2, 5), result.shape
    assert isinstance(result, np.ndarray)
    ok("test_heartbert_predict_logits_shape")
except Exception as e:
    fail("test_heartbert_predict_logits_shape", e)

try:
    from src.models.ecgpt import ECGPTClassifier, _ECGPatchTokenizer
    from unittest.mock import MagicMock
    clf = ECGPTClassifier(num_labels=5, patch_size=36)
    clf._tokenizer = _ECGPatchTokenizer(patch_size=36, vocab_size=256)
    mock_out = MagicMock()
    mock_out.logits = torch.randn(2, 5) * 5.0
    clf.model = MagicMock(return_value=mock_out)
    clf.model.eval = MagicMock()
    X = np.random.randn(2, 1000).astype(np.float32)
    result = clf.predict_logits(X)
    assert result.shape == (2, 5), result.shape
    assert isinstance(result, np.ndarray)
    ok("test_ecgpt_predict_logits_shape")
except Exception as e:
    fail("test_ecgpt_predict_logits_shape", e)

# ── explainability ────────────────────────────────────────────────────────────

try:
    from src.explainability.llm import build_ecg_prompt
    result_dict = {
        'predicted_classes':   ['NORM'],
        'class_probabilities': {'NORM': 0.9, 'MI': 0.1, 'STTC': 0.1, 'CD': 0.1, 'HYP': 0.1},
        'confidence_score':    0.9,
        'uncertainty':         None,
        'raw_logits':          [1.0, -1.0, -1.0, -1.0, -1.0],
        'uncertainty_level':   'not computed',
    }
    prompt = build_ecg_prompt(
        result_dict, saliency=None,
        lead_names=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'],
    )
    assert isinstance(prompt, str) and len(prompt) > 0
    ok("test_build_ecg_prompt_none_uncertainty")
except Exception as e:
    fail("test_build_ecg_prompt_none_uncertainty", e)

try:
    from src.models.heartbert import HeartBERTClassifier
    hb = HeartBERTClassifier()
    try:
        hb.get_attention_weights(np.zeros(1000, dtype=np.float32))
        fail("test_heartbert_get_attention_requires_load", "no AssertionError raised")
    except AssertionError:
        ok("test_heartbert_get_attention_requires_load")
except Exception as e:
    fail("test_heartbert_get_attention_requires_load", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    sys.exit(1)
