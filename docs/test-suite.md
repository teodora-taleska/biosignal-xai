# Test Suite

**89 tests across 8 files. Run with:**
```bash
pytest tests/ -v
```

---

## Philosophy

The tests target the pure-logic layer of the project — functions that take inputs and produce outputs with no side effects and no external dependencies. The goal is to catch regressions in the scientific code (signal processing, label mapping, metrics, explainability) that would silently produce wrong results rather than crashing loudly.

Three things are deliberately **not** tested here:

- **Model training** — training loops depend on the full PTB-XL dataset, GPU, and hours of compute. Correctness is verified through the reported benchmark numbers (AUC 0.922, Fmax 0.746).
- **Streamlit UI** — the app layer is integration-tested manually against the live deployment.
- **Qwen2 generation** — requires downloading a 400 MB model and GPU memory. `build_ecg_prompt` (the pure prompt-building function) is tested; the generation call is not.

The PEFT Transformer models (HeartBERT, ECG-PT, HuBERT-ECG) are also not unit-tested. Their `predict_logits` and `load_adapter` interfaces require loading HuggingFace checkpoints that are not committed to the repo. The inference contract those models expose is validated indirectly through `test_pipeline.py` using mock classifiers.

---

## Test files

### `test_dataset_full.py` — 4 tests
**What:** `ECGDatasetFull` — the PyTorch Dataset that loads PTB-XL records.

**Why:** The dataset class is the entry point for all training data. If it returns the wrong shape or misaligns signal and label, every downstream model trains on garbage with no error.

**How:** `wfdb.rdsamp` is patched with a fake that returns a known random signal. Tests check `len(ds)`, signal shape `(12, 1000)`, label shape `(5,)`, and that the correct label position is set to 1.0.

---

### `test_preprocess.py` — 18 tests
**What:** The four normalisation functions (`bandpass_filter`, `normalize_signal`, `normalize_minmax`, `normalize_robust`), the windowing function (`create_windows`), and the full pipeline (`preprocess_record`).

**Why:** Preprocessing bugs are silent. A wrong normalisation axis, a divide-by-zero on a flat lead, or a window count off by one will not raise an error — they will just feed corrupted tensors to the model. These functions run on every record at both training time and inference time.

**How:** Tests use a fixed-seed random signal `(1000, 12)`. Mathematical properties are asserted directly:
- `normalize_signal` → per-lead mean ≈ 0, std ≈ 1
- `normalize_minmax` → all values within the target range
- `normalize_robust` → all values finite (median/MAD path cannot divide by zero)
- `bandpass_filter` → a constant (DC) signal is strongly attenuated
- `create_windows` → exactly 7 windows for a 1000-sample signal with `window_size=250, stride=125`

Flat-lead edge cases (std = 0, range = 0, MAD = 0) are explicitly tested on every normalisation function because real PTB-XL records occasionally contain leads with no signal.

---

### `test_label_utils.py` — 11 tests
**What:** `_to_label_vec` (SCP superclass list → 5-element float vector) and `_scp_to_superclasses` (raw SCP code dict + reference DataFrame → superclass list).

**Why:** The label pipeline is the bridge between PTB-XL's SCP-ECG coding system and the model's output space. A mismatch here — wrong index, wrong filtering of zero-likelihood codes, wrong handling of NaN diagnostic classes — produces silently mislabelled training data.

**How:** A minimal `scp_statements` DataFrame is constructed inline. Tests cover: correct index for each superclass, multi-label output, unknown-class passthrough, zero-likelihood exclusion, NaN `diagnostic_class` exclusion, and sorted output order.

---

### `test_metrics.py` — 12 tests
**What:** `compute_probs` (logits → sigmoid), `compute_auc` (macro AUC), `compute_fmax` (threshold-optimised F1), `compute_auprc` (area under PR curve).

**Why:** These are the benchmark metrics reported in the README and compared against Strodthoff et al. 2020. A bug in `compute_fmax` (e.g. sweeping the wrong axis, wrong `zero_division` handling) would produce numbers that look plausible but are wrong. The tests include a "perfect predictions" fixture to verify that each metric reaches its theoretical maximum.

**How:** Two fixtures — perfect predictions (prob 0.95 where label=1, 0.05 where label=0) and random predictions over 50 synthetic records. The perfect fixture verifies that AUC > 0.99 and Fmax > 0.99. Range tests confirm all metrics stay in [0, 1]. The Fmax ≥ F1@0.5 test encodes the mathematical invariant that the maximised threshold cannot be worse than a fixed one.

`bootstrap_ci` is not tested here — it runs 1000 resamples and takes ~5 seconds, making it unsuitable for a unit test. Its correctness is exercised in notebook 03.

---

### `test_saliency.py` — 10 tests
**What:** `compute_saliency` (gradient backprop → saliency map) and `top_salient_leads` (rank leads by mean saliency).

**Why:** The saliency map is the primary XAI output shown in the Streamlit app. Shape or sign errors would produce a misleading visualisation without any exception being raised. The fallback path (uniform saliency when backprop fails) must also work correctly so the app never crashes mid-inference.

**How:** A minimal `_TinyECGModel` (`Linear(12→5)` after mean pooling) is defined inside the test file. It is differentiable, so `compute_saliency` can actually run a backward pass. Tests verify shape `(12, 1000)`, non-negativity (absolute gradients), float32 dtype, and correct behaviour across all five class indices. A `_BrokenModel` that raises in `forward` tests the fallback path. `top_salient_leads` is tested with a hand-crafted saliency array where one lead is set to 1 and all others to 0 — the correct lead must rank first.

---

### `test_llm.py` — 8 tests
**What:** `build_ecg_prompt` — the function that formats an inference result dict into a Qwen2 user prompt.

**Why:** This function is pure string logic with no model calls. If it formats the wrong class name, drops the uncertainty value, or silently ignores a `None` saliency, the Qwen2 model receives a malformed prompt and generates a nonsensical narrative. Because Qwen2 is not loaded in tests, this is the only coverage available for the LLM integration path.

**How:** A `_base_result` helper constructs a minimal result dict matching the `ECGInferencePipeline.predict()` output contract. Tests assert string membership: the predicted class name appears in the prompt, `'not computed'` appears when `saliency=None`, `'N/A'` appears when `uncertainty=None`, `'+/-'` notation appears when `uncertainty_per_class` is present.

---

### `test_fcn_wang.py` — 9 tests
**What:** `FCNWang` forward pass and utilities (`count_parameters`, `save`/`load` are not tested as they require disk I/O).

**Why:** FCNWang is the production model. Its forward pass shape must be `(B, 5)` for all valid batch sizes and sequence lengths. The adaptive pooling head means it should accept any sequence length T — this is important because TTA adds noise to signals that may be padded or trimmed.

**How:** Tests instantiate `FCNWang()` directly (no checkpoint needed — random weights are fine for shape/dtype checks). A notable edge case: BatchNorm1d in the head raises with batch size 1 in training mode — this is a known PyTorch constraint. The single-record test therefore runs in `.eval()` mode, which is how inference is always done in the app.

---

### `test_pipeline.py` — 17 tests
**What:** `ECGInferencePipeline` (the FCNWang inference wrapper) and `HeartBERTPipeline` (the HeartBERT wrapper), including `_uncertainty_level`.

**Why:** The pipeline is the contract between the model and the Streamlit app. It must: return all expected keys, handle both `(12, 1000)` and `(1000, 12)` input shapes, fall back to argmax when no class clears the 0.5 threshold, and correctly expose or suppress uncertainty depending on the model type.

**How:** Two lightweight `nn.Module` subclasses (`_DeterministicModel` and `_UncertaintyModel`) are defined inline using `nn.Parameter` to store fixed logits. This avoids loading any checkpoint while still exercising real `torch.no_grad` and `.to(device)` paths. `HeartBERTPipeline` uses `unittest.mock.MagicMock` for the classifier, with a custom `predict_logits` function that captures its input — allowing the Lead II extraction test to assert the exact row passed to the classifier.

---

## Summary table

| File | Tests | Src module |
|---|---|---|
| `test_dataset_full.py` | 4 | `src/preprocessing/dataset_full.py` |
| `test_preprocess.py` | 18 | `src/preprocessing/preprocess.py` |
| `test_label_utils.py` | 11 | `src/preprocessing/label_utils.py` |
| `test_metrics.py` | 12 | `src/evaluation/metrics.py` |
| `test_saliency.py` | 10 | `src/explainability/saliency.py` |
| `test_llm.py` | 8 | `src/explainability/llm.py` |
| `test_fcn_wang.py` | 9 | `src/models/fcn_wang.py` |
| `test_pipeline.py` | 17 | `src/inference/pipeline.py` |
| **Total** | **89** | |

---

## What is not tested and why

| Area | Reason |
|---|---|
| Training loops (`train_baseline.py`, `train_peft.py`) | Require the full PTB-XL dataset and GPU; correctness is validated by the published benchmark results |
| PEFT model classes (HeartBERT, ECG-PT, HuBERT-ECG) | Require HuggingFace checkpoint downloads not committed to the repo; the inference contract is covered via mocks in `test_pipeline.py` |
| Qwen2 generation (`generate_explanation`) | Requires a 400 MB model download; `build_ecg_prompt` (the pure input-formatting step) is tested instead |
| `bootstrap_ci` | 1000-resample loop (~5 s); tested in notebook 03 against real model outputs |
| Streamlit views and components | Integration-tested against the live app |
