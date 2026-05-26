# Transfer Training with PEFT (LoRA / DoRA)

## Overview

Three pretrained ECG foundation models are fine-tuned on PTB-XL 5-class
multi-label classification using parameter-efficient fine-tuning (PEFT).
No full fine-tuning is performed — only adapter layers are trained.

**Why PEFT?**  
Full fine-tuning of 93 M+ parameter models on a single GPU requires unfeasible
memory and risks catastrophic forgetting. LoRA and DoRA constrain updates to
< 1% of parameters, preserving pretrained representations while adapting to the
downstream task.

**LoRA vs DoRA:**

| Strategy | What is trained | Extra params vs LoRA | When to prefer |
|---|---|---|---|
| LoRA r=8 | Low-rank `BA` injected into Q/V attention | — | Fast convergence, tight VRAM budget |
| DoRA r=8 | LoRA direction + per-row magnitude scalar | ~37 k | More stable than LoRA when LR is higher |

Both are controlled by a single `apply_peft(use_dora=False/True)` call.

---

## Models

**HeartBERT** (`Bayesiano/HeartBERT`)  
RoBERTa encoder pretrained on ECG amplitude quantised to letter strings (MIT-BIH,
PTB-XL, European ST-T). Accepts single-lead (Lead II) input; the model converts
raw amplitudes to a 20-bin letter string before tokenisation. LoRA targets the
`query` and `value` projections in every encoder attention block.

**ECG-PT** (`Tconnector/ecg-pt`, falls back to GPT-2)  
Causal (decoder-only) Transformer pretrained with self-supervised ECG reconstruction.
Originally unsupervised; adapted here with a sequence-classification head on the last
token. Accepts single-lead input split into 36-sample patches quantised to token IDs.
LoRA targets the fused `c_attn` (QKV) projection of each GPT block.

**HuBERT-ECG** (`Edoardo-BS/hubert-ecg-base`)  
HuBERT-style foundation model pretrained on 9.1 M 12-lead ECGs across 164 conditions
(CC BY-NC 4.0, research use only). Accepts full 12-lead input at 100 Hz (1 000 samples).
The mean-pooled encoder output is projected to 5 logits via `LayerNorm → Linear`.
LoRA targets `q_proj` and `v_proj` in the HuBERT transformer stack.

---

## How to run

Open and run **`notebooks/04_transfer_training_peft.ipynb`** top-to-bottom.

**Prerequisites:**
- PTB-XL dataset in `data/` (see `docs/data-loading.md`)
- Notebooks 01–03 completed (for label files; training does not depend on
  their results directly)
- Internet access on first run (HuggingFace model download, ~100–350 MB each)

**GPU requirements:**

| Model | Approx VRAM | Approx time per experiment |
|---|---|---|
| HeartBERT (LoRA/DoRA) | ~3 GB | ~30–60 min (RTX 4060) |
| ECG-PT (LoRA/DoRA) | ~2 GB | ~20–40 min |
| HuBERT-ECG base (LoRA/DoRA) | ~5–7 GB | ~1.3–1.7 h |

Set `TRANSFORMERS_OFFLINE=1` after the first download to avoid network calls.

---

## Results directory layout

```
results/
  heartbert_lora_r8/
    best_adapter/         ← PEFT adapter weights (save_pretrained format)
    history.json          ← per-epoch train/val loss and AUC
    profiling.json        ← wall-clock time, GPU memory, param counts
  heartbert_dora_r8/      ← same structure
  ecgpt_lora_r8/
  ecgpt_dora_r8/
  hubert_ecg_lora_r8/
  hubert_ecg_dora_r8/
```

Adapter weights are small (< 5 MB each). Full backbone weights are NOT saved —
reload the HuggingFace checkpoint and re-apply the adapter to restore a model.
