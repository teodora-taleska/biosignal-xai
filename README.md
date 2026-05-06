# BioSignal-XAI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20PEFT-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

**Deep learning system for real-time ECG anomaly detection with explainable AI, built on Transformer architectures and Parameter-Efficient Fine-Tuning (PEFT).**

---

## Overview

BioSignal-XAI is an academic research project exploring the application of PEFT methods — specifically LoRA, DoRA, and QLoRA — to time-series Transformer architectures (PatchTST) for 12-lead ECG classification on the PTB-XL dataset. The system is designed to be explainable by default: predictions are paired with aleatoric uncertainty estimates and LLM-generated plain-English clinical summaries.

The longer-term goal is a multi-tenant SaaS clinical decision support platform where clinicians can upload wearable or hospital-grade ECG data and receive structured anomaly reports with confidence-calibrated predictions reviewed by a physician before delivery.

**This is a research prototype.** It is not a certified medical device and is not intended for clinical diagnosis.

---

## Research Contributions

- **PEFT on biomedical time series** — Systematic empirical comparison of LoRA and DoRA applied to PatchTST for ECG classification across five diagnostic superclasses, with analysis of parameter efficiency vs. performance trade-offs.
- **Aleatoric uncertainty quantification** — An uncertainty head that separates inherent signal ambiguity from model ignorance, providing per-prediction confidence scores that downstream consumers (e.g., a reviewing clinician) can act on.
- **LLM-generated clinical explanations** — Integration of Qwen-3 (1.7B) via vLLM to translate model outputs — predicted class, confidence, and saliency — into concise, structured plain-English summaries suitable for non-specialist review.

---

## Architecture Overview

```
PTB-XL ECG Records (WFDB)
        │
        ▼
┌───────────────────┐
│   Preprocessing   │  Normalization, windowing, lead selection
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   PatchTST        │  Time-series Transformer (patch-based tokenization)
│   + LoRA / DoRA   │  PEFT adapter layers — ~1–5% of total parameters trained
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────────────┐
│ Class  │  │ Uncertainty Head │  Aleatoric variance estimation
│ Logits │  └──────────────────┘
└────┬───┘
     │
     ▼
┌─────────────────────┐
│  Qwen-3 (1.7B)      │  vLLM inference — generates clinical text
│  Explanation Module │  conditioned on prediction + saliency
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  FastAPI Backend    │  REST API, multi-tenant auth, report storage
│  React Frontend     │  Upload interface, report viewer, dashboard
└─────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8 GB VRAM minimum; developed on RTX 4060 Laptop GPU)
- PTB-XL dataset (see [Dataset Setup](#dataset-setup))

### Installation

```bash
git clone https://github.com/your-username/biosignal-xai.git
cd biosignal-xai

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes
pip install wfdb scipy scikit-learn pandas numpy matplotlib tqdm
pip install jupyter
```

### Run notebooks

```bash
jupyter notebook notebooks/
```

---

## Project Structure

```
biosignal-xai/
├── app/                        # FastAPI backend + React frontend (Phase 5, planned)
├── configs/                    # YAML hyperparameter configs per experiment
├── data/                       # Local only — gitignored; see Dataset Setup
│   ├── records100/             # WFDB signal files at 100 Hz
│   ├── ptbxl_database.csv      # 21,837 records with metadata and labels
│   └── scp_statements.csv      # 71 SCP diagnostic codes with superclass mapping
├── docs/                       # Design docs, architecture notes
├── experiments/                # Saved experiment configs and run logs
├── notebooks/                  # Sequential research notebooks (see below)
├── results/                    # Model checkpoints, metrics, plots
└── src/
    ├── explainability/         # Saliency, attention rollout, LLM interface
    ├── inference/              # Real-time inference pipeline
    ├── models/
    │   └── baseline_cnn.py     # Baseline CNN classifier
    ├── preprocessing/
    │   ├── dataset.py          # PyTorch Dataset for PTB-XL
    │   ├── label_utils.py      # SCP code parsing and superclass mapping
    │   └── preprocess.py       # Normalization and windowing
    ├── training/
    │   └── train.py            # Training loop with logging and checkpointing
    └── utils/
        └── metrics.py          # AUC, F1, per-class metrics
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_loading.ipynb` | Download PTB-XL, inspect raw WFDB records, explore metadata and label distribution |
| `02_preprocessing.ipynb` | Normalization, windowing, train/val/test split by `strat_fold`, dataset statistics |
| `03_baseline_model.ipynb` | Train and evaluate the baseline CNN; establish AUC/F1 reference numbers |
| `04_transformer_peft.ipynb` | PatchTST + LoRA vs DoRA comparison; adapter configuration and training |
| `04b_leadwise_transformer.ipynb` | Per-lead Transformer variant; ablation on lead subsets |
| `05_uncertainty.ipynb` | Aleatoric uncertainty head; calibration curves; reliability diagrams |
| `06_explainability.ipynb` | Attention rollout, gradient saliency, Qwen-3 explanation generation |
| `07_realtime_demo.ipynb` | Simulated real-time streaming inference demo |

---

## Dataset Setup

### Option A — PhysioNet (official)

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

Or visit [https://physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/) and download manually. Free registration required.

### Option B — Kaggle mirror

Search for **PTB-XL ECG Dataset** on Kaggle. Several public mirrors exist that do not require PhysioNet registration.

### Expected layout after download

```
data/
├── ptbxl_database.csv
├── scp_statements.csv
└── records100/
    ├── 00000/
    │   ├── 00001_lr.dat
    │   ├── 00001_lr.hea
    │   └── ...
    └── ...
```

Set the `path` variable at the top of each notebook to point to your `data/` directory.

---

## Hardware Requirements

| Component | Minimum | Developed on |
|---|---|---|
| GPU VRAM | 8 GB | NVIDIA RTX 4060 Laptop (8 GB) |
| RAM | 16 GB | — |
| Storage | ~6 GB (dataset) | — |
| CUDA | 11.8+ | 12.1 |

QLoRA (4-bit quantization via `bitsandbytes`) is used throughout Phase 2 to keep the full fine-tuning pipeline within 8 GB VRAM. Full-precision training requires a larger GPU.

---

## Results

*Experiments in progress. Table will be updated as Phase 2 concludes.*

| Model | PEFT Method | AUC (macro) | F1 (macro) | Trainable Params |
|---|---|---|---|---|
| Baseline CNN | — | — | — | 100% |
| PatchTST | LoRA | — | — | — |
| PatchTST | DoRA | — | — | — |
| PatchTST | QLoRA | — | — | — |

---

## Roadmap

- [x] **Phase 1** — Data pipeline, preprocessing, baseline CNN
- [ ] **Phase 2** — PatchTST + LoRA/DoRA comparison; lead-wise Transformer ablation *(in progress)*
- [ ] **Phase 3** — Aleatoric uncertainty head; calibration analysis
- [ ] **Phase 4** — Qwen-3 (1.7B) explanation module via vLLM
- [ ] **Phase 5** — FastAPI + React multi-tenant SaaS interface

---

## Disclaimer

BioSignal-XAI is a **clinical decision support research prototype**. It is not a certified medical device under MDR, FDA 510(k), or any equivalent regulatory framework. Outputs from this system must not be used as the sole basis for clinical diagnosis or treatment decisions. All model outputs are intended to assist — not replace — qualified medical professionals.

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@misc{biosignalxai2026,
  author       = {Taleska, Teodora},
  title        = {BioSignal-XAI: ECG Anomaly Detection with Explainable PEFT-based Transformers},
  year         = {2026},
  howpublished = {\url{https://github.com/teodora-taleska/biosignal-xai}},
  note         = {University research project}
}
```

[//]: # (---)

[//]: # ()
[//]: # (## License)

[//]: # ()
[//]: # (MIT License — see [LICENSE]&#40;LICENSE&#41; for details.)
