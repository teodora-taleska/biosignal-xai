# BioSignal-XAI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%20%7C%20PEFT-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**ECG anomaly detection with explainable AI: FCN-Wang production model, PEFT fine-tuned Transformer comparison, gradient saliency, TTA uncertainty, and an interactive Streamlit clinical demo.**

> **Live demo:** [biosignalxai.streamlit.app](https://biosignalxai.streamlit.app) *(Streamlit Community Cloud)*

---

## Overview

BioSignal-XAI is an academic research project for 12-lead ECG classification on the PTB-XL dataset across five diagnostic superclasses (NORM, MI, STTC, CD, HYP).

The project has two layers:

1. **Research pipeline**: trains and evaluates multiple model families: a compact FCN baseline, and three Transformer architectures (HeartBERT, ECG-PT, HuBERT-ECG) fine-tuned with LoRA and DoRA via PEFT. The FCN-Wang baseline outperforms all PEFT models on this dataset.

2. **Interactive app**: a four-tab Streamlit app serving the FCN-Wang model live. It includes a real-time ECG monitor, gradient saliency explainability, TTA-based uncertainty estimates, and Qwen2-0.5B-Instruct clinical narrative generation.

**This is a research prototype.** It is not a certified medical device and must not be used for clinical diagnosis.

---

## Results

| Model | PEFT    | AUC (macro) | Fmax | Trainable params |
|---|---------|---|---|---|
| **FCN-Wang** *(production)* | /       | **0.922** | **0.746** | 309 k (100%) |
| HeartBERT | LoRA r8 | 0.820 | 0.594 | 889 k |
| HeartBERT | DoRA r8 | 0.817 | 0.589 | 908 k |
| ECG-PT | LoRA r8 | 0.619 | 0.412 | 299 k |
| ECG-PT | DoRA r8 | 0.616 | 0.417 | 326 k |
| HuBERT-ECG | LoRA r8 | 0.549 | 0.399 | 295 k |
| HuBERT-ECG | DoRA r8 | 0.516 | 0.399 | 313 k |

> PTB-XL benchmark (Strodthoff et al.): AUC 0.925, Fmax 0.643. FCN-Wang matches the paper benchmark at a 1.26 MB checkpoint.

FCN-Wang latency: **~1 ms/record** on GPU (vs. 21 ms for HeartBERT on the same hardware).

---

## Architecture

```
PTB-XL ECG (WFDB, 100 Hz, 12 leads, 10 s)
        │
        ▼
┌───────────────────────┐
│   Preprocessing       │  Bandpass filter (0.5–40 Hz), transpose to (12, 1000)
└──────────┬────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌──────────────────────────────────┐
│ FCN-Wang│  │  Transformer + PEFT (comparison) │
│ 309k    │  │  HeartBERT / ECG-PT / HuBERT-ECG │
│ params  │  │  LoRA / DoRA  (rank 8)            │
└────┬────┘  └──────────────────────────────────┘
     │
     ├── Sigmoid → 5-class multi-label prediction
     │
     ├── TTA Uncertainty (20 forward passes, σ=0.05 mV noise)
     │   └── per-class probability ± std
     │
     └── Gradient Saliency → top-3 salient leads
             │
             ▼
     Qwen2-0.5B-Instruct
     └── 3-5 sentence clinical narrative
```

---

## Streamlit App

Four tabs, served from the FCN-Wang checkpoint:

| Tab | Description                                                                                                                                                                                                |
|-----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Explorer** | Browse PTB-XL: class distribution over 21 k records, filterable curated-200 table, per-record waveform and prediction                                                                                      |
| **Preprocessing Pipeline** | Step-by-step educational visualisation: raw signal → bandpass filter → normalisation → sliding windows                                                                                                     |
| **Interactive Demo** | Select a patient from the 200-record test subset, watch the ECG scroll in real time, run inference, inspect confidence + uncertainty, then run XAI analysis (gradient saliency + Qwen2 clinical narrative) |
| **Real-Time Monitor** | Simulated streaming monitor: sliding window advances through a 10-second record, FCN-Wang classifies at each step, anomaly alert updates live                                                              |

### Cloud mode

On Streamlit Community Cloud, Qwen2 is never loaded at runtime, clinical narratives are served from a pre-generated cache (`app/data/narratives_cache.json`). All other tabs run fully live.

---

## Installation

### Conda (local development)
```bash
conda env create -f environment_local.yml
conda activate biosignal-xai
```

### pip
```bash
pip install -r requirements.txt
```

### Hardware
Developed on NVIDIA RTX 4060 Laptop GPU (8 GB VRAM, CUDA 12.1).
The app runs on CPU (Streamlit Cloud), inference is slower but functional.

---

## Dataset Setup

PTB-XL is a free public dataset from PhysioNet (21,799 10-second 12-lead ECGs).

```bash
# PhysioNet (official)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

Or download manually from [physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/) (free registration required).

Expected layout:
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

The full dataset (~2 GB) is gitignored. A curated 200-record subset for the app is embedded in `app/data/signals/` (~5 MB).

---

## Running Locally

```bash
# 1. Build app caches (first time only, needs full data/ directory)
python app/data/cache.py

# 2. Launch the Streamlit app
streamlit run app/main.py
```

For Qwen2 clinical narratives locally (requires ~400 MB download on first run):
```bash
python app/data/cache_narratives.py   # pre-generates narratives for cloud deployment
```

---

## Project Structure

```
biosignal-xai/
├── app/
│   ├── main.py                     # Streamlit entry point
│   ├── model.py                    # Cached model + inference wrappers
│   ├── components/                 # Reusable UI components
│   │   ├── confidence_gauge.py
│   │   ├── ecg_animation.py        # Animated ECG monitor (canvas + Web Audio)
│   │   ├── ecg_viewer.py           # Static 12-lead + saliency overlay
│   │   └── patient_card.py
│   ├── data/
│   │   ├── cache.py                # Build curated_200.json + predictions_cache.json
│   │   ├── cache_narratives.py     # Pre-generate Qwen2 narratives for cloud
│   │   ├── curated_200.json        # 200-record curated test subset index
│   │   ├── dataset_stats.json      # Pre-computed full-dataset statistics
│   │   ├── narratives_cache.json   # Pre-generated clinical narratives (cloud mode)
│   │   ├── predictions_cache.json  # FCN-Wang predictions for all 200 records
│   │   └── signals/                # Embedded WFDB signal files (~5 MB)
│   ├── model/
│   │   └── checkpoint.pt           # FCN-Wang weights (~1.2 MB)
│   └── views/                      # One file per tab
│       ├── data_explorer.py
│       ├── interactive_demo.py
│       ├── preprocessing.py
│       └── realtime_monitor.py
├── configs/
│   └── config.yaml                 # All hyperparameters and paths
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_ablation.ipynb
│   ├── 03_preprocessing_pipeline.ipynb
│   ├── 04_transfer_training_peft.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_explainability.ipynb
│   └── 07_inference_pipeline.ipynb
├── results/                        # Checkpoints, metrics JSON, training history
├── src/
│   ├── evaluation/                 # AUC, Fmax, AUPRC, bootstrap CI
│   ├── explainability/
│   │   ├── llm.py                  # Qwen2 prompt builder + generate_explanation
│   │   ├── saliency.py             # Gradient saliency (vanilla backprop)
│   │   └── uncertainty.py          # TTA uncertainty (20-run Monte Carlo)
│   ├── inference/
│   │   └── pipeline.py             # ECGInferencePipeline + HeartBERTPipeline
│   ├── models/
│   │   ├── fcn_wang.py             # FCN-Wang (3-layer FCN, production model)
│   │   ├── heartbert.py            # RoBERTa-base + PEFT classification head
│   │   ├── ecgpt.py                # Patch-based Transformer + PEFT
│   │   └── hubert_ecg.py           # HuBERT encoder + PEFT
│   ├── preprocessing/
│   │   ├── dataset.py              # PTB-XL PyTorch Dataset
│   │   ├── label_utils.py          # SCP code → superclass mapping
│   │   └── preprocess.py           # Bandpass filter, normalisation
│   ├── training/
│   │   ├── train_baseline.py       # FCN-Wang training loop
│   │   └── train_peft.py           # PEFT fine-tuning loop
│   └── utils/
│       ├── config.py               # YAML config loader
│       └── seed.py                 # Reproducibility seed
└── requirements.txt                # pip dependencies (cloud deployment)
```

---

## Notebooks

| Notebook | Description                                                                                                              |
|---|--------------------------------------------------------------------------------------------------------------------------|
| `01_data_exploration.ipynb` | PTB-XL dataset inspection: label distribution, signal quality, fold structure, per-class statistics                      |
| `02_preprocessing_ablation.ipynb` | Ablation over preprocessing configurations (filter type × normalisation × window size); selects best config for training |
| `03_preprocessing_pipeline.ipynb` | FCN-Wang training and full evaluation: AUC, Fmax, AUPRC with 1000-iter bootstrap CI                                      |
| `04_transfer_training_peft.ipynb` | LoRA and DoRA fine-tuning of HeartBERT, ECG-PT, and HuBERT-ECG; adapter configuration and training curves                |
| `05_model_comparison.ipynb` | Head-to-head comparison of all seven model variants; latency, parameter count, AUC, Fmax                                 |
| `06_explainability.ipynb` | Gradient saliency maps per lead and per class; Qwen2-0.5B-Instruct clinical narrative generation                         |
| `07_inference_pipeline.ipynb` | Production inference pipeline; TTA uncertainty (per-class ±); latency benchmarks (mean / median / p95)                   |

---

## Configuration

All hyperparameters, paths, and model settings live in `configs/config.yaml`. Edit that file only, no source code changes needed.

Key settings:
- `data.superclasses`: the five PTB-XL diagnostic superclasses
- `inference.threshold`: classification threshold (default: 0.5)
- `training.batch_size`, `training.lr`, `training.epochs`
- `uncertainty.n_runs`, `uncertainty.noise_std`: TTA parameters

---

## Disclaimer

BioSignal-XAI is a **research prototype**. It is not a certified medical device under MDR, FDA 510(k), or any equivalent regulatory framework. Model outputs must not be used as the sole basis for clinical diagnosis or treatment decisions. All outputs are intended to assist, not replace, qualified medical professionals.

---

## Citation

```bibtex
@misc{biosignalxai2026,
  author       = {Taleska, Teodora},
  title        = {BioSignal-XAI: ECG Anomaly Detection with Explainable AI},
  year         = {2026},
  howpublished = {\url{https://github.com/teodora-taleska/biosignal-xai}},
  note         = {University research project}
}
```
