"""
Quick verification that profiling integration works end-to-end.
Runs 2 epochs on a tiny HuBERT LoRA subset and asserts all contracts.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
from src.utils.config import CFG
from src.data.label_utils import load_all_labels
from src.data.dataset_full import ECGDatasetFull
from src.models.hubert_ecg_finetune import HuBERTECGPEFT
from src.training.train_peft import run_experiment
from src.utils.profiler import ExperimentProfiler

DATA_PATH = CFG['data']['path']
SMOKE     = os.path.join(CFG['paths']['results'], 'smoke')

Y        = load_all_labels(DATA_PATH + 'ptbxl_database.csv', DATA_PATH + 'scp_statements.csv')
train_df = Y[Y.strat_fold < 9].head(50)
val_df   = Y[Y.strat_fold == 9].head(10)
train_ds = ECGDatasetFull(train_df, DATA_PATH)
val_ds   = ECGDatasetFull(val_df,   DATA_PATH)

model = HuBERTECGPEFT(rank=8, use_dora=False)
auc, hist, prof = run_experiment(
    model, train_ds, val_ds,
    experiment_name='hubert_ecg_lora_r8',
    epochs=2,
    lr=5e-5,
    batch_size=8,
    save_dir=SMOKE,
)

# Contract 1: run_experiment returns 3 values
assert isinstance(auc,  float), f"auc must be float, got {type(auc)}"
assert isinstance(hist, list),  f"hist must be list, got {type(hist)}"
assert isinstance(prof, dict),  f"prof must be dict, got {type(prof)}"

# Contract 2: profiling dict has all required keys
required = [
    'experiment', 'total_time_sec', 'total_time_human',
    'avg_epoch_time_sec', 'epoch_times_sec',
    'peak_gpu_memory_gb', 'trainable_params',
    'total_params', 'trainable_pct', 'checkpoint_size_mb',
]
for key in required:
    assert key in prof, f"Missing key in profiling dict: {key}"

# Contract 3: epoch_times_sec has exactly 2 entries
assert len(prof['epoch_times_sec']) == 2, \
    f"Expected 2 epoch times, got {len(prof['epoch_times_sec'])}"

# Contract 4: history entries include epoch_time_sec
for entry in hist:
    assert 'epoch_time_sec' in entry, "Missing epoch_time_sec in history entry"

# Contract 5: profiling.json was written to disk
prof_path = os.path.join(SMOKE, 'hubert_ecg_lora_r8', 'profiling.json')
assert os.path.exists(prof_path), f"profiling.json not found at {prof_path}"

# Contract 6: ExperimentProfiler.load() returns the same dict
loaded = ExperimentProfiler.load(os.path.join(SMOKE, 'hubert_ecg_lora_r8'))
assert loaded is not None, "ExperimentProfiler.load() returned None"
assert loaded['experiment'] == 'hubert_ecg_lora_r8'
assert loaded['epoch_times_sec'] == prof['epoch_times_sec']

print("\nProfiling integration OK")
print(f"  total_time:       {prof['total_time_human']}")
print(f"  epoch_times_sec:  {prof['epoch_times_sec']}")
print(f"  peak_gpu_memory:  {prof['peak_gpu_memory_gb']} GB")
print(f"  trainable_params: {prof['trainable_params']:,}")
print(f"  checkpoint_size:  {prof['checkpoint_size_mb']} MB")
