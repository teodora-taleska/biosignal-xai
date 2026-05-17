"""
Smoke test: 2 epochs, small data subset, all models.

Verifies the full stack (data loading → model forward → training loop →
metrics) runs without errors. Not a quality benchmark — just a sanity check.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --epochs 3 --train 200 --val 50 --batch 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
from torch.utils.data import DataLoader

from src.utils.config import CFG
from src.preprocessing.label_utils import load_all_labels
from src.preprocessing.dataset_ablation import ECGDatasetAblation, ABLATION_CONFIGS
from src.preprocessing.dataset_full import ECGDatasetFull
from src.models.fcn_wang import fcn_wang
from src.models.dummy_classifier import DummyECGClassifier
from src.models.hubert_ecg_finetune import HuBERTECGClassifier, HuBERTECGPEFT
from src.models.leadwise_transformer import build_leadwise_with_peft
from src.training.train_baseline import quick_ablation_run
from src.training.train_peft import run_experiment
from src.evaluation.metrics import compute_auc, compute_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Smoke test all models')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--train',  type=int, default=100, help='training samples')
    p.add_argument('--val',    type=int, default=20,  help='validation samples')
    p.add_argument('--batch',  type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    print(f'Epochs : {args.epochs}  Train : {args.train}  Val : {args.val}  Batch : {args.batch}\n')

    DATA_PATH = CFG['data']['path']
    RESULTS   = os.path.join(CFG['paths']['results'], 'smoke')
    os.makedirs(RESULTS, exist_ok=True)

    Y = load_all_labels(
        DATA_PATH + 'ptbxl_database.csv',
        DATA_PATH + 'scp_statements.csv',
    )
    train_df = Y[Y.strat_fold < 9].head(args.train)
    val_df   = Y[Y.strat_fold == 9].head(args.val)

    # Window dataset (FCN-Wang, windowed inputs)
    preprocess_cfg = ABLATION_CONFIGS['bandpass_none_250']
    train_ds_w = ECGDatasetAblation(train_df, DATA_PATH, **preprocess_cfg)
    val_ds_w   = ECGDatasetAblation(val_df,   DATA_PATH, **preprocess_cfg)

    # Full-record dataset (HuBERT, Leadwise — need 1000-sample inputs)
    train_ds_f = ECGDatasetFull(train_df, DATA_PATH)
    val_ds_f   = ECGDatasetFull(val_df,   DATA_PATH)

    results = {}

    # ── FCN-Wang baseline ─────────────────────────────────────────────────────
    print('=' * 50)
    print('FCN-Wang baseline')
    fcn = fcn_wang()
    summary = quick_ablation_run(
        model           = fcn,
        train_ds        = train_ds_w,
        val_ds          = val_ds_w,
        experiment_name = 'smoke_fcn_wang',
        epochs          = args.epochs,
        batch_size      = args.batch,
        save_dir        = RESULTS,
    )
    results['fcn_wang'] = summary['best_auc']
    del fcn; torch.cuda.empty_cache()
    print(f"AUC: {summary['best_auc']:.4f}")

    # ── Dummy classifier ──────────────────────────────────────────────────────
    print('=' * 50)
    print('Dummy classifier')
    dummy = DummyECGClassifier()
    dummy._results_dir = RESULTS
    dummy.fit(train_ds_f)
    metrics = dummy.evaluate(val_ds_f)
    dummy.save_results(metrics, 'dummy_metrics.json')
    results['dummy'] = metrics['auc_macro']
    print(f"AUC: {metrics['auc_macro']:.4f}")

    # ── HuBERT-ECG 8 blocks ───────────────────────────────────────────────────
    print('=' * 50)
    print('HuBERT-ECG 8 blocks')
    model_B = HuBERTECGClassifier(
        size=CFG['model']['hubert_size'], blocks_to_unfreeze=8
    )
    auc_B, _, _ = run_experiment(
        model_B, train_ds_f, val_ds_f,
        experiment_name = 'smoke_hubert_8',
        epochs          = args.epochs,
        lr              = CFG['training']['lr_pretrained'],
        batch_size      = args.batch,
        save_dir        = RESULTS,
    )
    results['hubert_8'] = auc_B
    del model_B; torch.cuda.empty_cache()

    # ── HuBERT-ECG PEFT LoRA r=8 ─────────────────────────────────────────────
    print('=' * 50)
    print('HuBERT-ECG LoRA r=8')
    _probe = HuBERTECGPEFT(rank=8, use_dora=False).to(device)
    with torch.no_grad():
        _out = _probe(torch.randn(2, 12, 1000).to(device))
    assert _out.shape == (2, 5), f"Shape error: {_out.shape}"
    _p = _probe.count_parameters()
    assert _p['trainable'] / _p['total'] < 0.05, \
        f"LoRA trainable {_p['trainable']/_p['total']:.1%} exceeds 5%"
    print(f"Forward pass OK: (2,12,1000) → {_out.shape}  "
          f"LoRA params: {_p['trainable']:,}/{_p['total']:,}")
    del _probe, _out; torch.cuda.empty_cache()

    model_lora = HuBERTECGPEFT(rank=8, use_dora=False)
    auc_lora, _, _ = run_experiment(
        model_lora, train_ds_f, val_ds_f,
        experiment_name = 'smoke_hubert_lora_r8',
        epochs          = args.epochs,
        lr              = CFG['training']['lr_pretrained'],
        batch_size      = args.batch,
        save_dir        = RESULTS,
    )
    results['lora'] = auc_lora
    del model_lora; torch.cuda.empty_cache()

    # ── HuBERT-ECG PEFT DoRA r=8 ─────────────────────────────────────────────
    print('=' * 50)
    print('HuBERT-ECG DoRA r=8')
    model_dora = HuBERTECGPEFT(rank=8, use_dora=True)
    auc_dora, _, _ = run_experiment(
        model_dora, train_ds_f, val_ds_f,
        experiment_name = 'smoke_hubert_dora_r8',
        epochs          = args.epochs,
        lr              = CFG['training']['lr_pretrained'],
        batch_size      = args.batch,
        save_dir        = RESULTS,
    )
    results['dora'] = auc_dora
    del model_dora; torch.cuda.empty_cache()

    # ── Lead-wise Transformer LoRA r=8 ────────────────────────────────────────
    print('=' * 50)
    print('Lead-wise Transformer LoRA r=8')
    _probe_lw = build_leadwise_with_peft(rank=8, use_dora=False).to(device)
    with torch.no_grad():
        _lw_out = _probe_lw(torch.randn(2, 12, 1000).to(device))
    assert _lw_out.shape == (2, 5), f"Shape error: {_lw_out.shape}"
    _lw_p = _probe_lw.count_parameters()
    assert _lw_p['trainable'] / _lw_p['total'] < 0.10, \
        f"Leadwise LoRA trainable {_lw_p['trainable']/_lw_p['total']:.1%} exceeds 10%"
    print(f"Forward pass OK: (2,12,1000) → {_lw_out.shape}  "
          f"LoRA params: {_lw_p['trainable']:,}/{_lw_p['total']:,}")
    del _probe_lw, _lw_out; torch.cuda.empty_cache()

    model_lw = build_leadwise_with_peft(rank=8, use_dora=False)
    auc_lw, _, _ = run_experiment(
        model_lw, train_ds_f, val_ds_f,
        experiment_name = 'smoke_leadwise_lora_r8',
        epochs          = args.epochs,
        lr              = CFG['training']['lr_peft'],
        batch_size      = args.batch,
        save_dir        = RESULTS,
    )
    results['lw_lora'] = auc_lw
    del model_lw; torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 50)
    print('SMOKE TEST SUMMARY')
    print('=' * 50)
    labels = {
        'dummy':    'Dummy (prior)',
        'fcn_wang': 'FCN-Wang baseline',
        'hubert_8': 'HuBERT-ECG 8 blocks',
        'lora':     'HuBERT-ECG LoRA r=8',
        'dora':     'HuBERT-ECG DoRA r=8',
        'lw_lora':  'Lead-wise LoRA r=8',
    }
    for key, label in labels.items():
        print(f'  {label:<25s}  AUC {results[key]:.4f}')
    print()
    print(f'All outputs → {RESULTS}/')

    with open(os.path.join(RESULTS, 'smoke_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
