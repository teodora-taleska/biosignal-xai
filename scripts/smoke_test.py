"""
Smoke test all models: 2 epochs, small data subset.

Produces real history.json files under results/<experiment_name>/
so notebook cells that load history work immediately after.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --epochs 3 --train 200 --val 50 --batch 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import json

import torch
from torch.utils.data import DataLoader

from src.utils.config import CFG
from src.preprocessing.label_utils import load_all_labels
from src.preprocessing.dataset import ECGDataset
from src.preprocessing.dataset_full import ECGDatasetFull
from src.models.baseline_cnn import BaselineCNN
from src.models.dummy_classifier import DummyECGClassifier
from src.models.hubert_ecg_finetune import HuBERTECGClassifier, HuBERTECGPEFT
from src.models.leadwise_transformer import build_leadwise_with_peft
from src.training.train import train_model
from src.training.train_peft import run_experiment

from src.utils.metrics import compute_metrics


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
    print(f'Device: {device}')
    print(f'Epochs: {args.epochs}  Train: {args.train}  Val: {args.val}  Batch: {args.batch}\n')

    DATA_PATH = CFG['data']['path']
    RESULTS   = os.path.join(CFG['paths']['results'], 'smoke')
    os.makedirs(RESULTS, exist_ok=True)

    Y = load_all_labels(
        DATA_PATH + 'ptbxl_database.csv',
        DATA_PATH + 'scp_statements.csv',
    )
    train_df = Y[Y.strat_fold < 9].head(args.train)
    val_df   = Y[Y.strat_fold == 9].head(args.val)

    train_ds_w = ECGDataset(train_df, DATA_PATH)
    val_ds_w   = ECGDataset(val_df,   DATA_PATH)
    train_ds_f = ECGDatasetFull(train_df, DATA_PATH)
    val_ds_f   = ECGDatasetFull(val_df,   DATA_PATH)

    results = {}

    #  CNN baseline
    print('=' * 50)
    print('Baseline CNN')
    cnn = BaselineCNN()
    cnn = train_model(cnn, train_ds_w, val_ds_w, epochs=args.epochs, batch_size=args.batch,
                      save_dir=os.path.join(RESULTS, 'baseline_cnn'))

    cnn.eval()
    val_loader = DataLoader(val_ds_w, batch_size=args.batch)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_logits.append(cnn(x.to(device)).cpu())
            all_labels.append(y)
    cnn_metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    with open(os.path.join(RESULTS, 'baseline_cnn_metrics.json'), 'w') as f:
        json.dump(cnn_metrics, f, indent=2)

    results['cnn'] = cnn_metrics['auc_macro']
    del cnn; torch.cuda.empty_cache()
    print(f"AUC: {cnn_metrics['auc_macro']:.4f}")

    #  Dummy
    print('=' * 50)
    print('Dummy classifier')
    dummy = DummyECGClassifier()
    dummy._results_dir = RESULTS   # redirect to smoke/
    dummy.fit(train_ds_f)
    metrics = dummy.evaluate(val_ds_f)
    dummy.save_results(metrics, 'dummy_metrics.json')
    results['dummy'] = metrics['auc_macro']
    print(f"AUC: {metrics['auc_macro']:.4f}")

    #  HuBERT 8 blocks
    print()
    model_B = HuBERTECGClassifier(
        size=CFG['model']['hubert_size'], blocks_to_unfreeze=8
    )
    auc_B, _ = run_experiment(
        model_B, train_ds_f, val_ds_f,
        experiment_name='hubert_ecg_blocks8',
        epochs=args.epochs,
        lr=CFG['training']['lr_pretrained'],
        batch_size=args.batch,
        save_dir=RESULTS,
    )
    del model_B; torch.cuda.empty_cache()
    results['hubert_8'] = auc_B

    #  HuBERT PEFT — LoRA r=8
    print()
    _probe = HuBERTECGPEFT(rank=8, use_dora=False).to(device)
    _dummy_in = torch.randn(2, 12, 1000)
    with torch.no_grad():
        _out = _probe(_dummy_in.to(device))
    assert _out.shape == (2, 5), f"Shape error: {_out.shape}"
    print(f"Forward pass OK: (2, 12, 1000) -> {_out.shape}")
    _p = _probe.count_parameters()
    assert _p['trainable'] / _p['total'] < 0.05, \
        f"LoRA trainable {_p['trainable']/_p['total']:.1%} exceeds 5%"
    print(f"Parameter check OK: {_p['trainable']:,} / {_p['total']:,} = "
          f"{100*_p['trainable']/_p['total']:.1f}%")
    del _probe, _dummy_in, _out

    model_lora = HuBERTECGPEFT(rank=8, use_dora=False)
    auc_lora, _ = run_experiment(
        model_lora, train_ds_f, val_ds_f,
        experiment_name='hubert_ecg_lora_r8',
        epochs=args.epochs,
        lr=CFG['training']['lr_pretrained'],
        batch_size=args.batch,
        save_dir=RESULTS,
    )
    del model_lora; torch.cuda.empty_cache()
    results['lora'] = auc_lora

    #  HuBERT PEFT — DoRA r=8
    print()
    model_dora = HuBERTECGPEFT(rank=8, use_dora=True)
    auc_dora, _ = run_experiment(
        model_dora, train_ds_f, val_ds_f,
        experiment_name='hubert_ecg_dora_r8',
        epochs=args.epochs,
        lr=CFG['training']['lr_pretrained'],
        batch_size=args.batch,
        save_dir=RESULTS,
    )
    del model_dora; torch.cuda.empty_cache()
    results['dora'] = auc_dora

    #  Lead-wise PEFT — LoRA r=8
    print()
    # Shape + param assertions before training
    _probe_lw = build_leadwise_with_peft(rank=8, use_dora=False).to(device)
    _lw_dummy = torch.randn(2, 12, 1000)
    with torch.no_grad():
        _lw_out = _probe_lw(_lw_dummy.to(device))
    assert _lw_out.shape == (2, 5), f"Shape error: {_lw_out.shape}"
    _lw_p = _probe_lw.count_parameters()
    assert _lw_p['trainable'] / _lw_p['total'] < 0.10, \
        f"Leadwise LoRA trainable {_lw_p['trainable']/_lw_p['total']:.1%} exceeds 10%"
    print(f"Leadwise LoRA: (2,12,1000) -> {_lw_out.shape} OK | "
          f"{_lw_p['trainable']:,} / {_lw_p['total']:,} = "
          f"{100*_lw_p['trainable']/_lw_p['total']:.1f}%")
    del _probe_lw, _lw_dummy, _lw_out

    model_lw_lora = build_leadwise_with_peft(rank=8, use_dora=False)
    auc_lw_lora, _ = run_experiment(
        model_lw_lora, train_ds_f, val_ds_f,
        experiment_name='leadwise_lora_r8',
        epochs=args.epochs,
        lr=CFG['training']['lr_peft'],
        batch_size=args.batch,
        save_dir=RESULTS,
    )
    del model_lw_lora; torch.cuda.empty_cache()
    results['lw_lora'] = auc_lw_lora

    #  Summary
    print()
    print('=' * 50)
    print('SMOKE TEST SUMMARY')
    print('=' * 50)
    labels = {
        'dummy':    'Dummy (prior)',
        'cnn':      'CNN (baseline)',
        'hubert_8': 'HuBERT-ECG 8 blocks',
        'lora':     'HuBERT-ECG LoRA r=8',
        'dora':     'HuBERT-ECG DoRA r=8',
        'lw_lora':  'Lead-wise LoRA r=8',
    }
    for key, label in labels.items():
        print(f'  {label:<25s}  AUC {results[key]:.4f}')
    print()
    print(f'All outputs written to {RESULTS}/')


if __name__ == '__main__':
    main()
