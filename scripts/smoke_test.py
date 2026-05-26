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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch

from src.utils.config import CFG
from src.preprocessing.label_utils import load_all_labels
from src.preprocessing.dataset_ablation import ECGDatasetAblation, ABLATION_CONFIGS
from src.models.fcn_wang import fcn_wang
from src.training.train_baseline import quick_ablation_run


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

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 50)
    print('SMOKE TEST SUMMARY')
    print('=' * 50)
    labels = {
        'fcn_wang': 'FCN-Wang baseline',
    }
    for key, label in labels.items():
        print(f'  {label:<25s}  AUC {results[key]:.4f}')
    print()
    print(f'All outputs → {RESULTS}/')

    with open(os.path.join(RESULTS, 'smoke_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
