"""
Training loop for XResNet1D-101 on PTB-XL.

Matches the training protocol from Strodthoff et al. 2020:
  - AdamW optimiser
  - 1-cycle LR schedule (linear warm-up → cosine decay)
  - BCEWithLogitsLoss with pos_weight for class imbalance
  - Early stopping on validation macro-AUC (not loss)
  - Gradient clipping for stability
  - Full training history + profiling saved to JSON

Usage
-----
    from src.models.xresnet1d import xresnet1d101
    from src.training.train_xresnet import train_xresnet

    model = xresnet1d101()
    best_auc, history = train_xresnet(model, train_ds, val_ds, 'xresnet_baseline')
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_auc, compute_fmax, compute_probs
from src.utils.config import CFG
from src.utils.profiler import ExperimentProfiler


# Default pos_weight: inverse class frequencies in PTB-XL training set
# NORM: 44.5 %  MI: 25.6 %  STTC: 24.5 %  CD: 22.9 %  HYP: 12.4 %
_DEFAULT_POS_WEIGHT = torch.tensor([1.0, 1.74, 1.82, 1.94, 3.59])


def train_xresnet(
    model,
    train_ds,
    val_ds,
    experiment_name:   str   = 'xresnet_baseline',
    epochs:            int   = CFG['training']['epochs'],
    lr_peak:           float = 1e-3,
    batch_size:        int   = 64,
    weight_decay:      float = CFG['training']['weight_decay'],
    warmup_pct:        float = 0.3,
    patience:          int   = 8,
    grad_clip:         float = CFG['training']['grad_clip'],
    pos_weight:        torch.Tensor = None,
    num_workers:       int   = CFG['training']['num_workers'],
    save_dir:          str   = None,
) -> tuple[float, list, dict]:
    """
    Train XResNet1D-101 with 1-cycle LR and AUC-based early stopping.

    Parameters
    ----------
    model           : XResNet1d instance (untrained or pretrained)
    train_ds        : training Dataset (ECGDatasetAblation or ECGDatasetFull)
    val_ds          : validation Dataset
    experiment_name : used for save directory and profiling JSON key
    epochs          : maximum training epochs
    lr_peak         : peak learning rate for 1-cycle schedule
    batch_size      : samples per batch (64 recommended for 8 GB VRAM)
    weight_decay    : AdamW weight decay
    warmup_pct      : fraction of total steps used for linear warm-up
    patience        : early stopping — stop if AUC hasn't improved for N epochs
    grad_clip       : max gradient norm
    pos_weight      : BCEWithLogitsLoss positive class weights (length 5)
    num_workers     : DataLoader workers (0 on Windows)
    save_dir        : root directory for checkpoints; default CFG['paths']['results']

    Returns
    -------
    (best_auc, history, profiling_summary)
        best_auc  : float — best validation macro-AUC achieved
        history   : list of per-epoch dicts
        profiling : dict from ExperimentProfiler.summary()
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*64}")
    print(f" Experiment : {experiment_name}")
    print(f" Device     : {device}")
    p = model.count_parameters()
    print(f" Params     : {p['total']:,} total | {p['trainable']:,} trainable ({p['percentage']})")
    print(f" Epochs     : {epochs}  |  LR peak: {lr_peak:.0e}  |  BS: {batch_size}")
    print(f"{'='*64}")

    if save_dir is None:
        save_dir = CFG['paths']['results']
    exp_path = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_path, exist_ok=True)
    ckpt_path = os.path.join(exp_path, 'checkpoint.pt')

    model = model.to(device)

    # ---- Profiler ----
    profiler = ExperimentProfiler(experiment_name)
    profiler.log_model(model)
    profiler.start()

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_peak,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # ---- Loss ----
    if pos_weight is None:
        pos_weight = _DEFAULT_POS_WEIGHT
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # ---- 1-cycle LR schedule ----
    # PyTorch OneCycleLR: linear warm-up to lr_peak, then cosine annealing
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
    )

    total_steps = epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr         = lr_peak,
        total_steps    = total_steps,
        pct_start      = warmup_pct,
        anneal_strategy= 'cos',
        div_factor     = 25.0,       # initial_lr = lr_peak / 25
        final_div_factor=1e4,        # final_lr   = initial_lr / 1e4
    )

    # ---- Training loop ----
    best_auc          = 0.0
    history           = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        profiler.start_epoch()

        # -- Train --
        model.train()
        train_loss  = 0.0
        n_batches   = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n_batches  += 1

        # -- Validate --
        model.eval()
        val_loss   = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        probs      = compute_probs(all_logits)
        labels_np  = all_labels.numpy().astype(int)
        auc_res    = compute_auc(probs, labels_np)
        fmax_res   = compute_fmax(probs, labels_np)

        tl   = train_loss / n_batches
        vl   = val_loss   / len(val_loader)
        auc  = auc_res['auc_macro']
        fmax = fmax_res['fmax']
        current_lr = scheduler.get_last_lr()[0]

        profiler.end_epoch()

        print(f"Epoch {epoch+1:03d}/{epochs}  lr={current_lr:.2e}  "
              f"train={tl:.4f}  val={vl:.4f}  "
              f"AUC={auc:.4f}  Fmax={fmax:.4f}")

        epoch_rec = {
            'epoch':          epoch + 1,
            'lr':             current_lr,
            'train_loss':     round(tl, 5),
            'val_loss':       round(vl, 5),
            'auc_macro':      round(auc, 5),
            'fmax':           round(fmax, 5),
            'per_class_auc':  {k: round(v, 5) for k, v in auc_res['per_class_auc'].items()},
            'epoch_time_sec': profiler.epoch_times[-1],
        }
        history.append(epoch_rec)

        # -- Checkpoint & early stopping on AUC --
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Best AUC {auc:.4f} — saved → {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping (no AUC improvement for {patience} epochs)")
                break

    # ---- Save outputs ----
    profiler.end()
    profiler.log_checkpoint_size(ckpt_path)
    profiler.save(exp_path)
    profiler.print_summary()

    with open(os.path.join(exp_path, 'history.json'), 'w') as f:
        json.dump({
            'experiment': experiment_name,
            'best_auc':   best_auc,
            'history':    history,
        }, f, indent=2)

    print(f"\n{'='*64}")
    print(f" Done : {experiment_name}")
    print(f" Best AUC : {best_auc:.4f}")
    print(f" Saved    : {exp_path}/")
    print(f"{'='*64}\n")

    return best_auc, history, profiler.summary()



# Quick-run variant — for ablation (fewer epochs, subset of data)


def quick_ablation_run(
    model,
    train_ds,
    val_ds,
    experiment_name: str,
    epochs:          int = 10,
    lr_peak:         float = 5e-4,
    batch_size:      int = 64,
    save_dir:        str = None,
) -> dict:
    """
    Lightweight training run for ablation comparisons.

    Uses fewer epochs (10 default) and a lower peak LR to get a meaningful
    relative comparison without waiting for full convergence. The AUC ordering
    across preprocessing configs is stable after 10 epochs even if absolute
    values are below the final converged level.

    Parameters
    ----------
    save_dir : root directory for this run's checkpoint; defaults to
               CFG['paths']['results']. Pass e.g. results/ablation/ so all
               ablation conditions land in one dedicated subfolder.

    Returns a flat summary dict ready for ablation_results.json.
    """
    best_auc, history, profiling = train_xresnet(
        model           = model,
        train_ds        = train_ds,
        val_ds          = val_ds,
        experiment_name = experiment_name,
        epochs          = epochs,
        lr_peak         = lr_peak,
        batch_size      = batch_size,
        patience        = epochs,   # disable early stopping for fair comparison
        save_dir        = save_dir,
    )

    last = history[-1]
    return {
        'experiment':     experiment_name,
        'best_auc':       best_auc,
        'final_auc':      last['auc_macro'],
        'final_fmax':     last['fmax'],
        'per_class_auc':  last['per_class_auc'],
        'epochs_run':     len(history),
        'total_time_sec': profiling['total_time_sec'],
    }
