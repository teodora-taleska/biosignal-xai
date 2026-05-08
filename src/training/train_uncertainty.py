from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader

from src.models.uncertainty_head import aleatoric_loss
from src.utils.config import CFG
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.profiler import ExperimentProfiler


def run_uncertainty_experiment(
    model,
    train_ds,
    val_ds,
    experiment_name: str,
    epochs:      int   = CFG['training']['epochs'],
    lr:          float = CFG['training']['lr_peft'],
    batch_size:  int   = CFG['training']['batch_size'],
    save_dir:    str   = CFG['paths']['results'],
    num_workers: int   = CFG['training']['num_workers'],
    patience:    int   = 4,
) -> tuple[float, list, dict]:
    """
    Train AleatoricWrapper end-to-end with aleatoric uncertainty loss.

    Identical to run_experiment() in train_peft.py except:
      - model returns (mean, log_var) not logits
      - loss is aleatoric_loss(), not BCEWithLogitsLoss
      - history tracks mean_uncertainty per epoch
      - metrics are computed on mean (logits) only

    Returns:
        best_auc:  float
        history:   list of per-epoch dicts
        profiling: timing/memory summary dict

    Saves to results/{experiment_name}/:
        best_adapter/checkpoint.pt
        history.json
        profiling.json
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f" Experiment : {experiment_name}")
    print(f" Device     : {device}")
    params = model.count_parameters()
    print(f" Trainable  : {params['trainable']:,}  ({params['percentage']})")
    print(f"{'='*60}")

    model = model.to(device)

    profiler = ExperimentProfiler(experiment_name)
    profiler.log_model(model)
    profiler.start()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = CFG['training']['weight_decay'],
        betas        = (0.9, 0.999),
    )

    def lr_lambda(epoch):
        warmup = 2
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = device.type == 'cuda',
    )

    exp_path = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_path, exist_ok=True)

    best_auc          = 0.0
    history           = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        profiler.start_epoch()

        # TRAIN
        model.train()
        train_loss  = 0.0
        n_batches   = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            mean, log_var = model(x)
            loss = aleatoric_loss(mean, log_var, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=CFG['training']['grad_clip']
            )
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # VALIDATE
        model.eval()
        val_loss        = 0.0
        val_uncertainty = 0.0
        all_logits      = []
        all_labels      = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mean, log_var = model(x)
                val_loss        += aleatoric_loss(mean, log_var, y).item()
                val_uncertainty += torch.exp(log_var).mean().item()
                all_logits.append(mean.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics    = compute_metrics(all_logits, all_labels)

        tl         = train_loss / n_batches
        vl         = val_loss   / len(val_loader)
        epoch_unc  = val_uncertainty / len(val_loader)
        auc        = metrics['auc_macro']
        f1         = metrics['f1_macro']
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch+1:02d}/{epochs}  "
              f"lr={current_lr:.2e}  uncertainty={epoch_unc:.4f}")
        print(f"  train_loss={tl:.4f}  val_loss={vl:.4f}")
        print_metrics(metrics)
        profiler.end_epoch()

        entry = {
            'epoch':            epoch + 1,
            'lr':               current_lr,
            'train_loss':       tl,
            'val_loss':         vl,
            'auc_macro':        auc,
            'f1_macro':         f1,
            'per_class':        metrics['per_class_auc'],
            'mean_uncertainty': epoch_unc,
            'epoch_time_sec':   profiler.epoch_times[-1],
        }
        history.append(entry)

        if auc > best_auc:
            best_auc          = auc
            epochs_no_improve = 0
            model.save(os.path.join(exp_path, 'best_adapter'))
            print(f"  * Best saved -- AUC {auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping (no AUC improvement for {patience} epochs)")
                break

    with open(os.path.join(exp_path, 'history.json'), 'w') as f:
        json.dump({
            'experiment': experiment_name,
            'best_auc':   best_auc,
            'history':    history,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f" Done: {experiment_name}")
    print(f" Best AUC: {best_auc:.4f}")
    print(f" Saved to: {exp_path}/")
    print(f"{'='*60}\n")

    profiler.end()
    profiler.log_checkpoint_size(os.path.join(exp_path, 'best_adapter'))
    profiler.save(exp_path)
    profiler.print_summary()

    return best_auc, history, profiler.summary()
