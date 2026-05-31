"""
run_peft_experiment: generic PEFT training loop for tensor-input models.

Used by notebook 04 to train HuBERTECGClassifier (takes (B, 12, 1000) tensors).
HeartBERT and ECG-PT embed their own loops because they require HuggingFace
tokeniser steps that cannot run inside a standard DataLoader worker.
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_auc, compute_probs
from src.preprocessing.label_utils import SUPERCLASSES
from src.utils.profiler import ExperimentProfiler


def run_peft_experiment(
    model: nn.Module,
    train_ds,
    val_ds,
    experiment_name: str,
    epochs: int      = 25,
    lr: float        = 1e-4,
    batch_size: int  = 32,
    patience: int    = 2,
    save_dir: str    = "results/",
    num_workers: int = 0,
):
    """
    Training loop for PEFT models that accept (B, 12, 1000) float32 tensors.

    Parameters
    ----------
    model           : HuBERTECGClassifier (nn.Module) with count_parameters() and save()
    train_ds / val_ds : ECGDatasetFull instances
    experiment_name : used for results directory name and logging
    epochs          : maximum training epochs
    lr              : AdamW learning rate
    batch_size      : samples per gradient step (keep ≤ 32 for 8 GB VRAM)
    patience        : early stopping — halt if val AUC hasn't improved for N epochs
    save_dir        : parent directory for results
    num_workers     : DataLoader workers (0 on Windows)

    Returns
    -------
    best_auc  : float
    history   : list of per-epoch dicts
    profiling : dict from ExperimentProfiler.summary()
    """
    device    = next(model.parameters()).device
    save_path = os.path.join(save_dir, experiment_name)
    os.makedirs(save_path, exist_ok=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    p        = model.count_parameters()
    profiler = ExperimentProfiler(experiment_name)
    profiler.trainable_params = p["trainable"]
    profiler.total_params     = p["total"]

    print(f"\n{'='*60}")
    print(f" Experiment : {experiment_name}")
    print(f" Device     : {device}")
    print(f" Trainable  : {p['trainable']:,}  ({p['percentage']})")
    print(f"{'='*60}\n")

    profiler.start()
    best_auc, history = 0.0, []
    epochs_no_improve = 0

    for epoch in range(epochs):
        profiler.start_epoch()
        model.train()
        total_loss, n_batches = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        model.eval()
        val_logits_list, val_labels_list = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y    = x.to(device), y.to(device)
                logits  = model(x)
                val_loss += criterion(logits, y).item()
                val_logits_list.append(logits.cpu())
                val_labels_list.append(y.cpu())

        probs     = compute_probs(torch.cat(val_logits_list))
        auc_res   = compute_auc(probs, torch.cat(val_labels_list).numpy())
        auc_macro = auc_res["auc_macro"]
        per_cls   = auc_res["per_class_auc"]

        avg_train = total_loss / n_batches
        avg_val   = val_loss   / len(val_loader)

        print(f"Epoch {epoch+1:02d}/{epochs}  lr={lr_now:.2e}")
        print(f"  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")
        print(f"  AUC (macro): {auc_macro:.4f}")
        print(f"  Per-class AUC:")
        for sc, auc in zip(SUPERCLASSES, per_cls.values()):
            bar = "#" * int(auc * 20)
            print(f"    {sc:4s} : {auc:.3f}  {bar}")

        history.append({
            "epoch":          epoch + 1,
            "lr":             lr_now,
            "train_loss":     avg_train,
            "val_loss":       avg_val,
            "auc_macro":      auc_macro,
            "per_class_auc":  per_cls,
        })

        adapter_path = os.path.join(save_path, "best_adapter")
        if auc_macro > best_auc:
            best_auc = auc_macro
            epochs_no_improve = 0
            model.save(adapter_path)
            print(f"Saved → {adapter_path}")
            print(f"  * Best saved — AUC {best_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping (no AUC improvement for {patience} epochs)")
                profiler.end_epoch()
                break

        profiler.end_epoch()

    profiler.end()
    profiler.log_checkpoint_size(os.path.join(save_path, "best_adapter"))
    profiler.print_summary()
    profiler.save(save_path)

    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump({"experiment": experiment_name, "history": history}, f, indent=2)

    print(f"\n{'='*60}")
    print(f" Done: {experiment_name}")
    print(f" Best AUC: {best_auc:.4f}")
    print(f" Saved to: {save_path}/")
    print(f"{'='*60}\n")

    return best_auc, history, profiler.summary()
