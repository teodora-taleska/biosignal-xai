import torch
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.config import CFG

SUPERCLASSES = CFG['preprocessing']['superclasses']

def compute_metrics(all_logits, all_labels, threshold=0.5):
    """
    all_logits: tensor (N, 5) — raw model output (before sigmoid)
    all_labels: tensor (N, 5) — multi-hot ground truth
    threshold:  above this → predicted positive

    Returns dict with AUC, F1, and per-class breakdown.
    AUC is the key metric used in the PTB-XL benchmark papers.
    """
    probs  = torch.sigmoid(all_logits).numpy()
    preds  = (probs >= threshold).astype(int)
    labels = all_labels.numpy().astype(int)

    # Macro AUC — average AUC across all 5 classes
    try:
        auc = roc_auc_score(labels, probs, average='macro')
    except ValueError:
        auc = 0.0   # happens if a class has no positive samples in batch

    # Macro F1
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    # Per-class AUC (very useful for understanding where model fails)
    per_class = {}
    for i, sc in enumerate(SUPERCLASSES):
        try:
            per_class[sc] = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            per_class[sc] = 0.0

    return {
        'auc_macro': auc,
        'f1_macro':  f1,
        'per_class_auc': per_class
    }


def print_metrics(metrics):
    print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
    print(f"  F1  (macro): {metrics['f1_macro']:.4f}")
    print(f"  Per-class AUC:")
    for cls, val in metrics['per_class_auc'].items():
        bar = '#' * int(val * 20)
        print(f"    {cls:5s}: {val:.3f}  {bar}")