"""
Test-Time Augmentation (TTA) uncertainty estimation for deterministic ECG models.

Approach
--------
Run N forward passes on slightly noise-perturbed copies of the input signal.
For each class we get a distribution of probabilities across runs.  The std of
that distribution is the ± uncertainty for that class.

  prediction:  NORM 0.75 +/- 0.03   MI 0.12 +/- 0.01   ...

High per-class std means the sigmoid output shifts noticeably when the signal is
slightly perturbed -- the model is near the decision boundary for that class.

This is model-agnostic: no retraining, no architecture changes, works with any
deterministic feedforward network (FCN-Wang, HeartBERT, etc.).

Public API
----------
compute_tta_uncertainty(model, x_np, device, n_runs, noise_std)
    -> (mean_probs, std_probs, uncertainty_scalar, level_string)
tta_uncertainty_level(uncertainty) -> str
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ── Thresholds (calibrated on FCN-Wang / PTB-XL, noise_std=0.05, n_runs=20) ──
# PTB-XL signals span roughly -2..2 mV; noise_std=0.05 is ~2.5% of full range.
# Stable predictions (far from boundary) produce mean std < 0.01.
# Predictions near the 0.5 threshold produce mean std up to ~0.10.
_LOW_THRESH  = 0.015   # below: trustworthy
_HIGH_THRESH = 0.040   # above: treat with caution


def tta_uncertainty_level(uncertainty: float) -> str:
    """
    Convert a TTA uncertainty scalar (mean per-class std) to a quality label.

    Args:
        uncertainty: mean across per-class probability stds

    Returns:
        Descriptive string suitable for display and LLM prompts.
    """
    if uncertainty < _LOW_THRESH:
        return 'low - prediction stable under signal perturbations'
    elif uncertainty < _HIGH_THRESH:
        return 'moderate - some sensitivity to signal noise detected'
    else:
        return 'high - prediction changes with small noise, treat with caution'


def compute_tta_uncertainty(
    model:      nn.Module,
    x_np:       np.ndarray,
    device:     torch.device,
    n_runs:     int   = 20,
    noise_std:  float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """
    Estimate prediction uncertainty via Test-Time Augmentation.

    Runs *n_runs* forward passes: one clean pass followed by (n_runs-1) passes
    with independent Gaussian noise added to the input.

    Returns per-class mean and std probabilities so callers can express results as:
        class_probability +/- class_std

    The scalar uncertainty (mean of per-class stds) is also returned for summary
    displays and the human-readable level label.

    Args:
        model:      trained PyTorch model (eval mode, any deterministic arch)
        x_np:       preprocessed signal (12, 1000) float32
        device:     torch device the model lives on
        n_runs:     total forward passes (including the clean one); default 20
        noise_std:  std of additive Gaussian noise in signal units (mV);
                    default 0.05 (~2.5 % of the PTB-XL amplitude range)

    Returns:
        (mean_probs, std_probs, uncertainty, level) where:
            mean_probs  -- (n_classes,) float32 ndarray, mean probability per class
            std_probs   -- (n_classes,) float32 ndarray, std (the +/- per class)
            uncertainty -- float, mean of std_probs (summary scalar)
            level       -- human-readable quality label (str)
    """
    model.eval()

    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)

    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        # First pass: clean signal
        logits = model(x)
        probs_list.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())

        # Subsequent passes: perturbed signal
        for _ in range(n_runs - 1):
            noise  = torch.randn_like(x) * noise_std
            logits = model(x + noise)
            probs_list.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())

    probs_array = np.stack(probs_list)          # (n_runs, n_classes)
    mean_probs  = probs_array.mean(axis=0)      # (n_classes,) -- more robust than single pass
    std_probs   = probs_array.std(axis=0)       # (n_classes,) -- the +/- per class
    uncertainty = float(std_probs.mean())       # scalar summary

    return (
        mean_probs.astype(np.float32),
        std_probs.astype(np.float32),
        round(uncertainty, 4),
        tta_uncertainty_level(uncertainty),
    )
