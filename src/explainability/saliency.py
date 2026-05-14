from __future__ import annotations

import numpy as np
import torch


def compute_saliency(
    model,
    x_np: np.ndarray,
    target_class_idx: int,
    device: torch.device,
) -> np.ndarray:
    """
    Gradient-based input saliency for any ECG classifier.

    Works with both base models (return logits) and AleatoricWrapper
    (returns (mean, log_var) tuple) -- duck-typed automatically.

    Args:
        model:            nn.Module in eval mode
        x_np:             (12, 1000) float32 numpy array
        target_class_idx: index of the class to differentiate
        device:           torch.device

    Returns:
        (12, 1000) float32 numpy array of absolute gradient magnitudes.
        Falls back to a uniform array if backward pass fails.
    """
    model.eval()
    x_t = (
        torch.tensor(x_np, dtype=torch.float32)
        .unsqueeze(0)          # (1, 12, 1000)
        .to(device)
        .requires_grad_(True)
    )
    try:
        output = model(x_t)
        logits = output[0] if isinstance(output, tuple) else output
        logits[0, target_class_idx].backward()
        return x_t.grad[0].abs().cpu().numpy()   # (12, 1000)
    except Exception as e:
        print(f'  compute_saliency fallback (uniform): {e}')
        return np.ones((12, 1000), dtype=np.float32)


def top_salient_leads(
    saliency: np.ndarray,
    lead_names: list[str],
    top_k: int = 3,
) -> list[str]:
    """
    Return the names of the leads with the highest mean absolute saliency.

    Args:
        saliency:   (12, 1000) saliency array from compute_saliency()
        lead_names: list of 12 lead name strings
        top_k:      how many leads to return

    Returns:
        Ordered list of up to top_k lead names (most salient first).
    """
    scores = saliency.mean(axis=1)          # (12,) mean over time
    order  = np.argsort(scores)[::-1][:top_k]
    return [lead_names[i] for i in order]
