from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import CFG

NUM_CLASSES = len(CFG['data']['superclasses'])
_DROPOUT    = CFG['model']['leadwise']['dropout']


class AleatoricWrapper(nn.Module):
    """
    Wraps any ECG classifier to add aleatoric uncertainty.

    The base model's classifier is replaced with two heads:
      mean_head:    Linear -> logits (same as original)
      log_var_head: Linear -> log variance (uncertainty)

    During training: uses aleatoric_loss (NLL with uncertainty)
    During inference: returns (mean, log_var) from forward()

    Aleatoric uncertainty = noise inherent in the input signal.
    High uncertainty = noisy ECG, even perfect model struggles.
    Low uncertainty  = clean signal, prediction is trustworthy.
    """

    def __init__(self, base_model: nn.Module, hidden_dim: int):
        """
        Args:
            base_model:  trained HuBERTECGClassifier or LeadwiseTransformer
            hidden_dim:  feature size before classifier
                         768 for HuBERT, d_model (128) for LeadwiseTransformer
        """
        super().__init__()

        # Replace original classifier with Identity to expose features
        self.feature_extractor = base_model
        if hasattr(base_model, 'classifier'):
            self.feature_extractor.classifier = nn.Identity()

        # Two independent prediction heads
        self.mean_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, NUM_CLASSES),
        )

        self.log_var_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean:    (batch, 5) logits
            log_var: (batch, 5) log variance
        """
        features = self.feature_extractor(x)  # (batch, hidden_dim)
        mean     = self.mean_head(features)    # (batch, 5)
        log_var  = self.log_var_head(features) # (batch, 5)
        return mean, log_var

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {
            'trainable':  trainable,
            'total':      total,
            'percentage': f'{100 * trainable / total:.1f}%',
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, 'checkpoint.pt'))
        print(f'Saved -> {path}/checkpoint.pt')


def aleatoric_loss(
    mean:    torch.Tensor,
    log_var: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood loss with aleatoric uncertainty.

    Penalizes overconfident wrong predictions; rewards honest uncertainty
    on hard examples (noisy signals).

    Formula:
      L = mean( exp(-log_var) * BCE(mean, y) + 0.5 * log_var )

    - exp(-log_var): reduces the BCE penalty when the model is uncertain
    - 0.5 * log_var: regularisation that penalises claiming too much uncertainty

    Args:
        mean:    (batch, 5) raw logits from mean_head
        log_var: (batch, 5) log variance from log_var_head
        targets: (batch, 5) multi-hot ground truth

    Returns:
        scalar loss
    """
    # Per-element BCE (unreduced)
    bce = F.binary_cross_entropy_with_logits(
        mean, targets, reduction='none'
    )  # (batch, 5)

    # Uncertainty-weighted NLL
    loss = torch.exp(-log_var) * bce + 0.5 * log_var

    return loss.mean()
