"""
FCN-Wang Baseline for 12-lead ECG classification.

Implements the Fully Convolutional Network from:
  Wang, Z., Yan, W., Oates, T. (2017). "Time series classification from
  scratch with deep neural networks: A strong baseline." IJCNN 2017.

With the concat-pooling head from:
  Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. (2021). "Deep
  Learning for ECG Analysis: Benchmarks and Insights from PTB-XL."
  IEEE JBHI. (Appendix I)

The original Wang et al. FCN uses a single GlobalAvgPool + sigmoid head.
Strodthoff et al. replace it with a concat-pool + small MLP head, which
is what produces their reported AUROC = 0.925 on PTB-XL super-diag.

Architecture
------------
Three convolutional blocks (no downsampling between blocks):
  Block 1: Conv1d(12 → 128, k=8)  → BN → ReLU
  Block 2: Conv1d(128 → 256, k=5) → BN → ReLU
  Block 3: Conv1d(256 → 128, k=3) → BN → ReLU

Head (concat-pooling):
  GlobalAvgPool + GlobalMaxPool → concat → (B, 256)
  BN → Dropout(0.25) → Linear(256→128) → ReLU
  BN → Dropout(0.50) → Linear(128→5)

Input  : (B, 12, T)   — any T (adaptive pooling handles it)
Output : (B, 5)       — raw logits; use BCEWithLogitsLoss

Parameter count : ~270k–300k (vs ~12M for XResNet1D-101)

Usage
-----
    from src.models.fcn_wang import FCNWang, fcn_wang
    model = fcn_wang()                    # default: 12 leads, 5 classes
    model = FCNWang(in_channels=1, num_classes=5)  # customised
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn

from src.utils.config import CFG

NUM_CLASSES = len(CFG['data']['superclasses'])


# ── Building block ────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    """Conv1d → BatchNorm1d → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = -1):
        if padding < 0:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


# ── Main model ────────────────────────────────────────────────────────────────

class FCNWang(nn.Module):
    """
    FCN-Wang baseline with Strodthoff concat-pooling head.

    Parameters
    ----------
    in_channels : int
        Number of ECG leads (default 12).
    num_classes : int
        Number of output classes (default 5 for PTB-XL superclasses).
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes:  int = NUM_CLASSES,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Three FCN blocks — no spatial downsampling between blocks.
        # Kernel sizes (8, 5, 3) follow Wang et al. (2017) exactly.
        # padding = kernel_size // 2 keeps sequence length unchanged.
        # ------------------------------------------------------------------
        self.block1 = _ConvBnRelu(in_channels, 128, kernel_size=8, padding=4)
        self.block2 = _ConvBnRelu(128,         256, kernel_size=5, padding=2)
        self.block3 = _ConvBnRelu(256,         128, kernel_size=3, padding=1)

        # ------------------------------------------------------------------
        # Concat-pooling head (Strodthoff et al., Appendix I).
        # GlobalAvgPool and GlobalMaxPool each produce (B, 128, 1);
        # after flatten and cat the feature dim is 256.
        # ------------------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            # 128 (avg) + 128 (max) = 256 features
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.50),
            nn.Linear(128, num_classes),
            # No sigmoid — caller uses BCEWithLogitsLoss
        )

        self._init_weights()
        self._print_summary()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming init for Conv1d, Xavier for Linear, constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _print_summary(self) -> None:
        p = self.count_parameters()
        print(
            f"FCN-Wang | "
            f"Params: {p['total']:,} total, {p['trainable']:,} trainable "
            f"({p['percentage']})"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 12, T) — raw or preprocessed ECG, any sequence length T

        Returns:
            logits : (B, num_classes) — raw logits (no sigmoid)
        """
        x = self.block1(x)   # (B, 128, T)
        x = self.block2(x)   # (B, 256, T)
        x = self.block3(x)   # (B, 128, T)

        avg = self.gap(x).flatten(1)          # (B, 128)
        mx  = self.gmp(x).flatten(1)          # (B, 128)
        x   = torch.cat([avg, mx], dim=1)     # (B, 256)

        return self.head(x)                   # (B, num_classes)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {
            'trainable':  trainable,
            'total':      total,
            'percentage': f'{100 * trainable / total:.1f}%',
        }

    def save(self, path: str) -> None:
        """Save state_dict to path (file or directory)."""
        if os.path.isdir(path) or path.endswith('/') or path.endswith('\\'):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, 'checkpoint.pt')
        else:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f'Saved → {path}')

    @classmethod
    def load(cls, path: str, **kwargs) -> 'FCNWang':
        """Load from a checkpoint file or directory."""
        if os.path.isdir(path):
            path = os.path.join(path, 'checkpoint.pt')
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model


# ── Factory ───────────────────────────────────────────────────────────────────

def fcn_wang(**kwargs) -> FCNWang:
    """FCN-Wang baseline — ~270k params, AUROC ≈ 0.925 on PTB-XL super-diag."""
    return FCNWang(**kwargs)
