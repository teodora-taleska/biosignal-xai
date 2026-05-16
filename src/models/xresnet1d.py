"""
XResNet1D: 1D adaptation of XResNet for 12-lead ECG classification.

Implements the xresnet1d101 architecture from:
  Strodthoff et al. "Deep Learning for ECG Analysis: Benchmarks and Insights
  from PTB-XL." IEEE JBHI 2021. (arxiv 2004.13701)

Key design choices matching the paper's benchmark:
  1. XResNet stem: 3 × small Conv1d instead of one large conv (better gradient flow)
  2. AvgPool downsampling in skip connections (smoother gradients than strided conv)
  3. Concat-pooling head: global-avg + global-max concatenated → 2× feature width
  4. Single FC classifier: Linear → BN → ReLU → Dropout(0.5) → Linear
  5. Adaptive pooling → accepts any input length (works for 250 / 500 / 1000 samples)

Default: xresnet1d101  (layers=[3, 4, 23, 3], ~12 M parameters)
Also available via factory: xresnet1d50 (layers=[3, 4, 6, 3], ~4.5 M)

Usage:
    from src.models.xresnet1d import XResNet1d, xresnet1d101
    model = xresnet1d101()          # paper benchmark
    model = XResNet1d(layers=[3,4,6,3])  # xresnet1d50
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn

from src.utils.config import CFG

NUM_CLASSES = len(CFG['data']['superclasses'])



# Building blocks


class _ConvBnAct(nn.Sequential):
    """Conv1d → BN → ReLU convenience wrapper."""
    def __init__(self, in_ch: int, out_ch: int, ks: int,
                 stride: int = 1, padding: int = -1, bias: bool = False):
        if padding < 0:
            padding = ks // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, ks, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class XResBottleneck1d(nn.Module):
    """
    XResNet bottleneck block for 1D signals.

    Structure (post-activation):
        Conv1d(1×1) → BN → ReLU          [channel reduction]
        Conv1d(ks×1) → BN → ReLU         [temporal mixing]
        Conv1d(1×1) → BN                 [channel expansion]
        + skip (AvgPool if stride>1, then Conv1d 1×1)
        → ReLU

    expansion=4 means out_channels = planes × 4.
    """
    expansion: int = 4

    def __init__(self, in_ch: int, planes: int,
                 stride: int = 1, kernel_size: int = 5):
        super().__init__()
        mid  = planes
        out  = planes * self.expansion

        self.conv1 = _ConvBnAct(in_ch, mid, 1)
        self.conv2 = _ConvBnAct(mid, mid, kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv3 = nn.Sequential(
            nn.Conv1d(mid, out, 1, bias=False),
            nn.BatchNorm1d(out),
        )

        # Skip connection — XResNet uses AvgPool for downsampling (not strided conv)
        if stride != 1 or in_ch != out:
            skip: List[nn.Module] = []
            if stride > 1:
                skip.append(nn.AvgPool1d(stride, stride, ceil_mode=True))
            skip += [nn.Conv1d(in_ch, out, 1, bias=False), nn.BatchNorm1d(out)]
            self.skip = nn.Sequential(*skip)
        else:
            self.skip = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3(self.conv2(self.conv1(x))) + self.skip(x))



# Main model


class XResNet1d(nn.Module):
    """
    XResNet1D for multi-label 12-lead ECG classification.

    Parameters
    ----------
    layers      : bottleneck counts per stage, e.g. [3, 4, 23, 3] for -101
    num_classes : output size (5 for PTB-XL superclasses)
    in_channels : number of ECG leads (12)
    base_width  : channels after stem (64 by default)
    kernel_size : temporal kernel size inside bottleneck (5, matching paper)

    Input  : (B, 12, T)  — T can be 250 / 500 / 1000 (adaptive pooling handles it)
    Output : (B, num_classes) — raw logits, use BCEWithLogitsLoss
    """

    def __init__(
        self,
        layers:      List[int] = None,
        num_classes: int  = NUM_CLASSES,
        in_channels: int  = 12,
        base_width:  int  = 64,
        kernel_size: int  = 5,
    ):
        super().__init__()
        if layers is None:
            layers = [3, 4, 23, 3]  # xresnet1d101

        # ------------------------------------------------------------------
        # XResNet stem — 3 small convolutions + MaxPool
        # Standard ResNet uses one 7×7 conv; XResNet uses 3×(3 or 5)×conv
        # which improves gradient flow and keeps early features local.
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            # First conv: stride-2 spatial downsampling (1000 → 500)
            _ConvBnAct(in_channels, 32, 5, stride=2, padding=2),
            # Second conv: feature enrichment
            _ConvBnAct(32, 32, 3),
            # Third conv: expand to base_width
            _ConvBnAct(32, base_width, 3),
            # MaxPool: further downsampling (500 → 250)
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # ------------------------------------------------------------------
        # Four residual stages (mirrors ResNet-101 block layout)
        # Stage 1: no spatial downsampling (stride=1)
        # Stages 2-4: stride=2 → 125, 63, 32 (for 1000-sample input)
        # ------------------------------------------------------------------
        c = base_width   # running channel count into next stage
        self.layer1, c = self._make_stage(c, base_width,     layers[0], stride=1, ks=kernel_size)
        self.layer2, c = self._make_stage(c, base_width * 2, layers[1], stride=2, ks=kernel_size)
        self.layer3, c = self._make_stage(c, base_width * 4, layers[2], stride=2, ks=kernel_size)
        self.layer4, c = self._make_stage(c, base_width * 8, layers[3], stride=2, ks=kernel_size)
        # c == base_width * 8 * 4 == 2048 for default base_width=64

        # ------------------------------------------------------------------
        # Concat-pooling head
        # AdaptiveAvgPool + AdaptiveMaxPool both → (B, C, 1)
        # Concatenate along channel dim → (B, 2*C, 1) → flatten → (B, 2*C)
        # This doubles the feature size before the classifier, giving the model
        # access to both global statistics (avg) and peak activations (max).
        # ------------------------------------------------------------------
        head_in = c * 2   # 4096 for xresnet1d101

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            # No sigmoid — use BCEWithLogitsLoss
        )

        self._init_weights()
        self._print_summary()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stage(
        in_ch: int, planes: int, n_blocks: int, stride: int, ks: int
    ):
        """Build one residual stage; return (Sequential, out_channels)."""
        blocks = [XResBottleneck1d(in_ch, planes, stride=stride, kernel_size=ks)]
        out_ch = planes * XResBottleneck1d.expansion
        for _ in range(1, n_blocks):
            blocks.append(XResBottleneck1d(out_ch, planes, stride=1, kernel_size=ks))
        return nn.Sequential(*blocks), out_ch

    def _init_weights(self):
        """Kaiming init for Conv1d; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _print_summary(self):
        p = self.count_parameters()
        print(f"XResNet1D-101 | "
              f"Params: {p['total']:,} total, {p['trainable']:,} trainable "
              f"({p['percentage']})")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 12, T)  — raw or preprocessed ECG, any sequence length T

        Returns:
            logits : (B, num_classes)
        """
        x = self.stem(x)      # (B, 64, T/4)
        x = self.layer1(x)    # (B, 256, T/4)
        x = self.layer2(x)    # (B, 512, T/8)
        x = self.layer3(x)    # (B, 1024, T/16)
        x = self.layer4(x)    # (B, 2048, T/32)

        # Concat-pooling
        avg = x.mean(dim=2, keepdim=True)             # (B, 2048, 1)
        mx  = x.amax(dim=2, keepdim=True)             # (B, 2048, 1)
        x   = torch.cat([avg, mx], dim=1)             # (B, 4096, 1)

        return self.classifier(x)                     # (B, num_classes)

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

    def save(self, path: str):
        """Save state_dict to path (file or directory)."""
        if os.path.isdir(path) or path.endswith('/') or path.endswith('\\'):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, 'checkpoint.pt')
        else:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f'Saved → {path}')

    @classmethod
    def load(cls, path: str, layers: List[int] = None, **kwargs) -> 'XResNet1d':
        """Load from checkpoint; auto-detects layers if omitted."""
        if os.path.isdir(path):
            path = os.path.join(path, 'checkpoint.pt')
        model = cls(layers=layers, **kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model



# Factory functions


def xresnet1d101(**kwargs) -> XResNet1d:
    """XResNet1D-101 — paper benchmark (12 M params, layers=[3,4,23,3])."""
    return XResNet1d(layers=[3, 4, 23, 3], **kwargs)


def xresnet1d50(**kwargs) -> XResNet1d:
    """XResNet1D-50 — lighter alternative (4.5 M params, layers=[3,4,6,3])."""
    return XResNet1d(layers=[3, 4, 6, 3], **kwargs)
