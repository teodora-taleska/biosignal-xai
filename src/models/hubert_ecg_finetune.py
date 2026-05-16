import os

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

from src.utils.config import CFG

SUPERCLASSES = CFG['preprocessing']['superclasses']
NUM_CLASSES  = len(SUPERCLASSES)

# ──────────────────────────────────────────────────────────────
# HuBERT-ECG Fine-tuning Strategy
#
# HuBERT-ECG uses a custom HuBERT-based architecture pretrained
# on 9.1 million 12-lead ECGs in a self-supervised manner.
# It cannot be wrapped directly with LoRA via PEFT because it
# uses custom layer classes (HubertEncoderLayer, etc.)
#
# Instead we use "selective layer unfreezing" — the approach from
# the original paper:
#   - Freeze all layers
#   - Unfreeze only the last N transformer blocks
#   - Add a new classification head (always trainable)
#
# KEY ARCHITECTURE DETAIL (confirmed from model inspection):
#   - Input: 1 channel — model processes ONE lead at a time
#   - 12 encoder layers (encoder.layers.0 to encoder.layers.11)
#   - Hidden dim: 768
#   - We reshape (B, 12, 1000) → (B*12, 1, 1000) per forward pass
#     then mean-pool across leads before classifying
#
# This is parameter-efficient: unfreezing 4 of 12 blocks ≈ 33% params
# We compare N=4 vs N=8 blocks → our ablation study
#
# Paper:       https://doi.org/10.1101/2024.11.14.24317328
# HuggingFace: https://huggingface.co/Edoardo-BS/hubert-ecg-base
# ──────────────────────────────────────────────────────────────


class HuBERTECGClassifier(nn.Module):
    """
    HuBERT-ECG with selective layer unfreezing.

    KEY INSIGHT from architecture inspection:
    HuBERT-ECG takes 1-channel input (one lead at a time).
    For 12-lead ECG we process each lead independently
    through the shared backbone, then mean-pool the 12
    feature vectors before classifying.

    This is clinically sensible: the model learns
    universal ECG patterns applicable to any lead.

    Input:  (batch, 12, 1000)
    Per-lead: (batch*12, 1, 1000) → backbone → (batch*12, 768)
    Reshape:  (batch, 12, 768)    → mean     → (batch, 768)
    Output:   (batch, 5)
    """

    def __init__(
        self,
        size:               str   = CFG['model']['hubert_size'],
        blocks_to_unfreeze: int   = 4,
        dropout:            float = 0.2,
    ):
        super().__init__()
        self.size               = size
        self.blocks_to_unfreeze = blocks_to_unfreeze
        hidden_dim              = 768   # confirmed from print output

        # Load pretrained backbone
        print(f"Loading HuBERT-ECG-{size}...")
        self.backbone = AutoModel.from_pretrained(
            f"Edoardo-BS/hubert-ecg-{size}",
            trust_remote_code=True,
        )
        print("Loaded.")

        # Freeze all backbone params
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer blocks
        self._unfreeze_last_n_blocks(blocks_to_unfreeze)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

        self._print_parameter_summary()

    def _unfreeze_last_n_blocks(self, n: int):
        """
        Unfreeze last N encoder layers.
        Layer names confirmed from model printout:
        'encoder.layers.0.' through 'encoder.layers.11.'
        """
        total           = 12   # confirmed: 12 encoder layers
        blocks_to_train = set(range(total - n, total))

        unfrozen = 0
        for name, param in self.backbone.named_parameters():
            for idx in blocks_to_train:
                if f"encoder.layers.{idx}." in name:
                    param.requires_grad = True
                    unfrozen += 1
                    break

        print(f"Unfrozen {unfrozen} tensors from last {n} blocks "
              f"(layers {total-n}–{total-1})")

    def _print_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 12, 1000) — 12 leads, 1000 time steps

        Returns:
            logits: (batch, 5)
        """
        B, L, T = x.shape   # batch=B, leads=12, time=1000

        # Reshape: treat each lead as a separate sample
        # (B, 12, 1000) → (B*12, 1000)
        # HubertModel.forward() expects (batch, time) — the feature encoder
        # does its own [:, None] unsqueeze to (batch, 1, time) internally
        x = x.reshape(B * L, T)  # (B*12, 1000)

        out      = self.backbone(x)

        # Mean pool over time → (B*12, 768)
        features = out.last_hidden_state.mean(dim=1)

        # Reshape back and average across leads
        # (B*12, 768) → (B, 12, 768) → (B, 768)
        features = features.reshape(B, L, -1).mean(dim=1)

        return self.classifier(features)   # (B, 5)

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {
            "trainable":  trainable,
            "total":      total,
            "percentage": f"{100*trainable/total:.1f}%"
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "state_dict":         self.state_dict(),
            "size":               self.size,
            "blocks_to_unfreeze": self.blocks_to_unfreeze,
        }, f"{path}/checkpoint.pt")
        print(f"Saved -> {path}/checkpoint.pt")


# ──────────────────────────────────────────────────────────────────────────────


class HuBERTECGPEFT(nn.Module):
    """
    HuBERT-ECG with LoRA or DoRA (true PEFT).

    All 93M backbone weights frozen. LoRA/DoRA adapters injected into
    all 12 encoder blocks at q/k/v/out_proj — standard nn.Linear layers
    confirmed from model inspection. Trainable params: adapters + head
    (~2% of total for LoRA r=8, ~2.1% for DoRA r=8).

    Confirmed compatible target modules:
      encoder.layers.{i}.attention.{q,k,v,out}_proj  (Linear 768->768)

    Architecture note: backbone takes 2D input (B, T) — the feature
    encoder unsqueezes to (B, 1, T) internally. Same reshape trick as
    HuBERTECGClassifier: process 12 leads independently, mean-pool back.
    """

    def __init__(
        self,
        rank:     int   = 8,
        use_dora: bool  = False,
        dropout:  float = 0.2,
    ):
        super().__init__()
        self.rank     = rank
        self.use_dora = use_dora
        hidden_dim    = 768

        method = 'DoRA' if use_dora else 'LoRA'
        print(f"Loading HuBERT-ECG-base ({method} r={rank})...")
        backbone = AutoModel.from_pretrained(
            "Edoardo-BS/hubert-ecg-base",
            trust_remote_code=True,
        )
        print("Loaded.")

        # Freeze all backbone params before injecting adapters
        for param in backbone.parameters():
            param.requires_grad = False

        lora_cfg = LoraConfig(
            r              = rank,
            lora_alpha     = rank,   # alpha/r = 1 (multiplier=1x per config)
            target_modules = CFG['model']['hubert_peft']['target_modules'],
            lora_dropout   = 0.1,
            use_dora       = use_dora,
            bias           = "none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        # Classification head — fresh init, always trainable
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

        self._print_parameter_summary()

    def _print_parameter_summary(self):
        p = self.count_parameters()
        print(f"HuBERTECGPEFT | "
              f"Trainable: {p['trainable']:,} / {p['total']:,} ({p['percentage']})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 1000) — 12 leads, 1000 time steps
        Returns:
            logits: (B, 5)
        """
        B, L, T  = x.shape
        x        = x.reshape(B * L, T)                    # (B*12, 1000)
        out      = self.backbone(x)
        features = out.last_hidden_state.mean(dim=1)       # (B*12, 768)
        features = features.reshape(B, L, -1).mean(dim=1) # (B, 768)
        return self.classifier(features)                   # (B, 5)

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {
            "trainable":  trainable,
            "total":      total,
            "percentage": f"{100 * trainable / total:.1f}%",
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save LoRA/DoRA adapter weights only (small — a few MB vs 350MB full)
        self.backbone.save_pretrained(path, save_embedding_layers=False)
        # Save classifier head separately (not part of PEFT adapter)
        torch.save(
            self.classifier.state_dict(),
            os.path.join(path, "classifier.pt"),
        )
        print(f"Saved -> {path}")