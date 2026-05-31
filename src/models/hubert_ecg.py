"""
HuBERT-ECG — foundation ECG model pretrained on 9.1 M 12-lead ECGs.

Architecture : HuBERT-style convolutional feature extractor + Transformer encoder
Sizes        : small (30 M), base (93 M), large (188 M)
HuggingFace  : Edoardo-BS/hubert-ecg-{small|base|large}
License      : CC BY-NC 4.0 — research use only
Paper        : https://www.medrxiv.org/content/10.1101/2024.11.14.24317328

Adapted for PTB-XL 5-class multi-label classification.
trust_remote_code=True required — HuBERT-ECG ships its own model class.
"""

import os

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig


class _HuBERTHead(nn.Module):
    """Mean-pool encoder hidden states → linear classifier."""

    def __init__(self, encoder: nn.Module, hidden_dim: int, num_labels: int):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HuBERT feature extractor expects (B, T) — reshape leads into batch dim.
        B, num_leads, T = x.shape                          # (B, 12, 1000)
        x = x.reshape(B * num_leads, T)                    # (B*12, 1000)
        hidden = self.encoder(x).last_hidden_state          # (B*12, T', H)
        pooled = hidden.mean(dim=1)                         # (B*12, H)
        pooled = pooled.reshape(B, num_leads, -1).mean(1)   # (B, H) — mean over leads
        return self.classifier(pooled)                      # (B, num_labels)


class HuBERTECGClassifier(nn.Module):
    """
    HuBERT-ECG with classification head for PTB-XL 5-class multi-label task.

    Input  : (B, 12, 1000) float32 — 12-lead ECG at 100 Hz
    Output : (B, 5)        float32 — raw logits (apply sigmoid for probabilities)

    Usage
    -----
    model = HuBERTECGClassifier(size="base", num_labels=5)
    model.load()
    model.apply_peft(use_dora=False)
    model.to(device)
    logits = model(x_batch)           # use in run_peft_experiment()
    model.save("results/best_adapter")
    """

    def __init__(self, size: str = "base", num_labels: int = 5):
        super().__init__()
        assert size in ("small", "base", "large"), "size must be small, base, or large"
        self.size       = size
        self.num_labels = num_labels
        self._head      = None   # set by load(); registered as submodule via __setattr__

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self):
        """Download pretrained encoder from HuggingFace and attach classifier head."""
        hf_id = f"Edoardo-BS/hubert-ecg-{self.size}"
        print(f"Loading HuBERT-ECG ({self.size}) from {hf_id} ...")
        encoder    = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        hidden_dim = getattr(encoder.config, "hidden_size", 768)
        self._head = _HuBERTHead(encoder, hidden_dim, self.num_labels)
        print(f"  Done. Hidden dim: {hidden_dim}")
        return self

    # ── PEFT ──────────────────────────────────────────────────────────────────

    def apply_peft(
        self,
        r: int         = 16,
        alpha: int     = 32,
        dropout: float = 0.1,
        use_dora: bool = False,
    ):
        """
        Attach LoRA (use_dora=False) or DoRA (use_dora=True) to Q/K/V projections.
        Falls back to head-only fine-tuning if layer names don't match.
        """
        assert self._head is not None, "Call .load() first."
        cfg = LoraConfig(
            r              = r,
            lora_alpha     = alpha,
            lora_dropout   = dropout,
            target_modules = ["q_proj", "k_proj", "v_proj"],
            bias           = "none",
            use_dora       = use_dora,
        )
        try:
            self._head = get_peft_model(self._head, cfg)
            self._head.print_trainable_parameters()
        except Exception as e:
            print(f"  LoRA target mismatch ({e}). Falling back to head-only fine-tuning.")
            for name, param in self._head.named_parameters():
                param.requires_grad = "classifier" in name
            n = sum(p.numel() for p in self._head.parameters() if p.requires_grad)
            print(f"  Trainable params: {n:,}")
        return self

    # ── Parameters ───────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":      total,
            "trainable":  trainable,
            "percentage": f"{100 * trainable / total:.1f}%",
        }

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(x)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save PEFT adapter weights (or full head state if head-only)."""
        try:
            self._head.save_pretrained(path)
            print(f"Adapter saved → {path}")
        except AttributeError:
            os.makedirs(path, exist_ok=True)
            torch.save(self._head.state_dict(), os.path.join(path, "weights.pt"))
            print(f"Weights saved → {path}/weights.pt")

    # ── Adapter loading ───────────────────────────────────────────────────────

    def load_adapter(self, path: str):
        """Load a saved PEFT adapter into the already-loaded head."""
        from peft import PeftModel
        assert self._head is not None, "Call .load() first."
        self._head = PeftModel.from_pretrained(self._head, path)
        self._head.eval()
        return self
