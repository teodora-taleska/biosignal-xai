import os
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForClassification
from peft import LoraConfig, get_peft_model, PeftModel

from src.utils.config import CFG

# ──────────────────────────────────────────────────────────────
# WHY FROM SCRATCH:
#
# The only public pretrained PatchTST (IBM granite-timeseries-patchtst)
# was trained on ETTh1, an electrical dataset with 7 channels.
# Our ECG data has 12 channels, architecturally incompatible.
#
# Instead we train from scratch and use LoRA/DoRA as:
#   1. Parameter efficiency (fewer params = less overfitting on medical data)
#   2. Implicit regularization (low-rank constraint acts like weight decay)
#   3. Faster training (fewer gradients computed)
#
# "LoRA as Regularization" — Zhao et al. 2024
# ──────────────────────────────────────────────────────────────

SUPERCLASSES = CFG['data']['superclasses']


def build_patchtst_config(
        context_length: int = CFG['data']['window_size'],  # your window size
        patch_length: int = 25,  # 0.25 sec per patch at 100Hz
        d_model: int = 128,  # transformer hidden dim
        num_layers: int = 3,  # transformer depth
        num_heads: int = 8,  # attention heads
):
    """
    Build PatchTST config mapped to your ECG data.

    Input signal: (batch, 250, 12)
    Patching:     250 / 25 = 10 patches per lead
    Tokens:       10 patches × 12 leads + 1 CLS = 121 tokens
    Each token:   d_model=128 dimensional vector
    """
    assert context_length % patch_length == 0, \
        "context_length must be divisible by patch_length"
    assert d_model % num_heads == 0, \
        "d_model must be divisible by num_heads"

    return PatchTSTConfig(
        # Data dimensions
        num_input_channels=12,  # 12 ECG leads
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_length,  # no overlap between patches

        # Classification head
        num_targets=len(SUPERCLASSES),  # 5
        problem_type="multi_label_classification",

        # Transformer architecture
        d_model=d_model,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        ffn_dim=d_model * 2,  # standard = 2× or 4× d_model
        use_cls_token=True,  # CLS token for classification
        pooling_type="cls_token",

        # Regularization
        dropout=0.1,
        attention_dropout=0.1,
        head_dropout=0.2,

        # Preprocessing
        # We already normalized in preprocessing.py
        # so tell PatchTST not to normalize again
        scaling=None,

        # Channel independence
        # Each lead processed independently through same weights
        # Cross-lead reasoning is handled in 04b (lead-wise model)
        channel_attention=False,
    )


def build_lora_config(
        rank: int = 8,
        alpha: int = 16,
        use_dora: bool = False,
):
    """
    Build LoRA or DoRA config.

    target_modules: which linear layers get adapters
      - "q_proj", "v_proj" → standard choice from original LoRA paper
      - adding "k_proj", "out_proj" → more coverage, more params

    modules_to_save: layers that are fully unfrozen and trained normally
      - classifier head always needs to be trained from scratch
        because it's task-specific
    """
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        use_dora=use_dora,

        # Target Q and V projections inside attention
        # (standard from LoRA paper Section 4.2)
        target_modules=["q_proj", "v_proj"],

        # Classifier head is always fully trained
        modules_to_save=["classifier"],
    )


class PatchTSTClassifier(nn.Module):
    """
    PatchTST wrapped with LoRA or DoRA adapter.

    Handles the input shape mismatch between your DataLoader
    and PatchTST's expected format:
        DataLoader gives: (batch, 12, 250)  channels-first
        PatchTST expects: (batch, 250, 12)  time-first

    Usage:
        model = PatchTSTClassifier(rank=8, use_dora=False)  # LoRA
        model = PatchTSTClassifier(rank=8, use_dora=True)   # DoRA
        logits = model(x)  # shape (batch, 5)
    """

    def __init__(
            self,
            rank: int = 8,
            alpha: int = None,  # defaults to rank if None
            use_dora: bool = False,
            context_length: int = CFG['data']['window_size'],
            patch_length: int = 25,
            d_model: int = 128,
            num_layers: int = 3,
            num_heads: int = 8,
    ):
        super().__init__()

        if alpha is None:
            alpha = rank  # standard default

        # Build base model
        config = build_patchtst_config(
            context_length, patch_length, d_model, num_layers, num_heads
        )
        base_model = PatchTSTForClassification(config)

        # Wrap with PEFT adapter
        lora_cfg = build_lora_config(rank, alpha, use_dora)
        self.model = get_peft_model(base_model, lora_cfg)

        # Store config for saving/loading
        self.rank = rank
        self.use_dora = use_dora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 12, 250)  — from your DataLoader

        Returns:
            logits: (batch, 5)   — raw scores, pass through sigmoid for probs
        """
        # PatchTST wants (batch, time, channels)
        x = x.permute(0, 2, 1)  # → (batch, 250, 12)
        out = self.model(past_values=x)
        return out.prediction_logits  # (batch, 5)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def save(self, path: str):
        """Save only the adapter weights — small file."""
        os.makedirs(path, exist_ok=True)
        # save_embedding_layers=False: PatchTST has no vocabulary/tokenizer.
        # Without this, PEFT tries to check HF Hub for a tokenizer config,
        # fails (SSL/offline), and emits a spurious warning.
        self.model.save_pretrained(path, save_embedding_layers=False)

    @classmethod
    def load(cls, path: str, **kwargs) -> "PatchTSTClassifier":
        """
        Load a saved adapter back for inference.
        Used in your inference pipeline and product backend.
        """
        instance = cls(**kwargs)
        instance.model = PeftModel.from_pretrained(
            instance.model.base_model.model, path
        )
        instance.model.eval()
        return instance

    def count_parameters(self) -> dict:
        """Returns trainable vs total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "percentage": f"{100 * trainable / total:.2f}%"
        }