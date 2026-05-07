from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from src.utils.config import CFG

SUPERCLASSES = CFG['data']['superclasses']
NUM_CLASSES  = len(SUPERCLASSES)

_LW = CFG['model']['leadwise']


class LeadwiseTransformer(nn.Module):
    """
    Lead-wise Transformer for 12-lead ECG classification.

    Novel contribution: after per-lead temporal encoding (shared weights),
    a cross-lead attention layer lets each lead attend to all other leads,
    mirroring how cardiologists compare leads in diagnosis (e.g. ST elevation
    in II, III, aVF simultaneously for inferior MI).

    Architecture:
      Stage 1: Per-lead patch embedding (Conv1d projection)
      Stage 2: Temporal Transformer encoder (shared weights across all leads)
      Stage 3: Cross-lead attention (leads attend to each other)  <-- novel
      Stage 4: Mean aggregation across leads
      Stage 5: Classification head
    """

    def __init__(
        self,
        d_model:             int   = _LW['d_model'],
        num_temporal_layers: int   = _LW['num_temporal_layers'],
        num_heads:           int   = _LW['num_heads'],
        num_cross_heads:     int   = _LW['num_cross_heads'],
        dropout:             float = _LW['dropout'],
        head_dropout:        float = _LW['head_dropout'],
        patch_length:        int   = _LW['patch_length'],
        patch_stride:        int   = _LW['patch_stride'],
    ):
        super().__init__()

        self.d_model = d_model

        # Compute number of patches from config, no hardcoded 40/1000
        time_steps = CFG['data']['sampling_rate'] * 10   # 1000 for 10-sec recordings at 100 Hz
        n_patches  = (time_steps - patch_length) // patch_stride + 1  # (1000-25)//25+1 = 40

        #  Stage 1: Patch projection 
        # Each lead treated as a 1-channel time series.
        # (B*12, 1, 1000) → Conv1d → (B*12, d_model, n_patches) → permute → (B*12, n_patches, d_model)
        self.patch_proj = nn.Conv1d(
            in_channels  = 1,
            out_channels = d_model,
            kernel_size  = patch_length,
            stride       = patch_stride,
        )

        # CLS token: one per lead, shared across batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Sinusoidal PE for n_patches + 1 positions (patches + CLS)
        self.register_buffer('pe', self._sinusoidal_pe(n_patches + 1, d_model))

        #  Stage 2: Temporal encoder (shared weights across leads) ─
        # Pre-LN (norm_first=True) is more stable when training from scratch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = num_heads,
            dim_feedforward = d_model * 2,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_temporal_layers
        )

        #  Stage 3: Cross-lead attention (novel) 
        # Q = K = V = the 12 lead tokens; leads attend to each other
        self.cross_lead_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_cross_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.cross_lead_norm = nn.LayerNorm(d_model)

        #  Stage 5: Classification head ─
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model // 2, NUM_CLASSES),
        )

        self._init_weights()
        self._print_parameter_summary()

    #  Helpers 

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def _print_parameter_summary(self):
        p = self.count_parameters()
        print(f"LeadwiseTransformer | "
              f"Trainable: {p['trainable']:,} / {p['total']:,} ({p['percentage']})")

    #  Forward 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 1000) -- 12 leads, 1000 time steps

        Returns:
            logits: (B, 5)
        """
        B, L, T = x.shape  # batch, leads=12, time=1000

        # Stage 1: Patch embedding
        x   = x.reshape(B * L, 1, T)                  # (B*12, 1, 1000)
        x   = self.patch_proj(x)                       # (B*12, d_model, n_patches)
        x   = x.permute(0, 2, 1)                      # (B*12, n_patches, d_model)

        # Stage 2: Temporal encoder
        cls = self.cls_token.expand(B * L, -1, -1)    # (B*12, 1, d_model)
        x   = torch.cat([cls, x], dim=1)               # (B*12, n_patches+1, d_model)
        x   = x + self.pe                              # add sinusoidal PE
        x   = self.temporal_encoder(x)                 # (B*12, n_patches+1, d_model)
        x   = x[:, 0, :]                              # CLS token: (B*12, d_model)
        x   = x.reshape(B, L, -1)                     # (B, 12, d_model)

        # Stage 3: Cross-lead attention
        attn_out, _ = self.cross_lead_attn(x, x, x)   # (B, 12, d_model)
        x = self.cross_lead_norm(x + attn_out)         # residual + LN

        # Stage 4: Aggregate across leads
        x = x.mean(dim=1)                              # (B, d_model)

        # Stage 5: Classify
        return self.classifier(x)                      # (B, 5)

    #  Persistence 

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

    def apply_peft(self, rank: int = 8, use_dora: bool = False):
        """
        Wrap this model with LoRA or DoRA adapters.

        Targets "out_proj" (nn.Linear inside nn.MultiheadAttention) in:
          - temporal_encoder.layers.{i}.self_attn.out_proj  (3 temporal blocks)
          - cross_lead_attn.out_proj                         (cross-lead stage)

        After calling this, only adapter deltas + classifier are trainable.
        Returns a PeftModel with .save() and .count_parameters() patched for
        compatibility with run_experiment.

        Note: call apply_peft on a freshly-created model only. It mutates
        self.classifier (wraps it in ModulesToSaveWrapper).
        """
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r               = rank,
            lora_alpha      = rank,
            lora_dropout    = 0.05,
            bias            = "none",
            use_dora        = use_dora,
            target_modules  = ["out_proj"],
            modules_to_save = ["classifier"],
        )
        peft_model = get_peft_model(self, lora_cfg)
        peft_model.print_trainable_parameters()

        # Patch methods required by run_experiment (PeftModel lacks them)
        def _count():
            t = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            n = sum(p.numel() for p in peft_model.parameters())
            return {'trainable': t, 'total': n, 'percentage': f'{100*t/n:.1f}%'}

        def _save(path: str):
            os.makedirs(path, exist_ok=True)
            peft_model.save_pretrained(path)
            print(f'Saved -> {path}')

        peft_model.count_parameters = _count
        peft_model.save = _save
        return peft_model


def build_leadwise_with_peft(rank: int = 8, use_dora: bool = False) -> nn.Module:
    """
    Build a LeadwiseTransformer and immediately apply LoRA or DoRA.
    Returns a PeftModel ready for run_experiment.
    All architecture config comes from CFG['model']['leadwise'].
    """
    return LeadwiseTransformer().apply_peft(rank=rank, use_dora=use_dora)
