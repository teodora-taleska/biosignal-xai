import os
import torch
from peft import PeftModel

from src.utils.config import CFG


def load_xresnet(layers=None):
    """
    Load XResNet1D-101 from checkpoint.
    layers: e.g. [3,4,23,3] for -101 (default) or [3,4,6,3] for -50.
    Returns (model, ECGDatasetAblation config dict).
    """
    from src.models.xresnet1d import XResNet1d
    from src.preprocessing.dataset_ablation import ABLATION_CONFIGS

    path  = os.path.join(CFG['paths']['results'], 'xresnet_baseline', 'checkpoint.pt')
    model = XResNet1d(layers=layers)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    # Return the canonical preprocessing config used during training
    return model, ABLATION_CONFIGS['bandpass_zscore_250']


def load_hubert_blocks(n=8):
    """Load HuBERTECGClassifier (selective unfreezing). Returns (model, ECGDatasetFull)."""
    from src.models.hubert_ecg_finetune import HuBERTECGClassifier
    from src.preprocessing.dataset_full import ECGDatasetFull

    path = os.path.join(
        CFG['paths']['results'], f'hubert_ecg_blocks{n}', 'best_adapter', 'checkpoint.pt'
    )
    model = HuBERTECGClassifier(size='base', blocks_to_unfreeze=n)
    ckpt  = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ECGDatasetFull


def load_hubert_peft(rank=8, use_dora=False):
    """Load HuBERTECGPEFT (LoRA or DoRA). Returns (model, ECGDatasetFull)."""
    from src.models.hubert_ecg_finetune import HuBERTECGPEFT
    from src.preprocessing.dataset_full import ECGDatasetFull

    suffix       = 'dora' if use_dora else 'lora'
    adapter_path = os.path.join(
        CFG['paths']['results'], f'hubert_ecg_{suffix}_r{rank}', 'best_adapter'
    )
    model = HuBERTECGPEFT(rank=rank, use_dora=use_dora)
    # model.backbone is a PeftModel; base_model.model is the raw HuBERT backbone
    raw_backbone   = model.backbone.base_model.model
    model.backbone = PeftModel.from_pretrained(raw_backbone, adapter_path)
    model.classifier.load_state_dict(
        torch.load(os.path.join(adapter_path, 'classifier.pt'), map_location='cpu')
    )
    model.eval()
    return model, ECGDatasetFull


def load_leadwise():
    """Load LeadwiseTransformer from checkpoint. Returns (model, ECGDatasetFull)."""
    from src.models.leadwise_transformer import LeadwiseTransformer
    from src.preprocessing.dataset_full import ECGDatasetFull

    path = os.path.join(
        CFG['paths']['results'], 'leadwise_transformer', 'best_adapter', 'checkpoint.pt'
    )
    model = LeadwiseTransformer()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model, ECGDatasetFull
