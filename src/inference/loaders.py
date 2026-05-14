import os
import torch
from peft import PeftModel

from src.utils.config import CFG


def load_baseline_cnn():
    """Load BaselineCNN from checkpoint. Returns (model, ECGDataset)."""
    from src.models.baseline_cnn import BaselineCNN
    from src.data.dataset import ECGDataset

    path = os.path.join(CFG['paths']['results'], 'baseline_cnn', 'checkpoint.pt')
    model = BaselineCNN()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model, ECGDataset


def load_hubert_blocks(n=8):
    """Load HuBERTECGClassifier (selective unfreezing). Returns (model, ECGDatasetFull)."""
    from src.models.hubert_ecg_finetune import HuBERTECGClassifier
    from src.data.dataset_full import ECGDatasetFull

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
    from src.data.dataset_full import ECGDatasetFull

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
    from src.data.dataset_full import ECGDatasetFull

    path = os.path.join(
        CFG['paths']['results'], 'leadwise_transformer', 'best_adapter', 'checkpoint.pt'
    )
    model = LeadwiseTransformer()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model, ECGDatasetFull
