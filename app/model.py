"""
Streamlit-cached model and LLM loaders for the BioSignal-XAI app.

All heavy objects (FCNWang, Qwen) are loaded ONCE per Streamlit process
via @st.cache_resource.  Everything else calls these cached singletons.

Public API
----------
get_fcn_wang()             -> (model, device)
get_inference_pipeline()   -> ECGInferencePipeline
predict(signal_np)         -> dict
get_qwen3()                -> (qwen_model, qwen_tokenizer)  [optional]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch

from src.explainability.llm import load_qwen
from src.explainability.saliency import compute_saliency, top_salient_leads
from src.inference.pipeline import ECGInferencePipeline
from src.models.fcn_wang import FCNWang
from src.utils.config import CFG

# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_file(start: Path, rel: str) -> Path:
    """Walk upward from *start* until *rel* is found. Raises FileNotFoundError."""
    candidate = start / rel
    if candidate.exists():
        return candidate
    for parent in start.parents:
        candidate = parent / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find '{rel}' above {start}")


_REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_REL   = 'results/fcn_wang_baseline/checkpoint.pt'
CKPT_PATH  = _find_file(_REPO_ROOT, CKPT_REL)

SUPERCLASSES = CFG['data']['superclasses']
LEAD_NAMES   = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='Loading FCN-Wang model ...')
def get_fcn_wang() -> tuple[FCNWang, torch.device]:
    """Load FCN-Wang from checkpoint. Cached for the lifetime of the process."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = FCNWang.load(str(CKPT_PATH))
    model.to(device).eval()
    return model, device


@st.cache_resource(show_spinner='Initialising inference pipeline ...')
def get_inference_pipeline() -> ECGInferencePipeline:
    """Return a ready-to-use ECGInferencePipeline wrapping FCN-Wang."""
    model, device = get_fcn_wang()
    return ECGInferencePipeline(model=model, device=device, has_uncertainty=False)


@st.cache_resource(show_spinner='Loading Qwen2-0.5B-Instruct (first run downloads ~400 MB) ...')
def get_qwen3():
    """
    Load Qwen2-0.5B-Instruct for clinical narrative generation.

    Uses the same model as notebook 06_explainability.ipynb.
    Returns (model, tokenizer) or (None, None) if loading fails.
    """
    try:
        return load_qwen(model_id='Qwen/Qwen2-0.5B-Instruct')
    except Exception as exc:  # noqa: BLE001
        st.warning(f'Qwen not available: {exc}')
        return None, None


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Transpose raw PTB-XL signal to channels-first format.

    FCN-Wang was trained on ECGDatasetFull which applies no filtering --
    raw signals only, transposed to (12, 1000).

    Args:
        signal: (1000, 12) float32 raw PTB-XL signal

    Returns:
        (12, 1000) float32 array ready for FCN-Wang
    """
    return signal.T.astype(np.float32)  # (12, 1000)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(signal: np.ndarray) -> dict:
    """
    Run full-record inference on a raw (1000, 12) ECG signal.

    Returns prediction dict with keys:
        predicted_classes, class_probabilities, confidence_score,
        uncertainty (None), raw_logits, uncertainty_level
    """
    pipeline = get_inference_pipeline()
    x        = preprocess_signal(signal)   # (12, 1000)
    return pipeline.predict(x)


# ── Saliency ──────────────────────────────────────────────────────────────────

def get_saliency(
    signal: np.ndarray,
    target_class: Optional[str] = None,
    result: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute gradient saliency map for a signal.

    Args:
        signal:       raw (1000, 12) float32 signal
        target_class: class name to differentiate (default: top predicted class)
        result:       pre-computed predict() result (avoids re-inference)

    Returns:
        (12, 1000) float32 absolute gradient magnitude array
    """
    model, device = get_fcn_wang()
    x = preprocess_signal(signal)   # (12, 1000)

    if target_class is None:
        if result is None:
            result = predict(signal)
        target_class = result['predicted_classes'][0]

    target_idx = SUPERCLASSES.index(target_class)
    return compute_saliency(model, x, target_idx, device)


def get_top_leads(saliency: np.ndarray, top_k: int = 3) -> list[str]:
    """Return the top-k most salient lead names."""
    return top_salient_leads(saliency, LEAD_NAMES, top_k=top_k)
