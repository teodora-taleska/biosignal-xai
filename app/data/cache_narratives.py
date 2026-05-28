"""
Pre-generate Qwen2 clinical narratives for all 200 curated records.

Saves app/data/narratives_cache.json keyed by ecg_id string.
Used in cloud deployment mode so Qwen2 never loads at runtime.

Run once locally (requires GPU + Qwen2 downloaded):
    python app/data/cache_narratives.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explainability.llm import load_qwen, build_ecg_prompt, generate_explanation
from src.explainability.saliency import compute_saliency
from src.models.fcn_wang import FCNWang
from src.utils.config import CFG

# ── Paths ─────────────────────────────────────────────────────────────────────
_FILE    = Path(__file__).resolve()
APP_DATA = _FILE.parent

def _find_dir(start: Path, name: str) -> Path:
    candidate = start / name
    if candidate.exists():
        return candidate
    for parent in start.parents:
        candidate = parent / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find '{name}/' above {start}")

def _find_checkpoint(start: Path, rel: str) -> Path:
    candidate = start / rel
    if candidate.exists():
        return candidate
    for parent in start.parents:
        candidate = parent / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found: {rel}")

REPO_ROOT    = _FILE.parents[2]
DATA_DIR     = _find_dir(REPO_ROOT, 'data')
CKPT_PATH    = _find_checkpoint(REPO_ROOT, 'results/fcn_wang_baseline/checkpoint.pt')
SUPERCLASSES = CFG['data']['superclasses']
LEAD_NAMES   = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


def build_narratives() -> None:
    print('=== Building narratives cache (Qwen2-0.5B-Instruct) ===\n')

    # Load curated index + predictions cache
    with open(APP_DATA / 'curated_200.json') as f:
        curated = json.load(f)
    with open(APP_DATA / 'predictions_cache.json') as f:
        predictions = json.load(f)

    # Load FCN-Wang for saliency
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = FCNWang.load(str(CKPT_PATH))
    model.to(device).eval()
    print(f'FCN-Wang loaded  (device={device})')

    # Load Qwen2
    qwen_model, qwen_tok = load_qwen(model_id='Qwen/Qwen2-0.5B-Instruct')
    print()

    narratives: dict[str, str] = {}
    total = len(curated)

    for i, rec in enumerate(curated, 1):
        ecg_id  = str(rec['ecg_id'])
        pred    = predictions.get(ecg_id)

        if pred is None:
            print(f'  [{i}/{total}] ECG {ecg_id}: no prediction cached, skipping')
            continue

        # Load signal
        path      = str(DATA_DIR / rec['filename_lr'])
        signal, _ = wfdb.rdsamp(path)
        x_np      = signal.T.astype(np.float32)   # (12, 1000)

        # Gradient saliency for the top predicted class
        target_cls = pred['predicted_classes'][0]
        target_idx = SUPERCLASSES.index(target_cls)
        saliency   = compute_saliency(model, x_np, target_idx, device)

        # Build prompt and generate
        prompt     = build_ecg_prompt(
            result_dict  = pred,
            saliency     = saliency,
            lead_names   = LEAD_NAMES,
            true_classes = rec.get('superclass'),
        )
        narrative  = generate_explanation(prompt, qwen_model, qwen_tok)
        narratives[ecg_id] = narrative

        if i % 20 == 0 or i == total:
            print(f'  {i}/{total} done')

    out_path = APP_DATA / 'narratives_cache.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(narratives, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {out_path}  ({len(narratives)} narratives)')
    print('=== Done ===')


if __name__ == '__main__':
    build_narratives()
