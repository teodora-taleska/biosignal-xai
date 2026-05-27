"""
Build the two static caches used by the Streamlit app:

  app/data/curated_200.json       — 200-record index (40 per superclass, test fold 10)
  app/data/predictions_cache.json — FCN-Wang predictions keyed by ecg_id string

Run once from the repo root:
    python -m app.data.cache
or:
    python app/data/cache.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.fcn_wang import FCNWang
from src.preprocessing.label_utils import load_all_labels
from src.utils.config import CFG

# ── Paths ─────────────────────────────────────────────────────────────────────
_FILE    = Path(__file__).resolve()
APP_DATA = _FILE.parent   # app/data/

def _find_dir(start: Path, name: str) -> Path:
    candidate = start / name
    if candidate.exists():
        return candidate
    for parent in start.parents:
        candidate = parent / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find '{name}/' directory above {start}")

def _find_checkpoint(start: Path, rel: str) -> Path:
    candidate = start / rel
    if candidate.exists():
        return candidate
    for parent in start.parents:
        candidate = parent / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found searching above {start}: {rel}")

REPO_ROOT   = _FILE.parents[2]
DATA_DIR    = _find_dir(REPO_ROOT, 'data')
CKPT_PATH   = _find_checkpoint(REPO_ROOT, 'results/fcn_wang_baseline/checkpoint.pt')

SUPERCLASSES = CFG['data']['superclasses']
N_PER_CLASS  = 40
TEST_FOLD    = 10
SEED         = 42


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model() -> tuple[FCNWang, torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = FCNWang.load(str(CKPT_PATH))
    model.to(device).eval()
    print(f'  FCN-Wang loaded from {CKPT_PATH}  (device={device})')
    return model, device


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(signal: np.ndarray) -> np.ndarray:
    """Transpose raw (1000, 12) signal to (12, 1000) for FCN-Wang. No filtering."""
    return signal.T.astype(np.float32)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(model: FCNWang, x_np: np.ndarray, device: torch.device) -> dict:
    """Run the model on a (12, 1000) array. Returns prediction dict."""
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(x)
    probs  = torch.sigmoid(logits).squeeze(0).cpu()

    threshold = CFG['inference']['threshold']
    predicted = [SUPERCLASSES[i] for i, p in enumerate(probs) if p.item() >= threshold]
    if not predicted:
        predicted = [SUPERCLASSES[probs.argmax().item()]]

    return {
        'predicted_classes':   predicted,
        'class_probabilities': {cls: round(probs[i].item(), 4)
                                for i, cls in enumerate(SUPERCLASSES)},
        'confidence_score':    round(probs.max().item(), 4),
    }


# ── Curated index building ────────────────────────────────────────────────────

def _build_curated_index(df) -> list[dict]:
    """Select 40 records per superclass from test fold 10."""
    rng      = np.random.default_rng(SEED)
    selected = []
    seen_ids = set()

    for sc in SUPERCLASSES:
        sc_idx = SUPERCLASSES.index(sc)
        mask   = (df['strat_fold'] == TEST_FOLD) & (
            df['label_vec'].apply(lambda v: v[sc_idx] == 1.0)
        )
        subset = df[mask]

        if len(subset) < N_PER_CLASS:
            print(f'  WARNING: only {len(subset)} records for {sc} in fold 10')

        n    = min(N_PER_CLASS, len(subset))
        idxs = rng.choice(len(subset), size=n, replace=False)
        rows = subset.iloc[idxs]

        for ecg_id, row in rows.iterrows():
            if ecg_id in seen_ids:
                continue
            seen_ids.add(ecg_id)
            selected.append({
                'ecg_id':        int(ecg_id),
                'patient_id':    int(row['patient_id']),
                'age':           float(row['age']) if not np.isnan(row['age']) else None,
                'sex':           str(row['sex']),
                'filename_lr':   str(row['filename_lr']),
                'superclass':    list(row['superclass']),
                'label_vec':     list(row['label_vec']),
                'primary_class': sc,
            })

    print(f'  Curated index: {len(selected)} records')
    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def build_caches() -> None:
    print('=== Building app caches (FCN-Wang) ===')

    print('\n[1/4] Loading PTB-XL labels ...')
    df = load_all_labels(str(DATA_DIR / 'ptbxl_database.csv'),
                         str(DATA_DIR / 'scp_statements.csv'))

    print('\n[2/4] Selecting curated 200-record index ...')
    curated = _build_curated_index(df)

    print('\n[3/4] Loading FCN-Wang checkpoint ...')
    model, device = _load_model()

    print('\n[4/4] Running FCN-Wang inference on 200 records ...')
    predictions: dict[str, dict] = {}
    for i, rec in enumerate(curated, 1):
        path      = str(DATA_DIR / rec['filename_lr'])
        signal, _ = wfdb.rdsamp(path)
        x         = _preprocess(signal.astype(np.float32))
        pred      = _predict(model, x, device)
        predictions[str(rec['ecg_id'])] = pred

        if i % 20 == 0 or i == len(curated):
            print(f'  {i}/{len(curated)} done')

    out_curated = APP_DATA / 'curated_200.json'
    out_preds   = APP_DATA / 'predictions_cache.json'

    with open(out_curated, 'w') as f:
        json.dump(curated, f, indent=2)
    print(f'\nSaved: {out_curated}')

    with open(out_preds, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f'Saved: {out_preds}')

    print('\n=== Done ===')


if __name__ == '__main__':
    build_caches()
