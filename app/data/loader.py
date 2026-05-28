"""Data loading utilities for the BioSignal-XAI Streamlit app."""
from __future__ import annotations
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.utils.config import CFG

REPO_ROOT    = Path(__file__).resolve().parents[2]   # biosignal-xai/ (or worktree root)

# Data lives in the canonical repo root. When running from a git worktree
# (e.g. .claude/worktrees/<name>), walk up to find the actual data/ directory.
def _find_data_dir(start: Path) -> Path:
    candidate = start / 'data'
    if candidate.exists():
        return candidate
    # Walk up through parent directories looking for data/ptbxl_database.csv
    for parent in start.parents:
        candidate = parent / 'data'
        if (candidate / 'ptbxl_database.csv').exists():
            return candidate
    return start / 'data'  # fallback — will raise FileNotFoundError at load time

_DATA_DIR    = _find_data_dir(REPO_ROOT)
DATA_CSV     = _DATA_DIR / 'ptbxl_database.csv'
SCP_CSV      = _DATA_DIR / 'scp_statements.csv'
APP_DATA     = Path(__file__).parent                 # app/data/

SUPERCLASSES = CFG['data']['superclasses']           # ['NORM','MI','STTC','CD','HYP']
LEAD_NAMES   = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_metadata() -> pd.DataFrame:
    """Load PTB-XL metadata CSV with scp_codes parsed to dicts. Indexed by ecg_id."""
    df = pd.read_csv(DATA_CSV, index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    return df


def load_signal(filename_lr: str) -> np.ndarray:
    """Load a WFDB record by filename_lr. Returns (1000, 12) float32 array.

    Checks app/data/signals/ first (embedded for cloud deployment),
    then falls back to the full PTB-XL data directory.
    """
    # filename_lr is e.g. 'records100/00000/00001_lr' (no extension — WFDB convention)
    embedded = APP_DATA / 'signals' / filename_lr
    if Path(str(embedded) + '.hea').exists():
        path = str(embedded)
    else:
        path = str(_DATA_DIR / filename_lr)
    signal, _ = wfdb.rdsamp(path)
    return signal.astype(np.float32)


def load_curated_index() -> list[dict]:
    """Load curated 200-record index. Returns list of record dicts.
    Raises FileNotFoundError if cache not built yet (run app/data/cache.py)."""
    path = APP_DATA / 'curated_200.json'
    with open(path) as f:
        return json.load(f)


def load_predictions_cache() -> dict:
    """Load pre-computed FCN-Wang predictions keyed by ecg_id string.
    Raises FileNotFoundError if cache not built yet."""
    path = APP_DATA / 'predictions_cache.json'
    with open(path) as f:
        return json.load(f)


def load_dataset_stats() -> dict:
    """Load pre-computed dataset statistics (total records, class counts, etc.).
    Used as a fallback when ptbxl_database.csv is not available (cloud deployment).
    Returns dict with keys: total_records, unique_patients, class_counts."""
    path = APP_DATA / 'dataset_stats.json'
    with open(path) as f:
        return json.load(f)


def load_narratives_cache() -> dict:
    """Load pre-generated Qwen2 narratives keyed by ecg_id string.
    Returns empty dict if cache not built yet (run app/data/cache_narratives.py)."""
    path = APP_DATA / 'narratives_cache.json'
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)
