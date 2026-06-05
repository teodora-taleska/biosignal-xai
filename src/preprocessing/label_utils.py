import ast

import numpy as np
import pandas as pd

from src.utils.config import CFG

# The five PTB-XL diagnostic superclasses — order matches the label vector
SUPERCLASSES = CFG['data']['superclasses']  # ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# Per-class positive weights for BCEWithLogitsLoss, proportional to the
# negative-to-positive ratio in the PTB-XL training set (folds 1-8).
# NORM: 44.5 %  MI: 25.6 %  STTC: 23.9 %  CD: 22.9 %  HYP: 12.4 %
import torch as _torch
DEFAULT_POS_WEIGHT = _torch.tensor([1.0, 1.74, 1.82, 1.94, 3.59])

# Map from SCP-ECG diagnostic_class strings (as they appear in scp_statements.csv)
# to our canonical superclass abbreviations.
_DIAG_CLASS_MAP = {
    'NORM': 'NORM',
    'MI':   'MI',
    'STTC': 'STTC',
    'CD':   'CD',
    'HYP':  'HYP',
}


def _scp_to_superclasses(scp_codes: dict, scp_df: pd.DataFrame) -> list:
    """
    Convert a record's raw SCP code dict to a list of superclass labels.

    scp_codes: e.g. {'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}
    scp_df:    scp_statements.csv loaded as a DataFrame, indexed by SCP code string.

    Only codes with likelihood > 0 and a known diagnostic_class are considered.
    Returns a list like ['NORM'] or ['MI', 'CD'].
    """
    superclasses = set()
    for code, likelihood in scp_codes.items():
        if likelihood == 0:
            continue
        if code not in scp_df.index:
            continue
        diag_class = scp_df.loc[code, 'diagnostic_class']
        if pd.isna(diag_class):
            continue
        mapped = _DIAG_CLASS_MAP.get(str(diag_class).strip())
        if mapped:
            superclasses.add(mapped)
    return sorted(superclasses)


def _to_label_vec(superclasses: list) -> list:
    """Convert a list of superclass strings to a multi-hot float list (length 5)."""
    vec = [0.0] * len(SUPERCLASSES)
    for sc in superclasses:
        if sc in SUPERCLASSES:
            vec[SUPERCLASSES.index(sc)] = 1.0
    return vec


def load_all_labels(db_path: str, scp_path: str) -> pd.DataFrame:
    """
    Load PTB-XL metadata and map SCP codes to 5-superclass multi-hot labels.

    Parameters
    ----------
    db_path  : path to ptbxl_database.csv
    scp_path : path to scp_statements.csv

    Returns
    -------
    DataFrame indexed by ecg_id, with columns:
        patient_id, age, sex, filename_lr, filename_hr, strat_fold,
        scp_codes  (dict),
        superclass (list of matched superclass strings),
        label_vec  (list of 5 floats, multi-hot).

    Records with no matched superclass are dropped.
    Prints a brief class-distribution summary.
    """
    db = pd.read_csv(db_path, index_col='ecg_id')
    db['scp_codes'] = db['scp_codes'].apply(ast.literal_eval)

    scp = pd.read_csv(scp_path, index_col=0)

    db['superclass'] = db['scp_codes'].apply(
        lambda codes: _scp_to_superclasses(codes, scp)
    )

    # Drop records with no diagnostic superclass label
    valid = db['superclass'].apply(len) > 0
    db = db[valid].copy()

    db['label_vec'] = db['superclass'].apply(_to_label_vec)

    print(f"Records with valid labels: {len(db)}")
    print("Class distribution:")
    label_matrix = np.stack(db['label_vec'].values)
    for i, sc in enumerate(SUPERCLASSES):
        count = int(label_matrix[:, i].sum())
        pct   = 100 * count / len(db)
        print(f"  {sc}: {count} ({pct:.1f}%)")

    keep_cols = [
        'patient_id', 'age', 'sex',
        'filename_lr', 'filename_hr',
        'strat_fold', 'scp_codes', 'superclass', 'label_vec',
    ]
    keep_cols = [c for c in keep_cols if c in db.columns]
    return db[keep_cols]
