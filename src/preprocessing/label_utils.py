import pandas as pd
import ast
import numpy as np

# The 5 superclasses — order matters, this defines your output vector positions
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_label_mapping(scp_path):
    """
    Load scp_statements.csv and return only diagnostic entries.
    This gives us the mapping: SCP code → superclass
    e.g. 'AFIB' → 'NORM' is NOT diagnostic, 'MI' → 'MI' IS
    """
    agg = pd.read_csv(scp_path, index_col=0)
    agg = agg[agg.diagnostic == 1]  # keep only diagnostic labels
    return agg


def get_superclasses(scp_codes_dict, agg_df):
    """
    Convert one record's scp_codes dict → list of superclass strings.

    scp_codes_dict: e.g. {'NORM': 100, 'NDT': 0}  (from ptbxl_database.csv)
    Returns:        e.g. ['NORM']
    """
    result = []
    for code in scp_codes_dict.keys():
        if code in agg_df.index:
            superclass = agg_df.loc[code, 'diagnostic_class']
            if superclass in SUPERCLASSES and superclass not in result:
                result.append(superclass)
    return result


def to_multihot(superclass_list):
    """
    Convert list of superclass strings → multi-hot numpy vector.
    ['MI', 'CD'] → [0, 1, 0, 1, 0]
    This is what your model's output layer predicts.
    """
    vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for sc in superclass_list:
        if sc in SUPERCLASSES:
            vec[SUPERCLASSES.index(sc)] = 1.0
    return vec


def load_all_labels(db_path, scp_path):
    """
    Convenience function — loads metadata CSV and returns a dataframe
    with an added 'superclass' column (list) and 'label_vec' column (np array).
    """
    # Load metadata
    Y = pd.read_csv(db_path, index_col='ecg_id')
    Y['scp_codes'] = Y['scp_codes'].apply(ast.literal_eval)

    # Load label mapping
    agg = load_label_mapping(scp_path)

    # Map each record to superclasses
    Y['superclass'] = Y['scp_codes'].apply(lambda x: get_superclasses(x, agg))

    # Drop records with no diagnostic label (can't train on them)
    Y = Y[Y['superclass'].map(len) > 0].copy()

    # Create multi-hot vectors — list comprehension avoids pandas expanding
    # uniform-length arrays into a multi-column DataFrame via .apply()
    Y['label_vec'] = [to_multihot(sc) for sc in Y['superclass']]

    print(f"Records with valid labels: {len(Y)}")
    print(f"Class distribution:")
    for i, sc in enumerate(SUPERCLASSES):
        count = Y['label_vec'].apply(lambda v: v[i]).sum()
        print(f"  {sc}: {int(count)} ({100 * count / len(Y):.1f}%)")

    return Y