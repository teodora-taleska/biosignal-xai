from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from src.utils.config import CFG

SUPERCLASSES = CFG['data']['superclasses']


class DummyECGClassifier:
    """
    Multi-label dummy classifier predicting each class by its training frequency.

    Metrics dict format matches src/utils/metrics.py so results are directly comparable.
    """

    def __init__(self):
        self._strategy   = CFG['baseline']['dummy_strategy']
        self._threshold  = CFG['inference']['threshold']
        self._batch_size = CFG['training']['batch_size']
        self._results_dir = CFG['paths']['results']
        self._model   = MultiOutputClassifier(DummyClassifier(strategy=self._strategy))
        self._fitted  = False

    def _collect_labels(self, dataset) -> np.ndarray:
        # Read labels directly from the dataset dataframe — no signal I/O.
        # ECGDataset (windowed): index is a list of (record_idx, window_idx) tuples;
        # all windows of a record share one label, so index into df by record_idx.
        # ECGDatasetFull: one row per sample, so just stack df['label_vec'] directly.
        if hasattr(dataset, 'index'):
            return np.vstack([dataset.df.iloc[r]['label_vec'] for r, _ in dataset.index])
        return np.vstack(dataset.df['label_vec'].values)

    def fit(self, dataset) -> DummyECGClassifier:
        t_start = time.time()

        labels  = self._collect_labels(dataset)
        X_dummy = np.zeros((len(labels), 1))
        self._model.fit(X_dummy, labels)
        self._fitted = True

        print(f"Dummy classifier fitted (strategy='{self._strategy}') on class frequencies:")
        for i, cls in enumerate(SUPERCLASSES):
            freq = labels[:, i].mean()
            print(f"  {cls:5s}: {freq:.3f} ({freq * 100:.1f}%)")

        elapsed = round(time.time() - t_start, 2)
        dummy_profiling = {
            "experiment":         "dummy_classifier",
            "total_time_sec":     elapsed,
            "total_time_human":   f"{elapsed}s",
            "avg_epoch_time_sec": elapsed,
            "epoch_times_sec":    [elapsed],
            "peak_gpu_memory_mb": 0.0,
            "peak_gpu_memory_gb": 0.0,
            "trainable_params":   0,
            "total_params":       0,
            "trainable_pct":      0.0,
            "checkpoint_size_mb": 0.0,
        }
        save_dir = os.path.join(CFG['paths']['results'], 'dummy_classifier')
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'profiling.json'), 'w') as f:
            json.dump(dummy_profiling, f, indent=2)

        return self

    def evaluate(self, dataset) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate()")

        labels  = self._collect_labels(dataset)
        X_dummy = np.zeros((len(labels), 1))

        # predict_proba returns list of (N, 2) arrays, one per output class.
        # Column 1 is P(positive class).
        proba_list = self._model.predict_proba(X_dummy)
        probs = np.column_stack([p[:, 1] for p in proba_list])  # (N, n_classes)
        preds = (probs >= self._threshold).astype(int)

        try:
            auc = roc_auc_score(labels, probs, average='macro')
        except ValueError:
            auc = 0.0

        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        per_class = {}
        for i, cls in enumerate(SUPERCLASSES):
            try:
                per_class[cls] = roc_auc_score(labels[:, i], probs[:, i])
            except ValueError:
                per_class[cls] = 0.0

        return {
            'auc_macro':     auc,
            'f1_macro':      f1,
            'per_class_auc': per_class,
        }

    def save_results(self, metrics: dict, filename: str) -> None:
        os.makedirs(self._results_dir, exist_ok=True)
        path = os.path.join(self._results_dir, filename)
        output = {
            'model':         'DummyClassifier',
            'strategy':      self._strategy,
            'auc_macro':     round(metrics['auc_macro'], 4),
            'f1_macro':      round(metrics['f1_macro'],  4),
            'per_class_auc': {k: round(v, 4) for k, v in metrics['per_class_auc'].items()},
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved -> {path}")
