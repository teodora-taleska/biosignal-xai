from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from src.utils.config import CFG

SUPERCLASSES = CFG['data']['superclasses']


class ECGInferencePipeline:
    """
    End-to-end inference pipeline for ECG anomaly detection.

    Takes preprocessed ECG signal -> returns structured prediction with
    class probabilities, confidence score, and optional uncertainty.

    Used by:
      - notebooks/07_uncertainty.ipynb
      - app/backend/inference/ (product layer)
    """

    def __init__(
        self,
        model,
        device=None,
        has_uncertainty: bool = True,
    ):
        """
        Args:
            model:           trained AleatoricWrapper or base model
            device:          torch device (auto-detected if None)
            has_uncertainty: True if model returns (mean, log_var)
                             False if model returns logits only
        """
        self.model           = model
        self.has_uncertainty = has_uncertainty
        self.device          = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.model.eval()
        self.threshold = CFG['inference']['threshold']

    def predict(self, signal: np.ndarray) -> dict:
        """
        Run inference on one preprocessed ECG record.

        Args:
            signal: (1000, 12) or (12, 1000) float32 numpy array.
                    Pipeline normalises the shape to channels-first (12, 1000).

        Returns dict with keys:
            predicted_classes:   list of class names above threshold
            class_probabilities: dict {class: probability}
            confidence_score:    float -- max probability across classes
            uncertainty:         float or None -- mean aleatoric uncertainty
            raw_logits:          list -- raw mean_head output
            uncertainty_level:   human-readable quality label
        """
        if signal.shape == (1000, 12):
            signal = signal.T  # -> (12, 1000)

        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.has_uncertainty:
                mean, log_var = self.model(x)
                uncertainty: Optional[float] = torch.exp(log_var).mean().item()
            else:
                mean        = self.model(x)
                uncertainty = None

        probs = torch.sigmoid(mean).squeeze(0).cpu()  # (5,)

        predicted = [
            SUPERCLASSES[i]
            for i, p in enumerate(probs)
            if p.item() >= self.threshold
        ]

        # Fall back to argmax if nothing clears the threshold
        if not predicted:
            predicted = [SUPERCLASSES[probs.argmax().item()]]

        return {
            'predicted_classes':   predicted,
            'class_probabilities': {
                cls: round(probs[i].item(), 4)
                for i, cls in enumerate(SUPERCLASSES)
            },
            'confidence_score':    round(probs.max().item(), 4),
            'uncertainty':         round(uncertainty, 4) if uncertainty is not None else None,
            'raw_logits':          mean.squeeze(0).cpu().tolist(),
            'uncertainty_level':   self._uncertainty_level(uncertainty),
        }

    def predict_batch(self, signals: list) -> list:
        """Run predict() on a list of signals. Returns list of result dicts."""
        return [self.predict(s) for s in signals]

    @staticmethod
    def _uncertainty_level(uncertainty: Optional[float]) -> str:
        """Human-readable uncertainty label based on CFG threshold."""
        if uncertainty is None:
            return 'not computed'
        thresh = CFG['explainability']['uncertainty_threshold']
        if uncertainty < thresh * 0.5:
            return 'low -- signal was clean, result is trustworthy'
        elif uncertainty < thresh:
            return 'moderate -- some signal noise detected'
        else:
            return 'high -- noisy signal, treat result with caution'


class HeartBERTPipeline:
    """
    Inference pipeline for HeartBERT LoRA on PTB-XL 5-class classification.

    Wraps a loaded HeartBERTClassifier and exposes the same predict() dict
    interface as ECGInferencePipeline. Latency benchmark includes tokenisation.

    Usage
    -----
    pipeline = HeartBERTPipeline(loaded_heartbert_classifier)
    result   = pipeline.predict(signal_np)          # (12, 1000) or (1000,)
    report   = pipeline.benchmark(signals, n_warmup=5, n_runs=50)
    """

    def __init__(self, classifier):
        """
        Args:
            classifier: loaded HeartBERTClassifier (after .load() and .load_adapter())
        """
        self.classifier = classifier
        self.threshold  = CFG['inference']['threshold']

    def predict(self, signal: np.ndarray) -> dict:
        """
        Run inference on a single ECG record.

        Args:
            signal: (12, 1000), (1000, 12), or (1000,) float32 numpy array.
                    (1000, 12) is transposed to (12, 1000) automatically.
                    (12, 1000) -- Lead II (row index 1) is extracted.
                    (1000,)    -- used directly as Lead II.

        Returns dict with keys:
            predicted_classes, class_probabilities, confidence_score,
            uncertainty (always None), raw_logits, uncertainty_level
        """
        if signal.shape == (1000, 12):
            signal = signal.T                          # -> (12, 1000)
        if signal.ndim == 2:
            signal = signal[1]                         # Lead II, shape (1000,)

        logits = self.classifier.predict_logits(signal[np.newaxis])    # (1, 5)
        probs  = torch.sigmoid(torch.tensor(logits[0]))                # (5,)

        thresholds = getattr(self, 'thresholds', None) or [self.threshold] * len(SUPERCLASSES)
        predicted = [
            SUPERCLASSES[i]
            for i, p in enumerate(probs)
            if p.item() >= thresholds[i]
        ]
        if not predicted:
            predicted = [SUPERCLASSES[probs.argmax().item()]]

        return {
            'predicted_classes':   predicted,
            'class_probabilities': {
                c: round(probs[i].item(), 4)
                for i, c in enumerate(SUPERCLASSES)
            },
            'confidence_score':    round(probs.max().item(), 4),
            'uncertainty':         None,
            'raw_logits':          logits[0].tolist(),
            'uncertainty_level':   'not computed',
        }

    def benchmark(
        self,
        signals: list,
        n_warmup: int = 5,
        n_runs:   int = 50,
    ) -> dict:
        """
        Measure end-to-end wall-clock latency per record (tokenisation + inference).

        Args:
            signals:  list of numpy arrays accepted by predict()
            n_warmup: warm-up passes not included in measurements
            n_runs:   measured passes

        Returns dict with keys:
            n_runs, device, latency_ms_mean, latency_ms_median,
            latency_ms_p95, throughput_per_sec
        """
        import time
        n = len(signals)

        for i in range(n_warmup):
            self.predict(signals[i % n])

        times = []
        for i in range(n_runs):
            sig = signals[i % n]
            t0  = time.perf_counter()
            self.predict(sig)
            times.append((time.perf_counter() - t0) * 1000)    # ms

        times  = np.array(times)
        median = float(np.median(times))
        return {
            'n_runs':             n_runs,
            'device':             str(self.classifier.device),
            'latency_ms_mean':    round(float(np.mean(times)),            3),
            'latency_ms_median':  round(median,                           3),
            'latency_ms_p95':     round(float(np.percentile(times, 95)), 3),
            'throughput_per_sec': round(1000 / median,                    1),
        }
