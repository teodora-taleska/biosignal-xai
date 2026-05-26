"""
ECG-PT — GPT-2 based model adapted for PTB-XL 5-class multi-label classification.

Original pretraining: self-supervised reconstruction on cardiac time-series.
Adapted here for supervised classification by swapping the causal-LM head for
a sequence-classification head (last token hidden state → linear → 5 logits).

HuggingFace: Tconnector/ecg-pt (falls back to GPT-2 if checkpoint unavailable)
Reference: https://huggingface.co/Tconnector/ecg-pt
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

from src.evaluation.metrics import compute_auc, compute_probs
from src.utils.profiler import ExperimentProfiler


class _ECGPatchTokenizer:
    """Split a 1-D ECG into fixed-size patches and quantise to token IDs."""

    def __init__(self, patch_size: int = 36, vocab_size: int = 256):
        self.patch_size = patch_size
        self.vocab_size = vocab_size

    def tokenize_batch(self, X: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (N, seq_len) float32

        Returns
        -------
        torch.Tensor, shape (N, seq_len // patch_size), dtype=long
        """
        results = []
        for sig in X:
            n    = len(sig) // self.patch_size
            pats = sig[: n * self.patch_size].reshape(n, -1)
            lo, hi = pats.min(), pats.max()
            norm = (pats.mean(axis=1) - lo) / (hi - lo + 1e-8)
            ids  = (norm * (self.vocab_size - 1)).astype(int).clip(0, self.vocab_size - 1)
            results.append(ids)
        return torch.tensor(np.array(results), dtype=torch.long)


class ECGPTClassifier:
    """
    ECG-PT fine-tuned for PTB-XL 5-class multi-label classification.

    Input  : X shape (N, 1000) — single-lead (Lead II) raw ECG signals
    Output : sigmoid probabilities shape (N, 5)

    Usage
    -----
    model = ECGPTClassifier()
    model.load()
    model.apply_peft(use_dora=False)
    best_auc, history = model.fit(X_train, y_train, X_val, y_val,
                                  experiment_name="ecgpt_lora_r8")
    probs = model.predict(X_val)
    """

    HF_ID = "Tconnector/ecg-pt"

    def __init__(self, num_labels: int = 5, patch_size: int = 36):
        self.num_labels = num_labels
        self.patch_size = patch_size
        self.model      = None
        self._tokenizer = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self):
        """Download ECG-PT (falls back to GPT-2 if checkpoint is not public)."""
        try:
            AutoConfig.from_pretrained(self.HF_ID)
            base_id = self.HF_ID
            print(f"Loading ECG-PT from {self.HF_ID} ...")
        except Exception:
            base_id = "gpt2"
            print(f"{self.HF_ID} not available — loading GPT-2 (same architecture).")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_id,
            num_labels   = self.num_labels,
            problem_type = "multi_label_classification",
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self._tokenizer = _ECGPatchTokenizer(
            patch_size = self.patch_size,
            vocab_size = self.model.config.vocab_size,
        )
        self.model.to(self.device)
        print("  Done.")
        return self

    # ── PEFT ──────────────────────────────────────────────────────────────────

    def apply_peft(
        self,
        r: int         = 8,
        alpha: int     = 16,
        dropout: float = 0.05,
        use_dora: bool = False,
    ):
        """Attach LoRA (use_dora=False) or DoRA (use_dora=True) adapters."""
        assert self.model is not None, "Call .load() first."
        cfg = LoraConfig(
            task_type      = TaskType.SEQ_CLS,
            r              = r,
            lora_alpha     = alpha,
            lora_dropout   = dropout,
            target_modules = ["c_attn"],
            bias           = "none",
            use_dora       = use_dora,
        )
        self.model = get_peft_model(self.model, cfg)
        self.model.print_trainable_parameters()
        return self

    # ── Parameters ───────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total":      total,
            "trainable":  trainable,
            "percentage": f"{100 * trainable / total:.1f}%",
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        experiment_name: str = "ecgpt",
        epochs: int          = 15,
        lr: float            = 1e-4,
        batch_size: int      = 16,
        save_dir: str        = "results/",
    ):
        """
        Fine-tune on PTB-XL.

        Parameters
        ----------
        X_train / X_val : (N, 1000) float32 — single-lead ECG (Lead II)
        y_train / y_val : (N, 5)    float32 — multi-hot labels
        """
        assert self.model is not None, "Call .load() first."

        save_path = os.path.join(save_dir, experiment_name)
        os.makedirs(save_path, exist_ok=True)

        print("  Tokenising signals...")
        ids_train = self._tokenizer.tokenize_batch(X_train)
        ids_val   = self._tokenizer.tokenize_batch(X_val)

        train_loader = DataLoader(
            TensorDataset(ids_train, torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(ids_val, torch.tensor(y_val, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=0.01,
        )

        p        = self.count_parameters()
        profiler = ExperimentProfiler(experiment_name)
        profiler.trainable_params = p["trainable"]
        profiler.total_params     = p["total"]

        print(f"\n{'='*60}")
        print(f" Experiment : {experiment_name}")
        print(f" Device     : {self.device}")
        print(f" Trainable  : {p['trainable']:,}  ({p['percentage']})")
        print(f"{'='*60}\n")

        profiler.start()
        best_auc, history = 0.0, []

        for epoch in range(epochs):
            profiler.start_epoch()
            self.model.train()
            total_loss = 0.0

            for token_ids, labels in train_loader:
                token_ids, labels = token_ids.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self.model(input_ids=token_ids, labels=labels).loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            val_logits_list, val_labels_list = [], []
            val_loss = 0.0
            with torch.no_grad():
                for token_ids, labels in val_loader:
                    token_ids, labels = token_ids.to(self.device), labels.to(self.device)
                    out = self.model(input_ids=token_ids, labels=labels)
                    val_loss += out.loss.item()
                    val_logits_list.append(out.logits.cpu())
                    val_labels_list.append(labels.cpu())

            probs     = compute_probs(torch.cat(val_logits_list))
            auc_res   = compute_auc(probs, torch.cat(val_labels_list).numpy())
            auc_macro = auc_res["auc_macro"]
            avg_train = total_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)

            print(f"Epoch {epoch+1:02d}/{epochs}  "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  AUC={auc_macro:.4f}")
            history.append({
                "epoch":      epoch + 1,
                "train_loss": avg_train,
                "val_loss":   avg_val,
                "auc_macro":  auc_macro,
            })

            if auc_macro > best_auc:
                best_auc = auc_macro
                self.save(os.path.join(save_path, "best_adapter"))
                print(f"  * Best saved — AUC {best_auc:.4f}")

            profiler.end_epoch()

        profiler.end()
        profiler.log_checkpoint_size(os.path.join(save_path, "best_adapter"))
        profiler.print_summary()
        profiler.save(save_path)

        with open(os.path.join(save_path, "history.json"), "w") as f:
            json.dump({"experiment": experiment_name, "history": history}, f, indent=2)

        print(f"\n{'='*60}")
        print(f" Done: {experiment_name}  Best AUC: {best_auc:.4f}")
        print(f"{'='*60}\n")

        return best_auc, history

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities, shape (N, 5)."""
        assert self.model is not None, "Call .load() first."
        token_ids = self._tokenizer.tokenize_batch(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(input_ids=token_ids).logits).cpu()
        return probs.numpy()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save PEFT adapter weights."""
        self.model.save_pretrained(path)
