"""
HeartBERT — RoBERTa pretrained on ECG-as-text (Bayesiano/HeartBERT).

Reference: https://huggingface.co/Bayesiano/HeartBERT
Adapted for PTB-XL 5-class multi-label classification with PEFT (LoRA / DoRA).
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from src.evaluation.metrics import compute_auc, compute_probs
from src.utils.profiler import ExperimentProfiler


def _ecg_to_text(signal: np.ndarray, n_bins: int = 20) -> str:
    """Quantise a 1-D ECG waveform into a letter string."""
    bins    = np.linspace(signal.min(), signal.max(), n_bins)
    indices = np.digitize(signal, bins).clip(0, n_bins - 1)
    letters = "ABCDEFGHIJKLMNOPQRST"
    return " ".join(letters[i] for i in indices)


class HeartBERTClassifier:
    """
    HeartBERT fine-tuned for PTB-XL 5-class multi-label classification.

    Input  : X shape (N, 1000) — single-lead (Lead II) raw ECG signals
    Output : sigmoid probabilities shape (N, 5)

    Usage
    -----
    model = HeartBERTClassifier()
    model.load()
    model.apply_peft(use_dora=False)   # LoRA
    best_auc, history = model.fit(X_train, y_train, X_val, y_val,
                                  experiment_name="heartbert_lora_r8")
    probs = model.predict(X_val)
    model.save("results/heartbert_lora_r8/best_adapter")
    """

    HF_ID = "Bayesiano/HeartBERT"

    def __init__(self, num_labels: int = 5):
        self.num_labels = num_labels
        self.tokenizer  = None
        self.model      = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self):
        """Download pretrained weights from HuggingFace."""
        print(f"Loading HeartBERT from {self.HF_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.HF_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.HF_ID,
            num_labels              = self.num_labels,
            problem_type            = "multi_label_classification",
            ignore_mismatched_sizes = True,
        ).to(self.device)
        print("  Done.")
        return self

    # ── PEFT ─────────────────────────────────────────────────────────────────

    def apply_peft(
        self,
        r: int         = 8,
        alpha: int     = 16,
        dropout: float = 0.1,
        use_dora: bool = False,
    ):
        """Attach LoRA (use_dora=False) or DoRA (use_dora=True) adapters."""
        assert self.model is not None, "Call .load() first."
        cfg = LoraConfig(
            task_type      = TaskType.SEQ_CLS,
            r              = r,
            lora_alpha     = alpha,
            lora_dropout   = dropout,
            target_modules = ["query", "value"],
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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode(self, X: np.ndarray) -> dict:
        texts = [_ecg_to_text(x) for x in X]
        return self.tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = 512,
            return_tensors = "pt",
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        experiment_name: str = "heartbert",
        epochs: int          = 15,
        lr: float            = 2e-4,
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

        print("  Tokenising signals (this takes ~1 min for 17k records)...")
        enc_train = self._encode(X_train)
        enc_val   = self._encode(X_val)

        train_loader = DataLoader(
            TensorDataset(
                enc_train["input_ids"],
                enc_train["attention_mask"],
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                enc_val["input_ids"],
                enc_val["attention_mask"],
                torch.tensor(y_val, dtype=torch.float32),
            ),
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

            for ids, mask, labels in train_loader:
                ids, mask, labels = (
                    ids.to(self.device), mask.to(self.device), labels.to(self.device)
                )
                optimizer.zero_grad()
                loss = self.model(input_ids=ids, attention_mask=mask, labels=labels).loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            val_logits_list, val_labels_list = [], []
            val_loss = 0.0
            with torch.no_grad():
                for ids, mask, labels in val_loader:
                    ids, mask, labels = (
                        ids.to(self.device), mask.to(self.device), labels.to(self.device)
                    )
                    out = self.model(input_ids=ids, attention_mask=mask, labels=labels)
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
        enc = {k: v.to(self.device) for k, v in self._encode(X).items()}
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(**enc).logits).cpu()
        return probs.numpy()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save PEFT adapter weights."""
        self.model.save_pretrained(path)
