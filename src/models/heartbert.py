"""
HeartBERT — RoBERTa pretrained on ECG-as-text (Bayesiano/HeartBERT).

Reference: https://huggingface.co/Bayesiano/HeartBERT
Weights hosted on Google Drive (folder ID: 10flbRia9rDWeS8-TLScRUT6JBv81iN-4).
Adapted for PTB-XL 5-class multi-label classification with PEFT (LoRA / DoRA).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from src.evaluation.metrics import compute_auc, compute_probs
from src.utils.profiler import ExperimentProfiler


def _ecg_to_text(signal: np.ndarray, n_bins: int = 32) -> str:
    """Quantise a 1-D ECG waveform into a letter string (supports up to 52 bins)."""
    bins    = np.linspace(signal.min(), signal.max(), n_bins)
    indices = np.digitize(signal, bins).clip(0, n_bins - 1)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
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

    HF_ID      = "Bayesiano/HeartBERT"
    GDRIVE_ID  = "10flbRia9rDWeS8-TLScRUT6JBv81iN-4"
    CACHE_DIR  = Path.home() / ".cache" / "heartbert"

    def __init__(self, num_labels: int = 5):
        self.num_labels = num_labels
        self.tokenizer  = None
        self.model      = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def _ensure_local_weights(cls) -> Path:
        """
        Return a local directory containing the HeartBERT checkpoint.

        Priority:
          1. ~/.cache/heartbert  — already downloaded on a previous run
          2. Google Drive folder (ID: 10flbRia9rDWeS8-TLScRUT6JBv81iN-4)
          3. HuggingFace Hub (Bayesiano/HeartBERT) — if public
        """
        marker = cls.CACHE_DIR / "pytorch_model.bin"
        if marker.exists():
            print(f"HeartBERT weights found in cache: {cls.CACHE_DIR}")
            return cls.CACHE_DIR

        # Try Google Drive first (known-good source)
        try:
            import gdown
            print(f"Downloading HeartBERT from Google Drive → {cls.CACHE_DIR} ...")
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            gdown.download_folder(
                id     = cls.GDRIVE_ID,
                output = str(cls.CACHE_DIR),
                quiet  = False,
            )
            if marker.exists():
                print("  Download complete.")
                return cls.CACHE_DIR
            print("  WARNING: download finished but pytorch_model.bin not found.")
        except Exception as e:
            print(f"  Google Drive download failed: {e}")

        # Fall back to HuggingFace Hub
        try:
            AutoTokenizer.from_pretrained(cls.HF_ID, trust_remote_code=True)
            print(f"Loading HeartBERT from HuggingFace ({cls.HF_ID}) ...")
            return cls.HF_ID
        except Exception as e:
            print(f"  HuggingFace load also failed: {e}")

        return None

    def load(self):
        """Load HeartBERT weights (Google Drive cache → HF Hub → roberta-base fallback)."""
        source = self._ensure_local_weights()

        if source is not None:
            base_id = str(source)
            print(f"Loading HeartBERT from {base_id} ...")
        else:
            base_id = "roberta-base"
            print(
                "WARNING: HeartBERT weights could not be downloaded from Google Drive "
                "or HuggingFace.\n"
                "  Falling back to roberta-base (same architecture, general-language "
                "weights — NOT ECG-pretrained)."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(base_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_id,
            num_labels              = self.num_labels,
            problem_type            = "multi_label_classification",
            ignore_mismatched_sizes = True,
            attn_implementation     = "eager",
        ).to(self.device)
        print("  Done.")
        return self

    # ── PEFT ─────────────────────────────────────────────────────────────────

    def apply_peft(
        self,
        r: int         = 16,
        alpha: int     = 32,
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
        epochs: int          = 25,
        lr: float            = 2e-4,
        batch_size: int      = 16,
        patience: int        = 2,
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
        epochs_no_improve = 0

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
                epochs_no_improve = 0
                self.save(os.path.join(save_path, "best_adapter"))
                print(f"  * Best saved — AUC {best_auc:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping (no AUC improvement for {patience} epochs)")
                    profiler.end_epoch()
                    break

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

    # ── Adapter loading ───────────────────────────────────────────────────────

    def load_adapter(self, path: str):
        """Load a saved PEFT adapter into the already-loaded base model."""
        from peft import PeftModel
        assert self.model is not None, "Call .load() first."
        self.model = PeftModel.from_pretrained(self.model, path)
        self.model.eval()
        return self

    def predict_logits(self, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Return raw logits (before sigmoid), shape (N, 5).

        Runs inference in mini-batches to avoid OOM on large test sets.
        """
        assert self.model is not None, "Call .load() first."
        self.model.eval()
        all_logits = []
        for i in range(0, len(X), batch_size):
            chunk = X[i:i + batch_size]
            enc = {k: v.to(self.device) for k, v in self._encode(chunk).items()}
            with torch.no_grad():
                all_logits.append(self.model(**enc).logits.cpu())
        return torch.cat(all_logits).numpy()

    def get_attention_weights(self, x: np.ndarray) -> tuple:
        """
        Return last-layer CLS attention weights over ECG letter tokens.

        The RoBERTa tokenizer produces [CLS] + up to 510 letter tokens + [SEP]
        = 512 tokens total (max_length=512). This method returns the attention
        that the CLS token pays to each letter token, normalised to [0, 1].

        Args:
            x: (1000,) float32 Lead II signal for a single record

        Returns:
            positions: (N,) int numpy array -- Lead II sample indices (0-based)
            weights:   (N,) float32 numpy array -- normalised attention weights
        """
        assert self.model is not None, "Call .load() first."
        enc = {k: v.to(self.device) for k, v in self._encode(x[np.newaxis]).items()}
        self.model.eval()
        with torch.no_grad():
            out = self.model(**enc, output_attentions=True)
        # out.attentions: tuple of (1, num_heads, seq_len, seq_len), one per layer
        last_layer = out.attentions[-1]            # (1, num_heads, seq_len, seq_len)
        avg_heads  = last_layer[0].mean(dim=0)     # (seq_len, seq_len)
        cls_attn   = avg_heads[0].cpu().numpy()    # (seq_len,) -- CLS row
        # Drop CLS (index 0) and SEP (index -1); keep ECG letter tokens
        ecg_attn  = cls_attn[1:-1].astype(np.float32)
        ecg_attn  = ecg_attn / (ecg_attn.max() + 1e-8)   # normalise to [0, 1]
        positions = np.arange(len(ecg_attn), dtype=np.int32)
        return positions, ecg_attn
