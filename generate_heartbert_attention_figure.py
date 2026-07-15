"""
Generate heartbert-attention-mi.png for the blog post.

Loads the trained HeartBERT LoRA adapter, picks the first clean MI record
from the test fold, runs get_attention_weights(), and saves a publication-ready
figure of the attention overlay on Lead II.

Usage (from repo root, with biosignal-xai conda env active):
    python generate_heartbert_attention_figure.py

Output:
    results/figures/heartbert-attention-mi.png
    (copy this to your portfolio's public/blog-images/ folder)
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import wfdb

from src.preprocessing.preprocess import bandpass_filter, normalize_signal
from src.utils.config import CFG

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH    = Path(CFG["data"]["path"].rstrip("/"))
ADAPTER_PATH = Path("results/08_heartbert_lora_r8_filtered_zscore/best_adapter")
OUT_DIR      = Path("results/figures")
OUT_PATH     = OUT_DIR / "heartbert-attention-mi.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load PTB-XL index and find an MI test record ──────────────────────────────

print("Loading PTB-XL index...")
db = pd.read_csv(DATA_PATH / "ptbxl_database.csv", index_col="ecg_id")
db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

scp = pd.read_csv(DATA_PATH / "scp_statements.csv", index_col=0)

def is_pure_mi(scp_codes):
    """True if the record has MI as its only superclass."""
    classes = set()
    for code, likelihood in scp_codes.items():
        if likelihood == 0 or code not in scp.index:
            continue
        dc = scp.loc[code, "diagnostic_class"]
        if pd.notna(dc):
            classes.add(str(dc).strip())
    return classes == {"MI"}

test_mi = db[(db["strat_fold"] == 10) & db["scp_codes"].apply(is_pure_mi)]
print(f"Found {len(test_mi)} pure-MI test records.")

# Use the first record
row = test_mi.iloc[0]
print(f"Using ECG ID {row.name}  filename: {row['filename_lr']}")

# ── Load and preprocess signal ────────────────────────────────────────────────

signal, _ = wfdb.rdsamp(str(DATA_PATH / row["filename_lr"]))   # (1000, 12)
signal     = bandpass_filter(signal)                            # bandpass 0.5–40 Hz
signal     = normalize_signal(signal)                           # per-lead z-score
lead_ii    = signal[:, 1].astype(np.float32)                   # (1000,)

# ── Load HeartBERT + adapter ──────────────────────────────────────────────────

print("Loading HeartBERT...")
from src.models.heartbert import HeartBERTClassifier

model = HeartBERTClassifier()
model.load()
model.load_adapter(str(ADAPTER_PATH))
print("  Model ready.")

# ── Compute prediction and attention weights ──────────────────────────────────

probs = model.predict(lead_ii[np.newaxis])          # (1, 5)
superclasses = ["NORM", "MI", "STTC", "CD", "HYP"]
pred_idx = int(probs[0].argmax())
pred_class = superclasses[pred_idx]
confidence = float(probs[0, pred_idx])
print(f"Prediction: {pred_class}  confidence: {confidence:.2f}")
print(f"All probs: { {sc: f'{p:.2f}' for sc, p in zip(superclasses, probs[0])} }")

positions, weights = model.get_attention_weights(lead_ii)  # (N,), (N,)

# positions covers the tokenised portion of the signal (up to 510 samples)
# Map back to time axis at 100 Hz
t_full   = np.arange(1000) / 100.0          # 0.00 … 9.99 s
t_attn   = positions / 100.0               # attention time axis

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 5),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
)

# -- Top panel: Lead II signal with attention fill ----------------------------

ax = axes[0]
ax.plot(t_full, lead_ii, color="#1a1a2e", linewidth=0.9, zorder=2)

# Shade attention region: alpha proportional to attention weight
# Interpolate weights to full 1000 samples for smooth fill
weights_full = np.zeros(1000)
weights_full[positions] = weights
# Smooth with a small window for visual clarity
from numpy.lib.stride_tricks import sliding_window_view
kernel = np.ones(15) / 15
weights_smooth = np.convolve(weights_full, kernel, mode="same")
weights_smooth = np.clip(weights_smooth / (weights_smooth.max() + 1e-8), 0, 1)

# Fill each sample column with attention-scaled red alpha
for i in range(0, 1000 - 1):
    if weights_smooth[i] > 0.05:
        ax.axvspan(
            t_full[i], t_full[i + 1],
            alpha=float(weights_smooth[i]) * 0.55,
            color="#c0392b",
            linewidth=0,
            zorder=1,
        )

ax.set_xlim(0, 9.99)
ax.set_ylabel("Lead II (z-scored)", fontsize=10)
ax.tick_params(labelbottom=False)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cccccc")
ax.tick_params(colors="#555555")
ax.yaxis.label.set_color("#333333")

# Annotation box
ax.text(
    0.98, 0.95,
    f"Predicted: {pred_class}  ·  Confidence: {confidence:.0%}\n"
    f"True label: MI  ·  ECG ID: {row.name}",
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=8.5, color="#333333",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#dddddd", alpha=0.9),
)

# -- Bottom panel: attention weight curve -------------------------------------

ax2 = axes[1]
ax2.fill_between(t_attn, weights, alpha=0.4, color="#c0392b")
ax2.plot(t_attn, weights, color="#c0392b", linewidth=0.8)
ax2.set_xlim(0, 9.99)
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Time (s)", fontsize=10)
ax2.set_ylabel("Attention", fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#cccccc")
ax2.tick_params(colors="#555555")
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.yaxis.label.set_color("#333333")
ax2.xaxis.label.set_color("#333333")

# Vertical line at tokenisation cutoff (~5.1 s)
cutoff = 510 / 100.0
for a in axes:
    a.axvline(cutoff, color="#aaaaaa", linewidth=0.7, linestyle="--")
axes[0].text(
    cutoff + 0.05, axes[0].get_ylim()[1] * 0.92,
    "tokenisation\ncutoff",
    fontsize=7, color="#888888", va="top",
)

fig.suptitle(
    "HeartBERT: CLS attention weights over Lead II  (MI record)",
    fontsize=11, color="#1a1a2e", y=0.98,
)

plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT_PATH}")
