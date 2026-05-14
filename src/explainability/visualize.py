from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_DEFAULT_LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                       'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
_CLASS_COLORS = ['steelblue', 'tomato', 'darkorange', 'mediumseagreen', 'mediumpurple']


def plot_ecg_with_saliency(
    signal_np: np.ndarray,
    saliency:  np.ndarray,
    title:     str,
    pred_info: dict,
    lead_names: list[str] | None = None,
    save_path:  str | None = None,
) -> None:
    """
    3x4 grid of 12-lead ECG with per-lead saliency shown as a red fill
    on a twin axis behind each signal trace.

    Args:
        signal_np:  (12, 1000) ECG signal (channels-first)
        saliency:   (12, 1000) saliency array from compute_saliency()
        title:      figure suptitle string (e.g. case label)
        pred_info:  dict with keys true_cls, pred_cls, confidence, uncertainty
        lead_names: optional override for the 12 lead labels
        save_path:  if given, figure is saved here at dpi=120
    """
    if lead_names is None:
        lead_names = _DEFAULT_LEAD_NAMES

    t   = np.linspace(0, 10, signal_np.shape[1])
    sal = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True)
    axes = axes.flatten()

    for i, lead in enumerate(lead_names):
        ax  = axes[i]
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, sal[i], color='red', alpha=0.25)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax.plot(t, signal_np[i], color='black', linewidth=0.7)
        ax.set_title(lead, fontsize=9)
        ax.set_ylabel('mV', fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    for ax in axes[len(lead_names) - 4:]:
        ax.set_xlabel('Time (s)', fontsize=8)

    true_str = ', '.join(pred_info.get('true_cls', []))  or 'unknown'
    pred_str = ', '.join(pred_info.get('pred_cls', []))
    fig.suptitle(
        f'{title}  |  True: {true_str}  Pred: {pred_str}  '
        f'Conf: {pred_info.get("confidence", 0):.2f}  '
        f'Unc: {pred_info.get("uncertainty", 0):.4f}',
        fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_progressive_inference(
    checkpoints_sec:      np.ndarray,
    checkpoint_probs:     np.ndarray,
    checkpoint_uncertainty: np.ndarray | None,
    superclasses:         list[str],
    true_cls:             list[str],
    threshold:            float = 0.5,
    class_colors:         list[str] | None = None,
    save_path:            str | None = None,
) -> None:
    """
    Visualise how transformer predictions evolve as more ECG signal arrives.

    Top subplot:    per-class probability lines vs seconds received
    Bottom subplot: aleatoric uncertainty vs seconds received (if available)

    Args:
        checkpoints_sec:        (n,) seconds of signal available at each step
        checkpoint_probs:       (n, n_classes) running probability at each step
        checkpoint_uncertainty: (n,) uncertainty at each step, or None
        superclasses:           list of class name strings
        true_cls:               ground-truth class list for the title
        threshold:              prediction threshold horizontal line
        class_colors:           optional list of matplotlib color strings
        save_path:              if given, figure is saved here
    """
    if class_colors is None:
        class_colors = _CLASS_COLORS

    n_rows = 2 if checkpoint_uncertainty is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # --- Class probability evolution ---
    ax = axes[0]
    for i, cls in enumerate(superclasses):
        ax.plot(checkpoints_sec, checkpoint_probs[:, i],
                marker='o', markersize=5,
                color=class_colors[i % len(class_colors)],
                linewidth=1.8, label=cls)
    ax.axhline(threshold, linestyle='--', color='gray', alpha=0.7, label='threshold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Class probability', fontsize=11)
    ax.set_title(
        f'Progressive Transformer Inference  |  True: {", ".join(true_cls)}',
        fontsize=12,
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Uncertainty evolution ---
    if checkpoint_uncertainty is not None:
        ax2 = axes[1]
        ax2.plot(checkpoints_sec, checkpoint_uncertainty,
                 color='black', linewidth=1.8, marker='s', markersize=5)
        ax2.set_ylabel('Aleatoric uncertainty', fontsize=11)
        ax2.set_xlabel('Seconds of signal received', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Uncertainty vs Signal Length', fontsize=12)
    else:
        axes[0].set_xlabel('Seconds of signal received', fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
