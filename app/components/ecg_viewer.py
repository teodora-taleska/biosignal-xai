"""
ECG waveform viewer component.

Two rendering modes:
  1. Plain waveform  — overlaid leads on a single Plotly figure
  2. Saliency overlay — same waveform with a heatmap background showing
     gradient saliency (low→high = light blue → red), and a colorbar.

Usage:
    from app.components.ecg_viewer import render_ecg, render_ecg_with_saliency
    render_ecg(signal_np, lead_names)
    render_ecg_with_saliency(signal_np, saliency_np, lead_names, top_leads)
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

_BLUE   = '#1976d2'
_GREY   = '#90a4ae'
_FS     = 100  # Hz — PTB-XL 100 Hz records


def _time_axis(n_samples: int, fs: int = _FS) -> np.ndarray:
    return np.arange(n_samples) / fs  # seconds


def render_ecg(
    signal:     np.ndarray,
    lead_names: list[str] = LEAD_NAMES,
    title:      str       = 'ECG: 12 leads',
    height:     int       = 480,
    fs:         int       = _FS,
    key:        str       = 'ecg_viewer',
) -> None:
    """
    Render all 12 leads in a vertically stacked subplot grid (6 rows × 2 cols).

    Args:
        signal:     (1000, 12) or (12, 1000) float32 numpy array
        lead_names: list of 12 lead name strings
        title:      figure title
        height:     chart height in pixels
        fs:         sampling frequency (Hz)
    """
    # Normalise to (12, 1000) → then split to list of (1000,) per lead
    if signal.shape == (1000, 12):
        signal = signal.T
    n_leads, n_samples = signal.shape
    t = _time_axis(n_samples, fs)

    rows, cols = 6, 2
    subplot_titles = lead_names[:n_leads]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )

    for i, name in enumerate(lead_names[:n_leads]):
        row = i // cols + 1
        col = i  % cols + 1
        fig.add_trace(
            go.Scatter(
                x=t, y=signal[i],
                mode='lines',
                line=dict(color=_BLUE, width=1),
                name=name,
                showlegend=False,
                hovertemplate=f'{name}: %{{y:.3f}} mV  t=%{{x:.2f}}s<extra></extra>',
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=10, r=10, t=40, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0', title_text='s', title_font_size=10)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0', title_text='mV', title_font_size=10)

    st.plotly_chart(fig, use_container_width=True, key=key)


def render_ecg_with_saliency(
    signal:     np.ndarray,
    saliency:   np.ndarray,
    lead_names: list[str]       = LEAD_NAMES,
    top_leads:  list[str] | None = None,
    height:     int              = 520,
    fs:         int              = _FS,
    key:        str              = 'ecg_saliency',
) -> None:
    """
    Render ECG waveform with saliency heatmap overlay and colorbar.

    Top-salient leads are highlighted in blue; others in grey.
    A gradient colorbar (low→high = light blue → red) is shown.

    Args:
        signal:     (1000, 12) or (12, 1000) float32 array
        saliency:   (12, 1000) float32 saliency array from compute_saliency()
        lead_names: list of 12 lead name strings
        top_leads:  list of lead names to highlight; if None, all shown in blue
        height:     chart height in pixels
        fs:         sampling frequency
    """
    if signal.shape == (1000, 12):
        signal = signal.T          # (12, 1000)

    n_leads, n_samples = signal.shape
    t = _time_axis(n_samples, fs)
    top_set = set(top_leads or lead_names)

    rows, cols = 6, 2
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=lead_names[:n_leads],
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )

    # Shared saliency scale across all leads
    sal_min = saliency.min()
    sal_max = max(saliency.max(), sal_min + 1e-6)

    # Colorscale: white (low) → light-blue → red (high)
    colorscale = [
        [0.0,  '#e8f4fd'],
        [0.4,  '#90caf9'],
        [0.7,  '#ef9a9a'],
        [1.0,  '#b71c1c'],
    ]

    for i, name in enumerate(lead_names[:n_leads]):
        row = i // cols + 1
        col = i  % cols + 1
        sal_i = saliency[i]   # (1000,)

        # Heatmap background (1-pixel tall; y spans signal range)
        sig_lo = float(signal[i].min()) - 0.05
        sig_hi = float(signal[i].max()) + 0.05

        fig.add_trace(
            go.Heatmap(
                x=t,
                y=[sig_lo, sig_hi],
                z=[sal_i, sal_i],
                colorscale=colorscale,
                zmin=sal_min,
                zmax=sal_max,
                showscale=(i == 0),           # colorbar only on first lead
                colorbar=dict(
                    title=dict(text='Saliency', side='right'),
                    len=0.4, thickness=12,
                    y=0.85, x=1.02,
                    tickfont=dict(size=9),
                ) if i == 0 else None,
                hoverinfo='skip',
            ),
            row=row, col=col,
        )

        line_color = _BLUE if name in top_set else _GREY
        line_width = 1.5 if name in top_set else 0.8
        fig.add_trace(
            go.Scatter(
                x=t, y=signal[i],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                name=name,
                showlegend=False,
                hovertemplate=f'{name}: %{{y:.3f}} mV<extra></extra>',
            ),
            row=row, col=col,
        )

    # Title annotation listing top leads
    top_str = ', '.join(top_leads) if top_leads else '—'
    fig.update_layout(
        title=dict(
            text=f'ECG with saliency overlay · Top leads: {top_str}',
            font=dict(size=13),
        ),
        height=height,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=10, r=70, t=44, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True, key=key)

    # Colorbar legend caption
    st.caption('Saliency colorbar: light blue = low gradient magnitude → dark red = high')
