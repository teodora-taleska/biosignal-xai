"""
Tab 2:Preprocessing Pipeline.

Educational visualisation of the ECG preprocessing steps used in this project:
  Step 1: Raw signal
  Step 2: Bandpass filter (0.5–40 Hz)
  Step 3: Z-score normalisation (shown for comparison; NOT used for inference)
  Step 4: Sliding windows (250 samples, stride 125)

User picks a record from the curated subset; all steps rendered as Plotly charts.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data.loader import (
    load_curated_index,
    load_signal,
    SUPERCLASSES,
    LEAD_NAMES,
)
from src.preprocessing.preprocess import bandpass_filter, normalize_signal

_FS = 100   # 100 Hz
_BLUE = '#1976d2'
_ORANGE = '#f57c00'


# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_curated() -> list[dict]:
    return load_curated_index()


@st.cache_data(show_spinner='Loading signal …')
def _get_signal(filename_lr: str) -> np.ndarray:
    return load_signal(filename_lr)


# ── Step plots ────────────────────────────────────────────────────────────────

def _lead_strip(
    signal:    np.ndarray,      # (1000,)
    t:         np.ndarray,
    title:     str,
    color:     str = _BLUE,
    height:    int = 150,
    reference: np.ndarray | None = None,   # optional overlay (grey)
) -> go.Figure:
    """Single-lead time-series plot."""
    fig = go.Figure()
    if reference is not None:
        fig.add_trace(go.Scatter(
            x=t, y=reference, mode='lines',
            line=dict(color='#cfd8dc', width=1),
            name='raw', showlegend=True,
        ))
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode='lines',
        line=dict(color=color, width=1.2),
        name=title, showlegend=True,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=height,
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        margin=dict(l=10, r=10, t=32, b=10),
        xaxis_title='seconds',
        yaxis_title='mV',
        legend=dict(orientation='h', y=1.15, x=0),
    )
    return fig


def _window_diagram(signal: np.ndarray, t: np.ndarray) -> go.Figure:
    """Show sliding windows as vertical bands on lead II."""
    window_size = 250
    stride      = 125
    n = signal.shape[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode='lines',
        line=dict(color=_BLUE, width=1),
        name='Lead II', showlegend=False,
    ))

    # Alternating window bands
    idx = 0
    toggle = True
    while idx + window_size <= n:
        t_start = t[idx]
        t_end   = t[idx + window_size - 1]
        fig.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor='#bbdefb' if toggle else '#e3f2fd',
            opacity=0.3, line_width=0,
        )
        # Window label
        win_num = idx // stride
        fig.add_annotation(
            x=(t_start + t_end) / 2, y=signal.max() * 1.05,
            text=f'W{win_num}', showarrow=False,
            font=dict(size=9, color='#1976d2'),
        )
        idx    += stride
        toggle  = not toggle

    fig.update_layout(
        title=dict(text=f'Sliding windows (size={window_size}, stride={stride})', font=dict(size=12)),
        height=200,
        plot_bgcolor='#fff', paper_bgcolor='#fff',
        margin=dict(l=10, r=10, t=32, b=10),
        xaxis_title='seconds',
        yaxis_title='mV',
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(
        '<h3 style="margin-bottom:4px;">'
        '<span class="material-icons" style="vertical-align:middle;font-size:24px;color:#1976d2;">biotech</span>'
        ' Preprocessing Pipeline</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        'Visualise each preprocessing step applied to an ECG record. '
        'The model used in this demo was trained with **bandpass filter only** '
        '(no normalisation, 2.5 s windows).'
    )

    try:
        curated = _get_curated()
    except FileNotFoundError as e:
        st.error(f'Cache not built. Run `python app/data/cache.py` first.\n\n{e}')
        return

    # ── Record selector ───────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        sel_class = st.selectbox(
            'Filter by class', options=['All'] + SUPERCLASSES, key='pp_class'
        )
    filtered = curated if sel_class == 'All' else [
        r for r in curated if sel_class in r.get('superclass', [])
    ]
    with col2:
        ecg_ids = [r['ecg_id'] for r in filtered]
        chosen_id = st.selectbox(
            'ECG record', options=ecg_ids,
            format_func=lambda x: f'#{x}',
            key='pp_record',
        )

    rec = next(r for r in filtered if r['ecg_id'] == chosen_id)

    with st.spinner('Loading ECG signal …'):
        raw = _get_signal(rec['filename_lr'])   # (1000, 12)

    lead_idx  = LEAD_NAMES.index('II')          # always visualise Lead II for clarity
    t         = np.arange(raw.shape[0]) / _FS

    # Also let user pick which lead to visualise in steps 1–3
    chosen_lead = st.selectbox(
        'Lead to visualise', options=LEAD_NAMES, index=lead_idx, key='pp_lead'
    )
    li = LEAD_NAMES.index(chosen_lead)

    raw_lead = raw[:, li]

    st.divider()

    # ── Step 1: Raw ───────────────────────────────────────────────────────────
    with st.expander('**Step 1: Raw signal**', expanded=True):
        st.markdown(
            'The PTB-XL records are read directly from WFDB format. '
            'No preprocessing applied yet. Baseline wander and high-frequency noise are visible.'
        )
        st.plotly_chart(
            _lead_strip(raw_lead, t, f'Raw - {chosen_lead}', color='#607d8b'),
            width="stretch", key='pp_step1',
        )

    # ── Step 2: Bandpass ──────────────────────────────────────────────────────
    with st.expander('**Step 2: Bandpass filter (0.5-40 Hz)**', expanded=True):
        st.markdown(
            'A 4th-order Butterworth bandpass filter (0.5-40 Hz) removes '
            'baseline wander (< 0.5 Hz) and high-frequency EMG noise (> 40 Hz). '
            '**This is the only preprocessing step used for inference in this demo.**'
        )
        bp = bandpass_filter(raw)[:, li]
        fig = _lead_strip(bp, t, f'Bandpass - {chosen_lead}', color=_BLUE, reference=raw_lead)
        st.plotly_chart(fig, width="stretch", key='pp_step2')

    # ── Step 3: Z-score (educational) ────────────────────────────────────────
    with st.expander('**Step 3: Z-score normalisation (for reference)**'):
        st.markdown(
            'Z-score normalisation (subtract mean, divide by std) removes '
            'amplitude differences between patients. '
            '**Not applied for this model** (`bandpass_none_250` config), '
            'but shown here for educational comparison.'
        )
        bp_full  = bandpass_filter(raw)
        norm_lead = normalize_signal(bp_full)[:, li]
        fig = _lead_strip(norm_lead, t, f'Z-score - {chosen_lead}', color=_ORANGE, reference=bp)
        st.plotly_chart(fig, width="stretch", key='pp_step3')

    # ── Step 4: Sliding windows ───────────────────────────────────────────────
    with st.expander('**Step 4: Sliding windows (2.5 s, stride 1.25 s)**', expanded=True):
        st.markdown(
            'The 10-second record (1000 samples) is split into overlapping '
            '**250-sample (2.5 s) windows** with a **stride of 125 samples (1.25 s)**. '
            'Each window is passed independently through the model, and window '
            'predictions are aggregated by mean pooling at inference time.'
        )
        bp_lead = bandpass_filter(raw)[:, li]
        st.plotly_chart(_window_diagram(bp_lead, t), width="stretch", key='pp_step4')

        n_windows = (raw.shape[0] - 250) // 125 + 1
        c1, c2, c3 = st.columns(3)
        c1.metric('Record length', '1 000 samples (10 s)')
        c2.metric('Window size',   '250 samples (2.5 s)')
        c3.metric('Windows / record', str(n_windows))
