"""
Tab 4 -- Real-Time Monitor.

Simulates streaming ECG monitoring using PTB-XL records at 100 Hz.
A sliding window advances through the 10-second signal; FCN-Wang classifies
each position in real time and updates the probability display live.
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.data.loader import SUPERCLASSES, load_curated_index, load_signal

_FS = 100  # Hz
_CLASS_COLORS = {
    'NORM': '#388e3c', 'MI': '#d32f2f',
    'STTC': '#f57c00', 'CD': '#7b1fa2', 'HYP': '#0288d1',
}


# ── UI helpers ────────────────────────────────────────────────────────────────

def _icon(name: str, size: int = 20) -> str:
    return (
        f'<span class="material-icons" '
        f'style="vertical-align:middle;font-size:{size}px;">{name}</span>'
    )


def _section_header(icon_name: str, title: str) -> None:
    st.markdown(
        f'<div style="background:#f5f7fa;border-left:3px solid #1976d2;'
        f'padding:6px 12px;border-radius:4px;margin:12px 0 6px 0;">'
        f'<strong>{_icon(icon_name)} {title}</strong></div>',
        unsafe_allow_html=True,
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def _ecg_fig(signal: np.ndarray, start: int, end: int) -> go.Figure:
    """
    Plotly figure: full 10-second Lead II in grey, active window in blue.

    Args:
        signal: (1000, 12) raw signal
        start:  first sample of the active window
        end:    one-past-last sample of the active window
    """
    t       = np.arange(1000) / _FS          # 0.00 .. 9.99 s
    lead_ii = signal[:, 1]                   # Lead II

    fig = go.Figure()

    # Full signal (greyed background trace)
    fig.add_trace(go.Scatter(
        x=t, y=lead_ii,
        mode='lines',
        line=dict(color='#d0d7de', width=1),
        showlegend=False,
    ))

    # Active window (blue foreground trace)
    fig.add_trace(go.Scatter(
        x=t[start:end], y=lead_ii[start:end],
        mode='lines',
        line=dict(color='#1976d2', width=2),
        showlegend=False,
    ))

    # Window highlight band
    fig.add_vrect(
        x0=t[start], x1=t[end - 1],
        fillcolor='#1976d2', opacity=0.07,
        line_width=1, line_color='#1976d2',
    )

    fig.update_layout(
        height=220,
        margin=dict(l=50, r=20, t=8, b=40),
        xaxis=dict(title='Time (s)', gridcolor='#f0f0f0', zeroline=False),
        yaxis=dict(title='Lead II (mV)', gridcolor='#f0f0f0', zeroline=True,
                   zerolinecolor='#e0e0e0'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
    )
    return fig


def _prob_fig(probs: dict, predicted: list[str]) -> go.Figure:
    """Horizontal bar chart of class probabilities, highlighting predicted classes."""
    classes = list(probs.keys())
    values  = [probs[c] for c in classes]
    colors  = [
        _CLASS_COLORS[c] if c in predicted else '#cfd8dc'
        for c in classes
    ]

    fig = go.Figure(go.Bar(
        y=classes,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.0%}' for v in values],
        textposition='outside',
        cliponaxis=False,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=60, r=60, t=8, b=10),
        xaxis=dict(range=[0, 1.15], tickformat='.0%', gridcolor='#f0f0f0'),
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
    )
    return fig


# ── Core monitoring loop ──────────────────────────────────────────────────────

def _run_monitor(
    signal: np.ndarray,
    rec: dict,
    window_s: int,
    delay: float,
    step_s: float,
) -> None:
    """
    Animate the sliding-window ECG monitor.

    At each step the ECG chart updates (Lead II display) and FCN-Wang runs
    inference on the full 10-second signal.  Probability bars and the alert
    banner refresh after every step.

    Args:
        signal:   (1000, 12) raw ECG
        rec:      record dict from curated_200.json
        window_s: visible window size in seconds
        delay:    sleep between steps (controls animation speed)
        step_s:   how many seconds of signal to advance per step
    """
    from app.model import predict as live_predict

    window_n = window_s * _FS               # samples in sliding window
    step_n   = max(1, int(step_s * _FS))   # samples per animation step
    positions = list(range(0, 1000 - window_n + 1, step_n))
    if not positions:
        positions = [0]
    total = len(positions)

    # FCN-Wang prediction on the full signal (stable; called once upfront,
    # then updated every step so the user sees it settle from the start)
    full_result = live_predict(signal)

    # Layout: ECG left (2/3) | probs right (1/3)
    alert_ph = st.empty()
    prog_ph  = st.empty()
    ecg_col, prob_col = st.columns([2, 1])
    ecg_ph  = ecg_col.empty()
    prob_ph = prob_col.empty()

    for i, start in enumerate(positions):
        end = start + window_n

        # Progress
        prog_ph.progress(
            (i + 1) / total,
            text=f'Monitoring ... {end / _FS:.1f}s / {1000 // _FS}s',
        )

        # ECG chart (Lead II, sliding window highlighted)
        ecg_ph.plotly_chart(
            _ecg_fig(signal, start, end),
            width="stretch",
            key=f'rt_ecg_{i}',
        )

        # Probability bars (full-signal prediction, stable)
        prob_ph.plotly_chart(
            _prob_fig(full_result['class_probabilities'],
                      full_result['predicted_classes']),
            width="stretch",
            key=f'rt_prob_{i}',
        )

        # Alert banner
        predicted = full_result['predicted_classes']
        if predicted == ['NORM']:
            alert_ph.markdown(
                '<div style="background:#d4edda;border:1px solid #c3e6cb;border-radius:6px;'
                'padding:10px 16px;display:flex;align-items:center;gap:8px;">'
                '<span class="material-icons" style="color:#388e3c;font-size:20px;">check_circle</span>'
                '<span style="color:#155724;font-weight:500;">Sinus rhythm: within normal limits</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            classes_str = ', '.join(predicted)
            conf_str    = f'{full_result["confidence_score"]:.0%}'
            alert_ph.markdown(
                '<div style="background:#f8d7da;border:1px solid #f5c6cb;border-radius:6px;'
                'padding:10px 16px;display:flex;align-items:center;gap:8px;">'
                '<span class="material-icons" style="color:#d32f2f;font-size:20px;">warning</span>'
                f'<span style="color:#721c24;font-weight:500;">Anomaly detected: '
                f'<strong>{classes_str}</strong> (confidence {conf_str})</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        time.sleep(delay)

    # Done
    prog_ph.progress(1.0, text='Monitoring complete')

    true_cls = rec.get('superclass', [])
    if true_cls:
        correct = set(full_result['predicted_classes']) == set(true_cls)
        if correct:
            st.success(
                f'Prediction correct. True label: {", ".join(true_cls)}'
            )
        elif set(full_result['predicted_classes']) & set(true_cls):
            st.warning(
                f'Partial match. Predicted: {", ".join(full_result["predicted_classes"])}, '
                f'true: {", ".join(true_cls)}'
            )
        else:
            st.error(
                f'Incorrect. Predicted: {", ".join(full_result["predicted_classes"])}, '
                f'true: {", ".join(true_cls)}'
            )


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(
        f'<h3 style="margin-bottom:4px;">'
        f'{_icon("monitor_heart", 26)} Real-Time Monitor</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        'Simulated streaming ECG monitoring. '
        'A sliding window advances through a 10-second PTB-XL record; '
        'FCN-Wang (~310 k params, ~1 ms/record) classifies the signal in real time.'
    )

    try:
        curated = load_curated_index()
    except FileNotFoundError:
        st.error(
            'Cache not built yet. '
            'Run `python app/data/cache.py` from the repo root first.'
        )
        return

    # ── Settings ──────────────────────────────────────────────────────────────
    _section_header('tune', 'Monitor settings')

    options = {r['ecg_id']: r for r in curated}

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        chosen_id = st.selectbox(
            'Record',
            list(options.keys()),
            format_func=lambda x: (
                f'#{x}: {", ".join(options[x]["superclass"])}'
                + (f'  (age {int(options[x]["age"])})' if options[x].get('age') else '')
            ),
            key='rt_record',
        )
    with col2:
        speed_label = st.select_slider(
            'Speed',
            options=['Slow', 'Normal', 'Fast'],
            value='Normal',
            key='rt_speed',
        )
    with col3:
        window_s = st.selectbox(
            'Window',
            [3, 5, 7],
            index=1,
            format_func=lambda x: f'{x} s',
            key='rt_window',
        )

    rec = options[chosen_id]

    speed_params = {
        'Slow':   dict(delay=0.5,  step_s=0.5),
        'Normal': dict(delay=0.2,  step_s=0.5),
        'Fast':   dict(delay=0.05, step_s=0.25),
    }
    sp = speed_params[speed_label]

    # ── Record info ───────────────────────────────────────────────────────────
    _section_header('info', 'Record info')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('ECG ID',     f'#{rec["ecg_id"]}')
    m2.metric('True class', ', '.join(rec.get('superclass', ['?'])))
    m3.metric('Age',        str(int(rec['age'])) if rec.get('age') else '?')
    m4.metric('Model',      'FCN-Wang')

    st.divider()

    # ── Start button ──────────────────────────────────────────────────────────
    if st.button('Start monitoring', type='primary', key='rt_start'):
        with st.spinner('Loading ECG signal ...'):
            signal = load_signal(rec['filename_lr'])   # (1000, 12)

        _run_monitor(
            signal   = signal,
            rec      = rec,
            window_s = window_s,
            delay    = sp['delay'],
            step_s   = sp['step_s'],
        )
