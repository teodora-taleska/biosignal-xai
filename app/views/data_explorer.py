"""
Tab 1 — Data Explorer.

Lets the user browse the PTB-XL dataset:
  • High-level dataset statistics (21 k records, 5 superclasses)
  • Class distribution bar chart
  • Curated 200-record table with filters (class, sex, age range)
  • Record detail: waveform + patient card + cached prediction

All heavy data is @st.cache_data so reloads are instant.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from app.data.loader import (
    load_metadata,
    load_signal,
    load_curated_index,
    load_predictions_cache,
    SUPERCLASSES,
    LEAD_NAMES,
)
from app.components.patient_card import render_patient_card
from app.components.confidence_gauge import render_confidence_gauge
from app.components.ecg_viewer import render_ecg


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner='Loading PTB-XL metadata …')
def _get_metadata() -> pd.DataFrame:
    return load_metadata()


@st.cache_data(show_spinner='Loading curated index …')
def _get_curated() -> list[dict]:
    return load_curated_index()


@st.cache_data(show_spinner='Loading predictions cache …')
def _get_predictions() -> dict:
    return load_predictions_cache()


@st.cache_data(show_spinner='Loading signal …')
def _get_signal(filename_lr: str) -> np.ndarray:
    return load_signal(filename_lr)


# ── Class distribution chart ──────────────────────────────────────────────────

def _class_distribution_chart(df: pd.DataFrame) -> None:
    """Bar chart of superclass counts in the full dataset."""
    import plotly.graph_objects as go

    # Count records per superclass (multi-hot, so count each separately)
    counts: dict[str, int] = {}
    for sc in SUPERCLASSES:
        counts[sc] = int(df['scp_codes'].apply(
            lambda codes: any(
                c == sc for c in codes
            )
        ).sum())

    colors = {
        'NORM': '#388e3c', 'MI': '#d32f2f',
        'STTC': '#f57c00', 'CD': '#7b1fa2', 'HYP': '#0288d1',
    }

    fig = go.Figure(go.Bar(
        x           = list(counts.keys()),
        y           = list(counts.values()),
        marker_color= [colors[c] for c in counts],
        text        = [f'{v:,}' for v in counts.values()],
        textposition= 'outside',
        hovertemplate='%{x}: %{y:,} records<extra></extra>',
    ))
    fig.update_layout(
        title='Superclass distribution (full PTB-XL)',
        yaxis_title='Records',
        height=280,
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Curated table with filters ────────────────────────────────────────────────

def _curated_table(records: list[dict]) -> list[dict]:
    """Render filter controls and the filtered curated-200 table. Returns filtered list."""
    st.markdown('#### Curated 200-record test subset (fold 10)')

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        sel_class = st.multiselect(
            'Filter by class', options=SUPERCLASSES, default=SUPERCLASSES,
            key='de_class_filter',
        )
    with col2:
        sel_sex = st.selectbox(
            'Sex', options=['All', 'Male', 'Female'], key='de_sex_filter'
        )
    with col3:
        age_range = st.slider(
            'Age range', min_value=0, max_value=100, value=(0, 100),
            key='de_age_filter',
        )

    sex_map = {'Male': 'M', 'Female': 'F'}
    filtered = []
    for rec in records:
        # Class filter
        if not any(c in sel_class for c in rec.get('superclass', [])):
            continue
        # Sex filter
        if sel_sex != 'All':
            if str(rec.get('sex', '')).upper() != sex_map.get(sel_sex, ''):
                continue
        # Age filter
        age = rec.get('age')
        if age is not None and not (age_range[0] <= age <= age_range[1]):
            continue
        filtered.append(rec)

    if not filtered:
        st.warning('No records match the current filters.')
        return []

    # Build display DataFrame
    rows = []
    for rec in filtered:
        rows.append({
            'ECG ID':   rec['ecg_id'],
            'Age':      int(rec['age']) if rec['age'] is not None else '—',
            'Sex':      {'M': 'Male', 'F': 'Female'}.get(str(rec.get('sex','')).upper(), '—'),
            'Classes':  ', '.join(rec.get('superclass', [])),
            'Primary':  rec.get('primary_class', '—'),
        })
    df_show = pd.DataFrame(rows)
    st.dataframe(df_show, use_container_width=True, height=260)
    st.caption(f'{len(filtered)} records shown')
    return filtered


# ── Record detail panel ───────────────────────────────────────────────────────

def _record_detail(records: list[dict], predictions: dict) -> None:
    """Let user pick one record and show its waveform + predictions."""
    if not records:
        return

    st.markdown('#### Record detail')
    ecg_ids    = [r['ecg_id'] for r in records]
    chosen_id  = st.selectbox(
        'Select ECG ID', options=ecg_ids, format_func=lambda x: f'#{x}',
        key='de_record_select',
    )
    rec   = next(r for r in records if r['ecg_id'] == chosen_id)
    pred  = predictions.get(str(chosen_id))

    col_left, col_right = st.columns([1, 2])
    with col_left:
        render_patient_card(rec)
        if pred:
            st.markdown('**Model prediction (XResNet1D)**')
            render_confidence_gauge(pred, true_classes=rec.get('superclass'))
        else:
            st.info('No cached prediction for this record.')

    with col_right:
        with st.spinner('Loading ECG signal …'):
            sig = _get_signal(rec['filename_lr'])
        render_ecg(sig, lead_names=LEAD_NAMES, title=f'ECG #{chosen_id}')


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.subheader('📊 Data Explorer')

    # Dataset summary metrics
    try:
        df = _get_metadata()
        curated    = _get_curated()
        predictions = _get_predictions()
    except FileNotFoundError as e:
        st.error(f'Cache not built yet. Run `python app/data/cache.py` first.\n\n{e}')
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Total records',    f'{len(df):,}')
    m2.metric('Unique patients',  f'{df["patient_id"].nunique():,}')
    m3.metric('Superclasses',     str(len(SUPERCLASSES)))
    m4.metric('Curated subset',   str(len(curated)))

    st.divider()

    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        _class_distribution_chart(df)
    with col_info:
        st.markdown("""
**PTB-XL at a glance**
- 10-second, 12-lead ECGs at 100 Hz
- Multi-label: one record can have multiple diagnoses
- Test fold (fold 10) used for curated subset
- **NORM** = normal sinus rhythm
- **MI** = myocardial infarction
- **STTC** = ST/T-wave change
- **CD** = conduction disturbance
- **HYP** = hypertrophy
        """)

    st.divider()
    filtered = _curated_table(curated)
    st.divider()
    _record_detail(filtered, predictions)
