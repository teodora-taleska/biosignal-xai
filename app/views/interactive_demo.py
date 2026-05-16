"""
Tab 3 — Interactive Demo.

Full end-to-end clinical workflow for one selected ECG record:
  1. Patient selector (curated 200, filterable by class)
  2. Patient card + heartbeat audio
  3. Raw ECG waveform
  4. XResNet1D prediction (cached or live re-run)
  5. Confidence gauge with ground-truth comparison
  6. Saliency heatmap overlay (top-3 leads highlighted)
  7. Qwen3 clinical narrative (optional, on demand)
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app.components.audio import render_heartbeat
from app.components.confidence_gauge import render_confidence_gauge
from app.components.ecg_viewer import render_ecg, render_ecg_with_saliency
from app.components.patient_card import render_patient_card
from app.data.loader import (
    LEAD_NAMES,
    SUPERCLASSES,
    load_curated_index,
    load_predictions_cache,
    load_signal,
)


# ── Cached data ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_curated() -> list[dict]:
    return load_curated_index()


@st.cache_data(show_spinner=False)
def _get_predictions() -> dict:
    return load_predictions_cache()


@st.cache_data(show_spinner='Loading ECG signal …')
def _get_signal(filename_lr: str) -> np.ndarray:
    return load_signal(filename_lr)


# ── Section helpers ───────────────────────────────────────────────────────────

def _section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div style="background:#f5f7fa; border-left:3px solid #1976d2; '
        f'padding:6px 12px; border-radius:4px; margin:12px 0 6px 0;">'
        f'<strong>{icon} {title}</strong></div>',
        unsafe_allow_html=True,
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.subheader('🧠 Interactive Demo')
    st.markdown(
        'Select a patient from the curated 200-record test subset. '
        'The XResNet1D-101 model classifies the 12-lead ECG into one or more '
        'of 5 diagnostic superclasses. Gradient saliency explains which '
        'leads drove the prediction.'
    )

    # ── Data loading ──────────────────────────────────────────────────────────
    try:
        curated      = _get_curated()
        predictions  = _get_predictions()
    except FileNotFoundError as e:
        st.error(f'Cache not built. Run `python app/data/cache.py` first.\n\n{e}')
        return

    # ── Patient selector ──────────────────────────────────────────────────────
    _section_header('🔍', 'Patient selector')

    sel_col1, sel_col2 = st.columns([1, 2])
    with sel_col1:
        sel_class = st.selectbox(
            'Filter by class', options=['All'] + SUPERCLASSES, key='id_class'
        )
    filtered = curated if sel_class == 'All' else [
        r for r in curated if sel_class in r.get('superclass', [])
    ]
    with sel_col2:
        ecg_ids   = [r['ecg_id'] for r in filtered]
        chosen_id = st.selectbox(
            'ECG record',
            options   = ecg_ids,
            format_func = lambda x: f'#{x}',
            key='id_record',
        )

    rec  = next(r for r in filtered if r['ecg_id'] == chosen_id)
    pred = predictions.get(str(chosen_id))  # may be None

    # ── Patient info + audio ──────────────────────────────────────────────────
    _section_header('👤', 'Patient info')
    info_col, audio_col = st.columns([2, 1])
    with info_col:
        render_patient_card(rec)
    with audio_col:
        is_anomaly = bool(rec.get('superclass') and rec['superclass'] != ['NORM'])
        st.markdown('**Heartbeat**')
        render_heartbeat(is_anomaly=is_anomaly)

    # ── ECG waveform ──────────────────────────────────────────────────────────
    _section_header('📈', 'ECG waveform')
    with st.spinner('Loading ECG …'):
        raw = _get_signal(rec['filename_lr'])
    render_ecg(raw, lead_names=LEAD_NAMES, title=f'Raw ECG — Record #{chosen_id}', key='id_ecg_raw')

    # ── Prediction ────────────────────────────────────────────────────────────
    _section_header('🤖', 'XResNet1D-101 prediction')

    if pred is None:
        st.warning('No cached prediction. Click "Run inference" to compute one live.')
        if st.button('▶ Run inference', key='id_infer'):
            from app.model import predict as live_predict
            with st.spinner('Running XResNet1D …'):
                pred = live_predict(raw)
            st.success('Inference complete!')
            st.rerun()

    if pred:
        g_col, p_col = st.columns([2, 1])
        with g_col:
            render_confidence_gauge(pred, true_classes=rec.get('superclass'), key='id_gauge')
        with p_col:
            st.markdown('**Predicted:**')
            for cls in pred['predicted_classes']:
                st.markdown(
                    f'<span style="background:#1976d2; color:#fff; padding:3px 10px; '
                    f'border-radius:12px; font-size:14px; margin:2px;">{cls}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown('')
            st.metric('Confidence', f"{pred['confidence_score']:.0%}")

            true_classes = rec.get('superclass', [])
            correct = set(pred['predicted_classes']) == set(true_classes)
            if true_classes:
                st.markdown(
                    f'**Ground truth:** {", ".join(true_classes)}'
                )
                if correct:
                    st.success('✅ Correct prediction')
                else:
                    predicted_set = set(pred['predicted_classes'])
                    true_set      = set(true_classes)
                    if predicted_set & true_set:
                        st.warning('⚠️ Partial match')
                    else:
                        st.error('❌ Incorrect prediction')

    # ── Saliency ──────────────────────────────────────────────────────────────
    if pred:
        _section_header('🔍', 'Gradient saliency map')
        target_class = st.selectbox(
            'Target class for saliency',
            options=pred['predicted_classes'],
            key='id_saliency_target',
        )

        if st.button('🔍 Compute saliency', key='id_sal_btn'):
            from app.model import get_saliency, get_top_leads
            with st.spinner('Computing gradient saliency …'):
                sal = get_saliency(raw, target_class=target_class, result=pred)
                top = get_top_leads(sal, top_k=3)
            st.session_state['_sal']  = sal
            st.session_state['_top']  = top
            st.session_state['_sal_target'] = target_class

        if '_sal' in st.session_state and st.session_state.get('_sal_target') == target_class:
            sal = st.session_state['_sal']
            top = st.session_state['_top']
            st.markdown(f'**Top-3 salient leads:** {", ".join(top)}')
            render_ecg_with_saliency(raw, sal, lead_names=LEAD_NAMES, top_leads=top, key='id_ecg_sal')

    # ── LLM explanation ───────────────────────────────────────────────────────
    if pred:
        _section_header('💬', 'Clinical narrative (Qwen3-0.6B)')
        st.markdown(
            '_AI-generated interpretation — for educational purposes only. '
            'Always requires clinical correlation._'
        )

        if st.button('Generate clinical explanation', key='id_llm_btn'):
            from app.model import get_qwen3
            from src.explainability.llm import build_ecg_prompt, generate_explanation

            qwen_model, qwen_tok = get_qwen3()
            if qwen_model is None:
                st.warning('Qwen3 not available.')
            else:
                # Enrich result with dummy uncertainty fields if absent
                full_pred = dict(pred)
                full_pred.setdefault('uncertainty', None)
                full_pred.setdefault('uncertainty_level', 'not computed')

                sal = st.session_state.get('_sal')
                prompt = build_ecg_prompt(
                    result_dict  = full_pred,
                    saliency     = sal,
                    lead_names   = LEAD_NAMES,
                    true_classes = rec.get('superclass'),
                )
                with st.spinner('Qwen3 generating explanation …'):
                    explanation = generate_explanation(prompt, qwen_model, qwen_tok)
                st.info(explanation)
