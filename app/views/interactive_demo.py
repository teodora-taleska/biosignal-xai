"""
Tab 3 -- Interactive Demo.

Full end-to-end clinical workflow for one selected ECG record:
  1. Patient selector (curated 200, filterable by class)
  2. Patient card
  3. ECG monitor animation (Lead II, real-time scrolling)
  4. FCN-Wang prediction (cached or live re-run)
  5. Confidence gauge with ground-truth comparison
  6. XAI analysis (one-click): gradient saliency + Qwen2-0.5B clinical narrative
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app.components.ecg_animation import render_ecg_monitor
from app.components.confidence_gauge import render_confidence_gauge
from app.components.ecg_viewer import render_ecg_with_saliency
from app.components.patient_card import render_patient_card
import os

from app.data.loader import (
    LEAD_NAMES,
    SUPERCLASSES,
    load_curated_index,
    load_narratives_cache,
    load_predictions_cache,
    load_signal,
)

# Cloud mode: set CLOUD_MODE=true in Streamlit secrets or environment.
# In cloud mode, Qwen2 is never loaded — pre-cached narratives are served instead.
_CLOUD_MODE = os.getenv('CLOUD_MODE', 'false').lower() == 'true'


# ── Cached data ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_curated() -> list[dict]:
    return load_curated_index()


@st.cache_data(show_spinner=False)
def _get_predictions() -> dict:
    return load_predictions_cache()


@st.cache_data(show_spinner=False)
def _get_narratives() -> dict:
    return load_narratives_cache()


@st.cache_data(show_spinner='Loading ECG signal ...')
def _get_signal(filename_lr: str) -> np.ndarray:
    return load_signal(filename_lr)


# ── Section helpers ───────────────────────────────────────────────────────────

def _icon(name: str, size: int = 18) -> str:
    return (
        f'<span class="material-icons" '
        f'style="vertical-align:middle;font-size:{size}px;">{name}</span>'
    )


def _section_header(icon_name: str, title: str) -> None:
    st.markdown(
        f'<div style="background:#f5f7fa; border-left:3px solid #1976d2; '
        f'padding:6px 12px; border-radius:4px; margin:12px 0 6px 0;">'
        f'<strong>{_icon(icon_name)} {title}</strong></div>',
        unsafe_allow_html=True,
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(
        f'<h3 style="margin-bottom:4px;">'
        f'{_icon("psychology", 26)} Interactive Demo</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        'Select a patient from the curated 200-record test subset. '
        'FCN-Wang classifies the 12-lead ECG into one or more of '
        '5 diagnostic superclasses. Gradient saliency explains which '
        'leads drove the prediction.'
    )

    # ── Data loading ──────────────────────────────────────────────────────────
    try:
        curated      = _get_curated()
        predictions  = _get_predictions()
    except FileNotFoundError as e:
        st.error(f'Cache not built. Run `python app/data/cache.py` first.\n\n{e}')
        return

    narratives = _get_narratives() if _CLOUD_MODE else {}

    # ── Patient selector ──────────────────────────────────────────────────────
    _section_header('search', 'Patient selector')

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

    # ── Patient info ──────────────────────────────────────────────────────────
    _section_header('person', 'Patient info')
    render_patient_card(rec)

    # ── ECG monitor + heartbeat ───────────────────────────────────────────────
    _section_header('show_chart', 'ECG monitor')
    with st.spinner('Loading ECG …'):
        raw = _get_signal(rec['filename_lr'])
    is_anomaly = bool(rec.get('superclass') and rec['superclass'] != ['NORM'])
    st.caption('Press **Play** to scroll the lead II signal in real-time with heartbeat audio synced to detected R-peaks.')
    render_ecg_monitor(raw, is_anomaly=is_anomaly)

    # ── Prediction ────────────────────────────────────────────────────────────
    _section_header('smart_toy', 'FCN-Wang prediction')

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

            # Confidence score with ± stability range
            top_cls   = pred['predicted_classes'][0] if pred['predicted_classes'] else None
            unc_pc    = pred.get('uncertainty_per_class', {})
            conf_std  = unc_pc.get(top_cls, 0.0) if top_cls else 0.0
            conf_val  = pred['confidence_score']
            conf_label = (
                f"{conf_val:.0%} ± {conf_std:.0%}"
                if conf_std > 0 else f"{conf_val:.0%}"
            )

            metric_col, gap_col, info_col = st.columns([4, 1, 1])
            with metric_col:
                st.metric('Confidence', conf_label)
            with info_col:
                st.markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)
                with st.popover('ⓘ', use_container_width=True):
                    st.markdown(
                        '**What does this confidence score mean?**\n\n'
                        'The percentage shows how strongly the model believes '
                        'in its top prediction based on a single analysis of the ECG.\n\n'
                        'The **± figure** tells you how *stable* that answer is. '
                        'We run the same analysis 20 times with very small random '
                        'variations added to the signal, similar to the natural '
                        'noise present in any real ECG recording. '
                        'The ± shows how much the result changed across those 20 runs.\n\n'
                        '**How to read it:**\n'
                        '- **87% ± 2%**: High confidence, very stable. '
                        'The model gives the same answer consistently. '
                        'The true likelihood is reliably in the 85–89% range.\n'
                        '- **87% ± 15%**: High raw score, but unstable. '
                        'Small signal changes shift the answer noticeably. '
                        'Clinical review is recommended before acting on this result.\n'
                        '- **52% ± 3%**: Low confidence, stable. '
                        'The model is consistently uncertain, the signal may not '
                        'contain clear enough features to classify.\n\n'
                        '*This tool is for research purposes only and does not '
                        'replace clinical judgement.*'
                    )

            true_classes = rec.get('superclass', [])
            correct = set(pred['predicted_classes']) == set(true_classes)
            if true_classes:
                st.markdown(
                    f'**Ground truth:** {", ".join(true_classes)}'
                )
                if correct:
                    st.success('Correct prediction')
                else:
                    predicted_set = set(pred['predicted_classes'])
                    true_set      = set(true_classes)
                    if predicted_set & true_set:
                        st.warning('Partial match')
                    else:
                        st.error('Incorrect prediction')

    # ── XAI analysis ──────────────────────────────────────────────────────────
    if pred:
        _section_header('psychology', 'XAI Analysis')
        st.markdown(
            'Gradient saliency highlights the ECG segments that most influenced '
            'the prediction. Qwen2-0.5B-Instruct then generates a concise clinical '
            'interpretation. Both run together in one step.'
        )

        target_class = st.selectbox(
            'Target class for saliency',
            options=pred['predicted_classes'],
            key='id_saliency_target',
        )

        if st.button(
            'Run XAI analysis',
            type='primary',
            key='id_xai_btn',
        ):
            from app.model import get_saliency, get_top_leads

            # Step 1: gradient saliency (always computed live)
            with st.spinner('Computing gradient saliency ...'):
                sal = get_saliency(raw, target_class=target_class, result=pred)
                top = get_top_leads(sal, top_k=3)
            st.session_state['_sal']        = sal
            st.session_state['_top']        = top
            st.session_state['_sal_target'] = target_class

            # Step 2: narrative -- cached in cloud mode, live Qwen2 in local mode
            if _CLOUD_MODE:
                cached = narratives.get(str(rec['ecg_id']))
                st.session_state['_xai_explanation']        = cached
                st.session_state['_xai_explanation_target'] = target_class
            else:
                from app.model import get_qwen3
                from src.explainability.llm import build_ecg_prompt, generate_explanation
                qwen_model, qwen_tok = get_qwen3()
                if qwen_model is not None:
                    prompt = build_ecg_prompt(
                        result_dict  = pred,
                        saliency     = sal,
                        lead_names   = LEAD_NAMES,
                        true_classes = rec.get('superclass'),
                    )
                    with st.spinner('Qwen2 generating clinical narrative ...'):
                        explanation = generate_explanation(prompt, qwen_model, qwen_tok)
                    st.session_state['_xai_explanation']        = explanation
                    st.session_state['_xai_explanation_target'] = target_class
                else:
                    st.warning('Qwen2 model unavailable -- showing saliency only.')
                    st.session_state.pop('_xai_explanation', None)

        # Display results when available for the selected target class
        if (
            '_sal' in st.session_state
            and st.session_state.get('_sal_target') == target_class
        ):
            sal = st.session_state['_sal']
            top = st.session_state['_top']

            st.markdown(f'**Top-3 salient leads:** {", ".join(top)}')
            render_ecg_with_saliency(
                raw, sal,
                lead_names = LEAD_NAMES,
                top_leads  = top,
                key        = 'id_ecg_sal',
            )

            explanation = st.session_state.get('_xai_explanation')
            if (
                explanation is not None
                and st.session_state.get('_xai_explanation_target') == target_class
            ):
                _section_header('chat', 'Clinical narrative (Qwen2-0.5B-Instruct)')
                st.caption(
                    'AI-generated interpretation, for educational purposes only. '
                    'Always requires clinical correlation.'
                )
                st.info(explanation)
