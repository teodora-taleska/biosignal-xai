"""
BioSignal-XAI — Streamlit clinical demo.

Entry point: run from the repo root with:
    streamlit run app/app.py

Five tabs:
  1. Data Explorer      — browse the PTB-XL dataset
  2. Preprocessing      — visualise the preprocessing pipeline
  3. Interactive Demo   — pick a patient, run XResNet1D, see XAI
  4. Patient History    — [scaffolded] longitudinal trace comparison
  5. Real-Time Monitor  — [scaffolded] live inference streaming
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `app`, `src` etc. are importable
# regardless of which directory streamlit was launched from.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = 'BioSignal-XAI · ECG Demo',
    page_icon   = '🫀',
    layout      = 'wide',
    initial_sidebar_state = 'collapsed',
)

# ── Global CSS — Clinical Light theme ─────────────────────────────────────────
st.markdown("""
<style>
  /* Clinical white background */
  .main .block-container { background-color: #ffffff; padding-top: 1.5rem; }
  /* Blue accent on tab underlines */
  div[data-testid="stTabs"] button[aria-selected="true"] {
      border-bottom: 3px solid #1976d2 !important;
      color: #1976d2 !important;
      font-weight: 600;
  }
  div[data-testid="stTabs"] button { font-size: 14px; }
  /* Subtle metric boxes */
  [data-testid="metric-container"] {
      background-color: #f5f7fa;
      border: 1px solid #e0e4ea;
      border-radius: 8px;
      padding: 8px 12px;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="color:#1976d2; margin-bottom:0;">🫀 BioSignal-XAI</h2>'
    '<p style="color:#607d8b; margin-top:0; font-size:14px;">'
    'ECG anomaly detection · XResNet1D-101 · PTB-XL · Interactive clinical demo'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Tab routing ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📊 Data Explorer',
    '🔬 Preprocessing',
    '🧠 Interactive Demo',
    '📋 Patient History',
    '📡 Real-Time Monitor',
])

with tab1:
    from app.views.data_explorer import render
    render()

with tab2:
    from app.views.preprocessing import render
    render()

with tab3:
    from app.views.interactive_demo import render
    render()

with tab4:
    from app.views.patient_history import render
    render()

with tab5:
    from app.views.realtime_monitor import render
    render()
