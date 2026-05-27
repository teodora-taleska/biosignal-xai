"""
BioSignal-XAI -- Streamlit clinical demo.

Entry point: run from the repo root with:
    streamlit run app/main.py

Four tabs:
  1. Data Explorer    -- browse the PTB-XL dataset
  2. Preprocessing    -- visualise the preprocessing pipeline
  3. Interactive Demo -- pick a patient, run FCN-Wang, see XAI
  4. Live Monitor     -- real-time sliding-window ECG monitoring
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = 'BioSignal-XAI',
    page_icon   = 'app/assets/favicon.png' if Path('app/assets/favicon.png').exists() else None,
    layout      = 'wide',
    initial_sidebar_state = 'collapsed',
)

# ── Global CSS -- Clinical Light theme + Material Icons ───────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
  /* Force light mode */
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"] {
      background-color: #ffffff !important;
      color: #111111 !important;
  }
  .main .block-container {
      background-color: #ffffff !important;
      padding-top: 1.5rem;
  }
  /* Typography */
  p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6,
  [data-testid="stMarkdownContainer"],
  [data-testid="stText"],
  [data-testid="stCaption"] {
      color: #111111 !important;
  }
  /* Sidebar and widget labels */
  .stSelectbox label, .stMultiSelect label, .stSlider label,
  .stRadio label, .stCheckbox label, .stTextInput label {
      color: #111111 !important;
      font-size: 13px !important;
  }
  /* Metric labels and values */
  [data-testid="stMetricLabel"]  { color: #555555 !important; font-size: 12px !important; }
  [data-testid="stMetricValue"]  { color: #111111 !important; font-size: 20px !important; }
  /* Metric card background */
  [data-testid="metric-container"] {
      background-color: #f8f9fb !important;
      border: 1px solid #e0e4ea;
      border-radius: 8px;
      padding: 8px 14px;
  }
  /* Tabs -- active tab has blue underline */
  div[data-testid="stTabs"] button[aria-selected="true"] {
      border-bottom: 3px solid #1976d2 !important;
      color: #1976d2 !important;
      font-weight: 600;
  }
  div[data-testid="stTabs"] button {
      font-size: 13px;
      font-weight: 500;
      color: #444444 !important;
      padding: 8px 18px;
  }
  /* Expander headers */
  [data-testid="stExpander"] summary {
      color: #111111 !important;
  }
  /* Divider */
  hr { border-color: #e8eaed !important; }
  /* Caption */
  [data-testid="stCaptionContainer"] { color: #555555 !important; }
  /* Material Icons in headers */
  .material-icons {
      color: #1976d2;
      margin-right: 4px;
  }
  /* Primary button */
  [data-testid="stBaseButton-primary"] {
      background-color: #1976d2 !important;
      color: #ffffff !important;
      border-radius: 6px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── App header ────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:0;">'
    '<span class="material-icons" style="font-size:32px;color:#1976d2;">favorite</span>'
    '<div>'
    '<h2 style="color:#1976d2;margin:0;line-height:1.2;">BioSignal-XAI</h2>'
    '<p style="color:#555555;margin:0;font-size:13px;">'
    'ECG anomaly detection &nbsp;|&nbsp; FCN-Wang &nbsp;|&nbsp; PTB-XL &nbsp;|&nbsp; '
    'Interactive clinical demo'
    '</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Tab routing ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    'Data Explorer',
    'Preprocessing',
    'Interactive Demo',
    'Live Monitor',
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
    from app.views.realtime_monitor import render
    render()
