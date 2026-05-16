"""Tab 4 — Patient History (scaffolded placeholder)."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.subheader('📋 Patient History')
    st.info(
        '**Scaffolded — coming soon.**\n\n'
        'This tab will allow selecting a patient ID and comparing multiple '
        'ECG records over time, with longitudinal probability traces and '
        'trend annotations.',
        icon='🚧',
    )
    st.markdown("""
**Planned features:**
- Patient ID selector (from curated 200 subset)
- Timeline chart of class probabilities across visits
- Side-by-side ECG waveform comparison
- Delta annotations (improving / worsening class confidence)
    """)
