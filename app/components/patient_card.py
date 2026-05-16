"""
Patient info card component.

Renders a compact card with demographics and ground-truth labels.

Usage:
    from app.components.patient_card import render_patient_card
    render_patient_card(record_dict)
"""
from __future__ import annotations

import streamlit as st

# Clinical Light palette
_BLUE     = '#1976d2'
_LIGHT_BG = '#f5f7fa'
_BORDER   = '#e0e4ea'

# Superclass color chips
_CLASS_COLORS: dict[str, str] = {
    'NORM': '#388e3c',   # green
    'MI':   '#d32f2f',   # red
    'STTC': '#f57c00',   # orange
    'CD':   '#7b1fa2',   # purple
    'HYP':  '#0288d1',   # cyan-blue
}


def _class_chip(cls: str) -> str:
    color = _CLASS_COLORS.get(cls, '#607d8b')
    return (
        f'<span style="background:{color}; color:#fff; padding:2px 8px; '
        f'border-radius:10px; font-size:12px; margin-right:4px;">{cls}</span>'
    )


def render_patient_card(record: dict) -> None:
    """
    Render a styled patient info card.

    Args:
        record: dict from curated_200.json with keys:
                ecg_id, patient_id, age, sex, superclass, primary_class
    """
    age    = record.get('age')
    age_s  = f"{int(age)} yrs" if age is not None else "N/A"
    sex    = record.get('sex', '')
    sex_s  = {'0': 'Male', '1': 'Female', 0: 'Male', 1: 'Female',
              'M': 'Male', 'F': 'Female'}.get(str(sex), 'N/A')
    ecg_id = record.get('ecg_id', '—')

    classes   = record.get('superclass', [])
    chips_html = ''.join(_class_chip(c) for c in classes)

    card_html = f"""
<div style="
  background:{_LIGHT_BG};
  border:1px solid {_BORDER};
  border-left: 4px solid {_BLUE};
  border-radius:8px;
  padding:12px 16px;
  margin-bottom:8px;
  font-family: sans-serif;
">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
    <div>
      <span style="font-size:13px; color:#000000;">ECG ID</span><br>
      <span style="font-size:18px; font-weight:600; color:{_BLUE};">#{ecg_id}</span>
    </div>
    <div style="text-align:right;">
      <span style="font-size:13px; color:#000000;">{sex_s}</span><br>
      <span style="font-size:16px; font-weight:500;">{age_s}</span>
    </div>
  </div>
  <div style="margin-top:8px;">
    <span style="font-size:12px; color:#000000; margin-right:6px;">Diagnoses:</span>
    {chips_html if chips_html else '<span style="color:#000000; font-size:12px;">none</span>'}
  </div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)
