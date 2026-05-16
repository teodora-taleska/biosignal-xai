"""
Heartbeat audio component for the BioSignal-XAI Streamlit app.

Injects a small Web Audio API snippet via st.components.v1.html().
The heartbeat tone changes based on whether the ECG is normal or anomalous:
  - NORM: soft low-pitched double-beat (lub-dub ~60 BPM)
  - Anomaly: sharper, slightly higher pitch + shorter interval

Usage:
    from app.components.audio import render_heartbeat
    render_heartbeat(is_anomaly=False)
"""
from __future__ import annotations

import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Heartbeat parameters
# ---------------------------------------------------------------------------
_NORMAL_PARAMS = dict(
    freq1=80, freq2=100, dur1=0.08, dur2=0.06,
    gap=0.12, bpm=62, label='Normal sinus rhythm'
)
_ANOMALY_PARAMS = dict(
    freq1=130, freq2=160, dur1=0.06, dur2=0.05,
    gap=0.08, bpm=74, label='Anomaly detected'
)


def _build_js(p: dict, volume: float = 0.18) -> str:
    """Return a self-contained JS snippet that plays one heartbeat cycle."""
    return f"""
<div style="display:flex; align-items:center; gap:10px; margin:4px 0;">
  <button
    onclick="playBeat()"
    style="background:#1976d2; color:#fff; border:none; border-radius:6px;
           padding:6px 14px; cursor:pointer; font-size:13px;">
    ♥ Play heartbeat
  </button>
  <span style="font-size:12px; color:#000000;">{p['label']} · {p['bpm']} BPM</span>
</div>

<script>
function playBeat() {{
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const vol = {volume};

  function beat(t, freq, dur) {{
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type      = 'sine';
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(vol, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + dur + 0.02);
  }}

  const now = ctx.currentTime;
  beat(now,                  {p['freq1']}, {p['dur1']});
  beat(now + {p['gap']},     {p['freq2']}, {p['dur2']});
}}
</script>
"""


def render_heartbeat(is_anomaly: bool = False, volume: float = 0.18) -> None:
    """
    Render a clickable heartbeat button.

    Args:
        is_anomaly: True for anomaly tone (higher pitch, faster), False for normal
        volume:     Web Audio gain (0–1); default 0.18 is unobtrusive
    """
    params = _ANOMALY_PARAMS if is_anomaly else _NORMAL_PARAMS
    html   = _build_js(params, volume=volume)
    components.html(html, height=48)
