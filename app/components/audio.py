"""
Heartbeat audio component for the BioSignal-XAI Streamlit app.

Uses Web Audio API injected via st.components.v1.html().
The heartbeat tone changes based on whether the ECG is normal or anomalous:
  - NORM: soft low-pitched double-beat (lub-dub ~62 BPM)
  - Anomaly: sharper, higher pitch + faster tempo

Usage:
    from app.components.audio import render_heartbeat
    render_heartbeat(is_anomaly=False)
"""
from __future__ import annotations

import streamlit as st

_NORMAL_PARAMS = dict(
    freq1=80, freq2=100, dur1=0.08, dur2=0.06,
    gap=0.12, bpm=62, label='Normal sinus rhythm'
)
_ANOMALY_PARAMS = dict(
    freq1=140, freq2=170, dur1=0.06, dur2=0.05,
    gap=0.08, bpm=78, label='Anomaly detected'
)


def _build_html(p: dict, volume: float = 0.25) -> str:
    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:transparent;">
<div style="display:flex;align-items:center;gap:10px;font-family:sans-serif;">
  <button id="btn"
    style="background:#1976d2;color:#fff;border:none;border-radius:6px;
           padding:6px 16px;cursor:pointer;font-size:13px;line-height:1.4;">
    &#9829; Play heartbeat
  </button>
  <span style="font-size:12px;color:#000;">{p['label']} &middot; {p['bpm']} BPM</span>
</div>
<script>
(function() {{
  // Reuse one AudioContext across clicks (browsers limit how many can exist)
  var _ctx = null;

  function getCtx() {{
    if (!_ctx) {{
      _ctx = new (window.AudioContext || window.webkitAudioContext)();
    }}
    return _ctx;
  }}

  function beat(ctx, startTime, freq, dur, vol) {{
    var osc  = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, startTime);
    gain.gain.setValueAtTime(vol, startTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + dur);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(startTime);
    osc.stop(startTime + dur + 0.05);
  }}

  document.getElementById('btn').addEventListener('click', function() {{
    var ctx = getCtx();
    // Must resume — browsers suspend AudioContext until a user gesture
    var play = function() {{
      var now = ctx.currentTime;
      beat(ctx, now,              {p['freq1']}, {p['dur1']}, {volume});
      beat(ctx, now + {p['gap']}, {p['freq2']}, {p['dur2']}, {volume});
    }};
    if (ctx.state === 'suspended') {{
      ctx.resume().then(play);
    }} else {{
      play();
    }}
  }});
}})();
</script>
</body>
</html>"""


def render_heartbeat(is_anomaly: bool = False, volume: float = 0.25) -> None:
    """
    Render a clickable heartbeat button.

    Args:
        is_anomaly: True for anomaly tone (higher pitch, faster), False for normal
        volume:     Web Audio gain 0–1; default 0.25
    """
    params = _ANOMALY_PARAMS if is_anomaly else _NORMAL_PARAMS
    st.iframe(_build_html(params, volume=volume), height=44)
