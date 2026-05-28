"""
Animated ECG monitor component.

Scrolls the patient's actual lead II signal across a canvas in real-time
(100 samples / second = 10 seconds of data) and plays a heartbeat sound
timed to each detected R-peak.

Usage:
    from app.components.ecg_animation import render_ecg_monitor
    render_ecg_monitor(signal_np, is_anomaly=False)
"""
from __future__ import annotations

import json

import numpy as np
import streamlit as st


# ── R-peak detection ──────────────────────────────────────────────────────────

def detect_r_peaks(signal: np.ndarray) -> tuple[float, list[int]]:
    """
    Detect R-peaks from lead II and return (bpm, peak_sample_indices).

    Args:
        signal: (1000, 12) or (12, 1000) float32 array

    Returns:
        (bpm, list of integer sample indices)
    """
    from scipy.signal import find_peaks

    # Normalise to (1000, 12)
    if signal.shape == (12, 1000):
        signal = signal.T

    lead2 = signal[:, 1].astype(float)   # Lead II

    # Standardise so threshold is scale-independent
    mu, sd = lead2.mean(), lead2.std() + 1e-9
    lead2_z = (lead2 - mu) / sd

    # R-peaks are tall, separated by at least 40 samples (0.4 s → < 150 BPM)
    peaks, _ = find_peaks(lead2_z, height=0.6, distance=40)

    if len(peaks) >= 2:
        bpm = round(60.0 * (len(peaks) - 1) / ((peaks[-1] - peaks[0]) / 100.0), 1)
    else:
        bpm = 70.0

    return bpm, peaks.tolist()


# ── HTML / JS component ───────────────────────────────────────────────────────

def _build_monitor_html(
    lead2:      list[float],
    peaks:      list[int],
    bpm:        float,
    is_anomaly: bool,
    height:     int,
) -> str:
    freq1 = 140 if is_anomaly else 80
    freq2 = 175 if is_anomaly else 105
    status_color = '#e53935' if is_anomaly else '#43a047'
    status_label = 'ANOMALY' if is_anomaly else 'NORMAL'

    lead2_json = json.dumps(lead2)
    peaks_json = json.dumps(peaks)

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; font-family: monospace; }}
  #monitor {{
    background:#0d1117;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 8px 12px;
    width: 100%;
  }}
  #header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }}
  #lead-label {{ color:#4fc3f7; font-size:12px; letter-spacing:1px; }}
  #bpm-display {{
    font-size:22px;
    font-weight:bold;
    color:#e0e0e0;
  }}
  #bpm-unit {{ font-size:11px; color:#888; margin-left:3px; }}
  #status {{
    font-size:11px;
    font-weight:bold;
    color:{status_color};
    letter-spacing:2px;
    padding: 2px 8px;
    border: 1px solid {status_color};
    border-radius: 4px;
  }}
  canvas {{
    display:block;
    width:100%;
    height:130px;
    background:#0d1117;
  }}
  #controls {{
    margin-top:8px;
    display:flex;
    align-items:center;
    gap:12px;
  }}
  #btn {{
    background:#1976d2;
    color:#fff;
    border:none;
    border-radius:6px;
    padding:5px 16px;
    cursor:pointer;
    font-size:12px;
    font-family:monospace;
  }}
  #btn:hover {{ background:#1565c0; }}
  #progress {{
    font-size:11px;
    color:#555;
  }}
</style>
</head>
<body>
<div id="monitor">
  <div id="header">
    <span id="lead-label">LEAD II</span>
    <span id="bpm-display">{bpm:.0f}<span id="bpm-unit">BPM</span></span>
    <span id="status">{status_label}</span>
  </div>
  <canvas id="ecg"></canvas>
  <div id="controls">
    <button id="btn">▶ Play</button>
    <span id="progress">Click Play to animate the ECG signal</span>
  </div>
</div>

<script>
(function() {{
  const signal  = {lead2_json};
  const peaks   = new Set({peaks_json});
  const FS      = 100;
  const N       = signal.length;
  const FREQ1   = {freq1};
  const FREQ2   = {freq2};
  const VOL     = 0.28;

  const canvas  = document.getElementById('ecg');
  const ctx     = canvas.getContext('2d');
  const btn     = document.getElementById('btn');
  const prog    = document.getElementById('progress');

  // Set canvas pixel dimensions to match its CSS size
  function resizeCanvas() {{
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  || canvas.offsetWidth  || 600;
    canvas.height = rect.height || canvas.offsetHeight || 130;
  }}
  // Run after layout is painted
  requestAnimationFrame(() => {{ resizeCanvas(); drawGrid(); }});

  // Normalise signal for display: map to [0.1 * H, 0.9 * H]
  const lo = Math.min(...signal), hi = Math.max(...signal);
  const range = hi - lo || 1;
  function toY(v) {{
    const H = canvas.height;
    return H - ((v - lo) / range) * H * 0.8 - H * 0.1;
  }}

  // Draw grid
  function drawGrid() {{
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#1a2030';
    ctx.lineWidth = 0.5;
    // vertical lines every 25px (0.25 s)
    for (let x = 0; x < W; x += 25) {{
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }}
    // horizontal lines every 20px
    for (let y = 0; y < H; y += 20) {{
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }}
  }}

  // Audio
  let audioCtx = null;
  function getAudio() {{
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return audioCtx;
  }}
  function scheduleBeat(t) {{
    const ac = getAudio();
    function tone(startT, freq, dur) {{
      const o = ac.createOscillator(), g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = freq;
      g.gain.setValueAtTime(VOL, startT);
      g.gain.exponentialRampToValueAtTime(0.0001, startT + dur);
      o.connect(g); g.connect(ac.destination);
      o.start(startT); o.stop(startT + dur + 0.05);
    }}
    tone(t,        FREQ1, 0.08);
    tone(t + 0.13, FREQ2, 0.07);
  }}

  // Animation state
  let animId    = null;
  let startTs   = null;
  let audioBase = null;
  let running   = false;

  function stop() {{
    if (animId) {{ cancelAnimationFrame(animId); animId = null; }}
    running = false;
    btn.textContent = '▶ Play again';
    prog.textContent = 'Done — ' + (N / FS).toFixed(0) + ' s recorded';
  }}

  function start() {{
    const ac = getAudio();
    const resume = ac.state === 'suspended' ? ac.resume() : Promise.resolve();
    resume.then(() => {{
      // Schedule all beats upfront relative to audio clock
      audioBase = ac.currentTime + 0.05;
      peaks.forEach(p => {{
        scheduleBeat(audioBase + p / FS);
      }});

      startTs = null;
      running = true;
      btn.textContent = '⏹ Stop';
      requestAnimationFrame(frame);
    }});
  }}

  function frame(ts) {{
    if (!running) return;
    if (!startTs) startTs = ts;
    const elapsed = (ts - startTs) / 1000;     // seconds since start
    const cur     = Math.min(Math.floor(elapsed * FS), N - 1);

    const W = canvas.width, H = canvas.height;
    // px per sample — show a 6-second window
    const WIN  = Math.min(N, 600);             // 6 s at 100 Hz
    const pxPerSample = W / WIN;

    drawGrid();

    // Trace
    ctx.beginPath();
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = '#00e676';
    ctx.shadowBlur  = 3;

    const startSample = Math.max(0, cur - WIN + 1);
    for (let i = startSample; i <= cur; i++) {{
      const x = (i - startSample) * pxPerSample;
      const y = toY(signal[i]);
      if (i === startSample) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Blinking cursor dot
    if (cur > 0) {{
      const cx = Math.min(cur, WIN - 1) * pxPerSample;
      ctx.beginPath();
      ctx.arc(cx, toY(signal[cur]), 3, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
    }}

    prog.textContent = (cur / FS).toFixed(1) + ' / ' + (N / FS).toFixed(0) + ' s';

    if (cur < N - 1) {{
      animId = requestAnimationFrame(frame);
    }} else {{
      stop();
    }}
  }}

  btn.addEventListener('click', () => {{
    if (running) {{
      stop();
      drawGrid();
    }} else {{
      start();
    }}
  }});
}})();
</script>
</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────────

def render_ecg_monitor(
    signal:     np.ndarray,
    is_anomaly: bool = False,
    height:     int  = 260,
) -> tuple[float, list[int]]:
    """
    Render the animated ECG cardiac monitor widget.

    Args:
        signal:     raw (1000, 12) or (12, 1000) float32 ECG signal
        is_anomaly: True → red status badge + higher-pitched heartbeat
        height:     total component height in pixels

    Returns:
        (bpm, peak_indices) from the R-peak detector
    """
    if signal.shape == (12, 1000):
        signal = signal.T   # → (1000, 12)

    bpm, peaks = detect_r_peaks(signal)

    lead2 = signal[:, 1].tolist()   # Lead II as plain list for JSON
    html  = _build_monitor_html(lead2, peaks, bpm, is_anomaly, height)
    st.iframe(html, height=height)
    return bpm, peaks
