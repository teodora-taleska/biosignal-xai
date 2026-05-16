"""Tab 5 — Real-Time Monitor (scaffolded placeholder)."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.subheader('📡 Real-Time Monitor')
    st.info(
        '**Scaffolded — coming soon.**\n\n'
        'This tab will stream sliding windows of a PTB-XL record through '
        'the XResNet1D model in real time, animating the ECG waveform and '
        'updating class probability bars live.',
        icon='🚧',
    )
    st.markdown("""
**Planned features:**
- Animated ECG scrolling at 100 Hz (using `st.empty()` + `time.sleep`)
- Live class probability update every 2.5 s window
- Heartbeat audio sync
- Anomaly alert badge with timestamp
- Latency profiling display (ms/window)
    """)

    # Demonstrate live inference works (static single-record run)
    st.divider()
    st.markdown('**Live inference smoke test** (single record, no animation):')
    if st.button('▶ Run one inference'):
        import numpy as np
        from app.data.loader import load_curated_index, load_signal
        from app.model import predict

        records = load_curated_index()
        rec     = records[0]
        signal  = load_signal(rec['filename_lr'])
        result  = predict(signal)

        st.success(
            f"ECG #{rec['ecg_id']} → predicted: **{', '.join(result['predicted_classes'])}**  "
            f"| confidence: **{result['confidence_score']:.0%}**"
        )
