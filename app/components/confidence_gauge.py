"""
Confidence gauge component — horizontal probability bar chart.

Shows 5 superclass probabilities as a Plotly horizontal bar chart with
colour coding: green for high confidence predicted class, grey for others.

Usage:
    from app.components.confidence_gauge import render_confidence_gauge
    render_confidence_gauge(result_dict)
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

_COLOR_HIGH  = '#1976d2'   # predicted class bar
_COLOR_LOW   = '#b0bec5'   # other bars
_COLOR_TRUE  = '#388e3c'   # ground-truth highlight ring


def render_confidence_gauge(
    result: dict,
    true_classes: list[str] | None = None,
    height: int = 220,
    key: str = 'confidence_gauge',
) -> None:
    """
    Render a horizontal bar chart of class probabilities with ± uncertainty bars.

    Args:
        result:       prediction dict from app.model.predict()
        true_classes: optional list of ground-truth class names (adds green outline)
        height:       chart height in pixels
    """
    probs      = result['class_probabilities']
    predicted  = set(result['predicted_classes'])
    true_set   = set(true_classes or [])
    stds       = result.get('uncertainty_per_class', {})

    labels      = SUPERCLASSES
    values      = [probs[c] for c in labels]
    errors      = [stds.get(c, 0.0) for c in labels]
    colors      = [_COLOR_HIGH if c in predicted else _COLOR_LOW for c in labels]
    line_colors = [_COLOR_TRUE if c in true_set else 'rgba(0,0,0,0)' for c in labels]
    line_widths = [2 if c in true_set else 0 for c in labels]

    # Label shows mean ± std, e.g. "74% ± 3%"
    text_labels = [
        f'{v:.0%} ± {e:.0%}' if e > 0 else f'{v:.0%}'
        for v, e in zip(values, errors)
    ]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = labels,
        orientation = 'h',
        marker_color      = colors,
        marker_line_color = line_colors,
        marker_line_width = line_widths,
        error_x = dict(
            type      = 'data',
            array     = errors,
            visible   = any(e > 0 for e in errors),
            color     = '#555',
            thickness = 1.5,
            width     = 6,
        ),
        text        = text_labels,
        textposition= 'outside',
        hovertemplate='%{y}: %{x:.1%}<extra></extra>',
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 1.25], tickformat='.0%', showgrid=False),
        yaxis=dict(autorange='reversed'),
        margin=dict(l=10, r=60, t=10, b=10),
        height=height,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch", key=key)

    # Legend note if ground truth is shown
    if true_set:
        st.caption(
            f'🟦 Predicted  ·  🟩 Ground truth: {", ".join(sorted(true_set))}'
            '  ·  Error bars show ± stability under signal noise (20 passes)'
        )
    else:
        st.caption('Error bars show ± stability under signal noise (20 passes)')
