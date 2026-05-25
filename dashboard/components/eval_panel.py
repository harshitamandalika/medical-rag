from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from evaluation.ragas_evaluator import RAGASResult

METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}

METRIC_DESCRIPTIONS = {
    "faithfulness":      "Are all claims grounded in retrieved chunks?",
    "answer_relevancy":  "Does the answer address the question?",
    "context_precision": "Are retrieved chunks precisely relevant?",
    "context_recall":    "Did retrieval surface all needed information?",
}


def _score_color(score: float | None) -> str:
    if score is None:
        return "#6b7280"
    if score >= 0.75:
        return "#22c55e"
    if score >= 0.5:
        return "#f59e0b"
    return "#ef4444"


def _gauge(label: str, value: float | None) -> go.Figure:
    display_value = value if value is not None else 0
    color         = _score_color(value)

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = display_value,
        number= {"suffix": "", "valueformat": ".2f", "font": {"size": 28, "color": color}},
        title = {"text": label, "font": {"size": 13, "color": "#9ca3af"}},
        gauge = {
            "axis":       {"range": [0, 1], "tickwidth": 1, "tickcolor": "#374151"},
            "bar":        {"color": color},
            "bgcolor":    "#1f2937",
            "bordercolor":"#374151",
            "steps": [
                {"range": [0,    0.5],  "color": "#111827"},
                {"range": [0.5,  0.75], "color": "#1f2937"},
                {"range": [0.75, 1.0],  "color": "#1f2937"},
            ],
            "threshold": {
                "line":  {"color": "#ffffff", "width": 2},
                "thickness": 0.75,
                "value": display_value,
            },
        },
    ))
    fig.update_layout(
        height          = 180,
        margin          = dict(l=20, r=20, t=40, b=10),
        paper_bgcolor   = "rgba(0,0,0,0)",
        plot_bgcolor    = "rgba(0,0,0,0)",
        font            = {"color": "#e5e7eb"},
    )
    return fig


def render_eval_panel(ragas_result: RAGASResult | None = None) -> None:
    st.markdown("### Eval Scores")
    st.markdown("RAGAS metrics evaluated on every query using Gemini as judge LLM.")

    scores = ragas_result.to_dict() if ragas_result else {k: None for k in METRIC_LABELS}

    col1, col2 = st.columns(2)
    metrics     = list(METRIC_LABELS.items())

    for i, (key, label) in enumerate(metrics):
        col = col1 if i % 2 == 0 else col2
        with col:
            value = scores.get(key)
            st.plotly_chart(
                _gauge(label, value),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            color = _score_color(value)
            score_str = f"{value:.4f}" if value is not None else "n/a"
            st.markdown(
                f'<div style="text-align:center;margin-top:-15px;margin-bottom:10px;">'
                f'<span style="background:{color}22;color:{color};padding:2px 10px;'
                f'border-radius:12px;font-size:0.8rem;font-weight:600;">{score_str}</span>'
                f'<br><span style="color:#6b7280;font-size:0.72rem;">'
                f'{METRIC_DESCRIPTIONS[key]}</span></div>',
                unsafe_allow_html=True,
            )

    if ragas_result:
        scores_list = [v for v in scores.values() if v is not None]
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            color = _score_color(avg)
            st.markdown("---")
            st.markdown(
                f'<div style="text-align:center;padding:12px;background:#1f2937;'
                f'border-radius:10px;border:1px solid #374151;">'
                f'<span style="color:#9ca3af;font-size:0.85rem;">Overall RAGAS Score</span><br>'
                f'<span style="color:{color};font-size:2rem;font-weight:700;">{avg:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )