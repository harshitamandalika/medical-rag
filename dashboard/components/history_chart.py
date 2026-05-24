from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from evaluation.mlflow_logger import fetch_metric_history

METRICS = {
    "faithfulness":      "#22c55e",
    "answer_relevancy":  "#3b82f6",
    "context_precision": "#f59e0b",
    "context_recall":    "#a855f7",
}


def render_history_chart(n: int = 30) -> None:
    st.markdown("### Historical Trend")
    st.markdown(f"Last {n} queries logged to MLflow.")

    traces = []
    has_data = False

    for metric, color in METRICS.items():
        history = fetch_metric_history(metric, n=n)
        if not history:
            continue
        has_data = True
        traces.append(go.Scatter(
            x    = list(range(1, len(history) + 1)),
            y    = [h["value"] for h in history],
            name = metric.replace("_", " ").title(),
            mode = "lines+markers",
            line = {"color": color, "width": 2},
            marker={"size": 6, "color": color},
            hovertemplate=(
                "<b>%{text}</b><br>"
                + metric.replace("_", " ").title()
                + ": %{y:.4f}<extra></extra>"
            ),
            text=[h["question"][:60] + "..." for h in history],
        ))

    if not has_data:
        st.info("No historical data yet. Run queries to populate the trend chart.")
        return

    fig = go.Figure(traces)
    fig.update_layout(
        height        = 300,
        margin        = dict(l=10, r=10, t=20, b=10),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        legend        = {"font": {"color": "#9ca3af"}, "bgcolor": "rgba(0,0,0,0)"},
        xaxis         = {
            "title":      "Query number",
            "gridcolor":  "#1f2937",
            "tickcolor":  "#374151",
            "color":      "#6b7280",
        },
        yaxis         = {
            "title":      "Score",
            "range":      [0, 1],
            "gridcolor":  "#1f2937",
            "tickcolor":  "#374151",
            "color":      "#6b7280",
        },
        font={"color": "#e5e7eb"},
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})