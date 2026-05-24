from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title = "Medical RAG Eval",
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0f1a;
        color: #e5e7eb;
    }
    .stApp {
        background-color: #0a0f1a;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
        max-width: 100%;
    }
    h1, h2, h3, h4 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
    }
    .stTextArea textarea {
        background-color: #111827;
        color: #e5e7eb;
        border: 1px solid #374151;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
    .stButton button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        transition: background-color 0.2s;
    }
    .stButton button:hover {
        background-color: #1d4ed8;
    }
    .stSelectbox > div > div {
        background-color: #111827;
        border: 1px solid #374151;
        color: #e5e7eb;
    }
    .stSlider > div {
        color: #e5e7eb;
    }
    .stMetric {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.75rem;
    }
    .stMetric label {
        color: #6b7280;
        font-size: 0.8rem;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #e5e7eb;
        font-family: 'DM Mono', monospace;
        font-size: 1.1rem;
    }
    hr {
        border-color: #1f2937;
    }
    .stSpinner > div {
        border-top-color: #3b82f6;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1117;
    }
    .stAlert {
        background-color: #111827;
        border: 1px solid #374151;
        color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="border-bottom:1px solid #1f2937;padding-bottom:1rem;margin-bottom:1.5rem;">
    <h1 style="margin:0;font-size:1.6rem;color:#f9fafb;">
        Medical Literature RAG
        <span style="font-size:0.9rem;font-weight:400;color:#6b7280;margin-left:12px;">
            PubMed · Hybrid Retrieval · RAGAS Eval
        </span>
    </h1>
    <p style="margin:0.25rem 0 0;color:#6b7280;font-size:0.85rem;">
        Answers grounded in PubMed abstracts. Every claim cited with a PMID. Every query evaluated with RAGAS.
    </p>
</div>
""", unsafe_allow_html=True)


if "pipeline" not in st.session_state:
    with st.spinner("Loading pipeline - this takes about 2 seconds on first run..."):
        from pipeline.rag_pipeline import RAGPipeline
        st.session_state.pipeline = RAGPipeline(k=5)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_ragas" not in st.session_state:
    st.session_state.last_ragas = None


from dashboard.components.query_panel  import render_query_panel, render_answer
from dashboard.components.eval_panel   import render_eval_panel
from dashboard.components.history_chart import render_history_chart
from evaluation.ragas_evaluator        import evaluate_result
from evaluation.mlflow_logger          import log_run

left, right = st.columns([6, 4], gap="large")

with left:
    question, where, k = render_query_panel()

    if question:
        pipeline = st.session_state.pipeline
        pipeline.k = k

        with st.spinner("Retrieving and generating..."):
            result = pipeline.run(question, where=where)
            st.session_state.last_result = result

        render_answer(result)

        with st.spinner("Running RAGAS eval..."):
            ragas_result = evaluate_result(result)
            st.session_state.last_ragas = ragas_result

        log_run(result, ragas_result=ragas_result, source="live")
        st.rerun()

    elif st.session_state.last_result:
        render_answer(st.session_state.last_result)

with right:
    render_eval_panel(st.session_state.last_ragas)
    st.markdown("")
    render_history_chart(n=30)