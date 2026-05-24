from __future__ import annotations
import streamlit as st
from pipeline.rag_pipeline import PipelineResult

CONDITIONS = {
    "All conditions":   None,
    "Type 2 Diabetes":  "type2_diabetes",
    "Hypertension":     "hypertension",
    "Asthma":           "asthma",
    "Heart Failure":    "heart_failure",
    "Depression":       "depression",
    "Chronic Kidney":   "chronic_kidney",
}

CONFIDENCE_COLORS = {
    "high":     "#22c55e",
    "moderate": "#f59e0b",
    "low":      "#ef4444",
}


def render_query_panel() -> tuple[str | None, dict | None]:
    st.markdown("### Ask a Clinical Question")
    st.markdown(
        "Query is grounded in PubMed abstracts. "
        "Every claim in the answer is cited with a PMID."
    )

    question = st.text_area(
        label="Question",
        placeholder="e.g. What are the cardiovascular benefits of SGLT2 inhibitors?",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        condition_label = st.selectbox(
            "Filter by condition",
            options=list(CONDITIONS.keys()),
        )
    with col2:
        k = st.slider("Chunks (k)", min_value=3, max_value=10, value=5)

    submitted = st.button("Search", type="primary", use_container_width=True)

    if submitted and question.strip():
        condition_tag = CONDITIONS[condition_label]
        where = (
            {"condition": {"$eq": condition_tag}}
            if condition_tag else None
        )
        return question.strip(), where, k

    return None, None, 5


def render_answer(result: PipelineResult) -> None:
    confidence     = result.confidence.lower()
    conf_color     = CONFIDENCE_COLORS.get(confidence, "#6b7280")
    conf_label     = confidence.upper()

    st.markdown("---")
    st.markdown("#### Answer")

    st.markdown(
        f'<span style="background:{conf_color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;">'
        f'Confidence: {conf_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"*{result.confidence_reason}*")
    st.markdown("")
    st.markdown(result.answer)

    if result.cited_pmids:
        st.markdown("#### Cited PMIDs")
        cols = st.columns(min(len(result.cited_pmids), 5))
        for i, pmid in enumerate(result.cited_pmids):
            with cols[i % 5]:
                st.markdown(
                    f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">'
                    f'<button style="width:100%;padding:6px;border:1px solid #374151;'
                    f'border-radius:6px;background:#1f2937;color:#93c5fd;cursor:pointer;'
                    f'font-size:0.8rem;">PMID {pmid}</button></a>',
                    unsafe_allow_html=True,
                )

    st.markdown("")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total latency", f"{result.total_latency_ms:.0f}ms")
    with col2:
        st.metric("Retrieval", f"{result.retrieval_meta['retrieval_latency_ms']:.0f}ms")
    with col3:
        st.metric("Generation", f"{result.generation_meta['generation_latency_ms']:.0f}ms")