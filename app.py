import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="QuantScribe",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Path + secrets bootstrap
# -------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    google_key = st.secrets.get("GOOGLE_API_KEY", None)
except Exception:
    google_key = None

if google_key:
    os.environ["GOOGLE_API_KEY"] = google_key

# -------------------------
# Imports from project
# -------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

from quantscribe.config import get_settings
from quantscribe.embeddings.pipeline import EmbeddingPipeline
from quantscribe.llm.extraction_chain import build_extraction_chain
from quantscribe.llm.peer_comparison import run_peer_comparison
from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(56,189,248,0.10), rgba(168,85,247,0.08));
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2.4rem;
    }
    .subtle {
        color: rgba(250,250,250,0.75);
        font-size: 0.98rem;
    }
    .card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.8rem 1rem;
        background: rgba(255,255,255,0.02);
    }
    .metric-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        background: rgba(255,255,255,0.02);
        min-height: 120px;
    }
    .small-label {
        color: rgba(250,250,250,0.70);
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .big-number {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .bank-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.30);
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Cached loaders
# -------------------------
@st.cache_resource
def load_resources():
    settings = get_settings()
    embedder = EmbeddingPipeline()
    extraction_chain = build_extraction_chain(max_retries=settings.llm_max_retries)
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.1,
        google_api_key=settings.google_api_key,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    return embedder, extraction_chain, llm


@st.cache_resource
def load_bank_indices(selected_banks: tuple, fiscal_year: str = "FY25", doc_type: str = "annual_report"):
    bank_indices = {}
    for bank in selected_banks:
        index_name = f"{bank}_{doc_type}_{fiscal_year}"
        idx = BankIndex(index_name)
        idx.load("indices/active")
        bank_indices[index_name] = idx
    return bank_indices


# -------------------------
# Helpers
# -------------------------
def get_available_indices(index_dir="indices/active"):
    path = Path(index_dir)
    if not path.exists():
        return []

    banks = set()
    for f in path.glob("*.faiss"):
        stem = f.stem
        for suffix in ["_annual_report", "_earnings_call", "_investor_presentation"]:
            if suffix in stem:
                banks.add(stem.split(suffix)[0])
                break
    return sorted(banks)


def format_bank_pills(banks: List[str]) -> str:
    return "".join([f"<span class='bank-pill'>{b}</span>" for b in banks])


def build_custom_context(results_by_bank: Dict[str, List[dict]]) -> str:
    sections = []

    for bank, results in results_by_bank.items():
        chunks = []
        for r in results:
            meta = r["metadata"]
            content = meta.get("content", "").strip()
            if not content:
                continue

            chunks.append(
                f"""[BANK: {bank}]
[PAGE: {meta.get('page_number', 'N/A')}]
[SECTION: {meta.get('section_header', 'N/A')}]
[RELEVANCE: {r.get('score', 0.0):.3f}]
{content}"""
            )

        if chunks:
            bank_block = "\n\n---\n\n".join(chunks)
            sections.append(f"===== {bank} =====\n{bank_block}")

    return "\n\n\n".join(sections)


def run_custom_qa(question: str, retriever: PeerGroupRetriever, embedder: EmbeddingPipeline, llm, banks: List[str], top_k: int = 5):
    query_vector = embedder.embed_query(question)
    retrieved = retriever.retrieve(query_vector, banks, top_k_per_bank=top_k)
    context = build_custom_context(retrieved)

    prompt = f"""
You are a banking research assistant.
Answer the user's question using ONLY the context below.

Rules:
1. Do not invent facts.
2. If the answer is incomplete, say so clearly.
3. Compare banks where possible.
4. Cite evidence inline like [HDFC_BANK p.229].
5. After the answer, provide a short 'Evidence Used' bullet list.

User question:
{question}

Context:
{context}
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, retrieved


def render_retrieval_table(retrieved: Dict[str, List[dict]]):
    rows = []
    for bank, results in retrieved.items():
        for r in results:
            meta = r["metadata"]
            rows.append(
                {
                    "Bank": bank,
                    "Page": meta.get("page_number"),
                    "Section": meta.get("section_header"),
                    "Type": meta.get("content_type"),
                    "Score": round(r.get("score", 0.0), 3),
                    "Preview": (meta.get("content", "")[:180] + "...") if meta.get("content") else "",
                }
            )

    if rows:
        df = pd.DataFrame(rows).sort_values(["Bank", "Score"], ascending=[True, False])
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_bank_detail(ext):
    metrics_rows = []
    for m in ext.extracted_metrics:
        value = m.metric_value if m.metric_value is not None else m.qualitative_value
        metrics_rows.append(
            {
                "Metric": m.metric_name.replace("_", " ").title(),
                "Value": f"{value} {m.metric_unit or ''}".strip(),
                "Confidence": m.confidence.title(),
                "Page": m.citation.page_number,
                "Section": m.citation.section_header or "—",
            }
        )

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.markdown("<div class='metric-card'><div class='small-label'>Risk score</div>"
                    f"<div class='big-number'>{ext.risk_score:.1f}/10</div>"
                    f"<div>{ext.risk_rating.replace('_', ' ').title()}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='small-label'>Sentiment</div>"
                    f"<div class='big-number'>{ext.sentiment_score:.2f}</div>"
                    "<div>Disclosure tone</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><div class='small-label'>Executive summary</div>", unsafe_allow_html=True)
        st.write(ext.summary)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Extracted metrics")
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    with st.expander("Source excerpts"):
        for m in ext.extracted_metrics:
            st.markdown(
                f"**{m.metric_name}** · page {m.citation.page_number}  \n"
                f"{m.citation.source_excerpt}"
            )


# -------------------------
# Sidebar
# -------------------------
available_banks = get_available_indices()

with st.sidebar:
    st.markdown("## 🔎 QuantScribe")
    st.caption("RAG over indexed annual reports")

    mode = st.radio(
        "Mode",
        ["Theme Comparison", "Ask Reports"],
        index=0,
        horizontal=False,
    )

    selected_banks = st.multiselect(
        "Banks",
        available_banks,
        default=available_banks[:2] if len(available_banks) >= 2 else available_banks,
    )

    top_k = st.slider("Chunks per bank", 3, 10, 5)

    fiscal_year = st.selectbox("Fiscal year", ["FY25"], index=0)

    if mode == "Theme Comparison":
        selected_theme = st.selectbox(
            "Theme",
            [
                "credit_risk",
                "liquidity_risk",
                "unsecured_lending",
                "capital_adequacy",
                "market_risk",
                "operational_risk",
                "asset_quality_trend",
            ],
        )
    else:
        selected_theme = None

    st.divider()
    if st.button("Clear cache"):
        st.cache_resource.clear()
        st.rerun()

# -------------------------
# Main
# -------------------------
embedder, extraction_chain, llm = load_resources()

st.markdown(
    f"""
    <div class="hero">
        <h1>📊 QuantScribe</h1>
        <div class="subtle">Search, compare, and interrogate bank annual reports with retrieval-backed answers.</div>
        <div style="margin-top:0.8rem;">{format_bank_pills(selected_banks)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not selected_banks:
    st.warning("Select at least one bank to begin.")
    st.stop()

bank_indices = load_bank_indices(tuple(selected_banks), fiscal_year=fiscal_year)
retriever = PeerGroupRetriever(bank_indices)

if mode == "Theme Comparison":
    left, right = st.columns([2.4, 1])
    with left:
        st.markdown("### Thematic peer comparison")
        st.caption("Structured extraction for a predefined macro theme.")
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Theme**: `{selected_theme}`")
        st.markdown(f"**Banks**: {len(selected_banks)}")
        st.markdown(f"**Top-K**: {top_k}")
        st.markdown("</div>", unsafe_allow_html=True)

    if len(selected_banks) < 2:
        st.info("Theme comparison needs at least two banks.")
        st.stop()

    if st.button("Generate comparison", type="primary", use_container_width=False):
        with st.spinner("Running retrieval and thematic extraction..."):
            report = run_peer_comparison(
                theme=selected_theme,
                peer_group=selected_banks,
                retriever=retriever,
                embedding_pipeline=embedder,
                extraction_chain=extraction_chain,
                top_k_per_bank=top_k,
            )

        rank_df = pd.DataFrame([r.model_dump() for r in report.peer_ranking]).sort_values("risk_score", ascending=True)

        a, b, c = st.columns(3)
        with a:
            st.markdown("<div class='metric-card'><div class='small-label'>Best positioned</div>"
                        f"<div class='big-number'>{rank_df.iloc[0]['bank']}</div>"
                        f"<div>Risk score {rank_df.iloc[0]['risk_score']:.1f}</div></div>", unsafe_allow_html=True)
        with b:
            avg_score = rank_df["risk_score"].mean()
            st.markdown("<div class='metric-card'><div class='small-label'>Average risk score</div>"
                        f"<div class='big-number'>{avg_score:.2f}</div>"
                        "<div>Across selected banks</div></div>", unsafe_allow_html=True)
        with c:
            st.markdown("<div class='metric-card'><div class='small-label'>Banks compared</div>"
                        f"<div class='big-number'>{len(rank_df)}</div>"
                        f"<div>Theme: {selected_theme}</div></div>", unsafe_allow_html=True)

        st.markdown("### Peer ranking")
        fig = px.bar(
            rank_df,
            x="risk_score",
            y="bank",
            orientation="h",
            color="risk_score",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 10],
            text="risk_score",
            height=360,
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Risk score (lower is better)",
            yaxis_title="",
            coloraxis_colorbar_title="Score",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Cross-cutting insights")
        st.success(report.cross_cutting_insights)

        st.markdown("### Bank detail")
        for ext in sorted(report.extractions, key=lambda x: x.risk_score):
            with st.expander(f"{ext.bank_name} · score {ext.risk_score:.1f} · {ext.risk_rating.replace('_', ' ').title()}", expanded=False):
                render_bank_detail(ext)

else:
    st.markdown("### Ask the reports")
    st.caption("Free-text RAG over selected banks using the existing FAISS indices.")

    question = st.text_area(
        "Ask a question",
        placeholder="Example: Compare HDFC and SBI on GNPA, NNPA, and provisioning. Mention page numbers.",
        height=120,
    )

    q1, q2, q3 = st.columns([1.2, 1, 3])
    with q1:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with q2:
        show_evidence = st.checkbox("Show retrieved evidence", value=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
            st.stop()

        with st.spinner("Retrieving relevant chunks and generating answer..."):
            answer, retrieved = run_custom_qa(
                question=question,
                retriever=retriever,
                embedder=embedder,
                llm=llm,
                banks=selected_banks,
                top_k=top_k,
            )

        st.markdown("### Answer")
        st.markdown(f"<div class='card'>{answer}</div>", unsafe_allow_html=True)

        if show_evidence:
            st.markdown("### Retrieved evidence")
            render_retrieval_table(retrieved)