"""
QuantScribe — Streamlit Dashboard

RAG-backed thematic peer analysis for Indian BFSI sector.
Two modes:
  1. Theme Comparison — structured extraction + peer ranking
  2. Ask Reports — free-text RAG Q&A over indexed annual reports
"""

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

# ── Path + secrets bootstrap ──
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    google_key = st.secrets.get("GOOGLE_API_KEY", None)
except Exception:
    google_key = None

if google_key:
    os.environ["GOOGLE_API_KEY"] = google_key

# ── Project imports ──
from langchain_google_genai import ChatGoogleGenerativeAI

from quantscribe.config import get_settings
from quantscribe.embeddings.pipeline import EmbeddingPipeline
from quantscribe.llm.extraction_chain import build_extraction_chain
from quantscribe.llm.peer_comparison import run_peer_comparison
from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever


# ═══════════════════════════════════════════════════════
# SESSION STATE — persist results across reruns
# ═══════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "report": None,
        "report_theme": None,
        "report_banks": None,
        "qa_answer": None,
        "qa_retrieved": None,
        "qa_question": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ═══════════════════════════════════════════════════════
# CSS — uses color-mix() so it works on both light & dark
# ═══════════════════════════════════════════════════════

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

.hero {
    padding: 1.4rem 1.6rem;
    border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
    border-radius: 16px;
    background: color-mix(in srgb, #38bdf8 6%, transparent);
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0 0 0.3rem 0; font-size: 2.2rem; }
.hero .sub { opacity: 0.7; font-size: 0.95rem; }

.card {
    border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    background: color-mix(in srgb, currentColor 3%, transparent);
    margin-bottom: 0.5rem;
}

.metric-card {
    border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
    border-radius: 14px;
    padding: 1rem;
    background: color-mix(in srgb, currentColor 3%, transparent);
    min-height: 110px;
}
.metric-card .label { opacity: 0.6; font-size: 0.82rem; margin-bottom: 0.3rem; }
.metric-card .value { font-size: 2rem; font-weight: 700; line-height: 1.15; }
.metric-card .detail { opacity: 0.65; font-size: 0.85rem; margin-top: 0.15rem; }

.bank-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border-radius: 999px;
    font-size: 0.82rem;
    background: color-mix(in srgb, #3b82f6 12%, transparent);
    border: 1px solid color-mix(in srgb, #3b82f6 25%, transparent);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# CACHED LOADERS — lazy, only load what's needed
# ═══════════════════════════════════════════════════════

@st.cache_resource
def load_embedder():
    return EmbeddingPipeline()


@st.cache_resource
def load_extraction_chain():
    settings = get_settings()
    return build_extraction_chain(max_retries=settings.llm_max_retries)


@st.cache_resource
def load_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.1,
        google_api_key=settings.google_api_key,
        max_output_tokens=settings.llm_max_output_tokens,
    )


@st.cache_resource
def load_bank_indices(selected_banks: tuple, fiscal_year: str = "FY25", doc_type: str = "annual_report"):
    bank_indices = {}
    errors = []
    for bank in selected_banks:
        index_name = f"{bank}_{doc_type}_{fiscal_year}"
        idx = BankIndex(index_name)
        try:
            idx.load("indices/active")
            bank_indices[index_name] = idx
        except FileNotFoundError:
            errors.append(f"{bank}: index not found")
        except Exception as e:
            errors.append(f"{bank}: {str(e)[:80]}")
    return bank_indices, errors


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def get_available_indices(index_dir: str = "indices/active") -> list[str]:
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


def bank_pills_html(banks: list[str]) -> str:
    return "".join(f"<span class='bank-pill'>{b}</span>" for b in banks)


def build_qa_context(results_by_bank: Dict[str, List[dict]]) -> str:
    sections = []
    for bank, results in results_by_bank.items():
        chunks = []
        for r in results:
            meta = r["metadata"]
            content = meta.get("content", "").strip()
            if not content:
                continue
            chunks.append(
                f"[BANK: {bank}] [PAGE: {meta.get('page_number', '?')}] "
                f"[SECTION: {meta.get('section_header', '?')}]\n{content}"
            )
        if chunks:
            sections.append(f"===== {bank} =====\n" + "\n\n---\n\n".join(chunks))
    return "\n\n\n".join(sections)


def run_custom_qa(question, retriever, embedder, llm, banks, top_k=5):
    query_vector = embedder.embed_query(question)
    retrieved = retriever.retrieve(query_vector, banks, top_k_per_bank=top_k)
    context = build_qa_context(retrieved)

    prompt = f"""You are a banking research assistant.
Answer the user's question using ONLY the context below.

Rules:
1. Do not invent facts. If the answer is incomplete, say so.
2. Compare banks where possible.
3. Cite evidence inline like [HDFC_BANK p.229].
4. End with a short 'Evidence Used' bullet list.

Question: {question}

Context:
{context}"""

    response = llm.invoke(prompt)
    return (response.content if hasattr(response, "content") else str(response)), retrieved


# ═══════════════════════════════════════════════════════
# RENDERERS
# ═══════════════════════════════════════════════════════

def render_retrieval_table(retrieved):
    rows = []
    for bank, results in retrieved.items():
        for r in results:
            meta = r["metadata"]
            rows.append({
                "Bank": bank,
                "Page": meta.get("page_number"),
                "Section": meta.get("section_header", "—"),
                "Type": meta.get("content_type", "—"),
                "Score": round(r.get("score", 0.0), 3),
                "Preview": (meta.get("content", "")[:150] + "…") if meta.get("content") else "",
            })
    if rows:
        df = pd.DataFrame(rows).sort_values(["Bank", "Score"], ascending=[True, False])
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_bank_detail(ext):
    metrics_rows = []
    for m in ext.extracted_metrics:
        value = m.metric_value if m.metric_value is not None else m.qualitative_value
        metrics_rows.append({
            "Metric": m.metric_name.replace("_", " ").title(),
            "Value": f"{value} {m.metric_unit or ''}".strip(),
            "Confidence": m.confidence.title(),
            "Page": m.citation.page_number,
            "Section": m.citation.section_header or "—",
        })

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Risk score</div>"
            f"<div class='value'>{ext.risk_score:.1f}/10</div>"
            f"<div class='detail'>{ext.risk_rating.replace('_', ' ').title()}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Sentiment</div>"
            f"<div class='value'>{ext.sentiment_score:+.2f}</div>"
            f"<div class='detail'>Disclosure tone</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.caption("Executive summary")
        st.write(ext.summary)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Extracted metrics**")
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    with st.expander("Source excerpts"):
        for m in ext.extracted_metrics:
            st.markdown(
                f"**{m.metric_name}** · page {m.citation.page_number}  \n"
                f"_{m.citation.source_excerpt}_"
            )


def render_report(report):
    rank_df = pd.DataFrame(
        [r.model_dump() for r in report.peer_ranking]
    ).sort_values("risk_score", ascending=True)

    a, b, c = st.columns(3)
    with a:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Best positioned</div>"
            f"<div class='value'>{rank_df.iloc[0]['bank']}</div>"
            f"<div class='detail'>Risk score {rank_df.iloc[0]['risk_score']:.1f}</div></div>",
            unsafe_allow_html=True,
        )
    with b:
        avg = rank_df["risk_score"].mean()
        st.markdown(
            f"<div class='metric-card'><div class='label'>Average risk score</div>"
            f"<div class='value'>{avg:.2f}</div>"
            f"<div class='detail'>Across peer group</div></div>",
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Banks compared</div>"
            f"<div class='value'>{len(rank_df)}</div>"
            f"<div class='detail'>Theme: {report.query_theme}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Peer ranking")
    fig = px.bar(
        rank_df, x="risk_score", y="bank", orientation="h",
        color="risk_score", color_continuous_scale="RdYlGn_r",
        range_color=[0, 10], text="risk_score",
        height=max(200, len(rank_df) * 80),
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="Risk score (lower = better)",
        yaxis_title="",
        coloraxis_colorbar_title="Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cross-cutting insights")
    st.success(report.cross_cutting_insights)

    st.markdown("### Bank detail")
    for ext in sorted(report.extractions, key=lambda x: x.risk_score):
        label = (
            f"{ext.bank_name} · score {ext.risk_score:.1f} · "
            f"{ext.risk_rating.replace('_', ' ').title()}"
        )
        with st.expander(label, expanded=False):
            render_bank_detail(ext)


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════

available_banks = get_available_indices()

with st.sidebar:
    st.markdown("## 📊 QuantScribe")
    st.caption("RAG-backed peer analysis")

    mode = st.radio("Mode", ["Theme Comparison", "Ask Reports"], horizontal=False)

    selected_banks = st.multiselect(
        "Banks", available_banks,
        default=available_banks[:2] if len(available_banks) >= 2 else available_banks,
    )

    top_k = st.slider("Chunks per bank", 3, 15, 5)
    fiscal_year = st.selectbox("Fiscal year", ["FY25"], index=0)

    if mode == "Theme Comparison":
        selected_theme = st.selectbox(
            "Theme",
            ["credit_risk", "capital_adequacy", "liquidity_risk",
             "unsecured_lending", "market_risk", "operational_risk",
             "asset_quality_trend"],
        )
    else:
        selected_theme = None

    st.divider()
    if st.button("Reset everything"):
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ═══════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════

st.markdown(
    f"""<div class="hero">
    <h1>📊 QuantScribe</h1>
    <div class="sub">Search, compare, and interrogate bank annual reports with retrieval-backed answers.</div>
    <div style="margin-top:0.6rem;">{bank_pills_html(selected_banks)}</div>
    </div>""",
    unsafe_allow_html=True,
)

if not selected_banks:
    st.warning("Select at least one bank in the sidebar to begin.")
    st.stop()


# ═══════════════════════════════════════════════════════
# LOAD INDICES
# ═══════════════════════════════════════════════════════

bank_indices, load_errors = load_bank_indices(tuple(selected_banks), fiscal_year=fiscal_year)

if load_errors:
    for err in load_errors:
        st.error(f"Index load failed: {err}")

if not bank_indices:
    st.error("No FAISS indices loaded. Place `.faiss` + `_metadata.json` files in `indices/active/`.")
    st.stop()

retriever = PeerGroupRetriever(bank_indices)

with st.sidebar:
    st.divider()
    st.caption("Loaded indices")
    for name, idx in bank_indices.items():
        st.caption(f"  {name}: {idx.size:,} vectors")


# ═══════════════════════════════════════════════════════
# MODE: THEME COMPARISON
# ═══════════════════════════════════════════════════════

if mode == "Theme Comparison":
    left, right = st.columns([2.5, 1])
    with left:
        st.markdown("### Thematic peer comparison")
        st.caption("Structured extraction with risk scoring and citations.")
    with right:
        st.markdown(
            f"<div class='card'>"
            f"<b>Theme</b>: <code>{selected_theme}</code><br>"
            f"<b>Banks</b>: {len(selected_banks)}<br>"
            f"<b>Top-K</b>: {top_k}</div>",
            unsafe_allow_html=True,
        )

    if len(selected_banks) < 2:
        st.info("Theme comparison requires at least two banks.")
        st.stop()

    if st.button("Generate comparison", type="primary"):
        embedder = load_embedder()
        extraction_chain = load_extraction_chain()
        progress = st.progress(0, text="Embedding query...")

        try:
            progress.progress(15, text="Retrieving chunks from FAISS...")

            report = run_peer_comparison(
                theme=selected_theme,
                peer_group=selected_banks,
                retriever=retriever,
                embedding_pipeline=embedder,
                extraction_chain=extraction_chain,
                top_k_per_bank=top_k,
            )
            progress.progress(100, text="Done!")

            st.session_state.report = report
            st.session_state.report_theme = selected_theme
            st.session_state.report_banks = selected_banks

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)[:300]}")
            st.session_state.report = None

    report = st.session_state.report
    if report is not None:
        if (st.session_state.report_theme != selected_theme
                or st.session_state.report_banks != selected_banks):
            st.info(
                f"Showing cached results for **{st.session_state.report_theme}** "
                f"({', '.join(st.session_state.report_banks)}). "
                f"Click 'Generate comparison' to update."
            )
        render_report(report)
    else:
        st.markdown(
            "<div style='text-align:center; padding:3rem; opacity:0.4;'>"
            "Select a theme and click <b>Generate comparison</b> to start.</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════
# MODE: ASK REPORTS
# ═══════════════════════════════════════════════════════

elif mode == "Ask Reports":
    st.markdown("### Ask the reports")
    st.caption("Free-text RAG over selected banks.")

    question = st.text_area(
        "Your question",
        placeholder="Example: Compare HDFC and SBI on GNPA, NNPA, and provisioning. Cite page numbers.",
        height=110,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    if ask_clicked and question.strip():
        embedder = load_embedder()
        llm = load_llm()

        with st.spinner("Retrieving chunks and generating answer..."):
            try:
                answer, retrieved = run_custom_qa(
                    question=question, retriever=retriever,
                    embedder=embedder, llm=llm,
                    banks=selected_banks, top_k=top_k,
                )
                st.session_state.qa_answer = answer
                st.session_state.qa_retrieved = retrieved
                st.session_state.qa_question = question
            except Exception as e:
                st.error(f"Failed: {str(e)[:200]}")
                st.session_state.qa_answer = None
    elif ask_clicked:
        st.warning("Enter a question first.")

    if st.session_state.qa_answer:
        st.markdown("---")
        st.markdown("### Answer")
        st.markdown(f"<div class='card'>{st.session_state.qa_answer}</div>", unsafe_allow_html=True)

        show_evidence = st.checkbox("Show retrieved evidence", value=False)
        if show_evidence and st.session_state.qa_retrieved:
            st.markdown("### Retrieved chunks")
            render_retrieval_table(st.session_state.qa_retrieved)
