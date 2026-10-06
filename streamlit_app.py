"""
ResearchLens — Streamlit Community Cloud Edition
=================================================
Self-contained Streamlit application for autonomous academic literature synthesis,
PDF passage extraction, neural NLI claim verification, and citation integrity audits.

Reuses the core async pipeline from backend/services/research_workflow.py without
altering FastAPI or React behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# ── Ensure Project Root is on sys.path ─────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import psutil
import streamlit as st

# ── Defensive Compatibility Patch for torch.accelerator ─────────────────────────
# In torch < 2.6, torch.accelerator does not exist. Transformers 4.49+ calls
# torch.accelerator.current_accelerator() during import, causing an AttributeError.
import torch
if not hasattr(torch, "accelerator"):
    class _DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return None
    torch.accelerator = _DummyAccelerator()

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("researchlens.streamlit")

# ── 1. Map st.secrets to os.environ BEFORE importing backend.config ───────────
# This ensures that pydantic-settings picks up Streamlit Cloud secrets on startup.
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ[k] = str(v)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    os.environ[sub_k] = str(sub_v)
except Exception as e:
    logger.debug("st.secrets unavailable or empty: %s", e)

# ── 2. Memory & Resource-Conscious Streamlit Cloud Defaults ───────────────────
# Community Cloud provides ~2 CPU cores and ~2.7 GB RAM. We configure:
# - CPU device for PyTorch and NLI
# - Fast startup (lazy model loading on demand)
# - Lightweight DeBERTa-v3-small and BGE-small embeddings
# - Single-threaded PyTorch to prevent CPU thread thrashing and memory bloat
os.environ.setdefault("NLI_DEVICE", "cpu")
os.environ.setdefault("SKIP_MODEL_WARMUP", "true")
os.environ.setdefault("MAX_PDF_WORKERS", "2")
os.environ.setdefault("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Pick LLM provider: prioritize Gemini on Cloud if key is set, otherwise OpenAI
if os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here":
    os.environ.setdefault("LLM_PROVIDER", "gemini")
elif os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here":
    os.environ.setdefault("LLM_PROVIDER", "openai")
else:
    os.environ.setdefault("LLM_PROVIDER", "gemini")

# Writable SQLite DB path (works locally and in ephemeral cloud container)
_db_file = os.getenv("STREAMLIT_DB_PATH", str(_PROJECT_ROOT / "researchlens_st.db"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{Path(_db_file).resolve()}")

# Enforce 1 PyTorch thread for low-resource environments
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# ── 3. Import Backend Configuration & Database ────────────────────────────────
from backend.config import settings
from backend.models.database import PaperRecord, ResearchJob, SessionLocal, create_tables
from backend.models.schemas import (
    Citation,
    CitationStatus,
    ResearchInvestigation,
)
from backend.services.research_workflow import (
    _make_pipeline,
    job_to_investigation,
    run_research_pipeline,
)

# Apply conservative depth counts for Streamlit Cloud
settings.DEPTH_COUNTS = {"Quick": 3, "Standard": 6, "Deep": 12}
settings.MAX_PDF_WORKERS = 2

# ── 4. Concurrency Guard & Model Caching ──────────────────────────────────────
# Guard against concurrent runs: never execute more than one pipeline in memory
_PIPELINE_LOCK = threading.Lock()


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model():
    """Lazy-load the Sentence Transformers embedding model."""
    from backend.services.embeddings import get_model
    return get_model()


@st.cache_resource(show_spinner=False)
def get_cached_reranker():
    """Lazy-load the Cross-Encoder reranker."""
    from backend.services.ranker import get_reranker
    return get_reranker()


@st.cache_resource(show_spinner=False)
def get_cached_nli_components():
    """Lazy-load the DeBERTa-v3 NLI model on CPU."""
    from backend.services.nli_verifier import get_nli_components
    return get_nli_components()


def init_database() -> None:
    """Initialize SQLite tables and recover any stale in_progress jobs."""
    create_tables()
    db = SessionLocal()
    try:
        stale = db.query(ResearchJob).filter(ResearchJob.status == "in_progress").all()
        for j in stale:
            j.status = "failed"
            j.failure_reason = "Application restarted while this job was executing. Please retry."
        if stale:
            db.commit()
    finally:
        db.close()


init_database()


# ── 5. Helper Functions ────────────────────────────────────────────────────────

def get_current_memory_mb() -> float:
    """Return process RSS memory usage in MB."""
    try:
        proc = psutil.Process()
        return round(proc.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def render_verdict_badge(verdict: str) -> str:
    """Return styled HTML badge for NLI verdicts."""
    v = str(verdict).lower()
    if v in ("entails", "verified"):
        return '<span style="background-color:#065F46;color:#D1FAE5;padding:3px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">✓ ENTAILS</span>'
    elif v in ("neutral", "partially_supported"):
        return '<span style="background-color:#92400E;color:#FEF3C7;padding:3px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">~ NEUTRAL</span>'
    elif v in ("contradicts", "contradicted"):
        return '<span style="background-color:#991B1B;color:#FEE2E2;padding:3px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">✗ CONTRADICTS</span>'
    else:
        return '<span style="background-color:#4B5563;color:#F3F4F6;padding:3px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">? UNSUPPORTED</span>'


def build_markdown_report(inv: ResearchInvestigation) -> str:
    """Construct a clean, exportable Markdown document from investigation report."""
    if not inv.report:
        return f"# ResearchLens Investigation\n\n**Status:** {inv.status}\n\n**Question:** {inv.question}"

    rep = inv.report
    lines = [
        f"# Research Report: {inv.question}",
        "",
        f"- **Date**: {inv.createdAt[:10] if inv.createdAt else datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"- **Status**: `{inv.status.upper()}`",
        f"- **Research Confidence**: `{inv.research_confidence}`",
        f"- **Citation Integrity**: `{inv.citation_integrity}%`",
        f"- **Evidence Coverage**: `{inv.evidence_coverage}%`",
        f"- **Papers Analyzed**: {inv.papersAnalyzed}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        rep.executiveSummary or "No summary available.",
        "",
    ]

    if rep.comparisonTable:
        lines.extend([
            "## Methodological Comparison",
            "",
            "| Model / Architecture | Dataset | Performance / Metric | Venue / Year | Citations |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for row in rep.comparisonTable:
            perf = row.f1Score if row.f1Score != "Not extracted" else row.mapScore
            lines.append(
                f"| **{row.model}** | {row.dataset} | {perf} | {row.venue or ''} {row.year or ''} | {row.citationCount or ''} |"
            )
        lines.append("")

    if rep.findings:
        lines.append("## Detailed Research Findings")
        lines.append("")
        for sec in rep.findings:
            lines.append(f"### {sec.sectionTitle}")
            for para in sec.paragraphs:
                lines.append(para.text)
                lines.append("")

    if rep.limitations:
        lines.append("## Limitations & Boundary Conditions")
        lines.append("")
        for lim in rep.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    if rep.conclusion:
        lines.append("## Conclusion")
        lines.append("")
        lines.append(rep.conclusion)
        lines.append("")

    if rep.references:
        lines.append("## References")
        lines.append("")
        for idx, p in enumerate(rep.references, 1):
            authors_str = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            doi_link = f" [DOI](https://doi.org/{p.doi})" if p.doi else ""
            pdf_link = f" [PDF]({p.pdfUrl})" if p.pdfUrl else ""
            lines.append(f"{idx}. **{p.title}** ({p.publicationYear}). {authors_str}. *{p.journalConference or p.source}*.{doi_link}{pdf_link}")
        lines.append("")

    return "\n".join(lines)


# ── 6. Robust Pipeline Execution Thread ───────────────────────────────────────
# Streamlit Execution Strategy:
# - Streamlit script reruns from top-to-bottom on any widget change.
# - Running heavy async AI pipelines directly on Streamlit's main script thread
#   freezes the browser websocket and risks rerun interruptions.
# - Running the pipeline in a background thread protected by _PIPELINE_LOCK allows:
#   1. Clean async execution of run_research_pipeline in a dedicated event loop.
#   2. Main thread polls SQLite state using st.status and a 0.8s sleep,
#      reflecting pipeline steps live to the user.
#   3. If the pipeline encounters an error or triggers the evidence gate,
#      the exact status and failure reason are preserved and shown honestly.
def _run_pipeline_worker(job_id: str) -> None:
    """Worker function executed in background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    db = SessionLocal()
    try:
        logger.info("[Streamlit Worker] Starting job %s", job_id)
        loop.run_until_complete(run_research_pipeline(job_id, db))
        logger.info("[Streamlit Worker] Finished job %s", job_id)
    except Exception as exc:
        logger.exception("[Streamlit Worker] Pipeline failed for job %s: %s", job_id, exc)
        job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.failure_reason = f"Unhandled error: {exc}"
            db.commit()
    finally:
        db.close()
        loop.close()


# ── 7. Page Setup & Sidebar ───────────────────────────────────────────────────

st.set_page_config(
    page_title="ResearchLens — Autonomous Academic Synthesis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ResearchLens")
    st.caption("Autonomous Academic Research Synthesis")
    st.divider()

    st.subheader("⚙️ Investigation Parameters")
    depth_choice = st.radio(
        "Research Depth",
        options=["Quick", "Standard", "Deep"],
        index=0,
        help="Quick retrieves 3-4 papers (recommended for Community Cloud). Deep analyzes up to 12 papers.",
    )

    sources_selected = st.multiselect(
        "Academic Sources",
        options=["arXiv", "Semantic Scholar", "OpenAlex"],
        default=["arXiv", "Semantic Scholar", "OpenAlex"],
        help="Select academic repositories for concurrent paper retrieval.",
    )

    st.divider()

    # Read-only active system configuration
    st.subheader("🤖 Engine Configuration")
    active_provider = settings.LLM_PROVIDER.title()
    active_model = settings.GEMINI_MODEL if settings.LLM_PROVIDER.lower() == "gemini" else settings.OPENAI_MODEL
    llm_ready = settings.is_llm_configured

    st.markdown(f"**LLM Provider:** `{active_provider}` ({active_model})")
    if llm_ready:
        st.markdown("**LLM Status:** 🟢 `Ready`")
    else:
        st.markdown("**LLM Status:** 🔴 `Missing API Key`")

    st.markdown(f"**NLI Model:** `{settings.NLI_MODEL.split('/')[-1]}`")
    st.markdown(f"**Embedding:** `{settings.EMBEDDING_MODEL.split('/')[-1]}`")
    st.markdown(f"**Inference Device:** `{settings.NLI_DEVICE.upper()}`")

    # Memory Usage Telemetry
    mem_mb = get_current_memory_mb()
    mem_color = "green" if mem_mb < 1800 else ("orange" if mem_mb < 2400 else "red")
    st.markdown(f"**Process RAM:** :{mem_color}[{mem_mb:.1f} MB / 2700 MB]")

    st.divider()
    st.info(
        "💡 **Notice:** Storage on Streamlit Community Cloud is ephemeral. Investigation history may be reset when the container hibernates."
    )


# ── 8. Main Application Interface ─────────────────────────────────────────────

# Navigation Tabs
tab_research, tab_history = st.tabs(["🔬 Research Workspace", "📜 Investigation History"])

with tab_research:
    st.header("Autonomous Academic Literature Synthesis")
    st.markdown(
        "Enter a research inquiry. ResearchLens retrieves open-access peer-reviewed papers, extracts verbatim passages from PDFs, "
        "validates factual assertions with cross-encoder NLI, and generates a cited academic synthesis."
    )

    # Question Input
    default_question = "Which paper introduced the Transformer architecture, and what was its key idea?"
    question_input = st.text_area(
        "Research Question",
        value=st.session_state.get("prefill_question", default_question),
        height=90,
        placeholder="e.g. Which paper introduced the Transformer architecture, and what was its key idea?",
        help="Be as specific as possible. Mentioning key technical concepts or domains yields better candidate retrieval.",
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        run_clicked = st.button("🚀 Run Research Investigation", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear / Reset", use_container_width=True):
            st.session_state.pop("current_job_id", None)
            st.session_state.pop("prefill_question", None)
            st.rerun()

    # ── Pipeline Execution Flow ───────────────────────────────────────────────
    if run_clicked:
        if not question_input.strip():
            st.error("Please enter a research question.")
            st.stop()

        if not settings.is_llm_configured:
            st.error(
                f"No API key configured for {settings.LLM_PROVIDER.title()}. "
                f"Please add GEMINI_API_KEY or OPENAI_API_KEY in `.streamlit/secrets.toml` or environment variables."
            )
            st.stop()

        if not sources_selected:
            st.error("Please select at least one academic source in the sidebar.")
            st.stop()

        if _PIPELINE_LOCK.locked():
            st.warning("⚠️ Another research investigation is currently executing in memory. Please wait for it to complete.")
            st.stop()

        # Create new ResearchJob row in SQLite
        new_job_id = f"st-{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc)
        pipeline_defs = _make_pipeline()

        db_init = SessionLocal()
        try:
            job_obj = ResearchJob(
                id=new_job_id,
                question=question_input.strip(),
                depth=depth_choice,
                sources=json.dumps(sources_selected),
                status="in_progress",
                total_papers=settings.DEPTH_COUNTS.get(depth_choice, 3),
                created_at=now,
                updated_at=now,
            )
            job_obj.set_pipeline(pipeline_defs)
            db_init.add(job_obj)
            db_init.commit()
        finally:
            db_init.close()

        # Warm up cached models lazily before thread dispatch
        with st.spinner("Preparing ML components (embeddings, reranker, NLI)..."):
            get_cached_embedding_model()
            get_cached_reranker()
            get_cached_nli_components()

        # Launch worker thread with concurrency lock
        def _guarded_worker():
            with _PIPELINE_LOCK:
                _run_pipeline_worker(new_job_id)

        worker_thread = threading.Thread(target=_guarded_worker, daemon=True)
        worker_thread.start()

        # Poll status with st.status
        with st.status("🔬 Conducting Autonomous Research Investigation...", expanded=True) as status_box:
            step_container = st.empty()
            progress_bar = st.progress(0.0)

            total_steps = len(pipeline_defs)
            start_poll_time = time.time()

            while True:
                time.sleep(0.8)

                db_poll = SessionLocal()
                try:
                    current_job = db_poll.query(ResearchJob).filter(ResearchJob.id == new_job_id).first()
                    if not current_job:
                        break

                    current_pipeline = current_job.get_pipeline()
                    current_status = current_job.status
                    current_stats = current_job.get_stage_stats()
                finally:
                    db_poll.close()

                # Calculate progress and format step UI
                completed_count = sum(1 for s in current_pipeline if s.get("status") == "completed")
                active_step = next((s for s in current_pipeline if s.get("status") == "active"), None)

                progress_val = min(1.0, (completed_count + (0.5 if active_step else 0.0)) / total_steps)
                progress_bar.progress(progress_val)

                # Format step progress view
                lines = []
                for s in current_pipeline:
                    s_id = s.get("id")
                    s_name = s.get("name")
                    s_status = s.get("status")
                    s_desc = s.get("description", "")

                    if s_status == "completed":
                        lines.append(f"✅ **{s_name}**: Completed")
                    elif s_status == "active":
                        lines.append(f"🔄 **{s_name}**: *In progress... ({s_desc})*")
                    elif s_status == "failed":
                        lines.append(f"❌ **{s_name}**: Failed")
                    else:
                        lines.append(f"⏳ **{s_name}**: Pending")

                step_container.markdown("\n\n".join(lines))

                # Check completion conditions
                if current_status in ("completed", "completed_with_warnings"):
                    dur = int(time.time() - start_poll_time)
                    status_box.update(
                        label=f"🎉 Research Investigation Completed in {dur}s!",
                        state="complete",
                        expanded=False,
                    )
                    st.session_state["current_job_id"] = new_job_id
                    break
                elif current_status == "insufficient_evidence":
                    status_box.update(
                        label="⚠️ Investigation Completed: Insufficient Corroborating Evidence",
                        state="error",
                        expanded=True,
                    )
                    st.session_state["current_job_id"] = new_job_id
                    break
                elif current_status == "failed":
                    fail_msg = current_job.failure_reason or current_job.error_message or "Unknown failure"
                    status_box.update(
                        label=f"❌ Pipeline Execution Failed: {fail_msg}",
                        state="error",
                        expanded=True,
                    )
                    st.session_state["current_job_id"] = new_job_id
                    break

                # Safety check if thread died unexpectedly
                if not worker_thread.is_alive() and current_status == "in_progress":
                    status_box.update(label="❌ Worker thread terminated unexpectedly.", state="error", expanded=True)
                    st.session_state["current_job_id"] = new_job_id
                    break

        st.rerun()

    # ── Display Active Investigation Results ──────────────────────────────────
    active_job_id = st.session_state.get("current_job_id")
    if active_job_id:
        db_view = SessionLocal()
        job_record = None
        try:
            job_record = db_view.query(ResearchJob).filter(ResearchJob.id == active_job_id).first()
            if job_record:
                investigation = job_to_investigation(job_record)
            else:
                investigation = None
        finally:
            db_view.close()

        if not investigation:
            st.error(f"Investigation record '{active_job_id}' could not be loaded from database.")
        else:
            st.divider()

            # Investigation Header Banner
            status_colors = {
                "completed": "green",
                "completed_with_warnings": "orange",
                "insufficient_evidence": "red",
                "failed": "red",
            }
            status_color = status_colors.get(investigation.status, "gray")

            conf_colors = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
            conf_color = conf_colors.get(investigation.research_confidence, "gray")

            st.markdown(f"### 📋 Investigation: *{investigation.question}*")

            # Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Status</div><div class="metric-value">:{status_color}[{investigation.status.upper()}]</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">:{conf_color}[{investigation.research_confidence}]</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Citation Integrity</div><div class="metric-value">{investigation.citation_integrity}%</div></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Verified Claims</div><div class="metric-value">{investigation.verifiedClaims} <span style="font-size:1rem;color:#94A3B8;">/ {investigation.evidenceItems}</span></div></div>',
                    unsafe_allow_html=True,
                )
            with m5:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Papers / Passages</div><div class="metric-value">{investigation.papersAnalyzed} <span style="font-size:1rem;color:#94A3B8;">/ {investigation.passages_total}</span></div></div>',
                    unsafe_allow_html=True,
                )

            # Warning / Failure notice if applicable
            if investigation.failure_reason:
                st.warning(f"**Investigation Finding / Reason:** {investigation.failure_reason}")

            # Network warning notices if any scholarly API encountered delays
            retrieval_errs = investigation.debug.get("retrieval_errors") if investigation.debug else []
            if retrieval_errs:
                st.info(f"ℹ️ **Scholarly API Note:** {'; '.join(retrieval_errs)}")

            # ── Detailed Result Tabs ──────────────────────────────────────────
            res_tab_report, res_tab_audit, res_tab_papers, res_tab_diag = st.tabs([
                "📄 Synthesized Report",
                "🛡️ Citation & Claim Audit",
                "📚 Ingested Papers & Passages",
                "⏱️ Pipeline Diagnostics",
            ])

            # Tab 1: Synthesized Report
            with res_tab_report:
                if investigation.report:
                    rep = investigation.report

                    # Executive Summary
                    st.subheader("Executive Summary")
                    st.markdown(rep.executiveSummary)

                    # Comparison Table if available
                    if rep.comparisonTable:
                        st.subheader("Methodological Comparison")
                        table_data = []
                        for row in rep.comparisonTable:
                            table_data.append({
                                "Model / Approach": row.model,
                                "Dataset / Benchmark": row.dataset,
                                "Performance": row.f1Score if row.f1Score != "Not extracted" else row.mapScore,
                                "Venue / Year": f"{row.venue or ''} {row.year or ''}".strip(),
                                "Citations": row.citationCount or "",
                            })
                        st.dataframe(table_data, use_container_width=True)

                    # Findings Sections
                    if rep.findings:
                        st.subheader("Detailed Findings")
                        for sec in rep.findings:
                            with st.expander(f"📌 {sec.sectionTitle}", expanded=True):
                                for p_idx, para in enumerate(sec.paragraphs):
                                    st.markdown(para.text)
                                    if para.citations:
                                        cite_badges = []
                                        for c in para.citations:
                                            badge_color = "#065F46" if c.status == CitationStatus.VERIFIED else "#92400E"
                                            cite_badges.append(
                                                f"<span style='background-color:{badge_color};color:#FFF;padding:2px 6px;border-radius:4px;font-size:0.75rem;'>"
                                                f"[{c.badgeNumber}] {c.paperTitle} (p.{c.page})</span>"
                                            )
                                        st.markdown(" ".join(cite_badges), unsafe_allow_html=True)

                    # Limitations & Conclusion
                    if rep.limitations:
                        st.subheader("Limitations")
                        for lim in rep.limitations:
                            st.markdown(f"- {lim}")

                    if rep.conclusion:
                        st.subheader("Conclusion")
                        st.markdown(rep.conclusion)

                    # References
                    if rep.references:
                        st.subheader("References")
                        for idx, ref in enumerate(rep.references, 1):
                            authors_fmt = ", ".join(ref.authors[:3]) + (" et al." if len(ref.authors) > 3 else "")
                            ext_link = ref.pdfUrl or (f"https://doi.org/{ref.doi}" if ref.doi else "")
                            link_md = f" [[Link]({ext_link})]" if ext_link else ""
                            st.markdown(
                                f"{idx}. **{ref.title}** ({ref.publicationYear}). {authors_fmt}. *{ref.journalConference or ref.source}*.{link_md}"
                            )

                    # Downloads
                    st.divider()
                    down_col1, down_col2 = st.columns(2)
                    with down_col1:
                        md_report = build_markdown_report(investigation)
                        st.download_button(
                            label="📥 Download Report (.md)",
                            data=md_report,
                            file_name=f"research_report_{investigation.id}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    with down_col2:
                        json_investigation = json.dumps(investigation.model_dump(), indent=2)
                        st.download_button(
                            label="📥 Download Full Data (.json)",
                            data=json_investigation,
                            file_name=f"investigation_{investigation.id}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                else:
                    st.info("No synthesized report was generated for this investigation.")

            # Tab 2: Citation & Claim Audit
            with res_tab_audit:
                st.subheader("Neural NLI Claim Verification Audit")
                st.markdown(
                    "Every factual claim is evaluated against source paper passages using **DeBERTa-v3** cross-encoder NLI. "
                    "Claims that fail empirical corroboration are strictly marked as unsupported or filtered."
                )

                # Collect citations across report
                all_citations: List[Citation] = []
                if investigation.report and investigation.report.findings:
                    for sec in investigation.report.findings:
                        for p in sec.paragraphs:
                            all_citations.extend(p.citations)

                if all_citations:
                    for cite in all_citations:
                        badge_html = render_verdict_badge(cite.status.value)
                        with st.expander(f"Citation [{cite.badgeNumber}]: {cite.claim[:90]}...", expanded=False):
                            st.markdown(f"**Assertion:** {cite.claim}")
                            st.markdown(f"**Verdict:** {badge_html}", unsafe_allow_html=True)
                            st.markdown(f"**Source Paper:** *{cite.paperTitle}* ({cite.year}) — Page {cite.page}")
                            st.markdown("**Verbatim Source Passage:**")
                            st.info(f'"{cite.passage}"')

                            if cite.atomicClaims:
                                st.markdown("**Decomposed Atomic Assertions:**")
                                for a in cite.atomicClaims:
                                    a_badge = render_verdict_badge(a.verdict.value)
                                    st.markdown(
                                        f"- {a_badge} **Assertion:** {a.atomicClaim} *(Confidence: {a.confidence:.2f})*\n"
                                        f"  - *Matched Premise:* \"{a.matchedSentence}\"",
                                        unsafe_allow_html=True,
                                    )
                else:
                    st.info("No inline citations were found in the report.")

            # Tab 3: Ingested Papers & Passages
            with res_tab_papers:
                st.subheader("Retrieved Scholarly Literature")
                db_p = SessionLocal()
                try:
                    paper_recs = db_p.query(PaperRecord).filter(PaperRecord.job_id == active_job_id).all()
                    if paper_recs:
                        for idx, prec in enumerate(paper_recs, 1):
                            authors = prec.get_authors()
                            passages = prec.get_passages()
                            authors_disp = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

                            with st.expander(f"{idx}. {prec.title} ({prec.publication_year or 2024})"):
                                st.markdown(f"**Authors:** {authors_disp}")
                                st.markdown(f"**Source Repository:** `{prec.source or 'arXiv'}`")
                                if prec.doi:
                                    st.markdown(f"**DOI:** [{prec.doi}](https://doi.org/{prec.doi})")
                                if prec.pdf_url:
                                    st.markdown(f"**PDF:** [Direct Link]({prec.pdf_url})")
                                if prec.abstract:
                                    st.markdown(f"**Abstract:** {prec.abstract}")

                                st.markdown(f"**Extracted Passages ({len(passages)}):**")
                                for p_idx, pass_item in enumerate(passages[:5], 1):
                                    st.text(f"Passage {p_idx} (Page {pass_item.get('page', 1)}): {pass_item.get('text', '')[:250]}...")
                                if len(passages) > 5:
                                    st.caption(f"... and {len(passages) - 5} additional windowed passages.")
                    else:
                        st.info("No individual paper records were persisted for this job.")
                finally:
                    db_p.close()

            # Tab 4: Diagnostics
            with res_tab_diag:
                st.subheader("Pipeline Stage Diagnostics & Execution Trace")
                if investigation.stage_stats:
                    stat_rows = []
                    for st_item in investigation.stage_stats:
                        stat_rows.append({
                            "Stage": st_item.name,
                            "Duration (ms)": st_item.duration_ms,
                            "Duration (s)": round(st_item.duration_ms / 1000.0, 2),
                            "In Count": st_item.in_count,
                            "Out Count": st_item.out_count,
                            "Error": st_item.error or "None",
                        })
                    st.dataframe(stat_rows, use_container_width=True)

                if investigation.debug:
                    st.markdown("**Query Plan & Strategy:**")
                    st.json(investigation.debug.get("plan", {}))


# ── 9. Investigation History Tab ──────────────────────────────────────────────
with tab_history:
    st.header("📜 Investigation History")
    st.caption("Review previous research investigations stored in the local SQLite database.")

    db_h = SessionLocal()
    try:
        all_jobs = db_h.query(ResearchJob).order_by(ResearchJob.created_at.desc()).limit(30).all()
        if not all_jobs:
            st.info("No past research investigations found in the database.")
        else:
            job_rows = []
            for j in all_jobs:
                job_rows.append({
                    "Job ID": j.id,
                    "Question": j.question,
                    "Depth": j.depth,
                    "Status": j.status,
                    "Verified Claims": j.verified_claims,
                    "Citation Coverage": f"{j.citation_coverage}%",
                    "Created At": j.created_at.strftime("%Y-%m-%d %H:%M:%S") if j.created_at else "",
                })

            st.dataframe(job_rows, use_container_width=True)

            # Reopen job selector
            job_ids = [j.id for j in all_jobs]
            job_map = {j.id: j.question for j in all_jobs}
            selected_job_to_load = st.selectbox(
                "Select an investigation to open in Workspace:",
                options=job_ids,
                format_func=lambda x: f"{x} — {job_map[x][:70]}...",
            )

            col_load, col_del = st.columns([2, 1])
            with col_load:
                if st.button("📂 Load Selected Investigation", type="primary", use_container_width=True):
                    st.session_state["current_job_id"] = selected_job_to_load
                    st.rerun()

            with col_del:
                if st.button("🗑️ Delete Selected Job", use_container_width=True):
                    job_to_del = db_h.query(ResearchJob).filter(ResearchJob.id == selected_job_to_load).first()
                    if job_to_del:
                        db_h.delete(job_to_del)
                        db_h.commit()
                        if st.session_state.get("current_job_id") == selected_job_to_load:
                            st.session_state.pop("current_job_id", None)
                        st.success(f"Deleted job {selected_job_to_load}")
                        st.rerun()
    finally:
        db_h.close()
