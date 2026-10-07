"""
Research Workflow Orchestrator
================================
Implements the 8-phase architecture blueprint from imp pl.pdf:
  1. Query Planning: Classification, title prediction, targeted boolean search queries, sub-questions.
  2. Paper Retrieval: arXiv (with 3s throttling) + Semantic Scholar (tenacity backoff) with SQLite cache.
  3. Relevance Reranking: Multi-factor scoring (reranker + log citations + title bonus), deduplication.
  4. PDF Full-Text Extraction: PyMuPDF extraction, header/footer/reference strip, 3-sentence windows.
  5. Vector Indexing: L2-normalized embeddings into per-job FAISS IndexFlatIP.
  6. Evidence Extraction & NLI Verification: Sub-question passage retrieval, candidate assertions, DeBERTa-v3 batch verification on CUDA.
  7. Evidence Gate: If verified claims < threshold, halts with honest 'insufficient_evidence' status.
  8. Academic Synthesis: Rephrases verified statements into academic prose preserving [n] citations; comparison table & metadata in Python.

Every stage records StageStat(name, started_at, duration_ms, in_count, out_count, error).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from ..config import settings
from ..models.database import ResearchJob, PaperRecord
from ..models.schemas import (
    Citation,
    CitationStatus,
    Claim,
    EntailmentVerdict,
    IntegrityMetrics,
    Paper,
    PipelineStep,
    ResearchDepth,
    ResearchInvestigation,
    ResearchReport,
    ResearchSource,
    StageStat,
    StepStatus,
)
from .paper_retrieval import (
    retrieve_papers,
    enrich_papers_with_s2,
    enrich_papers_with_openalex,
    get_and_clear_retrieval_errors,
    get_and_clear_s2_call_records,
    get_source_status,
    reset_retrieval_session,
)
from .ranker import rank_papers, select_anchor_paper, normalize_paper_title, filter_and_deduplicate
from .pdf_extractor import extract_papers_batch
from .vector_store import VectorStore
from .evidence import extract_and_verify_evidence
from .local_llm_service import synthesize_report
from .query_planner import plan_queries
from .text_utils import split_into_sentences

logger = logging.getLogger(__name__)


# ─── Pipeline Step Definitions ────────────────────────────────────────────────

def _make_pipeline() -> list[dict]:
    return [
        {"id": "step-1", "name": "Query Planning",         "status": "active",   "description": "Decomposing query, classifying intent, predicting titles and sub-questions", "iconName": "Compass"},
        {"id": "step-2", "name": "Candidate Retrieval",    "status": "pending",  "description": "Querying arXiv, OpenAlex, and Semantic Scholar with rate limiting and cache", "iconName": "DownloadCloud"},
        {"id": "step-3", "name": "Relevance Reranking",    "status": "pending",  "description": "Multi-factor scoring: cross-encoder reranker, citations, and title match",     "iconName": "Filter"},
        {"id": "step-4", "name": "PDF Full-Text Extraction","status": "pending", "description": "Downloading PDFs, stripping headers/footers/references, segmenting windows",  "iconName": "Database"},
        {"id": "step-5", "name": "Vector Indexing",        "status": "pending",  "description": "Building per-job FAISS index with L2-normalized passage embeddings",          "iconName": "GitCompare"},
        {"id": "step-6", "name": "Evidence & Source Match","status": "pending",  "description": "Extracting candidate assertions and verifying via source match",              "iconName": "CheckCircle2"},
        {"id": "step-7", "name": "Academic Synthesis",     "status": "pending",  "description": "Synthesizing verified claims with bracketed citations and comparison table",   "iconName": "FileText"},
    ]


def _set_step(pipeline: list[dict], step_id: str, status: str, timestamp: bool = True) -> None:
    for step in pipeline:
        if step["id"] == step_id:
            step["status"] = status
            if timestamp:
                step["timestamp"] = datetime.now(timezone.utc).isoformat()


def _advance_step(pipeline: list[dict], done_id: str, next_id: Optional[str]) -> None:
    _set_step(pipeline, done_id, "completed")
    if next_id:
        _set_step(pipeline, next_id, "active")


# ─── Status Determination Rules ───────────────────────────────────────────────

def determine_investigation_status(
    answer_sent_count: int,
    integrity: float,
    anchor_conf: str = "high",
    is_abstract_only: bool = False,
    confidence_note: Optional[str] = None,
    candidate_count: int = 0,
    passages_total: int = 0,
) -> tuple[str, list[str]]:
    """
    Step 6C: Production Investigation Status Determination.
    Returns (status, status_reasons).
    Status rules:
    - If answer_sent_count == 0 or integrity == 0.0 -> 'insufficient_evidence'
    - Else if integrity < 0.80 or anchor_conf == 'uncertain' or is_abstract_only -> 'completed_with_warnings'
    - Else -> 'completed'
    """
    status_reasons: list[str] = []

    if answer_sent_count == 0 or integrity == 0.0:
        status = "insufficient_evidence"
        fail_reason = (
            f"Evidence gate failed: Zero verified answer sentences passed NLI verification. "
            f"Ingested {candidate_count} candidate papers across {passages_total} passages."
        )
        status_reasons.append(fail_reason)
    elif integrity < 0.80 or anchor_conf == "uncertain" or is_abstract_only:
        status = "completed_with_warnings"
        if integrity < 0.80:
            status_reasons.append(f"Citation integrity ({integrity:.1%}) is below the 80% threshold.")
        if anchor_conf == "uncertain":
            status_reasons.append(confidence_note or "Anchor paper selection is marked uncertain.")
        if is_abstract_only:
            status_reasons.append("Anchor paper was analyzed via abstract only (full-text PDF unavailable).")
    else:
        status = "completed"

    return status, status_reasons


def get_investigation_status(
    answer_sent_count: int,
    integrity: float,
    anchor_conf: str = "high",
    is_abstract_only: bool = False,
) -> str:
    """Convenience wrapper returning just the status string."""
    status, _ = determine_investigation_status(
        answer_sent_count=answer_sent_count,
        integrity=integrity,
        anchor_conf=anchor_conf,
        is_abstract_only=is_abstract_only,
    )
    return status


# ─── Integrity Metrics ────────────────────────────────────────────────────────

def _compute_metrics(
    citations: list[Citation],
    report: ResearchReport,
    verified_answer_count: Optional[int] = None,
) -> IntegrityMetrics:
    """
    Compute citation integrity formula:
      citation_score = weighted NLI score for that citation
        - VERIFIED            -> 1.0
        - PARTIALLY_SUPPORTED -> 0.5
        - UNSUPPORTED         -> 0.0
        - CONTRADICTED        -> 0.0
      coverage = sum(citation_scores) / (n_citations + n_uncited_sentences) * 100
    """
    uncited = _count_uncited_sentences(report)
    verified = partial = unsupported = contradicted = 0
    total_atomic = 0
    entailed_atomic = 0
    score_sum = 0.0

    for c in citations:
        atomics = c.atomicClaims or []
        total_atomic += len(atomics)
        entailed_atomic += sum(1 for a in atomics if a.verdict == EntailmentVerdict.ENTAILS)

        if c.status == CitationStatus.VERIFIED:
            verified += 1
            score_sum += 1.0
        elif c.status == CitationStatus.PARTIALLY_SUPPORTED:
            partial += 1
            score_sum += 0.5
        elif c.status == CitationStatus.UNSUPPORTED:
            unsupported += 1
        elif c.status == CitationStatus.CONTRADICTED:
            contradicted += 1

    n_citations = len(citations)
    total_assertions = n_citations + uncited
    avg = score_sum / total_assertions if total_assertions > 0 else 0.0
    coverage = max(0, min(100, round(avg * 100)))

    verified_claims_badge = verified_answer_count if verified_answer_count is not None else verified

    return IntegrityMetrics(
        citationCoverage=coverage,
        verifiedClaims=verified_claims_badge,
        partiallySupportedClaims=partial,
        unsupportedClaims=unsupported,
        contradictedClaims=contradicted,
        totalAtomicClaims=total_atomic,
        entailedAtomicClaims=entailed_atomic,
        potentialConflicts=contradicted,
        uncitedSentences=uncited,
    )


def _count_uncited_sentences(report: ResearchReport) -> int:
    """Count substantive sentences that lack an inline citation marker."""
    import re
    _CITE_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]|\[cite-[^\]]+\]")
    _SENT_RE = re.compile(r"(?<=[.?!])\s+")

    def _uncited_in(text: str) -> int:
        sentences = _SENT_RE.split(text)
        return sum(1 for s in sentences if len(s) > 20 and not _CITE_RE.search(s))

    count = 0
    if report.executiveSummary:
        count += _uncited_in(report.executiveSummary)
    for section in report.findings:
        for para in section.paragraphs:
            if not para.citations:
                sentences = _SENT_RE.split(para.text)
                count += max(1, len([s for s in sentences if len(s) > 20]))
            else:
                count += _uncited_in(para.text)
    return count


# ─── Insufficient Evidence Report ─────────────────────────────────────────────

def _build_insufficient_evidence_report(
    question: str,
    papers: list[Paper],
    citations: list[Citation],
    reason: str,
) -> ResearchReport:
    """Honest fallback report when evidence gate fails."""
    n_papers = len(papers)
    n_citations = len(citations)

    summary = (
        f"Insufficient empirical evidence found for query: \"{question}\"\n\n"
        f"Reason: {reason}\n\n"
        f"The pipeline retrieved {n_papers} candidate paper(s) and evaluated {n_citations} citation assertions. "
        f"However, neural NLI entailment verification on DeBERTa-v3 did not corroborate the required factual assertions.\n\n"
        f"Suggested actions:\n"
        f"  - Rephrase the question with more specific paper titles, author names, or technical keywords.\n"
        f"  - Increase research depth to 'Deep' to ingest additional candidate preprints.\n"
        f"  - Ensure the target topic has open-access papers available on arXiv or Semantic Scholar."
    )

    return ResearchReport(
        executiveSummary=summary,
        methodology="Retrieved literature was evaluated against strict DeBERTa-v3 NLI thresholds. Insufficient entailment observed.",
        findings=[],
        comparisonTable=[],
        computationalRequirements="",
        contradictoryEvidence="",
        limitations=["Strict evidence gate triggered: uncorroborated claims were rejected to prevent hallucination."],
        conclusion="No definitive verified conclusion can be drawn from the currently ingested corpus.",
        references=papers[:6],
    )


# ─── Main Pipeline Execution ──────────────────────────────────────────────────

async def run_research_pipeline(job_id: str, db: Session) -> None:
    """
    Main async research pipeline executing Phases 1 to 8.
    """
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        logger.error("[Pipeline] Job %s not found", job_id)
        return

    pipeline = _make_pipeline()
    job.set_pipeline(pipeline)
    db.commit()

    stage_stats: list[StageStat] = []
    debug_info: dict[str, Any] = {
        "seed_papers_enabled": getattr(settings, "SEED_PAPERS_ENABLED", True),
        "disable_cache": getattr(settings, "DISABLE_CACHE", False),
    }
    logger.info(
        "[Pipeline] Job %s config: SEED_PAPERS_ENABLED=%s, DISABLE_CACHE=%s",
        job_id,
        debug_info["seed_papers_enabled"],
        debug_info["disable_cache"],
    )

    def record_stage(name: str, t_start: float, in_count: int, out_count: int, error: Optional[str] = None):
        dur = int((time.time() - t_start) * 1000)
        stat = StageStat(
            name=name,
            started_at=datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
            duration_ms=dur,
            in_count=in_count,
            out_count=out_count,
            error=error,
        )
        stage_stats.append(stat)
        job.set_stage_stats([s.model_dump() for s in stage_stats])

    try:
        question = job.question
        depth = ResearchDepth(job.depth)
        sources = [ResearchSource(s) for s in job.get_sources()]

        # ─── Phase 1: Query Planning ──────────────────────────────────────────
        t0 = time.time()
        logger.info("[Pipeline] Stage 1: Query Planning for '%s'", question[:80])
        plan = await plan_queries(question)
        debug_info["plan"] = plan
        debug_info["planner_provider"] = plan.get("planner_provider", "none")
        debug_info["planner_model"] = plan.get("planner_model", "none")
        q_type = plan.get("question_type", "literature_review")
        queries = plan.get("queries", [question])
        title_guesses = plan.get("expected_titles", [])
        sub_questions = plan.get("sub_questions", [question])
        debug_info["queries_used"] = queries
        debug_info["titles_retrieved"] = title_guesses
        record_stage("query_planning", t0, in_count=1, out_count=len(queries))

        _advance_step(pipeline, "step-1", "step-2")
        job.set_pipeline(pipeline)
        db.commit()

        # ─── Phase 2: Candidate Retrieval ─────────────────────────────────────
        t0 = time.time()
        logger.info("[Pipeline] Stage 2: Candidate Retrieval across sources...")
        reset_retrieval_session()
        raw_papers = await retrieve_papers(
            question, depth, sources,
            api_key_s2=settings.SEMANTIC_SCHOLAR_API_KEY,
            custom_queries=queries,
            title_guesses=title_guesses,
            question_type=q_type,
        )
        debug_info["papers_discovered"] = len(raw_papers)
        debug_info["source_status"] = get_source_status()
        record_stage("candidate_retrieval", t0, in_count=len(queries), out_count=len(raw_papers))
        r_errors = get_and_clear_retrieval_errors()
        if r_errors:
            debug_info["retrieval_errors"] = r_errors
            s2_errs = [e for e in r_errors if "semantic scholar" in e.lower() or "s2" in e.lower()]
            if s2_errs:
                for step in pipeline:
                    if step["id"] == "step-2":
                        step["description"] = f"Retrieved candidates across sources (Warning: {s2_errs[0]})"

        # Record per-call Semantic Scholar statuses into stage_stats
        s2_calls = get_and_clear_s2_call_records()
        for sc in s2_calls:
            q_label = sc["query"][:30]
            stage_stats.append(StageStat(
                name=f"s2_query:{q_label}",
                started_at=sc.get("timestamp", datetime.now(timezone.utc).isoformat()),
                duration_ms=sc.get("duration_ms", 0),
                in_count=1,
                out_count=sc.get("papers_count", 0),
                error=sc.get("error") if sc.get("status_code", 200) != 200 else None,
            ))
        job.set_stage_stats([s.model_dump() for s in stage_stats])

        _advance_step(pipeline, "step-2", "step-3")
        job.set_pipeline(pipeline)
        db.commit()

        if not raw_papers:
            fail_reason = "No candidate papers could be retrieved for the specified query."
            if debug_info.get("retrieval_errors"):
                fail_reason += f" Retrieval warnings: {'; '.join(debug_info['retrieval_errors'])}"
            job.status = "insufficient_evidence"
            job.failure_reason = fail_reason
            job.set_report(_build_insufficient_evidence_report(question, [], [], fail_reason).model_dump())
            db.commit()
            return

        # ─── Phase 3: Relevance Reranking & Anchor Paper Selection ────────────
        t0 = time.time()
        logger.info("[Pipeline] Stage 3: Deduplicating and enriching %d candidate papers...", len(raw_papers))

        # 1. Deduplicate candidate papers
        deduped_candidates = filter_and_deduplicate(raw_papers)
        debug_info["unique_papers"] = len(deduped_candidates)

        # 2. Enrich candidate papers with S2 batch citations & metadata
        enriched_candidates = await enrich_papers_with_s2(deduped_candidates)

        # 3. Fallback: Enrich any remaining uncited candidate papers with OpenAlex
        enriched_candidates = await enrich_papers_with_openalex(enriched_candidates)

        anchor_paper = None
        alternate_paper = None
        anchor_debug: dict[str, Any] = {"anchor_paper_id": None, "anchor_rule": "none", "anchor_confidence": "none", "reason": "not_factual_lookup"}
        if q_type == "factual_lookup":
            anchor_paper, enriched_candidates, anchor_debug = await select_anchor_paper(enriched_candidates, title_guesses, question=question)
            alt_id = anchor_debug.get("alternate_paper_id")
            alt_title = anchor_debug.get("alternate_paper_title")
            if alt_id or alt_title:
                alternate_paper = next(
                    (p for p in enriched_candidates if (alt_id and p.id == alt_id) or (alt_title and normalize_paper_title(p.title) == normalize_paper_title(alt_title))),
                    None,
                )

        if not anchor_paper:
            anchor_debug["anchor_confidence"] = "none"
            anchor_debug["anchor_paper_id"] = None

        # 3. Rank papers and cut to top N AFTER enrichment
        ranked_papers = rank_papers(
            question=question,
            papers=enriched_candidates,
            question_type=q_type,
            title_guesses=title_guesses,
            depth=depth,
            anchor_paper=anchor_paper,
        )

        debug_info["relevant_papers"] = len(ranked_papers)

        if anchor_paper:
            # Map anchor paper to its assigned sequential ID in ranked_papers
            matched_p = next(
                (p for p in ranked_papers if normalize_paper_title(p.title) == normalize_paper_title(anchor_paper.title)),
                None,
            )
            anchor_id = matched_p.id if matched_p else anchor_paper.id
            anchor_debug["anchor_paper_id"] = anchor_id
            job.anchor_paper_id = anchor_id
            debug_info["anchor_paper_id"] = anchor_id
            anchor_paper.is_anchor = True
            if matched_p:
                matched_p.is_anchor = True
        else:
            job.anchor_paper_id = None
            debug_info["anchor_paper_id"] = None

        debug_info["anchor_paper"] = anchor_debug
        record_stage("relevance_reranking", t0, in_count=len(enriched_candidates), out_count=len(ranked_papers))
        debug_info["top_papers"] = [p.title for p in ranked_papers[:5]]

        _advance_step(pipeline, "step-3", "step-4")
        job.set_pipeline(pipeline)
        job.set_debug(debug_info)
        db.commit()

        # ─── Phase 4: PDF Extraction & Passage Chunking ───────────────────────
        t0 = time.time()
        logger.info("[Pipeline] Stage 4: Extracting full-text & chunking passages for %d papers...", len(ranked_papers))
        enriched_papers = await extract_papers_batch(ranked_papers, max_workers=settings.MAX_PDF_WORKERS)
        
        passages_total = sum(len(p.passages or []) for p in enriched_papers)
        full_text_count = sum(1 for p in enriched_papers if len(p.passages or []) > 0 and not getattr(p, "is_abstract_only", False))
        job.passages_total = passages_total
        debug_info["passages_total"] = passages_total
        debug_info["full_text_papers"] = full_text_count
        debug_info["passages_per_paper"] = {
            p.title[:60]: len(p.passages or []) for p in enriched_papers
        }
        record_stage("pdf_extraction", t0, in_count=len(ranked_papers), out_count=passages_total)

        # Persist papers and passages into DB
        for p in enriched_papers:
            passages_dicts = [pass_obj.model_dump() for pass_obj in (p.passages or [])]
            rec = PaperRecord(
                id=f"{job_id}::{p.id}",
                job_id=job_id,
                title=p.title,
                authors_json=json.dumps(p.authors),
                publication_year=p.publicationYear,
                journal_conference=p.journalConference,
                doi=p.doi,
                source=p.source,
                abstract=p.abstract,
                pdf_url=p.pdfUrl,
                full_text=p.fullText[:50000] if p.fullText else None,
                passages_json=json.dumps(passages_dicts),
            )
            db.merge(rec)
        db.commit()

        _advance_step(pipeline, "step-4", "step-5")
        job.set_pipeline(pipeline)
        db.commit()

        # ─── Phase 5: FAISS Vector Indexing ───────────────────────────────────
        t0 = time.time()
        logger.info("[Pipeline] Stage 5: Building FAISS index over %d passages...", passages_total)
        vector_store = VectorStore()
        try:
            vector_store.build(enriched_papers)
            record_stage("vector_indexing", t0, in_count=passages_total, out_count=passages_total)
        except Exception as exc:
            logger.warning("[Pipeline] Vector store build failed: %s", exc)
            record_stage("vector_indexing", t0, in_count=passages_total, out_count=0, error=str(exc))

        _advance_step(pipeline, "step-5", "step-6")
        job.set_pipeline(pipeline)
        db.commit()

        # ─── Phase 6: Evidence & NLI Verification Engine ─────────────────────
        t0 = time.time()
        logger.info("[Pipeline] Stage 6: Extracting & verifying evidence via source match...")
        claims, citations, rejected_claims, removal_counts = extract_and_verify_evidence(
            question=question,
            sub_questions=sub_questions,
            papers=enriched_papers,
            vector_store=vector_store,
            max_citations=15,
            anchor_paper_id=job.anchor_paper_id,
        )
        debug_info["rejected_claims"] = rejected_claims
        debug_info["evidence_filter_removals"] = removal_counts
        verified_count = sum(1 for c in citations if c.status == CitationStatus.VERIFIED)
        partial_count = sum(1 for c in citations if c.status == CitationStatus.PARTIALLY_SUPPORTED)
        total_cand_claims = len(claims) + len(rejected_claims)
        debug_info["candidate_claims"] = total_cand_claims
        evidence_bearing_count = len({c.paperId for c in citations if c.status == CitationStatus.VERIFIED})
        debug_info["evidence_bearing_papers"] = evidence_bearing_count
        debug_info["verified_claims"] = verified_count
        debug_info["contradicted_claims"] = sum(1 for c in citations if c.status == CitationStatus.CONTRADICTED)
        debug_info["insufficient_claims"] = len(rejected_claims) + sum(1 for c in citations if c.status == CitationStatus.UNSUPPORTED)
        record_stage("evidence_verification", t0, in_count=passages_total, out_count=len(citations))

        _advance_step(pipeline, "step-6", "step-7")
        job.set_pipeline(pipeline)
        db.commit()

        # ─── Phase 7: Evidence Gate Check ─────────────────────────────────────
        min_required_verified = 1 if q_type == "factual_lookup" else 2
        logger.info("[Pipeline] Gate check: verified=%d, partial=%d (required min=%d)", verified_count, partial_count, min_required_verified)

        if (verified_count + partial_count) < min_required_verified:
            fail_reason = (
                f"Evidence gate failed: Found {verified_count} verified and {partial_count} partially supported claims "
                f"(minimum required: {min_required_verified}). "
                f"Passages evaluated: {passages_total} across {len(enriched_papers)} papers."
            )
            logger.warning("[Pipeline] %s", fail_reason)
            job.status = "insufficient_evidence"
            job.failure_reason = fail_reason
            job.papers_analyzed = len(enriched_papers)
            job.total_papers = len(enriched_papers)
            job.evidence_items = len(citations)
            job.verified_claims = verified_count
            job.partially_supported_claims = partial_count
            job.unsupported_claims = sum(1 for c in citations if c.status == CitationStatus.UNSUPPORTED)
            job.contradicted_claims = sum(1 for c in citations if c.status == CitationStatus.CONTRADICTED)
            job.citation_coverage = 0.0
            
            insufficient_rep = _build_insufficient_evidence_report(question, enriched_papers, citations, fail_reason)
            job.set_report(insufficient_rep.model_dump())
            job.set_debug(debug_info)
            _advance_step(pipeline, "step-7", None)
            job.set_pipeline(pipeline)
            db.commit()
            return

        # ─── Phase 8: Academic Report Synthesis (Step 6A, 6B, 6C) ─────────────
        t0 = time.time()
        logger.info("[Pipeline] Stage 8: Synthesizing academic report from verified evidence...")
        anchor_conf_to_pass = anchor_debug.get("anchor_confidence", "none" if anchor_paper is None else "high")
        report, integrity, removed_sentences = await synthesize_report(
            question=question,
            depth=depth,
            papers=enriched_papers,
            claims=claims,
            citations=citations,
            anchor_paper=anchor_paper,
            anchor_confidence=anchor_conf_to_pass,
            alternate_paper=alternate_paper,
            anchor_rule=anchor_debug.get("anchor_rule", "none"),
            stage_stats=stage_stats,
            retrieval_warnings=debug_info.get("retrieval_errors"),
            debug_info=debug_info,
        )
        record_stage("report_synthesis", t0, in_count=len(citations), out_count=1)

        # Track Concept Coverage & Research Depth (Requirement 1 & 4)
        target_conc_list = plan.get("target_concepts", [])
        concept_cov = getattr(report, "concept_coverage", {})
        tot_conc = len(concept_cov) if concept_cov else max(1, len(target_conc_list))
        ver_conc = sum(1 for st in concept_cov.values() if st == "VERIFIED")
        research_depth_pct = round((ver_conc / max(1, tot_conc) * 100.0), 1)
        debug_info["research_depth"] = research_depth_pct
        debug_info["concept_coverage"] = concept_cov

        debug_info.setdefault("planner_provider", "none")
        debug_info.setdefault("planner_model", "none")
        debug_info.setdefault("writer_provider", "none")
        debug_info.setdefault("writer_model", "none")

        # Step 6C: Investigation Status Determination
        anchor_is_abstract_only = getattr(anchor_paper, "is_abstract_only", False) if anchor_paper else False
        anchor_conf = anchor_debug.get("anchor_confidence", "none" if anchor_paper is None else "high")
        confidence_note = anchor_debug.get("confidence_note")

        # Replace executiveSummary.split('. ') with the real sentence list
        exec_sentences = split_into_sentences(report.executiveSummary)
        # Verified claim sentences count (from verified_sentences details)
        ver_details = debug_info.get("verified_sentences", [])
        verified_claims_in_ans = sum(1 for d in ver_details if d.get("type") in ("claim", "qualification") and d.get("status") != "removed")
        claim_sent_count = verified_claims_in_ans if verified_claims_in_ans > 0 else (1 if (integrity > 0 and len(exec_sentences) >= 1) else 0)

        status, status_reasons = determine_investigation_status(
            answer_sent_count=claim_sent_count,
            integrity=integrity,
            anchor_conf=anchor_conf,
            is_abstract_only=anchor_is_abstract_only,
            confidence_note=confidence_note,
            candidate_count=len(enriched_papers),
            passages_total=job.passages_total,
        )

        # Check for LLM fallback warnings and make visible on job
        planner_prov = str(debug_info.get("planner_provider", "")).lower()
        writer_prov = str(debug_info.get("writer_provider", "")).lower()
        if "fallback" in planner_prov or "fallback" in writer_prov:
            fb_warn = "LLM fallback triggered: fell back to local model."
            if fb_warn not in status_reasons:
                status_reasons.append(fb_warn)
            if status == "completed":
                status = "completed_with_warnings"

        job.status = status
        job.failure_reason = "; ".join(status_reasons) if status_reasons else None

        metrics = _compute_metrics(citations, report, verified_answer_count=claim_sent_count)

        _advance_step(pipeline, "step-7", None)
        job.set_pipeline(pipeline)

        debug_info["status_reasons"] = status_reasons
        debug_info["citation_integrity"] = integrity
        debug_info["removed_sentences"] = removed_sentences

        # Finalize job record
        job.papers_analyzed = len(enriched_papers)
        job.total_papers = len(enriched_papers)
        job.evidence_items = len(citations)
        job.verified_claims = metrics.verifiedClaims
        job.partially_supported_claims = metrics.partiallySupportedClaims
        job.unsupported_claims = metrics.unsupportedClaims
        job.contradicted_claims = metrics.contradictedClaims
        job.potential_conflicts = metrics.potentialConflicts
        # B7: Validation Gate before saving report
        from .report_validator import validate_report
        val_errors = validate_report(report)
        if val_errors:
            logger.error("[Pipeline] Report validation gate failed for job %s: %s", job_id, val_errors)
            status = "generation_error"
            job.status = "generation_error"
            fail_desc = "Report validation gate failed: " + "; ".join(val_errors)
            job.failure_reason = fail_desc
            debug_info["validation_errors"] = val_errors
            if status_reasons:
                status_reasons.append(fail_desc)
            else:
                status_reasons = [fail_desc]

        job.set_report(report.model_dump())
        job.set_debug(debug_info)
        db.commit()

        logger.info(
            "[Pipeline] Job %s finalized with status '%s' (reasons: %s). Citation integrity: %d%%",
            job_id, job.status, status_reasons, job.citation_coverage
        )

    except Exception as exc:
        logger.exception("[Pipeline] Job %s failed with exception: %s", job_id, exc)
        job.status = "failed"
        job.error_message = str(exc)
        job.failure_reason = f"Execution error: {exc}"
        for step in pipeline:
            if step["status"] == "active":
                step["status"] = "failed"
        job.set_pipeline(pipeline)
        db.commit()


# ─── Read Helpers ─────────────────────────────────────────────────────────────

def job_to_investigation(job: ResearchJob) -> ResearchInvestigation:
    """Convert a DB job record to the ResearchInvestigation schema."""
    report_data = job.get_report()
    report = ResearchReport(**report_data) if report_data else None

    pipeline_data = job.get_pipeline()
    pipeline = [PipelineStep(**s) for s in pipeline_data]

    stage_stats_data = job.get_stage_stats()
    stage_stats = [StageStat(**s) for s in stage_stats_data]

    debug_blob = job.get_debug() or {}
    raw_cite_integrity = debug_blob.get("citation_integrity", (job.citation_coverage / 100.0 if job.citation_coverage else 0.0))
    cite_integrity_float = raw_cite_integrity / 100.0 if raw_cite_integrity > 1.0 else raw_cite_integrity

    total_claims = job.verified_claims + job.partially_supported_claims + job.unsupported_claims + job.contradicted_claims
    cand_claims = debug_blob.get("candidate_claims", total_claims)
    ev_coverage = round((job.verified_claims / max(1, cand_claims) * 100.0), 1) if cand_claims > 0 else 0.0

    # 13 Required Coverage Metrics (Requirement 1)
    papers_discovered = debug_blob.get("papers_discovered", job.total_papers)
    unique_papers = debug_blob.get("unique_papers", job.total_papers)
    relevant_papers = debug_blob.get("relevant_papers", job.papers_analyzed)
    full_text_papers = debug_blob.get("full_text_papers", job.papers_analyzed)
    evidence_bearing_papers = debug_blob.get("evidence_bearing_papers", min(job.papers_analyzed, job.verified_claims))
    candidate_claims = cand_claims
    verified_claims = job.verified_claims
    contradicted_claims = job.contradicted_claims
    insufficient_claims = debug_blob.get("insufficient_claims", job.unsupported_claims)
    citation_integrity = round(cite_integrity_float * 100.0, 1)
    evidence_coverage = ev_coverage
    research_depth = debug_blob.get("research_depth", 0.0)
    source_status = debug_blob.get("source_status", {})
    concept_coverage = debug_blob.get("concept_coverage", {})
    retrieval_warnings = debug_blob.get("retrieval_errors", [])

    # Honest Research Confidence metric (Requirement 1 & 6)
    if job.status == "insufficient_evidence" or job.verified_claims == 0:
        conf = "NONE"
    elif (
        cite_integrity_float < 0.60
        or research_depth < 30.0
        or (evidence_bearing_papers <= 1 and len(concept_coverage) >= 3)
        or bool(retrieval_warnings and evidence_bearing_papers <= 1)
    ):
        conf = "LOW"
    elif (
        job.verified_claims >= 4
        and cite_integrity_float >= 0.85
        and evidence_bearing_papers >= 3
        and job.contradicted_claims == 0
        and not retrieval_warnings
        and research_depth >= 50.0
    ):
        conf = "HIGH"
    else:
        conf = "MEDIUM"

    return ResearchInvestigation(
        id=job.id,
        question=job.question,
        depth=ResearchDepth(job.depth),
        sources=[ResearchSource(s) for s in job.get_sources()],
        status=job.status,
        papersAnalyzed=job.papers_analyzed,
        totalPapers=job.total_papers,
        evidenceItems=job.evidence_items,
        verifiedClaims=job.verified_claims,
        partiallySupportedClaims=job.partially_supported_claims,
        unsupportedClaims=job.unsupported_claims,
        contradictedClaims=job.contradicted_claims,
        potentialConflicts=job.potential_conflicts,
        citationCoverage=job.citation_coverage,

        # ── Coverage Metrics ──
        papers_discovered=papers_discovered,
        unique_papers=unique_papers,
        relevant_papers=relevant_papers,
        full_text_papers=full_text_papers,
        evidence_bearing_papers=evidence_bearing_papers,
        candidate_claims=candidate_claims,
        verified_claims=verified_claims,
        contradicted_claims=contradicted_claims,
        insufficient_claims=insufficient_claims,
        citation_integrity=citation_integrity,
        evidence_coverage=evidence_coverage,
        research_depth=research_depth,
        research_confidence=conf,
        source_status=source_status,
        concept_coverage=concept_coverage,
        retrieval_warnings=retrieval_warnings,

        uncitedSentences=job.uncited_sentences,
        passages_total=job.passages_total,
        failure_reason=job.failure_reason,
        anchor_paper_id=job.anchor_paper_id,
        stage_stats=stage_stats,
        debug=debug_blob,
        createdAt=job.created_at.isoformat() if job.created_at else "",
        updatedAt=job.updated_at.isoformat() if job.updated_at else "",
        pipeline=pipeline,
        report=report,
    )
