"""
ResearchLens LangGraph State Machine
====================================
Orchestrates the academic research pipeline as an explicit, stateful LangGraph
directed graph with a bounded, self-correcting retrieval-retry loop.

Flow:
    START -> preflight --(rejected)--> END
                 |
               (ok)
                 v
             plan_query -> retrieve -> rerank_fetch -> verify_claims
                                  ^                          |
                                  |                 [enough verified?]
                                  |                 /                \
                            rewrite_query <--- (no, < max_retries)  (yes or maxed)
                                                                       |
                                                                       v
                                                                  write_report -> END
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..models.schemas import (
    Citation,
    CitationStatus,
    Paper,
    ResearchDepth,
    ResearchReport,
    ResearchSource,
    StageStat,
)
from ..security import validate_research_query
from .evidence import extract_and_verify_evidence
from .local_llm_service import _call_llm, synthesize_report
from .pdf_extractor import extract_papers_batch
from .query_planner import plan_queries
from .ranker import (
    filter_and_deduplicate,
    normalize_paper_title,
    rank_papers,
    select_anchor_paper,
)
from .paper_retrieval import (
    enrich_papers_with_openalex,
    enrich_papers_with_s2,
    get_and_clear_retrieval_errors,
    get_and_clear_s2_call_records,
    get_s2_rate_limited_count,
    get_source_status,
    reset_retrieval_session,
    retrieve_papers,
)
from .text_utils import split_into_sentences
from .vector_store import VectorStore

try:
    from langsmith import traceable
except ImportError:
    def traceable(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f

logger = logging.getLogger(__name__)

# ─── Configuration Constants ──────────────────────────────────────────────────
MIN_VERIFIED_FACTUAL: int = 1
MIN_VERIFIED_REVIEW: int = 2
MAX_RETRIES: int = 2


# ─── State Definition ─────────────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    """
    State container for the ResearchLens LangGraph pipeline.
    All data is kept serializable (dicts, lists, strings, numbers, booleans, models).
    Heavy objects like database sessions are kept out of state.
    """
    question: str
    depth: str
    sources: List[str]
    allowed: bool
    reject_reason: str
    retries: int
    max_retries: int
    retry_history: List[Dict[str, Any]]

    # Query Planning
    intent: str
    sub_queries: List[str]
    expected_titles: List[str]
    sub_questions: List[str]
    target_concepts: List[str]
    planner_debug: Dict[str, Any]

    # Retrieval & Reranking
    papers: List[Paper]
    unique_papers_count: int
    ranked_papers: List[Paper]
    anchor_paper: Optional[Paper]
    anchor_debug: Dict[str, Any]
    alternate_paper: Optional[Paper]

    # Extraction & Indexing
    passages: List[Dict[str, Any]]
    passages_total: int
    vector_store: Optional[VectorStore]

    # Evidence & Verification
    claims: List[Any]
    citations: List[Citation]
    verified: List[Citation]
    rejected: List[str]
    evidence_bearing_papers_count: int

    # Final Synthesis & Diagnostics
    report: Optional[ResearchReport]
    limited_evidence: bool
    status: str
    status_reasons: List[str]
    stage_stats: List[Dict[str, Any]]
    debug_info: Dict[str, Any]
    failure_reason: Optional[str]


# ─── Helper: Query Rewriter ───────────────────────────────────────────────────

@traceable(name="researchlens_rewrite_queries", run_type="llm")
async def rewrite_search_queries(
    question: str,
    previous_queries: List[str],
    rejected_claims: List[str],
    intent: str = "literature_review",
) -> List[str]:
    """
    Generate broadened, alternative keyword queries when candidate claims fail verification.
    Uses LLM with a fallback heuristic keyword generator.
    """
    logger.info("[LangGraph:Rewrite] Rewriting queries for '%s' (prev=%s)", question[:60], previous_queries)

    prompt = (
        "You are an academic retrieval specialist. The initial search queries returned insufficient "
        "verified evidence for this research question.\n"
        f"Original Question: \"{question}\"\n"
        f"Previous Queries: {json.dumps(previous_queries)}\n"
        f"Uncorroborated / Missing Evidence: {json.dumps(rejected_claims[:4])}\n\n"
        "Generate 2 to 4 alternative academic keyword search phrases (2 to 5 words each). "
        "CRITICAL:\n"
        "1. Do NOT repeat previous queries verbatim.\n"
        "2. Broaden terminology or focus on core mechanisms and foundational methods.\n"
        "3. Avoid conversational words (what, which, why, paper, author).\n"
        "Return ONLY a JSON array of strings, e.g. [\"keyword query 1\", \"keyword query 2\"]."
    )

    try:
        raw, provider = await _call_llm(prompt, max_tokens=150)
        if raw and raw.strip():
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                cleaned = [str(q).strip() for q in parsed if str(q).strip() and len(str(q).strip()) > 3]
                if cleaned:
                    # Filter out exact duplicates of previous queries
                    new_q = [q for q in cleaned if q.lower() not in [p.lower() for p in previous_queries]]
                    if new_q:
                        logger.info("[LangGraph:Rewrite] LLM generated new queries (%s): %s", provider, new_q)
                        return new_q[:4]
    except Exception as exc:
        logger.warning("[LangGraph:Rewrite] LLM rewrite failed (%s), using heuristic rewrite", exc)

    # Heuristic fallback: broaden by stripping punctuation, extracting major technical nouns
    words = [
        w.lower() for w in re.findall(r"[A-Za-z0-9\-]+", question)
        if len(w) > 3 and w.lower() not in {
            "which", "what", "where", "when", "paper", "architecture", "model",
            "introduced", "proposed", "author", "authors", "their", "this"
        }
    ]
    variants = []
    if len(words) >= 2:
        variants.append(" ".join(words[:3]))
    if len(words) >= 4:
        variants.append(" ".join(words[2:5]))
    variants.append(f"{' '.join(words[:2])} review" if words else f"{question} review")

    fresh = [v for v in variants if v.lower() not in [p.lower() for p in previous_queries]]
    return fresh or [f"{question} survey"]


# ─── Graph Node Definitions ───────────────────────────────────────────────────

@traceable(name="researchlens_preflight", run_type="parser")
def preflight_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 1: Pre-flight Security & Query Guard.
    Enforces scope, character limits (10-500), and rejects casual chat or code pastes.
    """
    question = state.get("question", "").strip()
    is_valid, validation_err = validate_research_query(question)

    debug_info = dict(state.get("debug_info") or {})
    debug_info["preflight_valid"] = is_valid

    if not is_valid:
        logger.warning("[LangGraph:Preflight] Query rejected: %s", validation_err)
        return {
            "allowed": False,
            "reject_reason": validation_err or "Invalid research query.",
            "status": "failed",
            "failure_reason": validation_err,
            "retries": 0,
            "debug_info": debug_info,
        }

    return {
        "allowed": True,
        "reject_reason": "",
        "retries": 0,
        "retry_history": [],
        "debug_info": debug_info,
    }


@traceable(name="researchlens_plan_query", run_type="chain")
async def plan_query_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 2: Query Planning & Intent Classification.
    Decomposes research question, classifies intent, and predicts sub-questions and titles.
    """
    question = state["question"]
    logger.info("[LangGraph:PlanQuery] Planning queries for '%s'", question[:80])
    t0 = time.time()

    plan = await plan_queries(question)

    intent = plan.get("question_type", "literature_review")
    sub_queries = plan.get("queries", [question])
    expected_titles = plan.get("expected_titles", [])
    sub_questions = plan.get("sub_questions", [question])
    target_concepts = plan.get("target_concepts", [])

    debug_info = dict(state.get("debug_info") or {})
    debug_info["plan"] = plan
    debug_info["planner_provider"] = plan.get("planner_provider", "none")
    debug_info["planner_model"] = plan.get("planner_model", "none")
    debug_info["queries_used"] = sub_queries
    debug_info["titles_retrieved"] = expected_titles

    stage_stats = list(state.get("stage_stats") or [])
    stage_stats.append({
        "name": "query_planning",
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "duration_ms": int((time.time() - t0) * 1000),
        "in_count": 1,
        "out_count": len(sub_queries),
        "error": None,
    })

    return {
        "intent": intent,
        "sub_queries": sub_queries,
        "expected_titles": expected_titles,
        "sub_questions": sub_questions,
        "target_concepts": target_concepts,
        "planner_debug": plan,
        "debug_info": debug_info,
        "stage_stats": stage_stats,
    }


@traceable(name="researchlens_retrieve", run_type="retriever")
async def retrieve_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 3: Multi-Source Literature Retrieval.
    Queries arXiv, Semantic Scholar, and OpenAlex.
    If running inside a retry loop, merges newly discovered papers with existing candidates.
    """
    question = state["question"]
    sub_queries = state.get("sub_queries", [question])
    expected_titles = state.get("expected_titles", [])
    intent = state.get("intent", "literature_review")
    depth = ResearchDepth(state.get("depth", "normal"))
    sources = [ResearchSource(s) for s in state.get("sources", ["arxiv", "semantic_scholar", "openalex"])]

    retry_count = state.get("retries", 0)
    logger.info(
        "[LangGraph:Retrieve] Running candidate retrieval (retry=%d, queries=%d)...",
        retry_count,
        len(sub_queries),
    )
    t0 = time.time()

    reset_retrieval_session()
    new_raw_papers = await retrieve_papers(
        question,
        depth=depth,
        sources=sources,
        api_key_s2=settings.SEMANTIC_SCHOLAR_API_KEY,
        custom_queries=sub_queries,
        title_guesses=expected_titles,
        question_type=intent,
    )

    # Merge with any existing papers from previous attempts
    existing_papers = list(state.get("papers") or [])
    seen_identifiers = set()
    merged_papers: List[Paper] = []

    for p in existing_papers + new_raw_papers:
        key = (p.id or "") + "::" + normalize_paper_title(p.title)
        if key not in seen_identifiers:
            seen_identifiers.add(key)
            merged_papers.append(p)

    debug_info = dict(state.get("debug_info") or {})
    debug_info["papers_discovered"] = len(merged_papers)
    debug_info["source_status"] = get_source_status()

    r_errors = get_and_clear_retrieval_errors()
    if r_errors:
        debug_info["retrieval_errors"] = r_errors

    stage_stats = list(state.get("stage_stats") or [])
    stage_name = f"candidate_retrieval_retry_{retry_count}" if retry_count > 0 else "candidate_retrieval"
    stage_stats.append({
        "name": stage_name,
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "duration_ms": int((time.time() - t0) * 1000),
        "in_count": len(sub_queries),
        "out_count": len(merged_papers),
        "error": None,
    })

    return {
        "papers": merged_papers,
        "debug_info": debug_info,
        "stage_stats": stage_stats,
    }


@traceable(name="researchlens_rerank_fetch", run_type="chain")
async def rerank_fetch_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 4: Deduplication, MS-MARCO Reranking, PDF Extraction & Vector Indexing.
    Cleans watermarks, segments passages, and builds paper-scoped FAISS embeddings.
    """
    question = state["question"]
    papers = state.get("papers", [])
    intent = state.get("intent", "literature_review")
    expected_titles = state.get("expected_titles", [])
    depth = ResearchDepth(state.get("depth", "normal"))

    t0 = time.time()
    logger.info("[LangGraph:RerankFetch] Deduplicating and ranking %d candidate papers...", len(papers))

    deduped = filter_and_deduplicate(papers)
    unique_count = len(deduped)

    # Fast citation enrichment
    if get_s2_rate_limited_count() == 0 and any(getattr(p, "citationCount", 0) <= 0 for p in deduped[:30]):
        enriched_candidates = await enrich_papers_with_s2(deduped[:30]) + deduped[30:]
    else:
        enriched_candidates = deduped

    if any(getattr(p, "citationCount", 0) <= 0 for p in enriched_candidates[:25]):
        enriched_candidates = await enrich_papers_with_openalex(enriched_candidates)

    anchor_paper = None
    alternate_paper = None
    anchor_debug: Dict[str, Any] = {
        "anchor_paper_id": None,
        "anchor_rule": "none",
        "anchor_confidence": "none",
        "reason": "not_factual_lookup",
    }

    if intent == "factual_lookup":
        anchor_paper, enriched_candidates, anchor_debug = await select_anchor_paper(
            enriched_candidates,
            expected_titles,
            question=question,
        )
        alt_id = anchor_debug.get("alternate_paper_id")
        alt_title = anchor_debug.get("alternate_paper_title")
        if alt_id or alt_title:
            alternate_paper = next(
                (p for p in enriched_candidates if (alt_id and p.id == alt_id) or (alt_title and normalize_paper_title(p.title) == normalize_paper_title(alt_title))),
                None,
            )

    ranked_papers = rank_papers(
        question=question,
        papers=enriched_candidates,
        question_type=intent,
        title_guesses=expected_titles,
        depth=depth,
        anchor_paper=anchor_paper,
    )

    if anchor_paper:
        matched_p = next(
            (p for p in ranked_papers if normalize_paper_title(p.title) == normalize_paper_title(anchor_paper.title)),
            None,
        )
        anchor_id = matched_p.id if matched_p else anchor_paper.id
        anchor_debug["anchor_paper_id"] = anchor_id
        anchor_paper.is_anchor = True
        if matched_p:
            matched_p.is_anchor = True

    # PDF Full-text extraction
    logger.info("[LangGraph:RerankFetch] Extracting PDF passages for top %d papers...", len(ranked_papers))
    enriched_papers = await extract_papers_batch(ranked_papers, max_workers=settings.MAX_PDF_WORKERS)
    passages_total = sum(len(p.passages or []) for p in enriched_papers)

    # Build Vector Store
    vector_store = VectorStore()
    try:
        vector_store.build(enriched_papers, question=question)
    except Exception as exc:
        logger.warning("[LangGraph:RerankFetch] Vector store build failed: %s", exc)

    debug_info = dict(state.get("debug_info") or {})
    debug_info["unique_papers"] = unique_count
    debug_info["relevant_papers"] = len(ranked_papers)
    debug_info["anchor_paper"] = anchor_debug
    debug_info["passages_total"] = passages_total
    debug_info["top_papers"] = [p.title for p in ranked_papers[:5]]

    stage_stats = list(state.get("stage_stats") or [])
    stage_stats.append({
        "name": "relevance_rerank_and_pdf_extract",
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "duration_ms": int((time.time() - t0) * 1000),
        "in_count": len(papers),
        "out_count": passages_total,
        "error": None,
    })

    return {
        "ranked_papers": enriched_papers,
        "anchor_paper": anchor_paper,
        "alternate_paper": alternate_paper,
        "anchor_debug": anchor_debug,
        "passages_total": passages_total,
        "vector_store": vector_store,
        "unique_papers_count": unique_count,
        "debug_info": debug_info,
        "stage_stats": stage_stats,
    }


@traceable(name="researchlens_verify_claims", run_type="llm")
async def verify_claims_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 5: Two-Tier Evidence Extraction & NLI Corroboration.
    Evaluates candidate assertions against exact page sentences and DeBERTa-v3 entailment.
    Separates claims into verified vs rejected.
    """
    question = state["question"]
    sub_questions = state.get("sub_questions", [question])
    ranked_papers = state.get("ranked_papers", [])
    vector_store = state.get("vector_store")
    anchor_paper = state.get("anchor_paper")
    anchor_id = getattr(anchor_paper, "id", None)

    t0 = time.time()
    logger.info("[LangGraph:VerifyClaims] Verifying claims for %d papers...", len(ranked_papers))

    claims, citations, rejected_claims, removal_counts = extract_and_verify_evidence(
        question=question,
        sub_questions=sub_questions,
        papers=ranked_papers,
        vector_store=vector_store,
        max_citations=15,
        anchor_paper_id=anchor_id,
    )

    verified = [c for c in citations if c.status in (CitationStatus.VERIFIED, CitationStatus.PARTIALLY_SUPPORTED)]
    verified_count = sum(1 for c in citations if c.status == CitationStatus.VERIFIED)
    partial_count = sum(1 for c in citations if c.status == CitationStatus.PARTIALLY_SUPPORTED)

    debug_info = dict(state.get("debug_info") or {})
    debug_info["candidate_claims"] = len(claims) + len(rejected_claims)
    debug_info["verified_claims"] = verified_count
    debug_info["partial_claims"] = partial_count
    debug_info["rejected_claims"] = rejected_claims
    debug_info["evidence_filter_removals"] = removal_counts

    evidence_bearing = {c.paperId for c in citations if c.status == CitationStatus.VERIFIED}
    debug_info["evidence_bearing_papers"] = len(evidence_bearing)

    stage_stats = list(state.get("stage_stats") or [])
    stage_stats.append({
        "name": "evidence_verification",
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "duration_ms": int((time.time() - t0) * 1000),
        "in_count": state.get("passages_total", 0),
        "out_count": len(citations),
        "error": None,
    })

    return {
        "claims": claims,
        "citations": citations,
        "verified": verified,
        "rejected": rejected_claims,
        "evidence_bearing_papers_count": len(evidence_bearing),
        "debug_info": debug_info,
        "stage_stats": stage_stats,
    }


@traceable(name="researchlens_rewrite_query", run_type="llm")
async def rewrite_query_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 6: Self-Correction Retrieval Rewriter (The Retry Loop).
    Invoked when verified claims are below threshold and retries remain.
    Reformulates queries and increments retry counter.
    """
    question = state["question"]
    prev_queries = state.get("sub_queries", [question])
    rejected = state.get("rejected", [])
    intent = state.get("intent", "literature_review")
    current_retries = state.get("retries", 0)

    t0 = time.time()
    logger.info(
        "[LangGraph:RewriteNode] Verified claims (%d) below threshold. Triggering retry %d/%d...",
        len(state.get("verified", [])),
        current_retries + 1,
        state.get("max_retries", MAX_RETRIES),
    )

    new_sub_queries = await rewrite_search_queries(
        question=question,
        previous_queries=prev_queries,
        rejected_claims=rejected,
        intent=intent,
    )

    retry_history = list(state.get("retry_history") or [])
    retry_history.append({
        "retry_index": current_retries + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_queries": prev_queries,
        "rewritten_queries": new_sub_queries,
        "verified_before_retry": len(state.get("verified", [])),
        "rejected_before_retry": len(rejected),
    })

    stage_stats = list(state.get("stage_stats") or [])
    stage_stats.append({
        "name": f"query_rewrite_attempt_{current_retries + 1}",
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "duration_ms": int((time.time() - t0) * 1000),
        "in_count": len(prev_queries),
        "out_count": len(new_sub_queries),
        "error": None,
    })

    debug_info = dict(state.get("debug_info") or {})
    debug_info["retry_history"] = retry_history

    return {
        "sub_queries": new_sub_queries,
        "retries": current_retries + 1,
        "retry_history": retry_history,
        "debug_info": debug_info,
        "stage_stats": stage_stats,
    }


@traceable(name="researchlens_write_report", run_type="chain")
async def write_report_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node 7: Academic Synthesis & Citation Integrity Audit.
    Locks anchor citation at [1], builds comparison matrix, audits NLI sentence support.
    """
    question = state["question"]
    depth = ResearchDepth(state.get("depth", "normal"))
    ranked_papers = state.get("ranked_papers", [])
    claims = state.get("claims", [])
    citations = state.get("citations", [])
    verified = state.get("verified", [])
    anchor_paper = state.get("anchor_paper")
    anchor_debug = state.get("anchor_debug") or {}
    alternate_paper = state.get("alternate_paper")
    intent = state.get("intent", "literature_review")
    debug_info = dict(state.get("debug_info") or {})
    stage_stats_dicts = state.get("stage_stats") or []

    # Map stage stat dicts to StageStat schema objects for synthesis
    stage_stats = [
        StageStat(
            name=s.get("name", "stage"),
            started_at=s.get("started_at", datetime.now(timezone.utc).isoformat()),
            duration_ms=s.get("duration_ms", 0),
            in_count=s.get("in_count", 0),
            out_count=s.get("out_count", 0),
            error=s.get("error"),
        )
        for s in stage_stats_dicts
    ]

    min_required = MIN_VERIFIED_FACTUAL if intent == "factual_lookup" else MIN_VERIFIED_REVIEW
    limited_evidence = len(verified) < min_required

    t0 = time.time()
    logger.info(
        "[LangGraph:WriteReport] Synthesizing report (verified=%d, min_required=%d, limited=%s)...",
        len(verified),
        min_required,
        limited_evidence,
    )

    if not verified and not ranked_papers:
        # Honest fallback when zero literature was corroborated
        fail_reason = f"No corroborating empirical literature could be verified for '{question}'."
        from .research_workflow import _build_insufficient_evidence_report
        fallback_report = _build_insufficient_evidence_report(question, [], [], fail_reason)
        return {
            "report": fallback_report,
            "limited_evidence": True,
            "status": "insufficient_evidence",
            "failure_reason": fail_reason,
            "status_reasons": [fail_reason],
        }

    anchor_conf = anchor_debug.get("anchor_confidence", "none" if anchor_paper is None else "high")
    anchor_rule = anchor_debug.get("anchor_rule", "none")

    report, integrity, removed_sentences = await synthesize_report(
        question=question,
        depth=depth,
        papers=ranked_papers,
        claims=claims,
        citations=citations,
        anchor_paper=anchor_paper,
        anchor_confidence=anchor_conf,
        alternate_paper=alternate_paper,
        anchor_rule=anchor_rule,
        stage_stats=stage_stats,
        retrieval_warnings=debug_info.get("retrieval_errors"),
        debug_info=debug_info,
    )

    from .research_workflow import determine_investigation_status
    exec_sentences = split_into_sentences(report.executiveSummary)
    ver_details = debug_info.get("verified_sentences", [])
    verified_claims_in_ans = sum(1 for d in ver_details if d.get("type") in ("claim", "qualification") and d.get("status") != "removed")
    claim_sent_count = verified_claims_in_ans if verified_claims_in_ans > 0 else (1 if (integrity > 0 and len(exec_sentences) >= 1) else 0)

    anchor_is_abstract_only = getattr(anchor_paper, "is_abstract_only", False) if anchor_paper else False
    status_str, status_reasons = determine_investigation_status(
        answer_sent_count=claim_sent_count,
        integrity=integrity,
        anchor_conf=anchor_conf,
        is_abstract_only=anchor_is_abstract_only,
        confidence_note=anchor_debug.get("confidence_note"),
        candidate_count=len(ranked_papers),
        passages_total=state.get("passages_total", 0),
    )

    stage_stats_dicts.append({
        "name": "report_synthesis",
        "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "duration_ms": int((time.time() - t0) * 1000),
        "in_count": len(citations),
        "out_count": 1,
        "error": None,
    })

    return {
        "report": report,
        "limited_evidence": limited_evidence,
        "status": status_str,
        "status_reasons": status_reasons,
        "stage_stats": stage_stats_dicts,
        "debug_info": debug_info,
    }


# ─── Routing Functions (Conditional Edges) ────────────────────────────────────

def route_after_preflight(state: ResearchState) -> Literal["plan_query", "__end__"]:
    """Conditional edge after preflight: terminates early if query is rejected."""
    if state.get("allowed", False):
        return "plan_query"
    return END


def route_after_verify(state: ResearchState) -> Literal["write_report", "rewrite_query"]:
    """
    Conditional edge after verify_claims:
    If verified claims meet minimum threshold OR retries are exhausted, proceed to write_report.
    Otherwise, route to rewrite_query to self-correct retrieval.
    """
    intent = state.get("intent", "literature_review")
    min_required = MIN_VERIFIED_FACTUAL if intent == "factual_lookup" else MIN_VERIFIED_REVIEW
    verified_count = len(state.get("verified") or [])
    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)

    has_enough = verified_count >= min_required
    retries_exhausted = retries >= max_retries

    if has_enough or retries_exhausted:
        return "write_report"
    return "rewrite_query"


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_research_graph() -> Any:
    """Build and compile the LangGraph ResearchLens state machine."""
    workflow = StateGraph(ResearchState)

    # Add all processing nodes
    workflow.add_node("preflight", preflight_node)
    workflow.add_node("plan_query", plan_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank_fetch", rerank_fetch_node)
    workflow.add_node("verify_claims", verify_claims_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("write_report", write_report_node)

    # Connect edges
    workflow.add_edge(START, "preflight")
    workflow.add_conditional_edges("preflight", route_after_preflight)
    workflow.add_edge("plan_query", "retrieve")
    workflow.add_edge("retrieve", "rerank_fetch")
    workflow.add_edge("rerank_fetch", "verify_claims")
    workflow.add_conditional_edges("verify_claims", route_after_verify)
    workflow.add_edge("rewrite_query", "retrieve")  # The self-correcting retry loop
    workflow.add_edge("write_report", END)

    return workflow.compile()


# Singleton compiled graph built at module import
research_graph = build_research_graph()
