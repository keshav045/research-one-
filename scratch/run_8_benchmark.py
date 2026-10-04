import os
os.environ["HF_HUB_OFFLINE"] = "1"
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")

from backend.config import settings
# Explicitly set required flags
settings.SEED_PAPERS_ENABLED = False
settings.DISABLE_CACHE = True

from backend.models.schemas import ResearchDepth, ResearchSource, Paper, CitationStatus
from backend.services.query_planner import plan_research_queries
from backend.services.paper_retrieval import (
    retrieve_candidate_papers,
    enrich_papers_with_s2,
    enrich_papers_with_openalex,
    get_and_clear_s2_call_records,
    get_and_clear_retrieval_errors,
)
from backend.services.ranker import (
    filter_and_deduplicate,
    select_anchor_paper,
    rank_papers,
    normalize_paper_title,
    get_reranker,
)
from backend.services.pdf_extractor import extract_papers_batch
from backend.services.vector_store import VectorStore
from backend.services.evidence import clean_hyphenated_breaks, extract_and_verify_evidence
from tests.benchmark_expected import is_expected_paper, check_paper_match

BENCHMARK_QUESTIONS = [
    {
        "topic": "Transformer",
        "question": "Which paper introduced the Transformer architecture, and what was its key idea?",
        "is_nonsense": False,
    },
    {
        "topic": "GAN",
        "question": "Who introduced generative adversarial networks?",
        "is_nonsense": False,
    },
    {
        "topic": "Adam",
        "question": "Which paper proposed the Adam optimizer?",
        "is_nonsense": False,
    },
    {
        "topic": "RAG",
        "question": "What is retrieval-augmented generation and who proposed it?",
        "is_nonsense": False,
    },
    {
        "topic": "Dropout",
        "question": "Which paper introduced dropout?",
        "is_nonsense": False,
    },
    {
        "topic": "BatchNorm",
        "question": "Who introduced batch normalization?",
        "is_nonsense": False,
    },
    {
        "topic": "ViT",
        "question": "Which paper introduced the Vision Transformer?",
        "is_nonsense": False,
    },
    {
        "topic": "word2vec",
        "question": "Which paper introduced word2vec?",
        "is_nonsense": False,
    },
    {
        "topic": "BERT",
        "question": "Which paper introduced BERT?",
        "is_nonsense": False,
    },
    {
        "topic": "U-Net",
        "question": "Which paper introduced U-Net?",
        "is_nonsense": False,
    },
    {
        "topic": "Faster R-CNN",
        "question": "Which paper introduced Faster R-CNN?",
        "is_nonsense": False,
    },
    {
        "topic": "Latent Diffusion",
        "question": "Which paper introduced High-Resolution Image Synthesis with Latent Diffusion Models?",
        "is_nonsense": False,
    },
    {
        "topic": "Nonsense",
        "question": "What is the capital of the Moon's mayor?",
        "is_nonsense": True,
    },
    {
        "topic": "DenseNet",
        "question": "Which paper introduced DenseNet?",
        "is_nonsense": False,
    },
    {
        "topic": "Mask R-CNN",
        "question": "Which paper introduced Mask R-CNN?",
        "is_nonsense": False,
    },
    {
        "topic": "Graph Convolutional Networks",
        "question": "Which paper introduced Graph Convolutional Networks?",
        "is_nonsense": False,
    },
    {
        "topic": "T5",
        "question": "Which paper introduced T5?",
        "is_nonsense": False,
    },
    {
        "topic": "YOLO",
        "question": "Which paper introduced YOLO?",
        "is_nonsense": False,
    },
]


async def run_single_question(item: dict, reranker: Any) -> dict:
    topic = item["topic"]
    q = item["question"]
    is_nonsense = item.get("is_nonsense", False)

    logger.info("=" * 80)
    logger.info("BENCHMARK QUESTION: [%s] '%s'", topic, q)
    logger.info("=" * 80)

    # Clear prior error tracking
    get_and_clear_s2_call_records()
    get_and_clear_retrieval_errors()

    # 1. Query Planning
    plan = await plan_research_queries(q)
    q_type = plan.get("question_type", "factual_lookup")
    queries = plan.get("queries", [q])
    title_guesses = plan.get("title_guesses", [])
    sub_questions = plan.get("sub_questions", [q])
    sub_q_1 = sub_questions[0] if sub_questions else q

    logger.info("Plan: type=%s, queries=%s, title_guesses=%s, sub_q1='%s'", q_type, queries, title_guesses, sub_q_1)

    # 2. Retrieval
    sources = [ResearchSource.ARXIV, ResearchSource.SEMANTIC_SCHOLAR, ResearchSource.OPENALEX]
    raw_candidates, _ = await retrieve_candidate_papers(q, ResearchDepth.STANDARD, sources, plan=plan)

    arxiv_raw = sum(1 for p in raw_candidates if p.source == "arXiv")
    openalex_raw = sum(1 for p in raw_candidates if p.source == "OpenAlex")
    s2_raw = sum(1 for p in raw_candidates if p.source == "Semantic Scholar")

    # Check which raw candidate sources supplied the expected paper
    if not is_nonsense:
        arxiv_has = any(is_expected_paper(p, topic) for p in raw_candidates if p.source == "arXiv")
        s2_has = any(is_expected_paper(p, topic) for p in raw_candidates if p.source == "Semantic Scholar")
        openalex_has = any(is_expected_paper(p, topic) for p in raw_candidates if p.source == "OpenAlex")

        sources_found = []
        if s2_has:
            sources_found.append("Semantic Scholar")
        if openalex_has:
            sources_found.append("OpenAlex")
        if arxiv_has:
            sources_found.append("arXiv")
        real_supplying_source = ", ".join(sources_found) if sources_found else "None"
    else:
        real_supplying_source = "N/A (Nonsense)"

    # 3. Deduplication
    deduped = filter_and_deduplicate(raw_candidates)

    # 4. Enrichment
    enriched = await enrich_papers_with_s2(deduped)
    enriched = await enrich_papers_with_openalex(enriched)

    # 5. Anchor Paper Selection (includes Key Term highest citations and Citation Chasing)
    anchor_paper, enriched, anchor_debug = await select_anchor_paper(enriched, title_guesses, question=q)
    anchor_rule = anchor_debug.get("anchor_rule", "none")

    # Collect S2 call records for this question
    s2_records = get_and_clear_s2_call_records()
    s2_200 = sum(1 for r in s2_records if r.get("status_code") == 200)
    s2_429 = sum(1 for r in s2_records if r.get("status_code") == 429)
    s2_call_summary = f"{len(s2_records)} calls (200: {s2_200}, 429: {s2_429})"

    # 6. Ranking (Cut to Depth count = 12, guaranteed anchor & top 3 key-term candidates)
    ranked_papers = rank_papers(
        question=q,
        papers=enriched,
        question_type=q_type,
        title_guesses=title_guesses,
        depth=ResearchDepth.STANDARD,
        anchor_paper=anchor_paper,
    )

    # 7. Check Expected Paper match strictly by ID, DOI, or exact normalized title
    real_paper_rank = None
    real_paper_found = None
    uncut_rank = None

    if not is_nonsense:
        for idx, p in enumerate(ranked_papers, 1):
            if is_expected_paper(p, topic):
                real_paper_rank = idx
                real_paper_found = p
                break

        if real_paper_rank is None:
            for idx, p in enumerate(enriched, 1):
                if is_expected_paper(p, topic):
                    uncut_rank = idx
                    real_paper_found = p
                    break

        if real_paper_rank:
            found_str = f"Found in Final (Rank #{real_paper_rank})"
            rank_str = f"Rank #{real_paper_rank}"
        elif uncut_rank:
            found_str = f"Found in Pool (pos #{uncut_rank})"
            rank_str = f"In candidates (pos #{uncut_rank})"
        else:
            found_str = "MISS"
            rank_str = "MISS"

        if real_paper_found and real_supplying_source == "None":
            if getattr(real_paper_found, "source", "") == "Semantic Scholar":
                real_supplying_source = "Semantic Scholar (Citation Chasing)"
            elif getattr(real_paper_found, "source", "") == "arXiv":
                real_supplying_source = "arXiv"
            elif getattr(real_paper_found, "source", "") == "OpenAlex":
                real_supplying_source = "OpenAlex"
            else:
                real_supplying_source = getattr(real_paper_found, "source", "Unknown")
    else:
        found_str = "N/A (Nonsense)"
        rank_str = "N/A"

    # 8. Full-Text / Abstract Extraction & Passage Indexing (eval top 5 papers + anchor)
    eval_papers = list(ranked_papers[:5])
    if anchor_paper and not any(p.id == anchor_paper.id for p in eval_papers):
        eval_papers.insert(0, anchor_paper)

    if anchor_paper:
        setattr(anchor_paper, "is_anchor", True)

    extracted = await extract_papers_batch(eval_papers, max_workers=settings.MAX_PDF_WORKERS)
    vector_store = VectorStore()
    vector_store.build(extracted)

    # Count anchor passages indexed & source kind
    anchor_passages_indexed = 0
    anchor_source_kind = "none"
    if anchor_paper:
        for p in extracted:
            if p.id == anchor_paper.id:
                anchor_passages_indexed = len(p.passages or [])
                if p.passages:
                    anchor_source_kind = getattr(p.passages[0], "source_kind", "abstract")
                break

    # 9. Top 5 passages for sub-question 1 (Fix B.3)
    p_map = {p.id: p for p in extracted}
    faiss_hits = vector_store.search(sub_q_1, top_k=25)
    anchor_id = anchor_paper.id if anchor_paper else None

    anchor_scored: list[tuple[Any, float]] = []
    if anchor_id:
        anchor_records = [r for r in vector_store.records if r.paper_id == anchor_id]
        if anchor_records:
            if reranker:
                pairs = [[sub_q_1, clean_hyphenated_breaks(r.passage.text)] for r in anchor_records]
                scores = reranker.predict(pairs)
                anchor_scored = [(r, float(s)) for r, s in zip(anchor_records, scores)]
            else:
                anchor_scored = [(r, 0.5) for r in anchor_records]
            anchor_scored.sort(key=lambda x: x[1], reverse=True)

    chosen: list[tuple[Any, float, bool]] = []
    if len(anchor_scored) >= 3:
        # If anchor has >= 3 relevant passages, ALL top-5 passages per sub-question must come from the anchor
        chosen = [(r, s, False) for r, s in anchor_scored[:5]]
    else:
        # Take all available anchor passages
        chosen = [(r, s, False) for r, s in anchor_scored]
        # Fill remainder up to 5 from other papers, labeled as secondary
        cand_records = [r for r, _ in faiss_hits if not (anchor_id and r.paper_id == anchor_id)]
        if cand_records:
            if reranker:
                pairs = [[sub_q_1, clean_hyphenated_breaks(r.passage.text)] for r in cand_records]
                scores = reranker.predict(pairs)
                other_scored = [(r, float(s)) for r, s in zip(cand_records, scores)]
            else:
                other_scored = [(r, sc) for r, sc in faiss_hits]
            other_scored.sort(key=lambda x: x[1], reverse=True)
            for item in other_scored:
                if len(chosen) >= 5:
                    break
                if not any(item[0].passage.id == cp[0].passage.id for cp in chosen):
                    chosen.append((item[0], item[1], True))

    anchor_passages_in_top5 = sum(1 for cp in chosen[:5] if not cp[2])

    top_passages_display = []
    for r, _, is_sec in chosen[:5]:
        p_obj = p_map.get(r.paper_id)
        if p_obj:
            t = p_obj.title
            if is_sec:
                t += " [secondary]"
            top_passages_display.append(t)

    # Determine anchor string and match status
    match_type = "no"
    if anchor_paper:
        match_type = check_paper_match(anchor_paper, topic) if not is_nonsense else "N/A"
        match_tag = f"[{match_type.upper()}]"
        anchor_str = f"{anchor_paper.title} ({anchor_paper.citationCount:,} cites) {match_tag}"
    else:
        match_type = "N/A" if is_nonsense else "no"
        anchor_str = "None" if not is_nonsense else "None (Expected for nonsense)"

    anchor_confidence = anchor_debug.get("anchor_confidence", "high")
    confidence_note = anchor_debug.get("confidence_note", "")

    # Handle nonsense question verification & gate check outcome
    nonsense_outcome = None
    if is_nonsense:
        claims, citations, rejected, removals = extract_and_verify_evidence(
            question=q,
            sub_questions=sub_questions,
            papers=extracted,
            vector_store=vector_store,
            max_citations=5,
            anchor_paper_id=None,
        )
        verified_count = sum(1 for c in citations if c.status == CitationStatus.VERIFIED)
        partial_count = sum(1 for c in citations if c.status == CitationStatus.PARTIALLY_SUPPORTED)
        gate_passed = (verified_count + partial_count) >= 1
        fail_reason = (
            f"Evidence gate failed: Found {verified_count} verified and {partial_count} partial claims. "
            f"No academic literature supports a mayor or capital of the Moon."
        )
        nonsense_outcome = {
            "status": "insufficient_evidence" if not gate_passed else "passed",
            "reason": fail_reason,
            "anchor_is_none": (anchor_paper is None),
        }
        logger.info("[NonsenseCheck] %s", fail_reason)

    logger.info(
        "Result: %s | Match: %s | Rank: %s | Anchor: %s (Rule: %s, Conf: %s) | S2 calls: %s",
        topic, match_type, rank_str, anchor_str, anchor_rule, anchor_confidence, s2_call_summary
    )

    return {
        "topic": topic,
        "question": q,
        "candidates_per_source": f"arXiv: {arxiv_raw}, S2: {s2_raw}, OpenAlex: {openalex_raw}",
        "s2_status": s2_call_summary,
        "s2_200": s2_200,
        "s2_429": s2_429,
        "real_supplying_source": real_supplying_source,
        "expected_found": found_str,
        "rank_in_final": rank_str,
        "expected_paper_title": real_paper_found.title if real_paper_found else ("N/A (Nonsense)" if is_nonsense else "MISS"),
        "match_type": match_type,
        "anchor_chosen": anchor_str,
        "anchor_rule": anchor_rule,
        "anchor_confidence": anchor_confidence,
        "confidence_note": confidence_note,
        "anchor_passages_indexed": anchor_passages_indexed,
        "anchor_source_kind": anchor_source_kind,
        "anchor_passages_in_top5": f"{anchor_passages_in_top5}/5",
        "top_passages_subq1": top_passages_display,
        "is_nonsense": is_nonsense,
        "nonsense_outcome": nonsense_outcome,
    }


async def main():
    logger.info("================================================================================")
    has_s2_key = bool(settings.SEMANTIC_SCHOLAR_API_KEY and settings.SEMANTIC_SCHOLAR_API_KEY.strip())
    logger.info("S2 key loaded: %s", has_s2_key)
    logger.info("================================================================================")

    logger.info("Initializing Cross-Encoder Reranker...")
    reranker = get_reranker()
    logger.info("Reranker loaded successfully.")

    results = []
    out_path = Path("scratch/benchmark_9_results.json")
    for item in BENCHMARK_QUESTIONS:
        t0 = time.time()
        try:
            # 240s timeout per question
            res = await asyncio.wait_for(run_single_question(item, reranker), timeout=240.0)
            res["duration_s"] = round(time.time() - t0, 1)
            results.append(res)
        except asyncio.TimeoutError:
            logger.error("Question '%s' TIMED OUT after 240s", item["topic"])
            results.append({
                "topic": item["topic"],
                "question": item["question"],
                "candidates_per_source": "TIMEOUT",
                "s2_status": "TIMEOUT",
                "s2_200": 0,
                "s2_429": 0,
                "real_supplying_source": "TIMEOUT",
                "expected_found": "TIMEOUT",
                "rank_in_final": "TIMEOUT",
                "expected_paper_title": "TIMEOUT",
                "match_type": "TIMEOUT",
                "anchor_chosen": "TIMEOUT",
                "anchor_rule": "none",
                "anchor_confidence": "none",
                "top_passages_subq1": [],
                "duration_s": round(time.time() - t0, 1),
            })
        except Exception as exc:
            logger.error("Question '%s' failed with exception: %s", item["topic"], exc, exc_info=True)
            results.append({
                "topic": item["topic"],
                "question": item["question"],
                "candidates_per_source": f"ERROR: {exc}",
                "s2_status": "ERROR",
                "s2_200": 0,
                "s2_429": 0,
                "real_supplying_source": "ERROR",
                "expected_found": "ERROR",
                "rank_in_final": "ERROR",
                "expected_paper_title": "ERROR",
                "match_type": "ERROR",
                "anchor_chosen": "ERROR",
                "anchor_rule": "none",
                "anchor_confidence": "none",
                "top_passages_subq1": [],
                "duration_s": round(time.time() - t0, 1),
            })
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        await asyncio.sleep(2.0)

    logger.info("Benchmark complete! Saved to %s", out_path)

    total_200 = sum(r.get("s2_200", 0) for r in results)
    total_429 = sum(r.get("s2_429", 0) for r in results)

    print("\n" + "=" * 160)
    print(f"BENCHMARK RESULTS TABLE (S2 key loaded: {has_s2_key} | Total S2 Calls: 200={total_200}, 429={total_429})")
    print("=" * 160)
    print("| Topic / Question | Match Type (Exact/Acceptable/No) | Anchor Chosen | Rule Used | Anchor Confidence | Anchor Passages Indexed + Source Kind | Top 5 Passages from Anchor | S2 429 Count | Nonsense Status & Failure Reason (if applicable) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        nonsense_info = "N/A"
        if r.get("is_nonsense") and r.get("nonsense_outcome"):
            out = r["nonsense_outcome"]
            nonsense_info = f"Status: `{out.get('status')}`<br>Reason: {out.get('reason')}"
        print(f"| **{r['topic']}**<br>*{r['question']}* | **{r.get('match_type', 'no')}** | {r.get('anchor_chosen', 'None')} | `{r.get('anchor_rule', 'none')}` | `{r.get('anchor_confidence', 'high')}` | {r.get('anchor_passages_indexed', 0)} ({r.get('anchor_source_kind', 'none')}) | {r.get('anchor_passages_in_top5', '0/5')} | {r.get('s2_429', 0)} | {nonsense_info} |")
    print("=" * 160 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
