"""
Neural Reranked Retrieval Evaluation Module (eval_retriever_with_reranker.py).
Evaluates full retrieval pipeline: Multi-Source Retrieval + Deduplication + Cross-Encoder Reranking.
Metrics: Hit Rate@1, Hit Rate@5, MRR, NDCG@10, Top Rerank Scores, Latency.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.models.schemas import Paper, ResearchDepth, ResearchSource
from backend.services.paper_retrieval import retrieve_papers
from backend.services.ranker import rank_papers, normalize_paper_title
from evaluation.harness import EvaluationHarness
from evaluation.metric_registry import hit_rate_at_k, mrr_at_k, ndcg_at_k, latency_percentiles, keyword_overlap_score
from evaluation.eval_retriever import is_paper_relevant

from backend.services.query_planner import plan_queries

logger = logging.getLogger("researchlens.eval.reranker")


def eval_single_reranked(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    """Retrieves candidates and applies Cross-Encoder neural reranking."""
    question = item["question"]
    q_type = item.get("question_type", "factual_lookup")
    s2_key = getattr(settings, "SEMANTIC_SCHOLAR_API_KEY", "")
    sources = [ResearchSource.ARXIV, ResearchSource.OPENALEX]
    if s2_key:
        sources.append(ResearchSource.SEMANTIC_SCHOLAR)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    plan = loop.run_until_complete(plan_queries(question))
    title_guesses = plan.get("expected_titles", [])

    # 1. Retrieve candidates
    raw_papers = loop.run_until_complete(
        retrieve_papers(
            question=question,
            depth=ResearchDepth.STANDARD,
            sources=sources,
            custom_queries=plan.get("queries", [question]),
            title_guesses=title_guesses,
            question_type=q_type,
        )
    )

    # 2. Neural Reranking via rank_papers (incorporates CrossEncoder reranker)
    ranked = rank_papers(
        question=question,
        papers=raw_papers,
        question_type=q_type,
        title_guesses=title_guesses,
        depth=ResearchDepth.STANDARD,
    )

    relevance_flags = [is_paper_relevant(p, item) for p in ranked]
    hit_1 = bool(relevance_flags and relevance_flags[0])
    hit_5 = any(relevance_flags[:5])

    first_hit_rank = next((idx + 1 for idx, r in enumerate(relevance_flags) if r), None)
    mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0

    retrieved_ids = [p.id for p in ranked]
    ndcg = ndcg_at_k(retrieved_ids, [p.id for p, rel in zip(ranked, relevance_flags) if rel], k=10)

    top_score = getattr(ranked[0], "relevance_score", 0.0) if ranked else 0.0

    combined_titles = " ".join([getattr(p, "title", "") for p in ranked[:5]])
    kw_score = keyword_overlap_score(combined_titles, item.get("expected_facets", []))

    return {
        "query_id": item.get("id"),
        "question": question,
        "retrieved_count": len(ranked),
        "hit_at_1": hit_1,
        "hit_at_5": hit_5,
        "first_hit_rank": first_hit_rank,
        "mrr": mrr,
        "ndcg_at_10": round(ndcg, 3),
        "top_1_score": round(float(top_score), 3),
        "keyword_coverage": kw_score,
        "top_title": ranked[0].title if ranked else None
    }


def run_reranked_eval(limit: int = 10) -> Dict[str, Any]:
    """Executes Cross-Encoder reranked retrieval evaluation across benchmark dataset."""
    harness = EvaluationHarness()
    benchmark_data = harness.run_benchmark(
        name="Neural Reranked Hybrid Retrieval (arXiv + S2 + OpenAlex + CrossEncoder)",
        eval_fn=eval_single_reranked,
        limit=limit
    )

    results = benchmark_data["results"]
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    hits_1 = [1.0 if r.get("hit_at_1") else 0.0 for r in results]
    hits_5 = [1.0 if r.get("hit_at_5") else 0.0 for r in results]
    mrrs = [r.get("mrr", 0.0) for r in results]
    ndcgs = [r.get("ndcg_at_10", 0.0) for r in results]
    kw_scores = [r.get("keyword_coverage", 0.0) for r in results]
    top_scores = [r.get("top_1_score", 0.0) for r in results if "top_1_score" in r]

    summary = {
        "benchmark": "eval_retriever_with_reranker",
        "total_queries": len(results),
        "hit_rate_at_1": round(sum(hits_1) / max(len(hits_1), 1), 3),
        "hit_rate_at_5": round(sum(hits_5) / max(len(hits_5), 1), 3),
        "mrr": round(sum(mrrs) / max(len(mrrs), 1), 3),
        "ndcg_at_10": round(sum(ndcgs) / max(len(ndcgs), 1), 3),
        "avg_top_score": round(sum(top_scores) / max(len(top_scores), 1), 3),
        "avg_keyword_coverage": round(sum(kw_scores) / max(len(kw_scores), 1), 3),
        "latency_profile": latency_percentiles(latencies),
        "results": results
    }

    print("\n" + "="*60)
    print(" NEURAL RERANKED RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    print(f" Total Queries Evaluated: {summary['total_queries']}")
    print(f" Hit Rate @ 1:            {summary['hit_rate_at_1']*100:.1f}%")
    print(f" Hit Rate @ 5:            {summary['hit_rate_at_5']*100:.1f}%")
    print(f" Mean Reciprocal Rank:    {summary['mrr']:.3f}")
    print(f" NDCG @ 10:               {summary['ndcg_at_10']:.3f}")
    print(f" Avg Top Rerank Score:    {summary['avg_top_score']}")
    print(f" P50 Latency:             {summary['latency_profile']['p50']} ms")
    print(f" P90 Latency:             {summary['latency_profile']['p90']} ms")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neural Reranked Retrieval Evaluation")
    parser.add_argument("--limit", type=int, default=10, help="Max queries to evaluate")
    args = parser.parse_args()
    run_reranked_eval(limit=args.limit)
