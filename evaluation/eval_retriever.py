"""
Baseline Retrieval Evaluation Module (eval_retriever.py).
Evaluates multi-source paper retrieval across arXiv, Semantic Scholar, and OpenAlex
using baseline lexical and reciprocal-rank fusion without neural cross-encoder reranker.
Metrics: Hit Rate@1, Hit Rate@5, Recall@5, Recall@10, MRR, Latency.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.models.schemas import Paper, ResearchDepth, ResearchSource
from backend.services.paper_retrieval import retrieve_papers
from backend.services.ranker import normalize_paper_title
from evaluation.harness import EvaluationHarness
from evaluation.metric_registry import hit_rate_at_k, mrr_at_k, recall_at_k, latency_percentiles, keyword_overlap_score

from backend.services.query_planner import plan_queries

logger = logging.getLogger("researchlens.eval.retriever")


def rank_papers_lexical_only(papers: List[Paper], question: str) -> List[Paper]:
    """Lexical token and keyword overlap ranking without neural cross-encoder."""
    q_words = set(re.findall(r"\b[a-z]{3,}\b", question.lower()))
    def _score(p: Paper) -> float:
        title_words = set(re.findall(r"\b[a-z]{3,}\b", (p.title or "").lower()))
        overlap = len(q_words.intersection(title_words)) / max(len(q_words), 1)
        cites = min(1.0, math.log10(max(1, getattr(p, "citationCount", 0)) + 1) / 5.0)
        return 0.70 * overlap + 0.30 * cites
    return sorted(papers, key=_score, reverse=True)


def is_paper_relevant(paper: Paper, item: Dict[str, Any]) -> bool:
    """Checks whether a paper candidate matches ground-truth arXiv ID, DOI, or normalized title."""
    paper_arxiv = (getattr(paper, "arxiv_id", None) or "").lower().strip()
    paper_id = (getattr(paper, "id", "") or "").lower().strip()
    paper_doi = (getattr(paper, "doi", "") or "").lower().strip()
    norm_title = normalize_paper_title(getattr(paper, "title", ""))

    for exp_arx in item.get("expected_arxiv", []):
        exp = exp_arx.lower().strip()
        if exp and (exp in paper_arxiv or exp in paper_id):
            return True

    for exp_doi in item.get("expected_dois", []):
        exp = exp_doi.lower().strip()
        if exp and (exp in paper_doi or exp in paper_id):
            return True

    for exp_title in item.get("expected_titles", []):
        exp_norm = normalize_paper_title(exp_title)
        if exp_norm and (exp_norm in norm_title or norm_title in exp_norm):
            return True
        words_exp = set(exp_norm.split())
        words_paper = set(norm_title.split())
        if words_exp and (len(words_exp.intersection(words_paper)) / len(words_exp)) >= 0.65:
            return True

    return False


def eval_single_retrieval(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    """Retrieves candidates for a single benchmark question without cross-encoder reranking."""
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

    ranked = rank_papers_lexical_only(raw_papers, question)
    
    relevance_flags = [is_paper_relevant(p, item) for p in ranked]
    hit_1 = bool(relevance_flags and relevance_flags[0])
    hit_5 = any(relevance_flags[:5])
    
    first_hit_rank = next((idx + 1 for idx, r in enumerate(relevance_flags) if r), None)
    mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0

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
        "keyword_coverage": kw_score,
        "top_title": ranked[0].title if ranked else None
    }


def run_retrieval_eval(limit: int = 10) -> Dict[str, Any]:
    """Executes baseline retrieval evaluation across benchmark dataset."""
    harness = EvaluationHarness()
    benchmark_data = harness.run_benchmark(
        name="Baseline Multi-Source Retrieval (arXiv + S2 + OpenAlex + Lexical)",
        eval_fn=eval_single_retrieval,
        limit=limit
    )

    results = benchmark_data["results"]
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    hits_1 = [1.0 if r.get("hit_at_1") else 0.0 for r in results]
    hits_5 = [1.0 if r.get("hit_at_5") else 0.0 for r in results]
    mrrs = [r.get("mrr", 0.0) for r in results]
    kw_scores = [r.get("keyword_coverage", 0.0) for r in results]

    summary = {
        "benchmark": "eval_retriever",
        "total_queries": len(results),
        "hit_rate_at_1": round(sum(hits_1) / max(len(hits_1), 1), 3),
        "hit_rate_at_5": round(sum(hits_5) / max(len(hits_5), 1), 3),
        "mrr": round(sum(mrrs) / max(len(mrrs), 1), 3),
        "avg_keyword_coverage": round(sum(kw_scores) / max(len(kw_scores), 1), 3),
        "latency_profile": latency_percentiles(latencies),
        "results": results
    }

    print("\n" + "="*60)
    print(" BASELINE RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    print(f" Total Queries Evaluated: {summary['total_queries']}")
    print(f" Hit Rate @ 1:            {summary['hit_rate_at_1']*100:.1f}%")
    print(f" Hit Rate @ 5:            {summary['hit_rate_at_5']*100:.1f}%")
    print(f" Mean Reciprocal Rank:    {summary['mrr']:.3f}")
    print(f" Avg Keyword Coverage:    {summary['avg_keyword_coverage']*100:.1f}%")
    print(f" P50 Latency:             {summary['latency_profile']['p50']} ms")
    print(f" P90 Latency:             {summary['latency_profile']['p90']} ms")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline Retrieval Evaluation")
    parser.add_argument("--limit", type=int, default=10, help="Max queries to evaluate")
    args = parser.parse_args()
    run_retrieval_eval(limit=args.limit)
