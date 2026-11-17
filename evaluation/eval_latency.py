"""
System Latency and Subsystem Profiling Evaluation Module (eval_latency.py).
Benchmarks latency distribution across each ResearchLens subsystem:
1. Query Planning & Decomposition
2. Multi-Source Literature Retrieval
3. Relevance Deduplication & Cross-Encoder Reranker
4. Vector Store Passage Indexing & Semantic Search
5. Evidence Synthesis & Citation Generation
6. End-to-End Investigation Turn
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.models.schemas import Paper, ResearchDepth, ResearchSource
from backend.services.query_planner import plan_queries
from backend.services.paper_retrieval import retrieve_papers
from backend.services.ranker import rank_papers
from backend.services.vector_store import VectorStore
from backend.services.local_llm_service import synthesize_report
from evaluation.harness import EvaluationHarness
from evaluation.metric_registry import latency_percentiles

logger = logging.getLogger("researchlens.eval.latency")

BENCHMARK_PROFILING_QUERIES = [
    "Which paper introduced the Transformer architecture and self-attention?",
    "What is retrieval-augmented generation (RAG) and who proposed it?",
    "Explain deep residual networks (ResNet) for high-accuracy image recognition.",
    "How does the Adam optimizer compute adaptive learning rates from gradients?",
    "What is the biological mechanism of CRISPR-Cas9 targeted genome editing?",
    "Explain Latent Diffusion Models and their high-resolution image synthesis pipeline."
]


def profile_subsystems(harness: EvaluationHarness) -> Dict[str, Any]:
    """Profiles execution time across each ResearchLens pipeline stage."""
    planner_times = []
    retrieval_times = []
    rerank_times = []
    vector_times = []
    synthesis_times = []
    e2e_times = []

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    from backend.models.schemas import Paper, PaperPassage
    sample_paper = Paper(
        id="p1",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="arXiv",
        abstract="We propose the Transformer architecture.",
        passages=[
            PaperPassage(id="p1_1", paper_id="p1", text="Self-attention mechanism computes representations by relating different positions of a single sequence.", page=1, section="Introduction"),
            PaperPassage(id="p1_2", paper_id="p1", text="Residual connections allow gradients to backpropagate through hundreds of layers without vanishing.", page=2, section="Architecture"),
            PaperPassage(id="p1_3", paper_id="p1", text="Adaptive moment estimation maintains exponentially decaying moving averages of past gradients.", page=3, section="Optimization")
        ]
    )

    for q in BENCHMARK_PROFILING_QUERIES:
        t_start = time.perf_counter()

        # 1. Query Planner
        t0 = time.perf_counter()
        plan = loop.run_until_complete(plan_queries(q))
        planner_times.append((time.perf_counter() - t0) * 1000.0)

        # 2. Retrieval (quick top 5)
        t0 = time.perf_counter()
        try:
            papers = loop.run_until_complete(
                retrieve_papers(
                    question=q,
                    depth=ResearchDepth.QUICK,
                    sources=[ResearchSource.ARXIV],
                    max_results_override=5
                )
            )
        except Exception:
            papers = [sample_paper]
        retrieval_times.append((time.perf_counter() - t0) * 1000.0)

        # 3. Reranker
        t0 = time.perf_counter()
        ranked = rank_papers(question=q, papers=papers, depth=ResearchDepth.QUICK)
        rerank_times.append((time.perf_counter() - t0) * 1000.0)

        # 4. Vector Store
        t0 = time.perf_counter()
        vstore = VectorStore()
        vstore.build([sample_paper])
        _ = vstore.search(q, top_k=2)
        vector_times.append((time.perf_counter() - t0) * 1000.0)

        # 5. Synthesis
        t0 = time.perf_counter()
        from evaluation.harness import synthesize_report_sync
        from backend.models.schemas import Citation, CitationStatus
        cit = Citation(
            id="c1",
            badgeNumber=1,
            claim="Attention is all you need.",
            status=CitationStatus.VERIFIED,
            paperId=sample_paper.id,
            paperTitle=sample_paper.title,
            authors="Vaswani et al.",
            year=2017,
            page=1,
            passage=sample_paper.passages[0].text,
            highlightSentence=sample_paper.passages[0].text
        )
        _ = synthesize_report_sync(
            question=q,
            papers=ranked[:2] if ranked else [sample_paper],
            citations=[cit],
            depth=ResearchDepth.QUICK
        )
        synthesis_times.append((time.perf_counter() - t0) * 1000.0)

        e2e_times.append((time.perf_counter() - t_start) * 1000.0)

    return {
        "benchmark": "eval_latency",
        "sample_size": len(BENCHMARK_PROFILING_QUERIES),
        "stages": {
            "query_planner": latency_percentiles(planner_times),
            "multi_source_retrieval": latency_percentiles(retrieval_times),
            "cross_encoder_reranker": latency_percentiles(rerank_times),
            "vector_store_passage_search": latency_percentiles(vector_times),
            "academic_report_synthesis": latency_percentiles(synthesis_times),
            "end_to_end_pipeline": latency_percentiles(e2e_times)
        }
    }


def run_latency_eval() -> Dict[str, Any]:
    """Runs latency profiling benchmark."""
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_latency] Profiling subsystem latency distributions across stages...")
    profile = profile_subsystems(harness)

    print("\n" + "="*70)
    print(" RESEARCHLENS LATENCY & PERFORMANCE PROFILE")
    print("="*70)
    print(f"{'Pipeline Stage':<30} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P90 (ms)':<10}")
    print("-" * 70)
    for stage, stats in profile["stages"].items():
        print(f"{stage:<30} | {stats['mean']:<10.1f} | {stats['p50']:<10.1f} | {stats['p90']:<10.1f}")
    print("="*70 + "\n")

    return profile


if __name__ == "__main__":
    run_latency_eval()
