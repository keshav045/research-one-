"""
Generator Evaluation Module (eval_generator.py).
Evaluates LLM academic research synthesis against the ResearchLens 6-Point Academic Rubric:
1. Rigorous Technical Overview
2. Core Methodology & Mechanism
3. Mathematical / Theoretical Formulation
4. Empirical Benchmarks & Quantitative Results
5. Limitations & Critical Future Work Discussion
6. Bracketed Academic Citations ([1], [2]) linked to verified references.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.models.schemas import Paper, ResearchDepth, ResearchReport, Citation, CitationStatus
from evaluation.harness import EvaluationHarness, synthesize_report_sync
from evaluation.metric_registry import rubric_adherence_score, latency_percentiles

logger = logging.getLogger("researchlens.eval.generator")


def eval_single_generation(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    """Generates an academic research synthesis for a query and evaluates rubric adherence."""
    question = item["question"]
    expected_answer = item.get("expected_answer", "")
    
    paper_title = item.get("expected_titles", ["Foundational Research Paper"])[0]
    paper_id = item.get("expected_arxiv", ["arxiv:1706.03762"])[0]

    paper = Paper(
        id=paper_id,
        title=paper_title,
        authors=["Vaswani et al." if "transformer" in question.lower() else "Lead Author et al."],
        publicationYear=2017 if "transformer" in question.lower() else 2021,
        journalConference="NeurIPS",
        doi=item.get("expected_dois", ["10.48550/arxiv.1706.03762"])[0],
        source="arXiv",
        abstract=expected_answer or "Detailed empirical research and theoretical formulation of academic architecture."
    )

    citations = [
        Citation(
            id=f"cit_1_{item.get('id', 'q')}",
            badgeNumber=1,
            claim=expected_answer or "The architecture introduces multi-head self-attention mechanisms without sequential recurrence.",
            status=CitationStatus.VERIFIED,
            paperId=paper.id,
            paperTitle=paper.title,
            authors=paper.authors[0],
            year=paper.publicationYear,
            page=1,
            passage=expected_answer or "We propose the Transformer, an architecture based entirely on attention mechanisms.",
            highlightSentence=expected_answer or "We propose the Transformer, an architecture based entirely on attention mechanisms."
        ),
        Citation(
            id=f"cit_2_{item.get('id', 'q')}",
            badgeNumber=2,
            claim="Empirical evaluations on standard benchmark datasets demonstrate significant quantitative accuracy gains over prior baselines.",
            status=CitationStatus.VERIFIED,
            paperId=paper.id,
            paperTitle=paper.title,
            authors=paper.authors[0],
            year=paper.publicationYear,
            page=2,
            passage="Empirical evaluations on standard benchmark datasets demonstrate significant quantitative accuracy gains.",
            highlightSentence="Empirical evaluations on standard benchmark datasets demonstrate significant quantitative accuracy gains."
        )
    ]

    report, lat_sec, _ = synthesize_report_sync(
        question=question,
        papers=[paper],
        citations=citations,
        depth=ResearchDepth.QUICK,
        anchor_paper=paper
    )

    answer_text = report.executiveSummary if hasattr(report, "executiveSummary") else str(report)
    rubric = rubric_adherence_score(answer_text)

    return {
        "query_id": item.get("id"),
        "question": question,
        "answer_length": len(answer_text),
        "citations_count": len(getattr(report, "references", [])) or len(citations),
        "rubric_score": rubric["score"],
        "rubric_percentage": rubric["percentage"],
        "has_overview": rubric["breakdown"]["technical_overview"],
        "has_mechanism": rubric["breakdown"]["core_mechanism"],
        "has_math": rubric["breakdown"]["mathematical_formulation"],
        "has_benchmarks": rubric["breakdown"]["empirical_benchmarks"],
        "has_limitations": rubric["breakdown"]["limitations_future_work"],
        "has_citations": rubric["breakdown"]["academic_citations"]
    }


def run_generator_eval(limit: int = 10) -> Dict[str, Any]:
    """Runs generation evaluation across benchmark dataset."""
    harness = EvaluationHarness()
    benchmark_data = harness.run_benchmark(
        name="ResearchLens Academic Answer Generator Evaluation",
        eval_fn=eval_single_generation,
        limit=limit
    )

    results = benchmark_data["results"]
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    scores = [r.get("rubric_score", 0.0) for r in results]
    percentages = [r.get("rubric_percentage", 0.0) for r in results]

    overview_rate = sum(1.0 for r in results if r.get("has_overview")) / max(len(results), 1)
    mechanism_rate = sum(1.0 for r in results if r.get("has_mechanism")) / max(len(results), 1)
    math_rate = sum(1.0 for r in results if r.get("has_math")) / max(len(results), 1)
    benchmarks_rate = sum(1.0 for r in results if r.get("has_benchmarks")) / max(len(results), 1)
    limitations_rate = sum(1.0 for r in results if r.get("has_limitations")) / max(len(results), 1)
    citations_rate = sum(1.0 for r in results if r.get("has_citations")) / max(len(results), 1)

    summary = {
        "benchmark": "eval_generator",
        "total_evaluated": len(results),
        "mean_rubric_score": round(sum(scores) / max(len(scores), 1), 2),
        "mean_rubric_percentage": round(sum(percentages) / max(len(percentages), 1), 1),
        "overview_rate": round(overview_rate, 3),
        "mechanism_rate": round(mechanism_rate, 3),
        "math_rate": round(math_rate, 3),
        "benchmarks_rate": round(benchmarks_rate, 3),
        "limitations_rate": round(limitations_rate, 3),
        "citations_rate": round(citations_rate, 3),
        "latency_profile": latency_percentiles(latencies),
        "results": results
    }

    print("\n" + "="*60)
    print(" GENERATOR RUBRIC ADHERENCE EVALUATION RESULTS")
    print("="*60)
    print(f" Total Evaluated:         {summary['total_evaluated']}")
    print(f" Mean Rubric Score:       {summary['mean_rubric_score']} / 6.0 ({summary['mean_rubric_percentage']}%)")
    print(f" Technical Overview Rate: {summary['overview_rate']*100:.1f}%")
    print(f" Core Mechanism Rate:     {summary['mechanism_rate']*100:.1f}%")
    print(f" Math Formulation Rate:   {summary['math_rate']*100:.1f}%")
    print(f" Empirical Benchmarks:    {summary['benchmarks_rate']*100:.1f}%")
    print(f" Limitations / Future:    {summary['limitations_rate']*100:.1f}%")
    print(f" Academic Citations Rate: {summary['citations_rate']*100:.1f}%")
    print(f" P50 Latency:             {summary['latency_profile']['p50']} ms")
    print(f" P90 Latency:             {summary['latency_profile']['p90']} ms")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generator Rubric Evaluation")
    parser.add_argument("--limit", type=int, default=10, help="Max queries to evaluate")
    args = parser.parse_args()
    run_generator_eval(limit=args.limit)
