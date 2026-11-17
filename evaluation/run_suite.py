"""
Master Evaluation Suite Orchestrator (run_suite.py).
Executes the comprehensive ResearchLens evaluation architecture across all 13 evaluation modules,
aggregates the complete scorecard into 'evaluation_report.json',
produces the executive audit summary 'FULL_EVAL_REPORT.md',
and triggers 'promote.py' to evaluate production promotion gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import all evaluation modules
from evaluation.eval_ops import run_ops_eval
from evaluation.eval_retriever import run_retrieval_eval
from evaluation.eval_retriever_with_reranker import run_reranked_eval
from evaluation.eval_generator import run_generator_eval
from evaluation.eval_rag_pipeline import run_rag_eval
from evaluation.eval_leakage import run_leakage_eval
from evaluation.eval_safety import run_safety_eval
from evaluation.eval_scope import run_scope_eval
from evaluation.eval_toxicity import run_toxicity_eval
from evaluation.eval_reliability import run_reliability_eval
from evaluation.eval_latency import run_latency_eval
from evaluation.eval_cost import run_cost_eval
from evaluation.eval_application import run_application_eval
from evaluation.promote import evaluate_promotion_gates


def generate_markdown_report(scorecard: Dict[str, Any], promo_result: Dict[str, Any], output_path: Path):
    """Writes a GitHub Flavored Markdown report detailing all evaluation dimensions."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    md_lines = [
        "# ResearchLens — Comprehensive Production Evaluation Report",
        f"**Generated:** {ts} | **Evaluation Dataset:** `dataset.json` (30 Authentic Academic Research Benchmark Cases)",
        f"**Deployment Decision:** `{promo_result['decision']}`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Production Promotion Gates",
        "",
        "| Gate Name | Measured Value | Requirement | Gate Status |",
        "| :--- | :--- | :--- | :--- |"
    ]

    for g in promo_result["gates"]:
        st = "PASS" if g["passed"] else "FAIL"
        md_lines.append(f"| {g['name']} | {g['value']} | {g['threshold']} | **{st}** |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Multi-Source Retrieval & Reranker Performance",
        "",
        "| Retrieval Mode | Top-1 Precision (Hit@1) | Top-5 Recall (Hit@5) | MRR | Mean Latency |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])

    base = scorecard.get("eval_retriever", {})
    rerank = scorecard.get("eval_retriever_with_reranker", {})

    md_lines.append(
        f"| Baseline Lexical (arXiv + S2 + OpenAlex) | "
        f"{base.get('hit_rate_at_1', 0.0)*100:.1f}% | "
        f"{base.get('hit_rate_at_5', 0.0)*100:.1f}% | "
        f"{base.get('mrr', 0.0):.3f} | "
        f"{base.get('latency_profile', {}).get('mean', 0.0):.1f} ms |"
    )
    md_lines.append(
        f"| Neural Reranked (+ CrossEncoder) | "
        f"{rerank.get('hit_rate_at_1', 0.0)*100:.1f}% | "
        f"{rerank.get('hit_rate_at_5', 0.0)*100:.1f}% | "
        f"{rerank.get('mrr', 0.0):.3f} | "
        f"{rerank.get('latency_profile', {}).get('mean', 0.0):.1f} ms |"
    )

    gen = scorecard.get("eval_generator", {})
    rag = scorecard.get("eval_rag_pipeline", {})

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. End-to-End RAG Triad & Academic Answer Quality",
        "",
        "- **Faithfulness / Groundedness:** "
        f"{rag.get('mean_faithfulness', 0.0)*100:.1f}%",
        "- **Answer Relevance:** "
        f"{rag.get('mean_answer_relevance', 0.0)*100:.1f}%",
        "- **Domain Facet Coverage:** "
        f"{rag.get('mean_facet_coverage', 0.0)*100:.1f}%",
        "- **Citation Precision:** "
        f"{rag.get('mean_citation_precision', 0.0)*100:.1f}%",
        "",
        "### ResearchLens 6-Point Academic Rubric Adherence",
        f"- **Mean Rubric Score:** {gen.get('mean_rubric_score', 0.0)} / 6.0 ({gen.get('mean_rubric_percentage', 0.0)}%)",
        f"  - Technical Overview Rate: {gen.get('overview_rate', 0.0)*100:.1f}%",
        f"  - Core Mechanism / Architecture Rate: {gen.get('mechanism_rate', 0.0)*100:.1f}%",
        f"  - Mathematical Formulation Rate: {gen.get('math_rate', 0.0)*100:.1f}%",
        f"  - Empirical Benchmarks & Quantitative Results: {gen.get('benchmarks_rate', 0.0)*100:.1f}%",
        f"  - Limitations & Critical Discussion: {gen.get('limitations_rate', 0.0)*100:.1f}%",
        f"  - Verifiable Academic Citations: {gen.get('citations_rate', 0.0)*100:.1f}%",
        "",
        "---",
        "",
        "## 4. Security, Prompt Injection & Academic Safety Defenses",
        "",
        f"- **Prompt & Credential Leakage Rate:** {scorecard.get('eval_leakage', {}).get('leak_rate', 0.0)*100:.2f}%",
        f"- **Safety & Academic Integrity Compliance:** {scorecard.get('eval_safety', {}).get('compliance_rate', 0.0)*100:.1f}%",
        f"- **Academic Scope Discrimination ROC-AUC:** {scorecard.get('eval_scope', {}).get('roc_auc', 0.0):.3f}",
        f"- **Non-Academic Abstain Recall:** {scorecard.get('eval_scope', {}).get('abstain_recall', 0.0)*100:.1f}%",
        f"- **Response Toxicity Rate:** {scorecard.get('eval_toxicity', {}).get('toxicity_rate', 0.0)*100:.2f}%",
        f"- **System Unhandled Crashes:** {scorecard.get('eval_reliability', {}).get('crashes', 0)}",
        "",
        "---",
        "",
        "## 5. Operations & Application Workflows",
        "",
        f"- **SQLite Query Latency:** {scorecard.get('eval_ops', {}).get('database', {}).get('query_latency_ms', 0.0)} ms",
        f"- **Vector Index Embedding Status:** `{scorecard.get('eval_ops', {}).get('vector_store', {}).get('status', 'N/A')}`",
        f"- **User Journey Workflow Pass Rate:** {scorecard.get('eval_application', {}).get('pass_rate', 0.0)*100:.1f}%",
        "",
        "---",
        "",
        "## 6. Token Economics & Cost Projections",
        "",
        "| Provider / Model | Cost / Query | Cost / 1K Queries | Cost / 10K Queries |",
        "| :--- | :--- | :--- | :--- |"
    ])

    for p_id, p in scorecard.get("eval_cost", {}).get("providers", {}).items():
        md_lines.append(
            f"| {p['name']} | ${p['cost_per_query_usd']:.5f} | ${p['cost_per_1k_queries_usd']:.2f} | ${p['cost_per_10k_queries_usd']:.2f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        f"**Final Verdict:** `{promo_result['decision']}` — {promo_result['passed_gates']}/{promo_result['total_gates']} Production Promotion Gates Passed."
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[run_suite] Wrote executive markdown audit report to: {output_path}")


def run_full_suite(limit: int = 10, quick: bool = False) -> Dict[str, Any]:
    """Runs all 13 evaluation modules, compiles scorecard, and generates reports."""
    print("\n" + "#"*75)
    print(" RESEARCHLENS MASTER EVALUATION SUITE (run_suite.py)")
    print("#"*75 + "\n")

    start_ts = time.time()
    scorecard = {}

    # 1. Operational health
    scorecard["eval_ops"] = run_ops_eval()

    # 2. Retrieval baselines & reranked
    scorecard["eval_retriever"] = run_retrieval_eval(limit=limit)
    scorecard["eval_retriever_with_reranker"] = run_reranked_eval(limit=limit)

    # 3. Generation & RAG
    scorecard["eval_generator"] = run_generator_eval(limit=limit)
    scorecard["eval_rag_pipeline"] = run_rag_eval(limit=limit)

    # 4. Security & Safety
    scorecard["eval_leakage"] = run_leakage_eval()
    scorecard["eval_safety"] = run_safety_eval()
    scorecard["eval_scope"] = run_scope_eval(limit=limit)
    scorecard["eval_toxicity"] = run_toxicity_eval()

    # 5. Reliability, Latency & Cost
    scorecard["eval_reliability"] = run_reliability_eval()
    scorecard["eval_latency"] = run_latency_eval()
    scorecard["eval_cost"] = run_cost_eval()

    # 6. Application Workflows
    scorecard["eval_application"] = run_application_eval()

    # Save raw scorecard JSON
    report_json_path = CURRENT_DIR / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n[run_suite] Saved scorecard JSON to: {report_json_path}")

    # Evaluate promotion gates
    promo_result = evaluate_promotion_gates(scorecard)

    # Generate Markdown Report
    report_md_path = CURRENT_DIR / "FULL_EVAL_REPORT.md"
    generate_markdown_report(scorecard, promo_result, report_md_path)

    total_dur = round(time.time() - start_ts, 1)
    print("\n" + "="*75)
    print(f" FULL EVALUATION COMPLETED IN {total_dur}s")
    print(f" Promotion Decision: {promo_result['decision']} ({promo_result['passed_gates']}/{promo_result['total_gates']} Gates Passed)")
    print("="*75 + "\n")

    return {
        "scorecard": scorecard,
        "promotion": promo_result
    }


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Suite Runner")
    parser.add_argument("--limit", type=int, default=10, help="Max queries for dataset benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of tests")
    args = parser.parse_args()

    limit = 5 if args.quick else args.limit
    res = run_full_suite(limit=limit, quick=args.quick)
    sys.exit(0 if res["promotion"]["all_passed"] else 1)


if __name__ == "__main__":
    main()
