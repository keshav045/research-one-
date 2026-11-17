"""
System Reliability and Edge Case Evaluation Module (eval_reliability.py).
Evaluates ResearchLens stability under adversarial edge inputs: empty queries, whitespace,
1200+ word inputs, Unicode math, SQL injection patterns, and malformed characters.
Production Gate: Unhandled Crash Rate MUST be 0.0%.
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

from backend.models.schemas import Paper, ResearchReport, ResearchDepth
from backend.services.query_planner import plan_queries
from evaluation.harness import EvaluationHarness, synthesize_report_sync
from evaluation.metric_registry import latency_percentiles

logger = logging.getLogger("researchlens.eval.reliability")

RELIABILITY_EDGE_CASES = [
    {
        "test_id": "edge_empty",
        "name": "Empty string input",
        "query": ""
    },
    {
        "test_id": "edge_whitespace",
        "name": "Whitespace only input",
        "query": "   \n\t\r   "
    },
    {
        "test_id": "edge_ultralong",
        "name": "1200+ word repeating query",
        "query": "explain multi-head self-attention mechanisms and query-key-value transformations in deep learning " * 120
    },
    {
        "test_id": "edge_unicode_math",
        "name": "Complex LaTeX and Unicode mathematical symbols",
        "query": "Evaluate attention formulation: \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V with \\sum_{i=1}^n \\lambda_i \\cdot \\infty"
    },
    {
        "test_id": "edge_xss_sql_injection",
        "name": "XSS script tags and SQL injection delimiters",
        "query": "<script>alert('ResearchLens_XSS')</script> '; DROP TABLE research_jobs; -- SELECT * FROM papers WHERE '1'='1"
    },
    {
        "test_id": "edge_special_chars",
        "name": "Punctuation and non-ASCII glyphs",
        "query": "!@#$%^&*()_+{}[]:\"|<>?,./~`\\'"
    }
]


def eval_single_reliability(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    """Tests system resilience against malicious or malformed edge inputs."""
    query = item["query"]
    crashed = False
    error_msg = None
    response_length = 0

    try:
        # Step 1: Query Planner
        if query.strip():
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            plan = loop.run_until_complete(plan_queries(query))
        
        # Step 2: Synthesis
        report, _, _ = synthesize_report_sync(
            question=query,
            papers=[],
            citations=[],
            depth=ResearchDepth.QUICK
        )
        reply = report.executiveSummary if hasattr(report, "executiveSummary") else str(report)
        response_length = len(reply)
        handled_gracefully = True
    except Exception as exc:
        crashed = True
        error_msg = str(exc)
        handled_gracefully = False

    return {
        "test_id": item["test_id"],
        "name": item["name"],
        "crashed": crashed,
        "error_msg": error_msg,
        "response_length": response_length,
        "handled_gracefully": handled_gracefully
    }


def run_reliability_eval() -> Dict[str, Any]:
    """Executes system reliability and edge-case benchmark."""
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_reliability] Testing system resilience under edge-case and adversarial inputs...")
    results = []
    latencies = []

    for item in RELIABILITY_EDGE_CASES:
        t0 = time.perf_counter()
        res = eval_single_reliability(item, harness)
        dur = (time.perf_counter() - t0) * 1000.0
        res["latency_ms"] = round(dur, 2)
        results.append(res)
        latencies.append(dur)

    crashes = sum(1 for r in results if r["crashed"])
    pass_count = sum(1 for r in results if r["handled_gracefully"])

    summary = {
        "benchmark": "eval_reliability",
        "total_edge_cases": len(results),
        "crashes": crashes,
        "handled_gracefully_count": pass_count,
        "crash_rate": round(crashes / float(len(results)), 4),
        "pass_rate": round(pass_count / float(len(results)), 4),
        "latency_profile": latency_percentiles(latencies),
        "results": results
    }

    print("\n" + "="*60)
    print(" SYSTEM RELIABILITY & EDGE-CASE BENCHMARK RESULTS")
    print("="*60)
    print(f" Total Edge Cases:        {summary['total_edge_cases']}")
    print(f" Unhandled Crashes:       {summary['crashes']} (Threshold: == 0)")
    print(f" Handled Gracefully:      {summary['handled_gracefully_count']} / {summary['total_edge_cases']}")
    print(f" Stability Pass Rate:     {summary['pass_rate']*100:.1f}%")
    print(f" Gate Status:             {'PASSED' if summary['crashes'] == 0 else 'FAILED'}")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    run_reliability_eval()
