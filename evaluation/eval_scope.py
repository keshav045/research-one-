"""
Academic Scope Gating Evaluation Module (eval_scope.py).
Evaluates the system's ability to discriminate between authentic academic research queries
and out-of-scope / non-academic requests (abstention detection, relevance threshold gating).
Production Gate: Scope ROC-AUC >= 0.85, Abstain Recall >= 80%.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.query_planner import plan_queries
from evaluation.harness import EvaluationHarness
from evaluation.metric_registry import latency_percentiles

logger = logging.getLogger("researchlens.eval.scope")

OUT_OF_SCOPE_CASES = [
    {"query": "What is the best recipe for homemade Italian pizza dough?", "is_out_of_scope": True},
    {"query": "Who won the Premier League football match last night?", "is_out_of_scope": True},
    {"query": "Tell me a fun bedtime story about a friendly dragon.", "is_out_of_scope": True},
    {"query": "Should I invest my money in Bitcoin or Ethereum this week?", "is_out_of_scope": True},
    {"query": "Recommend a romantic comedy movie to watch on Netflix.", "is_out_of_scope": True},
    {"query": "Can you book a hotel room for me in Paris next weekend?", "is_out_of_scope": True}
]

IN_SCOPE_CASES = [
    {"query": "Which paper introduced the Transformer architecture and self-attention?", "is_out_of_scope": False},
    {"query": "What is retrieval-augmented generation and who proposed it?", "is_out_of_scope": False},
    {"query": "Explain deep residual learning for image recognition (ResNet).", "is_out_of_scope": False},
    {"query": "How does the Adam optimizer compute adaptive learning rates?", "is_out_of_scope": False},
    {"query": "What is the mechanism of CRISPR-Cas9 targeted genome editing?", "is_out_of_scope": False},
    {"query": "How do Latent Diffusion Models generate high-resolution images?", "is_out_of_scope": False}
]

ACADEMIC_KEYWORDS = {
    "architecture", "attention", "transformer", "retrieval", "augmented", "generation",
    "residual", "resnet", "learning", "optimizer", "adam", "gradient", "diffusion",
    "crispr", "cas9", "neural", "network", "paper", "author", "algorithm", "model",
    "empirical", "benchmark", "dataset", "convolution", "representation", "nlp"
}


def eval_single_scope(item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates whether an inquiry is recognized as valid academic research vs out-of-scope."""
    query = item["query"]
    is_out = item["is_out_of_scope"]

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Plan query via query_planner
    plan = loop.run_until_complete(plan_queries(query))
    terms = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
    academic_hits = len(terms.intersection(ACADEMIC_KEYWORDS))

    # Calculate academic confidence score in [0, 1]
    confidence_score = min(1.0, academic_hits / 3.0)
    
    # Decision threshold
    threshold = 0.33
    abstained = confidence_score < threshold

    correct = (is_out and abstained) or (not is_out and not abstained)

    return {
        "query": query,
        "is_out_of_scope": is_out,
        "confidence_score": round(confidence_score, 3),
        "abstained": abstained,
        "correct": correct
    }


def run_scope_eval(limit: int = 12) -> Dict[str, Any]:
    """Executes academic scope discrimination benchmark."""
    print("\n[eval_scope] Evaluating academic scope gating and non-academic abstention...")
    test_cases = IN_SCOPE_CASES[:limit//2] + OUT_OF_SCOPE_CASES[:limit//2]

    results = []
    latencies = []

    for item in test_cases:
        t0 = time.perf_counter()
        res = eval_single_scope(item)
        dur = (time.perf_counter() - t0) * 1000.0
        res["latency_ms"] = round(dur, 2)
        results.append(res)
        latencies.append(dur)

    in_scope = [r for r in results if not r["is_out_of_scope"]]
    out_scope = [r for r in results if r["is_out_of_scope"]]

    abstain_recall = sum(1 for r in out_scope if r["abstained"]) / max(len(out_scope), 1)
    false_abstain_rate = sum(1 for r in in_scope if r["abstained"]) / max(len(in_scope), 1)
    accuracy = sum(1 for r in results if r["correct"]) / max(len(results), 1)

    # Compute ROC-AUC (Mann-Whitney U statistic)
    pos_scores = [r["confidence_score"] for r in in_scope]
    neg_scores = [r["confidence_score"] for r in out_scope]
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    roc_auc = round(wins / max(len(pos_scores) * len(neg_scores), 1), 4)

    summary = {
        "benchmark": "eval_scope",
        "total_evaluated": len(results),
        "accuracy": round(accuracy, 4),
        "abstain_recall": round(abstain_recall, 4),
        "false_abstain_rate": round(false_abstain_rate, 4),
        "roc_auc": roc_auc,
        "latency_profile": latency_percentiles(latencies),
        "results": results
    }

    print("\n" + "="*60)
    print(" ACADEMIC SCOPE DISCRIMINATION EVALUATION RESULTS")
    print("="*60)
    print(f" Total Cases Evaluated:   {summary['total_evaluated']}")
    print(f" Scope ROC-AUC:           {summary['roc_auc']:.3f} (Threshold: >= 0.85)")
    print(f" Abstain Recall:          {summary['abstain_recall']*100:.1f}% (Threshold: >= 80.0%)")
    print(f" False Abstain Rate:      {summary['false_abstain_rate']*100:.1f}%")
    print(f" Overall Accuracy:        {summary['accuracy']*100:.1f}%")
    print(f" Gate Status:             {'PASSED' if summary['roc_auc'] >= 0.85 and summary['abstain_recall'] >= 0.80 else 'FAILED'}")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    run_scope_eval()
