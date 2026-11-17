"""
Cost and Token Economics Evaluation Module (eval_cost.py).
Evaluates token consumption and financial costs across LLM providers:
- Google Gemini 2.0 Flash
- OpenAI gpt-4o-mini
- Groq (Llama-3.3-70B)
- Local / In-Process Offline Mode ($0.00)
Projects operational research budgets for 1,000 to 10,000 investigations.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.models.schemas import Paper, ResearchReport, ResearchDepth
from evaluation.harness import EvaluationHarness, synthesize_report_sync
from evaluation.metric_registry import estimate_cost

logger = logging.getLogger("researchlens.eval.cost")

COST_BENCHMARK_PROMPTS = [
    "Which paper introduced the Transformer architecture and self-attention?",
    "What is retrieval-augmented generation (RAG) and who proposed it?",
    "Explain deep residual learning for image recognition (ResNet).",
    "How does the Adam optimizer compute adaptive learning rates from gradients?",
    "What is the biological mechanism of CRISPR-Cas9 targeted genome editing?"
]


def run_cost_eval() -> Dict[str, Any]:
    """Estimates token consumption and operational costs across providers."""
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_cost] Estimating token usage and API costs across providers...")

    total_prompt_chars = 0
    total_completion_chars = 0

    for query in COST_BENCHMARK_PROMPTS:
        context_text = f"Context passages for: {query}\n" + (
            "Paper 1: Theoretical foundation and methodology.\n"
            "Paper 2: Empirical benchmark results on GLUE, SQuAD, and ImageNet.\n"
            "Paper 3: Ablation studies and limitations.\n"
        ) * 5
        prompt_text = f"Context:\n{context_text}\nQuestion: {query}"
        total_prompt_chars += len(prompt_text)

        report, _, _ = synthesize_report_sync(
            question=query,
            papers=[],
            citations=[],
            depth=ResearchDepth.QUICK
        )
        reply = report.executiveSummary if hasattr(report, "executiveSummary") else str(report)
        total_completion_chars += len(reply)

    # 1 token ≈ 4 characters
    prompt_tokens = int(total_prompt_chars / 4.0)
    completion_tokens = int(total_completion_chars / 4.0)
    num_queries = len(COST_BENCHMARK_PROMPTS)

    avg_prompt_tokens = int(prompt_tokens / num_queries)
    avg_completion_tokens = int(completion_tokens / num_queries)
    avg_total_tokens = avg_prompt_tokens + avg_completion_tokens

    gemini_cost = estimate_cost(prompt_tokens, completion_tokens, provider="gemini")
    openai_cost = estimate_cost(prompt_tokens, completion_tokens, provider="openai")
    groq_cost = estimate_cost(prompt_tokens, completion_tokens, provider="groq")
    offline_cost = estimate_cost(prompt_tokens, completion_tokens, provider="offline")

    gemini_1k = round(gemini_cost["cost_usd"] * (1000.0 / num_queries), 3)
    openai_1k = round(openai_cost["cost_usd"] * (1000.0 / num_queries), 3)
    groq_1k = round(groq_cost["cost_usd"] * (1000.0 / num_queries), 3)

    summary = {
        "benchmark": "eval_cost",
        "sample_queries": num_queries,
        "avg_prompt_tokens_per_query": avg_prompt_tokens,
        "avg_completion_tokens_per_query": avg_completion_tokens,
        "avg_total_tokens_per_query": avg_total_tokens,
        "providers": {
            "gemini": {
                "name": "Google Gemini 2.0 Flash",
                "cost_per_query_usd": round(gemini_cost["cost_usd"] / num_queries, 6),
                "cost_per_1k_queries_usd": gemini_1k,
                "cost_per_10k_queries_usd": round(gemini_1k * 10, 2)
            },
            "openai": {
                "name": "OpenAI gpt-4o-mini",
                "cost_per_query_usd": round(openai_cost["cost_usd"] / num_queries, 6),
                "cost_per_1k_queries_usd": openai_1k,
                "cost_per_10k_queries_usd": round(openai_1k * 10, 2)
            },
            "groq": {
                "name": "Groq (Llama-3.3-70B)",
                "cost_per_query_usd": round(groq_cost["cost_usd"] / num_queries, 6),
                "cost_per_1k_queries_usd": groq_1k,
                "cost_per_10k_queries_usd": round(groq_1k * 10, 2)
            },
            "offline": {
                "name": "Local In-Process / Ollama",
                "cost_per_query_usd": 0.0,
                "cost_per_1k_queries_usd": 0.0,
                "cost_per_10k_queries_usd": 0.0
            }
        }
    }

    print("\n" + "="*70)
    print(" TOKEN ECONOMICS & LLM INFERENCE COST PROJECTIONS")
    print("="*70)
    print(f" Avg Prompt Tokens / Query:     {avg_prompt_tokens}")
    print(f" Avg Completion Tokens / Query: {avg_completion_tokens}")
    print(f" Avg Total Tokens / Query:      {avg_total_tokens}")
    print("-" * 70)
    print(f"{'Provider / Model':<26} | {'Cost / Query':<14} | {'Cost / 1K Jobs':<14} | {'Cost / 10K Jobs'}")
    print("-" * 70)
    for p_id, p in summary["providers"].items():
        print(f"{p['name']:<26} | ${p['cost_per_query_usd']:<13.5f} | ${p['cost_per_1k_queries_usd']:<13.2f} | ${p['cost_per_10k_queries_usd']:.2f}")
    print("="*70 + "\n")

    return summary


if __name__ == "__main__":
    run_cost_eval()
