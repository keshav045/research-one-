"""
Generation Quality Evaluation — LLM-as-a-Judge
===============================================
Evaluates the *quality of synthesized research reports* produced by ResearchLens
against human-written reference answers using OpenAI gpt-4o-mini as the judge.

Three Ragas-inspired metrics are computed per query:

  FAITHFULNESS   (1–5)
      Do all claims in the report have factual grounding in real papers?
      5 = every claim traceable; 1 = many hallucinations.

  ANSWER_RELEVANCE  (1–5)
      Does the report directly and completely address the posed research question?
      5 = fully on-point; 1 = misses the question entirely.

  COMPLETENESS   (1–5)
      Does the report cover the expected topical facets / sub-concepts?
      5 = all expected facets addressed; 1 = none addressed.

Usage (stand-alone):
    python -m evaluation.generation_eval --max 3

The module is also imported and called from evaluation/benchmark.py when
the --gen-eval flag is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── ensure project root on path ──────────────────────────────────────────────
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

logger = logging.getLogger(__name__)

# ── Judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict, objective academic research evaluator. "
    "You rate AI-generated research reports purely on accuracy, relevance, and coverage. "
    "Be consistent: a score of 5 is reserved for near-perfect quality."
)

_JUDGE_PROMPT = """\
You will evaluate a synthesized research report on three dimensions.

━━━ RESEARCH QUESTION ━━━
{question}

━━━ REFERENCE ANSWER (ground-truth) ━━━
{expected_answer}

━━━ EXPECTED TOPICAL FACETS ━━━
{facets}

━━━ SYNTHESIZED REPORT (to evaluate) ━━━
{report_text}

━━━ RATING INSTRUCTIONS ━━━
Rate the synthesized report on a scale of 1–5 for each dimension:

  FAITHFULNESS (1–5):
    Does every factual claim in the report appear to be grounded in actual
    published research?  Are there hallucinated results, made-up citations,
    or fabricated author names?
    5 = fully grounded, no detectable hallucinations
    3 = mostly grounded, 1–2 suspicious claims
    1 = many unsupported or fabricated claims

  ANSWER_RELEVANCE (1–5):
    Does the report directly and completely answer the specific research question?
    5 = directly and thoroughly addresses the question
    3 = partially answers, some tangential content
    1 = misses the point, does not address the question

  COMPLETENESS (1–5):
    Does the report cover all the expected topical facets listed above?
    5 = all major facets adequately discussed
    3 = roughly half the facets covered
    1 = none of the expected facets are addressed

Respond ONLY with valid JSON on a single line, no extra text:
{{"faithfulness": <int 1-5>, "answer_relevance": <int 1-5>, "completeness": <int 1-5>}}
"""


# ── Core async judge call ─────────────────────────────────────────────────────

async def _call_openai_judge(
    question: str,
    report_text: str,
    expected_answer: str,
    expected_facets: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Call gpt-4o-mini as a judge and return raw score dict."""
    try:
        import openai  # type: ignore
    except ImportError:
        raise RuntimeError("openai package is not installed – run: pip install openai")

    client = openai.AsyncOpenAI(api_key=api_key)

    facets_text = ", ".join(expected_facets) if expected_facets else "(none specified)"
    # Truncate very long reports to stay within token budget
    max_report_chars = 3000
    if len(report_text) > max_report_chars:
        report_text = report_text[:max_report_chars] + "\n… [truncated for evaluation]"

    user_msg = _JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        facets=facets_text,
        report_text=report_text,
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=80,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("OpenAI judge timed out for question: %s", question[:60])
        return {"faithfulness": 0, "answer_relevance": 0, "completeness": 0, "error": "timeout"}

    raw = response.choices[0].message.content.strip()
    logger.debug("Judge raw response: %s", raw)

    # Robust JSON extraction (handle markdown code fences)
    json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    if not json_match:
        logger.warning("Could not parse judge JSON from: %s", raw)
        return {"faithfulness": 0, "answer_relevance": 0, "completeness": 0, "error": "parse_failed", "raw": raw}

    try:
        scores = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode failed: %s | raw=%s", exc, raw)
        return {"faithfulness": 0, "answer_relevance": 0, "completeness": 0, "error": str(exc), "raw": raw}

    # Clamp to valid range
    for key in ("faithfulness", "answer_relevance", "completeness"):
        scores[key] = max(1, min(5, int(scores.get(key, 1))))

    scores["overall"] = round((scores["faithfulness"] + scores["answer_relevance"] + scores["completeness"]) / 3, 2)
    return scores


# ── High-level per-query evaluator ────────────────────────────────────────────

async def evaluate_report_quality(
    question: str,
    report_text: str,
    expected_answer: str,
    expected_facets: List[str],
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate one synthesized report against ground-truth.

    Returns a dict with keys:
        faithfulness, answer_relevance, completeness, overall  (all floats 1–5)
        error  (str, only on failure)
    """
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "completeness": 0.0,
            "overall": 0.0,
            "error": "OPENAI_API_KEY not set",
        }

    if not report_text or len(report_text.strip()) < 20:
        return {
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "completeness": 1.0,
            "overall": 1.0,
            "error": "report_empty",
        }

    return await _call_openai_judge(
        question=question,
        report_text=report_text,
        expected_answer=expected_answer,
        expected_facets=expected_facets,
        api_key=api_key,
    )


# ── Batch evaluator (retrieval-only mode) ─────────────────────────────────────

async def run_generation_benchmark(
    dataset_path: Optional[str] = None,
    max_items: int = 5,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the FULL generation quality evaluation:
      1. Spin up the ResearchLens pipeline for each query.
      2. Extract the synthesized report text.
      3. Call LLM-as-a-Judge for Faithfulness / Answer Relevance / Completeness.
      4. Aggregate results and return a summary dict.

    Args:
        dataset_path:   Path to evaluation/dataset.json.
        max_items:      Maximum number of queries to evaluate.
        openai_api_key: Override for OPENAI_API_KEY env var.
    """
    if dataset_path is None:
        dataset_path = str(Path(__file__).resolve().parent / "dataset.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    items = dataset[:max_items]
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    print("=" * 80)
    print("       RESEARCHLENS  GENERATION QUALITY EVALUATION  (LLM-as-a-Judge)")
    print("=" * 80)
    print(f"Dataset  : {dataset_path}")
    print(f"Queries  : {len(items)} of {len(dataset)}")
    print(f"Judge    : gpt-4o-mini")
    print(f"Metrics  : Faithfulness | Answer Relevance | Completeness  (each 1–5)")
    print()

    # Lazy imports — only needed here
    from backend.models.database import SessionLocal, create_tables, ResearchJob
    from backend.services.research_workflow import run_research_pipeline, job_to_investigation

    create_tables()

    per_query: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for idx, item in enumerate(items, start=1):
        q_id: str = item["id"]
        question: str = item["question"]
        expected_answer: str = item.get("expected_answer", "")
        expected_facets: List[str] = item.get("expected_facets", [])

        print(f"[{idx:02d}/{len(items):02d}]  {q_id}")
        print(f"  Q  : {question}")

        t0 = time.perf_counter()

        # ── 1. Run pipeline ──────────────────────────────────────────────────
        try:
            db = SessionLocal()
            job_id = f"geneval-{q_id}"
            db.query(ResearchJob).filter(ResearchJob.id == job_id).delete()
            db.commit()

            job = ResearchJob(
                id=job_id,
                question=question,
                depth="Quick",
                sources=json.dumps(["arXiv", "Semantic Scholar", "OpenAlex"]),
                status="in_progress",
            )
            db.add(job)
            db.commit()

            await run_research_pipeline(job_id, db)
            db.refresh(job)
            inv = job_to_investigation(job)
            db.close()
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", q_id, exc, exc_info=True)
            per_query.append({
                "id": q_id,
                "question": question,
                "faithfulness": 0,
                "answer_relevance": 0,
                "completeness": 0,
                "overall": 0,
                "error": str(exc),
                "latency_seconds": round(time.perf_counter() - t0, 2),
            })
            print(f"  ERROR (pipeline): {exc}\n")
            continue

        pipeline_latency = time.perf_counter() - t0

        # ── 2. Extract report text ───────────────────────────────────────────
        report = getattr(inv, "report", None)
        report_parts: List[str] = []
        if report:
            if getattr(report, "executiveSummary", None):
                report_parts.append(report.executiveSummary)
            for finding in getattr(report, "findings", []):
                for para in getattr(finding, "paragraphs", []):
                    if getattr(para, "text", None):
                        report_parts.append(para.text)
        report_text = " ".join(report_parts)

        print(f"  Pipeline: {inv.status}  |  Report length: {len(report_text)} chars  |  Latency: {pipeline_latency:.1f}s")

        # ── 3. LLM judge ────────────────────────────────────────────────────
        print(f"  Calling gpt-4o-mini judge …", end=" ", flush=True)
        judge_t0 = time.perf_counter()
        scores = await evaluate_report_quality(
            question=question,
            report_text=report_text,
            expected_answer=expected_answer,
            expected_facets=expected_facets,
            openai_api_key=api_key,
        )
        judge_latency = time.perf_counter() - judge_t0

        faith = scores.get("faithfulness", 0)
        rel   = scores.get("answer_relevance", 0)
        comp  = scores.get("completeness", 0)
        ovrl  = scores.get("overall", 0)
        err   = scores.get("error", "")

        status_str = f"Faithfulness={faith}/5  Relevance={rel}/5  Completeness={comp}/5  Overall={ovrl:.2f}/5"
        if err:
            status_str += f"  [ERROR: {err}]"
        print(f"done ({judge_latency:.1f}s)")
        print(f"  Scores: {status_str}")
        print()

        per_query.append({
            "id": q_id,
            "question": question,
            "domain": item.get("domain", ""),
            "question_type": item.get("question_type", ""),
            "difficulty": item.get("difficulty", ""),
            "faithfulness": faith,
            "answer_relevance": rel,
            "completeness": comp,
            "overall": ovrl,
            "report_length_chars": len(report_text),
            "pipeline_latency_seconds": round(pipeline_latency, 2),
            "judge_latency_seconds": round(judge_latency, 2),
            **({"error": err} if err else {}),
        })

    total_duration = time.perf_counter() - total_start

    # ── 4. Aggregate ─────────────────────────────────────────────────────────
    valid = [r for r in per_query if r.get("overall", 0) > 0]

    def _mean(key: str) -> float:
        vals = [r[key] for r in valid if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "evaluation_type":        "generation_quality_llm_as_judge",
        "judge_model":            "gpt-4o-mini",
        "queries_evaluated":      len(per_query),
        "queries_succeeded":      len(valid),
        "total_duration_seconds": round(total_duration, 2),
        # Aggregate scores (1–5 scale)
        "mean_faithfulness":      _mean("faithfulness"),
        "mean_answer_relevance":  _mean("answer_relevance"),
        "mean_completeness":      _mean("completeness"),
        "mean_overall":           _mean("overall"),
        # Normalised to 0–1 for comparison with other metric families
        "faithfulness_norm":      round(_mean("faithfulness") / 5, 3),
        "answer_relevance_norm":  round(_mean("answer_relevance") / 5, 3),
        "completeness_norm":      round(_mean("completeness") / 5, 3),
        "overall_norm":           round(_mean("overall") / 5, 3),
        "per_query":              per_query,
    }

    # ── 5. Print summary ─────────────────────────────────────────────────────
    print("=" * 80)
    print("            GENERATION QUALITY  —  SUMMARY")
    print("=" * 80)
    print(f"  Queries evaluated   : {len(per_query)}  (succeeded: {len(valid)})")
    print(f"  Total duration      : {total_duration:.1f}s")
    print()
    print(f"  Mean Faithfulness   : {summary['mean_faithfulness']:.2f} / 5   ({summary['faithfulness_norm']*100:.1f}%)")
    print(f"  Mean Answer Relevance: {summary['mean_answer_relevance']:.2f} / 5   ({summary['answer_relevance_norm']*100:.1f}%)")
    print(f"  Mean Completeness   : {summary['mean_completeness']:.2f} / 5   ({summary['completeness_norm']*100:.1f}%)")
    print(f"  Mean Overall        : {summary['mean_overall']:.2f} / 5   ({summary['overall_norm']*100:.1f}%)")
    print()

    # Grade
    overall_norm = summary["overall_norm"]
    if overall_norm >= 0.85:
        grade = "A  (Excellent)"
    elif overall_norm >= 0.70:
        grade = "B  (Good)"
    elif overall_norm >= 0.55:
        grade = "C  (Acceptable)"
    elif overall_norm >= 0.40:
        grade = "D  (Needs Improvement)"
    else:
        grade = "F  (Poor)"
    summary["grade"] = grade
    print(f"  Grade               : {grade}")
    print("=" * 80)

    # ── 6. Write JSON ─────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent / "generation_eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed report written to: {out_path}\n")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ResearchLens Generation Quality Evaluation (LLM-as-a-Judge).",
    )
    p.add_argument("--dataset", default=None, help="Path to dataset.json")
    p.add_argument("--max", type=int, default=5, dest="max_items", help="Number of queries to evaluate (default 5)")
    p.add_argument("--api-key", default=None, dest="api_key", help="OpenAI API key override")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()

    # Load .env
    env_path = root_dir / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    asyncio.run(
        run_generation_benchmark(
            dataset_path=args.dataset,
            max_items=args.max_items,
            openai_api_key=args.api_key,
        )
    )

