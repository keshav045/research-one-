"""
End-to-end Benchmark Execution: LLM Inference Efficiency
Research Question:
"What are the most effective techniques for reducing the computational cost and memory usage of large language models during inference?"
"""

import asyncio
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.database import SessionLocal, create_tables, ResearchJob
from backend.services.research_workflow import run_research_pipeline, job_to_investigation


async def main():
    create_tables()
    db = SessionLocal()

    q = "What are the most effective techniques for optimizing large language model inference in terms of memory usage, latency, and computational cost?"
    job_id = "bench-llm-inference-efficiency"

    # Clean previous run if exists
    db.query(ResearchJob).filter(ResearchJob.id == job_id).delete()
    db.commit()

    job = ResearchJob(
        id=job_id,
        question=q,
        depth="Quick",
        sources='["arXiv", "Semantic Scholar", "OpenAlex"]',
        status="in_progress",
    )
    db.add(job)
    db.commit()

    print(f"=== Starting Benchmark Pipeline for: '{q}' ===")
    await run_research_pipeline(job_id, db)

    db.refresh(job)
    inv = job_to_investigation(job)

    print("\n" + "="*80)
    print(f"=== BENCHMARK REPORT: {job_id} ===")
    print("="*80)
    print(f"Status: {inv.status}")
    print(f"\n--- REQUIRED RESEARCH METRICS ---")
    print(f"- discovered papers:        {inv.papers_discovered}")
    print(f"- unique papers:            {inv.unique_papers}")
    print(f"- relevant papers:          {inv.relevant_papers}")
    print(f"- full-text papers:         {inv.full_text_papers}")
    print(f"- evidence-bearing papers:  {inv.evidence_bearing_papers}")
    print(f"- candidate claims:         {inv.candidate_claims}")
    print(f"- verified claims:          {inv.verified_claims}")
    print(f"- contradicted claims:      {inv.contradicted_claims}")
    print(f"- insufficient claims:      {inv.insufficient_claims}")
    print(f"- citation integrity:       {inv.citation_integrity}%")
    print(f"- evidence coverage:        {inv.evidence_coverage}%")
    print(f"- research depth:           {inv.research_depth}%")
    print(f"- research confidence:      {inv.research_confidence}")
    print(f"- source failures / status: {json.dumps(inv.source_status or {}, indent=2)}")

    print(f"\n--- CONCEPT COVERAGE ---")
    if inv.concept_coverage:
        for concept, status in inv.concept_coverage.items():
            mark = "[VERIFIED]" if status == "VERIFIED" else "[INSUFFICIENT]"
            print(f"  {concept:<45} {mark}")
    else:
        print("  None recorded")

    if inv.report and inv.report.retrieval_warnings:
        print(f"\n--- RETRIEVAL WARNINGS ---")
        for w in inv.report.retrieval_warnings:
            print(f"  - {w}")

    if inv.report:
        print("\n--- EXECUTIVE SUMMARY ---")
        print(inv.report.executiveSummary)

        if inv.report.technique_comparison:
            print("\n--- TECHNIQUE COMPARISON ---")
            for tc in inv.report.technique_comparison:
                print(f"  - {tc.get('technique')}: Status={tc.get('evidence_status')}, Finding={tc.get('key_finding')}")

        if inv.report.insufficient_evidence:
            print("\n--- INSUFFICIENT EVIDENCE CONCEPTS ---")
            for ie in inv.report.insufficient_evidence:
                print(f"  - {ie}")

        print("\n--- REFERENCES ---")
        for ref in inv.report.references:
            print(f"  - [{ref.id}] {ref.title} ({ref.publicationYear}) - Authors: {', '.join(ref.authors[:3])}")

    db.close()
    print("\n=== Benchmark Completed Successfully ===")


if __name__ == "__main__":
    asyncio.run(main())
