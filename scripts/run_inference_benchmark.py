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

    q = "What are the most effective techniques for reducing the computational cost and memory usage of large language models during inference?"
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
    print(f"=== Benchmark Results for: {job_id} ===")
    print("="*80)
    print(f"Status: {inv.status}")
    print(f"Research Confidence: {inv.research_confidence}")
    print(f"Citation Integrity: {inv.citation_integrity}%")
    print(f"Evidence Coverage: {inv.evidence_coverage}%")
    print(f"Citation Coverage: {inv.citationCoverage}%")
    print(f"Papers Analyzed: {inv.papersAnalyzed}")
    print(f"Total Passages Extracted: {inv.passages_total}")
    print(f"Verified Claims: {inv.verifiedClaims}")
    print(f"Partially Supported Claims: {inv.partiallySupportedClaims}")
    print(f"Unsupported Claims: {inv.unsupportedClaims}")
    print(f"Contradicted Claims: {inv.contradictedClaims}")
    if inv.failure_reason:
        print(f"Failure / Warning Reasons: {inv.failure_reason}")

    if inv.report:
        print("\n--- Executive Summary ---")
        print(inv.report.executiveSummary)

        print("\n--- Comparison Table Rows ---")
        for i, row in enumerate(inv.report.comparisonTable):
            print(f"[{i+1}] {row.title} ({row.year}) | Venue: {row.venue} | Citations: {row.citationCount}")

        print("\n--- References ---")
        for ref in inv.report.references:
            print(f"- [{ref.id}] {ref.title} ({ref.publicationYear}) - Authors: {', '.join(ref.authors[:3])}")

        print("\n--- Findings & Verified Techniques ---")
        for sec in inv.report.findings:
            print(f"\nSection: {sec.sectionTitle}")
            for p in sec.paragraphs:
                print(f"Text: {p.text}")
                for c in p.citations:
                    print(f"  Badge [{c.badgeNumber}] (p.{c.page}): {c.claim[:100]} [{c.status}]")

    db.close()
    print("\n=== Benchmark Completed Successfully ===")


if __name__ == "__main__":
    asyncio.run(main())
