"""
End-to-end Pipeline Benchmark Test
Tests:
1. Benchmark query: "Which paper introduced the Transformer architecture, and what was its key idea?"
   Expected: Vaswani et al. 2017 ("Attention Is All You Need"), verified citations, page numbers, status "completed".
2. Negative query: "Which paper proved that unicorns built the pyramids using anti-gravity cheese?"
   Expected: Evidence gate triggers, status "insufficient_evidence", failure reason recorded, honest explanation.
"""

import asyncio
import os
import sys

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.models.database import SessionLocal, create_tables, ResearchJob
from backend.services.research_workflow import run_research_pipeline, job_to_investigation


import pytest


@pytest.mark.live
@pytest.mark.asyncio
async def test_benchmark_pipeline():
    create_tables()
    db = SessionLocal()

    # ── Test 1: Real Benchmark Query ──────────────────────────────────────────
    q1 = "Which paper introduced the Transformer architecture, and what was its key idea?"
    job1_id = "test-benchmark-transformer"
    
    # Remove existing if any
    db.query(ResearchJob).filter(ResearchJob.id == job1_id).delete()
    db.commit()

    job1 = ResearchJob(
        id=job1_id,
        question=q1,
        depth="Quick",
        sources='["arXiv", "Semantic Scholar"]',
        status="in_progress",
    )
    db.add(job1)
    db.commit()

    print(f"\n[TEST 1] Running pipeline for: '{q1}'...")
    await run_research_pipeline(job1_id, db)

    db.refresh(job1)
    inv1 = job_to_investigation(job1)

    print(f"[TEST 1 Result] Status: {inv1.status}")
    print(f"[TEST 1 Result] Papers analyzed: {inv1.papersAnalyzed}, Passages total: {inv1.passages_total}")
    print(f"[TEST 1 Result] Verified claims: {inv1.verifiedClaims}, Partially supported: {inv1.partiallySupportedClaims}")
    print(f"[TEST 1 Result] Citation coverage: {inv1.citationCoverage}%")
    
    if inv1.report:
        print(f"[TEST 1 Result] Report Executive Summary:\n{inv1.report.executiveSummary[:400]}...\n")
        print(f"[TEST 1 Result] Comparison Table Rows: {len(inv1.report.comparisonTable)}")
        print(f"[TEST 1 Result] References ({len(inv1.report.references)}):")
        for ref in inv1.report.references[:3]:
            print(f"  - [{ref.id}] {ref.title} ({ref.publicationYear}) - {', '.join(ref.authors[:2])}")
        
        # Check findings citations
        for sec in inv1.report.findings:
            print(f"  Section: {sec.sectionTitle}")
            for p in sec.paragraphs:
                for c in p.citations:
                    print(f"    - Citation [{c.badgeNumber}] (p.{c.page}): {c.claim[:80]} -> {c.status}")

    assert inv1.status in ("completed", "completed_with_warnings"), f"Expected completed or completed_with_warnings, got {inv1.status}"
    assert any("attention is all you need" in r.title.lower() for r in (inv1.report.references if inv1.report else [])), \
        "Expected 'Attention Is All You Need' in references"
    assert inv1.verifiedClaims + inv1.partiallySupportedClaims >= 1, "Expected at least 1 verified/supported claim"
    print("\n>>> TEST 1 PASSED SUCCESSFULLY! <<<\n")

    # ── Test 2: Negative/Nonsense Query ────────────────────────────────────────
    q2 = "Which paper proved that unicorns built the pyramids using anti-gravity cheese?"
    job2_id = "test-negative-nonsense"

    db.query(ResearchJob).filter(ResearchJob.id == job2_id).delete()
    db.commit()

    job2 = ResearchJob(
        id=job2_id,
        question=q2,
        depth="Quick",
        sources='["arXiv"]',
        status="in_progress",
    )
    db.add(job2)
    db.commit()

    print(f"\n[TEST 2] Running pipeline for nonsense query: '{q2}'...")
    await run_research_pipeline(job2_id, db)

    db.refresh(job2)
    inv2 = job_to_investigation(job2)

    print(f"[TEST 2 Result] Status: {inv2.status}")
    print(f"[TEST 2 Result] Failure reason: {inv2.failure_reason}")
    print(f"[TEST 2 Result] Verified claims: {inv2.verifiedClaims}")
    assert inv2.status == "insufficient_evidence", f"Expected insufficient_evidence, got {inv2.status}"
    print("\n>>> TEST 2 PASSED (Honest Evidence Gate Triggered)! <<<\n")

    db.close()


if __name__ == "__main__":
    asyncio.run(test_benchmark_pipeline())
