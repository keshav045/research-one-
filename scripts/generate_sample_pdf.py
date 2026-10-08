"""
Generate a complete, production-grade PDF Research Report
to verify that report synthesis, citation badges, references,
and single-draw PDF generation function without errors.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.models.schemas import (
    Citation,
    CitationStatus,
    Paper,
    ResearchInvestigation,
    ResearchDepth,
    ResearchSource,
)
from backend.services.local_llm_service import synthesize_report
from backend.services.pdf_report import build_pdf_report
from backend.services.report_validator import validate_report
from scripts.verify_acceptance import audit_pdf_bytes


async def generate_and_save_report(output_filename: str = "ResearchLens_Generated_Report.pdf") -> Path:
    print(f"Generating research report: {output_filename}...")

    q = "What are the core scaling laws and architectural innovations enabling modern Transformer models?"
    
    p1 = Paper(
        id="paper-1",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arXiv.1706.03762",
        source="arXiv",
        citationCount=95000,
        abstract="We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism to draw global dependencies between input and output.",
    )
    p2 = Paper(
        id="paper-2",
        title="Scaling Laws for Neural Language Models",
        authors=["Jared Kaplan", "Sam McCandlish", "Tom Henighan", "Tom B. Brown"],
        publicationYear=2020,
        journalConference="arXiv",
        doi="10.48550/arXiv.2001.08361",
        source="arXiv",
        citationCount=4200,
        abstract="We investigate empirical scaling laws for language model performance across model size, dataset size, and compute budget.",
    )
    p3 = Paper(
        id="paper-3",
        title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        authors=["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Re"],
        publicationYear=2022,
        journalConference="NeurIPS",
        doi="10.48550/arXiv.2205.14135",
        source="arXiv",
        citationCount=2100,
        abstract="FlashAttention tiles attention computation to reduce memory reads and writes between GPU HBM and SRAM.",
    )

    papers = [p1, p2, p3]

    citations = [
        Citation(
            id="c1",
            badgeNumber=1,
            paperId=p1.id,
            paperTitle=p1.title,
            authors="Vaswani et al.",
            year=2017,
            page=1,
            passage="We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism.",
            claim="Transformers eliminate sequential recurrence in favor of multi-head self-attention mechanisms.",
            status=CitationStatus.VERIFIED,
            highlightSentence="We propose the Transformer, a model architecture eschewing recurrence.",
        ),
        Citation(
            id="c2",
            badgeNumber=2,
            paperId=p2.id,
            paperTitle=p2.title,
            authors="Kaplan et al.",
            year=2020,
            page=2,
            passage="Cross-entropy loss scales as a power-law with compute, dataset size, and parameter count.",
            claim="Language model test performance follows predictable power-law scaling relationships.",
            status=CitationStatus.VERIFIED,
            highlightSentence="Cross-entropy loss scales as a power-law with compute, dataset size, and parameter count.",
        ),
        Citation(
            id="c3",
            badgeNumber=3,
            paperId=p3.id,
            paperTitle=p3.title,
            authors="Dao et al.",
            year=2022,
            page=1,
            passage="FlashAttention tiles computation to reduce memory reads and writes between high bandwidth memory and on-chip SRAM.",
            claim="IO-aware tiling in FlashAttention provides 2-4x speedups without approximation.",
            status=CitationStatus.VERIFIED,
            highlightSentence="FlashAttention tiles computation to reduce memory reads and writes.",
        ),
    ]

    claims = [c.claim for c in citations]

    report, _, _ = await synthesize_report(
        question=q,
        depth=ResearchDepth.STANDARD,
        papers=papers,
        claims=claims,
        citations=citations,
        anchor_paper=p1,
        anchor_confidence="high",
        debug_info={
            "papers_discovered": 120,
            "unique_papers": 85,
            "relevant_papers": 18,
            "full_text_papers": 12,
            "passages_total": 340,
            "evidence_bearing_papers": 3,
            "verified_claims": 3,
        },
    )

    # Validate report gate
    errors = validate_report(report)
    if errors:
        raise ValueError(f"Report validation gate failed: {errors}")
    print("Report structure passed validation gate with 0 errors.")

    now_iso = datetime.now(timezone.utc).isoformat()
    inv = ResearchInvestigation(
        id="sample-investigation-transformer",
        question=q,
        depth=ResearchDepth.STANDARD,
        sources=[ResearchSource.ARXIV, ResearchSource.OPENALEX],
        status="completed",
        report=report,
        papersAnalyzed=len(papers),
        totalPapers=len(papers),
        evidenceItems=len(citations),
        verifiedClaims=3,
        createdAt=now_iso,
        updatedAt=now_iso,
    )

    pdf_bytes = build_pdf_report(inv, validate=True)
    audit = audit_pdf_bytes(pdf_bytes)
    print(f"PDF audit results: {audit['draws']} draws across {audit['page_count']} pages | max draws per position: {audit['max_draws']}")
    assert audit["max_draws"] == 1, "PDF audit failed: duplicate draws detected!"

    out_path = root_dir / output_filename
    out_path.write_bytes(pdf_bytes)
    print(f"Successfully generated PDF report ({len(pdf_bytes):,} bytes) at: {out_path}")
    return out_path


if __name__ == "__main__":
    asyncio.run(generate_and_save_report())
