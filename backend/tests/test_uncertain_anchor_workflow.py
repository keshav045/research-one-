"""
Test workflow through uncertain anchor path with network and models mocked.
Verifies:
1. Pipeline completes without crashing (no JSON serialization error or missing imports).
2. Job reaches a final status (completed, completed_with_warnings, or insufficient_evidence).
3. debug_info is valid JSON and preserves alternate_paper_id / title without storing raw Paper objects.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backend.models.database import SessionLocal, create_tables, ResearchJob, Base
from backend.models.schemas import (
    Paper,
    PaperPassage,
    Claim,
    Citation,
    CitationStatus,
    ResearchReport,
    ReportSection,
    IntegrityMetrics,
)
from backend.services.research_workflow import run_research_pipeline


@pytest.mark.asyncio
async def test_workflow_uncertain_anchor_reaches_final_status(monkeypatch):
    create_tables()
    db = SessionLocal()

    job_id = "test-uncertain-anchor-job"
    db.query(ResearchJob).filter(ResearchJob.id == job_id).delete()
    db.commit()

    job = ResearchJob(
        id=job_id,
        question="Which paper introduced the Transformer architecture?",
        depth="Quick",
        sources='["arXiv", "Semantic Scholar"]',
        status="in_progress",
    )
    db.add(job)
    db.commit()

    p1 = Paper(
        id="arxiv:1706.03762",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arXiv.1706.03762",
        citationCount=100000,
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        source="arXiv",
        passages=[
            PaperPassage(
                id="pass-1",
                text="The dominant sequence transduction models are based on complex recurrent neural networks.",
                page=1,
                paper_id="arxiv:1706.03762",
            )
        ],
    )
    p2 = Paper(
        id="arxiv:2104.09864",
        title="RoFormer: Enhanced Transformer with Rotary Position Embedding",
        authors=["Su et al."],
        publicationYear=2021,
        journalConference="arXiv",
        doi="10.48550/arXiv.2104.09864",
        citationCount=2000,
        abstract="Positional encoding recently has shown effective in transformer architecture.",
        source="arXiv",
        passages=[],
    )

    # 1. Mock query planning
    plan_mock = AsyncMock(return_value={
        "question_type": "factual_lookup",
        "queries": ["Attention Is All You Need"],
        "expected_titles": ["Attention Is All You Need"],
        "sub_questions": ["What is Attention Is All You Need?"],
    })
    monkeypatch.setattr("backend.services.research_workflow.plan_queries", plan_mock)

    # 2. Mock candidate retrieval
    retrieve_mock = AsyncMock(return_value=[p1, p2])
    monkeypatch.setattr("backend.services.research_workflow.retrieve_papers", retrieve_mock)
    monkeypatch.setattr("backend.services.research_workflow.enrich_papers_with_s2", AsyncMock(return_value=[p1, p2]))
    monkeypatch.setattr("backend.services.research_workflow.enrich_papers_with_openalex", AsyncMock(return_value=[p1, p2]))

    # 3. Mock anchor selection with uncertain ensemble divergence
    anchor_debug_dict = {
        "anchor_paper_id": p1.id,
        "anchor_paper_title": p1.title,
        "anchor_citations": 100000,
        "anchor_rule": "ensemble_chasing_preferred",
        "anchor_confidence": "uncertain",
        "confidence_note": "Source paper uncertain: 'Attention Is All You Need' or 'RoFormer'",
        "alternate_paper_id": p2.id,
        "alternate_paper_title": p2.title,
    }
    monkeypatch.setattr(
        "backend.services.research_workflow.select_anchor_paper",
        AsyncMock(return_value=(p1, [p1, p2], anchor_debug_dict)),
    )

    # 4. Mock PDF extraction
    monkeypatch.setattr(
        "backend.services.research_workflow.extract_papers_batch",
        AsyncMock(return_value=[p1, p2]),
    )

    # 5. Mock VectorStore build
    monkeypatch.setattr("backend.services.research_workflow.VectorStore.build", MagicMock())

    # 6. Mock evidence extraction & verification
    mock_citation = Citation(
        id="cit-1",
        badgeNumber=1,
        claim="The dominant sequence transduction models are based on recurrent networks.",
        status=CitationStatus.VERIFIED,
        paperId=p1.id,
        paperTitle=p1.title,
        authors=", ".join(p1.authors),
        year=2017,
        page=1,
        passage="The dominant sequence transduction models are based on complex recurrent neural networks.",
        highlightSentence="The dominant sequence transduction models are based on complex recurrent neural networks.",
    )
    mock_claim = Claim(
        id="claim-1",
        text="The dominant sequence transduction models are based on recurrent networks.",
        status=CitationStatus.VERIFIED,
    )
    monkeypatch.setattr(
        "backend.services.research_workflow.extract_and_verify_evidence",
        MagicMock(return_value=([mock_claim], [mock_citation], [], {})),
    )

    # 7. Mock report synthesis
    dummy_report = ResearchReport(
        executiveSummary="Attention Is All You Need introduced the Transformer [1].",
        methodology="Methodology content",
        findings=[
            ReportSection(
                sectionTitle="Key Findings",
                paragraphs=[],
            )
        ],
        comparisonTable=[],
        conclusion="Conclusion text",
        references=[],
    )
    monkeypatch.setattr(
        "backend.services.research_workflow.synthesize_report",
        AsyncMock(return_value=(dummy_report, 1.0, [])),
    )

    # Run the pipeline
    try:
        await run_research_pipeline(job_id, db)
    finally:
        db.refresh(job)

    # Asserts
    print(f"Workflow finished with status: {job.status}, failure_reason: {job.failure_reason}")
    assert job.status in ("completed", "completed_with_warnings", "insufficient_evidence"), f"Job failed to reach final status: {job.status}"
    assert job.status != "failed", f"Job failed with reason: {job.failure_reason}"

    # Verify debug_info is valid JSON and clean
    debug = job.get_debug()
    assert isinstance(debug, dict)
    assert "anchor_paper" in debug
    assert debug["anchor_paper"]["anchor_confidence"] == "uncertain"
    assert debug["anchor_paper"]["alternate_paper_id"] in ("arxiv:2104.09864", p2.id)
    assert debug["anchor_paper"]["alternate_paper_title"] == p2.title
    # Crucial: alternate_paper Paper object must NOT be in the debug dict
    assert "alternate_paper" not in debug["anchor_paper"]

    # Verify raw debug_json parses as pure JSON
    parsed = json.loads(job.debug_json)
    assert parsed["anchor_paper"]["alternate_paper_title"] == p2.title

    db.query(ResearchJob).filter(ResearchJob.id == job_id).delete()
    db.commit()
