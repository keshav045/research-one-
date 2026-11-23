"""
Reproduction test for LangGraph research pipeline Deep run.
Verifies that candidate retrieval, rerank_fetch, verification, and synthesis
execute and populate all required state keys: passages, verified, rejected, report.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from backend.models.schemas import (
    Citation,
    CitationStatus,
    Claim,
    Paper,
    PaperPassage,
    ReportParagraph,
    ReportSection,
    ResearchReport,
)
from backend.services.research_graph import build_research_graph


@pytest.mark.asyncio
async def test_deep_run_reproduction(monkeypatch):
    """
    Simulates the Deep run for the Transformer architecture question.
    Stubs retrieval, PDF extraction, and NLI verification.
    Asserts that the final state contains:
      - passages
      - verified
      - rejected
      - report
    """
    question = (
        "Which paper introduced the Transformer architecture, "
        "and what was its BLEU score on the WMT 2014 English-to-German task?"
    )

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
                text=(
                    "The dominant sequence transduction models are based on complex recurrent neural networks. "
                    "On the WMT 2014 English-to-German translation task, the big transformer model achieves "
                    "a BLEU score of 28.4."
                ),
                page=1,
                paper_id="arxiv:1706.03762",
            )
        ],
    )

    # 1. Stub query planner
    mock_plan = {
        "question_type": "factual_lookup",
        "queries": ["Attention Is All You Need Transformer", "Transformer BLEU score WMT 2014"],
        "expected_titles": ["Attention Is All You Need"],
        "sub_questions": ["Which paper introduced the Transformer?", "What was its BLEU score on WMT 2014?"],
        "target_concepts": ["Transformer", "BLEU", "Attention"],
    }
    monkeypatch.setattr("backend.services.research_graph.plan_queries", AsyncMock(return_value=mock_plan))

    # 2. Stub candidate retrieval to return fixed paper
    monkeypatch.setattr("backend.services.research_graph.retrieve_papers", AsyncMock(return_value=[p1]))

    # 3. Stub PDF extraction to return fixed paper with passages
    monkeypatch.setattr("backend.services.research_graph.extract_papers_batch", AsyncMock(return_value=[p1]))

    # 4. Stub NLI evidence verification
    mock_citation = Citation(
        id="cite-1",
        badgeNumber=1,
        claim="The Transformer achieved a BLEU score of 28.4 on WMT 2014 English-to-German.",
        status=CitationStatus.VERIFIED,
        paperId=p1.id,
        paperTitle=p1.title,
        authors=", ".join(p1.authors),
        year=2017,
        page=1,
        passage=p1.passages[0].text,
        highlightSentence="On the WMT 2014 English-to-German translation task, the big transformer model achieves a BLEU score of 28.4.",
    )
    mock_claim = Claim(
        id="claim-1",
        text="The Transformer achieved a BLEU score of 28.4 on WMT 2014 English-to-German.",
        status=CitationStatus.VERIFIED,
    )
    monkeypatch.setattr(
        "backend.services.research_graph.extract_and_verify_evidence",
        MagicMock(return_value=([mock_claim], [mock_citation], [], {})),
    )

    # 5. Stub report synthesis
    mock_report = ResearchReport(
        executiveSummary="Attention Is All You Need introduced the Transformer architecture, scoring 28.4 BLEU on WMT 2014 [1].",
        methodology="Transformer architecture evaluation",
        findings=[
            ReportSection(
                sectionTitle="Empirical Results",
                paragraphs=[
                    ReportParagraph(
                        text="On the WMT 2014 English-to-German task, the big model achieved a BLEU score of 28.4 [1].",
                        citations=[mock_citation],
                    )
                ],
            )
        ],
        comparisonTable=[],
        conclusion="The Transformer established new state-of-the-art results.",
        references=[p1],
    )
    monkeypatch.setattr(
        "backend.services.research_graph.synthesize_report",
        AsyncMock(return_value=(mock_report, 1.0, [])),
    )

    # Build graph and run
    graph = build_research_graph()
    final_state = await graph.ainvoke({
        "question": question,
        "depth": "Deep",
        "sources": ["arXiv", "Semantic Scholar", "OpenAlex"],
    })

    # Assertions
    assert "passages" in final_state, f"Missing 'passages' in final state keys: {list(final_state.keys())}"
    assert "verified" in final_state, f"Missing 'verified' in final state keys: {list(final_state.keys())}"
    assert "rejected" in final_state, f"Missing 'rejected' in final state keys: {list(final_state.keys())}"
    assert "report" in final_state, f"Missing 'report' in final state keys: {list(final_state.keys())}"
    assert final_state["report"] is not None
    assert len(final_state["passages"]) > 0
    assert len(final_state["verified"]) > 0
