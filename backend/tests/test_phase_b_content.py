"""Tests for Phase B - Report content correctness & validation gate."""
import re
import pytest
from backend.models.schemas import (
    Citation, CitationStatus, ComparisonRow, Paper, ReportParagraph, ReportSection,
    ResearchDepth, ResearchInvestigation, ResearchReport,
)
from backend.services.local_llm_service import (
    anchor_state,
    build_comparison_table,
    build_limitations_from_stats,
    build_methodology_from_stats,
    collapse_repeated_citations,
    synthesize_report,
)
from backend.services.pdf_report import build_pdf_report
from backend.services.report_validator import validate_report
from backend.services.paper_retrieval import set_source_status, summarize_source_status
from streamlit_app import build_html_report, build_markdown_report


def _make_paper(pid: str = "p1", title: str = "Test Title", year: int = 2023, venue: str = "Venue X") -> Paper:
    return Paper(
        id=pid,
        title=title,
        authors=["Author One", "Author Two"],
        publicationYear=year,
        journalConference=venue,
        citationCount=42,
        abstract="Test abstract content.",
        source="Semantic Scholar",
        doi="10.1234/test",
    )


def _make_citation(
    cid: str = "c1",
    badge: int = 1,
    claim: str = "Test claim.",
    paper_id: str = "p1",
    title: str = "Test Title",
) -> Citation:
    return Citation(
        id=cid,
        badgeNumber=badge,
        claim=claim,
        status=CitationStatus.VERIFIED,
        paperId=paper_id,
        paperTitle=title,
        authors="Author One",
        year=2023,
        page=1,
        passage="Test passage.",
        highlightSentence=claim,
    )


def test_b1_comparison_table_fields():
    """B1: Table row shows year under Year and venue under Venue."""
    paper = _make_paper()
    rows = build_comparison_table([paper], [])
    assert len(rows) == 1
    row = rows[0]
    assert row.dataset == "Not extracted"
    assert row.f1Score == "Not extracted"
    assert row.mapScore == "Not extracted"
    assert row.year == "2023"
    assert row.venue == "Venue X"
    assert row.citationCount == "42"

    inv = ResearchInvestigation(
        id="inv-1",
        question="Test Question",
        depth=list(ResearchDepth)[0],
        sources=[],
        status="completed",
        createdAt="2026-10-07T00:00:00Z",
        updatedAt="2026-10-07T00:00:00Z",
        report=ResearchReport(
            executiveSummary="Summary text.",
            methodology="Methodology text.",
            findings=[],
            comparisonTable=rows,
            limitations=[],
            conclusion="Conclusion text.",
            references=[paper],
        ),
    )

    md = build_markdown_report(inv)
    assert "| Paper / Model | Year | Venue | Citations |" in md
    assert "| **Test Title** | 2023 | Venue X | 42 |" in md

    html = build_html_report(inv)
    assert "<th>Paper / Model</th><th>Year</th><th>Venue</th><th>Citations</th>" in html
    assert "<td><strong>Test Title</strong></td><td>2023</td><td>Venue X</td><td>42</td>" in html


@pytest.mark.asyncio
async def test_b2_citation_numbering_system():
    """B2: 2 claims from 1 paper yield reference-based citations where 1 <= n <= len(references)."""
    paper = _make_paper()
    cites = [
        _make_citation("c1", 1, "First claim for test paper.", paper.id, paper.title),
        _make_citation("c2", 2, "Second claim for same paper.", paper.id, paper.title),
    ]
    report, _, _ = await synthesize_report(
        question="What are empirical benefits of test method?",
        depth=list(ResearchDepth)[0],
        papers=[paper],
        claims=[],
        citations=cites,
    )
    assert len(report.references) == 1
    # Check all [n] in all text fields
    all_text = report.executiveSummary + " " + report.conclusion + " " + " ".join(
        p.text for sec in report.findings for p in sec.paragraphs
    )
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", all_text)]
    assert markers, "Expected citation markers"
    for m in markers:
        assert 1 <= m <= len(report.references), f"Citation marker [{m}] exceeds reference count {len(report.references)}"
    assert "[1][1]" not in all_text


def test_b3_no_raw_errors_in_prose():
    """B3: Simulating 429 produces clean summary without HTTP, 429, or all:\" in prose."""
    set_source_status("Semantic Scholar", "RATE_LIMITED")
    set_source_status("arXiv", "PARTIAL")
    set_source_status("OpenAlex", "OK")

    status_str = summarize_source_status()
    assert "rate limited" in status_str
    assert "429" not in status_str
    assert "HTTP" not in status_str

    limits = build_limitations_from_stats(
        anchor_paper=None,
        anchor_confidence="none",
        integrity=1.0,
        removed_sentences=[],
        retrieval_warnings=["Semantic Scholar HTTP 429 rate limit for query 'all:\"machine learning\"'"],
    )
    joined_limits = " ".join(limits)
    assert "429" not in joined_limits
    assert "HTTP" not in joined_limits
    assert 'all:"' not in joined_limits


@pytest.mark.asyncio
async def test_b4_summary_does_not_copy_findings():
    """B4: No sentence longer than 40 chars appears in both summary and findings."""
    paper = _make_paper()
    cites = [
        _make_citation(
            "c1", 1,
            "Transformers utilize self-attention mechanisms to aggregate tokens across layers.",
            paper.id, paper.title
        )
    ]
    report, _, _ = await synthesize_report(
        question="How do attention layers scale?",
        depth=list(ResearchDepth)[0],
        papers=[paper],
        claims=[],
        citations=cites,
    )
    exec_sents = [s.strip().lower() for s in re.split(r"[.!?]\s+", report.executiveSummary) if len(s.strip()) > 40]
    finding_sents = [
        s.strip().lower()
        for sec in report.findings
        for p in sec.paragraphs
        for s in re.split(r"[.!?]\s+", p.text)
        if len(s.strip()) > 40
    ]
    overlap = set(exec_sents).intersection(set(finding_sents))
    assert not overlap, f"Overlapping sentence found between summary and findings: {overlap}"


@pytest.mark.asyncio
async def test_b5_no_hardcoded_llm_inference_words_on_unrelated_questions():
    """B5: An unrelated question (e.g. biology or climate) must have zero LLM-inference jargon."""
    paper = _make_paper(title="Marine Microplastics Accumulation in Coastal Fish", venue="Marine Biology")
    cites = [
        _make_citation(
            "c1", 1,
            "Microplastics concentrate in digestive tracts of pelagic marine species.",
            paper.id, paper.title
        )
    ]
    report, _, _ = await synthesize_report(
        question="How do microplastics affect coastal marine organisms?",
        depth=list(ResearchDepth)[0],
        papers=[paper],
        claims=[],
        citations=cites,
    )
    all_report_text = (
        report.executiveSummary + " " +
        report.conclusion + " " +
        " ".join(sec.sectionTitle + " " + p.text for sec in report.findings for p in sec.paragraphs)
    ).lower()

    forbidden_inference_words = ["kv-cache", "pagedattention", "speculative decoding", "quantization", "vllm"]
    for w in forbidden_inference_words:
        assert w not in all_report_text, f"Hardcoded LLM-inference word '{w}' leaked into unrelated report!"


def test_b6_anchor_state_consistency():
    """B6: With anchor_paper=None and anchor_confidence='high', no text says 'confidence is high'."""
    label, sent = anchor_state(None, "high")
    assert label == "Not identified"
    assert "Not identified" in sent

    method = build_methodology_from_stats(
        question="What are methods?",
        papers=[],
        anchor_paper=None,
        anchor_confidence="high",
    )
    assert "confidence is high" not in method.lower()
    assert "Anchor confidence: None" in method


def test_b7_report_validator_gate():
    """B7: Validation gate catches all invariant violations."""
    paper = _make_paper()
    # 1. Valid report passes
    good_report = ResearchReport(
        executiveSummary="High-level overview of literature.",
        methodology="Methodology content.",
        findings=[ReportSection(
            sectionTitle="Section A",
            paragraphs=[ReportParagraph(
                text="Verbatim claim asserted here [1].",
                citations=[_make_citation("c1", 1, "Verbatim claim asserted here.", paper.id, paper.title)]
            )]
        )],
        comparisonTable=[],
        limitations=["Small sample."],
        conclusion="Foundational anchor paper: Not identified. Anchor confidence: None.",
        references=[paper],
    )
    assert validate_report(good_report) == []

    # 2. Out of range citation marker [99]
    bad_cite_report = good_report.model_copy(update={"executiveSummary": "Summary with out of range cite [99]."})
    assert any("outside valid reference range" in err for err in validate_report(bad_cite_report))

    # 3. Raw error string in report
    bad_error_report = good_report.model_copy(update={"conclusion": "Encountered HTTP 429 during fetch."})
    assert any("Raw error string" in err for err in validate_report(bad_error_report))

    # 4. Anchor contradiction
    bad_anchor_report = good_report.model_copy(update={
        "methodology": "Foundational anchor paper: Not identified.",
        "conclusion": "Anchor confidence is high.",
    })
    assert any("Anchor contradiction" in err for err in validate_report(bad_anchor_report))


@pytest.mark.asyncio
async def test_b8_coverage_warning_when_evidence_bearing_papers_one_or_fewer():
    """B8: When evidence-bearing papers are 1 or fewer, the coverage warning is present."""
    paper = _make_paper()
    cites = [
        _make_citation("c1", 1, "Single evidence paper finding.", paper.id, paper.title)
    ]
    report, _, _ = await synthesize_report(
        question="Comprehensive survey of all federated learning strategies?",
        depth=list(ResearchDepth)[0],
        papers=[paper],
        claims=[],
        citations=cites,
    )
    assert "Research coverage is insufficient to answer this question in general" in report.executiveSummary
    assert "Only 1 relevant paper(s) with usable evidence were retrieved" in report.executiveSummary
