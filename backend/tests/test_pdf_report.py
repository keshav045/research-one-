"""Regression tests for the PDF exporter (overprint, table columns, glyphs, citations)."""
import collections
import re

import pytest

fitz = pytest.importorskip("fitz")
from pypdf import PdfReader  # noqa: E402

from backend.models.schemas import (  # noqa: E402
    Citation, CitationStatus, ComparisonRow, Paper, ReportParagraph, ReportSection,
    ResearchDepth, ResearchInvestigation, ResearchReport,
)
from backend.services.pdf_report import build_pdf_report  # noqa: E402

TEXT_OP = re.compile(r"1 0 0 1 ([\d.\-]+) ([\d.\-]+) Tm\s*/\w+ [\d.]+ Tf[^\[]*\[<([0-9a-f]+)>\]TJ")


def _investigation(summary: str, n_cites: int = 2) -> ResearchInvestigation:
    paper = Paper(
        id="p1", title="Efficient Memory Management for Large Language Model Serving with PagedAttention",
        authors=["Woosuk Kwon", "Z. Li", "Siyuan Zhuang", "Y. Sheng"], publicationYear=2023,
        journalConference="Symposium on Operating Systems Principles", doi="10.1145/3600006.3613165",
        source="Semantic Scholar", abstract="x", citationCount=8884,
    )
    cites = [
        Citation(id=f"c{i}", badgeNumber=i, claim=f"claim {i}", status=CitationStatus.VERIFIED, paperId="p1",
                 paperTitle=paper.title, authors="Kwon et al.", year=2023, page=1, passage="p", highlightSentence="h")
        for i in range(1, n_cites + 1)
    ]
    report = ResearchReport(
        executiveSummary=summary, methodology="m",
        findings=[ReportSection(sectionTitle="Verified Findings", paragraphs=[ReportParagraph(text=summary, citations=cites)])],
        comparisonTable=[ComparisonRow(
            model=paper.title, dataset="2023", f1Score=paper.journalConference, mapScore="8,884",
            title=paper.title, authors="Kwon", year="2023", venue=paper.journalConference, citationCount="8,884")],
        limitations=["Citation integrity (33.3%) is below the 80% threshold."],
        conclusion="Foundational anchor paper: Not identified.", references=[paper],
    )
    return ResearchInvestigation(
        id="j1", question="What are the most effective techniques for optimizing large language model inference?",
        depth=list(ResearchDepth)[0], sources=[], status="completed",
        createdAt="2026-10-07T00:00:00Z", updatedAt="2026-10-07T00:00:00Z",
        report=report, research_confidence="LOW", citation_integrity=33.3, papersAnalyzed=1,
    )


def _pdf_text_and_draws(pdf_bytes: bytes):
    reader = PdfReader(__import__("io").BytesIO(pdf_bytes))
    draws = collections.Counter()
    for page_no, page in enumerate(reader.pages):
        data = page.get_contents().get_data().decode("latin-1")
        for x, y, _ in TEXT_OP.findall(data):
            draws[(page_no, x, y)] += 1
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text, draws


SUMMARY = ("Research coverage is currently limited. Only one relevant paper with usable evidence was identified. "
           "PagedAttention provides verified evidence for memory-efficient LLM serving [1].")


def test_every_line_is_drawn_once():
    _, draws = _pdf_text_and_draws(build_pdf_report(_investigation(SUMMARY), validate=False))
    assert draws, "no text drawn"
    assert max(draws.values()) == 1, f"overprinted text: max {max(draws.values())} draws at one position"


def test_extracted_text_has_no_repeats():
    text, _ = _pdf_text_and_draws(build_pdf_report(_investigation(SUMMARY), validate=False))
    assert text.count("Research coverage is currently limited") == 2  # summary + findings, not hundreds
    assert len(text) < 4000


def test_no_phantom_citation_and_no_question_mark_glyphs():
    text, _ = _pdf_text_and_draws(build_pdf_report(_investigation(SUMMARY + " More [2]."), validate=False))
    assert "[2]" not in text.split("References")[0]
    assert "? " not in text.replace("?'", "")  # bullet no longer rendered as '?'


def test_comparison_table_columns_are_correct():
    text, _ = _pdf_text_and_draws(build_pdf_report(_investigation(SUMMARY), validate=False))
    assert "Symposium on Operating Systems Principles" in text
    assert "2023" in text and "8,884" in text
