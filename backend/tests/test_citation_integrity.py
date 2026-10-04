"""
Tests for citation integrity calculation
==========================================
Covers:
  - _compute_metrics: 100%, partial, zero, mixed status, contradicted, atomic metadata
  - _count_uncited_sentences: exec summary, cited paras with/without inline markers
  - Placeholder report: no citation pre-stamped as VERIFIED
  - Invariant: headline score == breakdown-derivable score

Run from the project root:
    python -m pytest backend/tests/test_citation_integrity.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so the service module can be imported without heavy deps
# ---------------------------------------------------------------------------

import importlib.machinery

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


# Make sure the project root is on the path FIRST so relative imports resolve
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Stub optional deps that might not be installed ────────────────────────────
for _s in [
    "google",
    "google.generativeai",
]:
    sys.modules.setdefault(_s, _make_stub(_s))

# numpy IS installed — do NOT stub it.

# ── Stub the config module so it doesn't fail on missing env vars ─────────────
_cfg = types.ModuleType("backend.config")

class _Settings:
    GEMINI_API_KEY: str = "test-key"
    GEMINI_MODEL: str = "gemini-test"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    PDF_CACHE_DIR: str = "/tmp/pdf_cache"
    DATABASE_URL: str = "sqlite:///./test.db"
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    MAX_PDF_WORKERS: int = 2
    DEPTH_COUNTS: dict = {"Quick": 6, "Standard": 12, "Deep": 24}
    is_gemini_configured: bool = True

_cfg.settings = _Settings()  # type: ignore[attr-defined]
sys.modules["backend.config"] = _cfg

from backend.models.schemas import (  # noqa: E402
    Citation,
    CitationStatus,
    EntailmentVerdict,
    AtomicClaimVerification,
    ResearchReport,
    ReportSection,
    ReportParagraph,
    Paper,
)
from backend.services.research_workflow import (  # noqa: E402
    _compute_metrics,
    _count_uncited_sentences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_atomic(verdict: EntailmentVerdict, confidence: float = 0.9) -> AtomicClaimVerification:
    return AtomicClaimVerification(
        id="a-1",
        atomicClaim="The model achieves 95% accuracy.",
        matchedSentence="accuracy 95%",
        verdict=verdict,
        confidence=confidence,
        reasoning="test",
    )


def _make_citation(
    idx: int,
    status: CitationStatus,
    atomics: list[AtomicClaimVerification] | None = None,
) -> Citation:
    return Citation(
        id=f"cite-{idx}",
        badgeNumber=idx,
        claim=f"Claim {idx}.",
        status=status,
        paperId=f"paper-{idx}",
        paperTitle=f"Paper {idx}",
        authors="Author A",
        year=2024,
        page=1,
        passage="Some passage text used as source evidence.",
        highlightSentence="Some passage",
        atomicClaims=atomics or [],
        entailmentScore=1.0 if status == CitationStatus.VERIFIED else 0.0,
        extractionConfidence=0.9,
    )


def _make_report(
    exec_summary: str = "This is a short summary. [1] It discusses results.",
    sections: list[ReportSection] | None = None,
) -> ResearchReport:
    return ResearchReport(
        executiveSummary=exec_summary,
        methodology="Search: arXiv",
        findings=sections or [],
        comparisonTable=[],
        computationalRequirements="",
        contradictoryEvidence="",
        limitations=[],
        conclusion="Done.",
        references=[],
    )


def _make_section(text: str, has_citation: bool) -> ReportSection:
    cite = _make_citation(1, CitationStatus.VERIFIED) if has_citation else None
    return ReportSection(
        sectionTitle="Sec",
        paragraphs=[
            ReportParagraph(
                text=text,
                citations=[cite] if cite else [],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests: _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_100_percent_all_verified_fully_cited(self):
        """All citations VERIFIED, all exec-summary sentences have [N] markers => coverage 100%."""
        citations = [
            _make_citation(1, CitationStatus.VERIFIED),
            _make_citation(2, CitationStatus.VERIFIED),
            _make_citation(3, CitationStatus.VERIFIED),
        ]
        # Every sentence is followed by a citation marker
        report = _make_report(
            exec_summary="Confirmed result A. [1] Confirmed result B. [2] Confirmed result C. [3]",
            sections=[
                ReportSection(
                    sectionTitle="Sec",
                    paragraphs=[
                        ReportParagraph(
                            text="All sourced. [1]",
                            citations=[citations[0]],
                        )
                    ],
                )
            ],
        )
        metrics = _compute_metrics(citations, report)
        assert metrics.verifiedClaims == 3
        assert metrics.unsupportedClaims == 0
        assert metrics.contradictedClaims == 0
        assert metrics.potentialConflicts == 0
        assert metrics.citationCoverage == 100

    def test_zero_percent_all_unsupported_all_uncited(self):
        """All citations UNSUPPORTED, no inline markers in exec summary => coverage 0%."""
        citations = [
            _make_citation(1, CitationStatus.UNSUPPORTED),
            _make_citation(2, CitationStatus.UNSUPPORTED),
        ]
        report = _make_report(
            exec_summary=(
                "This paper claims extraordinary performance gains without evidence. "
                "No citations are provided for these statements here. "
                "The results are completely unverified by any source."
            ),
        )
        metrics = _compute_metrics(citations, report)
        assert metrics.verifiedClaims == 0
        assert metrics.unsupportedClaims == 2
        assert metrics.citationCoverage == 0

    def test_partial_coverage_mixed_statuses(self):
        """
        4 citations: 2 VERIFIED, 1 PARTIALLY_SUPPORTED, 1 UNSUPPORTED.
        score_sum = 2*1.0 + 1*0.5 + 1*0.0 = 2.5
        total = 4 + uncited_sentences
        coverage = round(2.5 / total * 100)
        """
        citations = [
            _make_citation(1, CitationStatus.VERIFIED),
            _make_citation(2, CitationStatus.VERIFIED),
            _make_citation(3, CitationStatus.PARTIALLY_SUPPORTED),
            _make_citation(4, CitationStatus.UNSUPPORTED),
        ]
        # Two uncited sentences before the cited ones
        report = _make_report(
            exec_summary=(
                "This is an uncited sentence about AI model compression techniques. "
                "This is another uncited sentence about evaluation benchmarks used. "
                "[1] Verified result. [2] Another verified result."
            ),
        )
        metrics = _compute_metrics(citations, report)
        assert metrics.verifiedClaims == 2
        assert metrics.partiallySupportedClaims == 1
        assert metrics.unsupportedClaims == 1
        n_uncited = metrics.uncitedSentences
        total = 4 + n_uncited
        expected = round((2 * 1.0 + 0.5) / total * 100)
        assert metrics.citationCoverage == expected

    def test_contradicted_contributes_zero_score(self):
        """CONTRADICTED citation scores 0 and increments potentialConflicts."""
        citations = [
            _make_citation(1, CitationStatus.VERIFIED),
            _make_citation(2, CitationStatus.CONTRADICTED),
        ]
        report = _make_report(exec_summary="Result A. [1] Result B. [2]")
        metrics = _compute_metrics(citations, report)
        assert metrics.contradictedClaims == 1
        assert metrics.potentialConflicts == 1
        n_uncited = metrics.uncitedSentences
        total = 2 + n_uncited
        expected = round(1.0 / total * 100)
        assert metrics.citationCoverage == expected

    def test_atomic_claims_reported_as_metadata_only(self):
        """
        Atomic claim counts are informational metadata.
        Coverage is derived from citation *status*, not atomic-claim re-sum.
        """
        atomics_v = [_make_atomic(EntailmentVerdict.ENTAILS)] * 3
        atomics_n = [_make_atomic(EntailmentVerdict.NEUTRAL)]

        citations = [
            _make_citation(1, CitationStatus.VERIFIED, atomics=atomics_v),
            _make_citation(2, CitationStatus.VERIFIED, atomics=[]),
            _make_citation(3, CitationStatus.PARTIALLY_SUPPORTED, atomics=atomics_n),
        ]
        report = _make_report(exec_summary="Cited. [1][2][3]")
        metrics = _compute_metrics(citations, report)

        assert metrics.totalAtomicClaims == 4   # 3 + 0 + 1
        assert metrics.entailedAtomicClaims == 3

        n_uncited = metrics.uncitedSentences
        total = 3 + n_uncited
        expected = round((1.0 + 1.0 + 0.5) / total * 100)
        assert metrics.citationCoverage == expected

    def test_no_citations_empty_report(self):
        """No citations, empty report text => coverage 0, no error."""
        report = _make_report(exec_summary="", sections=[])
        metrics = _compute_metrics([], report)
        assert metrics.citationCoverage == 0
        assert metrics.verifiedClaims == 0
        assert metrics.totalAtomicClaims == 0

    def test_headline_equals_breakdown_invariant(self):
        """
        Key invariant: headline citationCoverage == re-applying the documented
        formula to the breakdown counts from the same metrics object.
        """
        citations = [
            _make_citation(1, CitationStatus.VERIFIED),
            _make_citation(2, CitationStatus.PARTIALLY_SUPPORTED),
            _make_citation(3, CitationStatus.UNSUPPORTED),
        ]
        report = _make_report(
            exec_summary=(
                "An uncited opening sentence about the scope and context of this study. "
                "[1] Supported result. [2] Partially supported. [3] Unsupported claim."
            ),
        )
        metrics = _compute_metrics(citations, report)

        score = (
            metrics.verifiedClaims * 1.0
            + metrics.partiallySupportedClaims * 0.5
        )
        total = len(citations) + metrics.uncitedSentences
        expected = round(score / total * 100) if total > 0 else 0

        assert metrics.citationCoverage == expected, (
            f"Headline {metrics.citationCoverage}% != breakdown-derived {expected}% "
            f"(v={metrics.verifiedClaims}, p={metrics.partiallySupportedClaims}, "
            f"u={metrics.unsupportedClaims}, uncited={metrics.uncitedSentences})"
        )


# ---------------------------------------------------------------------------
# Tests: _count_uncited_sentences
# ---------------------------------------------------------------------------

class TestCountUncitedSentences:

    def test_fully_cited_exec_summary_zero_uncited(self):
        report = _make_report(exec_summary="Confirmed result. [1] Another result. [2]")
        assert _count_uncited_sentences(report) == 0

    def test_exec_summary_counts_uncited_sentences(self):
        report = _make_report(
            exec_summary=(
                "First uncited sentence about research methodology choices made. "
                "Second uncited sentence about evaluation benchmarks selected here. "
                "Third uncited sentence about experimental implications of findings."
            )
        )
        count = _count_uncited_sentences(report)
        assert count == 3

    def test_cited_paragraph_inline_markers_zero_extra(self):
        """A cited paragraph where all sentences have inline [N] markers -> 0 uncited.
        
        The marker must appear *within* or at the end of the sentence (before the
        sentence-split boundary). 'Claim. [1]' splits into 'Claim.' (uncited) and
        '[1]...' — so markers after the period don't count. Use 'Claim [1].' instead.
        """
        report = _make_report(
            exec_summary="",
            sections=[_make_section("Claim supported here [1]. Another supported claim [1].", has_citation=True)],
        )
        assert _count_uncited_sentences(report) == 0

    def test_cited_paragraph_with_uncited_sentences(self):
        """
        A paragraph with a citation but containing sentences without inline markers:
        those sentences must be counted as uncited.
        """
        report = _make_report(
            exec_summary="",
            sections=[
                _make_section(
                    "This sentence has a marker. [1] This one does not have any marker at all.",
                    has_citation=True,
                )
            ],
        )
        count = _count_uncited_sentences(report)
        assert count >= 1

    def test_fully_uncited_paragraph_counts_min_one(self):
        """A paragraph without any citations contributes at least 1 uncited unit."""
        report = _make_report(
            exec_summary="",
            sections=[_make_section("A very short text.", has_citation=False)],
        )
        count = _count_uncited_sentences(report)
        assert count >= 1

    def test_empty_report(self):
        report = _make_report(exec_summary="", sections=[])
        assert _count_uncited_sentences(report) == 0


# ---------------------------------------------------------------------------
# Tests: placeholder report must not pre-stamp VERIFIED
# ---------------------------------------------------------------------------

class TestPlaceholderReportNoFakeVerified:

    def test_placeholder_citations_are_not_verified(self):
        """
        _build_insufficient_evidence_report must NOT set CitationStatus.VERIFIED.
        Pre-stamping VERIFIED would inflate coverage artificially.
        """
        from backend.services.research_workflow import _build_insufficient_evidence_report

        paper = Paper(
            id="paper-1",
            title="A Study on Attention Mechanisms in Transformers",
            authors=["Alice Smith", "Bob Jones"],
            publicationYear=2024,
            journalConference="NeurIPS 2024",
            doi="10.1234/test.2024",
            source="arXiv",
            abstract="We study attention mechanisms and propose improvements.",
        )
        report = _build_insufficient_evidence_report("What is attention?", [paper], [], "Test reason")

        all_citations: list[Citation] = []
        for section in report.findings:
            for para in section.paragraphs:
                all_citations.extend(para.citations)

        pre_verified = [c for c in all_citations if c.status == CitationStatus.VERIFIED]
        assert pre_verified == [], (
            f"Insufficient evidence report pre-stamped {len(pre_verified)} citation(s) as VERIFIED: {[c.id for c in pre_verified]}"
        )
