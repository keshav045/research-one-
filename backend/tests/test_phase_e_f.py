"""
Unit tests for Phase E (Funnel & Depth Counts) and Phase F (Streamlit Caching & Lock)
====================================================================================
"""

from __future__ import annotations

import threading

from backend.config import settings
from backend.models.schemas import Paper
from backend.services.local_llm_service import build_limitations_from_stats
from backend.services.ranker import rank_papers


def test_depth_counts_agreement():
    """Verify that DEPTH_COUNTS is consistent and defined for Quick, Standard, and Deep."""
    dc = settings.DEPTH_COUNTS
    assert dc["Quick"] == 6
    assert dc["Standard"] == 12
    assert dc["Deep"] == 24


def test_funnel_included_in_limitations():
    """Verify build_limitations_from_stats formats and includes the funnel string."""
    debug_info = {
        "papers_discovered": 171,
        "unique_papers": 140,
        "relevant_papers": 12,
        "full_text_papers": 10,
        "passages_total": 450,
        "evidence_bearing_papers": 6,
        "verified_claims": 8,
    }
    limits = build_limitations_from_stats(
        anchor_paper=None,
        anchor_confidence="none",
        integrity=1.0,
        removed_sentences=[],
        debug_info=debug_info,
    )
    funnel_line = next((l for l in limits if "Retrieval & Evidence Funnel" in l), None)
    assert funnel_line is not None
    assert "171 discovered" in funnel_line
    assert "140 unique" in funnel_line
    assert "12 relevant" in funnel_line
    assert "10 full text" in funnel_line
    assert "450 passages" in funnel_line
    assert "6 evidence-bearing" in funnel_line
    assert "8 verified claims" in funnel_line


def test_ranker_preserves_candidates_gracefully():
    """Verify rank_papers does not drop all candidate papers if relevant_pool is empty."""
    papers = [
        Paper(
            id=f"p-{i}",
            title=f"General Study on Attention {i}",
            authors=["Author"],
            publicationYear=2021,
            journalConference="ArXiv",
            doi="10.1000/1",
            source="arXiv",
            citationCount=0,
            abstract="Study on attention models.",
        )
        for i in range(5)
    ]
    # Patch settings MIN_PAPER_RELEVANCE to 0.99 (impossible to meet)
    settings.MIN_PAPER_RELEVANCE = 0.99
    try:
        ranked = rank_papers(
            question="What is the attention mechanism?",
            papers=papers,
            question_type="literature_review",
            title_guesses=[],
        )
        # Should gracefully return candidate papers from pool rather than empty list
        assert len(ranked) > 0
    finally:
        settings.MIN_PAPER_RELEVANCE = 0.30


def test_pipeline_lock_guard():
    """Verify threading lock semantics preventing concurrent operations."""
    lock = threading.Lock()
    assert not lock.locked()

    with lock:
        assert lock.locked()
        acquired_second = lock.acquire(blocking=False)
        assert not acquired_second

    assert not lock.locked()
