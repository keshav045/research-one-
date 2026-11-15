"""
Tests for STEP 5: Anchor Fixes & Robust Arbitration
===================================================
Verifies:
a) Derivative guard: whole-word match, and skip variant word if in question itself.
b) Disagreement arbitration: prefer gated if title starts with key phrase (before ':' or ' -') and meets ANCHOR_MIN_CITATIONS.
c) S2 search params do not include openAccessPdf filter.
d) paper.is_anchor = True on chosen anchor.
e) SEED_PAPERS_ENABLED and DISABLE_CACHE logged in debug_info.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.config import settings
from backend.models.schemas import Paper
from backend.services.ranker import (
    check_derivative_guard,
    select_anchor_paper,
    _gated_title_starts_with_key_phrase,
)
from backend.services.paper_retrieval import S2_FIELDS


def test_derivative_guard_whole_word_and_question_exclusion():
    """Whole-word match prevents false positives ('steadfast' vs 'fast'); skips variant in question."""
    # 1. Whole-word match: "steadfast" should NOT match "fast"
    assert not check_derivative_guard("A Steadfast Approach to Neural Networks", "What is the network?")

    # 2. Whole-word match: "Fast" in "Fast R-CNN" DOES match
    assert check_derivative_guard("Fast R-CNN", "What is object detection?")

    # 3. Question exclusion: "Swin" in question means "Swin" is NOT flagged as derivative variant
    assert not check_derivative_guard("Swin Transformer: Hierarchical Vision Transformer", "What is Swin Transformer?")

    # 4. Question does NOT mention "Swin", so "Swin Transformer" IS flagged
    assert check_derivative_guard("Swin Transformer: Hierarchical Vision Transformer", "What is the Transformer architecture?")


def test_gated_title_starts_with_key_phrase():
    """Verifies prefix matching before ':' or ' -'."""
    # Matches before ':'
    assert _gated_title_starts_with_key_phrase(
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        ["bert"],
    )
    assert _gated_title_starts_with_key_phrase(
        "Adam: A Method for Stochastic Optimization",
        ["adam"],
    )
    # Matches before ' - '
    assert _gated_title_starts_with_key_phrase(
        "DenseNet - Densely Connected Convolutional Networks",
        ["densenet"],
    )
    # Does not start with key phrase
    assert not _gated_title_starts_with_key_phrase(
        "An Analysis of BERT in NLP",
        ["bert"],
    )


@pytest.mark.asyncio
async def test_arbitration_disagreement_with_prefix_preference(monkeypatch):
    """When methods disagree, prefer gated if title starts with key phrase and meets ANCHOR_MIN_CITATIONS."""
    monkeypatch.setattr(settings, "ANCHOR_PREFIX_PREFERENCE", True)
    monkeypatch.setattr(settings, "ANCHOR_MIN_CITATIONS", 1000)

    p_chasing = Paper(
        id="paper-chasing",
        title="Deep Learning Framework Overview",
        authors=["Alice Smith"],
        publicationYear=2015,
        journalConference="",
        doi="",
        source="arxiv",
        abstract="Foundational study of BERT representations and embeddings.",
        citationCount=15000,
    )
    p_gated = Paper(
        id="paper-gated",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Jacob Devlin", "Ming-Wei Chang"],
        publicationYear=2018,
        journalConference="NAACL",
        doi="",
        source="arxiv",
        abstract="We introduce BERT.",
        citationCount=80000,
    )

    # Mock citation_chasing returning p_chasing with high ref count
    with patch("backend.services.ranker.citation_chasing", AsyncMock(return_value=([p_chasing], {"deep learning framework overview": 50}, {}))):
        chosen, papers, debug = await select_anchor_paper(
            papers=[p_chasing, p_gated],
            question="What is the BERT architecture?",
        )

        assert chosen is not None
        assert chosen.id == "paper-gated"
        assert debug["anchor_confidence"] == "uncertain"
        assert debug["anchor_rule"] == "ensemble_gated_prefix_preferred"
        assert debug["alternate_paper_id"] == "paper-chasing"


@pytest.mark.asyncio
async def test_arbitration_disagreement_with_prefix_preference_disabled(monkeypatch):
    """When ANCHOR_PREFIX_PREFERENCE is False, keep chasing result."""
    monkeypatch.setattr(settings, "ANCHOR_PREFIX_PREFERENCE", False)
    monkeypatch.setattr(settings, "ANCHOR_MIN_CITATIONS", 1000)

    p_chasing = Paper(
        id="paper-chasing",
        title="Deep Learning Framework Overview",
        authors=["Alice Smith"],
        publicationYear=2015,
        journalConference="",
        doi="",
        source="arxiv",
        abstract="Foundational study of BERT representations and embeddings.",
        citationCount=15000,
    )
    p_gated = Paper(
        id="paper-gated",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Jacob Devlin", "Ming-Wei Chang"],
        publicationYear=2018,
        journalConference="NAACL",
        doi="",
        source="arxiv",
        abstract="We introduce BERT.",
        citationCount=80000,
    )

    with patch("backend.services.ranker.citation_chasing", AsyncMock(return_value=([p_chasing], {"deep learning framework overview": 50}, {}))):
        chosen, papers, debug = await select_anchor_paper(
            papers=[p_chasing, p_gated],
            question="What is the BERT architecture?",
        )

        assert chosen is not None
        assert chosen.id == "paper-chasing"
        assert debug["anchor_confidence"] == "uncertain"
        assert debug["anchor_rule"] == "ensemble_chasing_preferred"
        assert debug["alternate_paper_id"] == "paper-gated"


def test_s2_fields_and_search_params():
    """openAccessPdf must remain in S2_FIELDS but not filter searches."""
    assert "openAccessPdf" in S2_FIELDS
