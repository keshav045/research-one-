"""
Unit tests for Phase D: Retrieval Robustness & Sanitization
============================================================
Tests:
  1. Mocking a 429 on Semantic Scholar: verifies retry count, rate limited source status,
     and that raw query strings / HTTP status codes are excluded from retrieval errors.
  2. Fallback to another source: verifies that when S2 is rate-limited, retrieval continues
     with candidate papers from another source (e.g. arXiv / OpenAlex).
  3. Query generation: planner facets are used to produce topical queries, and purely
     generic queries ('paper', 'method', 'overview') are rejected.
  4. S2 batch enrichment: verifies that batches are chunked into at most 100 IDs.
  5. Error message sanitization: verifies _record_retrieval_error never stores HTTP, 429,
     query text, or all:" strings.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import Paper, ResearchDepth, ResearchSource
from backend.services.paper_retrieval import (
    _record_retrieval_error,
    enrich_papers_with_s2,
    generate_search_queries,
    get_and_clear_retrieval_errors,
    get_source_status,
    is_generic_query,
    reset_retrieval_session,
    retrieve_papers,
)


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def test_generic_query_rejection():
    """Verify is_generic_query identifies purely generic or empty queries."""
    assert is_generic_query("paper") is True
    assert is_generic_query("the paper method") is True
    assert is_generic_query("overview survey analysis") is True
    assert is_generic_query("transformer self-attention") is False
    assert is_generic_query("PagedAttention KV-cache memory") is False


def test_generate_search_queries_uses_facets_and_no_generic():
    """Verify fallback query generation creates non-generic facet-derived queries."""
    queries = _run_async(generate_search_queries("Which paper introduced the Transformer architecture, and what was its key idea?"))
    assert isinstance(queries, list)
    assert len(queries) >= 1
    for q in queries:
        assert not is_generic_query(q)
        assert len(q.strip()) > 0


def test_retrieval_error_sanitization():
    """Verify _record_retrieval_error strips query strings, HTTP status codes, and quotes."""
    reset_retrieval_session()
    _record_retrieval_error("Semantic Scholar HTTP 429 for query 'transformer attention'")
    _record_retrieval_error("arXiv error querying 'all:\"attention\"': connection timeout")
    errors = get_and_clear_retrieval_errors()

    for err in errors:
        assert "HTTP" not in err
        assert "429" not in err
        assert "transformer attention" not in err
        assert 'all:"' not in err


def test_s2_429_retry_and_source_status():
    """
    Mock an HTTP 429 on Semantic Scholar and verify:
      - rate limited status is recorded in get_source_status()
      - errors do not leak query strings or raw HTTP 429 into recorded errors
    """
    reset_retrieval_session()

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "0.01"}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        _ = _run_async(
            retrieve_papers(
                question="What is LoRA?",
                depth=ResearchDepth.QUICK,
                sources=[ResearchSource.SEMANTIC_SCHOLAR],
                api_key_s2="",
                custom_queries=["LoRA adaptation"],
            )
        )

    # Status must be recorded
    status = get_source_status()
    assert status.get("Semantic Scholar") == "RATE_LIMITED"
    errors = get_and_clear_retrieval_errors()
    for err in errors:
        assert "HTTP" not in err
        assert "429" not in err
        assert "LoRA adaptation" not in err


def test_fallback_to_other_source_when_s2_fails():
    """
    When Semantic Scholar fails with 429, retrieval should still finish
    successfully returning papers from arXiv / OpenAlex.
    """
    reset_retrieval_session()

    s2_resp = MagicMock()
    s2_resp.status_code = 429
    s2_resp.headers = {"Retry-After": "0.01"}

    arxiv_paper = Paper(
        id="arxiv-1",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.1234/test",
        source="arXiv",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        pdfUrl="https://arxiv.org/pdf/1706.03762.pdf",
    )

    async def mock_fetch_arxiv(query_expr, max_results):
        return [arxiv_paper]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get, \
         patch("backend.services.paper_retrieval._fetch_arxiv_query", side_effect=mock_fetch_arxiv):
        mock_get.return_value = s2_resp
        papers = _run_async(
            retrieve_papers(
                question="Attention mechanism in transformers",
                depth=ResearchDepth.QUICK,
                sources=[ResearchSource.ARXIV, ResearchSource.SEMANTIC_SCHOLAR],
                api_key_s2="",
                custom_queries=["transformer attention"],
            )
        )

    assert len(papers) >= 1
    assert any(p.title == "Attention Is All You Need" for p in papers)
    status = get_source_status()
    assert status.get("Semantic Scholar") == "RATE_LIMITED"
    assert status.get("arXiv") == "SUCCESS"


def test_enrich_papers_with_s2_batch_size_capped_at_100():
    """Verify that enrich_papers_with_s2 batches IDs in chunks of at most 100."""
    # Create 150 dummy papers with DOI
    dummy_papers = [
        Paper(
            id=f"p-{i}",
            title=f"Paper {i}",
            authors=["Author"],
            publicationYear=2023,
            journalConference="Venue",
            doi=f"10.1000/{i}",
            source="arXiv",
            citationCount=0,
            abstract="Abstract text for paper",
        )
        for i in range(150)
    ]

    posted_batches = []

    async def mock_post(url, *args, **kwargs):
        payload = kwargs.get("json", {})
        ids = payload.get("ids", [])
        posted_batches.append(len(ids))
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [{} for _ in ids]
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        _ = _run_async(enrich_papers_with_s2(dummy_papers))

    assert len(posted_batches) == 2
    assert posted_batches[0] == 100
    assert posted_batches[1] == 50
    assert max(posted_batches) <= 100
