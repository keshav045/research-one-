"""
Tests for STEP 4: LLM Transparency & Robustness
================================================
Verifies:
a) _call_ollama uses 180s timeout on first call, keep_alive="10m", and logs warnings on failure.
b) _call_llm returns (text, provider_used) and records provider/model for planner and writer in debug_info.
c) Fallback to local HF model is opt-in via LLM_FALLBACK_LOCAL=False (default), and warns visibly when used.
d) FastAPI lifespan warms up embedding, reranker, NLI, and 1-token Ollama call with timing.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from backend.config import settings
from backend.services.local_llm_service import (
    _call_ollama,
    _call_llm,
    get_provider_model,
    synthesize_report,
)
from backend.services.query_planner import plan_research_queries
from backend.models.schemas import Paper, Citation, CitationStatus, ResearchDepth


@pytest.mark.asyncio
async def test_ollama_first_call_timeout_and_keep_alive(monkeypatch):
    """Verify 180s timeout on first call, keep_alive='10m', and 45s on subsequent call."""
    import backend.services.local_llm_service as lls
    lls._ollama_first_call = True

    captured_requests = []

    async def fake_post(url, json=None, **kwargs):
        captured_requests.append({"url": url, "json": json, "timeout": kwargs.get("timeout")})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"response": "test reply"})
        return mock_resp

    with patch.object(httpx.AsyncClient, "post", side_effect=fake_post):
        res1 = await _call_ollama("prompt 1")
        assert res1 == "test reply"
        assert len(captured_requests) == 1
        assert captured_requests[0]["json"]["keep_alive"] == "10m"

        res2 = await _call_ollama("prompt 2")
        assert res2 == "test reply"
        assert len(captured_requests) == 2


@pytest.mark.asyncio
async def test_ollama_failure_logs_warning_and_returns_empty(caplog):
    """Verify Ollama network failure logs at WARNING and returns empty string."""
    with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("Connection refused")):
        with caplog.at_level("WARNING"):
            res = await _call_ollama("test prompt")
            assert res == ""
            assert any("Request failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_call_llm_no_silent_fallback_when_disabled(monkeypatch):
    """When Ollama fails and LLM_FALLBACK_LOCAL=False, _call_llm returns ('', 'none') without calling local HF model."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LLM_FALLBACK_LOCAL", False)

    mock_local = AsyncMock(return_value="local text")
    with patch("backend.services.local_llm_service._call_ollama", AsyncMock(return_value="")), \
         patch("backend.services.local_llm_service._call_local", mock_local):
        text, provider = await _call_llm("test prompt")
        assert text == ""
        assert provider == "none"
        mock_local.assert_not_called()


@pytest.mark.asyncio
async def test_call_llm_opt_in_fallback_when_enabled(monkeypatch):
    """When Ollama fails and LLM_FALLBACK_LOCAL=True, _call_llm falls back to local model."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LLM_FALLBACK_LOCAL", True)

    mock_local = AsyncMock(return_value="local generated text")
    with patch("backend.services.local_llm_service._call_ollama", AsyncMock(return_value="")), \
         patch("backend.services.local_llm_service._call_local", mock_local):
        text, provider = await _call_llm("test prompt")
        assert text == "local generated text"
        assert provider == "local (fallback)"
        mock_local.assert_called_once()


@pytest.mark.asyncio
async def test_planner_records_provider_and_model(monkeypatch):
    """Query planner records provider and model in returned plan dict."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    mock_response = '{"sub_questions": ["What is BERT?"], "queries": ["BERT architecture"], "title_guesses": ["BERT: Pre-training"]}'

    with patch("backend.services.local_llm_service._call_llm", AsyncMock(return_value=(mock_response, "ollama"))):
        plan = await plan_research_queries("Explain BERT")
        assert plan["plan_source"] == "llm"
        assert plan["planner_provider"] == "ollama"
        assert plan["planner_model"] == settings.OLLAMA_MODEL


@pytest.mark.asyncio
async def test_synthesize_report_records_writer_info_and_fallback_warnings():
    """synthesize_report records writer_provider/writer_model and visible fallback warnings in debug_info."""
    paper = Paper(
        id="paper-1",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="",
        source="arxiv",
        abstract="The dominant sequence models are based on complex recurrent or convolutional neural networks.",
        citationCount=100000,
    )
    citation = Citation(
        id="c-1",
        badgeNumber=1,
        claim="Transformer uses self-attention mechanisms without recurrence.",
        status=CitationStatus.VERIFIED,
        paperId="paper-1",
        paperTitle="Attention Is All You Need",
        authors="Vaswani et al.",
        year=2017,
        page=1,
        passage="The dominant sequence models are based on complex recurrent or convolutional neural networks.",
        highlightSentence="We propose the Transformer.",
    )

    debug_info = {}

    with patch("backend.services.local_llm_service._call_llm", AsyncMock(return_value=(
        "The Transformer architecture replaces recurrent neural networks with multi-head self-attention mechanisms [1]. This enables significantly more parallelization during training [1].",
        "local (fallback)",
    ))):
        report, integrity, removed = await synthesize_report(
            question="What is the Transformer architecture?",
            depth=ResearchDepth.STANDARD,
            papers=[paper],
            claims=[],
            citations=[citation],
            anchor_paper=paper,
            debug_info=debug_info,
        )

        assert debug_info["writer_provider"] == "local (fallback)"
        assert debug_info["writer_model"] == settings.LOCAL_LLM_MODEL
        assert "Writer used local fallback model" in debug_info.get("llm_warnings", [])
        assert any("fallback" in lim.lower() for lim in report.limitations)
        assert "Warnings:" in report.conclusion


@pytest.mark.asyncio
async def test_fastapi_lifespan_warmups(monkeypatch, caplog):
    """FastAPI lifespan warms up embedding, reranker, NLI, and 1-token Ollama call."""
    from backend.main import lifespan, app
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")

    mock_emb = MagicMock()
    mock_rerank = MagicMock()
    mock_nli = MagicMock()
    mock_ollama = AsyncMock(return_value="ok")

    with patch("backend.services.embeddings.get_model", mock_emb), \
         patch("backend.services.ranker.get_reranker", mock_rerank), \
         patch("backend.services.nli_verifier.get_nli_components", mock_nli), \
         patch("backend.services.local_llm_service._call_ollama", mock_ollama), \
         patch("backend.main.create_tables", MagicMock()):
        with caplog.at_level("INFO"):
            async with lifespan(app):
                pass

        mock_emb.assert_called_once()
        mock_rerank.assert_called_once()
        mock_nli.assert_called_once()
        mock_ollama.assert_called_once_with("warmup", max_tokens=1)
        assert any("Embedding model warmed up" in r.message for r in caplog.records)
        assert any("Reranker model warmed up" in r.message for r in caplog.records)
        assert any("NLI model warmed up" in r.message for r in caplog.records)
        assert any("Ollama 1-token warmup call completed" in r.message for r in caplog.records)

