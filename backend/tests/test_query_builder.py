"""
Tests for the query builder — Step 1 of the ResearchLens bug-fix plan.
=======================================================================
Covers:
  - settings is importable from paper_retrieval (import guard)
  - _simple_keyword_query never truncates the question
  - _simple_keyword_query strips stop-words and hardcoded filler terms
  - generate_search_queries (LLM path) produces plausible queries for the
    Transformer test question — asserted against the keyword fallback when
    Gemini is not configured, and a light integration smoke-test when it is.
  - retrieve_papers fans out queries and returns Vaswani et al. 2017 from
    arXiv (live network call; skipped when ARXIV_LIVE env var is not set).

Run from the project root:
    python -m pytest backend/tests/test_query_builder.py -v

To include the live arXiv integration test:
    ARXIV_LIVE=1 python -m pytest backend/tests/test_query_builder.py -v -k live
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make "backend" importable without installing the package
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Stub heavy optional dependencies that are not needed for unit tests
for _stub_name in [
    "faiss",
    "sentence_transformers",
    "google",
    "google.generativeai",
]:
    sys.modules.setdefault(_stub_name, types.ModuleType(_stub_name))

# Stub config so tests work without a real .env file
_cfg_mod = types.ModuleType("backend.config")


class _Settings:
    GEMINI_API_KEY: str = ""           # blank => not configured
    GEMINI_MODEL: str = "gemini-test"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    PDF_CACHE_DIR: str = "/tmp/pdf_cache"
    DATABASE_URL: str = "sqlite:///./test.db"
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    MAX_PDF_WORKERS: int = 2
    DEPTH_COUNTS: dict = {"Quick": 6, "Standard": 12, "Deep": 24}

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY) and self.GEMINI_API_KEY != "your_gemini_api_key_here"


_cfg_mod.settings = _Settings()  # type: ignore[attr-defined]
sys.modules["backend.config"] = _cfg_mod


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

# Now safe to import the module under test
from backend.services.paper_retrieval import (  # noqa: E402
    _simple_keyword_query,
    generate_search_queries,
    retrieve_papers,
)
from backend.models.schemas import ResearchDepth, ResearchSource  # noqa: E402

# ---------------------------------------------------------------------------
# Test question shared across tests
# ---------------------------------------------------------------------------

TRANSFORMER_QUESTION = (
    "Which paper introduced the Transformer architecture, and what was its key idea?"
)

# ---------------------------------------------------------------------------
# Guard: settings must be importable from paper_retrieval
# ---------------------------------------------------------------------------


class TestSettingsImportGuard:
    """
    The single most important structural test for Step 1:
    `settings` must be a live attribute inside paper_retrieval, not NameError.
    Before the fix, paper_retrieval.py never imported `settings`, so every
    call to generate_search_queries() raised NameError at runtime.
    """

    def test_settings_accessible_in_paper_retrieval_module(self):
        """paper_retrieval must have access to settings without NameError."""
        try:
            result = _run_async(
                generate_search_queries(TRANSFORMER_QUESTION)
            )
            # Should return a list (keyword fallback) without blowing up
            assert isinstance(result, list)
            assert len(result) >= 1
        except NameError as exc:
            pytest.fail(
                f"NameError raised -- settings is still not imported: {exc}"
            )


# ---------------------------------------------------------------------------
# _simple_keyword_query -- no truncation, no hardcoded filler
# ---------------------------------------------------------------------------


class TestSimpleKeywordQuery:

    def test_question_is_never_truncated(self):
        """
        The full question must contribute tokens. Previously a character limit
        cut questions like "Which paper introduced the Transformer architecture"
        to "Which paper introduced the Tra".
        """
        result = _simple_keyword_query(TRANSFORMER_QUESTION)
        # Key discriminating tokens that would be lost by mid-word truncation
        assert "transformer" in result.lower(), (
            f"'transformer' missing from query: {result!r}"
        )

    def test_no_hardcoded_filler_terms(self):
        """
        The output must NOT contain 'empirical', 'evaluation', or 'benchmark'
        unless those words actually appear in the question.
        """
        result = _simple_keyword_query(TRANSFORMER_QUESTION)
        for forbidden in ("empirical", "evaluation", "benchmark", "survey"):
            assert forbidden not in result.lower(), (
                f"Hardcoded filler term {forbidden!r} found in query: {result!r}"
            )

    def test_stop_words_removed(self):
        """Common stop-words must not appear as standalone tokens."""
        result = _simple_keyword_query(TRANSFORMER_QUESTION)
        tokens = set(result.lower().split())
        for stop in ("which", "the", "and", "was", "its"):
            assert stop not in tokens, (
                f"Stop-word {stop!r} leaked into query: {result!r}"
            )

    def test_returns_non_empty_string(self):
        assert _simple_keyword_query(TRANSFORMER_QUESTION).strip() != ""

    def test_short_question_returns_meaningful_tokens(self):
        result = _simple_keyword_query("What is BERT?")
        assert "bert" in result.lower()

    def test_very_long_question_not_truncated_mid_word(self):
        """
        The fallback tokeniser must not cut a word in half.
        Every token in the output must be a whole word (no partial strings).
        """
        import re

        long_q = (
            "Which paper introduced the Transformer architecture and replaced "
            "recurrence with self-attention mechanisms enabling parallel training?"
        )
        result = _simple_keyword_query(long_q)
        # Each token should be a complete word (alphanumeric + hyphens)
        tokens = result.split()
        for tok in tokens:
            assert re.fullmatch(r"[a-z0-9][-a-z0-9]*", tok), (
                f"Token looks truncated or malformed: {tok!r} in {result!r}"
            )


# ---------------------------------------------------------------------------
# generate_search_queries -- keyword fallback (no Gemini configured)
# ---------------------------------------------------------------------------


class TestGenerateSearchQueriesFallback:
    """
    When Gemini is not configured, generate_search_queries must fall back
    to _simple_keyword_query and still return useful queries for the
    Transformer question.
    """

    def test_returns_list_of_strings(self):
        result = _run_async(
            generate_search_queries(TRANSFORMER_QUESTION)
        )
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_fallback_contains_transformer_token(self):
        result = _run_async(
            generate_search_queries(TRANSFORMER_QUESTION)
        )
        combined = " ".join(result).lower()
        assert "transformer" in combined, (
            f"'transformer' missing from fallback queries: {result}"
        )

    def test_no_empty_queries(self):
        result = _run_async(
            generate_search_queries(TRANSFORMER_QUESTION)
        )
        assert all(q.strip() for q in result), (
            f"One or more empty queries returned: {result}"
        )


# ---------------------------------------------------------------------------
# generate_search_queries -- LLM path (Gemini mocked)
# ---------------------------------------------------------------------------


class TestGenerateSearchQueriesLLMPath:
    """
    Verify the LLM branch: when Gemini IS configured and returns a valid JSON
    array, generate_search_queries must return those queries (trimmed to <=3).
    """

    def _run_with_mock_gemini(self, mock_json_response: str) -> list[str]:
        """Helper: patch settings + genai, run generate_search_queries."""
        mock_response = MagicMock()
        mock_response.text = mock_json_response

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.GenerationConfig.return_value = MagicMock()
        mock_genai.configure.return_value = None

        old_google_genai = getattr(sys.modules.get("google"), "generativeai", None)
        if "google" in sys.modules:
            sys.modules["google"].generativeai = mock_genai

        try:
            with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
                result = _run_async(
                    generate_search_queries(TRANSFORMER_QUESTION)
                )
        finally:
            _cfg_mod.settings.GEMINI_API_KEY = ""  # type: ignore
            if "google" in sys.modules and old_google_genai is not None:
                sys.modules["google"].generativeai = old_google_genai

        return result

    def test_llm_queries_used_when_valid_json(self):
        queries = self._run_with_mock_gemini(
            '["Attention Is All You Need Vaswani", '
            '"transformer self-attention", '
            '"sequence model without recurrence"]'
        )
        assert "Attention Is All You Need Vaswani" in queries

    def test_at_most_3_queries_returned(self):
        queries = self._run_with_mock_gemini('["q1", "q2", "q3", "q4", "q5"]')
        assert len(queries) <= 3

    def test_no_empty_strings_in_llm_output(self):
        queries = self._run_with_mock_gemini(
            '["  ", "transformer self-attention", ""]'
        )
        assert all(q.strip() for q in queries)

    def test_falls_back_on_invalid_json(self):
        """If the LLM returns garbage JSON the fallback keyword query is used."""
        _cfg_mod.settings.GEMINI_API_KEY = "fake-key-for-test"  # type: ignore

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = ValueError("JSON parse error")

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.GenerationConfig.return_value = MagicMock()
        mock_genai.configure.return_value = None

        try:
            with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
                result = _run_async(
                    generate_search_queries(TRANSFORMER_QUESTION)
                )
        finally:
            _cfg_mod.settings.GEMINI_API_KEY = ""  # type: ignore

        # Must still return at least one usable query (keyword fallback)
        assert isinstance(result, list) and len(result) >= 1
        assert all(q.strip() for q in result)


# ---------------------------------------------------------------------------
# Live arXiv integration test -- only runs when ARXIV_LIVE=1
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ARXIV_LIVE") != "1",
    reason="Set ARXIV_LIVE=1 to run live arXiv integration tests",
)
class TestLiveArxivRetrievesVaswani:
    """
    Integration smoke-test: retrieve_papers on the Transformer question must
    include the Vaswani et al. 2017 'Attention Is All You Need' paper in the
    top results from arXiv.

    Requirements:
      - Network access to export.arxiv.org
      - ARXIV_LIVE=1 environment variable

    This test does NOT call the Gemini API; it uses the keyword fallback
    (Gemini key left blank in the stub settings).
    """

    def test_vaswani_2017_in_results(self):
        papers = _run_async(
            retrieve_papers(
                TRANSFORMER_QUESTION,
                depth=ResearchDepth.STANDARD,
                sources=[ResearchSource.ARXIV],
                api_key_s2="",
            )
        )

        assert papers, "retrieve_papers returned an empty list -- check network"

        titles_lower = [p.title.lower() for p in papers]

        # Primary assertion: the canonical paper must appear
        vaswani_found = any(
            "attention is all you need" in t for t in titles_lower
        )
        assert vaswani_found, (
            f"'Attention Is All You Need' not found in results.\n"
            f"Got titles: {[p.title for p in papers]}"
        )

        # Secondary assertion: it must be attributed to 2017
        vaswani_paper = next(
            p for p in papers if "attention is all you need" in p.title.lower()
        )
        assert vaswani_paper.publicationYear == 2017, (
            f"Expected year 2017, got {vaswani_paper.publicationYear}"
        )

    def test_no_hardcoded_terms_in_queries_sent_to_arxiv(self):
        """
        Re-run retrieval but intercept the arXiv HTTP call to verify that
        the search_query parameter does NOT contain 'empirical evaluation'
        or 'benchmark'.
        """
        import httpx

        captured_urls: list[str] = []

        original_get = httpx.AsyncClient.get

        async def _spy_get(self_inner, url, **kwargs):
            captured_urls.append(str(url))
            return await original_get(self_inner, url, **kwargs)

        with patch.object(httpx.AsyncClient, "get", _spy_get):
            _run_async(
                retrieve_papers(
                    TRANSFORMER_QUESTION,
                    depth=ResearchDepth.QUICK,
                    sources=[ResearchSource.ARXIV],
                    api_key_s2="",
                )
            )

        assert captured_urls, "No HTTP calls were captured"

        for url in captured_urls:
            url_lower = url.lower()
            assert "empirical+evaluation" not in url_lower and \
                   "empirical%20evaluation" not in url_lower and \
                   "empirical evaluation" not in url_lower, (
                f"Hardcoded 'empirical evaluation' found in arXiv URL: {url}"
            )
            assert "benchmark" not in url_lower, (
                f"Hardcoded 'benchmark' found in arXiv URL: {url}"
            )
