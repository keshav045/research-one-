"""
Phase 8 Unit Tests -- ResearchLens Implementation Plan
======================================================
- NLI: heuristic fallback NEVER returns ENTAILS
- Ranker: withdrawn papers dropped; exact-title paper ranks first
- Chunker: page numbers kept; references section removed
- Answer post-check: sentences without [n] are flagged

Run from project root:
    python -m pytest backend/tests/test_phase8.py -v
"""

from __future__ import annotations

import re
import sys
import types
import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make "backend" importable without installing the package
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Only stub optional deps that may not be installed (never stub torch/transformers)
for _stub_name in ["faiss", "sentence_transformers", "google", "google.generativeai"]:
    if _stub_name not in sys.modules:
        try:
            importlib.import_module(_stub_name)
        except ImportError:
            sys.modules[_stub_name] = types.ModuleType(_stub_name)

# Stub backend.config BEFORE importing any backend module
_cfg_mod = types.ModuleType("backend.config")


class _FakeSettings:
    LLM_PROVIDER = "local"
    LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    LOCAL_LLM_MAX_TOKENS = 512
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5:3b-instruct"
    QWEN_API_KEY = ""
    GEMINI_API_KEY = ""
    SEMANTIC_SCHOLAR_API_KEY = ""
    NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    NLI_DEVICE = "cpu"
    NLI_ENTAIL_THRESHOLD = 0.80
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEPTH_COUNTS = {"Quick": 6, "Standard": 12, "Deep": 24}
    DATABASE_URL = "sqlite:///./researchlens_test.db"
    PDF_CACHE_DIR = "./pdf_cache"
    MAX_PDF_WORKERS = 4
    is_gemini_configured = False
    is_qwen_configured = False


_cfg_mod.settings = _FakeSettings()
_cfg_mod.get_settings = lambda: _FakeSettings()
sys.modules["backend.config"] = _cfg_mod


# ---------------------------------------------------------------------------
# Test 1: NLI Fallback -- NEVER returns ENTAILS
# ---------------------------------------------------------------------------


class TestNliFallback:
    """Heuristic fallback must return NEUTRAL by default, never ENTAILS."""

    def test_heuristic_never_returns_entails_on_generic(self):
        from backend.services.nli_verifier import _heuristic_entailment
        from backend.models.schemas import EntailmentVerdict

        verdict, conf, reason = _heuristic_entailment(
            "The sky is blue.",
            "Neural networks achieve 95% accuracy.",
            "",
        )
        assert verdict != EntailmentVerdict.ENTAILS, (
            f"Heuristic must not return ENTAILS for unrelated pairs, got {verdict}"
        )

    def test_heuristic_neutral_on_empty_inputs(self):
        from backend.services.nli_verifier import _heuristic_entailment
        from backend.models.schemas import EntailmentVerdict

        verdict, conf, reason = _heuristic_entailment("", "Some claim.", "")
        assert verdict == EntailmentVerdict.NEUTRAL

    def test_heuristic_contradicts_on_directional_conflict(self):
        from backend.services.nli_verifier import _heuristic_entailment
        from backend.models.schemas import EntailmentVerdict

        # Use words that exactly match the heuristic patterns:
        # is_claim_pos matches 'improves'; is_prem_neg matches 'diminish'
        verdict, conf, reason = _heuristic_entailment(
            "Results diminish as the resolution increases.",
            "The model improves accuracy at higher resolutions.",
            "Results diminish as the resolution increases.",
        )
        assert verdict == EntailmentVerdict.CONTRADICTS, (
            f"Expected CONTRADICTS for directional conflict, got {verdict}. Reason: {reason}"
        )

    def test_heuristic_neutral_on_number_mismatch(self):
        from backend.services.nli_verifier import _heuristic_entailment
        from backend.models.schemas import EntailmentVerdict

        verdict, conf, reason = _heuristic_entailment(
            "The model was evaluated on a benchmark.",
            "The model achieves 98.5% accuracy with 3.2x speedup.",
            "",
        )
        assert verdict == EntailmentVerdict.NEUTRAL


# ---------------------------------------------------------------------------
# Test 2: Ranker -- withdrawn papers dropped, exact title ranks first
# ---------------------------------------------------------------------------


class TestRanker:
    def _make_paper(self, pid, title, year=2023, cites=10):
        from backend.models.schemas import Paper

        return Paper(
            id=pid,
            title=title,
            authors=["Author A"],
            publicationYear=year,
            journalConference="Conf",
            doi=f"10.0/{pid}",
            source="arXiv",
            abstract="This paper presents an important approach to the problem and evaluates it on standard benchmarks.",
        )

    def test_withdrawn_papers_dropped(self):
        """filter_and_deduplicate must exclude [WITHDRAWN] papers."""
        from backend.services.ranker import filter_and_deduplicate

        papers = [
            self._make_paper("p1", "Attention Is All You Need", 2017, 1000),
            self._make_paper("p2", "[WITHDRAWN] Bad Paper", 2022, 5),
            self._make_paper("p3", "BERT Pre-training", 2019, 800),
        ]
        filtered = filter_and_deduplicate(papers)
        assert all("withdrawn" not in p.title.lower() for p in filtered), (
            "Withdrawn paper must be excluded"
        )
        assert len(filtered) == 2, f"Expected 2 papers after filtering, got {len(filtered)}"

    def test_exact_title_ranks_first(self):
        """rank_papers with factual_lookup and title_guesses should rank exact-title paper first."""
        from backend.services.ranker import rank_papers, filter_and_deduplicate
        from backend.models.schemas import ResearchDepth

        papers = [
            self._make_paper("p1", "Unrelated Vision Paper", 2021, 5),
            self._make_paper("p2", "Attention Is All You Need", 2017, 1000),
            self._make_paper("p3", "GAN Paper About Image Synthesis", 2020, 50),
        ]
        # rank_papers calls filter_and_deduplicate internally
        ranked = rank_papers(
            question="Which paper introduced the Transformer architecture?",
            papers=papers,
            question_type="factual_lookup",
            title_guesses=["Attention Is All You Need"],
            depth=ResearchDepth.STANDARD,
        )
        assert len(ranked) > 0, "Ranked list must not be empty"
        # The factual_lookup branch adds +1.0 to score for 'attention is all you need'
        assert "attention" in ranked[0].title.lower(), (
            f"Expected 'Attention Is All You Need' to rank first, got: {ranked[0].title}"
        )


# ---------------------------------------------------------------------------
# Test 3: Chunker -- page numbers kept, references excluded
# ---------------------------------------------------------------------------


class TestPdfChunker:
    def test_passages_have_valid_page_numbers(self):
        try:
            import fitz

            if not hasattr(fitz, "open"):
                pytest.skip("fitz stubbed")
        except Exception:
            pytest.skip("fitz not available")
        from backend.services.pdf_extractor import _extract_passages_from_pdf_bytes

        doc = fitz.open()
        for i in range(2):
            page = doc.new_page()
            page.insert_text(
                (50, 50),
                f"Page {i+1}. This sentence discusses self-attention mechanisms. "
                "Transformers replace recurrent networks with attention. Multi-head attention enables parallelism.",
            )
        pdf_bytes = doc.tobytes()
        doc.close()
        passages = _extract_passages_from_pdf_bytes("test", pdf_bytes)
        assert len(passages) > 0, "Should extract at least 1 passage"
        for p in passages:
            assert p.page >= 1, f"Page must be >= 1, got {p.page}"

    def test_references_section_excluded(self):
        try:
            import fitz

            if not hasattr(fitz, "open"):
                pytest.skip("fitz stubbed")
        except Exception:
            pytest.skip("fitz not available")
        from backend.services.pdf_extractor import _extract_passages_from_pdf_bytes

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "Introduction\nThis paper presents self-attention mechanisms for NLP.\n\n"
            "References\n[1] Vaswani et al. 2017. Attention Is All You Need.\n"
            "[2] LeCun et al. 1989. Backpropagation applied to recognition.\n",
        )
        pdf_bytes = doc.tobytes()
        doc.close()
        passages = _extract_passages_from_pdf_bytes("test", pdf_bytes)
        for p in passages:
            assert "[1] Vaswani" not in p.text, "References text must be excluded"
            assert "[2] LeCun" not in p.text, "References text must be excluded"


# ---------------------------------------------------------------------------
# Test 4: Answer post-check -- uncited sentences detected and flagged
# ---------------------------------------------------------------------------


class TestAnswerPostCheck:
    CITATION_RE = re.compile(r"\[\d+\]")

    def _uncited_sentences(self, text: str) -> list:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        return [s for s in sentences if s.strip() and not self.CITATION_RE.search(s)]

    def test_uncited_sentence_detected(self):
        text = (
            "The Transformer introduced self-attention [1]. "
            "This was a major breakthrough with no citation. "
            "Multi-head attention was proposed by Vaswani et al. [2]."
        )
        bad = self._uncited_sentences(text)
        assert len(bad) >= 1, f"Expected at least 1 uncited sentence, got none"

    def test_fully_cited_answer_passes(self):
        text = (
            "The Transformer was introduced in 2017 [1]. "
            "It replaced recurrent networks with self-attention [2]. "
            "Multi-head attention enables parallel processing [3]."
        )
        bad = self._uncited_sentences(text)
        assert len(bad) == 0, f"All sentences cited, expected 0 bad: {bad}"


# ---------------------------------------------------------------------------
# Test 5: Semantic Scholar Retry -- 429 backoff & error logging
# ---------------------------------------------------------------------------


class TestSemanticScholarRetry:
    @pytest.mark.asyncio
    async def test_retry_on_429_then_success(self, monkeypatch):
        """HTTP 429 on first try, followed by 200 on second try -> succeeds."""
        import asyncio
        import httpx
        from backend.services.paper_retrieval import _fetch_s2_with_retry

        async def fake_sleep(sec):
            pass

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        calls = 0

        async def mock_get(self, url, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "paperId": "p123456789012",
                            "title": "Attention Is All You Need",
                            "authors": [{"name": "Vaswani"}],
                            "year": 2017,
                        }
                    ]
                },
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        fast_s2 = _fetch_s2_with_retry.retry_with(wait=lambda *args, **kwargs: 0)
        papers = await fast_s2("transformer", 5, {})
        assert len(papers) == 1
        assert "Attention Is All You Need" in papers[0].title
        assert calls == 2

    @pytest.mark.asyncio
    async def test_retry_exhaustion_records_error(self, monkeypatch):
        """HTTP 429 on all 4 attempts -> records retrieval error."""
        import asyncio
        import httpx
        from backend.services.paper_retrieval import (
            _fetch_semantic_scholar_query,
            get_and_clear_retrieval_errors,
        )
        from backend.services import paper_retrieval

        get_and_clear_retrieval_errors()

        async def fake_sleep(sec):
            pass

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def mock_get_429(self, url, **kwargs):
            return httpx.Response(429, headers={"Retry-After": "0"})

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get_429)
        monkeypatch.setattr(
            paper_retrieval,
            "_get_cached_response",
            lambda *args: None,
        )

        fast_s2 = paper_retrieval._fetch_s2_with_retry.retry_with(wait=lambda *args, **kwargs: 0)
        monkeypatch.setattr(paper_retrieval, "_fetch_s2_with_retry", fast_s2)

        results = await _fetch_semantic_scholar_query("never_cached_query_xyz", 5)
        assert results == []

        errors = get_and_clear_retrieval_errors()
        assert len(errors) > 0, "Expected at least one error recorded on retry exhaustion"
        assert any("Semantic Scholar" in err or "429" in err for err in errors)
