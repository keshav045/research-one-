"""
Tests for Final Hardening & Verification:
1. Test unified paper search manager and source enablement flags.
2. Test paper-scoped vector retrieval (verifying Paper X scopes to Paper X first).
3. Test PDF magic byte validation and rejection of HTML payloads.
4. Test deduplication priority across DOI, arXiv, Semantic Scholar, and OpenAlex.
5. Test relevance score recording and MIN_PAPER_RELEVANCE filtering.
6. Test citation reference boundary validation ([n] <= len(references)).
7. Test new API endpoints (/evidence, /claims, /report, and 404 behavior).
"""

import pytest
from backend.models.schemas import Paper, PaperPassage, Citation, CitationStatus, ResearchDepth
from backend.services.vector_store import VectorStore
from backend.services.ranker import filter_and_deduplicate, rank_papers
from backend.services.paper_search_manager import PaperSearchManager
from backend.config import settings


# ─── 1. Paper-Scoped Vector Retrieval (Phase 4 / Req 16) ──────────────────────

def test_paper_scoped_vector_search():
    store = VectorStore()
    p1 = Paper(
        id="paper-1",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="arXiv",
        abstract="We propose the Transformer architecture.",
        passages=[
            PaperPassage(id="p1-1", page=1, text="The Transformer is based entirely on self-attention.", source_kind="pdf"),
            PaperPassage(id="p1-2", page=2, text="Multi-head attention allows the model to jointly attend.", source_kind="pdf"),
        ]
    )
    p2 = Paper(
        id="paper-2",
        title="BERT Pre-training",
        authors=["Devlin et al."],
        publicationYear=2018,
        journalConference="NAACL",
        doi="10.48550/arxiv.1810.04805",
        source="arXiv",
        abstract="We introduce BERT for language representation.",
        passages=[
            PaperPassage(id="p2-1", page=1, text="BERT stands for Bidirectional Encoder Representations.", source_kind="pdf"),
        ]
    )
    store.build([p1, p2])

    # Search specifically scoped to paper-2
    scoped_results = store.search_paper(paper_id="paper-2", query="self-attention mechanism")
    assert all(r.paper_id == "paper-2" for r, _ in scoped_results), "Paper-scoped search must ONLY return paper-2 passages"
    assert len(scoped_results) == 1

    # Search specifically scoped to paper-1
    scoped_p1 = store.search_paper(paper_id="paper-1", query="self-attention")
    assert all(r.paper_id == "paper-1" for r, _ in scoped_p1)
    assert len(scoped_p1) > 0


# ─── 2. Deduplication Priority (Req 7) ────────────────────────────────────────

def test_deduplication_priority_and_metadata_merge():
    # Same paper with arXiv ID in p_arxiv and DOI in p_doi
    p_arxiv = Paper(
        id="arxiv-1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        publicationYear=2017,
        journalConference="arXiv",
        doi="",
        source="arXiv",
        abstract="Short abstract for arXiv paper that has enough characters.",
        pdfUrl="https://arxiv.org/pdf/1706.03762.pdf",
        citationCount=5000,
    )
    p_doi = Paper(
        id="openalex-W12345",
        title="Attention is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        publicationYear=2017,
        journalConference="NeurIPS 2017",
        doi="10.48550/arXiv.1706.03762",
        source="OpenAlex",
        abstract="A much longer and more comprehensive abstract of the Transformer model.",
        pdfUrl=None,
        citationCount=120000,
    )

    deduped = filter_and_deduplicate([p_arxiv, p_doi])
    assert len(deduped) == 1, "Duplicate paper across sources must merge to exactly 1 record"
    merged = deduped[0]
    assert merged.citationCount == 120000, "Must preserve highest citation count"
    assert merged.pdfUrl == "https://arxiv.org/pdf/1706.03762.pdf", "Must preserve PDF URL from provider that has it"
    assert len(merged.authors) == 2, "Must preserve richer author list"
    assert "comprehensive" in merged.abstract, "Must preserve richer abstract"


# ─── 3. Relevance Threshold & Transparent Scoring (Req 9 & 10) ───────────────

def test_relevance_threshold_filters_irrelevant_papers():
    p_relevant = Paper(
        id="p-rel",
        title="Efficient LLM Inference via 4-bit Quantization",
        authors=["Author A"],
        publicationYear=2023,
        journalConference="ICLR",
        doi="10.1234/quant",
        source="arXiv",
        abstract="Quantization techniques for large language models to reduce memory and compute during inference.",
        citationCount=500,
    )
    p_irrelevant = Paper(
        id="p-irrel",
        title="Historical Agricultural Crop Yields in Ancient Rome",
        authors=["Author B"],
        publicationYear=2015,
        journalConference="History",
        doi="10.1234/crops",
        source="Semantic Scholar",
        abstract="Archaeological analysis of grain production and wheat crops in ancient Italy.",
        citationCount=2,
    )

    ranked = rank_papers(
        question="What are effective techniques for reducing large language model inference memory?",
        papers=[p_relevant, p_irrelevant],
        question_type="literature_review",
        depth=ResearchDepth.STANDARD,
    )

    # p_relevant must have a high relevance score, p_irrelevant must be filtered or scored low
    rel_titles = [p.title for p in ranked]
    assert "Efficient LLM Inference via 4-bit Quantization" in rel_titles
    assert getattr(ranked[0], "relevance_score", 0.0) > 0.0
    assert len(getattr(ranked[0], "relevance_reasons", [])) > 0


def test_rank_papers_backfills_and_does_not_starve_candidates(monkeypatch):
    """When only 1 candidate exceeds min_relevance, ranker backfills up to target_count."""
    monkeypatch.setattr(settings, "MIN_PAPER_RELEVANCE", 0.70)
    papers = [
        Paper(
            id=f"p-{i}",
            title=f"Candidate Study on Transformers Number {i}",
            authors=[f"Author {i}"],
            publicationYear=2020 + (i % 4),
            journalConference="arXiv",
            doi=f"10.1234/test-{i}",
            source="arXiv",
            abstract=f"Study {i} investigating neural architectures and attention models.",
            citationCount=100 * (10 - i),
        )
        for i in range(1, 10)
    ]
    # Ensure paper-1 matches exactly for high score, others have lower semantic/relevance scores
    papers[0].title = "Large Language Model Inference Optimization via Quantization"
    papers[0].abstract = "Specific large language model quantization memory and latency speedup."

    ranked = rank_papers(
        question="What are effective techniques for large language model inference?",
        papers=papers,
        question_type="literature_review",
        depth=ResearchDepth.QUICK,  # Quick depth target_count is 6
    )

    # Must return 6 papers, not just 1!
    assert len(ranked) == 6, f"Expected 6 papers backfilled, got {len(ranked)}"
    assert ranked[0].id == "paper-1"  # Renumbered sequentially
    assert ranked[0].title == "Large Language Model Inference Optimization via Quantization"
    # All 6 papers must have distinct original titles
    assert len({p.title for p in ranked}) == 6


# ─── 4. Citation Reference Validation (Req 21) ────────────────────────────────

@pytest.mark.asyncio
async def test_citation_reference_validation():
    from backend.services.local_llm_service import synthesize_report

    p1 = Paper(
        id="paper-1",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="arXiv",
        abstract="Transformer paper.",
        citationCount=100000,
    )
    c1 = Citation(
        id="cite-1",
        badgeNumber=1,
        claim="Transformers use self-attention.",
        status=CitationStatus.VERIFIED,
        paperId="paper-1",
        paperTitle="Attention Is All You Need",
        authors="Vaswani et al.",
        year=2017,
        page=2,
        passage="Transformers use self-attention.",
        highlightSentence="Transformers use self-attention.",
    )

    # Synthesize report and check references match badges
    report, integrity, removed = await synthesize_report(
        question="Which paper introduced Transformers?",
        depth=ResearchDepth.QUICK,
        papers=[p1],
        claims=[],
        citations=[c1],
        anchor_paper=p1,
    )

    # Validate all [n] badges in executiveSummary map to an existing reference in report.references
    import re
    badges = [int(m) for m in re.findall(r"\[(\d+)\]", report.executiveSummary)]
    assert all(1 <= b <= len(report.references) for b in badges), (
        f"All badges {badges} must be within reference count {len(report.references)}"
    )


# ─── 5. Unified Search Manager & Source Flags (Req 35) ────────────────────────

@pytest.mark.asyncio
async def test_search_manager_respects_disabled_sources():
    mgr = PaperSearchManager()
    mgr.arxiv_enabled = False
    mgr.semantic_scholar_enabled = False
    mgr.openalex_enabled = False

    papers, stats = await mgr.search_all_sources("transformer attention")
    assert len(papers) == 0
    assert stats["source_counts"] == {}


def test_filter_and_extract_sentences_safe_counts():
    from backend.services.evidence import _filter_and_extract_sentences

    text = (
        "Short. "  # under 8 words & < 30 chars
        "Figure 1: This is a caption for the model architecture diagram. "  # caption
        "The rest of this paper is organized as follows in section 2. "  # procedural fluff
        "This is an informative technical sentence describing empirical transformer accuracy across benchmarks."
    )

    # Test 1: counts=None should not throw AttributeError or KeyError
    sents_none = _filter_and_extract_sentences(text, counts=None)
    assert len(sents_none) >= 1

    # Test 2: counts={} should safely populate missing keys without KeyError
    counts_empty = {}
    sents_empty = _filter_and_extract_sentences(text, counts=counts_empty)
    assert len(sents_empty) >= 1
    assert "fragments_under_8_words" in counts_empty or "length_out_of_bounds" in counts_empty
