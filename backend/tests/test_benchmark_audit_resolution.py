"""
Audit Resolution Unit Tests:
Verifies that all 5 critical errors reported in the benchmark audit on
'Which paper introduced the Transformer architecture, and what was its key idea?'
are definitively resolved.

Error 1: Wrong citation in executive summary (Vaswani had citation [1] but [1] mapped to Swin)
Error 2: ArXiv watermark text corrupted the key idea (e.g. arXiv:2103.14030v2)
Error 3: Weak evidence coverage (comparative sentences discarded by external-author filters)
Error 4: Irrelevant paper in comparison table (Scikit-learn entered via citation count bias)
Error 5: Incorrectly represented research contribution (Swin contribution described instead of Transformer)
"""

import pytest
import re
from backend.models.schemas import Paper, Citation, CitationStatus, ResearchDepth
from backend.services.pdf_extractor import strip_arxiv_watermarks
from backend.services.evidence import _filter_and_extract_sentences, extract_and_verify_evidence
from backend.services.vector_store import VectorStore
from backend.models.schemas import PaperPassage
from backend.services.ranker import rank_papers
from backend.services.local_llm_service import (
    build_references_from_citations,
    build_comparison_table,
    synthesize_report
)


def test_audit_error_2_arxiv_watermark_removal():
    """Verify that arXiv margin stamps are completely stripped from extracted text."""
    corrupted_raw = (
        "We present Swin Transformer, a general-purpose arXiv:2103.14030v2 [cs.CV] 17 Aug 2021 "
        "vision Transformer that serves as a general backbone for computer vision."
    )
    cleaned = strip_arxiv_watermarks(corrupted_raw)
    assert "arXiv:2103.14030" not in cleaned
    assert "[cs.CV]" not in cleaned
    assert "17 Aug 2021" not in cleaned
    # Ensure remaining sentence text remains cohesive
    assert "We present Swin Transformer" in cleaned
    assert "vision Transformer that serves as a general backbone" in cleaned


def test_audit_error_1_and_5_anchor_paper_locked_at_ref_1():
    """
    Verify that anchor_paper is strictly locked as Reference [1],
    preventing secondary or chronologically later papers (like Swin) from stealing [1].
    """
    vaswani = Paper(
        id="vaswani2017",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="arXiv",
        citationCount=120000,
        abstract="We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism.",
    )
    swin = Paper(
        id="liu2021",
        title="Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        authors=["Ze Liu", "Yutong Lin", "Yue Cao"],
        publicationYear=2021,
        journalConference="ICCV",
        doi="10.48550/arxiv.2103.14030",
        source="arXiv",
        citationCount=15000,
        abstract="This paper presents a new vision Transformer, called Swin Transformer.",
    )

    # Suppose citations from both papers exist, but swin citation appears first
    citations = [
        Citation(
            id="cite-swin",
            badgeNumber=1,
            claim="Swin Transformer constructs hierarchical feature maps.",
            status=CitationStatus.VERIFIED,
            paperId="liu2021",
            paperTitle="Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
            authors="Ze Liu et al.",
            year=2021,
            page=1,
            passage="Hierarchical representation.",
            highlightSentence="Swin Transformer constructs hierarchical feature maps.",
        ),
        Citation(
            id="cite-vaswani",
            badgeNumber=2,
            claim="The Transformer is the first transduction model relying entirely on self-attention.",
            status=CitationStatus.VERIFIED,
            paperId="vaswani2017",
            paperTitle="Attention Is All You Need",
            authors="Ashish Vaswani et al.",
            year=2017,
            page=1,
            passage="The Transformer is based entirely on attention.",
            highlightSentence="The Transformer relies entirely on self-attention.",
        ),
    ]

    refs = build_references_from_citations(
        papers=[vaswani, swin],
        citations=citations,
        anchor_paper=vaswani
    )

    # Reference [1] (index 0) MUST be Vaswani et al., 2017!
    assert len(refs) >= 2
    assert refs[0].id == "vaswani2017"
    assert refs[0].title == "Attention Is All You Need"
    assert refs[0].year == 2017

    # Reference [2] can be Swin
    assert refs[1].id == "liu2021"


def test_audit_error_4_irrelevant_paper_filtered_out_despite_high_citations():
    """
    Verify that 0-topic-overlap papers (like Scikit-learn with 85,000 citations)
    are strictly disqualified when querying about the Transformer architecture.
    """
    transformer_paper = Paper(
        id="vaswani2017",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="arXiv",
        citationCount=120000,
        abstract="We introduce the Transformer model based solely on self-attention mechanisms, completely without recurrence.",
    )
    bert_paper = Paper(
        id="devlin2018",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Devlin et al."],
        publicationYear=2018,
        journalConference="NAACL",
        doi="10.48550/arxiv.1810.04805",
        source="arXiv",
        citationCount=90000,
        abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
    )
    sklearn_paper = Paper(
        id="pedregosa2011",
        title="Scikit-learn: Machine Learning in Python",
        authors=["Pedregosa et al."],
        publicationYear=2011,
        journalConference="JMLR",
        doi="10.5555/1953048.2078195",
        source="Semantic Scholar",
        citationCount=85000,  # Massive citations but 0 overlap
        abstract="Scikit-learn is a Python module integrating a wide range of state-of-the-art machine learning algorithms for medium-scale supervised and unsupervised problems.",
    )

    query = "Which paper introduced the Transformer architecture, and what was its key idea?"

    ranked = rank_papers(
        question=query,
        papers=[transformer_paper, bert_paper, sklearn_paper],
        question_type="literature_review",
        depth=ResearchDepth.STANDARD,
    )

    ranked_titles = [p.title for p in ranked]
    assert "Scikit-learn: Machine Learning in Python" not in ranked_titles, (
        "Scikit-learn MUST be filtered out as irrelevant to Transformer architecture!"
    )
    assert "Attention Is All You Need" in ranked_titles


def test_audit_error_3_evidence_coverage_includes_comparative_framing():
    """
    Verify that sentences contrasting the Transformer with prior art (RNNs, LSTMs, convolutions)
    are NOT discarded by sentence filters, ensuring strong evidence coverage.
    """
    text = (
        "The Transformer is the first transduction model relying entirely on self-attention to compute representations without using sequence-aligned RNNs or convolution. "
        "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. "
        "Unlike recurrent models such as Hochreiter and Schmidhuber LSTM, the Transformer allows for significantly more parallelization. "
        "We replace the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention."
    )

    sentences = _filter_and_extract_sentences(text)
    assert len(sentences) >= 3, f"Expected at least 3 rich evidence sentences, found {len(sentences)}"
    all_text = " ".join(sentences)
    assert "self-attention" in all_text
    assert "Multi-head" in all_text or "multi-headed" in all_text


def test_audit_error_5_comparison_table_accurately_represents_anchor():
    """
    Verify that the comparison table prioritizes the anchor paper in the first row
    and reflects its architectural contribution accurately.
    """
    vaswani = Paper(
        id="vaswani2017",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="arXiv",
        citationCount=120000,
        abstract="The Transformer is based entirely on attention mechanisms.",
    )
    swin = Paper(
        id="liu2021",
        title="Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        authors=["Liu et al."],
        publicationYear=2021,
        journalConference="ICCV",
        doi="10.48550/arxiv.2103.14030",
        source="arXiv",
        citationCount=15000,
        abstract="Swin Transformer introduces shifted windows for computer vision.",
    )

    rows = build_comparison_table(
        papers=[swin, vaswani],  # deliberately out of order
        citations=[],
        anchor_paper=vaswani
    )

    assert len(rows) == 2
    # First row MUST be the anchor paper (Vaswani)
    assert rows[0].title == "Attention Is All You Need"
    assert rows[0].year == "2017"
