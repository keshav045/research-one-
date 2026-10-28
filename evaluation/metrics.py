"""
Information Retrieval & Grounding Evaluation Metrics
====================================================
Implements the complete metric suite from Section 7 of the Engineering Audit Report
(RL-010, P2): all metrics are now available for reproducible benchmarking.

Retrieval metrics:
  - Recall@K          Did the system retrieve the papers needed to answer the question?
  - MRR               How early does the first relevant paper appear?
  - NDCG@K            Is ranking quality good across multiple relevant papers?

Evidence metrics:
  - Evidence Precision  How much retrieved evidence is actually useful/supporting?
  - Evidence Recall     How much of the labeled supporting evidence was recovered?

Citation metrics:
  - Citation Precision  How many citations actually support their claims?
  - Citation Recall     How many important claims are adequately cited?

Report-quality metrics:
  - Unsupported Claim Rate  How often does the report assert something unsupported?
  - Source Diversity         Does the answer rely on a narrow or broad literature base?

Infrastructure metrics:
  - PDF Success Rate    How often can full text be resolved and parsed?
  - End-to-end Latency  How long does a complete research run take? (measured externally)
"""

from __future__ import annotations

import math
from typing import Collection, Sequence, Set


# ── Retrieval Metrics ─────────────────────────────────────────────────────────

def compute_recall_at_k(
    retrieved: Sequence[str],
    ground_truth: Set[str],
    k: int = 10,
) -> float:
    """Proportion of relevant items (papers/passages) retrieved in the top-K candidates.

    Args:
        retrieved:    Ordered list of retrieved item identifiers.
        ground_truth: Set of known-relevant item identifiers.
        k:            Cutoff rank.

    Returns:
        Float in [0, 1]; 1.0 if ground_truth is empty (no recall obligation).
    """
    if not ground_truth:
        return 1.0
    top_k = set(retrieved[:k])
    hits = len(top_k.intersection(ground_truth))
    return hits / len(ground_truth)


def compute_mrr(retrieved: Sequence[str], ground_truth: Set[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant candidate.

    Returns:
        Float in (0, 1]; 0.0 if no relevant item is found; 1.0 if empty ground_truth.
    """
    if not ground_truth:
        return 1.0
    for rank, item in enumerate(retrieved, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved: Sequence[str],
    ground_truth: Set[str],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at rank K.

    Assumes binary relevance (relevant = 1, not relevant = 0).

    Returns:
        Float in [0, 1]; 1.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 1.0

    # DCG
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        rel = 1.0 if item in ground_truth else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG (IDCG): all relevant items at the top positions
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return (dcg / idcg) if idcg > 0.0 else 0.0


# ── Evidence Metrics ──────────────────────────────────────────────────────────

def compute_evidence_precision(verified_claims: int, candidate_claims: int) -> float:
    """Precision of the evidence pipeline: verified claims / all candidate claims.

    High precision means the relevance + provenance gate is selective.

    Returns:
        Float in [0, 1]; 0.0 if no candidates were produced.
    """
    if candidate_claims <= 0:
        return 0.0
    return min(1.0, verified_claims / candidate_claims)


def compute_evidence_recall(
    recovered_passages: Collection[str],
    labeled_passages: Set[str],
) -> float:
    """Recall of the evidence pipeline against a labeled ground-truth passage set.

    Measures how much of the known-relevant evidence was actually recovered by
    the PDF extraction + vector retrieval pipeline.

    Args:
        recovered_passages: Identifiers (or normalized texts) of passages the
                            pipeline returned as supporting evidence.
        labeled_passages:   Ground-truth set of passage identifiers/texts
                            known to support the research question.

    Returns:
        Float in [0, 1]; 1.0 if labeled_passages is empty.
    """
    if not labeled_passages:
        return 1.0
    hits = len(set(recovered_passages).intersection(labeled_passages))
    return hits / len(labeled_passages)


# ── Citation Metrics ──────────────────────────────────────────────────────────

def compute_citation_precision(valid_citations: int, total_citations: int) -> float:
    """Proportion of inline citations [n] that resolve to valid references in 1..N.

    A citation is valid if its badge number n satisfies 1 ≤ n ≤ len(references)
    and the cited passage actually supports the claim it accompanies.

    Returns:
        Float in [0, 1]; 1.0 if total_citations == 0 (no citation errors possible).
    """
    if total_citations <= 0:
        return 1.0
    return min(1.0, valid_citations / total_citations)


def compute_citation_recall(
    cited_claims: int,
    total_important_claims: int,
) -> float:
    """Proportion of important claims that are adequately supported by at least one citation.

    Low citation recall means the report makes unsupported assertions about
    things that have known evidence in the retrieved literature.

    Args:
        cited_claims:           Number of important claims that carry a valid citation.
        total_important_claims: Total number of claims considered important enough
                                to require citation.

    Returns:
        Float in [0, 1]; 1.0 if total_important_claims == 0.
    """
    if total_important_claims <= 0:
        return 1.0
    return min(1.0, cited_claims / total_important_claims)


# ── Report Quality Metrics ────────────────────────────────────────────────────

def compute_unsupported_claim_rate(
    unsupported_claims: int,
    total_claims: int,
) -> float:
    """Proportion of extracted candidate claims rejected due to insufficient evidence.

    A low rate means the pipeline is producing well-grounded output; a high rate
    may indicate poor retrieval, low-quality PDFs, or overly strict provenance gates.

    Returns:
        Float in [0, 1]; 0.0 if no claims were processed.
    """
    if total_claims <= 0:
        return 0.0
    return min(1.0, unsupported_claims / total_claims)


def compute_source_diversity(paper_ids_used: Collection[str]) -> float:
    """Measure whether the report relies on a broad or narrow literature base.

    Returns the number of distinct papers cited in the final report as a simple
    diversity count. Use alongside a threshold (e.g., ≥ 3 papers = acceptable).

    Args:
        paper_ids_used: Collection of paper IDs that contributed at least one
                        verified citation to the final report.

    Returns:
        Integer count of unique sources cast to float for uniformity.
    """
    return float(len(set(paper_ids_used)))


# ── Infrastructure Metrics ────────────────────────────────────────────────────

def compute_pdf_success_rate(
    pdf_success_count: int,
    total_papers_attempted: int,
) -> float:
    """Proportion of papers for which full-text PDF was successfully resolved and parsed.

    A low PDF success rate indicates download failures, magic-byte rejections, or
    parsing errors. The pipeline falls back to abstract-only passages for failures.

    Returns:
        Float in [0, 1]; 1.0 if total_papers_attempted == 0.
    """
    if total_papers_attempted <= 0:
        return 1.0
    return min(1.0, pdf_success_count / total_papers_attempted)


# ── Aggregate Report Helper ───────────────────────────────────────────────────

def summarize_metrics(
    *,
    recall_5: float,
    recall_10: float,
    mrr: float,
    ndcg_10: float,
    evidence_precision: float,
    evidence_recall: float,
    citation_precision: float,
    citation_recall: float,
    unsupported_claim_rate: float,
    source_diversity: float,
    pdf_success_rate: float,
    latency_seconds: float,
) -> dict:
    """Bundle all metrics into a single serializable result dict.

    Use this to write per-query or aggregate benchmark results to JSON.
    """
    return {
        "recall_at_5":           round(recall_5, 4),
        "recall_at_10":          round(recall_10, 4),
        "mrr":                   round(mrr, 4),
        "ndcg_at_10":            round(ndcg_10, 4),
        "evidence_precision":    round(evidence_precision, 4),
        "evidence_recall":       round(evidence_recall, 4),
        "citation_precision":    round(citation_precision, 4),
        "citation_recall":       round(citation_recall, 4),
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        "source_diversity":      int(source_diversity),
        "pdf_success_rate":      round(pdf_success_rate, 4),
        "latency_seconds":       round(latency_seconds, 3),
    }
