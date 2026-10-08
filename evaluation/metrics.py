"""
Information Retrieval & Grounding Evaluation Metrics
===================================================
Implements standard metrics defined in Section 7 of the audit report:
- Recall@K: Proportion of relevant papers retrieved in top-K.
- MRR (Mean Reciprocal Rank): 1 / rank of first relevant paper.
- NDCG@K (Normalized Discounted Cumulative Gain): Ranking quality across relevant papers.
- Evidence Precision: Verified claims / total candidate claims.
- Citation Precision: Proportion of report citations resolving to valid references.
- Unsupported Claim Rate: Proportion of claims rejected due to lack of evidence.
"""

from __future__ import annotations

import math
from typing import Set, Sequence


def compute_recall_at_k(retrieved: Sequence[str], ground_truth: Set[str], k: int = 10) -> float:
    """Proportion of relevant items retrieved in top-K candidates."""
    if not ground_truth:
        return 1.0
    top_k = set(retrieved[:k])
    hits = len(top_k.intersection(ground_truth))
    return hits / len(ground_truth)


def compute_mrr(retrieved: Sequence[str], ground_truth: Set[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant candidate."""
    if not ground_truth:
        return 1.0
    for rank, item in enumerate(retrieved, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved: Sequence[str], ground_truth: Set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    if not ground_truth:
        return 1.0
    
    # DCG
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        rel = 1.0 if item in ground_truth else 0.0
        dcg += rel / math.log2(rank + 1)
        
    # Ideal DCG (IDCG)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    
    return (dcg / idcg) if idcg > 0.0 else 0.0


def compute_evidence_precision(verified_claims: int, candidate_claims: int) -> float:
    """How much of the candidate evidence passed relevance and provenance gates."""
    if candidate_claims <= 0:
        return 0.0
    return min(1.0, verified_claims / candidate_claims)


def compute_citation_precision(valid_citations: int, total_citations: int) -> float:
    """Proportion of inline citations [n] that resolve to references in 1..len(references)."""
    if total_citations <= 0:
        return 1.0
    return min(1.0, valid_citations / total_citations)


def compute_unsupported_claim_rate(unsupported_claims: int, total_claims: int) -> float:
    """Proportion of extracted candidate claims rejected due to insufficient evidence."""
    if total_claims <= 0:
        return 0.0
    return min(1.0, unsupported_claims / total_claims)
