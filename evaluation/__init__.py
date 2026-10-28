"""ResearchLens Evaluation & Benchmark Suite.

Provides the full metric suite from the Engineering Audit Report (RL-010):
  - Recall@K, MRR, NDCG@K              (retrieval)
  - Evidence Precision, Evidence Recall  (evidence)
  - Citation Precision, Citation Recall  (citations)
  - Unsupported Claim Rate               (report quality)
  - Source Diversity                     (report breadth)
  - PDF Success Rate                     (infrastructure)
  - summarize_metrics                    (aggregate helper)
"""
from evaluation.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_evidence_precision,
    compute_evidence_recall,
    compute_citation_precision,
    compute_citation_recall,
    compute_unsupported_claim_rate,
    compute_source_diversity,
    compute_pdf_success_rate,
    summarize_metrics,
)

__all__ = [
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_evidence_precision",
    "compute_evidence_recall",
    "compute_citation_precision",
    "compute_citation_recall",
    "compute_unsupported_claim_rate",
    "compute_source_diversity",
    "compute_pdf_success_rate",
    "summarize_metrics",
]
