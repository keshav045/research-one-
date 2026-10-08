"""
Unit tests for the Evaluation & Benchmark Metrics (Section 7 Audit Requirement)
"""

import pytest
from evaluation.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_evidence_precision,
    compute_citation_precision,
    compute_unsupported_claim_rate,
)


def test_recall_at_k():
    ground_truth = {"p1", "p2"}
    retrieved = ["p1", "p3", "p4", "p2", "p5"]
    assert compute_recall_at_k(retrieved, ground_truth, k=2) == 0.5
    assert compute_recall_at_k(retrieved, ground_truth, k=4) == 1.0
    assert compute_recall_at_k(retrieved, ground_truth, k=1) == 0.5
    assert compute_recall_at_k(["p8", "p9"], ground_truth, k=2) == 0.0


def test_mrr():
    ground_truth = {"target"}
    assert compute_mrr(["target", "p2", "p3"], ground_truth) == 1.0
    assert compute_mrr(["p1", "target", "p3"], ground_truth) == 0.5
    assert compute_mrr(["p1", "p2", "target"], ground_truth) == 1.0 / 3.0
    assert compute_mrr(["p1", "p2", "p3"], ground_truth) == 0.0


def test_ndcg_at_k():
    ground_truth = {"p1", "p2"}
    # Perfect ranking: hits at rank 1 and 2
    perfect = ["p1", "p2", "p3"]
    assert pytest.approx(compute_ndcg_at_k(perfect, ground_truth, k=5), 0.001) == 1.0

    # Suboptimal ranking: hits at rank 2 and 3
    suboptimal = ["p3", "p1", "p2"]
    ndcg = compute_ndcg_at_k(suboptimal, ground_truth, k=5)
    assert 0.0 < ndcg < 1.0


def test_evidence_and_citation_precision():
    assert compute_evidence_precision(4, 5) == 0.8
    assert compute_evidence_precision(0, 0) == 0.0
    assert compute_citation_precision(3, 3) == 1.0
    assert compute_citation_precision(2, 4) == 0.5
    assert compute_unsupported_claim_rate(1, 5) == 0.2
