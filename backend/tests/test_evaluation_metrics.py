"""
Unit tests for the Evaluation & Benchmark Metrics (RL-010 — Section 7 Audit Requirement).
Covers all 11 metrics now defined in evaluation/metrics.py.
"""

import pytest
from evaluation.metrics import (
    compute_citation_precision,
    compute_citation_recall,
    compute_evidence_precision,
    compute_evidence_recall,
    compute_mrr,
    compute_ndcg_at_k,
    compute_pdf_success_rate,
    compute_recall_at_k,
    compute_source_diversity,
    compute_unsupported_claim_rate,
    summarize_metrics,
)


# ─── Retrieval Metrics ────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_basic_recall(self):
        ground_truth = {"p1", "p2"}
        retrieved = ["p1", "p3", "p4", "p2", "p5"]
        assert compute_recall_at_k(retrieved, ground_truth, k=2) == 0.5
        assert compute_recall_at_k(retrieved, ground_truth, k=4) == 1.0
        assert compute_recall_at_k(retrieved, ground_truth, k=1) == 0.5

    def test_no_hit(self):
        assert compute_recall_at_k(["p8", "p9"], {"p1", "p2"}, k=2) == 0.0

    def test_empty_ground_truth(self):
        assert compute_recall_at_k(["p1"], set(), k=5) == 1.0

    def test_k_larger_than_retrieved(self):
        assert compute_recall_at_k(["p1"], {"p1"}, k=100) == 1.0


class TestMRR:
    def test_hit_at_rank_1(self):
        assert compute_mrr(["target", "p2", "p3"], {"target"}) == 1.0

    def test_hit_at_rank_2(self):
        assert compute_mrr(["p1", "target", "p3"], {"target"}) == 0.5

    def test_hit_at_rank_3(self):
        assert compute_mrr(["p1", "p2", "target"], {"target"}) == pytest.approx(1.0 / 3.0)

    def test_no_hit(self):
        assert compute_mrr(["p1", "p2", "p3"], {"target"}) == 0.0

    def test_empty_ground_truth(self):
        assert compute_mrr([], set()) == 1.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # Hits at rank 1 and 2 is ideal
        perfect = ["p1", "p2", "p3"]
        ndcg = compute_ndcg_at_k(perfect, {"p1", "p2"}, k=5)
        assert pytest.approx(ndcg, abs=1e-3) == 1.0

    def test_suboptimal_ranking(self):
        suboptimal = ["p3", "p1", "p2"]
        ndcg = compute_ndcg_at_k(suboptimal, {"p1", "p2"}, k=5)
        assert 0.0 < ndcg < 1.0

    def test_zero_ndcg(self):
        assert compute_ndcg_at_k(["p3", "p4"], {"p1", "p2"}, k=5) == 0.0

    def test_empty_ground_truth(self):
        assert compute_ndcg_at_k(["p1"], set(), k=5) == 1.0


# ─── Evidence Metrics ─────────────────────────────────────────────────────────

class TestEvidencePrecision:
    def test_normal(self):
        assert compute_evidence_precision(4, 5) == 0.8

    def test_zero_candidates(self):
        assert compute_evidence_precision(0, 0) == 0.0

    def test_perfect(self):
        assert compute_evidence_precision(10, 10) == 1.0

    def test_overflow_clipped(self):
        # Should never exceed 1.0
        assert compute_evidence_precision(10, 5) == 1.0


class TestEvidenceRecall:
    def test_full_recall(self):
        assert compute_evidence_recall(["p1", "p2"], {"p1", "p2"}) == 1.0

    def test_partial_recall(self):
        assert compute_evidence_recall(["p1"], {"p1", "p2"}) == 0.5

    def test_no_recall(self):
        assert compute_evidence_recall(["p3"], {"p1", "p2"}) == 0.0

    def test_empty_labeled(self):
        assert compute_evidence_recall(["p1"], set()) == 1.0

    def test_empty_recovered(self):
        assert compute_evidence_recall([], {"p1"}) == 0.0


# ─── Citation Metrics ─────────────────────────────────────────────────────────

class TestCitationPrecision:
    def test_perfect(self):
        assert compute_citation_precision(3, 3) == 1.0

    def test_half(self):
        assert compute_citation_precision(2, 4) == 0.5

    def test_zero_total(self):
        assert compute_citation_precision(0, 0) == 1.0


class TestCitationRecall:
    def test_perfect(self):
        assert compute_citation_recall(5, 5) == 1.0

    def test_partial(self):
        assert compute_citation_recall(3, 5) == 0.6

    def test_zero_total(self):
        assert compute_citation_recall(0, 0) == 1.0


# ─── Report Quality Metrics ───────────────────────────────────────────────────

class TestUnsupportedClaimRate:
    def test_normal(self):
        assert compute_unsupported_claim_rate(1, 5) == 0.2

    def test_all_unsupported(self):
        assert compute_unsupported_claim_rate(5, 5) == 1.0

    def test_none_unsupported(self):
        assert compute_unsupported_claim_rate(0, 5) == 0.0

    def test_zero_total(self):
        assert compute_unsupported_claim_rate(0, 0) == 0.0


class TestSourceDiversity:
    def test_distinct(self):
        assert compute_source_diversity(["p1", "p2", "p3"]) == 3.0

    def test_duplicates(self):
        assert compute_source_diversity(["p1", "p1", "p2"]) == 2.0

    def test_empty(self):
        assert compute_source_diversity([]) == 0.0


# ─── Infrastructure Metrics ───────────────────────────────────────────────────

class TestPDFSuccessRate:
    def test_full_success(self):
        assert compute_pdf_success_rate(10, 10) == 1.0

    def test_half(self):
        assert compute_pdf_success_rate(5, 10) == 0.5

    def test_no_papers(self):
        assert compute_pdf_success_rate(0, 0) == 1.0

    def test_zero_success(self):
        assert compute_pdf_success_rate(0, 5) == 0.0


# ─── Aggregate Helper ─────────────────────────────────────────────────────────

class TestSummarizeMetrics:
    def test_returns_all_keys(self):
        result = summarize_metrics(
            recall_5=0.8, recall_10=1.0, mrr=0.9, ndcg_10=0.85,
            evidence_precision=0.7, evidence_recall=0.6,
            citation_precision=0.95, citation_recall=0.8,
            unsupported_claim_rate=0.1, source_diversity=3.0,
            pdf_success_rate=0.75, latency_seconds=12.34,
        )
        expected_keys = {
            "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10",
            "evidence_precision", "evidence_recall",
            "citation_precision", "citation_recall",
            "unsupported_claim_rate", "source_diversity",
            "pdf_success_rate", "latency_seconds",
        }
        assert set(result.keys()) == expected_keys

    def test_rounding(self):
        result = summarize_metrics(
            recall_5=0.12345, recall_10=1.0, mrr=0.9, ndcg_10=0.85,
            evidence_precision=0.7, evidence_recall=0.6,
            citation_precision=0.95, citation_recall=0.8,
            unsupported_claim_rate=0.1, source_diversity=3.0,
            pdf_success_rate=0.75, latency_seconds=12.34567,
        )
        # 4 decimal places for metrics, 3 for latency
        assert result["recall_at_5"] == 0.1235
        assert result["latency_seconds"] == 12.346
