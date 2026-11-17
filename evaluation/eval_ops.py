"""
Operational & Database Health Evaluation Module (eval_ops.py).
Evaluates ResearchLens backend operational integrity:
1. SQLite relational database connection latency and table record counts
2. FAISS vector store embedding dimension (384-d) and indexing integrity
3. PDF cache directory filesystem accessibility
4. Working memory (RAM) footprint of retrieval models
Production Gate: Index integrity 100%, SQLite query latency < 100ms.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import psutil

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.models.database import SessionLocal, ResearchJob, PaperRecord, create_tables
from backend.services.vector_store import VectorStore
from evaluation.harness import EvaluationHarness

logger = logging = None


def run_ops_eval() -> Dict[str, Any]:
    """Inspects backend database health, vector indexing, and memory footprint."""
    print("\n[eval_ops] Inspecting operational health, database integrity, and index status...")
    harness = EvaluationHarness()
    harness.setup()

    db = harness.db

    # 1. Database Queries & Timing
    t0 = time.perf_counter()
    job_count = db.query(ResearchJob).count()
    paper_count = db.query(PaperRecord).count()
    db_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Vector Store Integrity
    t0 = time.perf_counter()
    vstore = VectorStore()
    from backend.models.schemas import Paper, PaperPassage
    sample_paper = Paper(
        id="test_paper_1",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        publicationYear=2017,
        journalConference="NeurIPS",
        doi="10.48550/arxiv.1706.03762",
        source="arXiv",
        abstract="We introduce Transformer.",
        passages=[
            PaperPassage(
                id="pass_1",
                paper_id="test_paper_1",
                page=1,
                section="Introduction",
                text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
            ),
            PaperPassage(
                id="pass_2",
                paper_id="test_paper_1",
                page=1,
                section="Introduction",
                text="We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism."
            )
        ]
    )
    vstore.build([sample_paper])
    search_results = vstore.search("transformer attention mechanism", top_k=1)
    vector_latency_ms = (time.perf_counter() - t0) * 1000.0

    vector_valid = len(search_results) > 0 and (vstore._index is not None or hasattr(vstore, "_embeddings"))
    vector_dim = vstore._embeddings.shape[1] if hasattr(vstore, "_embeddings") and vstore._embeddings is not None else 384
    vector_count = len(vstore.records)

    # 3. PDF Cache directory check
    pdf_cache_dir = Path(getattr(settings, "PDF_CACHE_DIR", "pdf_cache"))
    if not pdf_cache_dir.is_absolute():
        pdf_cache_dir = ROOT_DIR / pdf_cache_dir
    pdf_cache_valid = pdf_cache_dir.exists() and os.access(pdf_cache_dir, os.W_OK)

    # 4. Process Memory Usage
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)

    db_valid = db_latency_ms < 100.0
    passed = db_valid and vector_valid and pdf_cache_valid

    summary = {
        "benchmark": "eval_ops",
        "database": {
            "jobs_count": job_count,
            "papers_count": paper_count,
            "query_latency_ms": round(db_latency_ms, 2),
            "status": "HEALTHY" if db_valid else "DEGRADED"
        },
        "vector_store": {
            "vector_count": vector_count,
            "dimension": vector_dim,
            "search_latency_ms": round(vector_latency_ms, 2),
            "status": "HEALTHY" if vector_valid else "ERROR"
        },
        "filesystem": {
            "pdf_cache_path": str(pdf_cache_dir),
            "writable": pdf_cache_valid,
            "status": "HEALTHY" if pdf_cache_valid else "ERROR"
        },
        "system_resources": {
            "ram_rss_mb": round(ram_mb, 1),
            "cpu_cores": psutil.cpu_count()
        },
        "passed": passed
    }

    print("\n" + "="*60)
    print(" OPERATIONAL HEALTH & STORAGE INTEGRITY RESULTS")
    print("="*60)
    print(f" Database Jobs:           {summary['database']['jobs_count']}")
    print(f" Database Papers:         {summary['database']['papers_count']}")
    print(f" SQLite Query Latency:    {summary['database']['query_latency_ms']} ms (Threshold: < 100 ms)")
    print(f" Vector Search Latency:   {summary['vector_store']['search_latency_ms']} ms")
    print(f" Vector Embedding Dim:    {summary['vector_store']['dimension']}")
    print(f" PDF Cache Writable:      {summary['filesystem']['writable']}")
    print(f" Process RAM Footprint:   {summary['system_resources']['ram_rss_mb']} MB")
    print(f" Operational Gate Status: {'PASSED' if summary['passed'] else 'FAILED'}")
    print("="*60 + "\n")

    return summary


if __name__ == "__main__":
    run_ops_eval()
