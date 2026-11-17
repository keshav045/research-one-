"""
Evaluation Harness for ResearchLens Academic Research Assistant.
Serves as the common test runner, session lifecycle manager, and reporter.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.models.database import SessionLocal, create_tables
from backend.models.schemas import Paper, Citation, CitationStatus, ResearchDepth, ResearchReport
from backend.services.paper_search_manager import PaperSearchManager
from backend.services.ranker import normalize_paper_title
from backend.services.local_llm_service import synthesize_report

logger = logging.getLogger("researchlens.eval.harness")


def synthesize_report_sync(
    question: str,
    papers: Optional[List[Paper]] = None,
    citations: Optional[List[Citation]] = None,
    depth: ResearchDepth = ResearchDepth.QUICK,
    anchor_paper: Optional[Paper] = None,
) -> Tuple[ResearchReport, float, List[Dict[str, Any]]]:
    """Synchronous wrapper for synthesize_report with event loop management."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    p_list = papers or []
    c_list = citations or []
    anc = anchor_paper or (p_list[0] if p_list else None)

    return loop.run_until_complete(
        synthesize_report(
            question=question,
            depth=depth,
            papers=p_list,
            claims=[],
            citations=c_list,
            anchor_paper=anc
        )
    )


class EvaluationHarness:
    """
    Manages end-to-end evaluation runs, datasets, database lifecycle, and benchmark execution.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or str(CURRENT_DIR / "dataset.json")
        self.db = None
        self.search_manager = None
        self.dataset: List[Dict[str, Any]] = []
        self._initialized = False

    def setup(self):
        """Initializes database tables, search manager, and loads dataset."""
        if self._initialized:
            return

        logger.info("[Harness] Bootstrapping ResearchLens Evaluation Harness...")
        create_tables()
        self.db = SessionLocal()
        self.search_manager = PaperSearchManager()
        self.load_dataset()
        self._initialized = True
        logger.info("[Harness] ResearchLens Evaluation Harness ready.")

    def close(self):
        """Cleans up database session."""
        if self.db is not None:
            self.db.close()
            self.db = None
            self._initialized = False

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads evaluation benchmark questions from JSON dataset."""
        if not os.path.exists(self.dataset_path):
            logger.warning("[Harness] Dataset file not found: %s", self.dataset_path)
            self.dataset = []
            return self.dataset

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        logger.info("[Harness] Loaded %d evaluation benchmark queries from %s", len(self.dataset), self.dataset_path)
        return self.dataset

    def run_benchmark(
        self,
        name: str,
        eval_fn: Callable[[Dict[str, Any], "EvaluationHarness"], Dict[str, Any]],
        limit: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Executes eval_fn across dataset records, recording per-item latency and catching exceptions.
        """
        self.setup()
        records = list(self.dataset)

        if filter_fn:
            records = [r for r in records if filter_fn(r)]
        if limit:
            records = records[:limit]

        logger.info("[Harness] Starting benchmark '%s' across %d test cases...", name, len(records))
        start_time = time.perf_counter()
        results: List[Dict[str, Any]] = []
        durations: List[float] = []

        for idx, item in enumerate(records):
            item_start = time.perf_counter()
            try:
                res = eval_fn(item, self)
                item_dur = (time.perf_counter() - item_start) * 1000.0
                res["latency_ms"] = round(item_dur, 2)
                res["query_id"] = item.get("id", f"q_{idx}")
                res["question"] = item.get("question", "")
                res["domain"] = item.get("domain", "General")
                results.append(res)
                durations.append(item_dur)
            except Exception as exc:
                item_dur = (time.perf_counter() - item_start) * 1000.0
                logger.error("[Harness] Exception evaluating query '%s': %s", item.get("id", idx), exc)
                results.append({
                    "query_id": item.get("id", f"q_{idx}"),
                    "question": item.get("question", ""),
                    "error": str(exc),
                    "passed": False,
                    "latency_ms": round(item_dur, 2)
                })
                durations.append(item_dur)

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "name": name,
            "total_items": len(records),
            "total_time_ms": round(total_time_ms, 2),
            "results": results
        }
