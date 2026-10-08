"""
Unified Paper Search Manager
============================
Coordinates paper search across arXiv, Semantic Scholar, and OpenAlex.
Provides:
  - search_all_sources(query, sources, ...) -> tuple[list[Paper], dict]
  - search_arxiv(query, limit) -> list[Paper]
  - search_semantic_scholar(query, limit) -> list[Paper]
  - search_openalex(query, limit) -> list[Paper]
  - search statistics & error isolation
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..models.schemas import Paper
from .paper_retrieval import (
    _fetch_arxiv_query,
    _fetch_semantic_scholar_query,
    _fetch_openalex_query,
)

logger = logging.getLogger(__name__)


class PaperSearchManager:
    """Unified manager for multi-source academic paper retrieval."""

    def __init__(self):
        self.arxiv_enabled = getattr(settings, "ARXIV_ENABLED", True)
        self.semantic_scholar_enabled = getattr(settings, "SEMANTIC_SCHOLAR_ENABLED", True)
        self.openalex_enabled = getattr(settings, "OPENALEX_ENABLED", True)
        self.crossref_enabled = getattr(settings, "CROSSREF_ENABLED", False)
        self.core_enabled = getattr(settings, "CORE_ENABLED", False)

    async def search_arxiv(self, query: str, limit: int = 10) -> List[Paper]:
        """Search arXiv with query phrase."""
        if not self.arxiv_enabled:
            logger.info("[SearchManager] arXiv is disabled via configuration.")
            return []
        try:
            return await _fetch_arxiv_query(f'all:"{query}"', limit)
        except Exception as exc:
            logger.warning("[SearchManager] arXiv search failed for '%s': %s", query, exc)
            return []

    async def search_semantic_scholar(self, query: str, limit: int = 10) -> List[Paper]:
        """Search Semantic Scholar."""
        if not self.semantic_scholar_enabled:
            logger.info("[SearchManager] Semantic Scholar is disabled via configuration.")
            return []
        try:
            return await _fetch_semantic_scholar_query(query, limit)
        except Exception as exc:
            logger.warning("[SearchManager] Semantic Scholar search failed for '%s': %s", query, exc)
            return []

    async def search_openalex(self, query: str, limit: int = 10) -> List[Paper]:
        """Search OpenAlex works."""
        if not self.openalex_enabled:
            logger.info("[SearchManager] OpenAlex is disabled via configuration.")
            return []
        try:
            return await _fetch_openalex_query(query, limit)
        except Exception as exc:
            logger.warning("[SearchManager] OpenAlex search failed for '%s': %s", query, exc)
            return []

    async def search_all_sources(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        limit_per_source: int = 10,
    ) -> Tuple[List[Paper], Dict[str, Any]]:
        """
        Executes concurrent search across all enabled academic sources.
        Returns (papers, search_stats) with per-source discovery counts.
        """
        active_sources = sources or ["arXiv", "Semantic Scholar", "OpenAlex"]
        tasks: Dict[str, asyncio.Task] = {}

        if "arXiv" in active_sources and self.arxiv_enabled:
            tasks["arXiv"] = asyncio.create_task(self.search_arxiv(query, limit_per_source))
        if "Semantic Scholar" in active_sources and self.semantic_scholar_enabled:
            tasks["Semantic Scholar"] = asyncio.create_task(self.search_semantic_scholar(query, limit_per_source))
        if "OpenAlex" in active_sources and self.openalex_enabled:
            tasks["OpenAlex"] = asyncio.create_task(self.search_openalex(query, limit_per_source))

        all_papers: List[Paper] = []
        source_counts: Dict[str, int] = {}
        errors: Dict[str, str] = {}

        for src, task in tasks.items():
            try:
                res = await task
                source_counts[src] = len(res)
                all_papers.extend(res)
            except Exception as exc:
                logger.error("[SearchManager] Source '%s' error: %s", src, exc)
                errors[src] = str(exc)
                source_counts[src] = 0

        stats: Dict[str, Any] = {
            "query": query,
            "sources_searched": list(tasks.keys()),
            "source_counts": source_counts,
            "total_discovered": len(all_papers),
            "errors": errors,
        }

        logger.info(
            "[SearchManager] Discovered %d papers across sources: %s",
            len(all_papers),
            source_counts,
        )
        return all_papers, stats


_search_manager = PaperSearchManager()


def get_search_manager() -> PaperSearchManager:
    return _search_manager


async def search_all_sources(
    query: str,
    sources: Optional[List[str]] = None,
    limit_per_source: int = 10,
) -> Tuple[List[Paper], Dict[str, Any]]:
    """Module-level convenience wrapper."""
    return await _search_manager.search_all_sources(query, sources, limit_per_source)
