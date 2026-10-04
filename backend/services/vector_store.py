"""
Vector Store Service — FAISS
=============================
Builds and queries an in-memory FAISS index over paper passage embeddings.
Provides top-k semantic retrieval for any claim or query string.

Each research job gets its own index (no cross-job contamination).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.schemas import Paper, PaperPassage
from .embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)


@dataclass
class PassageRecord:
    """Metadata stored alongside each FAISS vector."""
    paper_id: str
    paper_title: str
    passage: PaperPassage


@dataclass
class VectorStore:
    """
    Per-job FAISS index wrapping paper passage embeddings.
    Provides semantic search over all passages from retrieved papers.
    """
    records: list[PassageRecord] = field(default_factory=list)
    _index: Optional[object] = field(default=None, repr=False)

    def build(self, papers: list[Paper]) -> None:
        """
        Builds the FAISS index from all passages across the given papers.
        Must be called before any search.
        """
        self.records.clear()
        texts: list[str] = []

        for paper in papers:
            if not paper.passages:
                continue
            for passage in paper.passages:
                self.records.append(
                    PassageRecord(
                        paper_id=paper.id,
                        paper_title=paper.title,
                        passage=passage,
                    )
                )
                texts.append(passage.text)

        if not texts:
            logger.warning("[FAISS] No passage texts to index")
            return

        logger.info("[FAISS] Building index over %d passages", len(texts))
        embeddings = embed_texts(texts)                      # (N, D) float32
        self._embeddings = embeddings
        dim = embeddings.shape[1]

        try:
            import faiss
            # IndexFlatIP: exact inner-product search on L2-normalized vectors = cosine similarity
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            self._index = index
            logger.info("[FAISS] Index built: %d vectors, dim=%d", index.ntotal, dim)
        except Exception as exc:
            logger.warning("[FAISS] faiss indexing unavailable (%s) - falling back to numpy cosine similarity", exc)
            self._index = None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[PassageRecord, float]]:
        """
        Semantic search: returns the top-k passages most similar to `query`.

        Returns:
            List of (PassageRecord, similarity_score) sorted desc by score.
        """
        if not self.records:
            logger.warning("[FAISS] VectorStore records empty")
            return []

        q_vec = embed_query(query)                            # (1, D)

        if self._index is not None:
            scores, indices = self._index.search(q_vec, min(top_k, len(self.records)))
            results: list[tuple[PassageRecord, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.records):
                    continue
                results.append((self.records[int(idx)], float(score)))
            return results

        # Fallback using numpy dot product
        if hasattr(self, "_embeddings") and self._embeddings is not None:
            sims = np.dot(self._embeddings, q_vec.squeeze())
            top_indices = np.argsort(sims)[::-1][:min(top_k, len(self.records))]
            return [(self.records[int(idx)], float(sims[idx])) for idx in top_indices]

        return []

    def find_best_passage_for_claim(
        self,
        claim: str,
        paper_id: Optional[str] = None,
        top_k: int = 3,
    ) -> Optional[tuple[PassageRecord, float]]:
        """
        Finds the single best passage supporting a claim.
        If `paper_id` is given, restricts results to that paper.
        """
        candidates = self.search(claim, top_k=top_k * 3)

        if paper_id:
            candidates = [(r, s) for r, s in candidates if r.paper_id == paper_id]

        return candidates[0] if candidates else None
