"""
Vector Store Service — FAISS & Semantic Search
===============================================
Builds and queries an in-memory FAISS index (or exact cosine numpy fallback)
over paper passage embeddings. Provides top-k semantic retrieval.

Optimizations:
1. Skip non-body sections (References, Bibliography, Acknowledgements, Appendix) via SKIP_NONBODY_SECTIONS.
2. Deduplicate near-identical / duplicate passages per paper using normalized text hashing.
3. Optional passage cap per paper via MAX_PASSAGES_PER_PAPER (ranked by keyword overlap with query).
4. Detailed performance and token metric telemetry in VectorStore.build.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import settings
from ..models.schemas import Paper, PaperPassage
from .embeddings import embed_query, embed_texts, get_model

logger = logging.getLogger(__name__)

NONBODY_SECTIONS = {
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "appendices",
}


def _normalize_passage_text(text: str) -> str:
    """Normalize text for duplicate detection: lowercase, strip punctuation, collapse whitespace."""
    t = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(t.split())


def _score_passage_overlap(passage_text: str, question_tokens: set[str]) -> int:
    """Fast keyword overlap score between question tokens and passage."""
    p_tokens = set(re.sub(r"[^\w\s]", "", passage_text.lower()).split())
    return len(p_tokens.intersection(question_tokens))


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

    def build(self, papers: list[Paper], question: Optional[str] = None) -> None:
        """
        Builds the FAISS index from filtered passages across the given papers.
        Must be called before any search.
        """
        self.records.clear()
        texts: list[str] = []

        total_skipped_nonbody = 0
        total_dropped_duplicates = 0
        total_dropped_capped = 0

        skip_nonbody = getattr(settings, "SKIP_NONBODY_SECTIONS", True)
        max_cap = getattr(settings, "MAX_PASSAGES_PER_PAPER", 0)

        q_tokens = set()
        if question:
            q_tokens = {w for w in re.sub(r"[^\w\s]", "", question.lower()).split() if len(w) > 2}

        for paper in papers:
            if not paper.passages:
                continue

            accepted_passages: list[PaperPassage] = []
            seen_hashes: set[str] = set()

            for passage in paper.passages:
                # 1. Skip non-body sections
                if skip_nonbody:
                    sec = (passage.section or "").strip().lower()
                    if sec in NONBODY_SECTIONS or any(sec.startswith(nb) for nb in NONBODY_SECTIONS):
                        total_skipped_nonbody += 1
                        continue

                # 2. Drop duplicate / near-duplicate passages within the same paper
                norm_text = _normalize_passage_text(passage.text)
                text_hash = hashlib.md5(norm_text.encode("utf-8")).hexdigest()
                if text_hash in seen_hashes:
                    total_dropped_duplicates += 1
                    continue
                seen_hashes.add(text_hash)

                accepted_passages.append(passage)

            # 3. Optional passage cap per paper
            if max_cap > 0 and len(accepted_passages) > max_cap:
                effective_cap = max(max_cap, 15)  # never cap below minimum needed for evidence
                if len(accepted_passages) > effective_cap:
                    orig_count = len(accepted_passages)
                    if q_tokens:
                        accepted_passages.sort(
                            key=lambda p: (_score_passage_overlap(p.text, q_tokens), -p.page),
                            reverse=True,
                        )
                    total_dropped_capped += orig_count - effective_cap
                    accepted_passages = accepted_passages[:effective_cap]
                    accepted_passages.sort(key=lambda p: p.page)
                    logger.info(
                        "[VectorStore] Capped passages for '%s': %d -> %d (dropped %d)",
                        paper.title[:40], orig_count, len(accepted_passages), orig_count - len(accepted_passages)
                    )

            for passage in accepted_passages:
                self.records.append(
                    PassageRecord(
                        paper_id=paper.id,
                        paper_title=paper.title,
                        passage=passage,
                    )
                )
                texts.append(passage.text)

        if total_skipped_nonbody > 0 or total_dropped_duplicates > 0 or total_dropped_capped > 0:
            logger.info(
                "[VectorStore.build] Filtered passages: nonbody_skipped=%d, duplicates_dropped=%d, "
                "capped_dropped=%d, indexed_passages=%d",
                total_skipped_nonbody, total_dropped_duplicates, total_dropped_capped, len(texts)
            )

        if not texts:
            logger.warning("[FAISS] No passage texts to index")
            return

        logger.info("[FAISS] Building index over %d passages", len(texts))
        t_emb_start = time.perf_counter()
        embeddings = embed_texts(texts)                      # (N, D) float32
        embed_sec = time.perf_counter() - t_emb_start
        self._embeddings = embeddings
        dim = embeddings.shape[1]

        faiss_add_sec = 0.0
        try:
            import faiss
            index = faiss.IndexFlatIP(dim)
            t_faiss_start = time.perf_counter()
            index.add(embeddings)
            faiss_add_sec = time.perf_counter() - t_faiss_start
            self._index = index
            logger.info("[FAISS] Index built: %d vectors, dim=%d", index.ntotal, dim)
        except Exception as exc:
            logger.warning("[FAISS] faiss indexing unavailable (%s) - falling back to numpy cosine similarity", exc)
            self._index = None

        try:
            tokenizer, _ = get_model()
            tok_counts = [len(tok_ids) for tok_ids in tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]]
            mean_tokens = float(np.mean(tok_counts)) if tok_counts else 0.0
            p95_tokens = float(np.percentile(tok_counts, 95)) if tok_counts else 0.0
        except Exception as exc:
            logger.warning("[VectorStore] Token count calculation failed: %s", exc)
            mean_tokens, p95_tokens = 0.0, 0.0

        logger.info(
            "[VectorStore.build] papers=%d, passages=%d, tokens_per_passage (mean=%.1f, p95=%.1f), "
            "embedding_seconds=%.3fs, faiss_add_seconds=%.3fs",
            len(papers), len(texts), mean_tokens, p95_tokens, embed_sec, faiss_add_sec
        )

    def search_paper(
        self,
        paper_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[PassageRecord, float]]:
        """
        Paper-scoped semantic search (Phase 4):
        FIRST restricts corpus strictly to Paper X, then searches within that subset.
        Avoids global search and post-filtering where Paper X might be excluded from global top-k.
        """
        if not self.records:
            return []

        paper_indices = [i for i, r in enumerate(self.records) if r.paper_id == paper_id]
        if not paper_indices:
            logger.info("[FAISS] No passages found for paper_id '%s'", paper_id)
            return []

        q_vec = embed_query(query)  # (1, D)
        q_flat = q_vec.squeeze()

        if hasattr(self, "_embeddings") and self._embeddings is not None:
            paper_embs = self._embeddings[paper_indices]
            sims = np.dot(paper_embs, q_flat)
            ranked_local = np.argsort(sims)[::-1][:min(top_k, len(paper_indices))]
            return [(self.records[paper_indices[idx]], float(sims[idx])) for idx in ranked_local]

        return []

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: Optional[str] = None,
    ) -> list[tuple[PassageRecord, float]]:
        """
        Semantic search: returns the top-k passages most similar to `query`.
        If `paper_id` is provided, strictly restricts retrieval to Paper X first.
        """
        if paper_id:
            return self.search_paper(paper_id, query, top_k=top_k)

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
        If `paper_id` is given, strictly scopes retrieval to Paper X first.
        """
        if paper_id:
            candidates = self.search_paper(paper_id, claim, top_k=top_k)
        else:
            candidates = self.search(claim, top_k=top_k)

        return candidates[0] if candidates else None
