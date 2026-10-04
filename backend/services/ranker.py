"""
Paper Ranker and Filter Service
===============================
Filters out withdrawn/empty papers, deduplicates candidates, and scores papers
using semantic reranking, citation counts, and title relevance bonuses.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Optional

from ..config import settings
from ..models.schemas import Paper, ResearchDepth
from .embeddings import embed_query, embed_texts
import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton Cross-Encoder Reranker ──────────────────────────────────────────

_reranker_model = None


def get_reranker():
    """Lazy load cross-encoder reranker for high-precision passage/abstract relevance."""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    try:
        from sentence_transformers import CrossEncoder
        logger.info("[Ranker] Loading CrossEncoder reranker: %s", settings.RERANK_MODEL)
        _reranker_model = CrossEncoder(settings.RERANK_MODEL, max_length=512)
        logger.info("[Ranker] CrossEncoder ready")
    except Exception as exc:
        logger.warning("[Ranker] CrossEncoder load failed (%s) - falling back to embeddings", exc)
        _reranker_model = None

    return _reranker_model


def normalize_paper_title(title: str) -> str:
    """
    Normalize paper title:
    - Lowercase
    - Drop a trailing version like 'v7' or 'v1.5'
    - Strip punctuation and non-alphanumeric characters (replace with space)
    - Collapse extra whitespace and strip
    """
    if not title:
        return ""
    t = title.strip().lower()
    t = re.sub(r"\s+v\d+(\.\d+)?$", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def validate_title_guesses(title_guesses: list[str]) -> list[str]:
    """
    Validate title guesses:
    Drop any guess that:
    - is empty or whitespace
    - starts with (case-insensitive): 'Introduction of', 'Paper title', 'A paper', 'The paper'
    - is shorter than 2 words
    Logs which guesses were dropped.
    """
    valid: list[str] = []
    _DISALLOWED_PREFIXES = (
        "introduction of",
        "paper title",
        "a paper",
        "the paper",
    )
    for raw in (title_guesses or []):
        g = raw.strip() if raw else ""
        if not g:
            logger.warning("[TitleGuess] Dropped empty title guess")
            continue
        lower_g = g.lower()
        if any(lower_g.startswith(prefix) for prefix in _DISALLOWED_PREFIXES):
            logger.warning("[TitleGuess] Dropped invalid title guess (disallowed prefix): '%s'", g)
            continue
        words = g.split()
        if len(words) < 2:
            logger.warning("[TitleGuess] Dropped invalid title guess (fewer than 2 words): '%s'", g)
            continue
        valid.append(g)
    return valid


def select_anchor_paper(
    papers: list[Paper],
    title_guesses: list[str],
) -> tuple[Optional[Paper], list[Paper], dict[str, Any]]:
    """
    Chooses an anchor paper for factual-lookup questions:
    - Validates title guesses (drops empty, disallowed prefixes, <2 words).
    - Normalizes titles (lowercase, strip punctuation and extra spaces, drop trailing version).
    - Anchor = retrieved paper whose normalized title EQUALS a normalized title guess exactly.
      Derivative papers like 'Cross-Attention is all you need: ...' must NOT match.
    - If several retrieved records match (e.g. same paper from arXiv and Semantic Scholar),
      merge them and keep the one with highest citationCount.
    - If no exact match, use fuzzy match only if similarity >= 0.95, otherwise anchor_paper_id = None.
    - Returns (anchor_paper, updated_papers_list, debug_dict).
    """
    import difflib

    valid_guesses = validate_title_guesses(title_guesses)
    if not papers or not valid_guesses:
        return None, papers, {"anchor_paper_id": None, "reason": "no_candidates_or_valid_guesses"}

    norm_guesses = [normalize_paper_title(g) for g in valid_guesses if normalize_paper_title(g)]
    if not norm_guesses:
        return None, papers, {"anchor_paper_id": None, "reason": "no_valid_guesses"}

    def get_cites(p: Paper) -> int:
        return getattr(p, "citationCount", 0) or 0

    # 1. Exact title match
    exact_matches: list[tuple[Paper, str]] = []
    for p in papers:
        p_norm = normalize_paper_title(p.title)
        for g_norm in norm_guesses:
            if p_norm == g_norm:
                exact_matches.append((p, g_norm))
                break

    if exact_matches:
        exact_matches.sort(
            key=lambda item: (get_cites(item[0]), 1 if item[0].pdfUrl else 0),
            reverse=True,
        )
        best_anchor, matched_guess = exact_matches[0]

        # Merge metadata from other exact matches
        merged_ids = [best_anchor.id]
        papers_to_remove = set()
        for other_p, _ in exact_matches[1:]:
            merged_ids.append(other_p.id)
            papers_to_remove.add(other_p.id)
            if not best_anchor.pdfUrl and other_p.pdfUrl:
                best_anchor.pdfUrl = other_p.pdfUrl
            if not best_anchor.doi and other_p.doi:
                best_anchor.doi = other_p.doi
            if not best_anchor.abstract and other_p.abstract:
                best_anchor.abstract = other_p.abstract
            if get_cites(other_p) > get_cites(best_anchor):
                best_anchor.citationCount = get_cites(other_p)
            if other_p.journalConference and other_p.journalConference not in ("arXiv", "Semantic Scholar", "unknown", ""):
                if best_anchor.journalConference in ("arXiv", "Semantic Scholar", "unknown", ""):
                    best_anchor.journalConference = other_p.journalConference

        updated_papers = [p for p in papers if p.id not in papers_to_remove]
        debug = {
            "anchor_paper_id": best_anchor.id,
            "anchor_paper_title": best_anchor.title,
            "anchor_doi": best_anchor.doi,
            "anchor_citations": get_cites(best_anchor),
            "match_type": "exact",
            "matched_guess": matched_guess,
            "merged_records": merged_ids,
        }
        logger.info(
            "[Anchor] Selected exact match anchor paper: '%s' (ID: %s, DOI: %s, citations: %d)",
            best_anchor.title,
            best_anchor.id,
            best_anchor.doi,
            get_cites(best_anchor),
        )
        return best_anchor, updated_papers, debug

    # 2. Anchor prefix rule (>= 3-word prefix, highest citationCount, >= ANCHOR_MIN_CITATIONS)
    min_citations = getattr(settings, "ANCHOR_MIN_CITATIONS", 500)
    prefix_matches: list[tuple[Paper, str]] = []
    for g_norm in norm_guesses:
        g_words = g_norm.split()
        if len(g_words) < 3:
            continue
        g_prefix_3 = " ".join(g_words[:3])
        for p in papers:
            p_norm = normalize_paper_title(p.title)
            p_words = p_norm.split()
            if len(p_words) < 3:
                continue

            is_prefix = False
            if p_norm.startswith(g_norm) or g_norm.startswith(p_norm):
                is_prefix = True
            elif p_norm.startswith(g_prefix_3):
                # Count matching leading words
                match_words = 0
                for w1, w2 in zip(g_words, p_words):
                    if w1 == w2:
                        match_words += 1
                    else:
                        break
                if match_words >= 3:
                    is_prefix = True

            if is_prefix and get_cites(p) >= min_citations:
                prefix_matches.append((p, g_norm))

    if prefix_matches:
        prefix_matches.sort(
            key=lambda item: (get_cites(item[0]), 1 if item[0].pdfUrl else 0),
            reverse=True,
        )
        best_anchor, matched_guess = prefix_matches[0]

        # Merge metadata from other matches with the same title key
        merged_ids = [best_anchor.id]
        papers_to_remove = set()
        best_key = _normalize_title(best_anchor.title)
        for other_p, _ in prefix_matches[1:]:
            if _normalize_title(other_p.title) == best_key:
                merged_ids.append(other_p.id)
                papers_to_remove.add(other_p.id)
                if not best_anchor.pdfUrl and other_p.pdfUrl:
                    best_anchor.pdfUrl = other_p.pdfUrl
                if not best_anchor.doi and other_p.doi:
                    best_anchor.doi = other_p.doi
                if not best_anchor.abstract and other_p.abstract:
                    best_anchor.abstract = other_p.abstract
                if get_cites(other_p) > get_cites(best_anchor):
                    best_anchor.citationCount = get_cites(other_p)
                if other_p.journalConference and other_p.journalConference not in ("arXiv", "Semantic Scholar", "unknown", ""):
                    if best_anchor.journalConference in ("arXiv", "Semantic Scholar", "unknown", ""):
                        best_anchor.journalConference = other_p.journalConference

        updated_papers = [p for p in papers if p.id not in papers_to_remove]
        debug = {
            "anchor_paper_id": best_anchor.id,
            "anchor_paper_title": best_anchor.title,
            "anchor_doi": best_anchor.doi,
            "anchor_citations": get_cites(best_anchor),
            "match_type": "prefix",
            "matched_guess": matched_guess,
            "merged_records": merged_ids,
        }
        logger.info(
            "[Anchor] Selected prefix match anchor paper: '%s' (ID: %s, citations: %d >= %d)",
            best_anchor.title,
            best_anchor.id,
            get_cites(best_anchor),
            min_citations,
        )
        return best_anchor, updated_papers, debug

    # 3. Fuzzy match fallback (similarity >= 0.95 required)
    fuzzy_matches: list[tuple[float, Paper, str]] = []
    for p in papers:
        p_norm = normalize_paper_title(p.title)
        for g_norm in norm_guesses:
            sim = difflib.SequenceMatcher(None, g_norm, p_norm).ratio()
            if sim >= 0.95:
                fuzzy_matches.append((sim, p, g_norm))

    if fuzzy_matches:
        fuzzy_matches.sort(key=lambda x: (x[0], get_cites(x[1])), reverse=True)
        best_sim, best_anchor, matched_guess = fuzzy_matches[0]
        debug = {
            "anchor_paper_id": best_anchor.id,
            "anchor_paper_title": best_anchor.title,
            "anchor_doi": best_anchor.doi,
            "anchor_citations": get_cites(best_anchor),
            "match_type": "fuzzy",
            "similarity": round(best_sim, 4),
            "matched_guess": matched_guess,
        }
        logger.info(
            "[Anchor] Selected fuzzy anchor paper: '%s' (similarity: %.3f, citations: %d)",
            best_anchor.title,
            best_sim,
            get_cites(best_anchor),
        )
        return best_anchor, papers, debug

    logger.info("[Anchor] No exact, prefix (>= %d cites), or >=0.95 fuzzy match found among %d candidate papers", min_citations, len(papers))
    return None, papers, {"anchor_paper_id": None, "reason": "no_match_above_threshold"}


def _normalize_title(title: str) -> str:
    """Normalize paper title for deduplication."""
    t = re.sub(r"\s+v\d+(\.\d+)?$", "", (title or "").lower().strip())
    return re.sub(r"[^a-z0-9]", "", t)


def _extract_id_key(paper: Paper) -> str:
    """Get canonical ID key (arXiv ID, DOI, or normalized title)."""
    if "10.48550/arxiv." in paper.doi.lower():
        raw_id = paper.doi.lower().split("10.48550/arxiv.")[-1].strip()
        return re.sub(r"v\d+$", "", raw_id)
    if paper.id.startswith("arxiv-"):
        raw_id = paper.id[len("arxiv-"):].strip()
        return re.sub(r"v\d+$", "", raw_id)
    if paper.doi:
        return paper.doi.lower().strip()
    return _normalize_title(paper.title)


def filter_and_deduplicate(papers: list[Paper]) -> list[Paper]:
    """
    Deduplicates papers by arXiv ID, DOI, and title.
    When duplicate records of the same paper are found (e.g. arXiv record + S2 record),
    merges them so the canonical record inherits S2's citationCount and publication venue.
    Drops papers with:
      - 'withdrawn' in title
      - empty abstract
      - publication year in the future
    """
    current_year = datetime.utcnow().year + 1
    key_to_paper: dict[str, Paper] = {}
    cleaned: list[Paper] = []

    for p in papers:
        # Check withdrawn
        if "withdrawn" in p.title.lower():
            logger.info("[Ranker] Dropping withdrawn paper: %s", p.title[:60])
            continue

        # Check empty abstract
        if not p.abstract or len(p.abstract.strip()) < 20:
            logger.info("[Ranker] Dropping paper with empty/short abstract: %s", p.title[:60])
            continue

        # Check future year
        if p.publicationYear > current_year:
            logger.info("[Ranker] Dropping future year paper (%d): %s", p.publicationYear, p.title[:60])
            continue

        # Canonical deduplication keys
        key = _extract_id_key(p)
        title_key = _normalize_title(p.title)

        existing = key_to_paper.get(key) or key_to_paper.get(title_key)
        if existing is not None:
            # Merge duplicate record into existing canonical record
            # 1. citationCount from S2
            p_cites = getattr(p, "citationCount", 0) or 0
            ex_cites = getattr(existing, "citationCount", 0) or 0
            if p_cites > ex_cites:
                existing.citationCount = p_cites
            # 2. venue from S2 (prefer specific venue over generic "arXiv" or "Semantic Scholar")
            if p.journalConference and p.journalConference not in ("arXiv", "Semantic Scholar", "unknown", ""):
                if existing.journalConference in ("arXiv", "Semantic Scholar", "unknown", ""):
                    existing.journalConference = p.journalConference
            # 3. PDF URL
            if not existing.pdfUrl and p.pdfUrl:
                existing.pdfUrl = p.pdfUrl
            # 4. DOI
            if not existing.doi and p.doi:
                existing.doi = p.doi
            # 5. Abstract
            if len(p.abstract or "") > len(existing.abstract or ""):
                existing.abstract = p.abstract

            # Map keys to existing
            key_to_paper[key] = existing
            key_to_paper[title_key] = existing
            continue

        key_to_paper[key] = p
        key_to_paper[title_key] = p
        cleaned.append(p)

    logger.info("[Ranker] Filtered & deduplicated %d candidates -> %d valid papers", len(papers), len(cleaned))
    return cleaned


def rank_papers(
    question: str,
    papers: list[Paper],
    question_type: str = "literature_review",
    title_guesses: Optional[list[str]] = None,
    depth: ResearchDepth = ResearchDepth.STANDARD,
    anchor_paper: Optional[Paper] = None,
) -> list[Paper]:
    """
    Ranks papers using a multi-factor scoring formula:
      Score = 0.6 * reranker(question, title + abstract) + 0.25 * log(citations + 1) + 0.15 * title_match
    """
    if not papers:
        return []

    depth_val = depth.value if hasattr(depth, "value") else str(depth)
    depth_counts = getattr(settings, "DEPTH_COUNTS", {"Quick": 6, "Standard": 12, "Deep": 24})
    target_count = depth_counts.get(depth_val, 12)

    cleaned = filter_and_deduplicate(papers)
    if not cleaned:
        return []

    # 1. Compute semantic relevance scores (Reranker or Embedding cosine)
    reranker = get_reranker()
    semantic_scores = []

    if reranker is not None:
        try:
            pairs = [[question, f"{p.title}. {p.abstract[:400]}"] for p in cleaned]
            raw_scores = reranker.predict(pairs)
            # Sigmoid / normalize to 0..1
            semantic_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]
        except Exception as exc:
            logger.warning("[Ranker] Reranker prediction failed: %s; falling back to embeddings", exc)
            reranker = None

    if not semantic_scores:
        try:
            q_emb = embed_query(question)
            doc_texts = [f"{p.title} {p.abstract[:300]}" for p in cleaned]
            doc_embs = embed_texts(doc_texts)
            # Dot product of normalized vectors
            q_vec = np.asarray(q_emb).squeeze()
            semantic_scores = [float(np.dot(q_vec, np.asarray(d_emb).squeeze())) for d_emb in doc_embs]
        except Exception as exc:
            logger.warning("[Ranker] Embedding scoring failed: %s; using equal weights", exc)
            semantic_scores = [0.5] * len(cleaned)

    # 2. Score each paper
    scored_papers: list[tuple[float, Paper]] = []
    title_guess_normalized = [_normalize_title(t) for t in (title_guesses or [])]

    # Max citation count for normalization
    max_cites = max([getattr(p, "citationCount", 0) for p in cleaned] + [1])

    for i, p in enumerate(cleaned):
        sem_score = semantic_scores[i]

        # Log-scaled citation score (0..1)
        cites = max(0, getattr(p, "citationCount", 0))
        cite_score = math.log1p(cites) / math.log1p(max(max_cites, 100))

        # Title match bonus
        title_norm = _normalize_title(p.title)
        title_match_bonus = 0.0

        for tg in title_guess_normalized:
            if tg and tg in title_norm:
                title_match_bonus = 1.0
                break

        # Check keyword overlap in title
        q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 3]
        if q_words:
            overlap = sum(1 for w in q_words if w in p.title.lower())
            title_match_bonus = max(title_match_bonus, overlap / len(q_words))

        # Weighted final score
        # For factual lookup: strongly boost exact title match and foundational papers
        if question_type == "factual_lookup":
            final_score = 0.45 * sem_score + 0.30 * cite_score + 0.25 * title_match_bonus
        else:
            final_score = 0.60 * sem_score + 0.25 * cite_score + 0.15 * title_match_bonus

        # Priority boost for verified anchor paper
        if anchor_paper and (p.id == anchor_paper.id or normalize_paper_title(p.title) == normalize_paper_title(anchor_paper.title)):
            final_score += 10.0

        scored_papers.append((final_score, p))

    # Sort descending by score
    scored_papers.sort(key=lambda x: x[0], reverse=True)

    ranked = [p for _, p in scored_papers]
    logger.info("[Ranker] Top paper: '%s' (score=%.3f)", ranked[0].title, scored_papers[0][0])

    # Re-assign sequential IDs: paper-1, paper-2, ...
    for idx, p in enumerate(ranked, start=1):
        p.id = f"paper-{idx}"

    return ranked[:target_count]
