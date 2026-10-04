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


_PREFIXES_TO_STRIP = (
    "introduction of",
    "introduction to",
    "the paper",
    "a paper",
    "paper title",
)
_LEFTOVER_PLACEHOLDERS = frozenset({
    "unknown",
    "placeholder",
    "title",
    "n a",
    "na",
    "none",
    "paper",
    "the",
    "a",
})


def validate_title_guesses(title_guesses: list[str]) -> list[str]:
    """
    Validate title guesses:
    - Strips prefixes ('Introduction of', 'Introduction to', 'The paper', 'A paper', 'Paper title')
    - Keeps remainder if it has >= 2 words
    - Drops leftover placeholders ('unknown', 'placeholder', 'title', etc.) and guesses with < 2 words
    - Logs each strip and drop.
    """
    valid: list[str] = []
    for raw in (title_guesses or []):
        g = raw.strip() if raw else ""
        if not g:
            logger.warning("[TitleGuess] Dropped empty title guess")
            continue

        stripped = g
        lower_g = stripped.lower()

        # Check and strip prefixes
        for prefix in _PREFIXES_TO_STRIP:
            if lower_g.startswith(prefix):
                remainder = stripped[len(prefix):].strip(" :-\t\n\r\"'")
                logger.info("[TitleGuess] Stripped prefix '%s' from '%s' -> remainder: '%s'", prefix, g, remainder)
                stripped = remainder
                lower_g = stripped.lower()
                break

        # Check leftover placeholder
        clean_norm = re.sub(r"[^a-z0-9\s]", " ", lower_g).strip()
        if clean_norm in _LEFTOVER_PLACEHOLDERS or not stripped:
            logger.warning("[TitleGuess] Dropped placeholder/empty title guess: '%s' (originally '%s')", stripped, g)
            continue

        words = stripped.split()
        if len(words) < 2:
            logger.warning("[TitleGuess] Dropped invalid title guess (fewer than 2 words): '%s' (originally '%s')", stripped, g)
            continue

        valid.append(stripped)
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


def _extract_arxiv_id(paper: Paper) -> Optional[str]:
    """Extract clean arXiv ID without version suffix."""
    if paper.doi and "10.48550/arxiv." in paper.doi.lower():
        raw_id = paper.doi.lower().split("10.48550/arxiv.")[-1].strip()
        return re.sub(r"v\d+$", "", raw_id)
    if paper.id.startswith("arxiv-"):
        raw_id = paper.id[len("arxiv-"):].strip()
        return re.sub(r"v\d+$", "", raw_id)
    if paper.pdfUrl and "arxiv.org/pdf/" in paper.pdfUrl:
        raw_id = paper.pdfUrl.split("arxiv.org/pdf/")[-1].replace(".pdf", "").strip()
        return re.sub(r"v\d+$", "", raw_id)
    return None


def _extract_doi(paper: Paper) -> Optional[str]:
    """Extract clean DOI (excluding arXiv proxy DOIs)."""
    if paper.doi:
        d = paper.doi.strip().lower()
        if d.startswith("https://doi.org/"):
            d = d[len("https://doi.org/"):].strip()
        elif d.startswith("http://doi.org/"):
            d = d[len("http://doi.org/"):].strip()
        if d and not d.startswith("10.48550/arxiv."):
            return d
    return None


def filter_and_deduplicate(papers: list[Paper]) -> list[Paper]:
    """
    Deduplicates candidate papers:
    - Checks in priority order: arXiv ID, DOI, then normalized title.
    - When duplicate records of the same paper are found, merges them:
      keeps the highest citationCount, merges pdfUrl, abstract, venue, and authors.
    - Drops papers with:
      - 'withdrawn' in title
      - empty abstract (< 20 chars)
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

        # Canonical deduplication keys in priority order: arXiv ID, DOI, title
        aid = _extract_arxiv_id(p)
        doi = _extract_doi(p)
        title_key = _normalize_title(p.title)

        existing: Optional[Paper] = None
        if aid and f"arxiv:{aid}" in key_to_paper:
            existing = key_to_paper[f"arxiv:{aid}"]
        elif doi and f"doi:{doi}" in key_to_paper:
            existing = key_to_paper[f"doi:{doi}"]
        elif f"title:{title_key}" in key_to_paper:
            existing = key_to_paper[f"title:{title_key}"]

        if existing is not None:
            # Merge duplicate record into existing canonical record
            # 1. Highest citationCount
            p_cites = getattr(p, "citationCount", 0) or 0
            ex_cites = getattr(existing, "citationCount", 0) or 0
            if p_cites > ex_cites:
                existing.citationCount = p_cites
            # 2. PDF URL
            if not existing.pdfUrl and p.pdfUrl:
                existing.pdfUrl = p.pdfUrl
            # 3. Abstract (prefer longer, richer text)
            if len(p.abstract or "") > len(existing.abstract or ""):
                existing.abstract = p.abstract
            # 4. DOI
            if not existing.doi and p.doi:
                existing.doi = p.doi
            # 5. Venue (prefer specific venue over generic "arXiv" or "Semantic Scholar" or "OpenAlex")
            if p.journalConference and p.journalConference not in ("arXiv", "Semantic Scholar", "OpenAlex", "unknown", ""):
                if existing.journalConference in ("arXiv", "Semantic Scholar", "OpenAlex", "unknown", ""):
                    existing.journalConference = p.journalConference
            # 6. Authors
            if (not existing.authors or existing.authors == ["Unknown"]) and (p.authors and p.authors != ["Unknown"]):
                existing.authors = p.authors

            # Map all keys of this paper to the canonical record
            if aid:
                key_to_paper[f"arxiv:{aid}"] = existing
            if doi:
                key_to_paper[f"doi:{doi}"] = existing
            key_to_paper[f"title:{title_key}"] = existing
            continue

        # New canonical paper
        if aid:
            key_to_paper[f"arxiv:{aid}"] = p
        if doi:
            key_to_paper[f"doi:{doi}"] = p
        key_to_paper[f"title:{title_key}"] = p
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
