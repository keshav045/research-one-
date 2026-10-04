"""
Paper Ranker and Filter Service
===============================
Filters out withdrawn/empty papers, deduplicates candidates, and scores papers
using semantic reranking, citation counts, and title relevance bonuses.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from datetime import datetime
from typing import Optional, Any

import httpx
import numpy as np

from ..config import settings
from ..models.schemas import Paper, ResearchDepth
from .embeddings import embed_query, embed_texts

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


def extract_key_terms(question: str) -> list[str]:
    """
    Extracts key term(s) from the question without an LLM:
    - Acronyms (e.g. ViT, GAN, RAG)
    - Phrases after 'introduced', 'proposed', 'developed', 'invented'
    - Method names with hyphens/digits (e.g. word2vec, doc2vec)
    - Multi-word methods (e.g. 'generative adversarial networks', 'batch normalization', 'retrieval-augmented generation')
    - Capitalized words/names (not at sentence start)
    """
    terms: list[str] = []
    # 1. Phrases after introduced / proposed / developed / invented
    m = re.search(r'(?:introduced|proposed|developed|invented)\s+(?:the\s+)?([A-Za-z0-9_\-\s]+?)(?:\?|\.|,|$|\band\b)', question, re.I)
    if m:
        phrase = m.group(1).strip()
        phrase_clean = re.sub(r'\s+(?:architecture|optimizer|algorithm|model|system|framework|approach)s?$', '', phrase, flags=re.I).strip()
        if phrase_clean and len(phrase_clean) > 2:
            terms.append(phrase_clean)
            if phrase != phrase_clean:
                terms.append(phrase)

    # 2. Acronyms in parens or standalone (e.g. ViT, GAN, RAG)
    for ac in re.findall(r'\b[A-Z0-9]{2,}\b', question):
        if ac.lower() not in ('who', 'what', 'which', 'nlp', 'the'):
            terms.append(ac)
    for ac_p in re.findall(r'\(([A-Za-z0-9_\-]+)\)', question):
        terms.append(ac_p)

    # 3. Method patterns like word2vec
    for w2v in re.findall(r'\b[a-zA-Z]+2[a-zA-Z0-9]+\b', question):
        terms.append(w2v)

    # 4. Multi-word phrases like 'generative adversarial networks', 'batch normalization', 'retrieval-augmented generation'
    for mw in re.findall(r'\b(?:generative adversarial networks?|batch normalization|retrieval-augmented generation|vision transformer|deep convolutional)\b', question, re.I):
        terms.append(mw)

    # 5. Capitalized words not at sentence start
    words = question.split()
    for w in words[1:]:
        clean_w = re.sub(r'[^a-zA-Z0-9_\-]', '', w)
        if clean_w and clean_w[0].isupper() and len(clean_w) > 2:
            if clean_w.lower() not in ('which', 'who', 'what', 'the', 'how'):
                terms.append(clean_w)

    seen = set()
    result = []
    for t in terms:
        t_clean = t.strip()
        if t_clean.lower() not in seen and len(t_clean) >= 2:
            seen.add(t_clean.lower())
            result.append(t_clean)
    return result


def _word_prefix_match(w1: str, w2: str) -> bool:
    """Compare words by exact match, stem, or 5-character prefix ('nets' matches 'networks')."""
    w1, w2 = w1.lower().strip(), w2.lower().strip()
    if not w1 or not w2:
        return False
    if w1 == w2:
        return True
    s1 = re.sub(r'(?:es|s)$', '', w1)
    s2 = re.sub(r'(?:es|s)$', '', w2)
    if s1 == s2 or s1 == w2 or s2 == w1:
        return True
    if (s1 == "net" and s2 == "network") or (s2 == "net" and s1 == "network"):
        return True
    if len(w1) >= 4 and len(w2) >= 4:
        min_len = min(len(w1), len(w2), 5)
        if w1[:min_len] == w2[:min_len]:
            return True
    return False


def _text_matches_key_term(text: str, key_term: str) -> bool:
    """Checks if text contains key_term using word boundary or 5-char prefix matching."""
    text_lower = text.lower()
    kt_lower = key_term.lower().strip()
    if not kt_lower:
        return False
    if kt_lower in text_lower:
        return True
    kt_words = kt_lower.split()
    text_words = re.findall(r'[a-z0-9]+', text_lower)
    if len(kt_words) == 1:
        return any(_word_prefix_match(kt_words[0], tw) for tw in text_words)
    n = len(kt_words)
    for i in range(len(text_words) - n + 1):
        window = text_words[i : i + n]
        if all(_word_prefix_match(kw, tw) for kw, tw in zip(kt_words, window)):
            return True
    return False


async def citation_chasing(
    candidates: list[Paper], key_terms: list[str]
) -> tuple[list[Paper], Counter, dict[str, list[Paper]]]:
    """
    Takes top 15 candidates by citationCount, fetches their references from S2 batch endpoint,
    counts reference frequency, filters for CS/ML field, logs top 5 most-referenced works,
    and returns (chased_papers, ref_counter, ref_citers).
    """
    sorted_cands = sorted(candidates, key=lambda p: getattr(p, "citationCount", 0) or 0, reverse=True)
    top_15 = sorted_cands[:15]
    if not top_15:
        return [], Counter(), {}

    from .paper_retrieval import _extract_s2_batch_id, _s2_limiter, S2_BATCH_API
    batch_ids: list[str] = []
    id_to_cand: dict[str, Paper] = {}
    for p in top_15:
        bid = _extract_s2_batch_id(p)
        if bid and bid not in batch_ids:
            batch_ids.append(bid)
            id_to_cand[bid] = p

    if not batch_ids:
        return [], Counter(), {}

    headers = {"User-Agent": "ResearchLens/2.0 (academic research tool)"}
    if getattr(settings, "SEMANTIC_SCHOLAR_API_KEY", "") and settings.SEMANTIC_SCHOLAR_API_KEY.strip():
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY.strip()

    ref_counter: Counter = Counter()
    ref_map: dict[str, dict] = {}
    ref_citers: dict[str, list[Paper]] = {}

    async with _s2_limiter:
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                params = {
                    "fields": "references.paperId,references.title,references.citationCount,references.year,references.externalIds,references.abstract,references.fieldsOfStudy,references.authors"
                }
                resp = await client.post(
                    S2_BATCH_API,
                    params=params,
                    json={"ids": batch_ids},
                    headers=headers,
                )
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_sec = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 3.0
                    logger.info("[CitationChasing] S2 batch rate limited; sleeping %.1fs before retry...", wait_sec)
                    _s2_limiter.penalize(wait_sec)
                    await asyncio.sleep(wait_sec)
                    resp = await client.post(
                        S2_BATCH_API,
                        params=params,
                        json={"ids": batch_ids},
                        headers=headers,
                    )
                from .paper_retrieval import _record_s2_call
                _record_s2_call(
                    f"batch_refs:{len(batch_ids)}",
                    resp.status_code,
                    0,
                    error=None if resp.status_code == 200 else f"HTTP {resp.status_code}",
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for idx_item, item in enumerate(data or []):
                        if not item or not isinstance(item, dict):
                            continue
                        cand_p = id_to_cand.get(batch_ids[idx_item]) if idx_item < len(batch_ids) else None
                        for ref in item.get("references", []) or []:
                            rtitle = ref.get("title")
                            if not rtitle or len(rtitle.strip()) < 5:
                                continue
                            fields = ref.get("fieldsOfStudy") or []
                            if fields and isinstance(fields, list):
                                cs_related = any(f.lower() in ("computer science", "mathematics", "engineering", "") for f in fields)
                                if not cs_related:
                                    continue
                            clean_t = rtitle.strip()
                            key = clean_t.lower()
                            ref_counter[key] += 1
                            if key not in ref_map or (ref.get("citationCount") or 0) > (ref_map[key].get("citationCount") or 0):
                                ref_map[key] = ref
                            if cand_p:
                                ref_citers.setdefault(key, []).append(cand_p)
                else:
                    logger.warning("[CitationChasing] S2 batch returned HTTP %d", resp.status_code)
        except Exception as exc:
            logger.warning("[CitationChasing] S2 batch failed: %s", exc)
            return [], Counter(), {}

    # Log the top 5 most-referenced works found
    top_5 = ref_counter.most_common(5)
    top_5_log = [f"'{ref_map[k].get('title')}' ({count}x cited, {ref_map[k].get('citationCount')} cites)" for k, count in top_5]
    logger.info("[CitationChasing] Top 5 most-referenced works found: %s", " | ".join(top_5_log) if top_5_log else "None")

    chased_papers: list[Paper] = []
    for k, count in ref_counter.most_common(30):
        r = ref_map[k]
        ext_ids = r.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        doi = ext_ids.get("DOI")
        clean_arxiv = re.sub(r"v\d+$", "", arxiv_id) if arxiv_id else None
        pid = f"arxiv-{clean_arxiv}" if clean_arxiv else (f"s2-{r.get('paperId')}" if r.get("paperId") else f"chased-{len(chased_papers)+1}")

        fields = r.get("fieldsOfStudy") or []
        fields_str = ", ".join(fields) if isinstance(fields, list) else str(fields)

        authors_raw = r.get("authors") or []
        authors_list = [
            a.get("name") if isinstance(a, dict) else str(a)
            for a in authors_raw
            if a
        ]
        if not authors_list:
            authors_list = ["Unknown"]

        pdf_url = f"https://arxiv.org/pdf/{clean_arxiv}" if clean_arxiv else None

        paper = Paper(
            id=pid,
            title=r.get("title") or "Untitled",
            authors=authors_list,
            citationCount=r.get("citationCount") or 0,
            publicationYear=r.get("year") or 2020,
            journalConference=fields_str or "Computer Science",
            doi=doi or "",
            source="Semantic Scholar",
            abstract=r.get("abstract") or f"Foundational work referenced {count} times by top candidate papers in literature review.",
            pdfUrl=pdf_url,
        )
        setattr(paper, "_ref_count", count)
        setattr(paper, "_citers", ref_citers.get(k, []))
        chased_papers.append(paper)

    return chased_papers, ref_counter, ref_citers


async def select_anchor_paper(
    papers: list[Paper],
    title_guesses: Optional[list[str]] = None,
    question: str = "",
) -> tuple[Optional[Paper], list[Paper], dict[str, Any]]:
    """
    Selects the foundational anchor paper for factual-lookup questions using simple rules:
    1. Exact title in the question (>= 4 words) or exact title-guess match.
    2. Else gate: candidates whose title OR abstract contains the key phrase
       (longest phrase first, stemmed by 5-character prefix). Hard AND gate.
    3. Among gated candidates with citationCount >= ANCHOR_MIN_CITATIONS (1000),
       pick the highest citationCount. If top two within 15%, set anchor_confidence='uncertain'
       and report both; prefer the earlier year only in that tie.
    4. If nothing passes the gate, citation chasing may run, but ONLY over citing papers
       that pass the key-phrase gate, and a chased candidate must also pass the gate
       or have reference share >= 0.6 over at least 5 gated citing papers. Otherwise anchor = None.
    5. No lift multiplier and no earliness weight in the main score.
    6. The nonsense question must end with anchor = None.
    """
    min_citations = getattr(settings, "ANCHOR_MIN_CITATIONS", 1000)

    def get_cites(p: Paper) -> int:
        return getattr(p, "citationCount", 0) or 0

    norm_q = normalize_paper_title(question)

    # 1. Exact title in the question (>= 4 words)
    exact_in_q: list[Paper] = []
    for p in papers:
        p_norm = normalize_paper_title(p.title)
        words = p_norm.split()
        if len(words) >= 4 and (p_norm in norm_q or norm_q in p_norm) and get_cites(p) >= min_citations:
            exact_in_q.append(p)
    if exact_in_q:
        exact_in_q.sort(key=lambda p: (get_cites(p), 1 if p.pdfUrl else 0), reverse=True)
        best_anchor = exact_in_q[0]
        logger.info("[Anchor] Rule 'exact_title_in_question' selected anchor: '%s' (cites=%d >= %d)", best_anchor.title, get_cites(best_anchor), min_citations)
        return best_anchor, papers, {
            "anchor_paper_id": best_anchor.id,
            "anchor_paper_title": best_anchor.title,
            "anchor_citations": get_cites(best_anchor),
            "anchor_rule": "exact_title_in_question",
            "anchor_confidence": "high",
        }

    # 1b. Exact title-guess match (>= min_citations)
    valid_guesses = validate_title_guesses(title_guesses) if title_guesses else []
    if valid_guesses:
        norm_guesses = [normalize_paper_title(g) for g in valid_guesses if normalize_paper_title(g)]
        exact_matches: list[Paper] = []
        for p in papers:
            p_norm = normalize_paper_title(p.title)
            if p_norm in norm_guesses and get_cites(p) >= min_citations:
                exact_matches.append(p)
        if exact_matches:
            exact_matches.sort(key=lambda p: (get_cites(p), 1 if p.pdfUrl else 0), reverse=True)
            best_anchor = exact_matches[0]
            logger.info("[Anchor] Rule 'exact_title_guess' selected anchor: '%s' (cites=%d >= %d)", best_anchor.title, get_cites(best_anchor), min_citations)
            return best_anchor, papers, {
                "anchor_paper_id": best_anchor.id,
                "anchor_paper_title": best_anchor.title,
                "anchor_citations": get_cites(best_anchor),
                "anchor_rule": "exact_title_guess",
                "anchor_confidence": "high",
            }

    # Extract key terms, prefer longest matching phrase
    raw_terms = extract_key_terms(question) if question else []
    key_terms = sorted(list(dict.fromkeys(raw_terms)), key=len, reverse=True)
    logger.info("[Anchor] Extracted key terms (longest first): %s", key_terms)

    # 2. Key phrase gate: candidates whose title OR abstract contains the key phrase
    # (longest phrase first, stemmed by 5-character prefix). Hard AND gate.
    gated_candidates: list[Paper] = []
    active_term = None

    for kt in key_terms:
        matching = [p for p in papers if _text_matches_key_term(f"{p.title} {p.abstract or ''}", kt)]
        if any(get_cites(p) >= min_citations for p in matching):
            gated_candidates = matching
            active_term = kt
            break
        elif matching and not gated_candidates:
            gated_candidates = matching
            active_term = kt

    if not gated_candidates and key_terms:
        for p in papers:
            p_text = f"{p.title} {p.abstract or ''}"
            if any(_text_matches_key_term(p_text, kt) for kt in key_terms):
                gated_candidates.append(p)

    logger.info("[Anchor] Key-phrase gate: %d / %d candidates passed gate (term='%s')", len(gated_candidates), len(papers), active_term or "None")

    # 3. Among gated candidates with citationCount >= ANCHOR_MIN_CITATIONS (1000), pick highest citationCount.
    qualified = [p for p in gated_candidates if get_cites(p) >= min_citations]
    if qualified:
        qualified.sort(key=lambda p: get_cites(p), reverse=True)
        top1 = qualified[0]
        runner_up = None
        anchor_confidence = "high"
        confidence_note = ""

        if len(qualified) > 1:
            top2 = qualified[1]
            c1 = get_cites(top1)
            c2 = get_cites(top2)
            if c1 > 0 and (c1 - c2) / c1 < 0.15:
                anchor_confidence = "uncertain"
                confidence_note = f"Source paper uncertain: '{top1.title}' or '{top2.title}' (margin < 15%)"
                y1 = getattr(top1, "publicationYear", 2024) or 2024
                y2 = getattr(top2, "publicationYear", 2024) or 2024
                if y2 < y1:
                    best_anchor = top2
                    runner_up = top1
                else:
                    best_anchor = top1
                    runner_up = top2
            else:
                best_anchor = top1
                runner_up = top2
        else:
            best_anchor = top1

        runner_up_str = f"'{runner_up.title}' ({get_cites(runner_up)} cites)" if runner_up else "None"
        logger.info(
            "[Anchor] Rule 'highest_citations_key_term' selected anchor: '%s' (%d cites, conf=%s)",
            best_anchor.title, get_cites(best_anchor), anchor_confidence
        )
        logger.info("[Anchor] Runner-up: %s", runner_up_str)
        if confidence_note:
            logger.warning("[Anchor] %s", confidence_note)

        return best_anchor, papers, {
            "anchor_paper_id": best_anchor.id,
            "anchor_paper_title": best_anchor.title,
            "anchor_citations": get_cites(best_anchor),
            "anchor_rule": "highest_citations_key_term",
            "anchor_confidence": anchor_confidence,
            "confidence_note": confidence_note,
            "runner_up": runner_up_str,
        }

    # 4. If nothing passes the gate, citation chasing may run,
    # but ONLY over citing papers that pass the key-phrase gate,
    # and a chased candidate must also pass the gate or have reference share >= 0.6 over at least 5 gated citing papers.
    # Otherwise anchor = None.
    gated_citers = [p for p in papers if any(_text_matches_key_term(f"{p.title} {p.abstract or ''}", kt) for kt in key_terms)]
    if len(gated_citers) >= 3:
        logger.info(
            "[Anchor] No gated candidate met %d citations. Running citation chasing over %d gated citing papers...",
            min_citations, len(gated_citers)
        )
        chased_papers, ref_counter, ref_citers = await citation_chasing(gated_citers, key_terms)
        n_gated = len(gated_citers)
        valid_chased: list[tuple[Paper, float]] = []

        for cp in chased_papers:
            rf = getattr(cp, "_ref_count", 0) or ref_counter.get(cp.title.strip().lower(), 0)
            ref_share = (rf / n_gated) if n_gated > 0 else 0.0
            passes_gate = any(_text_matches_key_term(f"{cp.title} {cp.abstract or ''}", kt) for kt in key_terms)
            if (passes_gate or (ref_share >= 0.60 and n_gated >= 5)) and get_cites(cp) >= min_citations:
                valid_chased.append((cp, ref_share))

        if valid_chased:
            valid_chased.sort(key=lambda x: (x[1], get_cites(x[0])), reverse=True)
            best_chased, best_share = valid_chased[0]
            logger.info(
                "[Anchor] Rule 'citation_chasing' selected anchor: '%s' (%d cites, share=%.2f)",
                best_chased.title, get_cites(best_chased), best_share
            )
            return best_chased, papers, {
                "anchor_paper_id": best_chased.id,
                "anchor_paper_title": best_chased.title,
                "anchor_citations": get_cites(best_chased),
                "anchor_rule": "citation_chasing",
                "anchor_confidence": "high",
                "ref_share": best_share,
            }

    # 6. Nonsense / no match: anchor = None
    logger.info("[Anchor] Rule 'none': No candidate paper met threshold or passed gate (min %d citations)", min_citations)
    return None, papers, {
        "anchor_paper_id": None,
        "anchor_rule": "none",
        "reason": "no_anchor_above_threshold",
    }


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
        # For factual lookup: raise citation weight (configurable) and log weights
        if question_type == "factual_lookup":
            sem_weight = getattr(settings, "FACTUAL_SEMANTIC_WEIGHT", 0.35)
            cite_weight = getattr(settings, "FACTUAL_CITATION_WEIGHT", 0.45)
            title_weight = getattr(settings, "FACTUAL_TITLE_WEIGHT", 0.20)
            if i == 0:
                logger.info(
                    "[Ranker] Factual lookup weights: semantic=%.2f, citation=%.2f, title=%.2f",
                    sem_weight,
                    cite_weight,
                    title_weight,
                )
            final_score = sem_weight * sem_score + cite_weight * cite_score + title_weight * title_match_bonus
        else:
            final_score = 0.60 * sem_score + 0.25 * cite_score + 0.15 * title_match_bonus

        scored_papers.append((final_score, p))

    # Sort descending by score
    scored_papers.sort(key=lambda x: x[0], reverse=True)
    ranked_pool = [p for _, p in scored_papers]

    # Selection cut for factual_lookup:
    # 1. The anchor must ALWAYS be in the final ranked set (placed at position 0).
    #    The cross-encoder must not be able to remove the anchor.
    # 2. Plus the top 3 key-term candidates by citationCount must be in the final set.
    if question_type == "factual_lookup":
        final_ranked: list[Paper] = []
        seen_pids = set()

        if anchor_paper:
            matched_anchor = next(
                (p for p in cleaned if p.id == anchor_paper.id or normalize_paper_title(p.title) == normalize_paper_title(anchor_paper.title)),
                anchor_paper,
            )
            final_ranked.append(matched_anchor)
            seen_pids.add(matched_anchor.id)

        # Top 3 key-term candidates by citationCount
        key_terms = extract_key_terms(question) if question else []
        key_term_cands: list[Paper] = []
        for p in cleaned:
            text = f"{p.title} {p.abstract or ''}".lower()
            for kt in key_terms:
                kt_clean = kt.lower().strip()
                if kt_clean and (kt_clean in text or re.search(r'\b' + re.escape(kt_clean) + r'\b', text)):
                    key_term_cands.append(p)
                    break
        key_term_cands.sort(key=lambda p: getattr(p, "citationCount", 0) or 0, reverse=True)
        for kt_p in key_term_cands:
            if len(final_ranked) >= 4:
                break
            if kt_p.id not in seen_pids:
                final_ranked.append(kt_p)
                seen_pids.add(kt_p.id)

        # Fill remaining slots from ranked_pool up to target_count
        for p in ranked_pool:
            if len(final_ranked) >= target_count:
                break
            if p.id not in seen_pids:
                final_ranked.append(p)
                seen_pids.add(p.id)

        ranked = final_ranked
    else:
        ranked = ranked_pool[:target_count]

    logger.info("[Ranker] Top paper: '%s' (cites=%d)", ranked[0].title, getattr(ranked[0], "citationCount", 0) or 0)

    # Re-assign sequential IDs: paper-1, paper-2, ...
    for idx, p in enumerate(ranked, start=1):
        p.id = f"paper-{idx}"

    return ranked
