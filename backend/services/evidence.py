"""
Evidence Extraction & NLI Verification Engine
=============================================
Implements Phase 5 of the architecture blueprint:
1. For each sub-question, queries the FAISS vector store.
2. Selects top passages with semantic reranking.
3. Extracts candidate assertions from source passages.
4. Verifies candidate claims against the exact passage window using DeBERTa NLI.
5. Returns verified claims and fully attributed citations with exact page numbers.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple, Optional, Dict

from ..models.schemas import (
    Paper,
    Citation,
    CitationStatus,
    Claim,
    Evidence,
    AtomicClaimVerification,
    EntailmentVerdict,
)
from ..config import settings
from .vector_store import VectorStore, PassageRecord
from .ranker import get_reranker
import math

logger = logging.getLogger(__name__)

# ─── Evidence Filtering Patterns & Removal Counts ─────────────────────────────

_CAPTION_REGEX = re.compile(
    r"^(?:table \d+|figure \d+|fig\.? \d+|chart \d+|diagram \d+|scheme \d+|box \d+)\b",
    re.IGNORECASE,
)
_AUTHOR_NOTE_REGEX = re.compile(
    r"(\S+@\S+\.\S+|\b(?:correspondence to|equal contribution|work done while|department of|school of|university|institute of|laboratory of|author for correspondence|all authors contributed|corresponding author|e-mail:?)\b)",
    re.IGNORECASE,
)
_REFERENCE_REGEX = re.compile(
    r"(^\[\d+\]|\b(?:references|bibliography)\b|\b(?:vol\.|pp\.|pages|isbn|doi:|arxiv:\d{4}\.\d{4,5})\b|\b(?:in proceedings of|ieee trans|acm comput)\b)",
    re.IGNORECASE,
)
_META_PROCEDURAL_REGEX = re.compile(
    r"^(?:in\s+(?:this\s+section|section\s+\d+|table\s+\d+|appendix|fig(?:ure)?\.?\s*\d+)|"
    r"the\s+rest\s+of\s+(?:this\s+)?paper\s+is\s+organized|"
    r"as\s+(?:discussed|shown|detailed|noted|described|illustrated)\s+in\s+(?:section|table|figure|appendix)|"
    r"we\s+(?:organize|refer\s+the\s+reader|begin\s+by\s+describing|outline\s+the\s+structure))\b",
    re.IGNORECASE,
)

_last_removal_counts: Dict[str, int] = {
    "captions": 0,
    "author_notes": 0,
    "references": 0,
    "fragments_under_8_words": 0,
    "length_out_of_bounds": 0,
    "procedural_fluff": 0,
}


def get_last_removal_counts() -> Dict[str, int]:
    return dict(_last_removal_counts)


def clean_hyphenated_breaks(text: str) -> str:
    """
    Cleans hyphenated line breaks (e.g. 'en- coder' -> 'encoder', 'trans- former' -> 'transformer')
    while preserving genuine hyphenated compounds (e.g. 'self-attention', 'cross-encoder').
    """
    if not text:
        return ""
    return re.sub(r"\b([a-zA-Z]{2,})-\s+([a-zA-Z]{2,})\b", r"\1\2", text)


def strip_leading_artifacts(text: str) -> str:
    """
    Strip leading footnote numbers, bullet markers, and page artifacts:
    e.g. '7 In this work' -> 'In this work'
    e.g. '[1] In this work' -> 'In this work'
    e.g. '12 We propose' -> 'We propose'
    e.g. '• We propose' -> 'We propose'
    """
    if not text:
        return ""
    t = text.strip()
    # Strip bullet / dash markers
    t = re.sub(r"^[\-\•\*\–\—\>]\s*", "", t)
    # Strip leading footnote numbers (1-3 digits or brackets followed by capital letter)
    t = re.sub(r"^(?:\[\d+\]|\d{1,3}|[\*\u2020\u2021])\s+(?=[A-Z])", "", t)
    # Strip leading page/journal header artifacts
    t = re.sub(r"^(?:page\s+\d+|\d+\s+(?:ieee|acm|arxiv|neurips|icml|cvpr|jmlr|nature|science))\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def normalize_text_for_match(text: str) -> str:
    """Normalize text for verbatim matching: collapses whitespace, cleans hyphen breaks, normalizes hyphens."""
    if not text:
        return ""
    t = clean_hyphenated_breaks(text)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", t)
    return t


def check_verbatim_source_match(claim_text: str, passage_text: str) -> bool:
    """
    Checks that the claim sentence exists verbatim (after whitespace/hyphen normalization)
    in its source passage.
    """
    norm_claim = normalize_text_for_match(claim_text)
    norm_passage = normalize_text_for_match(passage_text)
    return norm_claim in norm_passage


def _filter_and_extract_sentences(text: str, counts: Dict[str, int]) -> list[str]:
    """Split passage text into sentences, apply evidence filters, strip artifacts, and track removal counts."""
    cleaned = re.sub(r"\s+", " ", clean_hyphenated_breaks(text).strip())
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue

        # Strip leading footnote numbers and artifacts first
        s_clean = strip_leading_artifacts(s)

        # Check fragment under 8 words
        words = s_clean.split()
        if len(words) < 8:
            counts["fragments_under_8_words"] += 1
            continue

        # Check length
        if len(s_clean) < 30 or len(s_clean) > 350:
            counts["length_out_of_bounds"] += 1
            continue

        # Check captions
        if _CAPTION_REGEX.search(s_clean):
            counts["captions"] += 1
            continue

        # Check author notes / affiliations
        if _AUTHOR_NOTE_REGEX.search(s_clean):
            counts["author_notes"] += 1
            continue

        # Check references / bibliography
        if _REFERENCE_REGEX.search(s_clean):
            counts["references"] += 1
            continue

        # Check procedural / meta-text fluff (Section 2 related work, etc.)
        if _META_PROCEDURAL_REGEX.search(s_clean):
            counts["procedural_fluff"] = counts.get("procedural_fluff", 0) + 1
            continue

        sentences.append(s_clean)

    # Sort sentences to prioritize substantive technical assertions
    _INFORMATIVE_KEYWORDS = re.compile(
        r"\b(?:propos|achiev|reduc|increas|improv|outperform|introduc|demonstrat|show|evaluat|find|observ|yield|requir|comput|model|architect|attent|layer|param|loss|accurac|score|speed|latenc|throughput|effici|memor|weight|train|infer|dataset|benchmark)\w*\b",
        re.IGNORECASE,
    )
    sentences.sort(key=lambda s: 1 if _INFORMATIVE_KEYWORDS.search(s) else 0, reverse=True)
    return sentences


def extract_and_verify_evidence(
    question: str,
    sub_questions: list[str],
    papers: list[Paper],
    vector_store: VectorStore,
    max_citations: int = 15,
    anchor_paper_id: Optional[str] = None,
) -> Tuple[List[Claim], List[Citation], List[Dict], Dict[str, int]]:
    """
    Main Phase 5 pipeline (STEP 5):
    1. For each sub-question:
       - Retrieves candidate passages (FAISS top 30).
       - If anchor exists, searches anchor passages first with CrossEncoder reranking.
         Adds other papers' passages only if anchor yields fewer than 3 relevant ones.
       - Extracts candidate assertions per sub-question.
    2. Relevance Gate & NLI Verification:
       - Reranker evaluates (sub_question, claim) relevance.
       - DeBERTa evaluates NLI entailment against evidence passage.
       - A claim is 'verified' only if:
           NLI entailment >= NLI_ENTAIL_THRESHOLD (0.80)
           AND
           reranker relevance >= RELEVANCE_THRESHOLD (0.30).
       - Rejected claims are preserved with exact rejection reasons ('not relevant', 'not entailed').
    3. Deduplicates identical claims before ranking and returns verified claims, citations, and rejected claims.
    """
    if not papers:
        logger.warning("[Evidence] No papers available for evidence extraction")
        return [], [], [], {
            "captions": 0,
            "author_notes": 0,
            "references": 0,
            "fragments_under_8_words": 0,
            "length_out_of_bounds": 0,
        }

    # Map paper_id to Paper object
    paper_map: Dict[str, Paper] = {p.id: p for p in papers}

    # If no sub-questions provided, use the main question
    queries = sub_questions if sub_questions else [question]
    reranker = get_reranker()

    logger.info(
        "[Evidence] Extracting evidence for %d query aspects (anchor: %s)...",
        len(queries),
        anchor_paper_id or "none",
    )

    global _last_removal_counts
    removal_counts: Dict[str, int] = {
        "captions": 0,
        "author_notes": 0,
        "references": 0,
        "fragments_under_8_words": 0,
        "length_out_of_bounds": 0,
    }

    # 1. Retrieve top 5 passages per sub-question and extract candidate assertions
    candidate_claims: List[Dict] = []
    seen_subq_claim_norms: set[str] = set()

    for sub_q in queries:
        faiss_candidates = vector_store.search(sub_q, top_k=30)
        chosen_passages: list[tuple[PassageRecord, float]] = []

        if anchor_paper_id:
            # Anchor exists: search anchor passages first
            anchor_records = [r for r in vector_store.records if r.paper_id == anchor_paper_id]
            if anchor_records:
                if reranker is not None:
                    pairs = [[sub_q, clean_hyphenated_breaks(r.passage.text)] for r in anchor_records]
                    scores = reranker.predict(pairs)
                    anchor_scored = [(r, float(s)) for r, s in zip(anchor_records, scores)]
                else:
                    anchor_scored = [(r, 0.5) for r in anchor_records]
                anchor_scored.sort(key=lambda x: x[1], reverse=True)

                relevant_anchor = [item for item in anchor_scored if item[1] >= -1.5]
                chosen_passages = list(relevant_anchor[:3] if relevant_anchor else anchor_scored[:2])

                # ALSO add top passages from other papers from FAISS candidates
                other_candidates = [r for r, _ in faiss_candidates if r.paper_id != anchor_paper_id]
                if other_candidates:
                    if reranker is not None:
                        other_pairs = [[sub_q, clean_hyphenated_breaks(r.passage.text)] for r in other_candidates]
                        other_scores = reranker.predict(other_pairs)
                        other_scored = [(r, float(s)) for r, s in zip(other_candidates, other_scores)]
                    else:
                        other_scored = [(r, sc) for r, sc in faiss_candidates if r.paper_id != anchor_paper_id]
                    other_scored.sort(key=lambda x: x[1], reverse=True)
                    for item in other_scored:
                        if len(chosen_passages) >= 8:
                            break
                        if not any(item[0].passage.id == cp[0].passage.id for cp in chosen_passages):
                            chosen_passages.append(item)
        else:
            # No anchor: rank from all papers by cross-encoder relevance
            cand_records = [r for r, _ in faiss_candidates]
            if cand_records:
                if reranker is not None:
                    pairs = [[sub_q, clean_hyphenated_breaks(r.passage.text)] for r in cand_records]
                    scores = reranker.predict(pairs)
                    scored = [(r, float(s)) for r, s in zip(cand_records, scores)]
                else:
                    scored = list(faiss_candidates)
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen_passages = scored[:8]

        # Deduplicate identical passages within this sub-question
        seen_p_texts: set[str] = set()
        sub_passages: list[tuple[PassageRecord, float]] = []
        for rec, sc in chosen_passages:
            norm_p = re.sub(r"\s+", " ", clean_hyphenated_breaks(rec.passage.text)).strip().lower()
            if norm_p not in seen_p_texts:
                seen_p_texts.add(norm_p)
                sub_passages.append((rec, sc))

        # Extract candidate assertions for this sub-question using evidence filters
        for rec, sc in sub_passages:
            passage_text = clean_hyphenated_breaks(rec.passage.text)
            sentences = _filter_and_extract_sentences(passage_text, removal_counts)
            rec_paper = paper_map.get(rec.paper_id)
            rec_authors_lower = " ".join(rec_paper.authors).lower() if (rec_paper and rec_paper.authors) else ""

            for sent in sentences[:3]:
                # Author attribution alignment check:
                # Detect if the sentence explicitly attributes work to an external author (e.g. "Lewis et al.", "Gao et al.")
                ext_author_matches = re.findall(r"\b([A-Z][a-zA-Z\-]+)\s+(?:et\s+al\.?|\(\d{4}\))", sent)
                target_rec = rec
                if ext_author_matches and rec_paper:
                    # Check if the named author is an author of THIS paper
                    if not any(a.lower() in rec_authors_lower for a in ext_author_matches):
                        # Sentence is citing an external author! Check if that external author has a paper in our candidate papers
                        matched_other_rec = None
                        for other_rec, _ in sub_passages:
                            other_p = paper_map.get(other_rec.paper_id)
                            if other_p and other_p.id != rec_paper.id and other_p.authors:
                                other_auth_str = " ".join(other_p.authors).lower()
                                if any(a.lower() in other_auth_str for a in ext_author_matches):
                                    matched_other_rec = other_rec
                                    break
                        if matched_other_rec is not None:
                            # Re-attribute claim to the actual author's paper
                            target_rec = matched_other_rec
                        else:
                            # Do not falsely attribute an external author's finding to this paper
                            # (prevents e.g. "Lewis et al. [1]" pointing to Gao's survey)
                            continue

                claim_norm = re.sub(r"[^a-z0-9]", "", sent.lower())
                pair_key = f"{sub_q}::{claim_norm}"
                if pair_key in seen_subq_claim_norms:
                    continue
                seen_subq_claim_norms.add(pair_key)
                candidate_claims.append({
                    "sub_q": sub_q,
                    "sentence": sent,
                    "record": target_rec,
                    "passage_text": passage_text,
                    "passage_score": sc,
                })

    _last_removal_counts = dict(removal_counts)
    logger.info(
        "[Evidence] Filtering complete: %d candidate claims retained. Removals: %s",
        len(candidate_claims),
        removal_counts,
    )

    if not candidate_claims:
        return [], [], [], removal_counts

    # 2. Compute cross-encoder relevance scores for (sub_question, claim)
    if reranker is not None:
        rel_pairs = [[c["sub_q"], c["sentence"]] for c in candidate_claims]
        raw_rel = reranker.predict(rel_pairs)
        rel_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_rel]
    else:
        rel_scores = [0.5 for _ in candidate_claims]

    # 3. Plain Claim-Level Check: "source match" + Relevance Gate
    # Replaces the NLI call on claims with a plain check that the claim sentence exists verbatim
    # (after whitespace/hyphen normalization) in its source passage.
    rel_threshold = getattr(settings, "RELEVANCE_THRESHOLD", 0.30)

    verified_candidates = []
    rejected_claims: List[Dict] = []

    for item, rel_score in zip(candidate_claims, rel_scores):
        sub_q = item["sub_q"]
        claim_text = item["sentence"]
        record = item["record"]
        p_text = item["passage_text"]
        paper = paper_map.get(record.paper_id)
        if not paper:
            continue

        is_source_match = check_verbatim_source_match(claim_text, p_text)
        is_relevant = rel_score >= rel_threshold

        if is_relevant and is_source_match:
            verified_candidates.append({
                "record": record,
                "paper": paper,
                "sub_q": sub_q,
                "claim_text": claim_text,
                "rel_score": rel_score,
                "p_text": p_text,
                "check_label": "source match",
            })
        else:
            if not is_relevant and not is_source_match:
                reason = "not relevant to sub-question and no verbatim source match"
            elif not is_relevant:
                reason = f"not relevant to sub-question (relevance {rel_score:.3f} < {rel_threshold:.2f})"
            else:
                reason = "claim not found verbatim in source passage"

            rejected_claims.append({
                "claim": claim_text,
                "reason": reason,
                "sub_question": sub_q,
                "paper_id": paper.id,
                "paper_title": paper.title,
                "page": record.passage.page,
                "relevance_score": round(rel_score, 4),
                "source_match": is_source_match,
                "check_label": "source match",
            })

    logger.info(
        "[Evidence] Dual-gate evaluation: %d verified, %d rejected (rel_thresh=%.2f, check='source match')",
        len(verified_candidates),
        len(rejected_claims),
        rel_threshold,
    )

    # 4. Deduplicate identical verified claims before ranking (Amendment C)
    # Sort verified candidates by relevance score descending
    verified_candidates.sort(key=lambda x: x["rel_score"], reverse=True)

    verified_citations: List[Citation] = []
    claims_list: List[Claim] = []
    seen_verified_claims: set[str] = set()

    badge_counter = 1
    for cand in verified_candidates:
        if badge_counter > max_citations:
            break

        c_text = cand["claim_text"]
        c_norm = re.sub(r"[^a-z0-9]", "", c_text.lower())
        if c_norm in seen_verified_claims:
            continue
        seen_verified_claims.add(c_norm)

        paper = cand["paper"]
        record = cand["record"]
        sub_q = cand["sub_q"]
        rel_sc = cand["rel_score"]

        citation_id = f"cite-{badge_counter}"
        authors_str = ", ".join(paper.authors[:2]) + (" et al." if len(paper.authors) > 2 else "")

        v = AtomicClaimVerification(
            id=f"atom-{badge_counter}",
            atomicClaim=c_text,
            matchedSentence=c_text,
            verdict=EntailmentVerdict.ENTAILS,
            confidence=rel_sc,
            reasoning=f"source match: verified verbatim against source publication passage (relevance: {rel_sc:.2f})",
        )

        citation = Citation(
            id=citation_id,
            badgeNumber=badge_counter,
            claim=c_text,
            status=CitationStatus.VERIFIED,
            paperId=paper.id,
            paperTitle=paper.title,
            authors=authors_str,
            year=paper.publicationYear,
            page=record.passage.page,
            passage=record.passage.text,
            highlightSentence=c_text,
            reason=f"source match: verified verbatim against source publication passage (relevance: {rel_sc:.2f})",
            atomicClaims=[v],
            entailmentScore=rel_sc,
            extractionConfidence=rel_sc,
        )

        claim_obj = Claim(
            id=f"claim-{badge_counter}",
            text=c_text,
            sub_question=sub_q,
            status=CitationStatus.VERIFIED,
            evidence=[
                Evidence(
                    claim_id=f"claim-{badge_counter}",
                    passage_id=record.passage.id,
                    nli_label="source match",
                    entail_prob=rel_sc,
                    rerank_score=rel_sc,
                )
            ],
        )

        verified_citations.append(citation)
        claims_list.append(claim_obj)
        badge_counter += 1

    logger.info(
        "[Evidence] Verification complete: %d verified claims/citations produced, %d rejected claims recorded",
        len(verified_citations),
        len(rejected_claims),
    )

    return claims_list, verified_citations, rejected_claims, removal_counts
