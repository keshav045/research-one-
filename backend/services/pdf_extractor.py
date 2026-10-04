"""
PDF Extraction Service — PyMuPDF
=================================
Downloads candidate paper PDFs with caching and size limits.
Extracts clean full-text per page, strips headers/footers and references sections,
and chunks text into 3-sentence overlapping windows tagged with page numbers and sections.
Falls back to abstract when PDF download/parse is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import httpx

from ..config import settings
from ..models.schemas import Paper, PaperPassage

logger = logging.getLogger(__name__)

PDF_CACHE = Path(settings.PDF_CACHE_DIR)
PDF_CACHE.mkdir(parents=True, exist_ok=True)

MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def _cache_path(url: str) -> Path:
    digest = hashlib.md5(url.encode()).hexdigest()
    return PDF_CACHE / f"{digest}.pdf"


async def _download_pdf(url: str) -> Optional[bytes]:
    """Download a PDF, with local cache to avoid re-fetching (max 25MB, timeout 30s)."""
    cached = _cache_path(url)
    if cached.exists():
        return cached.read_bytes()

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "ResearchLens/2.0 (academic research tool)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            content = resp.content
            if len(content) > MAX_PDF_SIZE_BYTES:
                logger.warning("[PDF] Exceeds 25MB limit (%d bytes): %s", len(content), url)
                return None

            if "pdf" not in resp.headers.get("content-type", "") and not content.startswith(b"%PDF"):
                logger.warning("[PDF] Response is not valid PDF for %s", url)
                return None

            cached.write_bytes(content)
            logger.info("[PDF] Downloaded and cached %d bytes: %s", len(content), url)
            return content
    except Exception as exc:
        logger.warning("[PDF] Download failed for %s: %s", url, exc)
        return None


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.replace("\n", " ").strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 15]
    return sentences


def _extract_passages_from_pdf_bytes(paper_id: str, pdf_bytes: bytes) -> list[PaperPassage]:
    """
    Extract structured passages from PDF bytes:
      - PyMuPDF per-page extraction
      - Removes repeating header/footer lines across pages
      - Cuts references/bibliography section
      - Chunks text into 3-sentence windows with 1-sentence overlap
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = min(len(doc), 30)  # max 30 pages

        # 1. Identify common headers/footers to filter
        first_lines = []
        last_lines = []
        page_texts = []

        for p_no in range(total_pages):
            page = doc[p_no]
            text = page.get_text("text").strip()
            page_texts.append(text)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                first_lines.append(lines[0])
                if len(lines) > 1:
                    last_lines.append(lines[-1])

        # Find lines repeated on more than 30% of pages
        threshold = max(2, int(total_pages * 0.3))
        header_counts = Counter(first_lines)
        footer_counts = Counter(last_lines)
        bad_headers = {line for line, cnt in header_counts.items() if cnt >= threshold}
        bad_footers = {line for line, cnt in footer_counts.items() if cnt >= threshold}

        passages: list[PaperPassage] = []
        current_section = "Introduction"
        passage_seq = 1

        section_re = re.compile(
            r"^(?:\d+\.?\s+)?(Abstract|Introduction|Related Work|Background|Methodology|"
            r"Method|Model Architecture|Experiments?|Results?|Evaluation|Discussion|Conclusion|"
            r"References|Bibliography|Appendix)\b",
            re.IGNORECASE,
        )

        in_references = False

        for page_num, raw_text in enumerate(page_texts, start=1):
            if in_references:
                break

            lines = raw_text.split("\n")
            cleaned_lines = []

            for line in lines:
                l_str = line.strip()
                if not l_str or l_str in bad_headers or l_str in bad_footers:
                    continue

                # Check if we hit References / Bibliography
                if re.match(r"^(?:\d+\.?\s+)?(References|Bibliography)\b", l_str, re.IGNORECASE):
                    in_references = True
                    break

                # Detect section heading
                m = section_re.match(l_str)
                if m and len(l_str) < 80:
                    current_section = m.group(1).title()
                    continue

                # Clean hyphenated line break across consecutive lines (e.g. "en-" + "coder")
                if cleaned_lines and cleaned_lines[-1].endswith("-") and re.search(r"[a-zA-Z]-$", cleaned_lines[-1]) and re.match(r"^[a-zA-Z]", l_str):
                    cleaned_lines[-1] = cleaned_lines[-1][:-1] + l_str
                else:
                    cleaned_lines.append(l_str)

            page_body = " ".join(cleaned_lines).strip()
            # Clean any remaining inline hyphenated line breaks (e.g. "en- coder" -> "encoder")
            page_body = re.sub(r"\b([a-zA-Z]{2,})-\s+([a-z]{2,})\b", r"\1\2", page_body)
            if not page_body:
                continue

            # 2. Chunk text by 3-sentence windows with 1-sentence overlap
            sentences = _split_into_sentences(page_body)
            if not sentences:
                continue

            step = 2  # 3 sentences window, 1 overlap (step by 2)
            for i in range(0, len(sentences), step):
                chunk_sentences = sentences[i : i + 3]
                chunk_text = " ".join(chunk_sentences).strip()
                if len(chunk_text) < 50:
                    continue

                passages.append(
                    PaperPassage(
                        id=f"{paper_id}-p{passage_seq}",
                        paper_id=paper_id,
                        page=page_num,
                        section=current_section,
                        text=chunk_text,
                        embedding_idx=passage_seq - 1,
                        source_kind="pdf",
                    )
                )
                passage_seq += 1

        return passages

    except Exception as exc:
        logger.warning("[PDF] PyMuPDF parsing failed for %s: %s", paper_id, exc)
        return []


def create_abstract_fallback_passage(paper: Paper) -> list[PaperPassage]:
    """Creates a fallback passage from paper abstract when PDF cannot be retrieved."""
    abstract = (paper.abstract or "").strip()
    if not abstract:
        abstract = f"{paper.title}. Published by {', '.join(paper.authors[:3])} in {paper.publicationYear}."

    return [
        PaperPassage(
            id=f"{paper.id}-p1",
            paper_id=paper.id,
            page=1,
            section="Abstract",
            text=abstract,
            embedding_idx=0,
            source_kind="abstract",
        )
    ]


def resolve_paper_pdf_urls(paper: Paper) -> list[str]:
    """
    Resolves PDF candidates for a paper in exact priority order:
    1. paper.pdfUrl
    2. https://arxiv.org/pdf/<arXiv ID> if an arXiv ID is known
    3. S2 openAccessPdf URL
    4. OpenAlex open-access URL
    """
    candidates = []
    # 1. paper.pdfUrl
    if paper.pdfUrl and isinstance(paper.pdfUrl, str) and paper.pdfUrl.startswith("http"):
        candidates.append(paper.pdfUrl.strip())

    # 2. arXiv URL from id or doi
    arxiv_id = None
    doi_val = (getattr(paper, "doi", "") or "").lower()
    if "10.48550/arxiv." in doi_val:
        raw = doi_val.split("10.48550/arxiv.")[-1].strip()
        arxiv_id = re.sub(r"v\d+$", "", raw)
    elif (getattr(paper, "id", "") or "").startswith("arxiv-"):
        raw = paper.id[len("arxiv-"):].strip()
        arxiv_id = re.sub(r"v\d+$", "", raw)

    if arxiv_id:
        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
        if arxiv_url not in candidates:
            candidates.append(arxiv_url)

    # 3. S2 openAccessPdf URL or other OA URL stored in paper attributes
    oa_url = getattr(paper, "openAccessPdfUrl", None) or getattr(paper, "oa_url", None)
    if oa_url and isinstance(oa_url, str) and oa_url.startswith("http"):
        oa_clean = oa_url.strip()
        if oa_clean not in candidates:
            candidates.append(oa_clean)

    return candidates


async def extract_paper_passages(paper: Paper) -> list[PaperPassage]:
    """
    Downloads and extracts passages for a single paper with PyMuPDF.
    Attempts all resolved PDF URLs in priority order before falling back to abstract.
    Guarantees that a paper ALWAYS has at least 1 passage.
    """
    pdf_urls = resolve_paper_pdf_urls(paper)
    for url in pdf_urls:
        try:
            pdf_bytes = await _download_pdf(url)
            if pdf_bytes:
                passages = _extract_passages_from_pdf_bytes(paper.id, pdf_bytes)
                if passages:
                    paper.passages = passages
                    paper.passages_json = json.dumps([p.model_dump() for p in passages])
                    paper.pdfUrl = url
                    logger.info("[PDF] Extracted %d passages from %s via %s", len(passages), paper.title[:50], url)
                    return passages
        except Exception as exc:
            logger.debug("[PDF] Failed downloading %s: %s", url, exc)

    # Fallback to abstract if all PDF attempts fail
    passages = create_abstract_fallback_passage(paper)
    paper.passages = passages
    paper.passages_json = json.dumps([p.model_dump() for p in passages])
    is_anchor = getattr(paper, "is_anchor", False)
    if is_anchor:
        logger.warning("[Anchor-PDF] Using abstract only fallback for anchor '%s' (warning: full PDF text unavailable)", paper.title)
    else:
        logger.info("[PDF] Using abstract fallback (1 passage) for '%s'", paper.title[:50])
    return passages


async def extract_passages_for_all_papers(
    papers: list[Paper], max_workers: int = 4
) -> list[Paper]:
    """
    Extracts passages concurrently across candidate papers using a worker pool.
    """
    sem = asyncio.Semaphore(max_workers)

    async def _worker(p: Paper) -> Paper:
        async with sem:
            await extract_paper_passages(p)
            return p

    tasks = [_worker(p) for p in papers]
    await asyncio.gather(*tasks, return_exceptions=True)

    total_passages = sum(len(p.passages or []) for p in papers)
    logger.info("[PDF] Batch extraction complete: %d total passages across %d papers", total_passages, len(papers))
    return papers


extract_papers_batch = extract_passages_for_all_papers

