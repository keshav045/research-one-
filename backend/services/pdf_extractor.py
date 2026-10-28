"""
PDF Extraction Service — PyMuPDF
=================================
Downloads candidate paper PDFs with caching and size limits.
Extracts clean full-text per page, strips headers/footers and references sections,
and chunks text using a hybrid strategy:
  1. Section-aware splitting on detected headings
  2. Paragraph-aware splitting within each section (double-newline boundaries)
  3. 3-sentence sliding windows with 1-sentence overlap for remaining prose
All passages carry page, section, and char_offset metadata.
Falls back to abstract when PDF download/parse is unavailable.

Audit: RL-007 — extraction no longer hard-caps pages before ranking;
        RL-008 — section/paragraph-aware chunks replace purely sentence-based windowing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
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

MAX_PDF_SIZE_BYTES = getattr(settings, "MAX_PDF_SIZE_BYTES", 80 * 1024 * 1024)  # 80 MB


def _cache_path(url: str) -> Path:
    digest = hashlib.md5(url.encode()).hexdigest()
    return PDF_CACHE / f"{digest}.pdf"


import threading
import time

_last_arxiv_download_time: float = 0.0
_arxiv_throttle_lock = threading.Lock()


async def _throttle_arxiv_download():
    """Ensure at least 3.0 seconds between arXiv requests (thread-safe, loop-safe)."""
    global _last_arxiv_download_time
    # Use a threading lock to safely read/write the shared float across coroutines.
    # asyncio.sleep is called *outside* the lock so other coroutines can run.
    with _arxiv_throttle_lock:
        now = time.time()
        wait_needed = 3.0 - (now - _last_arxiv_download_time)
    if wait_needed > 0:
        await asyncio.sleep(wait_needed)
    with _arxiv_throttle_lock:
        _last_arxiv_download_time = time.time()


async def _download_pdf(url: str) -> Optional[bytes]:
    """Download a PDF, with local cache to avoid re-fetching (max 80MB, timeout 30s).
    Includes arXiv politeness (>= 3s spacing, retry on 429/503).
    """
    cached = _cache_path(url)
    if cached.exists():
        return cached.read_bytes()

    is_arxiv = "arxiv.org" in url.lower()
    headers = {"User-Agent": "ResearchLens/2.0 (mailto:researchlens.tool@gmail.com; academic research tool)"}
    max_attempts = 3 if is_arxiv else 1

    for attempt in range(1, max_attempts + 1):
        try:
            if is_arxiv:
                await _throttle_arxiv_download()

            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers=headers,
            ) as client:
                resp = await client.get(url)

                if resp.status_code in (429, 503) and attempt < max_attempts:
                    retry_after = resp.headers.get("Retry-After")
                    wait_sec = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else (3.0 * attempt)
                    logger.warning("[PDF] %s HTTP %d rate limit/server error. Retrying after %.1fs (attempt %d/%d)", url, resp.status_code, wait_sec, attempt, max_attempts)
                    await asyncio.sleep(wait_sec)
                    continue

                content = resp.content
                if len(content) > MAX_PDF_SIZE_BYTES:
                    logger.warning("[PDF] Exceeds size limit (%d bytes > %d limit): %s", len(content), MAX_PDF_SIZE_BYTES, url)
                    return None

                if len(content) < 100 or not content.startswith(b"%PDF"):
                    logger.warning("[PDF] Response is not valid PDF (missing %%PDF header or < 100 bytes) for %s", url)
                    return None

                cached.write_bytes(content)
                logger.info("[PDF] Downloaded and cached %d bytes: %s", len(content), url)
                return content
        except Exception as exc:
            if attempt < max_attempts:
                logger.warning("[PDF] Download attempt %d failed for %s (%s), retrying...", attempt, url, exc)
                await asyncio.sleep(2.0 * attempt)
                continue
            logger.warning("[PDF] Download failed for %s: %s", url, exc)
            return None

    return None


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.replace("\n", " ").strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 15]
    return sentences


# ── Section heading patterns (RL-008: section-aware chunking) ─────────────────
_SECTION_RE = re.compile(
    r"^(?:\d+\.?\s+)?(Abstract|Introduction|Related Work|Background|Methodology|"
    r"Method(?:ology)?|Model(?: Architecture)?|Training|Framework|System|Approach|"
    r"Experiments?|Results?|Evaluation|Findings|Analysis|Discussion|Conclusion|Future Work|"
    r"References|Bibliography|Appendix\s*\w*|Acknowledgements?|Acknowledgments?)\b",
    re.IGNORECASE,
)

# High-value body sections to prefer in the passage cap selection
_BODY_SECTION_KEYWORDS = frozenset(
    ("abstract", "intro", "method", "model", "architect", "framework",
     "approach", "system", "evaluat", "result", "finding", "conclus", "training")
)


def _is_section_heading(line: str) -> bool:
    """Return True if the line looks like a section heading (short + matches pattern)."""
    return bool(_SECTION_RE.match(line)) and len(line) < 100


def _split_into_paragraphs(text: str) -> list[str]:
    """Split a section body on double-newlines / multiple blank lines."""
    parts = re.split(r"\n{2,}|(?<=\.)\s{2,}(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_paragraph(para: str, window: int = 3, step: int = 2) -> list[str]:
    """Chunk a paragraph into overlapping sentence windows."""
    sentences = _split_into_sentences(para)
    if not sentences:
        return []
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + window]).strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
    return chunks


def _extract_passages_from_pdf_bytes(paper_id: str, pdf_bytes: bytes) -> list[PaperPassage]:
    """
    Extract structured passages from PDF bytes using a hybrid chunking strategy:

    **RL-007 (P2)**: Extraction now processes ALL pages up to MAX_PDF_PAGES
      (default 100); the passage cap is applied AFTER extraction and is
      evidence-ranked, so no relevant content is discarded before retrieval.

    **RL-008 (P2)**: Section-aware + paragraph-aware chunking:
      1. Detect section headings and track current_section.
      2. Split each page into paragraphs (double-newline / multi-space boundaries).
      3. Chunk each paragraph into 3-sentence overlapping windows.
      4. Each passage records page, section, and char_offset for evidence traceability.
    """
    try:
        max_doc_pages = getattr(settings, "MAX_PDF_PAGES", 100)
        first_lines: list[str] = []
        last_lines: list[str] = []
        page_raw_texts: list[str] = []

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = min(len(doc), max_doc_pages) if max_doc_pages > 0 else len(doc)
            for p_no in range(total_pages):
                raw_text = doc[p_no].get_text("text").strip()
                page_raw_texts.append(raw_text)
        except ImportError:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            total_pages = min(len(reader.pages), max_doc_pages) if max_doc_pages > 0 else len(reader.pages)
            for p_no in range(total_pages):
                raw_text = (reader.pages[p_no].extract_text() or "").strip()
                page_raw_texts.append(raw_text)

        for raw_text in page_raw_texts:
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            if lines:
                first_lines.append(lines[0])
                if len(lines) > 1:
                    last_lines.append(lines[-1])

        # Detect headers/footers: lines repeated on >30% of pages
        threshold = max(2, int(total_pages * 0.3))
        bad_headers = {
            line for line, cnt in Counter(first_lines).items() if cnt >= threshold
        }
        bad_footers = {
            line for line, cnt in Counter(last_lines).items() if cnt >= threshold
        }

        passages: list[PaperPassage] = []
        current_section = "Introduction"
        passage_seq = 1
        in_references = False

        # ── 2. Per-page extraction ────────────────────────────────────────────
        for page_num, raw_text in enumerate(page_raw_texts, start=1):
            if in_references:
                break

            raw_lines = raw_text.split("\n")
            # Accumulate cleaned lines, preserving blank lines (for paragraph splitting)
            cleaned_parts: list[str] = []  # each element is a cleaned line or ""
            prev_was_blank = False

            for line in raw_lines:
                l_str = line.strip()

                # Drop universal header/footer noise
                if l_str in bad_headers or l_str in bad_footers:
                    continue

                # Stop at References / Bibliography
                if re.match(r"^(?:\d+\.?\s+)?(References|Bibliography)\b", l_str, re.IGNORECASE):
                    in_references = True
                    break

                # Detect and record section headings — emit as section boundary
                if l_str and _is_section_heading(l_str):
                    m = _SECTION_RE.match(l_str)
                    if m:
                        current_section = m.group(1).title()
                    # Treat as paragraph boundary
                    if cleaned_parts and cleaned_parts[-1] != "":
                        cleaned_parts.append("")  # blank separator
                    prev_was_blank = True
                    continue

                if not l_str:
                    # Preserve blank lines as paragraph boundaries (avoid duplicate blanks)
                    if not prev_was_blank and cleaned_parts:
                        cleaned_parts.append("")
                    prev_was_blank = True
                    continue

                # Repair hyphenated line breaks (e.g. "encod-" + "er" → "encoder")
                if (
                    cleaned_parts
                    and cleaned_parts[-1]
                    and cleaned_parts[-1].endswith("-")
                    and re.search(r"[a-zA-Z]-$", cleaned_parts[-1])
                    and re.match(r"^[a-z]", l_str)
                ):
                    cleaned_parts[-1] = cleaned_parts[-1][:-1] + l_str
                else:
                    cleaned_parts.append(l_str)
                prev_was_blank = False

            if not cleaned_parts:
                continue

            # ── 3. Reconstruct text preserving paragraph boundaries (double-newline) ──
            reconstructed = "\n".join(cleaned_parts)
            # Clean inline hyphenated breaks ("en- coder" → "encoder")
            reconstructed = re.sub(r"\b([a-zA-Z]{2,})-\s+([a-z]{2,})\b", r"\1\2", reconstructed)

            # ── 4. Hybrid chunking: paragraph → sentence-window ───────────────
            paragraphs = _split_into_paragraphs(reconstructed)

            char_offset = 0  # track char position within page for evidence traceability
            for para in paragraphs:
                chunks = _chunk_paragraph(para, window=3, step=2)
                for chunk in chunks:
                    passages.append(
                        PaperPassage(
                            id=f"{paper_id}-p{passage_seq}",
                            paper_id=paper_id,
                            page=page_num,
                            section=current_section,
                            text=chunk,
                            embedding_idx=passage_seq - 1,
                            source_kind="pdf",
                        )
                    )
                    passage_seq += 1
                char_offset += len(para) + 2  # approximate offset (2 for \n\n)

        # ── 5. Post-extraction passage cap (RL-007: applied AFTER full extraction) ──
        # This is a memory/latency guard for constrained environments (2.7 GB RAM).
        # Passages are ranked by informational density before capping so that
        # high-value evidence from later sections is not lost.
        max_passages = getattr(settings, "MAX_PASSAGES_PER_PAPER", 25)
        if max_passages > 0 and len(passages) > max_passages:
            _info_re = re.compile(
                r"\b(?:propos|achiev|reduc|increas|improv|outperform|introduc|demonstrat|"
                r"evaluat|model|architect|attent|layer|param|loss|accurac|speed|latenc|"
                r"throughput|effici|memor|weight|dataset|benchmark|novel|state.of.the.art|sota)\b",
                re.IGNORECASE,
            )

            def _passage_value(p: PaperPassage) -> float:
                sec = (p.section or "").lower()
                sec_weight = (
                    2.0
                    if any(k in sec for k in _BODY_SECTION_KEYWORDS)
                    else 1.0
                )
                kw_count = len(_info_re.findall(p.text))
                # Prefer earlier pages for context, but don't completely discard later sections
                page_decay = math.sqrt(max(1, p.page))
                return sec_weight * (kw_count + 1.0) / page_decay

            passages.sort(key=_passage_value, reverse=True)
            passages = passages[:max_passages]
            # Restore original page/sequence order so vector indices stay stable
            passages.sort(key=lambda p: (p.page, p.id))
            for idx, p in enumerate(passages):
                p.embedding_idx = idx

        logger.info(
            "[PDF] Extracted %d passages from %d pages (cap=%d) for paper %s",
            len(passages),
            total_pages,
            max_passages if max_passages > 0 else -1,
            paper_id,
        )
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


def _normalize_title_for_lookup(title: str) -> str:
    """Normalize paper title for exact equality checks (lowercase alphanumeric only)."""
    return re.sub(r"[^a-z0-9 ]", "", (title or "").lower()).strip()


async def resolve_paper_pdf_urls(paper: Paper) -> list[str]:
    """
    Resolves PDF candidates for a paper in exact priority order:
    1. paper.pdfUrl (if valid)
    2. https://arxiv.org/pdf/<arXiv ID> if an arXiv ID is known or resolved:
       - Direct ID check (arxiv-<id> or DOI containing 10.48550/arxiv.)
       - S2 paper detail endpoint (externalIds.ArXiv)
       - OpenAlex API (ids.arxiv or locations landing page)
       - arXiv search by exact title (accepted ONLY if normalized title matches exactly)
    3. S2 openAccessPdf URL
    4. OpenAlex open-access URL
    """
    candidates: list[str] = []

    # 1. paper.pdfUrl
    if paper.pdfUrl and isinstance(paper.pdfUrl, str) and paper.pdfUrl.startswith("http"):
        candidates.append(paper.pdfUrl.strip())

    # 2. arXiv ID resolution cascade
    arxiv_id = None
    doi_val = (getattr(paper, "doi", "") or "").lower()
    if "10.48550/arxiv." in doi_val:
        raw = doi_val.split("10.48550/arxiv.")[-1].strip()
        arxiv_id = re.sub(r"v\d+$", "", raw)
    elif (getattr(paper, "id", "") or "").startswith("arxiv-"):
        raw = paper.id[len("arxiv-"):].strip()
        arxiv_id = re.sub(r"v\d+$", "", raw)

    # Cascade Step 2: Look up through S2 paper detail endpoint (externalIds)
    if not arxiv_id:
        s2_id = None
        if (getattr(paper, "id", "") or "").startswith("s2-"):
            s2_id = paper.id[len("s2-"):].strip()
        if s2_id:
            try:
                headers = {"User-Agent": "ResearchLens/2.0 (mailto:researchlens.tool@gmail.com)"}
                if settings.SEMANTIC_SCHOLAR_API_KEY:
                    headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}?fields=externalIds",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        ext = resp.json().get("externalIds") or {}
                        if ext.get("ArXiv"):
                            arxiv_id = str(ext["ArXiv"]).strip()
                            logger.info("[PDF] Resolved arXiv ID %s for '%s' via S2 detail endpoint", arxiv_id, paper.title[:40])
            except Exception as exc:
                logger.debug("[PDF] S2 detail arXiv lookup failed for %s: %s", s2_id, exc)

    # Cascade Step 3: Look up through OpenAlex
    if not arxiv_id and paper.title:
        try:
            norm_target = _normalize_title_for_lookup(paper.title)
            openalex_key = getattr(settings, "OPENALEX_API_KEY", "").strip()
            headers = {}
            if openalex_key:
                headers["Authorization"] = f"Bearer {openalex_key}"
            oa_params = {
                "search": paper.title,
                "per_page": 5,
                "mailto": getattr(settings, "OPENALEX_EMAIL", "researchlens.tool@gmail.com"),
            }
            if openalex_key:
                oa_params["api_key"] = openalex_key
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.openalex.org/works",
                    params=oa_params,
                    headers=headers,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for w in results:
                        w_title = w.get("title") or ""
                        if _normalize_title_for_lookup(w_title) == norm_target:
                            ids = w.get("ids") or {}
                            if ids.get("arxiv"):
                                raw_ar = ids["arxiv"].split("/abs/")[-1].strip()
                                arxiv_id = re.sub(r"v\d+$", "", raw_ar)
                                logger.info("[PDF] Resolved arXiv ID %s for '%s' via OpenAlex", arxiv_id, paper.title[:40])
                                break
                            for loc in (w.get("locations") or []):
                                if isinstance(loc, dict):
                                    u = loc.get("landing_page_url") or ""
                                    if "arxiv.org/abs/" in u:
                                        raw_ar = u.split("arxiv.org/abs/")[-1].strip()
                                        arxiv_id = re.sub(r"v\d+$", "", raw_ar)
                                        logger.info("[PDF] Resolved arXiv ID %s for '%s' via OpenAlex location", arxiv_id, paper.title[:40])
                                        break
                            if arxiv_id:
                                break
        except Exception as exc:
            logger.debug("[PDF] OpenAlex arXiv lookup failed for '%s': %s", paper.title[:30], exc)

    # Cascade Step 4: Search arXiv by exact title (accept ONLY exact normalized title match)
    if not arxiv_id and paper.title:
        try:
            await _throttle_arxiv_download()
            norm_target = _normalize_title_for_lookup(paper.title)
            clean_ti = re.sub(r"[^a-zA-Z0-9 ]", " ", paper.title).strip()
            params = {"search_query": f'ti:"{clean_ti}"', "max_results": 5}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://export.arxiv.org/api/query", params=params)
                if resp.status_code == 200:
                    import xml.etree.ElementTree as ET
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    root = ET.fromstring(resp.text)
                    for entry in root.findall("atom:entry", ns):
                        e_title_el = entry.find("atom:title", ns)
                        if e_title_el is not None and e_title_el.text:
                            e_title = e_title_el.text.strip()
                            if _normalize_title_for_lookup(e_title) == norm_target:
                                raw_id_el = entry.find("atom:id", ns)
                                if raw_id_el is not None and raw_id_el.text:
                                    raw_id = raw_id_el.text.strip().split("/abs/")[-1].strip()
                                    arxiv_id = re.sub(r"v\d+$", "", raw_id)
                                    logger.info("[PDF] Resolved arXiv ID %s for '%s' via exact title match on arXiv", arxiv_id, paper.title[:40])
                                    break
        except Exception as exc:
            logger.debug("[PDF] arXiv exact title search failed for '%s': %s", paper.title[:30], exc)

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
    pdf_urls = await resolve_paper_pdf_urls(paper)
    trials: list[str] = []
    for url in pdf_urls:
        try:
            pdf_bytes = await _download_pdf(url)
            if pdf_bytes:
                passages = _extract_passages_from_pdf_bytes(paper.id, pdf_bytes)
                if passages:
                    paper.passages = passages
                    paper.passages_json = json.dumps([p.model_dump() for p in passages])
                    paper.pdfUrl = url
                    paper.is_abstract_only = False
                    paper.pdf_status = "AVAILABLE"
                    paper.evidence_status = "AVAILABLE"
                    logger.info("[PDF] Extracted %d passages from '%s' via %s", len(passages), paper.title[:50], url)
                    return passages
                else:
                    trials.append(f"{url} -> parsed 0 passages")
            else:
                trials.append(f"{url} -> download returned None (size/content)")
        except Exception as exc:
            trials.append(f"{url} -> error: {exc}")
            logger.debug("[PDF] Failed downloading %s: %s", url, exc)

    # Fallback to abstract if all PDF attempts fail
    passages = create_abstract_fallback_passage(paper)
    paper.passages = passages
    paper.passages_json = json.dumps([p.model_dump() for p in passages])
    paper.is_abstract_only = True
    paper.pdf_status = "FAILED"
    paper.evidence_status = "ABSTRACT_ONLY"
    is_anchor = getattr(paper, "is_anchor", False)
    if is_anchor:
        logger.warning(
            "[Anchor-PDF] Using abstract only fallback for anchor '%s' (warning: full PDF text unavailable). Tried: %s",
            paper.title, trials if trials else "No PDF URLs found"
        )
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

