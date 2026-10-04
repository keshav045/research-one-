"""
Paper Retrieval Service
=======================
Fetches real academic papers from:
  1. arXiv API — free, throttled at 3s between calls, quoted phrase search
  2. Semantic Scholar API — with tenacity retry (4 attempts, exponential backoff)
  3. SQLite Search Cache — 24-hour TTL to save API rate limits
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import settings
from ..models.schemas import Paper, ResearchDepth, ResearchSource
from .query_planner import QueryPlan, plan_research_queries
from .ranker import validate_title_guesses

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "paperId,title,authors,year,publicationVenue,externalIds,abstract,openAccessPdf,citationCount,influentialCitationCount"

DB_PATH = Path(settings.DATABASE_URL.replace("sqlite:///", ""))


# ─── Query Extraction Compatibility Helpers ────────────────────────────────────

def _simple_keyword_query(question: str) -> str:
    """Strips stop words and extracts whole-word tokens without hardcoded filler."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "what", "which", "who", "whom", "whose", "when", "where", "how",
        "and", "but", "or", "nor", "for", "yet", "so", "of", "in", "on",
        "at", "by", "from", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "up",
        "down", "than", "that", "this", "these", "those", "i", "you", "he",
        "she", "it", "we", "they", "them", "their", "its", "if",
    }
    tokens = re.findall(r"[a-zA-Z0-9][-a-zA-Z0-9]*", question.lower())
    meaningful = [t for t in tokens if t not in stop_words]
    return " ".join(meaningful) if meaningful else " ".join(tokens)


async def generate_search_queries(question: str) -> list[str]:
    """Generates search queries for a research question (LLM or keyword fallback)."""
    is_configured = getattr(settings, "is_gemini_configured", False)
    if not is_configured and getattr(settings, "GEMINI_API_KEY", "") and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
        is_configured = True

    if is_configured:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(getattr(settings, "GEMINI_MODEL", "gemini-1.5-flash"))
            resp = model.generate_content(
                f"Generate 3 academic search queries for this question: '{question}'. Return ONLY a JSON array of strings."
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            queries = json.loads(raw)
            if isinstance(queries, list):
                valid = [str(q).strip() for q in queries if str(q).strip()]
                if valid:
                    return valid[:3]
        except Exception as exc:
            logger.warning("[generate_search_queries] LLM query generation failed (%s), using keyword fallback", exc)

    kw = _simple_keyword_query(question)
    return [kw] if kw else [question]



# ─── SQLite 24-Hour Search Cache ──────────────────────────────────────────────

def _get_cache_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_cache (
            source TEXT,
            query_key TEXT,
            response_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source, query_key)
        )
        """
    )
    conn.commit()
    return conn


def _get_cached_response(source: str, query: str) -> Optional[list[dict]]:
    try:
        conn = _get_cache_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT response_json, created_at FROM search_cache WHERE source = ? AND query_key = ?",
            (source, query.lower().strip())
        )
        row = cur.fetchone()
        conn.close()
        if row:
            raw_json, created_at = row
            # Check 24-hour TTL
            try:
                created_dt = datetime.fromisoformat(created_at.replace(" ", "T"))
                if datetime.utcnow() - created_dt < timedelta(hours=24):
                    logger.info("[Cache] Hit for %s: '%s'", source, query[:50])
                    return json.loads(raw_json)
            except Exception:
                return json.loads(raw_json)
    except Exception as exc:
        logger.debug("[Cache] Read error: %s", exc)
    return None


def _set_cached_response(source: str, query: str, data: list[dict]) -> None:
    try:
        conn = _get_cache_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO search_cache (source, query_key, response_json, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (source, query.lower().strip(), json.dumps(data))
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("[Cache] Write error: %s", exc)


# ─── Global Locks & Error Tracking ────────────────────────────────────────────

_arxiv_lock = asyncio.Lock()
_s2_lock = asyncio.Lock()
_last_arxiv_call_time = 0.0
_last_s2_call_time = 0.0
_retrieval_errors: list[str] = []
_s2_call_records: list[dict] = []


def _record_retrieval_error(err: str) -> None:
    if err and err not in _retrieval_errors:
        _retrieval_errors.append(err)


def get_and_clear_retrieval_errors() -> list[str]:
    global _retrieval_errors
    errors = list(_retrieval_errors)
    _retrieval_errors = []
    return errors


def _record_s2_call(query: str, status_code: int, papers_count: int = 0, error: Optional[str] = None, duration_ms: int = 0) -> None:
    _s2_call_records.append({
        "query": query,
        "status_code": status_code,
        "papers_count": papers_count,
        "error": error,
        "duration_ms": duration_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def get_and_clear_s2_call_records() -> list[dict]:
    global _s2_call_records
    records = list(_s2_call_records)
    _s2_call_records = []
    return records


async def _throttle_arxiv():
    """Ensure at least 3 seconds between successive arXiv API requests."""
    global _last_arxiv_call_time
    now = time.time()
    elapsed = now - _last_arxiv_call_time
    if elapsed < 3.0:
        await asyncio.sleep(3.0 - elapsed)
    _last_arxiv_call_time = time.time()


async def _throttle_s2():
    """Ensure at least 1.2 seconds between successive Semantic Scholar requests."""
    global _last_s2_call_time
    now = time.time()
    elapsed = now - _last_s2_call_time
    if elapsed < 1.2:
        await asyncio.sleep(1.2 - elapsed)
    _last_s2_call_time = time.time()


def _parse_arxiv_xml(xml_text: str) -> list[Paper]:
    """Parse arXiv Atom XML feed into Paper objects."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    papers: list[Paper] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("[arXiv] XML parse failed: %s", exc)
        return papers

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        if title_el is None or not title_el.text:
            continue
        title = " ".join(title_el.text.split())

        id_el = entry.find("atom:id", ns)
        raw_id = id_el.text.strip() if id_el is not None and id_el.text else ""
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

        summary_el = entry.find("atom:summary", ns)
        abstract = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""

        authors: list[str] = []
        for author in entry.findall("atom:author", ns):
            name_el = author.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        published_el = entry.find("atom:published", ns)
        pub_year = 2024
        if published_el is not None and published_el.text:
            try:
                pub_year = int(published_el.text[:4])
            except ValueError:
                pass

        journal_el = entry.find("arxiv:journal_ref", ns)
        journal = journal_el.text.strip() if journal_el is not None and journal_el.text else "arXiv"

        doi_el = entry.find("arxiv:doi", ns)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else (f"10.48550/arXiv.{arxiv_id}" if arxiv_id else "")

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None

        clean_id = f"arxiv-{arxiv_id}" if arxiv_id else f"paper-{len(papers)+1}"
        papers.append(
            Paper(
                id=clean_id,
                title=title,
                authors=authors or ["Unknown"],
                publicationYear=pub_year,
                journalConference=journal,
                doi=doi,
                source="arXiv",
                abstract=abstract,
                pdfUrl=pdf_url,
            )
        )

    return papers


async def _fetch_arxiv_query(search_expr: str, max_results: int) -> list[Paper]:
    """Fetch from arXiv using search expression with cache and throttle."""
    cached = _get_cached_response("arxiv", search_expr)
    if cached is not None:
        return [Paper(**item) for item in cached]

    await _throttle_arxiv()

    params = {
        "search_query": search_expr,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(ARXIV_API, params=params)
            if resp.status_code == 200:
                papers = _parse_arxiv_xml(resp.text)
                logger.info("[arXiv] Query '%s' -> %d papers", search_expr[:50], len(papers))
                _set_cached_response("arxiv", search_expr, [p.model_dump() for p in papers])
                return papers
            else:
                msg = f"arXiv HTTP {resp.status_code} for query '{search_expr[:50]}'"
                logger.warning("[arXiv] %s", msg)
                _record_retrieval_error(msg)
    except Exception as exc:
        msg = f"arXiv error querying '{search_expr[:50]}': {exc}"
        logger.warning("[arXiv] %s", msg)
        _record_retrieval_error(msg)

    return []


# ─── Semantic Scholar Fetcher with Tenacity ───────────────────────────────────

class RateLimitException(Exception):
    """Raised on HTTP 429 to trigger tenacity exponential backoff."""
    pass


class S2ServerError(Exception):
    """Raised on HTTP 5xx to trigger tenacity exponential backoff."""
    pass


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=2, max=10),
    retry=retry_if_exception_type((RateLimitException, S2ServerError, httpx.RequestError)),
    reraise=False,
)
async def _fetch_s2_with_retry(query: str, limit: int, headers: dict) -> list[Paper]:
    """Fetch from Semantic Scholar Graph API with automatic 429 and 5xx retries, >= 1.2s spacing, and per-call tracking."""
    async with _s2_lock:
        await _throttle_s2()
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                params = {
                    "query": query,
                    "limit": limit,
                    "fields": S2_FIELDS,
                    "openAccessPdf": "",
                }
                resp = await client.get(S2_SEARCH_API, params=params, headers=headers)
                dur_ms = int((time.time() - t0) * 1000)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_sec = int(retry_after) if retry_after and retry_after.isdigit() else 3
                    msg = f"Semantic Scholar HTTP 429 rate limit for query '{query[:40]}'"
                    logger.warning("[S2] %s. Retrying after %ds", msg, wait_sec)
                    _record_retrieval_error(msg)
                    _record_s2_call(query, 429, 0, error=msg, duration_ms=dur_ms)
                    await asyncio.sleep(wait_sec)
                    raise RateLimitException("Semantic Scholar 429 rate limit")

                if resp.status_code >= 500:
                    msg = f"Semantic Scholar HTTP {resp.status_code} server error for query '{query[:40]}'"
                    logger.warning("[S2] %s. Retrying...", msg)
                    _record_retrieval_error(msg)
                    _record_s2_call(query, resp.status_code, 0, error=msg, duration_ms=dur_ms)
                    raise S2ServerError(msg)

                if resp.status_code != 200:
                    msg = f"Semantic Scholar HTTP {resp.status_code} for query '{query[:40]}'"
                    logger.warning("[S2] %s: %s", msg, resp.text[:100])
                    _record_retrieval_error(msg)
                    _record_s2_call(query, resp.status_code, 0, error=msg, duration_ms=dur_ms)
                    return []

                data = resp.json()
                raw_papers = data.get("data", [])
                papers: list[Paper] = []

                for p_data in raw_papers:
                    title = p_data.get("title", "").strip()
                    if not title:
                        continue

                    abstract = (p_data.get("abstract") or "").strip()
                    ext_ids = p_data.get("externalIds") or {}
                    doi = ext_ids.get("DOI", "")
                    arxiv_id = ext_ids.get("ArXiv", "")

                    authors = [
                        a.get("name", "").strip()
                        for a in p_data.get("authors", [])
                        if a.get("name", "").strip()
                    ]

                    pdf_url = (p_data.get("openAccessPdf") or {}).get("url")
                    if not pdf_url and arxiv_id:
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

                    clean_id = f"s2-{p_data.get('paperId', '')[:12]}"
                    citation_count = p_data.get("citationCount") or 0

                    paper = Paper(
                        id=clean_id,
                        title=title,
                        authors=authors or ["Unknown"],
                        publicationYear=p_data.get("year") or 2024,
                        journalConference=p_data.get("publicationVenue", {}).get("name", "Semantic Scholar") if isinstance(p_data.get("publicationVenue"), dict) else "Semantic Scholar",
                        doi=doi or (f"10.48550/arXiv.{arxiv_id}" if arxiv_id else ""),
                        source="Semantic Scholar",
                        abstract=abstract,
                        pdfUrl=pdf_url,
                        citationCount=citation_count,
                        evidenceCount=0,
                    )
                    papers.append(paper)

                _record_s2_call(query, 200, len(papers), error=None, duration_ms=dur_ms)
                return papers
        except (RateLimitException, S2ServerError):
            raise
        except Exception as exc:
            dur_ms = int((time.time() - t0) * 1000)
            msg = f"Semantic Scholar network error for query '{query[:40]}': {exc}"
            logger.warning("[S2] %s", msg)
            _record_retrieval_error(msg)
            _record_s2_call(query, 0, 0, error=msg, duration_ms=dur_ms)
            raise


async def _fetch_semantic_scholar_query(query: str, limit: int) -> list[Paper]:
    """Cached wrapper for Semantic Scholar queries."""
    cached = _get_cached_response("semantic_scholar", query)
    if cached is not None:
        return [Paper(**item) for item in cached]

    headers = {"User-Agent": "ResearchLens/2.0 (academic research tool)"}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

    try:
        papers = await _fetch_s2_with_retry(query, limit, headers)
        if papers:
            logger.info("[S2] Query '%s' -> %d papers", query[:50], len(papers))
            _set_cached_response("semantic_scholar", query, [p.model_dump() for p in papers])
            return papers
    except Exception as exc:
        msg = f"Semantic Scholar error querying '{query[:50]}': {exc}"
        logger.warning("[S2] %s", msg)
        _record_retrieval_error(msg)

    return []
 

def _extract_s2_batch_id(paper: Paper) -> Optional[str]:
    """
    Extracts an arXiv ID or DOI suitable for Semantic Scholar's /graph/v1/paper/batch endpoint.
    arXiv IDs are normalized to remove version suffixes (e.g. '1706.03762v7' -> 'ARXIV:1706.03762').
    DOIs are prefixed with 'DOI:'.
    """
    if paper.doi:
        doi_clean = paper.doi.strip()
        if "10.48550/arxiv." in doi_clean.lower():
            raw_id = doi_clean.lower().split("10.48550/arxiv.")[-1].strip()
            clean_arxiv = re.sub(r"v\d+$", "", raw_id)
            return f"ARXIV:{clean_arxiv}"
        if not doi_clean.lower().startswith("10.48550/"):
            return f"DOI:{doi_clean}"

    if paper.id.startswith("arxiv-"):
        raw_id = paper.id[len("arxiv-"):].strip()
        clean_arxiv = re.sub(r"v\d+$", "", raw_id)
        return f"ARXIV:{clean_arxiv}"

    if paper.pdfUrl and "arxiv.org/pdf/" in paper.pdfUrl:
        raw_id = paper.pdfUrl.split("arxiv.org/pdf/")[-1].replace(".pdf", "").strip()
        clean_arxiv = re.sub(r"v\d+$", "", raw_id)
        return f"ARXIV:{clean_arxiv}"

    return None


async def enrich_papers_with_s2(papers: list[Paper]) -> list[Paper]:
    """
    Enriches candidate papers after deduplication using Semantic Scholar batch endpoint:
    POST /graph/v1/paper/batch with IDs like 'ARXIV:1706.03762' without version suffix or 'DOI:...',
    up to 500 per call, requesting fields citationCount,venue,year,authors,publicationVenue.
    Merges citationCount, venue, year, and authors into the existing record.
    Logs how many records were enriched.
    """
    if not papers:
        return papers

    id_to_papers: dict[str, list[Paper]] = {}
    for p in papers:
        s2_id = _extract_s2_batch_id(p)
        if s2_id:
            id_to_papers.setdefault(s2_id, []).append(p)

    if not id_to_papers:
        logger.info("[Enrichment] No candidate papers with arXiv ID or DOI found for S2 batch enrichment.")
        return papers

    all_ids = list(id_to_papers.keys())
    enriched_paper_ids: set[str] = set()

    headers = {"User-Agent": "ResearchLens/2.0 (academic research tool)"}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

    batch_size = 500
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        async with _s2_lock:
            await _throttle_s2()
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=25.0) as client:
                    params = {"fields": "citationCount,venue,year,authors,publicationVenue"}
                    resp = await client.post(
                        S2_BATCH_API,
                        params=params,
                        json={"ids": batch_ids},
                        headers=headers,
                    )
                    dur_ms = int((time.time() - t0) * 1000)

                    if resp.status_code == 429:
                        msg = f"Semantic Scholar batch rate limited (HTTP 429 for {len(batch_ids)} ids)"
                        logger.warning("[S2-Batch] %s", msg)
                        _record_s2_call(f"batch:{len(batch_ids)}", 429, 0, error=msg, duration_ms=dur_ms)
                        retry_after = resp.headers.get("Retry-After")
                        wait_sec = int(retry_after) if retry_after and retry_after.isdigit() else 3
                        await asyncio.sleep(wait_sec)
                        await _throttle_s2()
                        resp = await client.post(
                            S2_BATCH_API,
                            params=params,
                            json={"ids": batch_ids},
                            headers=headers,
                        )
                        dur_ms = int((time.time() - t0) * 1000)

                    if resp.status_code != 200:
                        msg = f"Semantic Scholar batch returned HTTP {resp.status_code}"
                        logger.warning("[S2-Batch] %s: %s", msg, resp.text[:120])
                        _record_s2_call(f"batch:{len(batch_ids)}", resp.status_code, 0, error=msg, duration_ms=dur_ms)
                        continue

                    batch_data = resp.json()
                    _record_s2_call(
                        f"batch:{len(batch_ids)}",
                        200,
                        len(batch_data) if isinstance(batch_data, list) else 0,
                        error=None,
                        duration_ms=dur_ms,
                    )

                    if isinstance(batch_data, list):
                        for s2_id, item in zip(batch_ids, batch_data):
                            if not item or not isinstance(item, dict):
                                continue
                            c_count = item.get("citationCount")
                            venue = item.get("venue") or (item.get("publicationVenue") or {}).get("name")
                            year = item.get("year")
                            authors_data = item.get("authors")

                            for target_paper in id_to_papers.get(s2_id, []):
                                modified = False
                                if c_count is not None and c_count > target_paper.citationCount:
                                    target_paper.citationCount = c_count
                                    modified = True
                                if venue and isinstance(venue, str) and venue.strip():
                                    v_clean = venue.strip()
                                    if target_paper.journalConference in ("arXiv", "Semantic Scholar", "unknown", "") or not target_paper.journalConference:
                                        target_paper.journalConference = v_clean
                                        modified = True
                                if year and isinstance(year, int) and (not target_paper.publicationYear or target_paper.publicationYear == 2024):
                                    target_paper.publicationYear = year
                                    modified = True
                                if authors_data and isinstance(authors_data, list):
                                    new_names = [a.get("name", "").strip() for a in authors_data if isinstance(a, dict) and a.get("name")]
                                    if new_names and (target_paper.authors == ["Unknown"] or not target_paper.authors):
                                        target_paper.authors = new_names
                                        modified = True
                                if modified:
                                    enriched_paper_ids.add(target_paper.id)

            except Exception as exc:
                dur_ms = int((time.time() - t0) * 1000)
                msg = f"Semantic Scholar batch exception: {exc}"
                logger.warning("[S2-Batch] %s", msg)
                _record_s2_call(f"batch:{len(batch_ids)}", 0, 0, error=msg, duration_ms=dur_ms)

    logger.info(
        "[Enrichment] S2 batch enriched %d / %d records with citation counts and metadata",
        len(enriched_paper_ids),
        len(papers),
    )
    return papers


# ─── Main Retrieval Entry Point ───────────────────────────────────────────────

async def retrieve_candidate_papers(
    question: str,
    depth: ResearchDepth,
    sources: list[ResearchSource],
    plan: Optional[QueryPlan] = None,
) -> tuple[list[Paper], QueryPlan]:
    """
    Retrieves candidates from arXiv and Semantic Scholar across planned queries.
    Returns merged candidate list and the query plan used.
    """
    if plan is None:
        plan = await plan_research_queries(question)

    depth_val = depth.value if hasattr(depth, "value") else str(depth)
    target_count = settings.DEPTH_COUNTS.get(depth_val, 12)
    fetch_limit = target_count + 6

    q_type = plan.get("question_type", "literature_review") if plan else "literature_review"
    s2_limit = 100 if q_type == "factual_lookup" else fetch_limit

    tasks = []

    valid_guesses = validate_title_guesses(plan.get("title_guesses", []))

    # 1. arXiv queries
    if ResearchSource.ARXIV in sources or "arXiv" in [s.value if hasattr(s, "value") else s for s in sources]:
        # Search valid title guesses first with ti:"..."
        for title in valid_guesses:
            clean_title = re.sub(r'["\']', '', title).strip()
            if clean_title:
                tasks.append(_fetch_arxiv_query(f'ti:"{clean_title}"', fetch_limit))

        # Search planned queries
        for q in plan.get("queries", []):
            phrase = re.sub(r'["\']', '', q).strip()
            if phrase:
                tasks.append(_fetch_arxiv_query(f'all:"{phrase}"', fetch_limit))

    # 2. Semantic Scholar queries
    if ResearchSource.SEMANTIC_SCHOLAR in sources or "Semantic Scholar" in [s.value if hasattr(s, "value") else s for s in sources]:
        for q in plan.get("queries", [])[:2]:
            tasks.append(_fetch_semantic_scholar_query(q, s2_limit))
        for title in valid_guesses:
            tasks.append(_fetch_semantic_scholar_query(title, s2_limit))

    # Run tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            candidates.extend(r)

    logger.info("[Retrieval] Raw candidates retrieved across %d calls: %d", len(tasks), len(candidates))
    return candidates, plan


async def retrieve_papers(
    question: str,
    depth: ResearchDepth,
    sources: list[ResearchSource],
    api_key_s2: Optional[str] = None,
    custom_queries: Optional[list[str]] = None,
    title_guesses: Optional[list[str]] = None,
    question_type: Optional[str] = None,
) -> list[Paper]:
    """Compatible wrapper returning list[Paper] directly."""
    plan: Optional[QueryPlan] = None
    if custom_queries or title_guesses or question_type:
        plan = {
            "question_type": question_type or "literature_review",
            "queries": custom_queries or [question],
            "title_guesses": title_guesses or [],
            "sub_questions": [question],
        }
    candidates, _ = await retrieve_candidate_papers(question, depth, sources, plan=plan)
    return candidates

