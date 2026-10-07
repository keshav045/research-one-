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
import uuid

from ..config import settings
from ..models.schemas import Paper, ResearchDepth, ResearchSource
from .query_planner import QueryPlan, plan_research_queries
from .ranker import validate_title_guesses

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch"
OPENALEX_API = "https://api.openalex.org/works"
S2_FIELDS = "paperId,title,authors,year,publicationVenue,externalIds,abstract,openAccessPdf,citationCount,influentialCitationCount"

DB_PATH = Path(settings.DATABASE_URL.replace("sqlite:///", ""))


# ─── Query Extraction Compatibility Helpers ────────────────────────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "what", "which", "who", "whom", "whose", "when", "where", "how",
    "and", "but", "or", "nor", "for", "yet", "so", "of", "in", "on",
    "at", "by", "from", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "up",
    "down", "than", "that", "this", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "them", "their", "its", "if",
})

GENERIC_QUERY_WORDS = frozenset({
    "paper", "papers", "study", "studies", "approach", "approaches", "method", "methods",
    "model", "models", "architecture", "architectures", "technique", "techniques",
    "analysis", "overview", "survey", "review", "evaluation", "benchmark", "performance",
    "research", "work", "investigation", "literature", "comparison", "comparative",
    "introduction", "introducing", "proposed", "proposal", "recent", "current",
    "state", "art", "sota", "system", "systems", "implementation", "results",
})


def is_generic_query(query: str) -> bool:
    """Returns True if the query contains only stop words or generic academic terms."""
    tokens = re.findall(r"[a-zA-Z0-9][-a-zA-Z0-9]*", query.lower())
    meaningful = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    if not meaningful:
        return True
    return all(t in GENERIC_QUERY_WORDS for t in meaningful)


def _simple_keyword_query(question: str) -> str:
    """Strips stop words and extracts whole-word tokens without hardcoded filler."""
    tokens = re.findall(r"[a-zA-Z0-9][-a-zA-Z0-9]*", question.lower())
    meaningful = [t for t in tokens if t not in _STOP_WORDS]
    return " ".join(meaningful) if meaningful else " ".join(tokens)


async def generate_search_queries(question: str) -> list[str]:
    """Generates search queries for a research question (LLM or planner facet fallback)."""
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
                valid = [str(q).strip() for q in queries if str(q).strip() and not is_generic_query(str(q))]
                if valid:
                    return valid[:3]
        except Exception as exc:
            logger.warning("[generate_search_queries] LLM query generation failed (%s), using keyword fallback", exc)

    # Planner facet-based fallback: topical queries derived from facets, deduplicated
    from .query_planner import extract_question_facets
    facets = extract_question_facets(question)
    kw_query = _simple_keyword_query(question)
    meaningful_q_tokens = [t for t in re.findall(r"[a-zA-Z0-9][-a-zA-Z0-9]*", question.lower()) if t not in _STOP_WORDS and t not in GENERIC_QUERY_WORDS]
    core_topic = " ".join(meaningful_q_tokens[:3]) if meaningful_q_tokens else kw_query

    candidate_queries: list[str] = []
    for f in facets:
        f_clean = re.sub(r"[^a-zA-Z0-9\s-]", "", f).strip()
        if f_clean and not is_generic_query(f_clean):
            if core_topic and core_topic.lower() not in f_clean.lower():
                candidate_queries.append(f"{core_topic} {f_clean}".strip())
            else:
                candidate_queries.append(f_clean)

    if kw_query and not is_generic_query(kw_query):
        candidate_queries.append(kw_query)

    deduped: list[str] = []
    seen = set()
    for q in candidate_queries:
        q_norm = q.lower().strip()
        if q_norm and q_norm not in seen and not is_generic_query(q):
            seen.add(q_norm)
            deduped.append(q)

    if deduped:
        return deduped[:4]
    return [kw_query] if kw_query else [question]



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
    if getattr(settings, "DISABLE_CACHE", False):
        return None
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
                    parsed = json.loads(raw_json)
                    if parsed:
                        logger.info("[Cache] Hit for %s: '%s'", source, query[:50])
                        return parsed
            except Exception:
                parsed = json.loads(raw_json)
                if parsed:
                    return parsed
    except Exception as exc:
        logger.debug("[Cache] Read error: %s", exc)
    return None


def _set_cached_response(source: str, query: str, data: list[dict]) -> None:
    if getattr(settings, "DISABLE_CACHE", False):
        return
    if not data:
        # Never cache failed or empty results
        return
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

_arxiv_lock_inst: Optional[asyncio.Lock] = None
_arxiv_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_arxiv_lock() -> asyncio.Lock:
    global _arxiv_lock_inst, _arxiv_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _arxiv_lock_inst is None or _arxiv_lock_loop != loop:
        _arxiv_lock_inst = asyncio.Lock()
        _arxiv_lock_loop = loop
    return _arxiv_lock_inst


class S2RateLimiter:
    """
    Process-wide mutual-exclusion rate limiter for Semantic Scholar:
    - Exactly ONE S2 call executing at any instant across the entire process.
    - >= 1.25s spacing (with API key) or >= 3.0s (unauthenticated) from the end of the previous call.
    - Honors Retry-After headers by shifting the next available window.
    - Lazy per-event-loop lock ensures thread and multi-loop safety.
    """
    def __init__(self):
        self._lock_inst: Optional[asyncio.Lock] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_call_end = 0.0

    def _get_lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._lock_inst is None or self._loop != loop:
            self._lock_inst = asyncio.Lock()
            self._loop = loop
        return self._lock_inst

    @property
    def _lock(self) -> asyncio.Lock:
        return self._get_lock()

    async def __aenter__(self):
        lock = self._get_lock()
        await lock.acquire()
        is_authenticated = bool(settings.SEMANTIC_SCHOLAR_API_KEY and settings.SEMANTIC_SCHOLAR_API_KEY.strip())
        min_interval = 1.25 if is_authenticated else 3.0
        now = time.time()
        elapsed = now - self._last_call_end
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        return self

    def penalize(self, wait_seconds: float):
        """Pushes the next call window into the future based on Retry-After."""
        now = time.time()
        self._last_call_end = max(self._last_call_end, now + wait_seconds)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        now = time.time()
        if self._last_call_end <= now:
            self._last_call_end = now
        try:
            self._get_lock().release()
        except RuntimeError:
            pass

_s2_limiter = S2RateLimiter()
_last_arxiv_call_time = 0.0
_retrieval_errors: list[str] = []
_s2_call_records: list[dict] = []
_source_status: dict[str, str] = {
    "arXiv": "PENDING",
    "Semantic Scholar": "PENDING",
    "OpenAlex": "PENDING",
}
_s2_429_count: int = 0


def _record_retrieval_error(err: str) -> None:
    if not err:
        return
    # Sanitize: strip query strings, raw HTTP codes, and raw syntax
    clean = re.sub(r"for query ['\"].*?['\"]", "", err, flags=re.IGNORECASE)
    clean = re.sub(r"querying ['\"].*?['\"]", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"HTTP\s*\d{3}", "service error", clean, flags=re.IGNORECASE)
    clean = re.sub(r"all:\".*?\"", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b429\b", "rate limit", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean and clean not in _retrieval_errors:
        _retrieval_errors.append(clean)


def _record_s2_rate_limited() -> None:
    global _s2_429_count
    _s2_429_count += 1
    _source_status["Semantic Scholar"] = "RATE_LIMITED"


def get_s2_rate_limited_count() -> int:
    return _s2_429_count


def set_source_status(source: str, status: str) -> None:
    _source_status[source] = status


def get_source_status() -> dict[str, str]:
    return dict(_source_status)


def summarize_source_status() -> str:
    """Returns a clean user-facing summary line of source statuses without raw HTTP or query text."""
    statuses = get_source_status()
    status_map = {
        "RATE_LIMITED": "rate limited",
        "ERROR": "unavailable",
        "OK": "ok",
        "PARTIAL": "partial results",
        "PENDING": "ok",
        "NO_RESULTS": "no results",
    }
    parts = []
    for src in ["Semantic Scholar", "arXiv", "OpenAlex"]:
        st = statuses.get(src, "PENDING")
        desc = status_map.get(st, st.lower())
        parts.append(f"{src}: {desc}")
    return "; ".join(parts)


def reset_retrieval_session() -> None:
    global _retrieval_errors, _s2_call_records, _source_status, _s2_429_count
    _retrieval_errors = []
    _s2_call_records = []
    _source_status = {
        "arXiv": "PENDING",
        "Semantic Scholar": "PENDING",
        "OpenAlex": "PENDING",
    }
    _s2_429_count = 0


def get_and_clear_retrieval_errors() -> list[str]:
    global _retrieval_errors
    errors = list(_retrieval_errors)
    _retrieval_errors = []
    return errors


def _record_s2_call(query: str, status_code: int, papers_count: int = 0, error: Optional[str] = None, duration_ms: int = 0) -> None:
    logger.info(
        "[S2-Call] HTTP %d for '%s' (%d papers, %dms)%s",
        status_code,
        query[:50],
        papers_count,
        duration_ms,
        f" - {error}" if error else "",
    )
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


def _format_arxiv_query_expr(phrase: str) -> str:
    """Format academic search phrase for arXiv API using boolean all:term AND all:term."""
    cleaned = re.sub(r'["\']', '', phrase).strip()
    if not cleaned:
        return ""
    if ":" in cleaned:
        return cleaned

    lower_p = cleaned.lower()
    if "pagedattention" in lower_p:
        return 'all:PagedAttention'
    if "flashattention" in lower_p:
        return 'all:FlashAttention'
    if "speculative decoding" in lower_p:
        return 'all:speculative AND all:decoding'
    if "kv cache" in lower_p or "kv-cache" in lower_p:
        return 'all:KV AND all:cache'
    if "quantization" in lower_p and ("weight-only" in lower_p or "weight only" in lower_p):
        if any(k in lower_p for k in ["llm", "language model", "transformer"]):
            return 'all:weight AND all:quantization AND all:LLM'
        return 'all:weight AND all:quantization'
    if "quantization" in lower_p:
        if any(k in lower_p for k in ["llm", "language model", "transformer"]):
            return 'all:quantization AND all:LLM'
        return 'all:quantization'
    if "pruning" in lower_p:
        if any(k in lower_p for k in ["llm", "language model", "transformer"]):
            return 'all:pruning AND all:LLM'
        return 'all:pruning'

    stop_words = frozenset(["the", "a", "an", "is", "are", "for", "in", "on", "of", "and", "or", "to", "with", "what", "how", "terms"])
    words = [w for w in re.findall(r'[a-zA-Z0-9\-]+', cleaned) if len(w) > 1 and w.lower() not in stop_words]
    if not words:
        return f'all:{cleaned}'
    if len(words) <= 2:
        return " AND ".join(f'all:{w}' for w in words)
    return " AND ".join(f'all:{w}' for w in words[:3])


async def _throttle_arxiv():
    """Ensure at least 3 seconds between successive arXiv API requests."""
    global _last_arxiv_call_time
    async with _get_arxiv_lock():
        now = time.time()
        elapsed = now - _last_arxiv_call_time
        if elapsed < 3.0:
            await asyncio.sleep(3.0 - elapsed)
        _last_arxiv_call_time = time.time()


async def _throttle_s2():
    """Spacing is automatically handled by _s2_limiter context manager."""
    pass


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
                logger.warning("[arXiv] HTTP %d for query '%s'", resp.status_code, search_expr[:50])
                _record_retrieval_error("arXiv service response error")
    except Exception as exc:
        logger.warning("[arXiv] arXiv error querying '%s': %s", search_expr[:50], exc)
        _record_retrieval_error("arXiv service connection error")

    return []


# ─── Semantic Scholar Fetcher with Tenacity ───────────────────────────────────

class RateLimitException(Exception):
    """Raised on HTTP 429 to trigger tenacity exponential backoff."""
    pass


class S2ServerError(Exception):
    """Raised on HTTP 5xx to trigger tenacity exponential backoff."""
    pass


async def _fetch_s2_with_retry(query: str, limit: int, headers: dict) -> list[Paper]:
    """Fetch from Semantic Scholar Graph API with process-wide rate limiting, Retry-After backoff, and tracking."""
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        async with _s2_limiter:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    params = {
                        "query": query,
                        "limit": limit,
                        "fields": S2_FIELDS,
                    }
                    if getattr(settings, "DEFAULT_FIELDS_OF_STUDY", ""):
                        params["fieldsOfStudy"] = settings.DEFAULT_FIELDS_OF_STUDY.strip()
                    resp = await client.get(S2_SEARCH_API, params=params, headers=headers)
                    dur_ms = int((time.time() - t0) * 1000)

                    if resp.status_code == 429:
                        is_authenticated = bool(settings.SEMANTIC_SCHOLAR_API_KEY and settings.SEMANTIC_SCHOLAR_API_KEY.strip())
                        retry_after = resp.headers.get("Retry-After")
                        wait_sec = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 3.0
                        logger.warning("[S2] Semantic Scholar HTTP 429 rate limit for query '%s'. Retry-After: %.1fs (attempt %d/%d)", query[:40], wait_sec, attempt, max_attempts)
                        _record_retrieval_error("Semantic Scholar rate limit reached")
                        _record_s2_rate_limited()
                        _record_s2_call(query, 429, 0, error="Semantic Scholar rate limit reached", duration_ms=dur_ms)
                        _s2_limiter.penalize(wait_sec)
                        if not is_authenticated:
                            return []
                        await asyncio.sleep(wait_sec)
                        continue

                    if resp.status_code >= 500:
                        logger.warning("[S2] Semantic Scholar HTTP %d server error for query '%s' (attempt %d/%d)", resp.status_code, query[:40], attempt, max_attempts)
                        _record_retrieval_error("Semantic Scholar server error")
                        _record_s2_call(query, resp.status_code, 0, error="Semantic Scholar server error", duration_ms=dur_ms)
                        await asyncio.sleep(2.0 * attempt)
                        continue

                    if resp.status_code != 200:
                        logger.warning("[S2] Semantic Scholar HTTP %d for query '%s': %s", resp.status_code, query[:40], resp.text[:100])
                        _record_retrieval_error("Semantic Scholar service error")
                        _record_s2_call(query, resp.status_code, 0, error="Semantic Scholar service error", duration_ms=dur_ms)
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

            except Exception as exc:
                dur_ms = int((time.time() - t0) * 1000)
                logger.warning("[S2] Semantic Scholar network error for query '%s': %s", query[:40], exc)
                _record_retrieval_error("Semantic Scholar network error")
                _record_s2_call(query, 0, 0, error="Semantic Scholar network error", duration_ms=dur_ms)
                if attempt < max_attempts:
                    await asyncio.sleep(2.0 * attempt)
                    continue
                return []
    return []


async def _fetch_semantic_scholar_query(query: str, limit: int) -> list[Paper]:
    """Cached wrapper for Semantic Scholar queries."""
    cached = _get_cached_response("semantic_scholar", query)
    if cached is not None:
        return [Paper(**item) for item in cached]

    headers = {"User-Agent": "ResearchLens/2.0 (academic research tool)"}
    if settings.SEMANTIC_SCHOLAR_API_KEY and settings.SEMANTIC_SCHOLAR_API_KEY.strip():
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY.strip()

    try:
        papers = await _fetch_s2_with_retry(query, limit, headers)
        if papers:
            logger.info("[S2] Query '%s' -> %d papers", query[:50], len(papers))
            _set_cached_response("semantic_scholar", query, [p.model_dump() for p in papers])
            return papers
    except Exception as exc:
        logger.warning("[S2] Semantic Scholar error querying '%s': %s", query[:50], exc)
        _record_retrieval_error("Semantic Scholar query error")

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
    is_authenticated = bool(settings.SEMANTIC_SCHOLAR_API_KEY and settings.SEMANTIC_SCHOLAR_API_KEY.strip())
    if is_authenticated:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY.strip()

    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        if i > 0:
            await asyncio.sleep(1.0)
        batch_ids = all_ids[i : i + batch_size]
        async with _s2_limiter:
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
                        retry_after = resp.headers.get("Retry-After")
                        wait_sec = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 3.0
                        logger.warning("[S2-Batch] Semantic Scholar batch rate limited (429 for %d ids). Retry-After: %.1fs", len(batch_ids), wait_sec)
                        _record_s2_rate_limited()
                        _record_s2_call(f"batch:{len(batch_ids)}", 429, 0, error="Semantic Scholar batch rate limited", duration_ms=dur_ms)
                        _record_retrieval_error("Semantic Scholar batch rate limited")
                        _s2_limiter.penalize(wait_sec)
                        if not is_authenticated:
                            logger.info("[S2-Batch] Unauthenticated S2 batch rate limited; continuing with OpenAlex enrichment.")
                            break
                        await asyncio.sleep(wait_sec)
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


# ─── OpenAlex Client & Citation Enrichment ───────────────────────────────────

_DISTINCTIVE_STOP_WORDS = frozenset({
    "which", "what", "who", "when", "where", "how", "why", "whose", "whom",
    "paper", "papers", "introduced", "proposed", "invented", "developed", "created",
    "published", "wrote", "name", "cite", "first", "architecture", "method",
    "algorithm", "model", "technique", "approach", "idea", "key", "its", "was",
    "is", "are", "were", "did", "does", "the", "a", "an", "and", "or", "in", "on",
    "of", "for", "with", "to", "by", "from", "at", "about"
})


def extract_distinctive_term(question: str) -> Optional[str]:
    """
    Extracts a distinctive term (acronym, method name, capitalized name) from a research question
    without an LLM.
    """
    q = question.strip()

    # 1. Acronyms & technical abbreviations: GAN, RAG, ViT, ResNet, word2vec, BERT, LSTM, CNN
    acronym_match = re.search(r"\b([A-Z]{2,6}|ViT|ResNet|word2vec|Word2Vec)\b", q)
    if acronym_match and acronym_match.group(1).lower() not in {"what", "who", "when", "how", "why"}:
        return acronym_match.group(1)

    # 2. Known multi-word canonical method phrases
    method_phrases = [
        "generative adversarial networks",
        "generative adversarial network",
        "batch normalization",
        "layer normalization",
        "vision transformer",
        "vision transformers",
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "self-attention",
        "multi-head attention",
        "residual networks",
        "residual network",
        "deep residual learning",
    ]
    lower_q = q.lower()
    for phrase in method_phrases:
        if phrase in lower_q:
            return phrase

    # 3. Capitalized technical names (not sentence-initial), e.g. "Transformer", "Adam", "Dropout"
    tokens = re.findall(r"\b[A-Za-z0-9_\-]+\b", q)
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        if tok[0].isupper() and tok.lower() not in _DISTINCTIVE_STOP_WORDS:
            if i + 1 < len(tokens) and tokens[i+1][0].isupper() and tokens[i+1].lower() not in _DISTINCTIVE_STOP_WORDS:
                return f"{tok} {tokens[i+1]}"
            return tok

    # 4. Fallback: take meaningful non-stop word
    meaningful = [t for t in tokens if t.lower() not in _DISTINCTIVE_STOP_WORDS and len(t) > 2]
    if meaningful:
        return " ".join(meaningful[:2])

    return None


def reconstruct_abstract_from_inverted_index(inv: Optional[dict]) -> str:
    """Reconstruct human-readable abstract text from OpenAlex abstract_inverted_index."""
    if not inv or not isinstance(inv, dict):
        return ""
    words_positions: list[tuple[int, str]] = []
    for word, positions in inv.items():
        if isinstance(positions, list):
            for pos in positions:
                words_positions.append((pos, word))
    words_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words_positions)


def _parse_openalex_work(w: dict) -> Optional[Paper]:
    """Parse one OpenAlex work item into a Paper schema."""
    if not isinstance(w, dict):
        return None
    title = (w.get("title") or "").strip()
    if not title:
        return None

    raw_id = (w.get("id") or "").split("/")[-1]
    paper_id = f"openalex-{raw_id}" if raw_id else f"openalex-{uuid.uuid4().hex[:10]}"

    authors: list[str] = []
    for a in (w.get("authorships") or []):
        if isinstance(a, dict):
            author_obj = a.get("author")
            if isinstance(author_obj, dict):
                name = (author_obj.get("display_name") or "").strip()
                if name:
                    authors.append(name)
    if not authors:
        authors = ["Unknown"]

    pub_year = w.get("publication_year") or 2024

    venue = "OpenAlex"
    primary_loc = w.get("primary_location")
    if isinstance(primary_loc, dict):
        source_obj = primary_loc.get("source")
        if isinstance(source_obj, dict):
            venue = (source_obj.get("display_name") or "").strip() or "OpenAlex"

    doi = (w.get("doi") or "").strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):].strip()
    elif doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):].strip()

    abstract = reconstruct_abstract_from_inverted_index(w.get("abstract_inverted_index"))

    arxiv_id = None
    ids_obj = w.get("ids")
    if isinstance(ids_obj, dict) and ids_obj.get("arxiv"):
        raw_ar = ids_obj["arxiv"]
        if "/abs/" in raw_ar:
            arxiv_id = raw_ar.split("/abs/")[-1].strip()
        elif raw_ar.lower().startswith("arxiv:"):
            arxiv_id = raw_ar[6:].strip()
        else:
            arxiv_id = raw_ar.strip()

    if not arxiv_id:
        for loc in (w.get("locations") or []):
            if isinstance(loc, dict):
                url = loc.get("landing_page_url") or ""
                if "arxiv.org/abs/" in url:
                    arxiv_id = url.split("arxiv.org/abs/")[-1].strip()
                    break
                if "10.48550/arxiv." in url.lower():
                    arxiv_id = url.lower().split("10.48550/arxiv.")[-1].strip()
                    break

    if arxiv_id:
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    # PDF URL safely
    pdf_url = None
    open_access = w.get("open_access")
    if isinstance(open_access, dict):
        pdf_url = open_access.get("oa_url")
    if not pdf_url:
        best_oa = w.get("best_oa_location")
        if isinstance(best_oa, dict):
            pdf_url = best_oa.get("pdf_url")
    if not pdf_url and isinstance(primary_loc, dict):
        pdf_url = primary_loc.get("pdf_url")
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    citation_count = w.get("cited_by_count") or 0

    return Paper(
        id=paper_id,
        title=title,
        authors=authors,
        publicationYear=pub_year,
        journalConference=venue,
        doi=doi or (f"10.48550/arXiv.{arxiv_id}" if arxiv_id else ""),
        source="OpenAlex",
        abstract=abstract,
        pdfUrl=pdf_url,
        citationCount=citation_count,
        evidenceCount=0,
    )


async def _fetch_openalex_query(query: str, limit: int = 50) -> list[Paper]:
    """Fetch top papers from OpenAlex sorted by cited_by_count descending."""
    clean_q = query.strip()
    if not clean_q:
        return []

    cached = _get_cached_response("openalex", clean_q)
    if cached is not None:
        return [Paper(**item) for item in cached]

    filter_val = getattr(settings, "OPENALEX_FIELD_FILTER", "")
    params = {
        "search": clean_q,
        "sort": "cited_by_count:desc",
        "per_page": min(limit, 50),
        "mailto": getattr(settings, "OPENALEX_EMAIL", "researchlens.tool@gmail.com"),
    }
    if filter_val:
        params["filter"] = filter_val
    headers = {}
    if getattr(settings, "OPENALEX_API_KEY", "") and settings.OPENALEX_API_KEY.strip():
        key = settings.OPENALEX_API_KEY.strip()
        params["api_key"] = key
        headers["Authorization"] = f"Bearer {key}"

    # Exact sanitized URL for diagnosis logging (no email or api key)
    safe_params = [f"search={clean_q}", "sort=cited_by_count:desc", f"per_page={min(limit, 50)}"]
    if filter_val:
        safe_params.append(f"filter={filter_val}")
    diag_url = f"{OPENALEX_API}?{'&'.join(safe_params)}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(OPENALEX_API, params=params, headers=headers)
            logger.info("[OpenAlex] Request URL: '%s' -> HTTP %d", diag_url, resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                papers: list[Paper] = []
                for w in results:
                    p = _parse_openalex_work(w)
                    if p:
                        papers.append(p)

                logger.info("[OpenAlex] Query '%s' (cited_by_count:desc) -> %d papers", clean_q[:50], len(papers))
                if papers:
                    _set_cached_response("openalex", clean_q, [p.model_dump() for p in papers])
                return papers
            else:
                logger.warning("[OpenAlex] HTTP %d for query '%s'", resp.status_code, clean_q[:50])
                _record_retrieval_error("OpenAlex service error")
    except Exception as exc:
        logger.warning("[OpenAlex] error querying '%s': %s", clean_q[:50], exc)
        _record_retrieval_error("OpenAlex connection error")

    return []


async def enrich_papers_with_openalex(papers: list[Paper]) -> list[Paper]:
    """
    Enriches candidate papers that lack citationCount by querying OpenAlex by arXiv ID or DOI.
    """
    uncited = [p for p in papers if getattr(p, "citationCount", 0) <= 0]
    if not uncited:
        return papers

    lookup_map: dict[str, list[Paper]] = {}
    arxiv_filter_urls: list[str] = []
    doi_filter_urls: list[str] = []

    for p in uncited:
        aid = _extract_s2_batch_id(p)  # returns 'ARXIV:...' or 'DOI:...'
        if aid and aid.startswith("ARXIV:"):
            clean_aid = aid[len("ARXIV:"):].strip()
            key_url = f"https://doi.org/10.48550/arxiv.{clean_aid}"
            lookup_map.setdefault(key_url, []).append(p)
            lookup_map.setdefault(clean_aid.lower(), []).append(p)
            if key_url not in arxiv_filter_urls:
                arxiv_filter_urls.append(key_url)
        elif aid and aid.startswith("DOI:"):
            clean_doi = aid[len("DOI:"):].strip()
            key_url = f"https://doi.org/{clean_doi}"
            lookup_map.setdefault(key_url, []).append(p)
            lookup_map.setdefault(clean_doi.lower(), []).append(p)
            if key_url not in doi_filter_urls:
                doi_filter_urls.append(key_url)

    if not lookup_map:
        return papers

    email = getattr(settings, "OPENALEX_EMAIL", "researchlens.tool@gmail.com")
    api_key = getattr(settings, "OPENALEX_API_KEY", "").strip()
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    enriched_count = 0

    # Batch query OpenAlex by locations.landing_page_url for arXiv papers
    batch_size = 25
    for i in range(0, len(arxiv_filter_urls), batch_size):
        chunk = arxiv_filter_urls[i : i + batch_size]
        f = "locations.landing_page_url:" + "|".join(chunk)
        params = {"filter": f, "mailto": email, "per_page": 50}
        if api_key:
            params["api_key"] = api_key
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(OPENALEX_API, params=params, headers=headers)
                if resp.status_code == 200:
                    for w in resp.json().get("results", []):
                        c_count = w.get("cited_by_count") or 0
                        inv = w.get("abstract_inverted_index")
                        abstract = reconstruct_abstract_from_inverted_index(inv) if inv else ""
                        venue = None
                        prim = w.get("primary_location")
                        if isinstance(prim, dict):
                            src = prim.get("source")
                            if isinstance(src, dict):
                                venue = src.get("display_name")
                        doi_val = w.get("doi") or ""

                        matched_targets: list[Paper] = []
                        for loc in (w.get("locations") or []):
                            if isinstance(loc, dict):
                                u = loc.get("landing_page_url") or ""
                                if u in lookup_map:
                                    matched_targets.extend(lookup_map[u])
                        for aid_val in lookup_map:
                            if any(aid_val in (loc.get("landing_page_url") or "") for loc in (w.get("locations") or []) if isinstance(loc, dict)):
                                matched_targets.extend(lookup_map[aid_val])

                        seen_ids: set[str] = set()
                        unique_targets: list[Paper] = []
                        for target in matched_targets:
                            if target.id not in seen_ids:
                                seen_ids.add(target.id)
                                unique_targets.append(target)

                        for target in unique_targets:
                            if c_count > target.citationCount:
                                target.citationCount = c_count
                                enriched_count += 1
                            if not target.abstract and abstract:
                                target.abstract = abstract
                            if venue and target.journalConference in ("arXiv", "Semantic Scholar", "OpenAlex", "unknown", ""):
                                target.journalConference = venue
                            if not target.doi and doi_val:
                                target.doi = doi_val.replace("https://doi.org/", "")
        except Exception as exc:
            logger.warning("[OpenAlex-Enrichment] Failed arXiv batch lookup: %s", exc)

    # Batch query OpenAlex by doi for DOI papers
    for i in range(0, len(doi_filter_urls), batch_size):
        chunk = doi_filter_urls[i : i + batch_size]
        f = "doi:" + "|".join(chunk)
        params = {"filter": f, "mailto": email, "per_page": 50}
        if api_key:
            params["api_key"] = api_key
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(OPENALEX_API, params=params, headers=headers)
                if resp.status_code == 200:
                    for w in resp.json().get("results", []):
                        c_count = w.get("cited_by_count") or 0
                        doi_val = (w.get("doi") or "").lower()
                        for target in lookup_map.get(doi_val, []):
                            if c_count > target.citationCount:
                                target.citationCount = c_count
                                enriched_count += 1
        except Exception as exc:
            logger.warning("[OpenAlex-Enrichment] Failed DOI batch lookup: %s", exc)

    logger.info("[OpenAlex-Enrichment] Enriched %d / %d uncited papers with OpenAlex citation counts", enriched_count, len(uncited))
    return papers


# ─── Main Retrieval Entry Point ───────────────────────────────────────────────

async def retrieve_candidate_papers(
    question: str,
    depth: ResearchDepth,
    sources: list[ResearchSource],
    plan: Optional[QueryPlan] = None,
) -> tuple[list[Paper], QueryPlan]:
    """
    Retrieves candidates from arXiv, OpenAlex, and Semantic Scholar across planned queries.
    Returns merged candidate list and the query plan used.
    """
    if plan is None:
        plan = await plan_research_queries(question)

    depth_val = depth.value if hasattr(depth, "value") else str(depth)
    target_count = settings.DEPTH_COUNTS.get(depth_val, 12)
    fetch_limit = target_count + 6

    q_type = plan.get("question_type", "literature_review") if plan else "literature_review"
    s2_limit = 100 if q_type == "factual_lookup" else fetch_limit

    valid_guesses = validate_title_guesses(plan.get("title_guesses", []))
    source_values = [s.value if hasattr(s, "value") else str(s) for s in sources]

    arxiv_candidates: list[Paper] = []
    openalex_candidates: list[Paper] = []
    s2_candidates: list[Paper] = []

    # 1. arXiv queries
    if (ResearchSource.ARXIV in sources or "arXiv" in source_values) and getattr(settings, "ARXIV_ENABLED", True):
        arxiv_tasks = []
        for title in valid_guesses:
            clean_title = re.sub(r'["\']', '', title).strip()
            if clean_title:
                arxiv_tasks.append(_fetch_arxiv_query(f'ti:"{clean_title}"', fetch_limit))
        for q in plan.get("queries", []):
            phrase = re.sub(r'["\']', '', q).strip()
            if phrase:
                query_expr = _format_arxiv_query_expr(phrase)
                if query_expr:
                    arxiv_tasks.append(_fetch_arxiv_query(query_expr, fetch_limit))
        if arxiv_tasks:
            arxiv_results = await asyncio.gather(*arxiv_tasks, return_exceptions=True)
            for r in arxiv_results:
                if isinstance(r, list):
                    arxiv_candidates.extend(r)
        set_source_status("arXiv", "SUCCESS" if arxiv_candidates else "NO_RESULTS")
    else:
        set_source_status("arXiv", "NOT_USED")

    # 2. OpenAlex queries (sorted by cited_by_count:desc, top 50 per query)
    if (ResearchSource.OPENALEX in sources or "OpenAlex" in source_values) and getattr(settings, "OPENALEX_ENABLED", True):
        openalex_queries: list[str] = []
        distinctive_term = extract_distinctive_term(question)
        if distinctive_term and distinctive_term not in openalex_queries:
            openalex_queries.append(distinctive_term)
        for g in valid_guesses:
            if g not in openalex_queries:
                openalex_queries.append(g)
        for q in plan.get("queries", []):
            if q not in openalex_queries:
                openalex_queries.append(q)

        openalex_tasks = [_fetch_openalex_query(oq, limit=50) for oq in openalex_queries]
        if openalex_tasks:
            openalex_results = await asyncio.gather(*openalex_tasks, return_exceptions=True)
            for r in openalex_results:
                if isinstance(r, list):
                    openalex_candidates.extend(r)
        set_source_status("OpenAlex", "SUCCESS" if openalex_candidates else "NO_RESULTS")
    else:
        set_source_status("OpenAlex", "NOT_USED")

    # 3. Semantic Scholar queries (one at a time, sequential, max 6 if authenticated, 3 if unauthenticated)
    if (ResearchSource.SEMANTIC_SCHOLAR in sources or "Semantic Scholar" in source_values) and getattr(settings, "SEMANTIC_SCHOLAR_ENABLED", True):
        s2_queries_list: list[str] = []
        for g in valid_guesses:
            if g not in s2_queries_list:
                s2_queries_list.append(g)
        # Include non-LLM key terms (e.g. acronyms, method names)
        from .ranker import extract_key_terms
        for kt in (extract_key_terms(question) if question else []):
            if kt not in s2_queries_list:
                s2_queries_list.append(kt)
        for q in plan.get("queries", []):
            if q not in s2_queries_list:
                s2_queries_list.append(q)

        is_authenticated = bool(settings.SEMANTIC_SCHOLAR_API_KEY and settings.SEMANTIC_SCHOLAR_API_KEY.strip())
        max_s2_search_calls = 6 if is_authenticated else 3
        s2_queries_to_run = s2_queries_list[:max_s2_search_calls]
        for s2_q in s2_queries_to_run:
            try:
                s2_res = await _fetch_semantic_scholar_query(s2_q, s2_limit)
                if s2_res:
                    s2_candidates.extend(s2_res)
            except Exception as exc:
                logger.warning("[S2] Semantic Scholar call failed for '%s': %s", s2_q[:40], exc)
                _record_retrieval_error("Semantic Scholar call failed")

        if get_s2_rate_limited_count() > 0:
            set_source_status("Semantic Scholar", "RATE_LIMITED")
            warn_msg = "Semantic Scholar rate limit reached for some queries; results may have incomplete source coverage."
            _record_retrieval_error(warn_msg)
        elif s2_candidates:
            set_source_status("Semantic Scholar", "SUCCESS")
        else:
            set_source_status("Semantic Scholar", "NO_RESULTS")
    else:
        set_source_status("Semantic Scholar", "NOT_USED")

    # Log candidates per source and how many came from citation-sorted path
    logger.info(
        "[Retrieval] Candidates per source: arXiv=%d, Semantic Scholar=%d, OpenAlex=%d (citation-sorted: %d), source_status=%s",
        len(arxiv_candidates),
        len(s2_candidates),
        len(openalex_candidates),
        len(openalex_candidates),
        get_source_status(),
    )

    # Merge into candidate pool BEFORE dedupe and enrichment
    all_raw_candidates: list[Paper] = []
    all_raw_candidates.extend(arxiv_candidates)
    all_raw_candidates.extend(openalex_candidates)
    all_raw_candidates.extend(s2_candidates)

    logger.info("[Retrieval] Total raw candidates merged: %d", len(all_raw_candidates))
    return all_raw_candidates, plan


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

