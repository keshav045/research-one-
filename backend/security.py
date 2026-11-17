"""
Security & Rate Limiting Module for ResearchLens
=================================================
Provides:
1. API Key Authentication (via X-API-Key or Authorization Bearer header).
   - Enabled when settings.API_KEY is configured.
   - Disabled / open in local development when settings.API_KEY is unset or empty.
2. In-memory IP Sliding Window Rate Limiting:
   - General endpoint limit (default 60 requests/minute).
   - Research job start limit (default 10 requests/minute).
"""

from __future__ import annotations

import collections
import logging
import re
import secrets
import threading
import time
from typing import Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from .config import settings

logger = logging.getLogger(__name__)

# Security scheme headers for OpenAPI docs
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer_scheme = HTTPBearer(auto_error=False)


# ─── Client IP Resolution ─────────────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For if behind a reverse proxy."""
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


# ─── Sliding Window Rate Limiter ──────────────────────────────────────────────

class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)

    def check(self, key: str, max_requests: int, window_seconds: float = 60.0) -> None:
        """
        Check if request under `key` exceeds max_requests within window_seconds.
        Raises HTTPException 429 if rate limit is exceeded.
        """
        if not getattr(settings, "RATE_LIMIT_ENABLED", True) or max_requests <= 0:
            return

        now = time.monotonic()
        cutoff = now - window_seconds

        with self._lock:
            timestamps = self._requests[key]
            # Prune timestamps older than window
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

            if len(timestamps) >= max_requests:
                oldest = timestamps[0]
                retry_after = max(1, int(window_seconds - (now - oldest)))
                logger.warning(
                    "[RateLimiter] Client '%s' exceeded limit (%d req / %ds). Retry-After: %ds",
                    key, max_requests, int(window_seconds), retry_after,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {max_requests} requests per {int(window_seconds)}s.",
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)

    def reset(self) -> None:
        """Clear all tracked request timestamps (useful for tests)."""
        with self._lock:
            self._requests.clear()


rate_limiter = SlidingWindowRateLimiter()


async def rate_limit_general(request: Request) -> None:
    """Rate limit dependency for general endpoints (default 60 req/min)."""
    client_ip = get_client_ip(request)
    limit = getattr(settings, "RATE_LIMIT_PER_MINUTE", 60)
    rate_limiter.check(f"general:{client_ip}", max_requests=limit, window_seconds=60.0)


async def rate_limit_research(request: Request) -> None:
    """Rate limit dependency for research pipeline execution (default 10 req/min)."""
    client_ip = get_client_ip(request)
    limit = getattr(settings, "RATE_LIMIT_RESEARCH_PER_MINUTE", 10)
    rate_limiter.check(f"research:{client_ip}", max_requests=limit, window_seconds=60.0)


# ─── API Key Authentication ───────────────────────────────────────────────────

async def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header_scheme),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(http_bearer_scheme),
) -> Optional[str]:
    """
    Validates API key from X-API-Key header or Authorization Bearer header.
    - If settings.API_KEY is not set or empty: allows request (development mode).
    - If settings.API_KEY is set: requires valid matching key using constant-time comparison.
    """
    configured_key = getattr(settings, "API_KEY", None)
    if not configured_key or not configured_key.strip():
        # Open / dev mode when no key configured
        return None

    # Check X-API-Key header first
    provided = api_key_header
    # Fall back to Authorization: Bearer <key>
    if not provided and bearer_creds:
        provided = bearer_creds.credentials

    if not provided or not secrets.compare_digest(provided.strip(), configured_key.strip()):
        logger.warning(
            "[Security] Unauthorized request to '%s' from IP '%s' (invalid or missing API Key)",
            request.url.path, get_client_ip(request),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing or invalid API Key. Provide via 'X-API-Key' header or 'Bearer' token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided


# ─── Academic Integrity & Safety Guard ────────────────────────────────────────

UNETHICAL_ACADEMIC_PATTERNS = [
    (r"\b(?:fabricat|forge|fake)\s+(?:\d+\s+)?(?:paper|citation|doi|reference|author)", "citation_fraud"),
    (r"\b(?:write\s+(?:my|a)\s+.*?(?:dissertation|thesis|essay|paper)\s+.*?(?:without\s+citing|verbatim|as\s+my\s+own)|ghostwrit)", "ghostwriting_plagiarism"),
    (r"\b(?:jailbreak|dan\s*(?:\(|do\s*anything\s*now)|ignore\s+all\s+(?:academic|previous)\s+limitations)", "jailbreak_attempt"),
    (r"\b(?:malware|keylogger|trojan|ransomware|exploit\s+script)", "malicious_code"),
    (r"\b(?:ddos|denial\s+of\s+service|attack\s+against\s+(?:arxiv|openalex|crossref|server))", "academic_sabotage"),
    (r"\b(?:generate\s+false\s+benchmark|falsif(?:y|ication)\s+(?:data|number|result))", "data_falsification")
]


def check_academic_safety(query: str) -> tuple[bool, Optional[str]]:
    """
    Evaluates whether an inquiry violates academic ethics or safety guardrails.
    Returns (is_safe, refusal_reason).
    """
    q_lower = (query or "").lower()
    for pattern, category in UNETHICAL_ACADEMIC_PATTERNS:
        if re.search(pattern, q_lower):
            return False, f"Refused: Request violates academic integrity policy ({category.replace('_', ' ')})."
    return True, None


# ─── Academic Scope & Relevance Guard ─────────────────────────────────────────

NON_ACADEMIC_PATTERNS = [
    (r"\b(?:recipe|cook(?:ing)?|bake|baking|pizza|dough|pasta|burger|cake|ingredient|delicious|dinner|breakfast|lunch)\b", "culinary_recipe"),
    (r"\b(?:football|cricket|premier league|nba|fifa|match score|game score|who won the match|sports team)\b", "sports_and_games"),
    (r"\b(?:bedtime story|fairy tale|dragon|netflix|movie recommendation|romantic comedy|song lyrics|celebrity gossip|poem|write a story)\b", "entertainment_fiction"),
    (r"\b(?:book a hotel|flight ticket|room in paris|restaurant near me|vacation in|cheap shoes|buy product|grocery store)\b", "commercial_lifestyle"),
    (r"^(?:hi|hello|hey|good morning|good evening|how are you|who are you|what is your name|tell me a joke)\b", "casual_chat"),
    (r"\b(?:invest (?:my )?money in (?:bitcoin|ethereum|crypto|stocks)|crypto price prediction|stock pick)\b", "financial_speculation"),
    (r"\b(?:my mom|my dad|my dog|my cat|my friend|my boyfriend|my girlfriend|my trip|my vacation|walked down the street)\b", "personal_narrative"),
]

ACADEMIC_SCHOLARLY_KEYWORDS = {
    # Scholarly metadata & literature terminology
    "paper", "papers", "study", "studies", "author", "authors", "doi", "arxiv", "journal", "conference",
    "proceedings", "literature", "review", "findings", "empirical", "methodology", "algorithm",
    "architecture", "architectures", "mechanism", "mechanisms", "framework", "formulation",
    "theorem", "hypothesis", "dataset", "datasets", "benchmark", "benchmarks", "sota",
    "ablation", "evaluation", "citation", "citations", "survey", "paradigm", "proof", "experiment",
    # Computer Science, Machine Learning & AI
    "transformer", "transformers", "attention", "self-attention", "neural", "deep learning",
    "machine learning", "gradient", "gradients", "backpropagation", "convolution", "convolutional",
    "cnn", "rnn", "lstm", "diffusion", "latent diffusion", "gan", "generative adversarial",
    "reinforcement learning", "lora", "fine-tuning", "rag", "retrieval", "retrieval-augmented",
    "embeddings", "embedding", "vector", "bert", "resnet", "encoder", "decoder", "nlp",
    "computer vision", "optimization", "adam", "sgd", "loss function", "stochastic",
    "hyperparameter", "llm", "llms", "language model", "tokenization", "tokens",
    "prompt", "prompting", "react", "chain-of-thought", "cot", "augmented", "imagenet",
    # Natural, Physical & Formal Sciences
    "crispr", "cas9", "genome", "genomics", "gene", "genes", "dna", "rna", "protein", "proteins",
    "mutation", "mutations", "clinical", "efficacy", "pathway", "therapy", "cellular", "receptor",
    "biology", "biological", "biomedical", "pharmacology", "immunology", "quantum", "thermodynamics",
    "relativity", "particle", "gravity", "semiconductor", "spectroscopy", "superconductor",
    "differential equation", "linear algebra", "eigenvalue", "eigenvalues", "manifold", "topology",
    # Social Sciences & Economics
    "econometrics", "macroeconomic", "macroeconomics", "microeconomics", "monetary policy",
    "inflation", "sociological", "psychological", "cognitive", "behavioral", "ethnography",
    "regression", "causality", "correlation", "game theory", "annealing", "qubit"
}

INQUIRY_VERBS_PATTERN = (
    r"^(?:what|which|how|why|who|whom|whose|is|are|was|were|do|does|did|can|could|should|would|"
    r"compare|comparing|contrast|explain|evaluate|analyze|investigate|review|survey|synthesize|"
    r"discuss|summarize|examine|assess|describe|empirical|benchmark|benchmarks|overview|advances|study|studies)\b"
)

MIN_QUESTION_LENGTH = 10
MAX_QUESTION_LENGTH = 500


def check_academic_scope(query: str) -> tuple[bool, Optional[str]]:
    """
    Evaluates whether an inquiry is related to academic research / scientific literature.
    Rejects raw narrative paragraphs, stories, and non-academic domains.
    Returns (is_in_scope, refusal_reason).
    """
    q_clean = (query or "").strip()
    q_lower = q_clean.lower()

    # 1. Check for explicit non-academic categories (recipes, sports, entertainment, casual chat)
    for pattern, category in NON_ACADEMIC_PATTERNS:
        if re.search(pattern, q_lower):
            category_label = category.replace("_", " ").title()
            return (
                False,
                f"Out of Scope ({category_label}): ResearchLens is an academic research assistant designed to analyze scientific papers, methodologies, and literature reviews. Please ask a research-related question."
            )

    # 2. Rejection of narrative blocks: multiple declarative sentences with no inquiry starter and no question mark
    sentences = [s.strip() for s in re.split(r"[\.\n]+", q_clean) if s.strip()]
    has_qmark = "?" in q_clean
    starts_inquiry = bool(re.match(INQUIRY_VERBS_PATTERN, q_clean, re.IGNORECASE))

    if len(sentences) >= 2 and not has_qmark and not starts_inquiry:
        return (
            False,
            "Invalid Inquiry: The input appears to be a narrative paragraph or statement rather than a research question. Please formulate your query as a specific research question (e.g. 'How does X work?' or 'Compare X and Y')."
        )

    # 3. Check for question framing (must contain question mark, inquiry verb, or scholarly comparative phrase)
    if not has_qmark and not starts_inquiry:
        has_scholarly_topic_phrase = bool(
            re.search(
                r"\b(?:empirical (?:benchmark|study|analysis)|state of the art in|literature review on|advances in|comparison of|benchmark of)\b",
                q_lower,
            )
        )
        if not has_scholarly_topic_phrase:
            return (
                False,
                "Invalid Inquiry: Missing question framing. Please ask a specific question ending with '?' or starting with an inquiry keyword such as 'Compare', 'Explain', 'What is', or 'How does'."
            )

    # 4. Extract word tokens (3+ letters) and check intersection with academic concepts
    tokens = set(re.findall(r"\b[a-z]{3,}\b", q_lower))
    academic_hits = len(tokens.intersection(ACADEMIC_SCHOLARLY_KEYWORDS))

    # 5. Check for structural academic inquiry patterns
    has_scholarly_structure = bool(
        re.search(
            r"\b(?:which paper|what paper|who (?:introduced|proposed|discovered)|"
            r"compare\s+.*?\s+and|difference between\s+.*?\s+and|"
            r"what is the mechanism of|how does\s+.*?\s+(?:work|operate|reduce|improve|scale)|"
            r"what are the limitations of|state of the art in|findings on|evidence for)\b",
            q_lower,
        )
    )

    if academic_hits >= 1 or has_scholarly_structure:
        return True, None

    return (
        False,
        "Out of Scope: No academic or scientific concepts detected. Please ask a question related to published research papers, algorithms, architectures, scientific studies, or literature reviews."
    )


def validate_research_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Pre-flight validator for research questions before entering the pipeline.
    Validates:
      1. Minimum & maximum character limits (10 to 500 characters).
      2. Academic safety, anti-ghostwriting, and ethics policies.
      3. Academic scope and relevance to research literature.
    Returns (is_valid, error_message).
    """
    if not query or not query.strip():
        return False, "Please enter a research question."

    q_clean = query.strip()
    if len(q_clean) < MIN_QUESTION_LENGTH:
        return (
            False,
            f"Question is too short ({len(q_clean)} characters). Minimum question length is {MIN_QUESTION_LENGTH} characters. Please provide a more descriptive research question."
        )

    if len(q_clean) > MAX_QUESTION_LENGTH:
        return (
            False,
            f"Question is too long ({len(q_clean)} characters). Maximum question length is {MAX_QUESTION_LENGTH} characters. Please condense your question."
        )

    # Check academic ethics and safety
    is_safe, safety_err = check_academic_safety(q_clean)
    if not is_safe:
        return False, safety_err

    # Check academic scope
    is_scope, scope_err = check_academic_scope(q_clean)
    if not is_scope:
        return False, scope_err

    return True, None


