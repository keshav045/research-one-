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
