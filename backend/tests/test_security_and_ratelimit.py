"""
Tests for API Key Authentication & In-Memory Rate Limiting
===========================================================
Verifies:
1. API key authentication enforcement when API_KEY is set.
2. Permissive local development mode when API_KEY is unset/empty.
3. Both X-API-Key and Bearer authorization schemes.
4. Sliding window rate limiter behavior and 429 response on threshold exceed.
"""

import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException

from backend.config import settings
from backend.security import (
    rate_limiter,
    verify_api_key,
    rate_limit_research,
    get_client_ip,
)
from fastapi.security import HTTPAuthorizationCredentials


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset in-memory rate limiter state before each test."""
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def _dummy_request(ip: str = "127.0.0.1", headers: dict = None) -> MagicMock:
    req = MagicMock()
    req.client.host = ip
    req.headers = headers or {}
    req.url.path = "/api/research"
    return req


# ── 1. Client IP Resolution ───────────────────────────────────────────────────

def test_get_client_ip_direct_and_forwarded():
    # Direct client
    req1 = _dummy_request(ip="192.168.1.100")
    assert get_client_ip(req1) == "192.168.1.100"

    # Forwarded header (reverse proxy / cloud load balancer)
    req2 = _dummy_request(headers={"x-forwarded-for": "203.0.113.195, 70.41.3.18"})
    assert get_client_ip(req2) == "203.0.113.195"


# ── 2. Rate Limiting Tests ────────────────────────────────────────────────────

def test_sliding_window_rate_limiter_allows_then_blocks():
    key = "test:1.2.3.4"
    # Allow 3 requests
    rate_limiter.check(key, max_requests=3, window_seconds=60.0)
    rate_limiter.check(key, max_requests=3, window_seconds=60.0)
    rate_limiter.check(key, max_requests=3, window_seconds=60.0)

    # 4th request must be rejected with 429
    with pytest.raises(HTTPException) as exc_info:
        rate_limiter.check(key, max_requests=3, window_seconds=60.0)

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
    assert "Retry-After" in exc_info.value.headers


@pytest.mark.asyncio
async def test_rate_limit_research_dependency(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_RESEARCH_PER_MINUTE", 2)

    req = _dummy_request(ip="10.0.0.1")

    # 2 requests succeed
    await rate_limit_research(req)
    await rate_limit_research(req)

    # 3rd request fails
    with pytest.raises(HTTPException) as exc:
        await rate_limit_research(req)
    assert exc.value.status_code == 429


# ── 3. API Key Authentication Tests ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_api_key_open_mode_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", None)
    req = _dummy_request()

    # When no API key is configured, request passes freely
    result = await verify_api_key(req, api_key_header=None, bearer_creds=None)
    assert result is None


@pytest.mark.asyncio
async def test_verify_api_key_rejects_missing_key_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret-token-xyz")
    req = _dummy_request()

    # Missing key raises 401
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(req, api_key_header=None, bearer_creds=None)
    assert exc.value.status_code == 401
    assert "Unauthorized" in exc.value.detail


@pytest.mark.asyncio
async def test_verify_api_key_rejects_invalid_key(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret-token-xyz")
    req = _dummy_request()

    # Invalid key raises 401
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(req, api_key_header="wrong-token", bearer_creds=None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_api_key_accepts_valid_header(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret-token-xyz")
    req = _dummy_request()

    # Valid X-API-Key passes
    res = await verify_api_key(req, api_key_header="secret-token-xyz", bearer_creds=None)
    assert res == "secret-token-xyz"


@pytest.mark.asyncio
async def test_verify_api_key_accepts_valid_bearer(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "secret-token-xyz")
    req = _dummy_request()

    bearer = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-token-xyz")
    res = await verify_api_key(req, api_key_header=None, bearer_creds=bearer)
    assert res == "secret-token-xyz"
