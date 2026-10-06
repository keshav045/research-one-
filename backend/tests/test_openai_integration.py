"""
Unit tests for OpenAI LLM integration and provider routing.
Verifies that OpenAI completions are called correctly when configured
and fallback mechanisms handle missing credentials cleanly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.config import settings
from backend.services.local_llm_service import _call_openai, _call_llm, get_provider_model


@pytest.mark.asyncio
async def test_call_openai_unconfigured(monkeypatch):
    """When OPENAI_API_KEY is empty, _call_openai should immediately return empty string."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    res = await _call_openai("test prompt")
    assert res == ""


@pytest.mark.asyncio
async def test_call_openai_success(monkeypatch):
    """When OPENAI_API_KEY is provided, _call_openai calls AsyncOpenAI client and returns text."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-testkey123456789")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "gpt-4o-mini")

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Synthesized academic answer with [1]."
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        res = await _call_openai("Synthesize this")
        assert res == "Synthesized academic answer with [1]."
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_call_llm_routes_to_openai(monkeypatch):
    """When LLM_PROVIDER is openai and key is configured, _call_llm returns provider 'openai'."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-testkey")

    with patch("backend.services.local_llm_service._call_openai", AsyncMock(return_value="Synthesized claim text.")):
        text, provider = await _call_llm("test prompt")
        assert text == "Synthesized claim text."
        assert provider == "openai"


def test_get_provider_model_openai(monkeypatch):
    """get_provider_model returns OPENAI_MODEL when provider is openai."""
    monkeypatch.setattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
    assert get_provider_model("openai") == "gpt-4o-mini"
    assert get_provider_model("OpenAI") == "gpt-4o-mini"
