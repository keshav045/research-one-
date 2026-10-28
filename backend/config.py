"""Backend configuration loaded from .env via pydantic-settings (with BaseModel fallback)."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore[assignment]
    SettingsConfigDict = dict  # type: ignore[misc,assignment]
    _HAS_PYDANTIC_SETTINGS = False


def _load_env_files() -> None:
    """Load backend/.env and root .env into os.environ if not already set."""
    env_paths = [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]
    for p in env_paths:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass


_load_env_files()


class Settings(BaseSettings):
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            # Load backend/.env first, then root .env if keys exist there
            env_file=(str(Path(__file__).parent / ".env"), str(Path(__file__).parent.parent / ".env")),
            env_file_encoding="utf-8",
            extra="ignore",
        )

    def __init__(self, **kwargs: Any):
        if not _HAS_PYDANTIC_SETTINGS:
            fields_dict = getattr(self.__class__, "model_fields", None) or getattr(self.__class__, "__fields__", {})
            for field_name, field_info in fields_dict.items():
                if field_name not in kwargs and field_name in os.environ:
                    raw_val = os.environ[field_name]
                    default_val = getattr(field_info, "default", None)
                    if isinstance(default_val, bool):
                        kwargs[field_name] = raw_val.lower() in ("1", "true", "yes")
                    elif isinstance(default_val, int):
                        try:
                            kwargs[field_name] = int(raw_val)
                        except ValueError:
                            pass
                    elif isinstance(default_val, float):
                        try:
                            kwargs[field_name] = float(raw_val)
                        except ValueError:
                            pass
                    else:
                        kwargs[field_name] = raw_val
        super().__init__(**kwargs)

    # LLM Provider: "openai" (OpenAI API) | "gemini" | "qwen" | "ollama" | "local"
    LLM_PROVIDER: str = "openai"

    # OpenAI (Primary cloud LLM provider)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Gemini (secondary / alternative)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Qwen (Alibaba DashScope — OpenAI-compatible)
    QWEN_API_KEY: str = ""
    QWEN_MODEL: str = "qwen-plus"

    # Local LLM Fallback (active when OpenAI / cloud fails)
    LOCAL_LLM_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
    LOCAL_LLM_MAX_TOKENS: int = 1536
    LLM_FALLBACK_LOCAL: bool = True
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b-instruct"

    # Semantic Scholar
    SEMANTIC_SCHOLAR_API_KEY: str = ""

    # OpenAlex Polite Pool & API Key (Configured via environment)
    OPENALEX_EMAIL: str = ""
    OPENALEX_API_KEY: str = ""
    OPENALEX_FIELD_FILTER: str = ""

    # Domain / Field of Study Filters (Optional dynamic filter)
    DEFAULT_FIELDS_OF_STUDY: str = ""

    # Source Configuration
    ARXIV_ENABLED: bool = True
    SEMANTIC_SCHOLAR_ENABLED: bool = True
    OPENALEX_ENABLED: bool = True
    CROSSREF_ENABLED: bool = False
    CORE_ENABLED: bool = False

    # Relevance Thresholds
    MIN_PAPER_RELEVANCE: float = 0.30

    # Feature Flags
    SEED_PAPERS_ENABLED: bool = True
    DISABLE_CACHE: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./researchlens.db"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000


    # Security & Rate Limiting
    API_KEY: Optional[str] = None
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_RESEARCH_PER_MINUTE: int = 10

    # ML Models
    NLI_MODEL: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    NLI_DEVICE: str = "cuda"
    NLI_ENTAIL_THRESHOLD: float = 0.80

    # Embedding & Vector Indexing Optimizations
    TORCH_NUM_THREADS: int = 4
    EMBED_MAX_LENGTH: int = 128
    SKIP_NONBODY_SECTIONS: bool = True
    MAX_PDF_PAGES: int = 100
    MAX_PASSAGES_PER_PAPER: int = 25
    EMBED_CACHE_ENABLED: bool = True
    EMBED_QUANTIZE: bool = False
    EMBEDDING_BACKEND: str = "auto"  # "auto" | "openai" | "local"
    RELEVANCE_THRESHOLD: float = 0.30
    ANCHOR_MIN_CITATIONS: int = 1000
    ANCHOR_WEIGHT_REF: float = 0.5
    ANCHOR_WEIGHT_CITES: float = 0.3
    ANCHOR_WEIGHT_EARLINESS: float = 0.2
    ANCHOR_TOPICAL_MIN: float = 0.35
    ANCHOR_MARGIN_THRESHOLD: float = 0.10
    ANCHOR_VARIANT_WORDS: list[str] = [
        "3d", "sentence-", "group", "fast", "swin", "rotary", "survey", "overview", "review"
    ]
    ANCHOR_PREFIX_PREFERENCE: bool = True

    # Factual Lookup Ranking Weights
    FACTUAL_SEMANTIC_WEIGHT: float = 0.35
    FACTUAL_CITATION_WEIGHT: float = 0.45
    FACTUAL_TITLE_WEIGHT: float = 0.20

    DEPTH_COUNTS: dict[str, int] = {"Quick": 6, "Standard": 12, "Deep": 24}

    # File system
    PDF_CACHE_DIR: str = "./pdf_cache"
    MAX_PDF_WORKERS: int = 6
    MAX_PDF_SIZE_BYTES: int = 80 * 1024 * 1024  # 80 MB PDF size cap

    @property
    def is_openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY) and self.OPENAI_API_KEY.strip() not in ("your_openai_api_key_here", "xxx", "")

    @property
    def is_ollama_configured(self) -> bool:
        return self.LLM_PROVIDER.lower() == "ollama"

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY) and self.GEMINI_API_KEY.strip() not in ("your_gemini_api_key_here", "xxx", "")

    @property
    def is_qwen_configured(self) -> bool:
        return bool(self.QWEN_API_KEY) and self.QWEN_API_KEY.strip() not in ("your_qwen_api_key_here", "xxx", "")

    @property
    def is_local_llm_configured(self) -> bool:
        return bool(self.LLM_FALLBACK_LOCAL)

    @property
    def is_llm_configured(self) -> bool:
        """True if the active provider OR any fallback cloud/local LLM provider is ready."""
        provider = self.LLM_PROVIDER.lower()
        if provider == "openai" and self.is_openai_configured:
            return True
        if provider == "gemini" and self.is_gemini_configured:
            return True
        if provider == "qwen" and self.is_qwen_configured:
            return True
        if provider == "ollama":
            return True
        # Auto-detect if user configured another cloud key (e.g. GEMINI_API_KEY while LLM_PROVIDER=openai)
        return (
            self.is_openai_configured
            or self.is_gemini_configured
            or self.is_qwen_configured
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
