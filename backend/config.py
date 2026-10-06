"""Backend configuration loaded from .env via pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load backend/.env first, then root .env if keys exist there
        env_file=(str(Path(__file__).parent / ".env"), str(Path(__file__).parent.parent / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider: "openai" (OpenAI API) | "gemini" | "qwen"
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

    # Local LLM Fallback (disabled — cloud LLM is active)
    LOCAL_LLM_MODEL: str = "Qwen/Qwen2.5-0.5B-Instruct"
    LOCAL_LLM_MAX_TOKENS: int = 1536
    LLM_FALLBACK_LOCAL: bool = False
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b-instruct"

    # Semantic Scholar
    SEMANTIC_SCHOLAR_API_KEY: str = ""

    # OpenAlex Polite Pool & API Key
    OPENALEX_EMAIL: str = "researchlens.tool@gmail.com"
    OPENALEX_API_KEY: str = ""
    OPENALEX_FIELD_FILTER: str = "topics.field.id:17"

    # Domain / Field of Study Filters
    DEFAULT_FIELDS_OF_STUDY: str = "Computer Science"

    # Source Configuration
    ARXIV_ENABLED: bool = True
    SEMANTIC_SCHOLAR_ENABLED: bool = True
    OPENALEX_ENABLED: bool = True
    CROSSREF_ENABLED: bool = False
    CORE_ENABLED: bool = False

    # Relevance Thresholds
    MIN_PAPER_RELEVANCE: float = 0.40

    # Feature Flags
    SEED_PAPERS_ENABLED: bool = True
    DISABLE_CACHE: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./researchlens.db"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ML Models
    NLI_MODEL: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    NLI_DEVICE: str = "cuda"
    NLI_ENTAIL_THRESHOLD: float = 0.80
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
    MAX_PDF_WORKERS: int = 4
    MAX_PDF_SIZE_BYTES: int = 80 * 1024 * 1024  # 80 MB PDF size cap

    @property
    def is_openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY) and self.OPENAI_API_KEY.strip() not in ("your_openai_api_key_here", "")

    @property
    def is_ollama_configured(self) -> bool:
        return self.LLM_PROVIDER.lower() == "ollama"

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY) and self.GEMINI_API_KEY != "your_gemini_api_key_here"

    @property
    def is_qwen_configured(self) -> bool:
        return bool(self.QWEN_API_KEY) and self.QWEN_API_KEY != "your_qwen_api_key_here"

    @property
    def is_local_llm_configured(self) -> bool:
        return False

    @property
    def is_llm_configured(self) -> bool:
        """True if any cloud LLM provider is ready."""
        provider = self.LLM_PROVIDER.lower()
        if provider == "openai":
            return self.is_openai_configured
        if provider == "gemini":
            return self.is_gemini_configured
        if provider == "qwen":
            return self.is_qwen_configured
        return (
            self.is_openai_configured
            or self.is_gemini_configured
            or self.is_qwen_configured
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
