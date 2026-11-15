"""
ResearchLens FastAPI Backend
==============================
Full pipeline implementation:
  - Paper retrieval: arXiv API + Semantic Scholar API
  - PDF extraction: PyMuPDF
  - Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
  - Vector DB: FAISS
  - Claim verification: Hugging Face NLI (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
  - Report generation: Gemini API
  - Database: SQLite (default) → PostgreSQL (production)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("researchlens")

# ── Load .env from backend and root directories ──────────────────────────────
from dotenv import load_dotenv
_root_env = Path(__file__).parent.parent / ".env"
_backend_env = Path(__file__).parent / ".env"
if _root_env.exists():
    load_dotenv(_root_env)
if _backend_env.exists():
    load_dotenv(_backend_env, override=True)
else:
    load_dotenv(Path(__file__).parent / ".env.example")

import sys
_backend_dir = Path(__file__).resolve().parent
_root_dir = _backend_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

try:
    from .config import settings
    from .models.database import create_tables
    from .routers.research import router as research_router, direct_router
except (ImportError, ValueError):
    from backend.config import settings
    from backend.models.database import create_tables
    from backend.routers.research import router as research_router, direct_router


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== ResearchLens Backend starting ===")
    logger.info("S2 key loaded: %s", bool(settings.SEMANTIC_SCHOLAR_API_KEY.strip()))
    logger.info("Database: %s", settings.DATABASE_URL)
    logger.info("OpenAI configured: %s (model: %s)", settings.is_openai_configured, settings.OPENAI_MODEL)
    logger.info("Gemini configured: %s", settings.is_gemini_configured)
    logger.info("NLI model: %s", settings.NLI_MODEL)
    logger.info("Embedding model: %s", settings.EMBEDDING_MODEL)

    # Create DB tables
    create_tables()
    logger.info("Database tables ready")

    # Jobs left in_progress by a previous process can never finish; mark them failed.
    try:
        from .models.database import SessionLocal, ResearchJob
        _db = SessionLocal()
        try:
            stale = _db.query(ResearchJob).filter(ResearchJob.status == "in_progress").all()
            for _j in stale:
                _j.status = "failed"
                _j.failure_reason = "Server restarted while this job was running. Please retry."
            _db.commit()
            if stale:
                logger.warning("Marked %d stale in_progress job(s) as failed", len(stale))
        finally:
            _db.close()
    except Exception as exc:
        logger.warning("Stale job cleanup failed: %s", exc)

    # Ensure PDF cache directory exists
    Path(settings.PDF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # Configure PyTorch thread count (defaults to settings.TORCH_NUM_THREADS)
    try:
        import torch
        torch.set_num_threads(getattr(settings, "TORCH_NUM_THREADS", 2))
    except Exception:
        pass

    # Model pre-warmup (skip when SKIP_MODEL_WARMUP=true to enable fast cloud boot on low RAM)
    skip_warmup = os.getenv("SKIP_MODEL_WARMUP", "false").lower() in ("1", "true", "yes")
    if not skip_warmup:
        import time

        t0 = time.time()
        try:
            from .services.embeddings import get_model as get_embedding_model
            get_embedding_model()
            logger.info("[Startup] Embedding model warmed up in %.2fs", time.time() - t0)
        except Exception as exc:
            logger.warning("[Startup] Embedding model warmup failed: %s", exc)

        t0 = time.time()
        try:
            from .services.ranker import get_reranker
            get_reranker()
            logger.info("[Startup] Reranker model warmed up in %.2fs", time.time() - t0)
        except Exception as exc:
            logger.warning("[Startup] Reranker model warmup failed: %s", exc)

        t0 = time.time()
        try:
            from .services.nli_verifier import get_nli_components
            get_nli_components()
            logger.info("[Startup] NLI model warmed up in %.2fs", time.time() - t0)
        except Exception as exc:
            logger.warning("[Startup] NLI model warmup failed: %s", exc)

        if settings.LLM_PROVIDER.lower() == "ollama":
            t0 = time.time()
            try:
                from .services.local_llm_service import _call_ollama
                await _call_ollama("warmup", max_tokens=1)
                logger.info("[Startup] Ollama 1-token warmup call completed in %.2fs", time.time() - t0)
            except Exception as exc:
                logger.warning("[Startup] Ollama warmup failed: %s", exc)
    else:
        logger.info("[Startup] Model pre-warmup skipped (models will load lazily on demand to conserve RAM)")

    yield

    logger.info("=== ResearchLens Backend shutting down ===")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResearchLens API",
    description=(
        "Academic research synthesis API with real paper retrieval (arXiv + Semantic Scholar), "
        "PDF extraction (PyMuPDF), semantic embeddings (Sentence Transformers), "
        "FAISS vector search, Hugging Face NLI claim verification, and Gemini report generation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allows local dev servers, Streamlit, and custom CORS_ORIGINS
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:8501",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:8501",
]
if _cors_origins_env:
    for _origin in _cors_origins_env.split(","):
        _cleaned = _origin.strip()
        if _cleaned and _cleaned not in _allowed_origins:
            _allowed_origins.append(_cleaned)

_has_wildcard = "*" in _allowed_origins
_origin_regex = os.getenv("CORS_ORIGIN_REGEX")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if not _has_wildcard else ["*"],
    allow_origin_regex=_origin_regex,
    allow_credentials=not _has_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(research_router)
app.include_router(direct_router)


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_configured": settings.is_llm_configured,
        "openai_configured": settings.is_openai_configured,
        "openai_model": settings.OPENAI_MODEL if settings.is_openai_configured else None,
        "gemini_configured": settings.is_gemini_configured,
        "qwen_configured": settings.is_qwen_configured,
        "nli_model": settings.NLI_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "database": settings.DATABASE_URL.split("///")[-1],
    }


# ── Entrypoint (python main.py) ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
