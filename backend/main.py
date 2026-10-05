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
    logger.info("Gemini configured: %s", settings.is_gemini_configured)
    logger.info("NLI model: %s", settings.NLI_MODEL)
    logger.info("Embedding model: %s", settings.EMBEDDING_MODEL)

    # Create DB tables
    create_tables()
    logger.info("Database tables ready")

    # Ensure PDF cache directory exists
    Path(settings.PDF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # Configure PyTorch to use 1 thread to conserve memory in low-RAM cloud containers
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    # Optional model pre-warmup (default: skipped to enable fast boot and stay within 512MB RAM)
    skip_warmup = os.getenv("SKIP_MODEL_WARMUP", "true").lower() in ("1", "true", "yes")
    if not skip_warmup:
        logger.info("[Startup] Warming up ML models...")
        try:
            from .services.embeddings import get_model as get_embedding_model
            get_embedding_model()
        except Exception as exc:
            logger.warning("[Startup] Embedding warmup failed: %s", exc)
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
# Allows local dev servers, Netlify deploy URLs (*.netlify.app), and custom CORS_ORIGINS
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]
if _cors_origins_env:
    for _origin in _cors_origins_env.split(","):
        _cleaned = _origin.strip()
        if _cleaned and _cleaned not in _allowed_origins:
            _allowed_origins.append(_cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if "*" not in _allowed_origins else ["*"],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
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
        "local_model": settings.LOCAL_LLM_MODEL if settings.LLM_PROVIDER == "local" else None,
        "qwen_configured": settings.is_qwen_configured,
        "gemini_configured": settings.is_gemini_configured,
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
