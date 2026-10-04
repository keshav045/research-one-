"""
ResearchLens FastAPI Backend
==============================
Full pipeline implementation:
  - Paper retrieval: arXiv API + Semantic Scholar API
  - PDF extraction: PyMuPDF
  - Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
  - Vector DB: FAISS
  - Claim verification: Hugging Face NLI (cross-encoder/nli-deberta-v3-small)
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

# ── Load .env from backend directory ──────────────────────────────────────────
from dotenv import load_dotenv
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(Path(__file__).parent / ".env.example")

from .config import settings
from .models.database import create_tables
from .routers.research import router as research_router


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== ResearchLens Backend starting ===")
    logger.info("Database: %s", settings.DATABASE_URL)
    logger.info("Gemini configured: %s", settings.is_gemini_configured)
    logger.info("NLI model: %s", settings.NLI_MODEL)
    logger.info("Embedding model: %s", settings.EMBEDDING_MODEL)

    # Create DB tables
    create_tables()
    logger.info("Database tables ready")

    # Ensure PDF cache directory exists
    Path(settings.PDF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

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
# Allows the Vite dev server (localhost:517x) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(research_router)


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
