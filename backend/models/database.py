"""
ResearchLens Backend — SQLAlchemy Database Models & Session
Database: SQLite (default) | PostgreSQL (production)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..config import settings


# ─── Engine & Session Factory ─────────────────────────────────────────────────

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ─── ORM Models ───────────────────────────────────────────────────────────────


class ResearchJob(Base):
    """Represents one research investigation."""

    __tablename__ = "research_jobs"

    id = Column(String, primary_key=True)
    question = Column(Text, nullable=False)
    depth = Column(String, nullable=False)           # Quick | Standard | Deep
    sources = Column(Text, nullable=False)            # JSON list of sources
    status = Column(String, nullable=False, default="in_progress")
    anchor_paper_id = Column(String, nullable=True)
    papers_analyzed = Column(Integer, default=0)
    total_papers = Column(Integer, default=0)
    evidence_items = Column(Integer, default=0)
    verified_claims = Column(Integer, default=0)
    partially_supported_claims = Column(Integer, default=0)
    unsupported_claims = Column(Integer, default=0)
    contradicted_claims = Column(Integer, default=0)
    potential_conflicts = Column(Integer, default=0)
    citation_coverage = Column(Float, default=0.0)
    uncited_sentences = Column(Integer, default=0)
    passages_total = Column(Integer, default=0)
    stage_stats_json = Column(Text, default="[]")
    failure_reason = Column(Text, nullable=True)
    debug_json = Column(Text, nullable=True)
    pipeline_json = Column(Text, default="[]")       # JSON pipeline steps
    report_json = Column(Text, nullable=True)        # full JSON report when done
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_sources(self) -> list[str]:
        return json.loads(self.sources)

    def get_pipeline(self) -> list[dict]:
        return json.loads(self.pipeline_json or "[]")

    def set_pipeline(self, steps: list[dict]) -> None:
        self.pipeline_json = json.dumps(steps)

    def get_report(self) -> Optional[dict]:
        return json.loads(self.report_json) if self.report_json else None

    def set_report(self, report: dict) -> None:
        self.report_json = json.dumps(report)

    def get_stage_stats(self) -> list[dict]:
        return json.loads(self.stage_stats_json or "[]")

    def set_stage_stats(self, stats: list[dict]) -> None:
        self.stage_stats_json = json.dumps(stats)

    def get_debug(self) -> dict:
        return json.loads(self.debug_json or "{}")

    def set_debug(self, d: dict) -> None:
        self.debug_json = json.dumps(d)


class PaperRecord(Base):
    """A retrieved academic paper (cached to avoid re-fetching)."""

    __tablename__ = "papers"

    id = Column(String, primary_key=True)           # e.g. arxiv:2306.00978
    job_id = Column(String, nullable=False, index=True)
    title = Column(Text, nullable=False)
    authors_json = Column(Text, nullable=False)     # JSON list
    publication_year = Column(Integer)
    journal_conference = Column(Text)
    doi = Column(Text)
    source = Column(String)                         # arXiv | Semantic Scholar
    abstract = Column(Text)
    pdf_url = Column(Text)
    full_text = Column(Text, nullable=True)         # extracted via PyMuPDF
    passages_json = Column(Text, nullable=True)     # JSON list of PaperPassage
    claims_supported_count = Column(Integer, default=0)
    evidence_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def get_authors(self) -> list[str]:
        return json.loads(self.authors_json)

    def get_passages(self) -> list[dict]:
        return json.loads(self.passages_json) if self.passages_json else []


# ─── Helpers ──────────────────────────────────────────────────────────────────


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
