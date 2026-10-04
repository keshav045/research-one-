"""FastAPI router for research job management."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..models.database import ResearchJob, PaperRecord, get_db
from ..models.schemas import (
    Paper,
    ResearchInvestigation,
    StartResearchRequest,
)
from ..services.research_workflow import job_to_investigation, run_research_pipeline

router = APIRouter(prefix="/api/research", tags=["research"])


# ─── Start Research ────────────────────────────────────────────────────────────


@router.post("", response_model=ResearchInvestigation, status_code=status.HTTP_202_ACCEPTED)
async def start_research(
    req: StartResearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ResearchInvestigation:
    """
    Create a new research job and immediately start the pipeline in the background.
    Returns the job record (status=in_progress) straight away.
    """
    job_id = f"research-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    from ..services.research_workflow import _make_pipeline
    pipeline = _make_pipeline()

    job = ResearchJob(
        id=job_id,
        question=req.question,
        depth=req.depth.value,
        sources=str([s.value for s in req.sources]).replace("'", '"'),
        status="in_progress",
        total_papers={"Quick": 6, "Standard": 8, "Deep": 12}.get(req.depth.value, 8),
        created_at=now,
        updated_at=now,
    )
    job.set_pipeline(pipeline)
    db.add(job)
    db.commit()
    db.refresh(job)

    # Schedule the pipeline asynchronously — do NOT await here
    background_tasks.add_task(_run_pipeline_with_new_session, job_id)

    return job_to_investigation(job)


async def _run_pipeline_with_new_session(job_id: str) -> None:
    """Create a fresh DB session for the background task (thread-safety)."""
    from ..models.database import SessionLocal
    db = SessionLocal()
    try:
        await run_research_pipeline(job_id, db)
    finally:
        db.close()


# ─── Get Research Status ───────────────────────────────────────────────────────


@router.get("/{job_id}", response_model=ResearchInvestigation)
async def get_research(
    job_id: str,
    db: Session = Depends(get_db),
) -> ResearchInvestigation:
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Research job '{job_id}' not found")
    return job_to_investigation(job)


# ─── List All Research Jobs ────────────────────────────────────────────────────


@router.get("", response_model=list[ResearchInvestigation])
async def list_research(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[ResearchInvestigation]:
    jobs = (
        db.query(ResearchJob)
        .order_by(ResearchJob.created_at.desc())
        .limit(limit)
        .all()
    )
    return [job_to_investigation(j) for j in jobs]


# ─── Delete Research Job ───────────────────────────────────────────────────────


@router.delete("/{job_id}")
async def delete_research(
    job_id: str,
    db: Session = Depends(get_db),
) -> Response:
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Research job '{job_id}' not found")
    db.delete(job)
    db.commit()
    return Response(status_code=204)


# ─── Papers for a Job ─────────────────────────────────────────────────────────


@router.get("/{job_id}/papers", response_model=list[Paper])
async def get_job_papers(
    job_id: str,
    db: Session = Depends(get_db),
) -> list[Paper]:
    records = db.query(PaperRecord).filter(PaperRecord.job_id == job_id).all()
    papers: list[Paper] = []
    for r in records:
        authors = r.get_authors()
        passages = [
            {"id": p.get("id", ""), "page": p.get("page", 1), "section": p.get("section", ""), "text": p.get("text", "")}
            for p in r.get_passages()
        ]
        papers.append(
            Paper(
                id=r.id.split("::")[-1],
                title=r.title,
                authors=authors,
                publicationYear=r.publication_year or 2024,
                journalConference=r.journal_conference or "",
                doi=r.doi or "",
                source=r.source or "arXiv",
                abstract=r.abstract or "",
                pdfUrl=r.pdf_url,
                claimsSupportedCount=r.claims_supported_count,
                evidenceCount=r.evidence_count,
            )
        )
    return papers


# ─── Debug Diagnostics for a Job ──────────────────────────────────────────────


@router.get("/{job_id}/debug")
async def get_job_debug(
    job_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Phase 2 Observability: Returns full diagnostic info for a job including
    queries used, titles retrieved, per-paper passage counts, and stage timing.
    """
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Research job '{job_id}' not found")

    debug_blob = job.get_debug()
    return {
        "job_id": job.id,
        "question": job.question,
        "status": job.status,
        "failure_reason": job.failure_reason,
        # Counts
        "papers_analyzed": job.papers_analyzed,
        "passages_total": job.passages_total,
        "verified_claims": job.verified_claims,
        "partially_supported_claims": job.partially_supported_claims,
        "citation_coverage": job.citation_coverage,
        # Phase 2 fields — from debug_json
        "queries_used": debug_blob.get("queries_used", []),
        "titles_retrieved": debug_blob.get("titles_retrieved", []),
        "top_papers": debug_blob.get("top_papers", []),
        "passages_per_paper": debug_blob.get("passages_per_paper", {}),
        "retrieval_errors": debug_blob.get("retrieval_errors", []),
        # Timing breakdown
        "stage_stats": job.get_stage_stats(),
        # Full raw debug blob for advanced diagnostics
        "debug": debug_blob,
    }

