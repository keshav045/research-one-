"""FastAPI router for research job management."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..config import settings
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
        sources=json.dumps([s.value for s in req.sources]),
        status="in_progress",
        total_papers=settings.DEPTH_COUNTS.get(req.depth.value, 12),
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
    limit: int = Query(20, ge=1, le=100),
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
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Research job '{job_id}' not found")

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


# ─── Evidence for a Job ───────────────────────────────────────────────────────


@router.get("/{job_id}/evidence")
async def get_job_evidence(
    job_id: str,
    db: Session = Depends(get_db),
) -> dict:
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Research job '{job_id}' not found")

    report_data = job.get_report()
    citations = []
    if report_data and "findings" in report_data:
        for sec in report_data.get("findings", []):
            for para in sec.get("paragraphs", []):
                for cite in para.get("citations", []):
                    citations.append(cite)

    records = db.query(PaperRecord).filter(PaperRecord.job_id == job_id).all()
    passages = []
    for r in records:
        for p in r.get_passages():
            passages.append({
                "paper_id": r.id.split("::")[-1],
                "paper_title": r.title,
                **p,
            })

    return {
        "job_id": job.id,
        "evidence_items": job.evidence_items,
        "citations": citations,
        "passages": passages,
    }


# ─── Claims for a Job ─────────────────────────────────────────────────────────


@router.get("/{job_id}/claims")
async def get_job_claims(
    job_id: str,
    db: Session = Depends(get_db),
) -> dict:
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Research job '{job_id}' not found")

    report_data = job.get_report()
    claims = []
    if report_data and "findings" in report_data:
        for sec in report_data.get("findings", []):
            for para in sec.get("paragraphs", []):
                for cite in para.get("citations", []):
                    claims.append({
                        "badge_number": cite.get("badgeNumber"),
                        "claim": cite.get("claim"),
                        "status": cite.get("status"),
                        "paper_id": cite.get("paperId"),
                        "paper_title": cite.get("paperTitle"),
                        "page": cite.get("page"),
                        "passage": cite.get("passage"),
                    })

    debug_blob = job.get_debug()
    return {
        "job_id": job.id,
        "total_claims": len(claims),
        "verified_claims": job.verified_claims,
        "partially_supported_claims": job.partially_supported_claims,
        "unsupported_claims": job.unsupported_claims,
        "contradicted_claims": job.contradicted_claims,
        "claims": claims,
        "rejected_claims": debug_blob.get("rejected_claims", []),
    }


# ─── Report for a Job ─────────────────────────────────────────────────────────


@router.get("/{job_id}/report")
async def get_job_report(
    job_id: str,
    db: Session = Depends(get_db),
) -> dict:
    job = db.query(ResearchJob).filter(ResearchJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Research job '{job_id}' not found")

    report_data = job.get_report()
    if not report_data:
        raise HTTPException(status_code=404, detail=f"Report not yet available for job '{job_id}' (status: {job.status})")
    return report_data


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


# ─── Direct Compatibility Router (/research) ──────────────────────────────────

direct_router = APIRouter(prefix="/research", tags=["research-direct"])
direct_router.add_api_route("", start_research, methods=["POST"], response_model=ResearchInvestigation, status_code=status.HTTP_202_ACCEPTED)
direct_router.add_api_route("/{job_id}", get_research, methods=["GET"], response_model=ResearchInvestigation)
direct_router.add_api_route("", list_research, methods=["GET"], response_model=list[ResearchInvestigation])
direct_router.add_api_route("/{job_id}", delete_research, methods=["DELETE"])
direct_router.add_api_route("/{job_id}/papers", get_job_papers, methods=["GET"], response_model=list[Paper])
direct_router.add_api_route("/{job_id}/evidence", get_job_evidence, methods=["GET"])
direct_router.add_api_route("/{job_id}/claims", get_job_claims, methods=["GET"])
direct_router.add_api_route("/{job_id}/report", get_job_report, methods=["GET"])
direct_router.add_api_route("/{job_id}/debug", get_job_debug, methods=["GET"])

