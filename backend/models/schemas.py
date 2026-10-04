"""Pydantic schemas — shared between API request/response and internal services."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Enumerations ─────────────────────────────────────────────────────────────


class ResearchDepth(str, Enum):
    QUICK = "Quick"
    STANDARD = "Standard"
    DEEP = "Deep"


class ResearchSource(str, Enum):
    ARXIV = "arXiv"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    OPENALEX = "OpenAlex"
    UPLOADED = "Uploaded Papers"


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class EntailmentVerdict(str, Enum):
    ENTAILS = "entails"
    NEUTRAL = "neutral"
    CONTRADICTS = "contradicts"
    UNSUPPORTED = "unsupported"


# ─── Pipeline Step ────────────────────────────────────────────────────────────


class PipelineStep(BaseModel):
    id: str
    name: str
    status: StepStatus
    description: str
    timestamp: Optional[str] = None
    iconName: Optional[str] = None


# ─── Data Contracts (Implementation Plan) ────────────────────────────────────


class Passage(BaseModel):
    id: str
    paper_id: str
    page: int
    section: str = ""
    text: str
    embedding_idx: Optional[int] = None
    source_kind: str = "pdf"  # "pdf" or "abstract"


class Evidence(BaseModel):
    claim_id: str
    passage_id: str
    nli_label: str  # "entails" | "neutral" | "contradicts"
    entail_prob: float
    rerank_score: float = 0.0


class Claim(BaseModel):
    id: str
    text: str
    sub_question: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    status: CitationStatus = CitationStatus.UNSUPPORTED


class StageStat(BaseModel):
    name: str
    started_at: str
    duration_ms: int
    in_count: int
    out_count: int
    error: Optional[str] = None


class JobResult(BaseModel):
    papers: list[Paper] = Field(default_factory=list)
    passages_total: int = 0
    claims: list[Claim] = Field(default_factory=list)
    answer_text: str = ""
    status: str = "completed"  # completed | insufficient_evidence | failed
    stage_stats: list[StageStat] = Field(default_factory=list)
    failure_reason: Optional[str] = None


# ─── Paper / Passage ──────────────────────────────────────────────────────────


class PaperPassage(BaseModel):
    id: str
    page: int
    section: str = ""
    text: str
    paper_id: Optional[str] = None
    embedding_idx: Optional[int] = None
    source_kind: str = "pdf"


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    publicationYear: int
    journalConference: str
    doi: str
    source: str
    abstract: str
    pdfUrl: Optional[str] = None
    citationCount: int = 0
    claimsSupportedCount: int = 0
    evidenceCount: int = 0
    passages: Optional[list[PaperPassage]] = None
    passages_json: Optional[str] = None
    fullText: Optional[str] = None


# ─── Citation & NLI ───────────────────────────────────────────────────────────


class AtomicClaimVerification(BaseModel):
    id: str
    atomicClaim: str
    matchedSentence: str
    verdict: EntailmentVerdict
    confidence: float
    reasoning: str


class Citation(BaseModel):
    id: str
    badgeNumber: int
    claim: str
    status: CitationStatus
    paperId: str
    paperTitle: str
    authors: str
    year: int
    page: int
    passage: str
    highlightSentence: str
    reason: Optional[str] = None
    atomicClaims: Optional[list[AtomicClaimVerification]] = None
    entailmentScore: Optional[float] = None
    extractionConfidence: Optional[float] = None


# ─── Report ───────────────────────────────────────────────────────────────────


class ComparisonRow(BaseModel):
    model: str
    architectureType: str
    dataset: str
    f1Score: str
    mapScore: str
    fpsThroughput: str
    parametersM: str
    gflops: str
    citationId: str


class ReportParagraph(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)


class ReportSection(BaseModel):
    sectionTitle: str
    paragraphs: list[ReportParagraph]


class ResearchReport(BaseModel):
    executiveSummary: str
    methodology: str
    findings: list[ReportSection]
    comparisonTable: list[ComparisonRow] = Field(default_factory=list)
    computationalRequirements: str = ""
    contradictoryEvidence: str = ""
    limitations: list[str] = Field(default_factory=list)
    conclusion: str
    references: list[Paper] = Field(default_factory=list)


# ─── Research Investigation ───────────────────────────────────────────────────


class ResearchInvestigation(BaseModel):
    id: str
    question: str
    depth: ResearchDepth
    sources: list[ResearchSource]
    status: str  # in_progress | completed | insufficient_evidence | failed
    papersAnalyzed: int = 0
    totalPapers: int = 0
    evidenceItems: int = 0
    verifiedClaims: int = 0
    partiallySupportedClaims: int = 0
    unsupportedClaims: int = 0
    contradictedClaims: int = 0
    potentialConflicts: int = 0
    citationCoverage: float = 0.0
    uncitedSentences: int = 0
    passages_total: int = 0
    failure_reason: Optional[str] = None
    anchor_paper_id: Optional[str] = None
    stage_stats: list[StageStat] = Field(default_factory=list)
    createdAt: str
    updatedAt: str
    pipeline: list[PipelineStep] = Field(default_factory=list)
    report: Optional[ResearchReport] = None
    debug: Optional[dict[str, Any]] = None



# ─── API Request/Response Schemas ─────────────────────────────────────────────


class StartResearchRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500)
    depth: ResearchDepth = ResearchDepth.STANDARD
    sources: list[ResearchSource] = Field(
        default=[ResearchSource.ARXIV, ResearchSource.SEMANTIC_SCHOLAR, ResearchSource.OPENALEX]
    )


class GenerateReportRequest(BaseModel):
    investigation_id: str


class IntegrityMetrics(BaseModel):
    citationCoverage: float
    verifiedClaims: int
    partiallySupportedClaims: int
    unsupportedClaims: int
    contradictedClaims: int
    totalAtomicClaims: int
    entailedAtomicClaims: int
    potentialConflicts: int
    uncitedSentences: int
