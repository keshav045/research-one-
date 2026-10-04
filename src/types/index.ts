export type CitationStatus = 'verified' | 'partially_supported' | 'unsupported' | 'contradicted';

export type ResearchDepth = 'Quick' | 'Standard' | 'Deep';

export const DEPTH_COUNTS: Record<ResearchDepth, number> = {
  Quick: 6,
  Standard: 12,
  Deep: 24,
};

export type ResearchSource = 'arXiv' | 'Semantic Scholar' | 'OpenAlex' | 'Uploaded Papers';

export type StepStatus = 'completed' | 'active' | 'pending' | 'failed';

export interface PipelineStep {
  id: string;
  name: string;
  status: StepStatus;
  description: string;
  timestamp?: string;
  iconName?: string;
}

export type EntailmentVerdict = 'entails' | 'neutral' | 'contradicts' | 'unsupported';

export interface AtomicClaimVerification {
  id: string;
  atomicClaim: string;
  matchedSentence: string;
  verdict: EntailmentVerdict;
  confidence: number;
  reasoning: string;
}

export interface PaperPassage {
  id: string;
  page: number;
  section: string;
  text: string;
}

export interface Citation {
  id: string; // e.g. "citation-001"
  badgeNumber: number; // e.g. 1
  claim: string;
  status: CitationStatus;
  paperId: string;
  paperTitle: string;
  authors: string;
  year: number;
  page: number;
  passage: string;
  highlightSentence: string;
  reason?: string;
  atomicClaims?: AtomicClaimVerification[];
  entailmentScore?: number;
  extractionConfidence?: number;
}

export interface EvidenceItem {
  id: string; // e.g. "evidence-001"
  evidenceNumber: number; // e.g. 23
  paperId: string;
  paperTitle: string;
  authors: string;
  year: number;
  model: string;
  dataset: string;
  metric: string;
  value: string;
  page: number;
  status: CitationStatus;
  passage: string;
  highlightSentence?: string;
  evidenceType?: string;
  atomicClaims?: AtomicClaimVerification[];
  entailmentScore?: number;
}

export interface Paper {
  id: string; // e.g. "paper-001"
  title: string;
  authors: string[];
  publicationYear: number;
  journalConference: string;
  doi: string;
  source: ResearchSource;
  abstract: string;
  pdfUrl?: string;
  claimsSupportedCount: number;
  evidenceCount: number;
  passages?: PaperPassage[];
  fullText?: string;
}

export interface ComparisonRow {
  model: string;
  architectureType: string;
  dataset: string;
  f1Score: string;
  mapScore: string;
  fpsThroughput: string;
  parametersM: string;
  gflops: string;
  citationId: string;
}

export interface ContradictionIssue {
  id: string;
  type: 'conflicting_evidence' | 'unsupported_claim';
  title: string;
  claim: string;
  reason?: string;
  paperA?: {
    id: string;
    title: string;
    authors: string;
    statement: string;
    page: number;
  };
  paperB?: {
    id: string;
    title: string;
    authors: string;
    statement: string;
    page: number;
  };
  suggestedAction?: string;
}

export interface ResearchReport {
  executiveSummary: string;
  methodology: string;
  findings: {
    sectionTitle: string;
    paragraphs: {
      text: string;
      citations: Citation[];
    }[];
  }[];
  comparisonTable: ComparisonRow[];
  computationalRequirements: string;
  contradictoryEvidence: string;
  limitations: string[];
  conclusion: string;
  references: Paper[];
}

export interface StageStat {
  name: string;
  started_at: string;
  duration_ms: number;
  in_count: number;
  out_count: number;
  error?: string;
}

export interface ResearchInvestigation {
  id: string;
  question: string;
  depth: ResearchDepth;
  sources: ResearchSource[];
  status: 'in_progress' | 'completed' | 'insufficient_evidence' | 'failed';
  papersAnalyzed: number;
  totalPapers: number;
  evidenceItems: number;
  verifiedClaims: number;
  partiallySupportedClaims: number;
  unsupportedClaims: number;
  contradictedClaims: number;
  potentialConflicts: number;
  citationCoverage: number; // e.g. 94 (Citation Integrity %)
  uncitedSentences?: number;
  passages_total?: number;
  failure_reason?: string;
  stage_stats?: StageStat[];
  debug?: Record<string, any>;
  createdAt: string;
  updatedAt: string;
  pipeline: PipelineStep[];
  report?: ResearchReport;
}

export interface AppSettings {
  llmProvider: 'Gemini' | 'OpenAI';
  geminiModel: string;
  openaiModel: string;
  maxPapers: number;
  researchDepth: ResearchDepth;
  maxIterations: number;
  sources: {
    arxiv: boolean;
    semanticScholar: boolean;
    uploadedDocuments: boolean;
  };
  appearance: 'Dark' | 'Light' | 'System';
}
