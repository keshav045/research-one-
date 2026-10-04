import React from 'react';
import {
  Layers, CheckCircle2, AlertTriangle, ExternalLink, BookOpen, ShieldCheck, ChevronRight, FileWarning
} from 'lucide-react';
import { ResearchInvestigation, Citation } from '../../types';
import { StatCard } from './StatCard';
import { CitationBadge } from './CitationBadge';
import { ComparisonTable } from './ComparisonTable';
import { SourceBadge } from '../common/SourceBadge';

interface ReportViewerProps {
  investigation: ResearchInvestigation;
  onSelectCitation: (citation: Citation) => void;
  onSelectCitationById: (id: string) => void;
  onOpenPaperModal: (paperId: string) => void;
  onOpenIntegrityDashboard: () => void;
}

export const ReportViewer: React.FC<ReportViewerProps> = ({
  investigation,
  onSelectCitation,
  onSelectCitationById,
  onOpenPaperModal,
  onOpenIntegrityDashboard,
}) => {
  const { report } = investigation;
  if (!report) return null;
  const isInsufficient = investigation.status === 'insufficient_evidence';

  const citationMap = new Map<number, Citation>();
  report.findings?.forEach(f => {
    f.paragraphs?.forEach(p => {
      p.citations?.forEach(c => {
        if (!citationMap.has(c.badgeNumber)) {
          citationMap.set(c.badgeNumber, c);
        }
      });
    });
  });

  const renderExecutiveSummary = () => {
    const text = report.executiveSummary;
    if (!text) return null;

    const regex = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        elements.push(text.slice(lastIndex, match.index));
      }
      const numbers = match[1].split(',').map(n => parseInt(n.trim(), 10)).filter(n => !isNaN(n));
      numbers.forEach((num, idx) => {
        const cite = citationMap.get(num);
        if (cite) {
          elements.push(
            <CitationBadge
              key={`exec-${num}-${match!.index}-${idx}`}
              citation={cite}
              onClick={onSelectCitation}
            />
          );
        } else {
          elements.push(
            <span
              key={`exec-raw-${num}-${match!.index}-${idx}`}
              className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-gray-100 text-gray-700 ml-0.5"
            >
              [{num}]
            </span>
          );
        }
      });
      lastIndex = regex.lastIndex;
    }

    if (lastIndex < text.length) {
      elements.push(text.slice(lastIndex));
    }

    return elements;
  };

  return (
    <article className="max-w-3xl mx-auto px-4 sm:px-8 py-8 space-y-10 text-gray-700 leading-relaxed">
      {/* Header — always shown */}
      <header className="space-y-4 pb-6 border-b border-gray-200">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className={`px-2 py-0.5 rounded border text-[11px] font-mono font-semibold uppercase ${
              isInsufficient ? 'border-orange-300 bg-orange-50 text-orange-700' :
              investigation.status === 'completed_with_warnings' ? 'border-amber-300 bg-amber-50 text-amber-800' :
              'border-gray-300 bg-gray-50 text-gray-600'
            }`}>
              {isInsufficient ? 'Insufficient Evidence' : (investigation.status === 'completed_with_warnings' ? 'Completed with Warnings' : 'Synthesis Report')}
            </span>
            <span className="text-gray-300">•</span>
            <span className="text-[11px] text-gray-400 font-mono">{investigation.id}</span>
          </div>
          <button
            onClick={onOpenIntegrityDashboard}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-gray-300 bg-white hover:border-black text-xs font-medium text-gray-700 hover:text-black transition-colors"
          >
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" /> Citation Integrity: {investigation.citationCoverage != null ? `${investigation.citationCoverage}%` : 'N/A'}
          </button>
        </div>

        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Research Results</p>
          <h1 className="text-xl sm:text-2xl font-bold text-gray-900 leading-snug">
            "{investigation.question}"
          </h1>
        </div>

        {/* Stat Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 pt-1">
          <StatCard label="Papers" value={investigation.papersAnalyzed} subtext="analyzed" icon={BookOpen} />
          <StatCard label="Evidence" value={investigation.evidenceItems} subtext="items" icon={Layers} />
          <StatCard label="Verified" value={investigation.verifiedClaims} subtext="claims" icon={CheckCircle2} />
          <StatCard label="Conflicts" value={investigation.potentialConflicts} icon={AlertTriangle} onClick={onOpenIntegrityDashboard} />
          <StatCard label="Citation Integrity" value={investigation.citationCoverage != null ? `${investigation.citationCoverage}%` : 'N/A'} icon={ShieldCheck} onClick={onOpenIntegrityDashboard} />
        </div>
      </header>

      {/* Completed with Warnings banner */}
      {investigation.status === 'completed_with_warnings' && (
        <section className="p-4 rounded-lg border border-amber-300 bg-amber-50 space-y-1.5">
          <div className="flex items-center gap-2 text-amber-900 font-semibold text-sm">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
            <span>Completed with warnings</span>
          </div>
          <p className="text-xs text-amber-800 leading-relaxed">
            {investigation.failure_reason || (investigation.debug?.status_reasons || []).join('; ') || 'Anchor or citation integrity constraints detected.'}
          </p>
        </section>
      )}

      {/* Phase 7: Insufficient Evidence notice — rendered instead of full report */}
      {isInsufficient ? (
        <>
          <section className="p-5 rounded-lg border-2 border-orange-200 bg-orange-50 space-y-3">
            <div className="flex items-center gap-2">
              <FileWarning className="w-5 h-5 text-orange-600 shrink-0" />
              <h2 className="text-base font-bold text-orange-900">Insufficient Evidence</h2>
            </div>
            {investigation.failure_reason && (
              <p className="text-sm text-orange-800 leading-relaxed">{investigation.failure_reason}</p>
            )}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-white/70 rounded-lg p-3 text-center">
                <div className="text-lg font-bold font-mono text-orange-700">{investigation.papersAnalyzed}</div>
                <div className="text-[10px] text-orange-600 uppercase tracking-wider">Papers Retrieved</div>
              </div>
              <div className="bg-white/70 rounded-lg p-3 text-center">
                <div className="text-lg font-bold font-mono text-orange-700">{investigation.passages_total ?? 0}</div>
                <div className="text-[10px] text-orange-600 uppercase tracking-wider">Passages Indexed</div>
              </div>
              <div className="bg-white/70 rounded-lg p-3 text-center">
                <div className="text-lg font-bold font-mono text-orange-700">{investigation.verifiedClaims}</div>
                <div className="text-[10px] text-orange-600 uppercase tracking-wider">Claims Verified</div>
              </div>
            </div>
            {report.executiveSummary && (
              <p className="text-sm text-orange-700 border-t border-orange-200 pt-3 leading-relaxed">{report.executiveSummary}</p>
            )}
          </section>

          {/* Always show retrieved papers even when evidence is insufficient */}
          {report.references && report.references.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
                <span className="text-gray-400 font-mono text-sm">01.</span>
                Retrieved Sources ({report.references.length})
              </h2>
              <p className="text-xs text-gray-500">These papers were retrieved but did not yield enough verifiable evidence to complete a report.</p>
              <div className="space-y-2">
                {report.references.map((paper, idx) => (
                  <div
                    key={paper.id}
                    onClick={() => onOpenPaperModal(paper.id)}
                    className="p-3 rounded border border-gray-200 hover:border-orange-300 bg-white transition-colors cursor-pointer flex items-start justify-between gap-3 group"
                  >
                    <div className="space-y-0.5 min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-[10px] font-mono font-semibold text-gray-400">[{idx + 1}]</span>
                        <SourceBadge source={paper.source} />
                      </div>
                      <p className="text-sm font-semibold text-gray-900 group-hover:underline leading-snug">{paper.title}</p>
                      <p className="text-xs text-gray-500">{paper.authors.slice(0, 3).join(', ')} • {paper.publicationYear}</p>
                    </div>
                    <ExternalLink className="w-3.5 h-3.5 text-gray-400 group-hover:text-black shrink-0 mt-1" />
                  </div>
                ))}
              </div>
            </section>
          )}
        </>
      ) : (
        <>
      {/* 1. Executive Summary */}
      <section className="space-y-2">
        <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
          <span className="text-gray-400 font-mono text-sm">01.</span> Executive Summary
        </h2>
        <div className="p-4 rounded border border-gray-200 bg-gray-50 text-sm text-gray-800 leading-relaxed">
          {renderExecutiveSummary()}
        </div>
      </section>

      {/* 2. Methodology */}
      <section className="space-y-3">
        <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
          <span className="text-gray-400 font-mono text-sm">02.</span> Research Methodology & Search Protocol
        </h2>
        <div className="p-4 rounded border border-gray-200 bg-gray-50 text-sm text-gray-800 leading-relaxed whitespace-pre-line space-y-2">
          {report.methodology}
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-1">
          <div className="p-2.5 rounded border border-gray-200 bg-white">
            <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400 block mb-0.5">Citation Transparency</span>
            <span className="text-xs font-semibold text-gray-900">Sentence-level Quote Grounding</span>
          </div>
          <div className="p-2.5 rounded border border-gray-200 bg-white">
            <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400 block mb-0.5">Reproducibility</span>
            <span className="text-xs font-semibold text-gray-900">Documented Queries & Criteria</span>
          </div>
          <div className="p-2.5 rounded border border-gray-200 bg-white">
            <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400 block mb-0.5">Research Traceability</span>
            <span className="text-xs font-semibold text-gray-900">Direct Page & DOI Mappings</span>
          </div>
        </div>
      </section>

      {/* 3. Findings */}
      <section className="space-y-4">
        <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
          <span className="text-gray-400 font-mono text-sm">03.</span> Research Findings & Citation Verification
        </h2>
        <div className="space-y-4">
          {report.findings.map((section, idx) => (
            <div key={idx} className="p-4 rounded border border-gray-200 space-y-3">
              <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-1.5">
                <ChevronRight className="w-4 h-4 text-gray-400" /> {section.sectionTitle}
              </h3>
              <div className="space-y-2 text-sm leading-relaxed text-gray-700">
                {section.paragraphs.map((p, pIdx) => (
                  <div key={pIdx} className="leading-relaxed">
                    <span>{p.text}</span>
                    {p.citations.map(cite => (
                      <CitationBadge key={cite.id} citation={cite} onClick={onSelectCitation} />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 4. Comparison Table */}
      {report.comparisonTable && report.comparisonTable.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
            <span className="text-gray-400 font-mono text-sm">04.</span> Performance Comparison
          </h2>
          <ComparisonTable rows={report.comparisonTable} onSelectCitation={onSelectCitationById} />
        </section>
      )}

      {/* 5. Computational Requirements */}
      {report.computationalRequirements && report.computationalRequirements.trim() && (
        <section className="space-y-2">
          <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
            <span className="text-gray-400 font-mono text-sm">05.</span> Computational Requirements
          </h2>
          <div className="p-4 rounded border border-gray-200 bg-gray-50 text-sm text-gray-700 leading-relaxed whitespace-pre-line">
            {report.computationalRequirements}
          </div>
        </section>
      )}

      {/* 6. Contradictory Evidence */}
      {report.contradictoryEvidence && report.contradictoryEvidence.trim() && report.contradictoryEvidence.toLowerCase() !== 'none' && (
        <section className="space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
              <span className="text-gray-400 font-mono text-sm">06.</span> Contradictory Evidence
            </h2>
            <button onClick={onOpenIntegrityDashboard} className="text-xs text-gray-500 hover:text-black underline">
              Resolve →
            </button>
          </div>
          <div className="p-4 rounded border border-gray-300 bg-gray-50 text-sm text-gray-700 leading-relaxed whitespace-pre-line">
            <p className="font-semibold text-gray-800 mb-1 flex items-center gap-1.5">
              <AlertTriangle className="w-4 h-4 text-gray-600" /> Literature Discrepancy Identified:
            </p>
            <p>{report.contradictoryEvidence}</p>
          </div>
        </section>
      )}

      {/* 7. Limitations */}
      {report.limitations && report.limitations.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
            <span className="text-gray-400 font-mono text-sm">07.</span> Limitations
          </h2>
          <ul className="space-y-1.5 text-sm text-gray-700">
            {report.limitations.map((limit, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="mt-2 w-1.5 h-1.5 rounded-full bg-gray-400 shrink-0" />
                {limit}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* 8. Conclusion */}
      <section className="space-y-2">
        <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
          <span className="text-gray-400 font-mono text-sm">08.</span> Conclusion & Decision Framework
        </h2>
        <div className="p-4 rounded border border-gray-200 bg-gray-50 text-sm text-gray-800 font-medium leading-relaxed whitespace-pre-line">
          {report.conclusion}
        </div>
      </section>

      {/* 9. References */}
      <section className="space-y-3 pt-4 border-t border-gray-200">
        <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
          <span className="text-gray-400 font-mono text-sm">09.</span> References ({report.references.length})
        </h2>
        <div className="space-y-3">
          {report.references.map((paper, idx) => (
            <div
              key={paper.id}
              onClick={() => onOpenPaperModal(paper.id)}
              className="p-4 rounded border border-gray-200 hover:border-black bg-white transition-colors cursor-pointer flex flex-col sm:flex-row sm:items-start justify-between gap-3 group"
            >
              <div className="space-y-1.5 min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[10px] font-mono font-semibold text-gray-500">[{idx + 1}]</span>
                  <SourceBadge source={paper.source} />
                  <span className="text-[10px] font-mono text-gray-400">{paper.doi}</span>
                </div>
                <p className="text-sm font-bold text-gray-900 group-hover:underline leading-snug">{paper.title}</p>
                <p className="text-xs text-gray-500">{paper.authors.join(", ")} • {paper.journalConference} ({paper.publicationYear})</p>
                {paper.abstract && (
                  <p className="text-xs text-gray-600 line-clamp-2 mt-1 leading-relaxed">
                    <span className="font-semibold text-gray-700">Contribution: </span>
                    {paper.abstract}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-1 text-xs text-gray-400 group-hover:text-black shrink-0 self-end sm:self-start">
                <span>View Details</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </div>
            </div>
          ))}
        </div>
      </section>
        </>
      )}
    </article>
  );
};
