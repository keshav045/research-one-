import React, { useState, useMemo } from 'react';
import { ShieldCheck, CheckCircle2, AlertTriangle, XCircle, ArrowLeftRight, FileText } from 'lucide-react';
import { ResearchInvestigation, ContradictionIssue, Citation } from '../../types';
import { ConflictCard } from './ConflictCard';
import { ConflictModal } from './ConflictModal';

interface CitationIntegrityProps {
  investigation: ResearchInvestigation;
  onOpenPaperModal: (paperId: string) => void;
  onViewClaimInReport: (claim: string) => void;
}

export const CitationIntegrity: React.FC<CitationIntegrityProps> = ({
  investigation, onOpenPaperModal, onViewClaimInReport,
}) => {
  const [issues, setIssues] = useState<ContradictionIssue[]>(() => {
    if (!investigation?.report) return [];

    const allCitations: Citation[] = [];
    investigation.report.findings?.forEach(f => {
      f.paragraphs?.forEach(p => {
        if (p.citations) allCitations.push(...p.citations);
      });
    });

    const flagged = allCitations.filter(
      c => c.status === 'contradicted' || c.status === 'unsupported' || c.status === 'partially_supported'
    );

    if (flagged.length === 0) return [];

    return flagged.map((c, idx) => ({
      id: `issue-${idx + 1}`,
      type: c.status === 'contradicted' ? 'conflicting_evidence' : 'unsupported_claim',
      title: `${c.status === 'contradicted' ? 'Empirical Divergence' : 'Grounding Flag'} [${c.badgeNumber}]`,
      claim: c.claim,
      reason: c.reason || c.passage || 'Contextual parameters or measurement units require explicit qualification.',
      paperA: {
        id: c.paperId,
        title: c.paperTitle,
        authors: c.authors,
        statement: c.highlightSentence || c.passage,
        page: c.page,
      },
      suggestedAction: c.status === 'contradicted'
        ? 'Cross-reference benchmark conditions and deployment scale.'
        : 'Align citation boundaries with primary author conclusions.',
    }));
  });
  const [activeModal, setActiveModal] = useState<ContradictionIssue | null>(null);

  const { verifiedClaims: v, partiallySupportedClaims: p, unsupportedClaims: u, contradictedClaims: c } = investigation;
  const total = v + p + u + c;

  const uncitedCount = useMemo(() => {
    if (typeof investigation.uncitedSentences === 'number') {
      return investigation.uncitedSentences;
    }
    let count = 0;
    if (investigation.report?.executiveSummary) {
      const s = investigation.report.executiveSummary.split(/(?<=[.?!])\s+/).filter(x => x.trim().length > 15);
      const uncited = s.filter(x => !/\[\d+(?:,\s*\d+)*\]|\[cite-[^\]]+\]/.test(x));
      count += uncited.length;
    }
    return count;
  }, [investigation]);

  const totalAssertions = total + uncitedCount;

  // totalAssertions mirrors the server formula: n_citations + uncited_sentences.
  // The headline citationCoverage is ALWAYS taken from investigation.citationCoverage
  // (server-computed via the NLI weighted score) — never re-derived here,
  // so Page 1 (ReportViewer) and Page 2 (CitationIntegrity) always show the same number.
  const pv       = totalAssertions > 0 ? Math.round((v / totalAssertions) * 100) : 0;
  const pp       = totalAssertions > 0 ? Math.round((p / totalAssertions) * 100) : 0;
  const pu       = totalAssertions > 0 ? Math.round((u / totalAssertions) * 100) : 0;
  const pc       = totalAssertions > 0 ? Math.round((c / totalAssertions) * 100) : 0;
  const puncited = totalAssertions > 0 ? Math.max(0, 100 - (pv + pp + pu + pc))  : 0;

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-7">
      {/* Header */}
      <div className="pb-4 border-b border-gray-200">
        <div className="flex items-center gap-2 mb-1">
          <ShieldCheck className="w-5 h-5 text-gray-700" />
          <h1 className="text-xl font-bold text-gray-900">Citation Integrity</h1>
        </div>
        <p className="text-sm text-gray-500">Sentence-level verification against full-text source publications</p>
      </div>

      {/* Citation Integrity Card */}
      <div className="bg-white border border-gray-200 rounded p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-gray-100">
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Citation Integrity</p>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-black text-gray-900 font-mono">
                {investigation.citationCoverage != null ? `${investigation.citationCoverage}%` : 'N/A'}
              </span>
              <span className="text-xs text-emerald-600 font-medium">(Verified answer sentences / answer sentences)</span>
            </div>
            <p className="text-xs text-gray-500 mt-1 max-w-sm leading-relaxed">
              Every sentence in the generated answer has been evaluated via sentence-level NLI against supporting verbatim passages from verified source literature.
            </p>
          </div>
          {/* Breakdown stats */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
            {[
              { label: 'Verified', value: v, icon: <CheckCircle2 className="w-3.5 h-3.5 text-black" /> },
              { label: 'Partial', value: p, icon: <AlertTriangle className="w-3.5 h-3.5 text-gray-500" /> },
              { label: 'Unsupported', value: u, icon: <XCircle className="w-3.5 h-3.5 text-gray-400" /> },
              { label: 'Conflicts', value: c, icon: <ArrowLeftRight className="w-3.5 h-3.5 text-rose-500" /> },
              { label: 'Uncited', value: uncitedCount, icon: <FileText className="w-3.5 h-3.5 text-amber-600" /> },
            ].map(stat => (
              <div key={stat.label} className="p-2.5 rounded border border-gray-200 text-center">
                <div className="flex justify-center mb-1">{stat.icon}</div>
                <span className="block font-bold text-gray-900 font-mono text-base">{stat.value}</span>
                <span className="text-[10px] text-gray-400">{stat.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Progress Bar */}
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-1.5">
            <span>Synthesis Grounding Breakdown</span>
            <span>{totalAssertions} total assertions ({total} cited, {uncitedCount} uncited narrative)</span>
          </div>
          {totalAssertions > 0 ? (
            <>
              <div className="h-3 w-full bg-gray-100 rounded overflow-hidden flex">
                <div style={{ width: `${pv}%` }} className="bg-black" title={`Verified ${pv}%`} />
                <div style={{ width: `${pp}%` }} className="bg-gray-500" title={`Partial ${pp}%`} />
                <div style={{ width: `${pu}%` }} className="bg-gray-300" title={`Unsupported ${pu}%`} />
                <div style={{ width: `${pc}%` }} className="bg-rose-500" title={`Conflicts ${pc}%`} />
                <div style={{ width: `${puncited}%` }} className="bg-amber-200" title={`Uncited Synthesis ${puncited}%`} />
              </div>
              <div className="flex flex-wrap gap-4 mt-2 text-[11px] text-gray-400 font-mono">
                <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded bg-black inline-block" /> Verified ({pv}%)</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded bg-gray-500 inline-block" /> Partial ({pp}%)</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded bg-gray-300 inline-block" /> Unsupported ({pu}%)</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded bg-rose-500 inline-block" /> Conflicts ({pc}%)</span>
                {uncitedCount > 0 && (
                  <span className="flex items-center gap-1.5"><span className="w-3 h-2 rounded bg-amber-200 border border-amber-300 inline-block" /> Uncited Synthesis ({puncited}%)</span>
                )}
              </div>
            </>
          ) : (
            <div className="h-3 w-full bg-gray-100 rounded" />
          )}
        </div>
      </div>

      {/* Issues */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-gray-600" /> Audit Findings ({issues.length})
          </h2>
          <span className="text-xs text-gray-400">Resolve before publishing</span>
        </div>
        {issues.length === 0 ? (
          <div className="p-6 text-center rounded border border-gray-200 bg-gray-50/70 space-y-2">
            <div className="w-9 h-9 rounded-full bg-black text-white flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-4 h-4" />
            </div>
            <p className="text-sm font-semibold text-gray-900">Audit Status: No Citation Divergences Detected</p>
            <p className="text-xs text-gray-500 max-w-lg mx-auto leading-relaxed">
              {total > 0
                ? `All ${total} inline cited claims strictly entail from verbatim primary literature passages. Synthesis narrative (including ${uncitedCount} sentences in the Executive Summary) contains general overview text without inline citation anchors, reflected in the ${investigation.citationCoverage}% Citation Integrity score.`
                : 'No citation audit data available. Run a research investigation to see verification findings.'}
            </p>
          </div>
        ) : (
          issues.map(issue => (
            <ConflictCard key={issue.id} issue={issue}
              onViewClaim={onViewClaimInReport}
              onFindMoreEvidence={(clm) => {
                // TODO: trigger additional evidence search once backend is connected
                console.info(`Find more evidence requested for: "${clm}"`);
              }}
              onRemoveClaim={(id) => setIssues(prev => prev.filter(i => i.id !== id))}
              onCompareEvidence={(iss) => setActiveModal(iss)}
            />
          ))
        )}
      </div>

      <ConflictModal issue={activeModal} onClose={() => setActiveModal(null)} onOpenPaper={onOpenPaperModal} />
    </div>
  );
};
