import React, { useState } from 'react';
import { X, BookOpen, ExternalLink, Bookmark, BookmarkCheck, FileText, Layers, CheckCircle, Copy, Check } from 'lucide-react';
import { Paper, EvidenceItem, Citation } from '../../types';
import { SourceBadge } from '../common/SourceBadge';

interface PaperDetailsModalProps {
  paper: Paper | null;
  evidenceItems: EvidenceItem[];
  citations: Citation[];
  onClose: () => void;
}

export const PaperDetailsModal: React.FC<PaperDetailsModalProps> = ({ paper, evidenceItems, citations, onClose }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'evidence' | 'claims' | 'metadata'>('overview');
  const [isSaved, setIsSaved] = useState(false);
  const [copiedBib, setCopiedBib] = useState(false);

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  if (!paper) return null;

  const paperEvidence = evidenceItems.filter(e => e.paperId === paper.id);
  const paperCitations = citations.filter(c => c.paperId === paper.id);

  const bibtex = `@article{${paper.id},
  title={${paper.title}},
  author={${paper.authors.join(' and ')}},
  journal={${paper.journalConference}},
  year={${paper.publicationYear}},
  doi={${paper.doi}}
}`;

  const copyBibtex = () => {
    navigator.clipboard.writeText(bibtex);
    setCopiedBib(true);
    setTimeout(() => setCopiedBib(false), 2000);
  };

  return (
    <div onClick={onClose} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div onClick={e => e.stopPropagation()} className="bg-white border border-gray-300 rounded max-w-3xl w-full max-h-[90vh] flex flex-col shadow-xl overflow-hidden">
        {/* Header */}
        <div className="p-5 border-b border-gray-200">
          <div className="flex items-start justify-between gap-4 mb-3">
            <div className="flex items-center gap-2">
              <SourceBadge source={paper.source} />
              <span className="text-[11px] font-mono text-gray-400">{paper.doi}</span>
            </div>
            <button onClick={onClose} className="p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100" aria-label="Close">
              <X className="w-5 h-5" />
            </button>
          </div>
          <h2 className="text-lg font-bold text-gray-900 leading-snug mb-1">{paper.title}</h2>
          <p className="text-sm text-gray-500">{paper.authors.join(", ")} • {paper.journalConference} ({paper.publicationYear})</p>

          <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-gray-100">
            <a href={paper.pdfUrl || `https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-black bg-black text-white text-xs font-medium hover:bg-gray-800 transition-colors">
              <FileText className="w-3.5 h-3.5" /> Open PDF
            </a>
            <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-gray-300 bg-white text-xs font-medium text-gray-700 hover:border-black hover:text-black transition-colors">
              <ExternalLink className="w-3.5 h-3.5" /> DOI
            </a>
            <button onClick={() => setIsSaved(!isSaved)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-gray-300 bg-white text-xs font-medium text-gray-700 hover:border-black hover:text-black transition-colors">
              {isSaved ? <><BookmarkCheck className="w-3.5 h-3.5" /> Saved</> : <><Bookmark className="w-3.5 h-3.5" /> Save</>}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center px-5 border-b border-gray-200 text-xs">
          {([
            { id: 'overview', label: 'Overview', icon: BookOpen },
            { id: 'evidence', label: `Evidence (${paperEvidence.length})`, icon: Layers },
            { id: 'claims', label: `Claims (${paperCitations.length})`, icon: CheckCircle },
            { id: 'metadata', label: 'BibTeX', icon: FileText },
          ] as const).map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 py-2.5 px-3 font-medium border-b-2 transition-colors ${isActive ? 'border-black text-black' : 'border-transparent text-gray-400 hover:text-gray-700'}`}>
                <Icon className="w-3.5 h-3.5" /> {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 text-sm text-gray-700">
          {activeTab === 'overview' && (
            <div className="space-y-4">
              <p className="leading-relaxed">{paper.abstract}</p>
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded border border-gray-200 bg-gray-50">
                  <p className="text-[10px] text-gray-400 mb-0.5">Evidence Points</p>
                  <p className="font-bold text-gray-900 font-mono">{paperEvidence.length} items</p>
                </div>
                <div className="p-3 rounded border border-gray-200 bg-gray-50">
                  <p className="text-[10px] text-gray-400 mb-0.5">Claims Verified</p>
                  <p className="font-bold text-gray-900 font-mono">{paper.claimsSupportedCount} claims</p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'evidence' && (
            <div className="space-y-3">
              {paperEvidence.length === 0
                ? <p className="text-center py-6 text-gray-400">No evidence units for this paper.</p>
                : paperEvidence.map(e => (
                  <div key={e.id} className="p-3 rounded border border-gray-200 bg-gray-50 space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold">Evidence #{e.evidenceNumber}</span>
                      <span className="font-mono text-gray-500">p. {e.page}</span>
                    </div>
                    <p className="font-medium text-gray-900 text-sm">{e.model} → <span className="font-bold">{e.value} {e.metric}</span> on {e.dataset}</p>
                    <p className="text-xs text-gray-500 italic">"{e.passage}"</p>
                  </div>
                ))}
            </div>
          )}

          {activeTab === 'claims' && (
            <div className="space-y-3">
              {paperCitations.length === 0
                ? <p className="text-center py-6 text-gray-400">No citation claims for this paper.</p>
                : paperCitations.map(c => (
                  <div key={c.id} className="p-3 rounded border border-gray-200 bg-gray-50 space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="font-bold">Claim #{c.badgeNumber}</span>
                      <span className="uppercase font-mono text-gray-500">{c.status}</span>
                    </div>
                    <p className="font-medium text-gray-900 leading-relaxed">"{c.claim}"</p>
                    <p className="text-[11px] text-gray-500 italic">"{c.passage}"</p>
                  </div>
                ))}
            </div>
          )}

          {activeTab === 'metadata' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-2 p-3 rounded border border-gray-200 bg-gray-50 text-xs font-mono">
                <div>DOI: {paper.doi}</div>
                <div>Source: {paper.source}</div>
                <div>Year: {paper.publicationYear}</div>
                <div className="col-span-2">Venue: {paper.journalConference}</div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">BibTeX</p>
                  <button onClick={copyBibtex} className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-black">
                    {copiedBib ? <><Check className="w-3 h-3 text-black" /> Copied</> : <><Copy className="w-3 h-3" /> Copy</>}
                  </button>
                </div>
                <pre className="p-3 rounded border border-gray-200 bg-gray-50 text-[11px] font-mono text-gray-700 overflow-x-auto whitespace-pre-wrap">{bibtex}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
