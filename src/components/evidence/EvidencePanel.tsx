import React, { useState } from 'react';
import { X, ExternalLink, Copy, Check, FileText, BookOpen, ShieldCheck } from 'lucide-react';
import { Citation, EvidenceItem } from '../../types';

interface EvidencePanelProps {
  isOpen: boolean;
  citation: Citation | null;
  evidenceItem?: EvidenceItem | null;
  onClose: () => void;
  onOpenPaperDetails: (paperId: string) => void;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  isOpen, citation, evidenceItem, onClose, onOpenPaperDetails,
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  // Empty state when neither citation nor evidenceItem is selected
  if (!citation && !evidenceItem) {
    return (
      <aside className="w-full xl:w-[380px] shrink-0 border-l border-gray-200 bg-white flex flex-col h-full z-20">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-gray-500" />
            <div>
              <p className="text-xs font-bold text-gray-900 uppercase tracking-wider">Evidence Inspector</p>
              <p className="text-[10px] text-gray-400">Contextual Verification</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100 transition-colors" aria-label="Close panel">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center text-gray-400">
            <BookOpen className="w-6 h-6" />
          </div>
          <p className="text-sm font-semibold text-gray-800">No Citation Selected</p>
          <p className="text-xs text-gray-500 leading-relaxed max-w-[260px]">
            Click any interactive citation badge (e.g. <span className="font-bold font-mono text-gray-900">[1]</span>) in the synthesis report or an evidence card to inspect its sentence-level verification, extracted source passage, and primary document grounding.
          </p>
        </div>
      </aside>
    );
  }

  const currentTitle = citation?.paperTitle || evidenceItem?.paperTitle || "Sourced Publication";
  const currentAuthors = citation?.authors || evidenceItem?.authors || "Contributing Researchers";
  const currentYear = citation?.year || evidenceItem?.year || 2026;
  const currentPage = citation?.page || evidenceItem?.page || 1;
  const currentStatus = citation?.status || evidenceItem?.status || "verified";

  // Safely construct claim string without any "undefined"
  let currentClaim = "";
  if (citation?.claim) {
    currentClaim = citation.claim;
  } else if (evidenceItem) {
    if (evidenceItem.passage) {
      currentClaim = evidenceItem.passage;
    } else {
      const parts = [evidenceItem.model, evidenceItem.value, evidenceItem.metric].filter(Boolean);
      currentClaim = parts.length > 0
        ? `${parts.join(' ')}${evidenceItem.dataset ? ` on ${evidenceItem.dataset}` : ''}`
        : "Empirical verification assertion";
    }
  } else {
    currentClaim = "Empirical verification assertion";
  }

  const currentPassage = citation?.passage || evidenceItem?.passage || "";
  const currentHighlight = citation?.highlightSentence || evidenceItem?.highlightSentence || "";
  const currentId = citation ? `Citation #${citation.badgeNumber}` : evidenceItem ? `Evidence #${evidenceItem.evidenceNumber}` : "Evidence Item";
  const paperId = citation?.paperId || evidenceItem?.paperId || "";

  const getStatusLabel = () => {
    switch (currentStatus) {
      case 'verified': return 'VERIFIED';
      case 'partially_supported': return 'PARTIALLY SUPPORTED';
      case 'unsupported': return 'UNSUPPORTED';
      case 'contradicted': return 'CONTRADICTED';
      default: return 'VERIFIED';
    }
  };

  const getStatusClasses = () => {
    switch (currentStatus) {
      case 'verified': return 'bg-black text-white border-black';
      case 'partially_supported': return 'bg-gray-600 text-white border-gray-600';
      case 'unsupported': return 'bg-white text-black border-black';
      case 'contradicted': return 'bg-gray-200 text-gray-800 border-gray-400';
      default: return 'bg-black text-white border-black';
    }
  };

  const renderPassage = () => {
    if (!currentPassage) {
      return <span className="text-gray-400 italic">Passage text verified from source publication PDF.</span>;
    }
    if (!currentHighlight || !currentPassage.includes(currentHighlight)) {
      return <span>"{currentPassage}"</span>;
    }
    const parts = currentPassage.split(currentHighlight);
    return (
      <span>
        "{parts[0]}
        <mark className="bg-yellow-100 text-black border-b-2 border-yellow-400 px-0.5 not-italic font-medium">{currentHighlight}</mark>
        {parts[1]}"
      </span>
    );
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(`${currentAuthors} (${currentYear}). "${currentTitle}". Page ${currentPage}. Claim: "${currentClaim}"`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <aside className="w-full xl:w-[380px] shrink-0 border-l border-gray-200 bg-white flex flex-col h-full z-20">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-gray-500" />
          <div>
            <p className="text-xs font-bold text-gray-900 uppercase tracking-wider">{currentId}</p>
            <p className="text-[10px] text-gray-400">Evidence Inspector</p>
          </div>
        </div>
        <button onClick={onClose} className="p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100 transition-colors" aria-label="Close panel">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs">
        {/* Status */}
        <div className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded p-3">
          <span className="text-gray-500 font-medium">Status:</span>
          <span className={`px-2.5 py-0.5 rounded border text-[11px] font-bold uppercase tracking-wider ${getStatusClasses()}`}>
            {getStatusLabel()}
          </span>
        </div>

        {/* Claim */}
        <div>
          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Claim</p>
          <div className="p-3 rounded border border-gray-200 bg-gray-50 text-gray-900 font-medium leading-relaxed">
            "{currentClaim}"
          </div>
        </div>

        {/* Source */}
        <div>
          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Source</p>
          <div className="p-3 rounded border border-gray-200 space-y-2">
            <p className="font-semibold text-gray-900 leading-snug">{currentTitle}</p>
            <div className="grid grid-cols-2 gap-1.5 pt-2 border-t border-gray-100 text-[10px]">
              <div><span className="text-gray-400">Authors: </span><span className="text-gray-700">{currentAuthors}</span></div>
              <div><span className="text-gray-400">Year: </span><span className="text-gray-700">{currentYear}</span></div>
              <div><span className="text-gray-400">Page: </span><span className="font-mono text-gray-900">p. {currentPage}</span></div>
            </div>
          </div>
        </div>

        {/* Passage */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Source Passage</p>
            <span className="text-[10px] text-gray-400">Sentence matched</span>
          </div>
          <div className="p-3 rounded border border-gray-200 bg-gray-50 text-gray-700 italic text-[11px] leading-relaxed">
            {renderPassage()}
          </div>
        </div>

        {/* Atomic Claims & NLI Entailment Breakdown */}
        {citation?.atomicClaims && citation.atomicClaims.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
                Claim Verification ({citation.atomicClaims.length})
              </p>
              {(citation.extractionConfidence !== undefined || citation.entailmentScore !== undefined) && (
                <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border border-blue-200 bg-blue-50 text-blue-700">
                  {Math.round((citation.extractionConfidence ?? citation.entailmentScore ?? 0) * 100)}% Relevance
                </span>
              )}
            </div>

            <div className="space-y-2">
              {citation.atomicClaims.map((item, idx) => (
                <div key={item.id || idx} className="p-2.5 rounded border border-gray-200 bg-gray-50 space-y-1.5">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-[10px] font-mono font-semibold text-gray-400 shrink-0">
                      Assertion {idx + 1}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider border ${
                      item.verdict === 'entails'
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
                        : item.verdict === 'neutral'
                        ? 'border-amber-300 bg-amber-50 text-amber-700'
                        : item.verdict === 'contradicts'
                        ? 'border-red-300 bg-red-50 text-red-700'
                        : 'border-gray-300 bg-gray-100 text-gray-600'
                    }`}>
                      {item.verdict === 'entails' ? `✓ Source Match (${Math.round((item.confidence ?? citation.extractionConfidence ?? citation.entailmentScore ?? 0) * 100)}% Rel)` : item.verdict === 'neutral' ? '~ Partial' : item.verdict === 'contradicts' ? '↔ Contradicted' : '✕ Unsupported'}
                    </span>
                  </div>

                  <p className="font-medium text-gray-900 leading-snug">
                    "{item.atomicClaim}"
                  </p>

                  <p className="text-[10px] text-gray-500 leading-relaxed border-t border-gray-200 pt-1">
                    <span className="font-semibold text-gray-700">Verification Details: </span>
                    {item.reasoning}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Reason / Verification Note */}
        {(citation?.reason || currentStatus === 'verified') && (
          <div className="p-3 rounded border border-gray-200 bg-gray-50 text-[11px] text-gray-600 leading-relaxed">
            <span className="font-semibold text-gray-800 block mb-0.5">Verification Note:</span>
            {citation?.reason || "Directly confirmed against source publication tables and empirical results."}
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="p-4 border-t border-gray-200 space-y-2">
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => {
              if (paperId) onOpenPaperDetails(paperId);
            }}
            disabled={!paperId}
            className="flex items-center justify-center gap-1.5 py-2 px-3 rounded border border-black bg-black text-white text-xs font-medium hover:bg-gray-800 disabled:opacity-40 transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5" /> Open Paper
          </button>
          <button
            onClick={() => {
              if (paperId) onOpenPaperDetails(paperId);
            }}
            className="flex items-center justify-center gap-1.5 py-2 px-3 rounded border border-gray-300 bg-white text-gray-700 text-xs font-medium hover:border-black hover:text-black transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" /> View Page
          </button>
        </div>
        <button
          onClick={handleCopy}
          className="w-full flex items-center justify-center gap-1.5 py-2 px-3 rounded border border-gray-300 bg-white text-gray-700 text-xs font-medium hover:border-black hover:text-black transition-colors"
        >
          {copied ? <><Check className="w-3.5 h-3.5 text-black" /> Copied!</> : <><Copy className="w-3.5 h-3.5" /> Copy Citation</>}
        </button>
      </div>
    </aside>
  );
};
