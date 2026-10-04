import React, { useState } from 'react';
import { Citation, CitationStatus } from '../../types';

interface CitationBadgeProps {
  citation: Citation;
  onClick: (citation: Citation) => void;
}

export const CitationBadge: React.FC<CitationBadgeProps> = ({ citation, onClick }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  const getStatusConfig = (status: CitationStatus) => {
    switch (status) {
      case 'verified':
        return { symbol: '✓', label: 'Verified', badgeClass: 'bg-black text-white border-black' };
      case 'partially_supported':
        return { symbol: '~', label: 'Partially Supported', badgeClass: 'bg-gray-600 text-white border-gray-600' };
      case 'unsupported':
        return { symbol: '✕', label: 'Unsupported', badgeClass: 'bg-white text-black border-black' };
      case 'contradicted':
        return { symbol: '↔', label: 'Contradicted', badgeClass: 'bg-gray-200 text-gray-800 border-gray-400' };
      default:
        return { symbol: '?', label: 'Unknown', badgeClass: 'bg-gray-100 text-gray-500 border-gray-300' };
    }
  };

  const config = getStatusConfig(citation.status);

  return (
    <span className="relative inline-block mx-0.5 align-baseline">
      <button
        type="button"
        onClick={() => onClick(citation)}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onFocus={() => setShowTooltip(true)}
        onBlur={() => setShowTooltip(false)}
        className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[11px] font-mono font-semibold border citation-mark ${config.badgeClass}`}
        aria-label={`Citation [${citation.badgeNumber}] ${config.label}`}
      >
        [{citation.badgeNumber} {config.symbol}]
      </button>

      {showTooltip && (
        <div
          role="tooltip"
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-3 bg-white border border-gray-300 rounded shadow-lg z-50 pointer-events-none text-left text-xs"
        >
          <div className="flex items-center justify-between border-b border-gray-100 pb-1.5 mb-2">
            <span className="font-semibold text-gray-900">{config.label}</span>
            <span className="text-gray-400 font-mono">p. {citation.page}</span>
          </div>
          <div className="text-gray-700 font-medium mb-0.5 line-clamp-1">{citation.paperTitle}</div>
          <div className="text-gray-400 mb-2">{citation.authors} ({citation.year})</div>
          <div className="p-2 rounded border border-gray-100 bg-gray-50 text-[10px] text-gray-600 italic line-clamp-2 leading-relaxed">
            "{citation.highlightSentence || citation.passage}"
          </div>
          <div className="mt-1.5 text-[10px] text-gray-400 text-right">Click to inspect →</div>
        </div>
      )}
    </span>
  );
};
