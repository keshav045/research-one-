import React from 'react';
import { ExternalLink } from 'lucide-react';
import { EvidenceItem } from '../../types';

interface EvidenceCardProps {
  item: EvidenceItem;
  onViewSource: (item: EvidenceItem) => void;
}

export const EvidenceCard: React.FC<EvidenceCardProps> = ({ item, onViewSource }) => {
  const getStatusBadge = () => {
    switch (item.status) {
      case 'verified': return <span className="px-1.5 py-0.5 rounded border border-black bg-black text-white text-[10px] font-mono">✓ Verified</span>;
      case 'partially_supported': return <span className="px-1.5 py-0.5 rounded border border-gray-500 bg-gray-100 text-gray-700 text-[10px] font-mono">~ Partial</span>;
      case 'unsupported': return <span className="px-1.5 py-0.5 rounded border border-black bg-white text-black text-[10px] font-mono">✕ Unsupported</span>;
      case 'contradicted': return <span className="px-1.5 py-0.5 rounded border border-gray-400 bg-gray-200 text-gray-700 text-[10px] font-mono">↔ Contradicted</span>;
      default: return null;
    }
  };

  return (
    <div className="rounded border border-gray-200 bg-white p-4 flex flex-col justify-between hover:border-gray-400 transition-colors">
      <div>
        <div className="flex items-center justify-between gap-2 pb-2.5 border-b border-gray-100 mb-2.5">
          <span className="text-xs font-bold text-gray-900 font-mono">Evidence #{item.evidenceNumber}</span>
          {getStatusBadge()}
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs mb-2.5 pb-2.5 border-b border-gray-100">
          <div><span className="text-gray-400 block text-[10px]">Model:</span><span className="font-semibold text-gray-900 truncate block">{item.model}</span></div>
          <div><span className="text-gray-400 block text-[10px]">Dataset:</span><span className="font-mono text-gray-700 truncate block">{item.dataset}</span></div>
          <div><span className="text-gray-400 block text-[10px]">Metric:</span><span className="text-gray-700 font-mono">{item.metric}</span></div>
          <div><span className="text-gray-400 block text-[10px]">Value:</span><span className="font-bold text-gray-900 font-mono">{item.value}</span></div>
        </div>
        <p className="text-[11px] font-semibold text-gray-800 line-clamp-1 mb-0.5">{item.paperTitle}</p>
        <p className="text-[10px] text-gray-400">{item.authors} ({item.year}) • Verbatim p. {item.page}</p>
        {item.atomicClaims && item.atomicClaims.length > 0 && (
          <span className="inline-block mt-1 text-[10px] font-mono text-gray-500 bg-gray-50 px-1.5 py-0.5 rounded border border-gray-100">
            {item.atomicClaims.filter(a => a.verdict === 'entails').length}/{item.atomicClaims.length} verified via source match
          </span>
        )}
      </div>
      <button
        onClick={() => onViewSource(item)}
        className="mt-3 w-full py-1.5 px-3 rounded border border-gray-300 hover:border-black text-gray-700 hover:text-black text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
      >
        <ExternalLink className="w-3 h-3" /> View Source
      </button>
    </div>
  );
};
