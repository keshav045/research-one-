import React from 'react';
import { 
  Calendar, 
  BookOpen, 
  CheckCircle2, 
  ArrowRight, 
  ShieldCheck, 
  Layers, 
  Download, 
  Trash2 
} from 'lucide-react';
import { ResearchInvestigation } from '../../types';
import { exportReportToPdf } from '../../utils/exportPdf';

interface ResearchCardProps {
  item: ResearchInvestigation;
  onSelect: (item: ResearchInvestigation) => void;
  onDelete?: (item: ResearchInvestigation) => void;
}

export const ResearchCard: React.FC<ResearchCardProps> = ({ item, onSelect, onDelete }) => {
  const formattedDate = new Date(item.createdAt).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  const formattedTime = new Date(item.createdAt).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  const handleDownloadPdf = (e: React.MouseEvent) => {
    e.stopPropagation();
    exportReportToPdf(item);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onDelete) {
      onDelete(item);
    }
  };

  return (
    <div
      onClick={() => onSelect(item)}
      className="p-5 rounded-lg border border-gray-200 bg-white hover:border-gray-900 hover:shadow-sm transition-all duration-200 cursor-pointer flex flex-col justify-between gap-4 group"
    >
      {/* Top Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
          <span className="flex items-center gap-1 font-medium text-gray-700">
            <Calendar className="w-3.5 h-3.5 text-gray-400" />
            {formattedDate} · {formattedTime}
          </span>
          <span className="text-gray-300">•</span>
          <span className="px-2 py-0.5 rounded border border-gray-200 bg-gray-50 text-[11px] font-semibold text-gray-700">
            {item.depth} Depth
          </span>
          {item.sources && item.sources.map(src => (
            <span key={src} className="px-1.5 py-0.5 rounded bg-gray-100 text-[10px] text-gray-600 font-mono">
              {src}
            </span>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border border-emerald-200 bg-emerald-50 text-[11px] font-medium text-emerald-800">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block animate-pulse" /> Completed
          </span>
        </div>
      </div>

      {/* Main Question */}
      <div className="space-y-2">
        <h3 className="text-base font-bold text-gray-900 group-hover:text-black leading-snug">
          {item.question}
        </h3>

        {item.report?.executiveSummary && (
          <p className="text-xs text-gray-600 line-clamp-2 leading-relaxed bg-gray-50/70 p-2.5 rounded border border-gray-100">
            {item.report.executiveSummary}
          </p>
        )}
      </div>

      {/* Metrics Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 py-2.5 px-3 rounded bg-gray-50/50 border border-gray-100 text-xs">
        <div className="flex items-center gap-1.5">
          <BookOpen className="w-3.5 h-3.5 text-gray-400 shrink-0" />
          <span className="text-gray-500">Papers:</span>
          <strong className="text-gray-900 font-mono">{item.papersAnalyzed}</strong>
        </div>
        <div className="flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-gray-400 shrink-0" />
          <span className="text-gray-500">Evidence:</span>
          <strong className="text-gray-900 font-mono">{item.evidenceItems}</strong>
        </div>
        <div className="flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-gray-400 shrink-0" />
          <span className="text-gray-500">Verified:</span>
          <strong className="text-gray-900 font-mono">{item.verifiedClaims}</strong>
        </div>
        <div className="flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-gray-400 shrink-0" />
          <span className="text-gray-500">Citation Integrity:</span>
          <strong className="text-gray-900 font-mono">{item.citationCoverage}%</strong>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-100">
        <div className="flex items-center gap-2">
          {item.report && (
            <button
              onClick={handleDownloadPdf}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded border border-gray-200 bg-white text-xs font-medium text-gray-600 hover:text-black hover:border-black transition-colors"
              title="Download Report as PDF"
            >
              <Download className="w-3 h-3" />
              <span>PDF</span>
            </button>
          )}
          {onDelete && (
            <button
              onClick={handleDelete}
              className="p-1 rounded text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors"
              title="Delete investigation"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-gray-900 group-hover:translate-x-0.5 transition-transform">
          Open Synthesis Report <ArrowRight className="w-3.5 h-3.5" />
        </span>
      </div>
    </div>
  );
};
