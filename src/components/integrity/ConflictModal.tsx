import React from 'react';
import { X, GitCompare, BookOpen } from 'lucide-react';
import { ContradictionIssue } from '../../types';

interface ConflictModalProps {
  issue: ContradictionIssue | null;
  onClose: () => void;
  onOpenPaper: (paperId: string) => void;
}

export const ConflictModal: React.FC<ConflictModalProps> = ({ issue, onClose, onOpenPaper }) => {
  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  if (!issue) return null;

  return (
    <div onClick={onClose} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div onClick={e => e.stopPropagation()} className="bg-white border border-gray-300 rounded max-w-3xl w-full p-6 shadow-xl space-y-5">
        <div className="flex items-center justify-between pb-3 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-gray-600" />
            <div>
              <h3 className="text-base font-bold text-gray-900">Contradiction Comparison</h3>
              <p className="text-xs text-gray-400">Cross-document evidence reconciliation</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100" aria-label="Close">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div>
          <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Subject of Divergence</p>
          <div className="p-3 rounded border border-gray-200 bg-gray-50 text-sm text-gray-900 font-medium">
            "{issue.claim}"
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { paper: issue.paperA, label: 'Perspective A', borderClass: 'border-gray-300' },
            { paper: issue.paperB, label: 'Perspective B', borderClass: 'border-gray-400' },
          ].map(({ paper, label, borderClass }) => paper && (
            <div key={label} className={`p-4 rounded border ${borderClass} bg-gray-50 space-y-2.5`}>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-gray-700">{label}</span>
                <span className="text-[10px] text-gray-400">p. {paper.page}</span>
              </div>
              <p className="text-xs font-semibold text-gray-800 leading-snug">{paper.title}</p>
              <p className="text-[10px] text-gray-400">{paper.authors}</p>
              <div className="p-2.5 rounded border border-gray-200 bg-white text-[11px] text-gray-700 italic leading-relaxed">
                "{paper.statement}"
              </div>
              <button
                onClick={() => { onClose(); onOpenPaper(paper.id); }}
                className="w-full py-1.5 px-3 rounded border border-gray-300 hover:border-black text-xs text-gray-700 hover:text-black flex items-center justify-center gap-1.5 transition-colors"
              >
                <BookOpen className="w-3.5 h-3.5" /> Open Paper
              </button>
            </div>
          ))}
        </div>

        {issue.suggestedAction && (
          <div className="p-3.5 rounded border border-gray-200 bg-gray-50 text-xs text-gray-600 leading-relaxed">
            <span className="font-semibold text-gray-800 block mb-0.5">Resolution Note:</span>
            {issue.suggestedAction}
          </div>
        )}

        <div className="flex justify-end pt-2 border-t border-gray-100">
          <button onClick={onClose} className="px-4 py-2 rounded border border-gray-300 hover:border-black text-sm font-medium text-gray-700 hover:text-black transition-colors">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
