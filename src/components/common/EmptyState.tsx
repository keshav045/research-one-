import React from 'react';
import { Search, FolderOpen, ShieldCheck, ArrowRight } from 'lucide-react';

interface EmptyStateProps {
  type?: 'research' | 'evidence' | 'history' | 'custom';
  title?: string;
  description?: string;
  actionText?: string;
  onAction?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  type = 'research',
  title,
  description,
  actionText,
  onAction
}) => {
  let defaultTitle = "Start your first research investigation.";
  let defaultDesc = "Enter a research question to discover papers, extract evidence, and verify claims.";

  if (type === 'evidence') {
    defaultTitle = "No verified evidence found.";
    defaultDesc = "No extracted evidence matches the selected filter criteria.";
  } else if (type === 'history') {
    defaultTitle = "Your completed research reports will appear here.";
    defaultDesc = "Run an investigation to store reproducible synthesis and citation audits.";
  }

  return (
    <div className="flex flex-col items-center justify-center p-12 text-center rounded border border-dashed border-gray-300 bg-gray-50 my-6">
      <div className="w-12 h-12 rounded border border-gray-300 bg-white flex items-center justify-center mb-4 text-gray-400">
        {type === 'evidence' ? <ShieldCheck className="w-6 h-6" /> : type === 'history' ? <FolderOpen className="w-6 h-6" /> : <Search className="w-6 h-6" />}
      </div>
      <h3 className="text-sm font-semibold text-gray-900 mb-1">{title || defaultTitle}</h3>
      <p className="text-sm text-gray-500 max-w-md mb-5 leading-relaxed">{description || defaultDesc}</p>
      {actionText && onAction && (
        <button
          onClick={onAction}
          className="inline-flex items-center gap-2 px-4 py-2 rounded border border-black bg-black text-white text-sm font-medium hover:bg-gray-800 transition-colors"
        >
          {actionText} <ArrowRight className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
};
