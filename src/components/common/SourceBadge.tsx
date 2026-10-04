import React from 'react';
import { Database, FileText, UploadCloud } from 'lucide-react';
import { ResearchSource } from '../../types';

interface SourceBadgeProps {
  source: ResearchSource | string;
  className?: string;
}

export const SourceBadge: React.FC<SourceBadgeProps> = ({ source, className = '' }) => {
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded border border-gray-300 bg-gray-100 text-xs font-medium text-gray-700 ${className}`}>
      {source === 'arXiv' && <FileText className="w-3 h-3" />}
      {source === 'Semantic Scholar' && <Database className="w-3 h-3" />}
      {(source === 'Uploaded Papers' || source === 'Uploaded Documents') && <UploadCloud className="w-3 h-3" />}
      <span>{source}</span>
    </span>
  );
};
