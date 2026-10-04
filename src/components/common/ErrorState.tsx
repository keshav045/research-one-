import React from 'react';
import { AlertTriangle, RefreshCw, SlidersHorizontal } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  onChangeSources?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = "Research failed",
  message = "Unable to retrieve papers from the selected sources.",
  onRetry,
  onChangeSources
}) => {
  return (
    <div className="rounded border border-red-300 bg-red-50 p-8 max-w-xl mx-auto my-8 text-center">
      <div className="w-10 h-10 rounded border border-red-300 bg-red-100 flex items-center justify-center mx-auto mb-4 text-red-500">
        <AlertTriangle className="w-5 h-5" />
      </div>
      <h3 className="text-base font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-sm text-gray-600 mb-5 leading-relaxed max-w-md mx-auto">{message}</p>
      <div className="flex items-center justify-center gap-3">
        {onRetry && (
          <button onClick={onRetry} className="inline-flex items-center gap-2 px-4 py-2 rounded border border-gray-300 bg-white text-gray-800 text-sm font-medium hover:bg-gray-50 transition-colors">
            <RefreshCw className="w-3.5 h-3.5" /> Retry
          </button>
        )}
        {onChangeSources && (
          <button onClick={onChangeSources} className="inline-flex items-center gap-2 px-4 py-2 rounded border border-black bg-black text-white text-sm font-medium hover:bg-gray-800 transition-colors">
            <SlidersHorizontal className="w-3.5 h-3.5" /> Change Sources
          </button>
        )}
      </div>
    </div>
  );
};
