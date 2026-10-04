import React from 'react';
import { ArrowLeftRight, Eye, FileSearch, Trash2, GitCompare } from 'lucide-react';
import { ContradictionIssue } from '../../types';

interface ConflictCardProps {
  issue: ContradictionIssue;
  onViewClaim: (claim: string) => void;
  onFindMoreEvidence: (claim: string) => void;
  onRemoveClaim: (id: string) => void;
  onCompareEvidence: (issue: ContradictionIssue) => void;
}

export const ConflictCard: React.FC<ConflictCardProps> = ({
  issue, onViewClaim, onFindMoreEvidence, onRemoveClaim, onCompareEvidence,
}) => {
  const isUnsupported = issue.type === 'unsupported_claim';

  return (
    <div className={`rounded border p-5 bg-white ${isUnsupported ? 'border-gray-300' : 'border-gray-400'}`}>
      <div className="flex items-center gap-2 mb-3">
        {isUnsupported
          ? <span className="px-2 py-0.5 rounded border border-gray-300 bg-gray-50 text-[10px] font-semibold text-gray-600 uppercase">Unsupported Claim</span>
          : <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded border border-gray-400 bg-gray-100 text-[10px] font-semibold text-gray-700 uppercase"><ArrowLeftRight className="w-3 h-3" /> Conflicting Evidence</span>
        }
        <span className="text-sm font-semibold text-gray-800">{issue.title}</span>
      </div>

      <div className="space-y-3 mb-4">
        <div className="p-3 rounded border border-gray-200 bg-gray-50 text-sm text-gray-900 font-medium leading-relaxed">
          "{issue.claim}"
        </div>
        {issue.reason && <p className="text-sm text-gray-500 leading-relaxed">{issue.reason}</p>}

        {!isUnsupported && issue.paperA && issue.paperB && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="p-3 rounded border border-gray-200 bg-gray-50 space-y-1">
              <span className="font-semibold text-gray-700 block">Paper A — {issue.paperA.authors} (p.{issue.paperA.page})</span>
              <p className="text-gray-600 italic">"{issue.paperA.statement}"</p>
            </div>
            <div className="p-3 rounded border border-gray-200 bg-gray-50 space-y-1">
              <span className="font-semibold text-gray-700 block">Paper B — {issue.paperB.authors} (p.{issue.paperB.page})</span>
              <p className="text-gray-600 italic">"{issue.paperB.statement}"</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-gray-100">
        {isUnsupported ? (
          <>
            <button onClick={() => onViewClaim(issue.claim)} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-gray-300 text-xs text-gray-700 hover:border-black hover:text-black transition-colors">
              <Eye className="w-3.5 h-3.5" /> View Claim
            </button>
            <button onClick={() => onFindMoreEvidence(issue.claim)} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-black bg-black text-white text-xs hover:bg-gray-800 transition-colors">
              <FileSearch className="w-3.5 h-3.5" /> Find More Evidence
            </button>
            <button onClick={() => onRemoveClaim(issue.id)} className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-gray-300 text-xs text-gray-500 hover:text-red-600 hover:border-red-300 transition-colors">
              <Trash2 className="w-3.5 h-3.5" /> Remove Claim
            </button>
          </>
        ) : (
          <button onClick={() => onCompareEvidence(issue)} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-black bg-black text-white text-xs hover:bg-gray-800 transition-colors">
            <GitCompare className="w-3.5 h-3.5" /> Compare Evidence
          </button>
        )}
      </div>
    </div>
  );
};
