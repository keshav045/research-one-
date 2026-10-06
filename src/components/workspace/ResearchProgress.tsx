import React from 'react';
import { Layers, Loader2, CheckCircle, AlertTriangle, Cpu, Clock, Activity } from 'lucide-react';
import { ResearchInvestigation } from '../../types';
import { PipelineStep } from './PipelineStep';

interface ResearchProgressProps {
  investigation: ResearchInvestigation;
  onViewReport?: () => void;
}

export const ResearchProgress: React.FC<ResearchProgressProps> = ({ investigation, onViewReport }) => {
  const steps = investigation.pipeline || [];
  const completedCount = steps.filter(s => s.status === 'completed').length;
  const activeStep = steps.find(s => s.status === 'active');

  const isCompleted = investigation.status === 'completed';
  const isWarnings = investigation.status === 'completed_with_warnings';
  const isFinished = isCompleted || isWarnings;
  const isInsufficient = investigation.status === 'insufficient_evidence';
  const isFailed = investigation.status === 'failed';
  const isRunning = investigation.status === 'in_progress';

  const percentProgress = isFinished
    ? 100
    : steps.length > 0
    ? Math.round((completedCount / steps.length) * 100)
    : 10;

  return (
    <div className="max-w-3xl mx-auto px-3 sm:px-4 py-5 sm:py-8 pb-24 sm:pb-8">
      {/* Header Card */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 sm:p-6 mb-5 sm:mb-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-gray-100 mb-4">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <span className="text-[11px] font-bold text-gray-400 uppercase tracking-wider font-mono">
                Neural Research Engine
              </span>
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                  isRunning
                    ? 'bg-amber-50 text-amber-700 border border-amber-200'
                    : isCompleted
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : isWarnings
                    ? 'bg-amber-50 text-amber-800 border border-amber-300'
                    : isInsufficient
                    ? 'bg-orange-50 text-orange-700 border border-orange-200'
                    : 'bg-red-50 text-red-700 border border-red-200'
                }`}
              >
                {isRunning && <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />}
                {isCompleted && <CheckCircle className="w-3.5 h-3.5" />}
                {isWarnings && <AlertTriangle className="w-3.5 h-3.5" />}
                {isInsufficient && <AlertTriangle className="w-3.5 h-3.5" />}
                {isRunning ? 'Processing Pipeline' : investigation.status
                  .replace(/_/g, ' ')
                  .replace(/\b\w/g, c => c.toUpperCase())}
              </span>
            </div>
            <h2 className="text-base sm:text-lg font-bold text-gray-900 leading-snug break-words">
              "{investigation.question}"
            </h2>
          </div>

          {(isFinished || isInsufficient) && onViewReport && (
            <button
              onClick={onViewReport}
              className="w-full sm:w-auto shrink-0 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-black text-white text-xs font-semibold hover:bg-gray-800 transition-all shadow-sm"
            >
              View Synthesized Report →
            </button>
          )}
        </div>

        {/* Diagnostic Stats Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
          <div className="p-3 rounded-lg border border-gray-200 bg-gray-50/50">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mb-0.5">Papers Retrieved</p>
            <p className="text-sm font-bold text-gray-900 font-mono">
              {investigation.papersAnalyzed || 0}
            </p>
          </div>
          <div className="p-3 rounded-lg border border-gray-200 bg-gray-50/50">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mb-0.5">Passages Indexed</p>
            <p className="text-sm font-bold text-gray-900 font-mono">
              {investigation.passages_total || 0}
            </p>
          </div>
          <div className="p-3 rounded-lg border border-gray-200 bg-gray-50/50">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mb-0.5">Verified Claims</p>
            <p className="text-sm font-bold text-emerald-600 font-mono">
              {investigation.verifiedClaims || 0}
            </p>
          </div>
          <div className="p-3 rounded-lg border border-gray-200 bg-gray-50/50">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mb-0.5">Citation Integrity</p>
            <p className="text-sm font-bold text-gray-900 font-mono">
              {investigation.citationCoverage != null ? `${investigation.citationCoverage}%` : 'N/A'}
            </p>
          </div>
        </div>

        {/* Progress Bar */}
        <div>
          <div className="flex justify-between text-xs text-gray-600 mb-1.5 font-medium">
            <span>
              {isRunning
                ? activeStep ? `Stage: ${activeStep.name}...` : 'Executing...'
                : isCompleted ? 'Research Pipeline Completed'
                : isWarnings ? 'Completed with Warnings'
                : isInsufficient ? 'Evidence Gate Flagged Insufficient Evidence'
                : 'Pipeline Failed'}
            </span>
            <span className="font-mono font-bold text-gray-900">{percentProgress}%</span>
          </div>
          <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden border border-gray-200">
            <div
              className={`h-full transition-all duration-500 rounded-full ${
                isCompleted ? 'bg-emerald-600' : isWarnings ? 'bg-amber-500' : isInsufficient ? 'bg-orange-500' : isFailed ? 'bg-red-500' : 'bg-black'
              }`}
              style={{ width: `${percentProgress}%` }}
            />
          </div>
        </div>

        {/* Notice for Warnings */}
        {isWarnings && investigation.failure_reason && (
          <div className="mt-4 p-4 rounded-lg border border-amber-200 bg-amber-50 text-amber-900 text-xs leading-relaxed">
            <div className="flex items-center gap-2 font-bold mb-1 text-amber-800">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>Completed with Warnings</span>
            </div>
            <p className="text-amber-800">{investigation.failure_reason}</p>
          </div>
        )}

        {/* Notice for Insufficient Evidence — Phase 7: show real reason + counts */}
        {isInsufficient && (
          <div className="mt-4 p-4 rounded-lg border border-orange-200 bg-orange-50 text-orange-900 text-xs leading-relaxed">
            <div className="flex items-center gap-2 font-bold mb-2">
              <AlertTriangle className="w-4 h-4 text-orange-600 shrink-0" />
              <span>Insufficient Evidence</span>
            </div>
            {investigation.failure_reason && (
              <p className="mb-2 text-orange-800">{investigation.failure_reason}</p>
            )}
            <div className="grid grid-cols-3 gap-2 mt-2">
              <div className="bg-white/60 rounded p-2 text-center">
                <div className="font-mono font-bold text-orange-700">{investigation.papersAnalyzed || 0}</div>
                <div className="text-orange-600 text-[10px] uppercase tracking-wider">Papers Retrieved</div>
              </div>
              <div className="bg-white/60 rounded p-2 text-center">
                <div className="font-mono font-bold text-orange-700">{investigation.passages_total || 0}</div>
                <div className="text-orange-600 text-[10px] uppercase tracking-wider">Passages Indexed</div>
              </div>
              <div className="bg-white/60 rounded p-2 text-center">
                <div className="font-mono font-bold text-orange-700">{investigation.verifiedClaims || 0}</div>
                <div className="text-orange-600 text-[10px] uppercase tracking-wider">Claims Verified</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Execution Stages */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 mb-6 shadow-sm">
        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2 font-mono">
          <Layers className="w-4 h-4 text-black" /> Pipeline Stages
        </h3>
        <div className="divide-y divide-gray-100">
          {steps.map((step, idx) => (
            <PipelineStep key={step.id} step={step} index={idx} />
          ))}
        </div>
      </div>

      {/* Real Stage Stats Breakdown (if available) */}
      {investigation.stage_stats && investigation.stage_stats.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2 font-mono">
            <Activity className="w-4 h-4 text-black" /> Execution Benchmarks
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {investigation.stage_stats.map((s, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 rounded-lg border border-gray-100 bg-gray-50/50 text-xs">
                <span className="font-medium text-gray-700 capitalize">{s.name.replace(/_/g, ' ')}</span>
                <span className="font-mono text-gray-500">
                  {s.duration_ms < 1000 ? `${s.duration_ms}ms` : `${(s.duration_ms / 1000).toFixed(1)}s`}
                  {s.out_count > 0 && ` (${s.out_count} items)`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
