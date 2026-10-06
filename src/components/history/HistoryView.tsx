import React, { useState, useMemo } from 'react';
import { 
  Clock, 
  PlusCircle, 
  Trash2, 
  Search, 
  BookOpen, 
  CheckCircle2, 
  ShieldCheck, 
  Filter, 
  X 
} from 'lucide-react';
import { ResearchInvestigation } from '../../types';
import { ResearchCard } from './ResearchCard';
import { EmptyState } from '../common/EmptyState';

interface HistoryViewProps {
  history: ResearchInvestigation[];
  onSelectInvestigation: (item: ResearchInvestigation) => void;
  onNewResearch: () => void;
  onDeleteInvestigation?: (item: ResearchInvestigation) => void;
  onClearHistory?: () => void;
}

type DepthFilter = 'all' | 'Quick' | 'Standard' | 'Deep' | 'high_coverage';

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  onSelectInvestigation,
  onNewResearch,
  onDeleteInvestigation,
  onClearHistory,
}) => {
  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState<DepthFilter>('all');

  // Overview metrics across all history
  const totalPapers = useMemo(() => history.reduce((sum, h) => sum + (h.papersAnalyzed || 0), 0), [history]);
  const totalVerified = useMemo(() => history.reduce((sum, h) => sum + (h.verifiedClaims || 0), 0), [history]);
  const avgCoverage = useMemo(() => {
    if (history.length === 0) return 0;
    const sum = history.reduce((acc, h) => acc + (h.citationCoverage || 0), 0);
    return Math.round(sum / history.length);
  }, [history]);

  // Filtered investigations
  const filtered = useMemo(() => {
    return history.filter(item => {
      // Query filter
      if (query.trim()) {
        const q = query.toLowerCase();
        const matchQ = item.question.toLowerCase().includes(q) ||
          item.report?.executiveSummary?.toLowerCase().includes(q) ||
          item.sources.some(s => s.toLowerCase().includes(q));
        if (!matchQ) return false;
      }

      // Depth / Category filter
      if (activeFilter === 'Quick' || activeFilter === 'Standard' || activeFilter === 'Deep') {
        if (item.depth !== activeFilter) return false;
      } else if (activeFilter === 'high_coverage') {
        if (item.citationCoverage < 90) return false;
      }

      return true;
    });
  }, [history, query, activeFilter]);

  return (
    <div className="max-w-4xl mx-auto px-3 sm:px-4 py-5 sm:py-8 pb-24 sm:pb-8 space-y-5 sm:space-y-6">
      {/* 1. Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-gray-200">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-7 h-7 rounded border border-gray-200 bg-gray-50 flex items-center justify-center">
              <Clock className="w-4 h-4 text-gray-700" />
            </div>
            <h1 className="text-xl font-bold text-gray-900">Research History</h1>
          </div>
          <p className="text-sm text-gray-500">
            Chronological archive of deep literature syntheses, empirical evidence extractions, and citation audits.
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {history.length > 0 && onClearHistory && (
            <button
              onClick={() => {
                if (window.confirm('Are you sure you want to clear all research history?')) {
                  onClearHistory();
                }
              }}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded border border-gray-200 hover:border-red-300 hover:text-red-600 bg-white text-gray-500 text-xs font-medium transition-colors"
              title="Clear all history entries"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear History</span>
            </button>
          )}

          <button
            onClick={onNewResearch}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded border border-black bg-black text-white text-xs font-semibold hover:bg-gray-800 transition-colors shadow-sm"
          >
            <PlusCircle className="w-3.5 h-3.5" />
            <span>New Investigation</span>
          </button>
        </div>
      </div>

      {/* 2. Overview Metrics Ribbon */}
      {history.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3.5 rounded-lg border border-gray-200 bg-white shadow-xs space-y-1">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>Investigations</span>
              <Clock className="w-3.5 h-3.5 text-gray-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-gray-900">{history.length}</p>
            <p className="text-[10px] text-gray-400">Total sessions archived</p>
          </div>

          <div className="p-3.5 rounded-lg border border-gray-200 bg-white shadow-xs space-y-1">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>Papers Evaluated</span>
              <BookOpen className="w-3.5 h-3.5 text-gray-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-gray-900">{totalPapers}</p>
            <p className="text-[10px] text-gray-400">Peer-reviewed sources</p>
          </div>

          <div className="p-3.5 rounded-lg border border-gray-200 bg-white shadow-xs space-y-1">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>Verified Claims</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-gray-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-gray-900">{totalVerified}</p>
            <p className="text-[10px] text-gray-400">Grounded facts extracted</p>
          </div>

          <div className="p-3.5 rounded-lg border border-gray-200 bg-white shadow-xs space-y-1">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>Avg Citation Integrity</span>
              <ShieldCheck className="w-3.5 h-3.5 text-gray-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-gray-900">{avgCoverage}%</p>
            <p className="text-[10px] text-gray-400">Average citation integrity</p>
          </div>
        </div>
      )}

      {/* 3. Search Bar & Filter Tabs */}
      {history.length > 0 && (
        <div className="space-y-3">
          <div className="relative">
            <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search past investigations by question or topic..."
              className="w-full pl-9 pr-8 py-2 rounded-lg border border-gray-200 bg-white text-xs text-gray-900 placeholder-gray-400 focus:outline-none focus:border-black transition-colors"
            />
            {query && (
              <button
                onClick={() => setQuery('')}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-black"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mr-1 flex items-center gap-1">
                <Filter className="w-3 h-3" /> Filter:
              </span>

              {[
                { id: 'all' as DepthFilter, label: `All (${history.length})` },
                { id: 'Deep' as DepthFilter, label: 'Deep (~12)' },
                { id: 'Standard' as DepthFilter, label: 'Standard (~8)' },
                { id: 'Quick' as DepthFilter, label: 'Quick (~6)' },
                { id: 'high_coverage' as DepthFilter, label: '≥90% Integrity' },
              ].map(f => (
                <button
                  key={f.id}
                  onClick={() => setActiveFilter(f.id)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                    activeFilter === f.id
                      ? 'bg-black text-white border border-black'
                      : 'bg-white text-gray-600 border border-gray-200 hover:border-gray-400 hover:text-black'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <span className="text-[11px] text-gray-400 font-mono">
              Showing {filtered.length} of {history.length}
            </span>
          </div>
        </div>
      )}

      {/* 4. Investigation Cards List */}
      {history.length === 0 ? (
        <EmptyState
          type="history"
          title="No research history yet"
          description="Synthesize your first topic on the home screen to build your research archive."
          actionText="Start First Investigation"
          onAction={onNewResearch}
        />
      ) : filtered.length === 0 ? (
        <div className="p-12 text-center rounded-lg border border-gray-200 bg-white space-y-3">
          <p className="text-sm font-semibold text-gray-800">No matching investigations found</p>
          <p className="text-xs text-gray-500">
            No research sessions matched "{query}" with the current filter settings.
          </p>
          <button
            onClick={() => {
              setQuery('');
              setActiveFilter('all');
            }}
            className="px-3 py-1.5 rounded border border-gray-300 text-xs font-medium text-gray-700 hover:border-black hover:text-black"
          >
            Clear Filters
          </button>
        </div>
      ) : (
        <div className="space-y-3.5">
          {filtered.map((item, index) => (
            <ResearchCard
              key={`${item.id}-${index}`}
              item={item}
              onSelect={onSelectInvestigation}
              onDelete={onDeleteInvestigation}
            />
          ))}
        </div>
      )}
    </div>
  );
};
