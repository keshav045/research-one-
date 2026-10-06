import React, { useState, useMemo } from 'react';
import { Layers } from 'lucide-react';
import { EvidenceItem } from '../../types';
import { SearchBar } from '../common/SearchBar';
import { FilterBar, FilterState } from './FilterBar';
import { EvidenceCard } from './EvidenceCard';
import { EmptyState } from '../common/EmptyState';

interface EvidenceExplorerProps {
  evidenceList: EvidenceItem[];
  onInspectEvidence: (item: EvidenceItem) => void;
}

export const EvidenceExplorer: React.FC<EvidenceExplorerProps> = ({ evidenceList, onInspectEvidence }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<FilterState>({ dataset: 'all', model: 'all', metric: 'all', status: 'all' });

  const datasets = useMemo(() => Array.from(new Set(evidenceList.map(e => e.dataset))), [evidenceList]);
  const models = useMemo(() => Array.from(new Set(evidenceList.map(e => e.model))), [evidenceList]);
  const metrics = useMemo(() => Array.from(new Set(evidenceList.map(e => e.metric))), [evidenceList]);

  const filteredEvidence = useMemo(() => {
    return evidenceList.filter(item => {
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const matches = item.model.toLowerCase().includes(q) || item.dataset.toLowerCase().includes(q) || item.metric.toLowerCase().includes(q) || item.paperTitle.toLowerCase().includes(q) || item.passage.toLowerCase().includes(q);
        if (!matches) return false;
      }
      if (filters.dataset !== 'all' && item.dataset !== filters.dataset) return false;
      if (filters.model !== 'all' && item.model !== filters.model) return false;
      if (filters.metric !== 'all' && item.metric !== filters.metric) return false;
      if (filters.status !== 'all' && item.status !== filters.status) return false;
      return true;
    });
  }, [evidenceList, searchQuery, filters]);

  return (
    <div className="max-w-5xl mx-auto px-3 sm:px-4 py-5 sm:py-8 pb-24 sm:pb-8 space-y-4 sm:space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-gray-200">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Layers className="w-5 h-5 text-gray-600" />
            <h1 className="text-xl font-bold text-gray-900">Evidence Explorer</h1>
          </div>
          <p className="text-sm text-gray-500">Extracted empirical data points from literature synthesis</p>
        </div>
        <span className="text-sm font-mono text-gray-500 border border-gray-200 rounded px-2.5 py-1">
          {filteredEvidence.length} / {evidenceList.length}
        </span>
      </div>

      <SearchBar value={searchQuery} onChange={setSearchQuery} placeholder="Search evidence by model, dataset, or passage..." />
      <FilterBar filters={filters} onChange={setFilters} datasets={datasets} models={models} metrics={metrics} />

      {filteredEvidence.length === 0 ? (
        <EmptyState type="evidence" actionText="Clear Filters" onAction={() => { setSearchQuery(''); setFilters({ dataset: 'all', model: 'all', metric: 'all', status: 'all' }); }} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredEvidence.map(item => <EvidenceCard key={item.id} item={item} onViewSource={onInspectEvidence} />)}
        </div>
      )}
    </div>
  );
};
