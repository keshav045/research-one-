import React from 'react';
import { Filter, RotateCcw } from 'lucide-react';

export interface FilterState {
  dataset: string;
  model: string;
  metric: string;
  status: string;
}

interface FilterBarProps {
  filters: FilterState;
  onChange: (filters: FilterState) => void;
  datasets: string[];
  models: string[];
  metrics: string[];
}

export const FilterBar: React.FC<FilterBarProps> = ({ filters, onChange, datasets, models, metrics }) => {
  const handleChange = (key: keyof FilterState, val: string) => onChange({ ...filters, [key]: val });
  const hasActive = Object.values(filters).some(v => v !== 'all');

  const selectClass = "bg-white border border-gray-300 rounded px-2.5 py-1.5 text-xs text-gray-700 focus:outline-none focus:border-black";

  return (
    <div className="flex flex-wrap items-center gap-2 p-3 rounded border border-gray-200 bg-gray-50">
      <div className="flex items-center gap-1.5 text-xs text-gray-500 font-medium mr-1">
        <Filter className="w-3.5 h-3.5" /> Filters:
      </div>
      <select value={filters.dataset} onChange={e => handleChange('dataset', e.target.value)} className={selectClass}>
        <option value="all">All Datasets</option>
        {datasets.map(d => <option key={d} value={d}>{d}</option>)}
      </select>
      <select value={filters.model} onChange={e => handleChange('model', e.target.value)} className={selectClass}>
        <option value="all">All Models</option>
        {models.map(m => <option key={m} value={m}>{m}</option>)}
      </select>
      <select value={filters.metric} onChange={e => handleChange('metric', e.target.value)} className={selectClass}>
        <option value="all">All Metrics</option>
        {metrics.map(m => <option key={m} value={m}>{m}</option>)}
      </select>
      <select value={filters.status} onChange={e => handleChange('status', e.target.value)} className={selectClass}>
        <option value="all">All Statuses</option>
        <option value="verified">✓ Verified</option>
        <option value="partially_supported">~ Partially Supported</option>
        <option value="unsupported">✕ Unsupported</option>
        <option value="contradicted">↔ Contradicted</option>
      </select>
      {hasActive && (
        <button onClick={() => onChange({ dataset: 'all', model: 'all', metric: 'all', status: 'all' })} className="ml-auto inline-flex items-center gap-1 px-2.5 py-1.5 rounded border border-gray-300 text-xs text-gray-500 hover:text-black hover:border-black transition-colors">
          <RotateCcw className="w-3 h-3" /> Reset
        </button>
      )}
    </div>
  );
};
