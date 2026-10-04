import React, { useState } from 'react';
import { BookOpen } from 'lucide-react';
import { Paper } from '../../types';
import { SourceBadge } from '../common/SourceBadge';
import { SearchBar } from '../common/SearchBar';
import { EmptyState } from '../common/EmptyState';

interface SourcesViewProps {
  papers: Paper[];
  onSelectPaper: (paper: Paper) => void;
  onNewResearch?: () => void;
}

export const SourcesView: React.FC<SourcesViewProps> = ({ papers, onSelectPaper, onNewResearch }) => {
  const [search, setSearch] = useState('');

  const filtered = papers.filter(p => {
    if (!search) return true;
    const q = search.toLowerCase();
    return p.title.toLowerCase().includes(q) || p.authors.some(a => a.toLowerCase().includes(q)) || p.doi.toLowerCase().includes(q);
  });

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-5">
      <div className="flex items-center justify-between pb-4 border-b border-gray-200">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <BookOpen className="w-5 h-5 text-gray-600" />
            <h1 className="text-xl font-bold text-gray-900">Sources Library</h1>
          </div>
          <p className="text-sm text-gray-500">{papers.length} sourced publications</p>
        </div>
      </div>

      {papers.length === 0 ? (
        <EmptyState
          type="research"
          title="No Sources Loaded Yet"
          description="Start an investigation to automatically retrieve, parse, and verify literature sources from arXiv and Semantic Scholar."
          actionText={onNewResearch ? "Start Research" : undefined}
          onAction={onNewResearch}
        />
      ) : (
        <>
          <SearchBar value={search} onChange={setSearch} placeholder="Search by title, author, or DOI..." />

          {filtered.length === 0 ? (
            <div className="text-center py-12 border border-dashed border-gray-200 rounded p-6 bg-white space-y-2">
              <p className="text-sm font-semibold text-gray-800">No matching publications found</p>
              <p className="text-xs text-gray-500">No results found for "{search}".</p>
              <button
                onClick={() => setSearch('')}
                className="mt-2 px-3 py-1.5 rounded border border-gray-300 text-xs font-medium text-gray-700 hover:border-black hover:text-black transition-colors"
              >
                Clear Search
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {filtered.map(paper => (
                <div
                  key={paper.id}
                  onClick={() => onSelectPaper(paper)}
                  className="p-4 rounded border border-gray-200 hover:border-black bg-white transition-colors cursor-pointer group flex flex-col justify-between gap-3"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <SourceBadge source={paper.source} />
                      <span className="text-[11px] text-gray-400 font-mono">{paper.publicationYear}</span>
                    </div>
                    <h3 className="text-sm font-bold text-gray-900 group-hover:underline leading-snug">{paper.title}</h3>
                    <p className="text-xs text-gray-500">{paper.authors.join(", ")}</p>
                    <p className="text-[11px] text-gray-400">{paper.journalConference}</p>
                    <p className="text-xs text-gray-500 line-clamp-2 leading-relaxed">{paper.abstract}</p>
                  </div>
                  <div className="pt-2 border-t border-gray-100 flex items-center justify-between text-xs text-gray-400">
                    <span>{paper.evidenceCount} evidence pts • <span className="text-gray-700 font-medium">{paper.claimsSupportedCount} source matches</span></span>
                    <span className="text-gray-400 group-hover:text-black">Details →</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};
