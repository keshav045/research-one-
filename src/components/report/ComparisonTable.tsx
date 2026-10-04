import React from 'react';
import { ExternalLink } from 'lucide-react';
import { ComparisonRow } from '../../types';

interface ComparisonTableProps {
  rows: ComparisonRow[];
  onSelectCitation: (citationId: string) => void;
}

export const ComparisonTable: React.FC<ComparisonTableProps> = ({ rows, onSelectCitation }) => {
  return (
    <div className="w-full overflow-x-auto rounded border border-gray-200 my-5">
      <table className="w-full text-left text-xs border-collapse">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50 text-gray-500 font-semibold uppercase tracking-wider text-[10px]">
            <th className="py-2.5 px-3">Paper / Title</th>
            <th className="py-2.5 px-3">Authors</th>
            <th className="py-2.5 px-3">Year</th>
            <th className="py-2.5 px-3">Venue</th>
            <th className="py-2.5 px-3 text-right">Citations</th>
            <th className="py-2.5 px-3 text-center">Cite</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 text-gray-700">
          {rows.map((row, idx) => (
            <tr key={idx} className="hover:bg-gray-50 transition-colors">
              <td className="py-2.5 px-3 font-semibold text-gray-900 max-w-xs">{row.title || row.model}</td>
              <td className="py-2.5 px-3 text-gray-600">{row.authors || row.architectureType}</td>
              <td className="py-2.5 px-3 font-mono text-[11px]">{row.year || row.dataset}</td>
              <td className="py-2.5 px-3 text-gray-600">{row.venue || row.f1Score}</td>
              <td className="py-2.5 px-3 text-right font-mono font-medium">{row.citationCount || row.mapScore}</td>
              <td className="py-2.5 px-3 text-center">
                <button
                  onClick={() => onSelectCitation(row.citationId)}
                  className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-black transition-colors"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
