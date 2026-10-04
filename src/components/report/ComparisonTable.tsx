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
            <th className="py-2.5 px-3">Method / Model</th>
            <th className="py-2.5 px-3">Architecture Type</th>
            <th className="py-2.5 px-3">Dataset / Benchmark</th>
            <th className="py-2.5 px-3 text-right">Quality Impact</th>
            <th className="py-2.5 px-3 text-right">Comp. Ratio</th>
            <th className="py-2.5 px-3 text-right">Speed / Throughput</th>
            <th className="py-2.5 px-3 text-right">Memory (VRAM)</th>
            <th className="py-2.5 px-3 text-right">Compute / Calib</th>
            <th className="py-2.5 px-3 text-center">Cite</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 text-gray-700">
          {rows.map((row, idx) => (
            <tr key={idx} className="hover:bg-gray-50 transition-colors">
              <td className="py-2.5 px-3 font-semibold text-gray-900">{row.model}</td>
              <td className="py-2.5 px-3 text-gray-500">{row.architectureType}</td>
              <td className="py-2.5 px-3 font-mono text-[10px]">{row.dataset}</td>
              <td className="py-2.5 px-3 text-right font-bold font-mono text-gray-900">{row.f1Score}</td>
              <td className="py-2.5 px-3 text-right font-mono">{row.mapScore}</td>
              <td className="py-2.5 px-3 text-right font-mono">{row.fpsThroughput}</td>
              <td className="py-2.5 px-3 text-right font-mono text-gray-500">{row.parametersM}</td>
              <td className="py-2.5 px-3 text-right font-mono text-gray-500">{row.gflops}</td>
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
