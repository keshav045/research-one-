import React, { useState } from 'react';
import { ArrowRight, CheckSquare, Square, BookOpen, Search, ShieldCheck } from 'lucide-react';
import { ResearchDepth, ResearchSource } from '../../types';

interface ResearchHomeProps {
  onStartResearch: (params: { question: string; depth: ResearchDepth; sources: ResearchSource[] }) => void;
}

export const ResearchHome: React.FC<ResearchHomeProps> = ({ onStartResearch }) => {
  const [question, setQuestion] = useState(
    "Compare the performance and computational requirements of different Vision Transformer architectures for industrial defect detection."
  );
  const [depth, setDepth] = useState<ResearchDepth>('Standard');
  const [sources, setSources] = useState<ResearchSource[]>(['arXiv', 'Semantic Scholar', 'OpenAlex']);

  const exampleQuestions = [
    "Compare RAG architectures for enterprise document search.",
    "How do Vision Transformers compare with CNNs for defect detection?",
    "What are the main limitations of multimodal LLMs in clinical diagnostics?"
  ];

  const toggleSource = (source: ResearchSource) => {
    if (sources.includes(source)) {
      if (sources.length > 1) setSources(sources.filter(s => s !== source));
    } else {
      setSources([...sources, source]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    onStartResearch({ question: question.trim(), depth, sources });
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-12 sm:py-16">
      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-2 mb-4">
          <ShieldCheck className="w-5 h-5 text-gray-900" />
          <span className="text-sm font-semibold text-gray-900">ResearchLens</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 leading-tight mb-3">
          Research deeper.<br />Cite with confidence.
        </h1>
        <p className="text-gray-500 text-base leading-relaxed">
          An AI agent that investigates complex research questions, compares empirical evidence, and verifies every citation against source passages.
        </p>
      </div>

      {/* Form Card */}
      <div className="bg-white border border-gray-200 rounded p-6 space-y-6">
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Research Question */}
          <div>
            <label htmlFor="research-question-input" className="block text-sm font-semibold text-gray-900 mb-1.5 flex items-center gap-1.5">
              <Search className="w-4 h-4" /> Research Question
            </label>
            <textarea
              id="research-question-input"
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your research question..."
              className="w-full bg-white border border-gray-300 rounded px-3 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-black transition-colors resize-none leading-relaxed"
            />
          </div>

          {/* Depth + Sources */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 pt-1 border-t border-gray-100">
            {/* Depth */}
            <div>
              <p className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">Research Depth</p>
              <div className="grid grid-cols-3 gap-1.5">
                {(['Quick', 'Standard', 'Deep'] as ResearchDepth[]).map(d => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setDepth(d)}
                    className={`py-1.5 px-2 rounded border text-xs font-medium transition-colors ${
                      depth === d
                        ? 'bg-black text-white border-black'
                        : 'bg-white text-gray-600 border-gray-300 hover:border-black hover:text-black'
                    }`}
                  >
                    <span>{d}</span>
                    <span className="block text-[10px] font-normal opacity-60">
                      {d === 'Quick' ? '~6 papers' : d === 'Standard' ? '~12 papers' : '~24 papers'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Sources */}
            <div>
              <p className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">Sources</p>
              <div className="space-y-1.5">
                {([
                  { id: 'arXiv' as ResearchSource, name: 'arXiv' },
                  { id: 'Semantic Scholar' as ResearchSource, name: 'Semantic Scholar' },
                  { id: 'OpenAlex' as ResearchSource, name: 'OpenAlex' },
                ]).map(src => {
                  const checked = sources.includes(src.id);
                  return (
                    <div
                      key={src.id}
                      onClick={() => toggleSource(src.id)}
                      className="flex items-center gap-2 px-2.5 py-1.5 rounded border border-gray-200 hover:border-gray-400 cursor-pointer text-sm select-none bg-white"
                    >
                      {checked
                        ? <CheckSquare className="w-4 h-4 text-black shrink-0" />
                        : <Square className="w-4 h-4 text-gray-300 shrink-0" />}
                      <span className={checked ? 'text-gray-900 font-medium' : 'text-gray-500'}>{src.name}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={!question.trim()}
            className="w-full py-3 px-6 rounded border border-black bg-black text-white font-semibold text-sm hover:bg-gray-800 transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Start Research <ArrowRight className="w-4 h-4" />
          </button>
        </form>
      </div>

      {/* Example Questions */}
      <div className="mt-8 space-y-3">
        <div className="flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5 text-gray-400" />
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Example Questions</p>
        </div>
        <div className="space-y-2">
          {exampleQuestions.map((q, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setQuestion(q)}
              className="w-full p-3 text-left rounded border border-gray-200 hover:border-gray-400 bg-white text-sm text-gray-700 hover:text-black transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
