import React, { useState } from 'react';
import { ArrowRight, CheckSquare, Square, BookOpen, Search, ShieldCheck, Sparkles, CheckCircle2 } from 'lucide-react';
import { ResearchDepth, ResearchSource, ResearchInvestigation } from '../../types';
import * as mockService from '../../services/mockService';

interface ResearchHomeProps {
  onStartResearch: (params: { question: string; depth: ResearchDepth; sources: ResearchSource[] }) => void;
  onSelectSampleInvestigation?: (inv: ResearchInvestigation) => void;
}

export const ResearchHome: React.FC<ResearchHomeProps> = ({ onStartResearch, onSelectSampleInvestigation }) => {
  const [question, setQuestion] = useState(
    "What are the most effective techniques for reducing the computational cost and memory usage of large language models during inference?"
  );
  const [depth, setDepth] = useState<ResearchDepth>('Standard');
  const [sources, setSources] = useState<ResearchSource[]>(['arXiv', 'Semantic Scholar', 'OpenAlex']);

  const sampleInvestigations = mockService.getMockHistory().slice(0, 3);

  const exampleQuestions = [
    "What are the most effective techniques for reducing the computational cost and memory usage of large language models during inference?",
    "Which paper introduced the Transformer architecture, and what was its key idea?",
    "Which paper introduced the Vision Transformer and how did it adapt self-attention?",
    "What is retrieval-augmented generation (RAG) and who proposed it?"
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
    <div className="max-w-2xl mx-auto px-4 py-6 sm:py-14 pb-24 sm:pb-16 select-none">
      {/* Header */}
      <div className="mb-8 sm:mb-10 text-left">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded bg-black flex items-center justify-center text-white shrink-0">
            <ShieldCheck className="w-3.5 h-3.5" />
          </div>
          <span className="text-xs sm:text-sm font-bold tracking-tight text-gray-900">ResearchLens Engine</span>
          <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
            Citation Integrity Verified
          </span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-extrabold text-gray-900 leading-tight mb-2 sm:mb-3">
          Research deeper.<br />Cite with confidence.
        </h1>
        <p className="text-gray-600 text-sm sm:text-base leading-relaxed">
          An AI research agent that retrieves peer-reviewed literature, extracts empirical evidence passages, and verifies citations via sentence-level NLI entailment.
        </p>
      </div>

      {/* Form Card */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 sm:p-6 shadow-xs space-y-5">
        <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-5">
          {/* Research Question */}
          <div>
            <label htmlFor="research-question-input" className="block text-xs sm:text-sm font-semibold text-gray-900 mb-1.5 flex items-center gap-1.5">
              <Search className="w-4 h-4 text-gray-500" /> Research Question
            </label>
            <textarea
              id="research-question-input"
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your research question (e.g. techniques for LLM inference efficiency)..."
              className="w-full bg-white border border-gray-300 rounded-lg px-3 py-2.5 text-xs sm:text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-black transition-colors resize-none leading-relaxed"
            />
          </div>

          {/* Depth + Sources */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1 border-t border-gray-100">
            {/* Depth */}
            <div>
              <p className="text-xs font-semibold text-gray-700 uppercase tracking-wider mb-2">Research Depth</p>
              <div className="grid grid-cols-3 gap-1.5">
                {(['Quick', 'Standard', 'Deep'] as ResearchDepth[]).map(d => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setDepth(d)}
                    className={`py-2 px-1.5 rounded-lg border text-xs font-medium transition-colors text-center ${
                      depth === d
                        ? 'bg-black text-white border-black shadow-xs'
                        : 'bg-white text-gray-600 border-gray-300 hover:border-black hover:text-black'
                    }`}
                  >
                    <span>{d}</span>
                    <span className="block text-[10px] font-normal opacity-70">
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
                      className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-gray-200 hover:border-gray-400 cursor-pointer text-xs sm:text-sm select-none bg-white transition-colors"
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
            className="w-full py-3 px-5 rounded-lg border border-black bg-black text-white font-semibold text-xs sm:text-sm hover:bg-gray-800 transition-all shadow-sm flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Start Research Investigation <ArrowRight className="w-4 h-4" />
          </button>
        </form>
      </div>

      {/* Featured Verified Studies (1-tap demo exploration on phone) */}
      {sampleInvestigations.length > 0 && onSelectSampleInvestigation && (
        <div className="mt-8 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-amber-500" />
              <p className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                Instant Benchmark Studies (Ready to View)
              </p>
            </div>
            <span className="text-[10px] text-gray-400 font-mono">Verified NLI</span>
          </div>
          <div className="grid grid-cols-1 gap-2">
            {sampleInvestigations.map((inv) => (
              <div
                key={inv.id}
                onClick={() => onSelectSampleInvestigation(inv)}
                className="p-3.5 rounded-xl border border-gray-200 bg-white hover:border-black cursor-pointer transition-all shadow-2xs group flex items-start justify-between gap-3"
              >
                <div className="min-w-0 flex-1 space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
                      <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                      {inv.citationCoverage}% Integrity
                    </span>
                    <span className="text-[10px] text-gray-400 font-mono">
                      {inv.papersAnalyzed} papers • {inv.verifiedClaims} verified facts
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm font-semibold text-gray-900 group-hover:text-black line-clamp-2">
                    {inv.question}
                  </p>
                </div>
                <div className="shrink-0 text-gray-400 group-hover:text-black self-center">
                  <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Example Questions */}
      <div className="mt-8 space-y-3">
        <div className="flex items-center gap-1.5">
          <BookOpen className="w-3.5 h-3.5 text-gray-400" />
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Example Research Queries</p>
        </div>
        <div className="space-y-1.5">
          {exampleQuestions.map((q, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setQuestion(q)}
              className="w-full p-2.5 sm:p-3 text-left rounded-lg border border-gray-200 hover:border-gray-400 bg-white text-xs sm:text-sm text-gray-700 hover:text-black transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
