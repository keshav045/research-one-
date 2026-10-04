import React, { useState } from 'react';
import { X, Settings, Cpu, Sliders, Database, Sun, Moon, Laptop, Check, Key } from 'lucide-react';
import { AppSettings, ResearchDepth } from '../../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSave: (settings: AppSettings) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, settings, onSave }) => {
  const [formData, setFormData] = useState<AppSettings>({ ...settings });
  const [prevSettings, setPrevSettings] = useState<AppSettings>(settings);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [savedSuccess, setSavedSuccess] = useState(false);

  if (settings !== prevSettings) {
    setPrevSettings(settings);
    setFormData({ ...settings });
  }

  const handleClose = React.useCallback(() => {
    if (settings.appearance === 'Dark') {
      document.documentElement.classList.add('dark');
    } else if (settings.appearance === 'Light') {
      document.documentElement.classList.remove('dark');
    } else {
      const isSysDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.documentElement.classList.toggle('dark', isSysDark);
    }
    onClose();
  }, [settings.appearance, onClose]);

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') handleClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [handleClose]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
    setSavedSuccess(true);
    setTimeout(() => { setSavedSuccess(false); onClose(); }, 800);
  };

  return (
    <div onClick={handleClose} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div onClick={e => e.stopPropagation()} className="bg-white border border-gray-300 rounded max-w-xl w-full p-6 shadow-xl space-y-5 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-gray-600" />
            <div>
              <h2 className="text-base font-bold text-gray-900">Settings</h2>
              <p className="text-xs text-gray-400">Configure LLM, sources, and depth parameters</p>
            </div>
          </div>
          <button onClick={handleClose} className="p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100" aria-label="Close">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5 text-sm">
          {/* LLM Provider */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5"><Cpu className="w-3.5 h-3.5" /> LLM Provider</p>
            <div className="grid grid-cols-2 gap-2">
              {(['Gemini', 'OpenAI'] as const).map(prov => {
                const isSelected = formData.llmProvider === prov;
                return (
                  <button key={prov} type="button" onClick={() => setFormData({ ...formData, llmProvider: prov })}
                    className={`p-3 rounded border text-left flex items-center justify-between transition-colors ${isSelected ? 'border-black bg-black text-white' : 'border-gray-300 bg-white text-gray-700 hover:border-black'}`}>
                    <div>
                      <div className="font-semibold">{prov}</div>
                      <div className="text-[10px] opacity-70 font-mono">{prov === 'Gemini' ? 'Gemini 3.8 / 3.5 Flash' : 'GPT-4o'}</div>
                    </div>
                    {isSelected && <Check className="w-4 h-4" />}
                  </button>
                );
              })}
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1 flex items-center gap-1"><Key className="w-3 h-3" /> API Key (Optional)</p>
              <input type="password" value={apiKeyInput} onChange={e => setApiKeyInput(e.target.value)} placeholder="sk-... or AIzaSy..."
                className="w-full bg-white border border-gray-300 rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-black transition-colors" />
            </div>
          </div>

          {/* Research Params */}
          <div className="space-y-3 pt-3 border-t border-gray-200">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5"><Sliders className="w-3.5 h-3.5" /> Research Parameters</p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Max Papers: <strong className="text-gray-900 font-mono">{formData.maxPapers}</strong></label>
                <input type="range" min="4" max="30" step="2" value={formData.maxPapers} onChange={e => setFormData({ ...formData, maxPapers: Number(e.target.value) })}
                  className="w-full accent-black cursor-pointer" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Default Depth:</label>
                <select value={formData.researchDepth} onChange={e => setFormData({ ...formData, researchDepth: e.target.value as ResearchDepth })}
                  className="w-full bg-white border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-black">
                  <option value="Quick">Quick (~6 papers)</option>
                  <option value="Standard">Standard (~12 papers)</option>
                  <option value="Deep">Deep (~24 papers)</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Max Iterations: <strong className="text-gray-900 font-mono">{formData.maxIterations}</strong></label>
              <input type="range" min="1" max="8" value={formData.maxIterations} onChange={e => setFormData({ ...formData, maxIterations: Number(e.target.value) })}
                className="w-full accent-black cursor-pointer" />
            </div>
          </div>

          {/* Sources */}
          <div className="space-y-2 pt-3 border-t border-gray-200">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5"><Database className="w-3.5 h-3.5" /> Scholarly Sources</p>
            <div className="space-y-1.5">
              {([
                { key: 'arxiv' as const, label: 'arXiv API Integration' },
                { key: 'semanticScholar' as const, label: 'Semantic Scholar API' },
              ]).map(({ key, label }) => (
                <label key={key} className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={formData.sources[key]} onChange={e => setFormData({ ...formData, sources: { ...formData.sources, [key]: e.target.checked } })}
                    className="accent-black" />
                  <span className="text-gray-700">{label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Appearance */}
          <div className="space-y-2 pt-3 border-t border-gray-200">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Appearance</p>
            <div className="grid grid-cols-3 gap-2">
              {(['Dark', 'Light', 'System'] as const).map(mode => {
                const isSelected = formData.appearance === mode;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => {
                      setFormData({ ...formData, appearance: mode });
                      if (mode === 'Dark') {
                        document.documentElement.classList.add('dark');
                        document.documentElement.setAttribute('data-theme', 'dark');
                      } else if (mode === 'Light') {
                        document.documentElement.classList.remove('dark');
                        document.documentElement.removeAttribute('data-theme');
                      } else {
                        const isSysDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                        document.documentElement.classList.toggle('dark', isSysDark);
                      }
                    }}
                    className={`py-2 px-3 rounded border text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors ${
                      isSelected
                        ? 'border-blue-600 bg-blue-600 text-white shadow-sm'
                        : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
                    }`}
                  >
                    {mode === 'Dark' && <Moon className="w-3.5 h-3.5" />}
                    {mode === 'Light' && <Sun className="w-3.5 h-3.5" />}
                    {mode === 'System' && <Laptop className="w-3.5 h-3.5" />}
                    {mode}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Submit */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-gray-200">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 rounded border border-gray-300 text-sm font-medium text-gray-700 hover:border-black hover:text-black transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-5 py-2 rounded border border-black bg-black text-white text-sm font-semibold hover:bg-gray-800 flex items-center gap-1.5 transition-colors"
            >
              {savedSuccess ? <><Check className="w-4 h-4" /> Saved!</> : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
