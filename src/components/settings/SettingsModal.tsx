import React, { useState } from 'react';
import { X, Settings, Cpu, Sliders, Database, Sun, Moon, Laptop, Check, Key, Globe, Sparkles, Activity } from 'lucide-react';
import { AppSettings, ResearchDepth } from '../../types';
import { getCustomApiUrl, setCustomApiUrl, getAppMode, setAppMode, checkBackendHealth } from '../../services/api';

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

  // API base URL & Mode state
  const [apiUrl, setApiUrl] = useState(getCustomApiUrl());
  const [appMode, setLocalAppMode] = useState<'live' | 'demo'>(getAppMode());
  const [testingConnection, setTestingConnection] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

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

  const handleTestConnection = async () => {
    setTestingConnection(true);
    setTestResult(null);
    setCustomApiUrl(apiUrl);
    const res = await checkBackendHealth();
    setTestingConnection(false);
    if (res.ok) {
      setTestResult({ ok: true, message: `Connected! Latency: ${res.latencyMs}ms` });
    } else {
      setTestResult({ ok: false, message: `Unreachable: ${res.error || 'Connection refused'}` });
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setCustomApiUrl(apiUrl);
    setAppMode(appMode);
    onSave(formData);
    setSavedSuccess(true);
    setTimeout(() => { setSavedSuccess(false); onClose(); }, 700);
  };

  return (
    <div onClick={handleClose} className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/60 backdrop-blur-xs select-none">
      <div onClick={e => e.stopPropagation()} className="bg-white border border-gray-200 rounded-2xl max-w-xl w-full p-4 sm:p-6 shadow-2xl space-y-4 sm:space-y-5 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-gray-700" />
            <div>
              <h2 className="text-sm sm:text-base font-bold text-gray-900">Settings & Connection</h2>
              <p className="text-xs text-gray-500">Configure Cloud/Local API, Demo Mode, and models</p>
            </div>
          </div>
          <button onClick={handleClose} className="p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100 transition-colors" aria-label="Close">
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-5 text-xs sm:text-sm">
          {/* Backend Connection & Mode */}
          <div className="space-y-2.5 p-3 sm:p-3.5 rounded-xl border border-gray-200 bg-gray-50/70">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold text-gray-700 uppercase tracking-wider flex items-center gap-1.5">
                <Globe className="w-3.5 h-3.5 text-blue-600" /> Backend Engine & Static Mode
              </p>
            </div>

            {/* Mode Toggle */}
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setLocalAppMode('live')}
                className={`p-2.5 rounded-lg border text-left transition-all ${
                  appMode === 'live'
                    ? 'border-black bg-black text-white'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-black'
                }`}
              >
                <div className="flex items-center gap-1 font-semibold text-xs">
                  <Activity className="w-3.5 h-3.5" /> Live Backend API
                </div>
                <div className="text-[10px] opacity-70 mt-0.5">Render / Localhost FastAPI</div>
              </button>

              <button
                type="button"
                onClick={() => setLocalAppMode('demo')}
                className={`p-2.5 rounded-lg border text-left transition-all ${
                  appMode === 'demo'
                    ? 'border-emerald-600 bg-emerald-600 text-white'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-emerald-600'
                }`}
              >
                <div className="flex items-center gap-1 font-semibold text-xs">
                  <Sparkles className="w-3.5 h-3.5" /> Static Demo (Offline)
                </div>
                <div className="text-[10px] opacity-70 mt-0.5">Standalone Benchmarks</div>
              </button>
            </div>

            {/* API URL Input */}
            <div className="space-y-1 pt-1">
              <div className="flex items-center justify-between text-xs text-gray-600">
                <span>FastAPI Backend URL</span>
                <span className="text-[10px] text-gray-400">e.g. https://xxx.onrender.com</span>
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={apiUrl}
                  onChange={e => setApiUrl(e.target.value)}
                  placeholder="https://researchlens-backend.onrender.com or http://localhost:8000"
                  className="flex-1 bg-white border border-gray-300 rounded-lg px-3 py-1.5 text-xs font-mono text-gray-900 focus:outline-none focus:border-black"
                />
                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={testingConnection}
                  className="px-3 py-1.5 rounded-lg border border-gray-300 bg-white hover:border-black text-xs font-medium text-gray-700 hover:text-black shrink-0 transition-colors disabled:opacity-50"
                >
                  {testingConnection ? 'Testing...' : 'Test'}
                </button>
              </div>
              {testResult && (
                <p className={`text-[11px] font-medium mt-1 ${testResult.ok ? 'text-emerald-600' : 'text-amber-600'}`}>
                  {testResult.message}
                </p>
              )}
            </div>
          </div>

          {/* LLM Provider */}
          <div className="space-y-2">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" /> LLM Provider
            </p>
            <div className="grid grid-cols-2 gap-2">
              {(['OpenAI', 'Gemini'] as const).map(prov => {
                const isSelected = formData.llmProvider === prov;
                return (
                  <button
                    key={prov}
                    type="button"
                    onClick={() => setFormData({ ...formData, llmProvider: prov })}
                    className={`p-2.5 sm:p-3 rounded-lg border text-left flex items-center justify-between transition-colors ${
                      isSelected ? 'border-black bg-black text-white' : 'border-gray-300 bg-white text-gray-700 hover:border-black'
                    }`}
                  >
                    <div>
                      <div className="font-semibold text-xs sm:text-sm">{prov}</div>
                      <div className="text-[10px] opacity-70 font-mono">
                        {prov === 'OpenAI' ? 'GPT-4o / GPT-4o-mini' : 'Gemini 2.0 Flash'}
                      </div>
                    </div>
                    {isSelected && <Check className="w-4 h-4" />}
                  </button>
                );
              })}
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1 flex items-center gap-1">
                <Key className="w-3 h-3" /> API Key (Optional)
              </p>
              <input
                type="password"
                value={apiKeyInput}
                onChange={e => setApiKeyInput(e.target.value)}
                placeholder="sk-... or AIzaSy..."
                className="w-full bg-white border border-gray-300 rounded-lg px-3 py-2 text-xs sm:text-sm font-mono focus:outline-none focus:border-black transition-colors"
              />
            </div>
          </div>

          {/* Research Parameters */}
          <div className="space-y-3 pt-2 border-t border-gray-100">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" /> Research Parameters
            </p>
            <div className="grid grid-cols-2 gap-3 sm:gap-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">
                  Max Papers: <strong className="text-gray-900 font-mono">{formData.maxPapers}</strong>
                </label>
                <input
                  type="range"
                  min="4"
                  max="30"
                  step="2"
                  value={formData.maxPapers}
                  onChange={e => setFormData({ ...formData, maxPapers: Number(e.target.value) })}
                  className="w-full accent-black cursor-pointer"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">
                  Depth: <strong className="text-gray-900">{formData.researchDepth}</strong>
                </label>
                <select
                  value={formData.researchDepth}
                  onChange={e => setFormData({ ...formData, researchDepth: e.target.value as ResearchDepth })}
                  className="w-full bg-white border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs text-gray-900 focus:outline-none focus:border-black"
                >
                  <option value="Quick">Quick (~6 papers)</option>
                  <option value="Standard">Standard (~12 papers)</option>
                  <option value="Deep">Deep (~24 papers)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Appearance Theme */}
          <div className="space-y-2 pt-2 border-t border-gray-100">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Appearance Theme</p>
            <div className="grid grid-cols-3 gap-2">
              {(['Light', 'Dark', 'System'] as const).map(mode => {
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
                    className={`py-2 px-3 rounded-lg border text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors ${
                      isSelected
                        ? 'border-black bg-black text-white shadow-xs'
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

          {/* Footer Submit */}
          <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-gray-200">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 rounded-lg border border-gray-300 text-xs sm:text-sm font-medium text-gray-700 hover:border-black hover:text-black transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-5 py-2 rounded-lg border border-black bg-black text-white text-xs sm:text-sm font-semibold hover:bg-gray-800 flex items-center gap-1.5 transition-colors shadow-xs"
            >
              {savedSuccess ? <><Check className="w-4 h-4" /> Saved!</> : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
