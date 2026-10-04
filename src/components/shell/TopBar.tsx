import { Menu, Download, PanelRightClose, PanelRightOpen, Sparkles, LogIn, Sun, Moon, Cpu } from 'lucide-react';
import { ResearchInvestigation } from '../../types';
import { UserMenu } from '../auth/UserMenu';
import { useAuth } from '../../context/AuthContext';
import { exportReportToPdf } from '../../utils/exportPdf';

interface TopBarProps {
  investigation: ResearchInvestigation | null;
  evidencePanelOpen: boolean;
  onToggleEvidencePanel: () => void;
  onOpenMobileMenu: () => void;
  onOpenSettings: () => void;
  onOpenProfile: () => void;
  onOpenAuthModal: () => void;
  theme?: 'Dark' | 'Light' | 'System';
  onToggleTheme?: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  investigation,
  evidencePanelOpen,
  onToggleEvidencePanel,
  onOpenMobileMenu,
  onOpenSettings,
  onOpenProfile,
  onOpenAuthModal,
  theme,
  onToggleTheme,
}) => {
  const { user } = useAuth();
  const isRunning = investigation?.status === 'in_progress';

  return (
    <header className="h-12 border-b border-gray-200 bg-white px-4 flex items-center justify-between sticky top-0 z-20">
      {/* Left */}
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onOpenMobileMenu}
          className="p-1.5 md:hidden rounded text-gray-400 hover:text-black hover:bg-gray-100"
          aria-label="Open navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2 text-sm text-gray-500 truncate">
          <span className="font-medium text-gray-400 hidden sm:inline">ResearchLens</span>
          <span className="text-gray-300 hidden sm:inline">/</span>
          <span className="text-gray-900 font-medium truncate max-w-xs sm:max-w-md">
            {investigation ? investigation.question : "Research Workspace"}
          </span>
        </div>
      </div>

      {/* Right */}
      <div className="flex items-center gap-2 shrink-0">
        {/* Local Neural Engine status badge */}
        <div
          className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded border border-emerald-300 bg-emerald-50 text-[11px] font-medium text-emerald-800"
          title="DeBERTa-v3 NLI & Qwen Local LLM Engine Active on CUDA"
        >
          <Cpu className="w-3 h-3 text-emerald-600" />
          <span>Local Engine Active</span>
        </div>

        {/* Status pill */}
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded border border-gray-200 bg-gray-50 text-xs text-gray-700">
          <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-amber-500 animate-pulse' : 'bg-black'}`} />
          <span>{isRunning ? 'Researching...' : 'Ready'}</span>
          {investigation && (
            <span className="text-gray-400 ml-1">{investigation.depth}</span>
          )}
        </div>

        <button
          onClick={() => {
            if (investigation?.report) {
              exportReportToPdf(investigation);
            }
          }}
          disabled={!investigation?.report}
          className="p-1.5 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-gray-400 hover:text-black hover:bg-gray-100"
          title={investigation?.report ? 'Download report as PDF' : 'No report available to download'}
        >
          <Download className="w-4 h-4" />
        </button>

        {onToggleTheme && (
          <button
            onClick={onToggleTheme}
            className="p-1.5 rounded transition-colors text-gray-400 hover:text-black hover:bg-gray-100"
            title={`Theme: ${theme || 'Light'} (Click to toggle)`}
            aria-label="Toggle theme"
          >
            {theme === 'Dark' ? (
              <Sun className="w-4 h-4 text-amber-500" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </button>
        )}

        <button
          onClick={onToggleEvidencePanel}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded border text-xs font-medium transition-colors ${
            evidencePanelOpen
              ? 'bg-black text-white border-black'
              : 'bg-white text-gray-600 border-gray-300 hover:border-black hover:text-black'
          }`}
          title="Toggle Evidence Panel"
        >
          {evidencePanelOpen
            ? <><PanelRightClose className="w-3.5 h-3.5" /><span className="hidden sm:inline">Hide Panel</span></>
            : <><PanelRightOpen className="w-3.5 h-3.5" /><span className="hidden sm:inline">Evidence Panel</span></>
          }
        </button>

        {/* User: show avatar menu or sign-in button */}
        {user ? (
          <UserMenu
            onOpenSettings={onOpenSettings}
            onOpenProfile={onOpenProfile}
          />
        ) : (
          <button
            onClick={onOpenAuthModal}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-black bg-black text-white text-xs font-semibold hover:bg-gray-800 transition-colors"
          >
            <LogIn className="w-3.5 h-3.5" />
            Sign In
          </button>
        )}
      </div>
    </header>
  );
};
