import React from 'react';
import { Menu, Download, PanelRightClose, PanelRightOpen, LogIn, Sun, Moon, Cpu, Globe, Sparkles } from 'lucide-react';
import { ResearchInvestigation } from '../../types';
import { UserMenu } from '../auth/UserMenu';
import { useAuth } from '../../context/AuthContext';
import { exportReportToPdf } from '../../utils/exportPdf';
import { getAppMode, getApiBaseUrl } from '../../services/api';

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
  const isDemo = investigation?.id?.startsWith('demo-') || investigation?.id?.startsWith('bench-') || getAppMode() === 'demo';
  const isCloud = getApiBaseUrl().includes('onrender.com');

  return (
    <header className="h-12 border-b border-gray-200 bg-white px-3 sm:px-4 flex items-center justify-between sticky top-0 z-20 select-none">
      {/* Left */}
      <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1 mr-2">
        <button
          onClick={onOpenMobileMenu}
          className="p-1.5 md:hidden rounded text-gray-500 hover:text-black hover:bg-gray-100 transition-colors shrink-0"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-1.5 sm:gap-2 text-xs sm:text-sm text-gray-500 min-w-0">
          <span className="font-bold text-gray-900 hidden xs:inline tracking-tight shrink-0">
            ResearchLens
          </span>
          <span className="text-gray-300 hidden xs:inline">/</span>
          <span className="text-gray-800 font-medium truncate max-w-[140px] xs:max-w-[200px] sm:max-w-xs md:max-w-md">
            {investigation ? investigation.question : "Research Workspace"}
          </span>
        </div>
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-1 sm:gap-2 shrink-0">
        {/* Engine mode badge */}
        <div
          onClick={onOpenSettings}
          className="cursor-pointer hidden sm:flex items-center gap-1 px-2 py-0.5 rounded border text-[11px] font-medium transition-colors border-emerald-300 bg-emerald-50 text-emerald-800 hover:border-emerald-500"
          title={`Active Engine: ${isDemo ? 'Static Demo Mode (Offline Ready)' : isCloud ? 'Render Cloud FastAPI Backend' : 'Local CUDA / NLI Engine'}. Click to configure settings.`}
        >
          {isDemo ? (
            <>
              <Sparkles className="w-3 h-3 text-emerald-600" />
              <span>Static Demo</span>
            </>
          ) : isCloud ? (
            <>
              <Globe className="w-3 h-3 text-emerald-600" />
              <span>Cloud Engine</span>
            </>
          ) : (
            <>
              <Cpu className="w-3 h-3 text-emerald-600" />
              <span>Local Engine</span>
            </>
          )}
        </div>

        {/* Status pill */}
        <div className="flex items-center gap-1 sm:gap-1.5 px-2 py-0.5 rounded border border-gray-200 bg-gray-50 text-[11px] sm:text-xs text-gray-700">
          <span className={`w-2 h-2 rounded-full shrink-0 ${isRunning ? 'bg-amber-500 animate-pulse' : 'bg-emerald-600'}`} />
          <span className="hidden xs:inline">{isRunning ? 'Running...' : 'Ready'}</span>
        </div>

        {/* PDF Download (when report exists) */}
        {investigation?.report && (
          <button
            onClick={() => exportReportToPdf(investigation)}
            className="p-1.5 rounded transition-colors text-gray-600 hover:text-black hover:bg-gray-100"
            title="Download report as PDF"
            aria-label="Download report as PDF"
          >
            <Download className="w-4 h-4" />
          </button>
        )}

        {/* Theme toggle */}
        {onToggleTheme && (
          <button
            onClick={onToggleTheme}
            className="p-1.5 rounded transition-colors text-gray-500 hover:text-black hover:bg-gray-100"
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

        {/* Evidence Panel toggle */}
        <button
          onClick={onToggleEvidencePanel}
          className={`flex items-center gap-1.5 px-2 sm:px-2.5 py-1 sm:py-1.5 rounded border text-xs font-medium transition-colors ${
            evidencePanelOpen
              ? 'bg-black text-white border-black'
              : 'bg-white text-gray-600 border-gray-300 hover:border-black hover:text-black'
          }`}
          title="Toggle Evidence Panel"
        >
          {evidencePanelOpen
            ? <><PanelRightClose className="w-3.5 h-3.5" /><span className="hidden md:inline">Hide Evidence</span></>
            : <><PanelRightOpen className="w-3.5 h-3.5" /><span className="hidden md:inline">Evidence</span></>
          }
        </button>

        {/* User: avatar menu or sign-in */}
        {user ? (
          <UserMenu
            onOpenSettings={onOpenSettings}
            onOpenProfile={onOpenProfile}
          />
        ) : (
          <button
            onClick={onOpenAuthModal}
            className="flex items-center gap-1 px-2.5 py-1 rounded border border-black bg-black text-white text-xs font-semibold hover:bg-gray-800 transition-colors"
          >
            <LogIn className="w-3 h-3" />
            <span className="hidden xs:inline">Sign In</span>
          </button>
        )}
      </div>
    </header>
  );
};
