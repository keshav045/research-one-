import React, { useState, useEffect, useCallback } from 'react';
import { 
  Sidebar, 
  NavigationTab 
} from './components/shell/Sidebar';
import { TopBar } from './components/shell/TopBar';
import { MobileBottomNav } from './components/shell/MobileBottomNav';
import { ResearchHome } from './components/home/ResearchHome';
import { ResearchProgress } from './components/workspace/ResearchProgress';
import { ReportViewer } from './components/report/ReportViewer';
import { EvidencePanel } from './components/evidence/EvidencePanel';
import { EvidenceExplorer } from './components/evidence/EvidenceExplorer';
import { CitationIntegrity } from './components/integrity/CitationIntegrity';
import { SourcesView } from './components/papers/SourcesView';
import { PaperDetailsModal } from './components/papers/PaperDetailsModal';
import { HistoryView } from './components/history/HistoryView';
import { SettingsModal } from './components/settings/SettingsModal';
import { EmptyState } from './components/common/EmptyState';
import { AuthModal } from './components/auth/AuthModal';
import { ProfileModal } from './components/auth/ProfileModal';
import { useAuth } from './context/AuthContext';

import {
  ResearchInvestigation,
  Citation,
  EvidenceItem,
  Paper,
  AppSettings,
  ResearchSource,
  ResearchDepth,
} from './types';
import * as api from './services/api';

export const App: React.FC = () => {
  // Navigation & View State
  const [currentTab, setCurrentTab] = useState<NavigationTab>('home');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Investigation & Research State
  const [activeInvestigation, setActiveInvestigation] = useState<ResearchInvestigation | null>(null);
  const [historyList, setHistoryList] = useState<ResearchInvestigation[]>([]);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [evidenceList, setEvidenceList] = useState<EvidenceItem[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);

  // Evidence Panel state
  const [evidencePanelOpen, setEvidencePanelOpen] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [selectedEvidenceItem, setSelectedEvidenceItem] = useState<EvidenceItem | null>(null);

  // Modals
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);

  // Auth
  const { user } = useAuth();

  // Settings & Theme
  const SETTINGS_STORAGE_KEY = 'researchlens_app_settings';

  const DEFAULT_SETTINGS: AppSettings = {
    llmProvider: 'Gemini',
    geminiModel: 'gemini-2.0-flash',
    openaiModel: 'gpt-4o',
    maxPapers: 12,
    researchDepth: 'Standard',
    maxIterations: 3,
    sources: {
      arxiv: true,
      semanticScholar: true,
      uploadedDocuments: false,
    },
    appearance: 'Light',
  };

  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (saved) return { ...DEFAULT_SETTINGS, ...JSON.parse(saved) };
    } catch {
      // fallback
    }
    return DEFAULT_SETTINGS;
  });

  // Apply Appearance Theme and Persist Settings
  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    } catch {}

    const applyTheme = (isDark: boolean) => {
      if (isDark) {
        document.documentElement.classList.add('dark');
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.classList.remove('dark');
        document.documentElement.removeAttribute('data-theme');
      }
    };

    if (settings.appearance === 'Dark') {
      applyTheme(true);
    } else if (settings.appearance === 'Light') {
      applyTheme(false);
    } else {
      const media = window.matchMedia('(prefers-color-scheme: dark)');
      applyTheme(media.matches);
      const listener = (e: MediaQueryListEvent) => applyTheme(e.matches);
      media.addEventListener('change', listener);
      return () => media.removeEventListener('change', listener);
    }
  }, [settings]);

  const handleToggleTheme = () => {
    const isCurrentlyDark = document.documentElement.classList.contains('dark');
    const newAppearance = isCurrentlyDark ? 'Light' : 'Dark';
    setSettings(prev => ({ ...prev, appearance: newAppearance }));
  };

  /**
   * Helper to load an investigation's papers, citations, and derived evidence
   */
  const loadInvestigationData = useCallback(async (inv: ResearchInvestigation) => {
    setActiveInvestigation(inv);
    
    // Fetch papers from API/Mock
    const jobPapers = await api.getJobPapers(inv.id);
    const allPapers = jobPapers.length > 0 ? jobPapers : ((inv as any).papers || inv.report?.references || []);
    setPapers(allPapers);

    // Extract citations from report findings
    const extractedCitations: Citation[] = [];
    if (inv.report?.findings) {
      for (const sec of inv.report.findings) {
        for (const p of sec.paragraphs) {
          if (p.citations) {
            for (const c of p.citations) {
              if (!extractedCitations.some(ec => ec.id === c.id)) {
                extractedCitations.push(c);
              }
            }
          }
        }
      }
    }
    setCitations(extractedCitations);

    // Build evidence explorer items
    const derivedEvidence: EvidenceItem[] = extractedCitations.map((c, idx) => {
      const comp = inv.report?.comparisonTable?.find(r => r.citationId === c.id);
      return {
        id: `evidence-${idx + 1}`,
        evidenceNumber: idx + 1,
        paperId: c.paperId,
        paperTitle: c.paperTitle,
        authors: c.authors,
        year: c.year,
        model: comp ? comp.model : (c.claim.split(' ')[0] || 'Model'),
        dataset: comp ? comp.dataset : 'Benchmark Evaluation',
        metric: comp ? 'Throughput / Quality' : 'Empirical Claim',
        value: comp ? (comp.fpsThroughput || comp.f1Score || 'Reported Result') : 'Verified passage',
        page: c.page,
        status: c.status,
        passage: c.passage,
        highlightSentence: c.highlightSentence,
        evidenceType: 'Extracted Passage',
        atomicClaims: c.atomicClaims,
        entailmentScore: c.entailmentScore,
      };
    });
    setEvidenceList(derivedEvidence);
  }, []);

  // Load history on mount
  useEffect(() => {
    api.getResearchHistory().then(history => {
      if (history && history.length > 0) {
        setHistoryList(history);
      }
    }).catch(err => console.warn('[History] Could not load past jobs:', err));
  }, []);

  // Handler: Start New Research Flow
  const handleStartResearch = async (params: {
    question: string;
    depth: ResearchDepth;
    sources: ResearchSource[];
  }) => {
    // Require login before running research
    if (!user) {
      setAuthModalOpen(true);
      return;
    }
    // Reset previous session data
    setPapers([]);
    setEvidenceList([]);
    setCitations([]);
    setSelectedCitation(null);
    setSelectedEvidenceItem(null);
    setEvidencePanelOpen(false);

    try {
      const newJob = await api.startResearch(params);
      setActiveInvestigation(newJob);
      setCurrentTab('workspace');
    } catch (err: any) {
      console.error('[StartResearch] Error:', err);
    }
  };

  // Handler: Select a pre-computed sample benchmark study (1-tap demo on mobile)
  const handleSelectSampleInvestigation = async (inv: ResearchInvestigation) => {
    await loadInvestigationData(inv);
    setCurrentTab('report');
  };

  // Polling backend/simulation while investigation is in progress
  useEffect(() => {
    if (!activeInvestigation || activeInvestigation.status !== 'in_progress') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const fresh = await api.getResearch(activeInvestigation.id);
        if (!fresh) return;

        setActiveInvestigation(fresh);

        const finalStatuses = ['completed', 'completed_with_warnings', 'insufficient_evidence', 'failed'];
        if (finalStatuses.includes(fresh.status)) {
          await loadInvestigationData(fresh);
          setHistoryList(prev => [fresh, ...prev.filter(h => h.id !== fresh.id)]);
          if (fresh.status !== 'failed') {
            setCurrentTab('report');
          }
          setEvidencePanelOpen(false);
        }
      } catch (err) {
        console.error('[Polling] Error fetching job status:', err);
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [activeInvestigation?.id, activeInvestigation?.status, loadInvestigationData]);

  // Handler: Select Citation from report
  const handleSelectCitation = (citation: Citation) => {
    setSelectedCitation(citation);
    setSelectedEvidenceItem(null);
    setEvidencePanelOpen(true);
  };

  // Handler: Select Citation by ID (from ComparisonTable)
  const handleSelectCitationById = (id: string) => {
    let found = citations.find(c => c.id === id);
    if (!found && activeInvestigation?.report) {
      for (const f of activeInvestigation.report.findings) {
        for (const p of f.paragraphs) {
          const match = p.citations?.find(c => c.id === id);
          if (match) {
            found = match;
            break;
          }
        }
        if (found) break;
      }
    }
    if (found) {
      setSelectedCitation(found);
      setSelectedEvidenceItem(null);
      setEvidencePanelOpen(true);
    }
  };

  // Handler: Inspect evidence item from Evidence Explorer
  const handleInspectEvidence = (item: EvidenceItem) => {
    setSelectedEvidenceItem(item);
    const matchingCitation = citations.find(c => c.paperId === item.paperId && c.page === item.page);
    setSelectedCitation(matchingCitation || null);
    setEvidencePanelOpen(true);
  };

  // Handler: Open Paper Details Modal
  const handleOpenPaperModal = (paperId: string) => {
    const found = papers.find(p => p.id === paperId);
    if (found) setSelectedPaper(found);
  };

  const hasReport = Boolean(activeInvestigation?.report);

  return (
    <div className="flex h-screen bg-white text-gray-900 overflow-hidden font-sans antialiased">
      {/* 1. Left Navigation Sidebar (Desktop dock + Mobile off-canvas drawer) */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={(tab) => {
          setCurrentTab(tab);
          setMobileMenuOpen(false);
        }}
        onNewResearch={() => {
          setCurrentTab('home');
          setMobileMenuOpen(false);
        }}
        onOpenSettings={() => {
          setSettingsOpen(true);
          setMobileMenuOpen(false);
        }}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        isOpenOnMobile={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      {/* 2. Main Content Canvas */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <TopBar
          investigation={activeInvestigation}
          evidencePanelOpen={evidencePanelOpen}
          onToggleEvidencePanel={() => setEvidencePanelOpen(!evidencePanelOpen)}
          onOpenMobileMenu={() => setMobileMenuOpen(true)}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenProfile={() => setProfileModalOpen(true)}
          onOpenAuthModal={() => setAuthModalOpen(true)}
          theme={settings.appearance}
          onToggleTheme={handleToggleTheme}
        />

        <div className="flex-1 flex min-h-0 overflow-hidden relative">
          {/* Main Scrollable Canvas */}
          <main className="flex-1 overflow-y-auto min-w-0">

            {/* Home — research question form */}
            {currentTab === 'home' && (
              <ResearchHome 
                onStartResearch={handleStartResearch} 
                onSelectSampleInvestigation={handleSelectSampleInvestigation}
              />
            )}

            {/* Workspace — live pipeline progress */}
            {currentTab === 'workspace' && activeInvestigation && (
              <ResearchProgress
                key={activeInvestigation.id}
                investigation={activeInvestigation}
                onViewReport={() => setCurrentTab('report')}
              />
            )}
            {currentTab === 'workspace' && !activeInvestigation && (
              <div className="max-w-2xl mx-auto px-4 py-12">
                <EmptyState
                  title="No active research job"
                  description="Start a new research investigation from the home screen to see the pipeline progress here."
                  actionText="Start Research"
                  onAction={() => setCurrentTab('home')}
                />
              </div>
            )}

            {/* Report — generated research report */}
            {currentTab === 'report' && hasReport && activeInvestigation && (
              <ReportViewer
                investigation={activeInvestigation}
                onSelectCitation={handleSelectCitation}
                onSelectCitationById={handleSelectCitationById}
                onOpenPaperModal={handleOpenPaperModal}
                onOpenIntegrityDashboard={() => setCurrentTab('citations')}
              />
            )}
            {currentTab === 'report' && !hasReport && (
              <div className="max-w-2xl mx-auto px-4 py-12">
                <EmptyState
                  title="No report yet"
                  description="Complete a research investigation or pick an instant benchmark study to generate a verified synthesis report."
                  actionText="Start Research"
                  onAction={() => setCurrentTab('home')}
                />
              </div>
            )}

            {/* Evidence Explorer */}
            {currentTab === 'evidence' && (
              <EvidenceExplorer
                evidenceList={evidenceList}
                onInspectEvidence={handleInspectEvidence}
              />
            )}

            {/* Citation Integrity Dashboard */}
            {currentTab === 'citations' && activeInvestigation && (
              <CitationIntegrity
                investigation={activeInvestigation}
                onOpenPaperModal={handleOpenPaperModal}
                onViewClaimInReport={() => setCurrentTab('report')}
              />
            )}
            {currentTab === 'citations' && !activeInvestigation && (
              <div className="max-w-2xl mx-auto px-4 py-12">
                <EmptyState
                  title="No citation data yet"
                  description="Run a research investigation or inspect an instant benchmark to see citation integrity analysis."
                  actionText="Start Research"
                  onAction={() => setCurrentTab('home')}
                />
              </div>
            )}

            {/* Sources Library */}
            {currentTab === 'sources' && (
              <SourcesView
                papers={papers}
                onSelectPaper={(p) => setSelectedPaper(p)}
                onNewResearch={() => setCurrentTab('home')}
              />
            )}

            {/* Research History */}
            {currentTab === 'history' && (
              <HistoryView
                history={historyList}
                onSelectInvestigation={async (inv) => {
                  await loadInvestigationData(inv);
                  setCurrentTab('report');
                }}
                onNewResearch={() => setCurrentTab('home')}
                onDeleteInvestigation={(targetItem) => {
                  setHistoryList(prev => prev.filter(h => h.id !== targetItem.id));
                }}
                onClearHistory={() => setHistoryList([])}
              />
            )}

          </main>

          {/* 3. Contextual Evidence Inspector Panel */}
          {evidencePanelOpen && (
            <EvidencePanel
              isOpen={evidencePanelOpen}
              citation={selectedCitation || (citations.length > 0 ? citations[0] : null)}
              evidenceItem={selectedEvidenceItem}
              onClose={() => setEvidencePanelOpen(false)}
              onOpenPaperDetails={handleOpenPaperModal}
            />
          )}
        </div>

        {/* 4. Mobile Bottom Navigation Bar (Smart Ergonomic Touch Controls) */}
        <MobileBottomNav
          currentTab={currentTab}
          onSelectTab={(tab) => {
            setCurrentTab(tab);
            setMobileMenuOpen(false);
          }}
          onOpenMenu={() => setMobileMenuOpen(true)}
          hasActiveReport={hasReport}
          isRunning={activeInvestigation?.status === 'in_progress'}
        />
      </div>

      {/* Global Modals */}
      <PaperDetailsModal
        paper={selectedPaper}
        evidenceItems={evidenceList}
        citations={citations}
        onClose={() => setSelectedPaper(null)}
      />

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onSave={(newSettings) => setSettings(newSettings)}
      />

      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
      />

      <ProfileModal
        isOpen={profileModalOpen}
        onClose={() => setProfileModalOpen(false)}
        history={historyList}
      />
    </div>
  );
};

export default App;
