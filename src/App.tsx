import React, { useState, useEffect } from 'react';
import { 
  Sidebar, 
  NavigationTab 
} from './components/shell/Sidebar';
import { TopBar } from './components/shell/TopBar';
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

  // Investigation & Research State — starts completely empty
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
    geminiModel: 'gemini-3.5-flash-lite',
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
      // fallback to defaults
    }
    return DEFAULT_SETTINGS;
  });

  // Apply Appearance Theme and Persist Settings
  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // ignore
    }

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
      // System
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
      alert(`Could not start research: ${err?.message || err}`);
    }
  };

  // Polling backend while investigation is in progress
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
          // Fetch real papers from DB
          const jobPapers = await api.getJobPapers(fresh.id);
          const allPapers = jobPapers.length > 0 ? jobPapers : (fresh.report?.references || []);
          setPapers(allPapers);

          // Extract citations from report findings
          const extractedCitations: Citation[] = [];
          if (fresh.report?.findings) {
            for (const sec of fresh.report.findings) {
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
            const comp = fresh.report?.comparisonTable?.find(r => r.citationId === c.id);
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

          setHistoryList(prev => [fresh, ...prev.filter(h => h.id !== fresh.id)]);
          if (fresh.status !== 'failed') {
            setCurrentTab('report');
          }
          setEvidencePanelOpen(false);
        }
      } catch (err) {
        console.error('[Polling] Error fetching job status:', err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeInvestigation?.id, activeInvestigation?.status]);


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
      for (const sec of activeInvestigation.report.findings) {
        for (const p of sec.paragraphs) {
          const match = p.citations?.find(c => c.id === id);
          if (match) { found = match; break; }
        }
        if (found) break;
      }
    }
    if (found) handleSelectCitation(found);
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
      {/* 1. Left Navigation Sidebar */}
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
        onOpenSettings={() => setSettingsOpen(true)}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Mobile Drawer Overlay */}
      {mobileMenuOpen && (
        <div 
          onClick={() => setMobileMenuOpen(false)}
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
        />
      )}

      {/* 2. Main Content Area */}
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

        <div className="flex-1 flex min-h-0 overflow-hidden">
          {/* Main Scrollable Canvas */}
          <main className="flex-1 overflow-y-auto min-w-0">

            {/* Home — research question form */}
            {currentTab === 'home' && (
              <ResearchHome onStartResearch={handleStartResearch} />
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
                  description="Complete a research investigation to generate a synthesis report with citation verification."
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
                  description="Run a research investigation to see citation integrity analysis here."
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
                onSelectInvestigation={(inv) => {
                  setActiveInvestigation(inv);
                  setCurrentTab('report');
                }}
                onNewResearch={() => setCurrentTab('home')}
                onDeleteInvestigation={(targetItem) => {
                  setHistoryList(prev => {
                    const idx = prev.indexOf(targetItem);
                    if (idx !== -1) {
                      return prev.filter((_, i) => i !== idx);
                    }
                    let removed = false;
                    return prev.filter(h => {
                      if (!removed && h.id === targetItem.id) {
                        removed = true;
                        return false;
                      }
                      return true;
                    });
                  });
                }}
                onClearHistory={() => setHistoryList([])}
              />
            )}


          </main>

          {/* 3. Contextual Evidence Inspector Panel */}
          {(currentTab === 'report' || currentTab === 'evidence') && evidencePanelOpen && (
            <EvidencePanel
              isOpen={evidencePanelOpen}
              citation={selectedCitation || (citations.length > 0 ? citations[0] : null)}
              evidenceItem={selectedEvidenceItem}
              onClose={() => setEvidencePanelOpen(false)}
              onOpenPaperDetails={handleOpenPaperModal}
            />
          )}
        </div>
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
