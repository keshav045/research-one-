import React from 'react';
import {
  PlusCircle,
  FileText,
  Clock,
  Layers,
  CheckCircle,
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  X,
} from 'lucide-react';

export type NavigationTab =
  | 'home'
  | 'workspace'
  | 'report'
  | 'evidence'
  | 'citations'
  | 'sources'
  | 'history';

interface SidebarProps {
  currentTab: NavigationTab;
  onSelectTab: (tab: NavigationTab) => void;
  onNewResearch: () => void;
  onOpenSettings: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  isOpenOnMobile?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  onNewResearch,
  onOpenSettings,
  collapsed,
  onToggleCollapse,
  isOpenOnMobile = false,
  onCloseMobile,
}) => {
  const navItems = [
    { id: 'report' as NavigationTab, label: 'Research Report', icon: FileText },
    { id: 'history' as NavigationTab, label: 'Research History', icon: Clock },
  ];

  const toolItems = [
    { id: 'evidence' as NavigationTab, label: 'Evidence Explorer', icon: Layers },
    { id: 'citations' as NavigationTab, label: 'Citation Checker', icon: CheckCircle },
    { id: 'sources' as NavigationTab, label: 'Sources Library', icon: BookOpen },
  ];

  const handleTabClick = (tab: NavigationTab) => {
    onSelectTab(tab);
    if (onCloseMobile) onCloseMobile();
  };

  const handleNewResearchClick = () => {
    onNewResearch();
    if (onCloseMobile) onCloseMobile();
  };

  const handleSettingsClick = () => {
    onOpenSettings();
    if (onCloseMobile) onCloseMobile();
  };

  const renderNavContent = (isMobileView: boolean) => (
    <>
      {/* Brand Header */}
      <div className="flex items-center justify-between px-3 py-4 border-b border-gray-200">
        <div
          onClick={() => handleTabClick('home')}
          className="flex items-center gap-2.5 cursor-pointer overflow-hidden"
          title="ResearchLens Home"
        >
          <div className="w-8 h-8 rounded border border-gray-900 bg-black flex items-center justify-center text-white shrink-0">
            <ShieldCheck className="w-4 h-4" />
          </div>
          {(!collapsed || isMobileView) && (
            <div className="flex flex-col min-w-0">
              <span className="font-bold text-sm tracking-tight text-gray-900">ResearchLens</span>
              <span className="text-[11px] text-gray-500 truncate">Citation Verification</span>
            </div>
          )}
        </div>

        {/* Collapse toggle (desktop only) */}
        {!isMobileView && (
          <button
            onClick={onToggleCollapse}
            className="hidden md:flex p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100 transition-colors"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        )}

        {/* Close button (mobile only) */}
        {isMobileView && (
          <button
            onClick={onCloseMobile}
            className="p-1.5 rounded text-gray-500 hover:text-black hover:bg-gray-100 transition-colors"
            aria-label="Close navigation"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* New Research Button */}
      <div className="p-2.5">
        <button
          onClick={handleNewResearchClick}
          className={`w-full flex items-center justify-center gap-2 rounded border border-black bg-black text-white font-medium text-sm py-2 hover:bg-gray-800 transition-colors ${
            collapsed && !isMobileView ? 'px-2' : 'px-3'
          }`}
          title="Start New Research"
        >
          <PlusCircle className="w-4 h-4 shrink-0" />
          {(!collapsed || isMobileView) && <span>New Research</span>}
        </button>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-2 space-y-5 py-2">
        <div>
          {(!collapsed || isMobileView) && (
            <div className="px-2 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
              Navigation
            </div>
          )}
          <nav className="space-y-0.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleTabClick(item.id)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-black text-white'
                      : 'text-gray-600 hover:text-black hover:bg-gray-100'
                  }`}
                  title={item.label}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {(!collapsed || isMobileView) && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </div>

        <div>
          {(!collapsed || isMobileView) && (
            <div className="px-2 pb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
              Tools
            </div>
          )}
          <nav className="space-y-0.5">
            {toolItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleTabClick(item.id)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-black text-white'
                      : 'text-gray-600 hover:text-black hover:bg-gray-100'
                  }`}
                  title={item.label}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {(!collapsed || isMobileView) && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Settings */}
      <div className="p-2 border-t border-gray-200">
        <button
          onClick={handleSettingsClick}
          className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded text-sm font-medium text-gray-600 hover:text-black hover:bg-gray-100 transition-colors"
          title="Settings"
        >
          <Settings className="w-4 h-4 shrink-0" />
          {(!collapsed || isMobileView) && <span>Settings</span>}
        </button>
      </div>
    </>
  );

  return (
    <>
      {/* Desktop Sidebar: Only visible on md screens and up */}
      <aside
        className={`hidden md:flex flex-col h-screen border-r border-gray-200 bg-white text-gray-800 transition-all duration-300 select-none z-30 shrink-0 ${
          collapsed ? 'w-14' : 'w-60'
        }`}
      >
        {renderNavContent(false)}
      </aside>

      {/* Mobile Off-canvas Drawer: Slides in from left when isOpenOnMobile is true */}
      {isOpenOnMobile && (
        <div
          onClick={onCloseMobile}
          className="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-xs transition-opacity"
          aria-hidden="true"
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-72 max-w-[85vw] bg-white border-r border-gray-200 flex flex-col h-full shadow-2xl transition-transform duration-300 ease-in-out md:hidden ${
          isOpenOnMobile ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {renderNavContent(true)}
      </aside>
    </>
  );
};
