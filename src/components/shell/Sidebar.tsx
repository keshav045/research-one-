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
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  onNewResearch,
  onOpenSettings,
  collapsed,
  onToggleCollapse,
}) => {
  const navItems = [
    { id: 'report' as NavigationTab, label: 'Research', icon: FileText },
    { id: 'history' as NavigationTab, label: 'History', icon: Clock },
  ];

  const toolItems = [
    { id: 'evidence' as NavigationTab, label: 'Evidence Explorer', icon: Layers },
    { id: 'citations' as NavigationTab, label: 'Citation Checker', icon: CheckCircle },
    { id: 'sources' as NavigationTab, label: 'Sources', icon: BookOpen },
  ];

  return (
    <aside
      className={`relative flex flex-col h-screen border-r border-gray-200 bg-white text-gray-800 transition-all duration-300 select-none z-30 ${
        collapsed ? 'w-14' : 'w-60'
      }`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between px-3 py-4 border-b border-gray-200">
        <div
          onClick={() => onSelectTab('home')}
          className="flex items-center gap-2.5 cursor-pointer overflow-hidden"
          title="ResearchLens Home"
        >
          <div className="w-8 h-8 rounded border border-gray-900 bg-black flex items-center justify-center text-white shrink-0">
            <ShieldCheck className="w-4 h-4" />
          </div>
          {!collapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-bold text-sm tracking-tight text-gray-900">ResearchLens</span>
              <span className="text-[11px] text-gray-500 truncate">Citation Verification</span>
            </div>
          )}
        </div>

        <button
          onClick={onToggleCollapse}
          className="hidden md:flex p-1 rounded text-gray-400 hover:text-black hover:bg-gray-100 transition-colors"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* New Research Button */}
      <div className="p-2.5">
        <button
          onClick={onNewResearch}
          className={`w-full flex items-center justify-center gap-2 rounded border border-black bg-black text-white font-medium text-sm py-2 hover:bg-gray-800 transition-colors ${
            collapsed ? 'px-2' : 'px-3'
          }`}
          title="Start New Research"
        >
          <PlusCircle className="w-4 h-4 shrink-0" />
          {!collapsed && <span>New Research</span>}
        </button>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-2 space-y-5 py-2">
        <div>
          {!collapsed && (
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
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-black text-white'
                      : 'text-gray-600 hover:text-black hover:bg-gray-100'
                  }`}
                  title={item.label}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </div>

        <div>
          {!collapsed && (
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
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-black text-white'
                      : 'text-gray-600 hover:text-black hover:bg-gray-100'
                  }`}
                  title={item.label}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom */}
      <div className="p-2 border-t border-gray-200">
        <button
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded text-sm font-medium text-gray-600 hover:text-black hover:bg-gray-100 transition-colors"
          title="Settings"
        >
          <Settings className="w-4 h-4 shrink-0" />
          {!collapsed && <span>Settings</span>}
        </button>
      </div>
    </aside>
  );
};
