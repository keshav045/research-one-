import React from 'react';
import { 
  Search, 
  FileText, 
  Layers, 
  BookOpen, 
  Clock, 
  Menu 
} from 'lucide-react';
import { NavigationTab } from './Sidebar';

interface MobileBottomNavProps {
  currentTab: NavigationTab;
  onSelectTab: (tab: NavigationTab) => void;
  onOpenMenu: () => void;
  hasActiveReport: boolean;
  isRunning: boolean;
}

export const MobileBottomNav: React.FC<MobileBottomNavProps> = ({
  currentTab,
  onSelectTab,
  onOpenMenu,
  hasActiveReport,
  isRunning,
}) => {
  const tabs = [
    {
      id: 'home' as NavigationTab,
      label: 'Search',
      icon: Search,
    },
    {
      id: (isRunning ? 'workspace' : (hasActiveReport ? 'report' : 'workspace')) as NavigationTab,
      label: isRunning ? 'Running' : (hasActiveReport ? 'Report' : 'Process'),
      icon: FileText,
      badge: isRunning,
    },
    {
      id: 'evidence' as NavigationTab,
      label: 'Evidence',
      icon: Layers,
    },
    {
      id: 'sources' as NavigationTab,
      label: 'Sources',
      icon: BookOpen,
    },
    {
      id: 'history' as NavigationTab,
      label: 'History',
      icon: Clock,
    },
  ];

  return (
    <nav 
      aria-label="Mobile Navigation"
      className="md:hidden fixed bottom-0 left-0 right-0 z-30 bg-white/95 backdrop-blur-md border-t border-gray-200 px-1 py-1 flex items-center justify-around shadow-lg select-none"
      style={{ paddingBottom: 'calc(env(safe-area-inset-bottom, 0px) + 4px)' }}
    >
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = currentTab === tab.id || (tab.id === 'workspace' && currentTab === 'report') || (tab.id === 'report' && currentTab === 'workspace');
        return (
          <button
            key={tab.id}
            onClick={() => onSelectTab(tab.id)}
            className={`flex flex-col items-center justify-center flex-1 py-1 px-1 rounded-md transition-colors relative ${
              isActive 
                ? 'text-black font-semibold' 
                : 'text-gray-500 hover:text-black'
            }`}
          >
            <div className="relative">
              <Icon className={`w-5 h-5 ${isActive ? 'stroke-[2.5]' : 'stroke-[1.8]'}`} />
              {tab.badge && (
                <span className="absolute -top-1 -right-1.5 w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              )}
            </div>
            <span className="text-[10px] tracking-tight mt-0.5 leading-none">
              {tab.label}
            </span>
          </button>
        );
      })}

      {/* Menu / Drawer button */}
      <button
        onClick={onOpenMenu}
        className="flex flex-col items-center justify-center flex-1 py-1 px-1 rounded-md text-gray-500 hover:text-black transition-colors"
        aria-label="Open full menu"
      >
        <Menu className="w-5 h-5 stroke-[1.8]" />
        <span className="text-[10px] tracking-tight mt-0.5 leading-none">
          More
        </span>
      </button>
    </nav>
  );
};
