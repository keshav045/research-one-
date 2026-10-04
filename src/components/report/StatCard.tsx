import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon?: LucideIcon;
  onClick?: () => void;
}

export const StatCard: React.FC<StatCardProps> = ({ label, value, subtext, icon: Icon, onClick }) => {
  return (
    <div
      onClick={onClick}
      className={`rounded border border-gray-200 p-3 bg-white transition-colors ${onClick ? 'cursor-pointer hover:border-black' : ''}`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider truncate">{label}</span>
        {Icon && <Icon className="w-3.5 h-3.5 text-gray-300 shrink-0" />}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-xl font-bold text-gray-900 font-mono">{value}</span>
        {subtext && <span className="text-[10px] text-gray-400">{subtext}</span>}
      </div>
    </div>
  );
};
