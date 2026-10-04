import React from 'react';
import { Search, X } from 'lucide-react';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  value,
  onChange,
  placeholder = "Search...",
  className = ""
}) => {
  return (
    <div className={`relative flex items-center ${className}`}>
      <Search className="w-4 h-4 text-gray-400 absolute left-3 pointer-events-none" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-white border border-gray-300 rounded px-9 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-black transition-colors"
      />
      {value && (
        <button onClick={() => onChange('')} className="absolute right-3 text-gray-400 hover:text-black">
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
};
