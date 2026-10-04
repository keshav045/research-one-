import React, { useState } from 'react';
import {
  X, User, Mail, Calendar, BookOpen, ShieldCheck,
  Edit2, Check, Layers, Clock,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { ResearchInvestigation } from '../../types';

interface ProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  history: ResearchInvestigation[];
}

export const ProfileModal: React.FC<ProfileModalProps> = ({ isOpen, onClose, history }) => {
  const { user, updateProfile, signOut } = useAuth();
  const [editingName, setEditingName] = useState(false);
  const [nameInput, setNameInput] = useState('');

  React.useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  if (!isOpen || !user) return null;

  const totalPapers = history.reduce((sum, h) => sum + (h.papersAnalyzed || 0), 0);
  const totalVerified = history.reduce((sum, h) => sum + (h.verifiedClaims || 0), 0);
  const avgCoverage = history.length
    ? Math.round(history.reduce((sum, h) => sum + h.citationCoverage, 0) / history.length)
    : 0;

  const handleSaveName = () => {
    if (nameInput.trim()) updateProfile({ name: nameInput.trim() });
    setEditingName(false);
  };

  const startEditName = () => {
    setNameInput(user.name);
    setEditingName(true);
  };

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
    >
      <div
        onClick={e => e.stopPropagation()}
        className="bg-white border border-gray-200 rounded-xl w-full max-w-lg shadow-2xl overflow-hidden max-h-[90vh] flex flex-col"
      >
        {/* Header */}
        <div className="bg-black px-6 pt-6 pb-8 text-white relative shrink-0">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-1 rounded text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>

          {/* Avatar */}
          <div className="flex items-end gap-4">
            <span
              className="w-16 h-16 rounded-2xl flex items-center justify-center text-white text-2xl font-bold shrink-0 border-2 border-white/20"
              style={{ backgroundColor: user.avatarColor }}
            >
              {user.initials}
            </span>
            <div className="mb-1 min-w-0">
              {/* Editable name */}
              {editingName ? (
                <div className="flex items-center gap-2">
                  <input
                    autoFocus
                    value={nameInput}
                    onChange={e => setNameInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleSaveName(); if (e.key === 'Escape') setEditingName(false); }}
                    className="bg-white/10 border border-white/30 rounded px-2 py-1 text-white text-base font-bold w-40 focus:outline-none focus:border-white"
                  />
                  <button onClick={handleSaveName} className="p-1 rounded bg-white/20 hover:bg-white/30">
                    <Check className="w-3.5 h-3.5 text-white" />
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold truncate">{user.name}</h2>
                  <button onClick={startEditName} className="p-1 rounded text-white/50 hover:text-white hover:bg-white/10">
                    <Edit2 className="w-3 h-3" />
                  </button>
                </div>
              )}
              <p className="text-white/60 text-xs mt-0.5">{user.email}</p>
            </div>
          </div>
        </div>

        {/* Body */}
        <div className="overflow-y-auto flex-1 p-6 space-y-5">

          {/* Stats */}
          <div>
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">Research Stats</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {[
                { label: 'Investigations', value: history.length, icon: BookOpen },
                { label: 'Papers Read', value: totalPapers, icon: Layers },
                { label: 'Claims Verified', value: totalVerified, icon: ShieldCheck },
                { label: 'Avg Integrity', value: `${avgCoverage}%`, icon: Check },
              ].map(({ label, value, icon: Icon }) => (
                <div key={label} className="p-3 rounded-lg border border-gray-200 bg-gray-50 text-center">
                  <Icon className="w-4 h-4 text-gray-400 mx-auto mb-1" />
                  <p className="text-lg font-bold text-gray-900 font-mono">{value}</p>
                  <p className="text-[10px] text-gray-400 font-medium">{label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Account info */}
          <div>
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">Account Details</p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-3 py-2 border-b border-gray-100">
                <User className="w-4 h-4 text-gray-400 shrink-0" />
                <span className="text-gray-500 w-20 shrink-0">Name</span>
                <span className="font-medium text-gray-900">{user.name}</span>
              </div>
              <div className="flex items-center gap-3 py-2 border-b border-gray-100">
                <Mail className="w-4 h-4 text-gray-400 shrink-0" />
                <span className="text-gray-500 w-20 shrink-0">Email</span>
                <span className="font-medium text-gray-900">{user.email}</span>
              </div>
              <div className="flex items-center gap-3 py-2 border-b border-gray-100">
                <Calendar className="w-4 h-4 text-gray-400 shrink-0" />
                <span className="text-gray-500 w-20 shrink-0">Member since</span>
                <span className="font-medium text-gray-900">
                  {new Date(user.createdAt).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                </span>
              </div>
            </div>
          </div>

          {/* Recent history */}
          {history.length > 0 && (
            <div>
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">Recent Investigations</p>
              <div className="space-y-2">
                {history.slice(0, 4).map(item => (
                  <div key={item.id} className="flex items-start gap-3 p-3 rounded-lg border border-gray-100 bg-gray-50">
                    <Clock className="w-3.5 h-3.5 text-gray-400 mt-0.5 shrink-0" />
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-gray-800 truncate">"{item.question}"</p>
                      <p className="text-[11px] text-gray-400 mt-0.5">
                        {item.papersAnalyzed} papers · {item.verifiedClaims} verified · {item.citationCoverage}% integrity
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Sign out */}
          <button
            onClick={() => { signOut(); onClose(); }}
            className="w-full py-2.5 rounded-lg border border-red-200 text-red-600 text-sm font-semibold hover:bg-red-50 transition-colors"
          >
            Sign Out
          </button>
        </div>
      </div>
    </div>
  );
};
