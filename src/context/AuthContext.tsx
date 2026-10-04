import React, { createContext, useContext, useState, useCallback } from 'react';
import { User, AuthState } from '../types/auth';

// ─── Storage Helpers ──────────────────────────────────────────────────────────

const STORAGE_KEY = 'researchlens_auth';
const ACCOUNTS_KEY = 'researchlens_accounts';

/** Returns a deterministic avatar background color from a string. */
function getAvatarColor(id: string): string {
  const colors = [
    '#1a1a2e', '#16213e', '#0f3460', '#533483',
    '#2b2d42', '#1b4332', '#023e8a', '#6a0572',
    '#3d405b', '#264653',
  ];
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = id.charCodeAt(i) + ((hash << 5) - hash);
  return colors[Math.abs(hash) % colors.length];
}

/** Derives initials from a full name. */
function getInitials(name: string): string {
  return name
    .split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
}

/** Loads stored accounts (email -> hashed password). */
function loadAccounts(): Record<string, { passwordHash: string; user: User }> {
  try {
    return JSON.parse(localStorage.getItem(ACCOUNTS_KEY) || '{}');
  } catch {
    return {};
  }
}

/** Saves accounts back to storage. */
function saveAccounts(accounts: Record<string, { passwordHash: string; user: User }>) {
  localStorage.setItem(ACCOUNTS_KEY, JSON.stringify(accounts));
}

/** Very simple hash — good enough for a client-side demo (not cryptographic). */
function simpleHash(str: string): string {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) hash = (hash * 33) ^ str.charCodeAt(i);
  return (hash >>> 0).toString(16);
}

// ─── Context ──────────────────────────────────────────────────────────────────

interface AuthContextValue extends AuthState {
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  updateProfile: (updates: Partial<Pick<User, 'name'>>) => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

// ─── Provider ─────────────────────────────────────────────────────────────────

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });
  const [isLoading] = useState(false);

  const signUp = useCallback(async (name: string, email: string, password: string) => {
    const accounts = loadAccounts();
    if (accounts[email.toLowerCase()]) {
      throw new Error('An account with this email already exists.');
    }
    if (password.length < 6) {
      throw new Error('Password must be at least 6 characters.');
    }

    const id = `user-${Date.now().toString(36)}`;
    const newUser: User = {
      id,
      name: name.trim(),
      email: email.toLowerCase().trim(),
      initials: getInitials(name.trim()),
      avatarColor: getAvatarColor(id),
      createdAt: new Date().toISOString(),
    };

    accounts[email.toLowerCase()] = { passwordHash: simpleHash(password), user: newUser };
    saveAccounts(accounts);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newUser));
    setUser(newUser);
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const accounts = loadAccounts();
    const record = accounts[email.toLowerCase()];
    if (!record || record.passwordHash !== simpleHash(password)) {
      throw new Error('Incorrect email or password.');
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(record.user));
    setUser(record.user);
  }, []);

  const signOut = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
  }, []);

  const updateProfile = useCallback((updates: Partial<Pick<User, 'name'>>) => {
    setUser(prev => {
      if (!prev) return prev;
      const updated: User = {
        ...prev,
        ...updates,
        initials: updates.name ? getInitials(updates.name) : prev.initials,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));

      // Also update in accounts store
      const accounts = loadAccounts();
      if (accounts[prev.email]) {
        accounts[prev.email].user = updated;
        saveAccounts(accounts);
      }
      return updated;
    });
  }, []);

  return (
    <AuthContext.Provider value={{ user, isLoading, signUp, signIn, signOut, updateProfile }}>
      {children}
    </AuthContext.Provider>
  );
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}
