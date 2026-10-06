/**
 * ResearchLens Frontend API Service
 * =================================
 * Single Source of Truth with Dual Mode:
 * 1. Live Cloud Backend (FastAPI on Render / Localhost)
 * 2. Static Interactive Demo Mode (Full offline client-side simulation & benchmarks)
 */

import {
  ResearchInvestigation,
  ResearchDepth,
  ResearchSource,
  Paper,
} from '../types';
import * as mockService from './mockService';

const STORAGE_API_KEY = 'researchlens_custom_api_url';
const STORAGE_MODE_KEY = 'researchlens_app_mode';

export function getCustomApiUrl(): string {
  try {
    const saved = localStorage.getItem(STORAGE_API_KEY);
    if (saved && saved.trim()) return saved.trim();
  } catch {}
  return (import.meta.env.VITE_API_URL || 'http://localhost:8000').trim();
}

export function setCustomApiUrl(url: string) {
  try {
    localStorage.setItem(STORAGE_API_KEY, url.trim());
  } catch {}
}

export function getAppMode(): 'live' | 'demo' {
  try {
    const saved = localStorage.getItem(STORAGE_MODE_KEY);
    if (saved === 'demo' || saved === 'live') return saved;
  } catch {}
  return 'live';
}

export function setAppMode(mode: 'live' | 'demo') {
  try {
    localStorage.setItem(STORAGE_MODE_KEY, mode);
  } catch {}
}

export function getApiBaseUrl(): string {
  return getCustomApiUrl().replace(/\/+$/, '');
}

/**
 * Check if the configured backend server is reachable and responsive.
 */
export async function checkBackendHealth(): Promise<{ ok: boolean; latencyMs?: number; error?: string }> {
  const url = getApiBaseUrl();
  const start = performance.now();
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 4000);
    const res = await fetch(`${url}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (res.ok) {
      return { ok: true, latencyMs: Math.round(performance.now() - start) };
    }
    return { ok: false, error: `Status ${res.status}` };
  } catch (err: any) {
    return { ok: false, error: err?.message || 'Connection refused' };
  }
}

/**
 * Start a new research investigation.
 * If backend is unavailable or user selected Demo mode, executes interactive client simulation.
 */
export async function startResearch(params: {
  question: string;
  depth: ResearchDepth;
  sources: ResearchSource[];
  forceDemo?: boolean;
}): Promise<ResearchInvestigation> {
  const mode = getAppMode();

  if (params.forceDemo || mode === 'demo') {
    return mockService.startMockResearch(params);
  }

  const url = getApiBaseUrl();
  try {
    const res = await fetch(`${url}/api/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Server returned ${res.status}: ${errorText}`);
    }

    return (await res.json()) as ResearchInvestigation;
  } catch (err: any) {
    // Surface failure clearly instead of substituting canned demo data for a real question
    throw new Error(
      `Could not reach the research backend (${err?.message || 'network error'}). It may be waking up — please retry in a moment, or switch to Demo mode in Settings.`
    );
  }
}

/**
 * Poll research investigation status by ID.
 */
export async function getResearch(id: string): Promise<ResearchInvestigation | null> {
  if (id.startsWith('demo-') || id.startsWith('bench-') || getAppMode() === 'demo') {
    const mock = mockService.getMockInvestigation(id);
    if (mock) return mock;
  }

  const url = getApiBaseUrl();
  try {
    const res = await fetch(`${url}/api/research/${id}`);
    if (res.ok) {
      return (await res.json()) as ResearchInvestigation;
    }
  } catch (err) {
    // Check mock fallback
    const mock = mockService.getMockInvestigation(id);
    if (mock) return mock;
  }
  return null;
}

/**
 * Retrieve the ranked and extracted papers for a research job.
 */
export async function getJobPapers(id: string): Promise<Paper[]> {
  if (id.startsWith('demo-') || id.startsWith('bench-') || getAppMode() === 'demo') {
    return mockService.getMockPapers(id);
  }

  const url = getApiBaseUrl();
  try {
    const res = await fetch(`${url}/api/research/${id}/papers`);
    if (res.ok) {
      const papers = (await res.json()) as Paper[];
      if (papers && papers.length > 0) return papers;
    }
  } catch (err) {}

  return mockService.getMockPapers(id);
}

/**
 * Retrieve debug diagnostics for a research job.
 */
export async function getJobDebug(id: string): Promise<any> {
  const url = getApiBaseUrl();
  try {
    const res = await fetch(`${url}/api/research/${id}/debug`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {}
  return null;
}

/**
 * Retrieve history list of past investigations.
 * Combines live backend history with bundled benchmark studies.
 */
export async function getResearchHistory(): Promise<ResearchInvestigation[]> {
  const mockHistory = mockService.getMockHistory();
  const url = getApiBaseUrl();

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2500);
    const res = await fetch(`${url}/api/research`, { signal: controller.signal });
    clearTimeout(timeout);
    if (res.ok) {
      const liveHistory = (await res.json()) as ResearchInvestigation[];
      if (Array.isArray(liveHistory) && liveHistory.length > 0) {
        // Merge without duplicates
        const seen = new Set(liveHistory.map(h => h.id));
        const combined = [...liveHistory];
        for (const m of mockHistory) {
          if (!seen.has(m.id)) {
            combined.push(m);
            seen.add(m.id);
          }
        }
        return combined;
      }
    }
  } catch (err) {}

  return mockHistory;
}

/**
 * Delete a research job.
 */
export async function deleteResearch(id: string): Promise<boolean> {
  if (id.startsWith('demo-') || id.startsWith('bench-')) {
    return mockService.deleteMockInvestigation(id);
  }

  const url = getApiBaseUrl();
  try {
    const res = await fetch(`${url}/api/research/${id}`, {
      method: 'DELETE',
    });
    if (res.status === 204) return true;
  } catch (err) {}

  return mockService.deleteMockInvestigation(id);
}
