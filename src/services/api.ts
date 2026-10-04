/**
 * ResearchLens Frontend API Service
 * =================================
 * Single Source of Truth: All research jobs, pipeline progress, paper retrieval,
 * and verified evidence are driven by the FastAPI backend.
 */

import {
  ResearchInvestigation,
  ResearchDepth,
  ResearchSource,
  Paper,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Start a new research investigation on the backend.
 * Triggers asynchronous background pipeline execution.
 */
export async function startResearch(params: {
  question: string;
  depth: ResearchDepth;
  sources: ResearchSource[];
}): Promise<ResearchInvestigation> {
  const res = await fetch(`${API_BASE_URL}/api/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Failed to start research (${res.status}): ${errorText}`);
  }

  return (await res.json()) as ResearchInvestigation;
}

/**
 * Poll research investigation status by ID.
 * Returns the current state, pipeline steps, stage_stats, and report (when finished).
 */
export async function getResearch(id: string): Promise<ResearchInvestigation | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/research/${id}`);
    if (res.ok) {
      return (await res.json()) as ResearchInvestigation;
    }
  } catch (err) {
    console.warn(`[API] Failed to get research for ${id}:`, err);
  }
  return null;
}

/**
 * Retrieve the ranked and extracted papers for a research job.
 */
export async function getJobPapers(id: string): Promise<Paper[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/research/${id}/papers`);
    if (res.ok) {
      return (await res.json()) as Paper[];
    }
  } catch (err) {
    console.warn(`[API] Failed to get papers for ${id}:`, err);
  }
  return [];
}

/**
 * Retrieve debug diagnostics for a research job.
 */
export async function getJobDebug(id: string): Promise<any> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/research/${id}/debug`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn(`[API] Failed to get debug for ${id}:`, err);
  }
  return null;
}

/**
 * Retrieve history list of past investigations.
 */
export async function getResearchHistory(): Promise<ResearchInvestigation[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/research`);
    if (res.ok) {
      return (await res.json()) as ResearchInvestigation[];
    }
  } catch (err) {
    console.warn('[API] Failed to get research history:', err);
  }
  return [];
}

/**
 * Delete a research job.
 */
export async function deleteResearch(id: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/research/${id}`, {
      method: 'DELETE',
    });
    return res.status === 204;
  } catch (err) {
    console.warn(`[API] Failed to delete research ${id}:`, err);
  }
  return false;
}
