import {
  SystemInfo,
  ModelInspectResult,
  BenchmarkInfo,
  DatasetInspectResult,
  PreflightResponse,
  StartOptimizationRequest,
  RunSummaryItem,
  RunDetailItem,
} from '../types';

const BASE_URL = '/api/v1';

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${BASE_URL}/system`);
  if (!res.ok) throw new Error('Failed to fetch system hardware information');
  return res.json();
}

export async function fetchPopularModels(): Promise<any[]> {
  const res = await fetch(`${BASE_URL}/models/popular`);
  if (!res.ok) throw new Error('Failed to fetch popular models');
  return res.json();
}

export async function inspectModel(modelId: string): Promise<ModelInspectResult> {
  const res = await fetch(`${BASE_URL}/models/inspect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to inspect model architecture');
  }
  return res.json();
}

export async function fetchBenchmarks(): Promise<BenchmarkInfo[]> {
  const res = await fetch(`${BASE_URL}/benchmarks`);
  if (!res.ok) throw new Error('Failed to fetch benchmark catalog');
  return res.json();
}

export async function inspectDataset(source: string): Promise<DatasetInspectResult> {
  const res = await fetch(`${BASE_URL}/datasets/inspect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to inspect dataset schema');
  }
  return res.json();
}

export async function previewDataset(source: string, limit: number = 3): Promise<any[]> {
  const res = await fetch(`${BASE_URL}/datasets/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, limit }),
  });
  if (!res.ok) throw new Error('Failed to preview dataset samples');
  const data = await res.json();
  return data.samples || [];
}

export async function runPreflight(params: StartOptimizationRequest): Promise<PreflightResponse> {
  const res = await fetch(`${BASE_URL}/optimization/preflight`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Preflight planning failed');
  }
  return res.json();
}

export async function startOptimization(params: StartOptimizationRequest): Promise<{ job_id: string; run_id: string; status: string; message: string }> {
  const res = await fetch(`${BASE_URL}/optimization/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to start optimization job');
  }
  return res.json();
}

export async function fetchActiveOptimization(): Promise<{ active: boolean; job: any }> {
  const res = await fetch(`${BASE_URL}/optimization/active`);
  if (!res.ok) throw new Error('Failed to check active optimization');
  return res.json();
}

export async function fetchRuns(status?: string, model?: string): Promise<RunSummaryItem[]> {
  const query = new URLSearchParams();
  if (status) query.set('status', status);
  if (model) query.set('model', model);
  const res = await fetch(`${BASE_URL}/runs?${query.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch runs library');
  return res.json();
}

export async function fetchRunDetails(runId: string): Promise<RunDetailItem> {
  const res = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) throw new Error(`Failed to load details for run '${runId}'`);
  return res.json();
}

export async function fetchRunReport(runId: string): Promise<{ summary_markdown: string }> {
  const res = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}/report`);
  if (!res.ok) throw new Error('Report not found for this run');
  return res.json();
}

export async function cancelRun(runId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to request cancellation');
}

export async function resumeRun(runId: string): Promise<{ job_id: string; run_id: string }> {
  const res = await fetch(`${BASE_URL}/runs/${encodeURIComponent(runId)}/resume`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to resume run');
  }
  return res.json();
}

export async function compareRuns(runA: string, runB: string): Promise<any> {
  const res = await fetch(`${BASE_URL}/runs/compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`);
  if (!res.ok) throw new Error('Failed to compare runs');
  return res.json();
}
