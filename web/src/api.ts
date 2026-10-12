// Typed client for api/main.py (same origin in production; Vite proxies /api in dev).

export interface Health {
  status: 'ok' | 'degraded';
  kg: 'snapshot' | 'live' | null;
  snapshot_generated_at: string | null;
  kg_error: string | null;
  artifacts_error: string | null;
}

export interface Cell { model: string; mfr: string }

export type Condition = 'c_rate_1' | 'c_rate_2' | 'soc_transition_pct';
export interface Recipe { profile: string; c_rate_1: number; c_rate_2: number; soc_transition_pct: number }

export interface Predictor {
  limits: Record<Condition, [number, number]>;
  profiles: { id: string; policy: string }[];
  presets: { pass: Recipe; refuse: Recipe };
  nominal_ah: number;
  n_training_cells: number;
  k_neighbors: number;
  threshold: number;
  cv_rmse_cycles: number;
  cv_mape_pct: number;
}

export interface Prediction {
  abstain: boolean;
  coverage: number;
  threshold: number;
  prediction_cycles: number;
  band_lo: number;
  band_hi: number;
}

export interface Discrepancy {
  claim: number; med: number; mn: number; mx: number; q1: number; q3: number;
  n: number; gap: number; comparable: boolean; rationale: string;
  claim_conditions: string; measured_conditions: string;
}

export interface Claim {
  property: string; value: unknown; unit: string | null;
  stated_conditions: string | null; page: number | null; document: string;
}

export interface Promise_ {
  model: string;
  values: number[];
  discrepancy: Discrepancy | null;
  claims: Claim[];
  mismatch: string | null;
}

export interface SpecRow {
  cell: string; property: string; value_a: string; value_b: string;
  source_a: string; source_b: string; verdict: string; detail: string;
  provenance: string; fragment_a: string; fragment_b: string;
}

export interface Specs {
  rows: SpecRow[];
  problems: SpecRow[];
  n_comparisons: number;
  n_consistent: number;
  n_single_document: number;
}

export interface Instances { instances: string[]; views: string[]; k: [number, number] }

export interface Neighbor { nbr: string; weight: number; same_group: boolean; life: number | null; policy: string }
export interface Graph { center: string; center_life: number | null; neighbors: Neighbor[] }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === 'string') detail = body.detail;
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const get = <T>(path: string, params?: Record<string, string | number>) =>
  request<T>(params ? `${path}?${new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))}` : path);

export const api = {
  health: () => get<Health>('/api/health'),
  cells: () => get<{ cells: Cell[] }>('/api/cells'),
  predictor: () => get<Predictor>('/api/predictor'),
  predict: (recipe: Recipe, signal?: AbortSignal) =>
    request<Prediction>('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(recipe),
      signal,
    }),
  promise: (model: string) => get<Promise_>('/api/promise', { model }),
  specs: () => get<Specs>('/api/specs'),
  instances: () => get<Instances>('/api/instances'),
  graph: (center: string, view: string, k: number) => get<Graph>('/api/graph', { center, view, k }),
};
