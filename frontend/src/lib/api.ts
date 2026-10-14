import { API_BASE } from './deploy'
import type { Alert, Incident, ScenarioRequest, SessionSummary, StepResult, SystemInfo, Timeline } from './types'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  // no-store: never show cached (possibly stale) model output when the backend is unavailable
  const r = await fetch(API_BASE + path, { headers: { 'Content-Type': 'application/json' }, cache: 'no-store', ...init })
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try { const j = await r.json(); msg = j.detail ?? msg } catch { /* not json */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}
const post = <T,>(p: string, body: unknown = {}) => req<T>(p, { method: 'POST', body: JSON.stringify(body) })

/* eslint-disable @typescript-eslint/no-explicit-any */
export const api = {
  system: () => req<SystemInfo>('/api/system'),
  assets: () => req<{ id: string; label: string; name: string; substation: string; source: string; test_failures: number }[]>('/api/assets'),
  timeline: () => req<Timeline>('/api/timeline'),
  log: (asset: string, start: number, end: number) => req<Record<string, any>[]>(`/api/log?asset=${asset}&start=${start}&end=${end}`),
  snapshot: (index: number) => req<{ index: number; assets: Record<string, Record<string, any>>; feed: any[]; active_alerts: any[] }>(`/api/snapshot?index=${index}`),
  createSession: (start_index: number, assets?: string[], stop_index?: number) => post<SessionSummary>('/api/sessions', { start_index, assets, stop_index }),
  step: (id: string, n: number) => post<StepResult>(`/api/sessions/${id}/step`, { n }),
  dropSession: (id: string) => req(`/api/sessions/${id}`, { method: 'DELETE' }),
  incidents: () => req<Incident[]>('/api/incidents'),
  replay: (asset: string, eventId: number, hours_before: number) => post<SessionSummary>(`/api/incidents/${asset}/${eventId}/replay`, { hours_before, hours_after: 1 }),
  alerts: () => req<Alert[]>('/api/alerts'),
  ack: (id: string) => post(`/api/alerts/${id}/ack`, { by: 'operator' }),
  unack: (id: string) => req(`/api/alerts/${id}/ack`, { method: 'DELETE' }),
  failures: () => req<Record<string, any>>('/api/analytics/failures'),
  models: () => req<Record<string, any>>('/api/models'),
  patterns: () => req<{ signal: string; alarms: number; followed_by_failure: number; share_followed: number }[]>('/api/patterns'),
  methodology: () => req<Record<string, any>>('/api/methodology'),
  scenario: (body: ScenarioRequest) => post<SessionSummary>('/api/scenarios', body),
  scenarioValidation: () => req<Record<string, any>>('/api/scenarios/validation'),
  upload: (asset: string, rows: Record<string, string | number>[]) => post<SessionSummary>('/api/uploads', { asset, rows }),
  sampleUrl: (asset: string, fault: string, month: number) => `${API_BASE}/api/samples/synthetic.csv?asset=${asset}&fault=${fault}&month=${month}`,
}
