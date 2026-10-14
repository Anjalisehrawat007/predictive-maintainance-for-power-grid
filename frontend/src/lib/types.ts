export type Status = 'WARMUP' | 'NORMAL' | 'WARNING' | 'CRITICAL' | 'OFFLINE'

export interface Signal { signal: string; share: number; direction?: string }
export interface Health {
  score: number
  components: Record<'thermal_stability' | 'load_stability' | 'sensor_quality' | 'failure_risk' | 'model_confidence', number>
  inputs?: Record<string, number | boolean>
}
/** One engine output (live session record, or a row of the precomputed engine log). */
export interface EngineRecord {
  index: number
  timestamp: string
  asset: string
  sensors: Record<string, number>
  offline: boolean
  status: Status
  raw_score: number | null
  risk: number | null
  confidence: number | null
  member_scores?: number[]
  trend?: 'rising' | 'falling' | 'stable'
  trend_delta_1h?: number
  warning_since?: string | null
  warning_window_min?: number | null
  signals?: Signal[]
  attribution?: Signal[]
  temporal_attribution?: number[]
  health?: Health
  timings_ms?: Record<string, number>
  ground_truth?: { failure_event: number; sim_phase: string; sim_event_type: string }
  lead_time_min?: number | null
  source: 'live' | 'log'
}
export interface FeedEvent { ts: string; asset: string; kind: string; severity: string; message: string; index: number; lead_time_min?: number }
export interface Alert {
  id: string; asset: string; severity: Status; opened: string; updated?: string; closed?: string | null; closed_by?: string
  risk: number; raw_score: number; confidence: number; warning_window_min: number | null; signals: Signal[]
  status: 'open' | 'resolved' | 'acknowledged'; index: number; source: string; outcome?: string; ack?: { by: string; at: string; note: string }
}
export interface SystemInfo {
  status: string; disclosure: string
  model: { name: string; type: string; parameters: number; seq_len: number; window_hours: number; horizon_min: number; feature_set: string; n_features: number; onnx_runtime: boolean; ensemble_members: number; calibration: string; selected_in: string }
  thresholds: { warning_raw: number; critical_raw: number; warning_risk: number; critical_risk: number; persistence_steps: number; confidence_bins?: Record<'HIGH' | 'MEDIUM' | 'LOW', number> }
  dataset: { name: string; assets: { id: string; label: string }[]; sampling_min: number; test_period: [string, string]; test_index: [number, number] }
  benchmark: Record<string, number | string>
  engine_timings: { median_ms?: Record<string, number>; p95_ms?: Record<string, number>; n_steps?: number; onnx_inference?: boolean }
  health_weights: Record<string, number>
}
export interface Timeline {
  assets: Record<string, { index: number[]; risk: (number | null)[]; status: number[]; confidence: (number | null)[]; health: (number | null)[] }>
  timestamps: string[]
  status_codes: Record<string, number>
  failures: { asset: string; event_id: number; type: string; failure_idx: number; failure_time: string; label: string }[]
}
export interface Incident {
  asset: string; label: string; event_id: number; type: string; failure_idx: number; onset_idx: number
  failure_time: string; onset_time: string; class?: string; signal_strength?: number; signal_metric?: string
  detected?: boolean; lead_time_min?: number | null; engine_lead_time_min?: number | null
}
export type Fault = 'none' | 'thermal_runaway' | 'load_oscillation' | 'intermittent_arcing' | 'sudden'
export interface ScenarioRequest { asset: string; month: number; fault: Fault; severity: number; precursor_hours: number; seed: number }
/** Session context: a held-out incident (local), a synthetic scenario or an upload (public). */
export interface SessionContext extends Partial<Incident> { fault?: Fault; severity?: number; month?: number; seed?: number; precursor_hours?: number; rows?: number }
export interface SessionSummary { id: string; assets: string[]; mode: string; start: number; stop: number; cursor: number; incident?: SessionContext | null; warmup_ms: number; start_timestamp: string; stop_timestamp: string }
export interface StepResult { session: string; cursor: number; done: boolean; records: Record<string, EngineRecord[]>; events: FeedEvent[]; alerts: Alert[] }
