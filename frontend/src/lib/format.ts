import type { EngineRecord, Status } from './types'

export const pct = (v: number | null | undefined, d = 0) => (v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(d)}%`)
/** Calibrated risk: never display 100 % (isotonic calibration can saturate on validation; that is not certainty). */
export const riskPct = (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? '—' : v >= 0.995 ? '≥ 99%' : `${(v * 100).toFixed(d)}%`)
/** Model-confidence bin; cut points come from the API (`/api/system` → thresholds.confidence_bins). */
export const confBin = (c: number | null | undefined, bins?: Record<string, number>) =>
  c == null || !bins ? '' : c >= bins.HIGH ? 'HIGH' : c >= bins.MEDIUM ? 'MEDIUM' : 'LOW'
const MODEL_NAMES: Record<string, string> = { gru_attention: 'GRU + temporal attention', gru: 'GRU', lstm: 'LSTM', lstm_attention: 'LSTM + attention' }
export const modelName = (type: string | undefined) => (type ? MODEL_NAMES[type] ?? type : '—')
/** Live engine record for the UI. While an asset is offline (after a trip) its rows are never scored, so the
 *  engine's internal risk/score/confidence (computed only to keep buffers continuous) are not displayed. */
export const liveRecord = (x: EngineRecord): EngineRecord =>
  ({ ...x, source: 'live' as const, ...(x.offline ? { risk: null, raw_score: null, confidence: null } : {}) })
export const num = (v: number | null | undefined, d = 2) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(d))
export const dur = (min: number | null | undefined) => {
  if (min == null || Number.isNaN(min)) return '—'
  const h = Math.floor(min / 60), m = Math.round(min % 60)
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m`
}
export const ts = (s: string | undefined | null) => (s ? s.replace('T', ' ').slice(0, 16) : '—')
export const time = (s: string | undefined | null) => (s ? s.replace('T', ' ').slice(11, 16) : '—')
export const STATUS_CODE: Record<number, Status> = { 0: 'WARMUP', 1: 'NORMAL', 2: 'WARNING', 3: 'CRITICAL', 4: 'OFFLINE' }
export const minutesBetween = (a: string, b: string) => (new Date(b.replace(' ', 'T')).getTime() - new Date(a.replace(' ', 'T')).getTime()) / 60000

/** Convert an engine-log row (flat) into the EngineRecord shape used by the UI. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function fromLog(r: Record<string, any>, asset: string, horizonMin: number | null = null): EngineRecord {
  const sensors: Record<string, number> = {}
  for (const c of ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']) sensors[c] = r[c]
  let attribution: { signal: string; share: number }[] = []
  try {
    const a = typeof r.attribution === 'string' ? JSON.parse(r.attribution) : r.attribution ?? {}
    attribution = Object.entries(a as Record<string, number>).map(([signal, share]) => ({ signal, share })).sort((x, y) => y.share - x.share)
  } catch { /* empty */ }
  const signals = [r.signal1 && { signal: r.signal1, share: r.signal1_share }, r.signal2 && { signal: r.signal2, share: r.signal2_share }].filter(Boolean) as { signal: string; share: number }[]
  return {
    index: r.index, timestamp: r.timestamp, asset, sensors, offline: !!r.offline, status: r.status, raw_score: r.offline ? null : r.raw_score, risk: r.offline ? null : r.risk,
    confidence: r.offline ? null : r.confidence, trend: r.trend ?? undefined, trend_delta_1h: r.trend_delta_1h ?? undefined, warning_since: r.warning_since,
    warning_window_min: r.status === 'WARNING' || r.status === 'CRITICAL' ? horizonMin : null, signals, attribution,
    health: r.health == null ? undefined : {
      score: r.health,
      components: { thermal_stability: r.h_thermal_stability, load_stability: r.h_load_stability, sensor_quality: r.h_sensor_quality, failure_risk: r.h_failure_risk, model_confidence: r.h_model_confidence },
    },
    ground_truth: { failure_event: r.failure_event, sim_phase: r.sim_phase, sim_event_type: r.sim_event_type },
    lead_time_min: r.lead_time_min, timings_ms: r.t_total_ms ? { total: r.t_total_ms } : undefined, source: 'log',
  }
}
