import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { ArrowDownRight, ArrowRight, ArrowUpRight } from 'lucide-react'
import { confBin, dur, minutesBetween, pct, riskPct } from '../lib/format'
import { C } from '../lib/palette'
import StatusBadge from './StatusBadge'
import type { EngineRecord } from '../lib/types'
import { Derived } from './Panel'
import { useGrid, useModelSpec } from '../store/grid'

const STAGE_LABEL: Record<string, string> = { NORMAL: 'NORMAL OPERATION', WARNING: 'EARLY WARNING', CRITICAL: 'CRITICAL', OFFLINE: 'ASSET OFFLINE', WARMUP: 'WARMING UP' }

export function TrendIcon({ trend, size = 14 }: { trend?: string; size?: number }) {
  if (trend === 'rising') return <ArrowUpRight size={size} className="text-warn" aria-label="rising" />
  if (trend === 'falling') return <ArrowDownRight size={size} className="text-ok" aria-label="falling" />
  return <ArrowRight size={size} className="text-ink-3" aria-label="stable" />
}

/** Operational read-out for one asset: every value is model output or a labelled derived metric. */
export function IntelligenceCard({ rec, label }: { rec?: EngineRecord; label: string }) {
  const spec = useModelSpec()
  const bins = useGrid((s) => s.system?.thresholds.confidence_bins)
  if (!rec) return <div className="p-4 text-ink-3 text-sm">No data.</div>
  const since = rec.warning_since ? minutesBetween(rec.warning_since, rec.timestamp) : null
  const conf = rec.confidence
  const row = 'flex items-baseline justify-between gap-3 border-b border-line/70 py-1.5'
  return (
    <div className="text-[12.5px]">
      <div className="mb-2 flex items-center justify-between">
        <div><div className="label">Asset</div><div className="text-[15px] font-semibold">Transformer {label}</div></div>
        <StatusBadge status={rec.status} size="lg" />
      </div>
      <div className={row}><span className="label">Current risk (calibrated)</span>
        <motion.span key={Math.round((rec.risk ?? 0) * 1000)} initial={{ opacity: 0.4 }} animate={{ opacity: 1 }} className="num text-[22px] font-semibold">{riskPct(rec.risk, 1)}</motion.span></div>
      <div className={row}><span className="label">Raw model score</span><span className="num text-ink-2">{rec.raw_score == null ? '—' : rec.raw_score.toFixed(3)}</span></div>
      <div className={row}><span className="label">Trend (1 h)</span><span className="flex items-center gap-1 num"><TrendIcon trend={rec.trend} />{rec.trend ?? '—'} <span className="text-ink-3">({rec.trend_delta_1h == null ? '—' : `${rec.trend_delta_1h >= 0 ? '+' : ''}${(rec.trend_delta_1h * 100).toFixed(1)} pp`})</span></span></div>
      <div className={row}><span className="label">Model confidence</span><span className="num">{pct(conf)} <span className="text-[10.5px] text-ink-3">{confBin(conf, bins)}</span></span></div>
      <div className={row}><span className="label">Warning window</span><span className="num">{rec.warning_window_min ? `failure predicted ≤ ${dur(rec.warning_window_min)}` : '—'}</span></div>
      <div className={row}><span className="label">Warning active for</span><span className="num">{since == null ? '—' : dur(since)}</span></div>
      <div className={row}><span className="label">Primary model-attributed signal</span><span className="text-right">{rec.signals?.[0]?.signal ?? '—'} <span className="num text-ink-3">{rec.signals?.[0] ? pct(rec.signals[0].share) : ''}</span></span></div>
      <div className={row}><span className="label">Secondary signal</span><span className="text-right">{rec.signals?.[1]?.signal ?? '—'} <span className="num text-ink-3">{rec.signals?.[1] ? pct(rec.signals[1].share) : ''}</span></span></div>
      <div className="flex items-baseline justify-between gap-3 py-1.5"><span className="label">Status</span><span className="font-mono text-[12px] font-semibold">{STAGE_LABEL[rec.status]}</span></div>
      <p className="mt-1 text-[10.5px] leading-snug text-ink-3">Confidence = agreement of 3 independently-seeded models (model confidence, not physical certainty). Calibrated risk = P(failure within {spec.horizon}) after isotonic calibration fitted on validation data.</p>
    </div>
  )
}

export function HealthBreakdown({ rec }: { rec?: EngineRecord }) {
  const h = rec?.health
  const items: [string, keyof NonNullable<typeof h>['components']][] = [
    ['Thermal stability', 'thermal_stability'], ['Load stability', 'load_stability'], ['Sensor quality', 'sensor_quality'],
    ['Failure risk (inverse)', 'failure_risk'], ['Model confidence', 'model_confidence']]
  const col = (v: number) => (v >= 75 ? C.ok : v >= 45 ? C.warn : C.crit)
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-baseline gap-2"><span className="label">Health</span><span className="num text-[26px] font-semibold" style={{ color: h ? col(h.score) : undefined }}>{h ? Math.round(h.score) : '—'}</span><span className="text-ink-3 text-[12px]">/ 100</span></div>
        <Derived>Derived application health indicator</Derived>
      </div>
      <ul className="space-y-1.5">
        {items.map(([l, k]) => {
          const v = h?.components[k]
          return (
            <li key={k} className="grid grid-cols-[130px_1fr_34px] items-center gap-2 text-[11.5px]">
              <span className="text-ink-2">{l}</span>
              <div className="h-1.5 overflow-hidden rounded bg-line" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={v ?? 0} aria-label={l}>
                <motion.div className="h-full rounded" style={{ background: v == null ? C.line : col(v) }} animate={{ width: `${v ?? 0}%` }} transition={{ duration: 0.4 }} />
              </div>
              <span className="num text-right text-ink-2">{v == null ? '—' : Math.round(v)}</span>
            </li>
          )
        })}
      </ul>
      <p className="mt-2 text-[10.5px] leading-snug text-ink-3">Weighted 40/20/20/10/10 (risk/thermal/load/sensor/confidence). Thermal and load stability are scaled against each asset's own training-period distribution. Not an official grid-reliability standard.</p>
    </div>
  )
}

type Pattern = { signal: string; alarms: number; followed_by_failure: number; share_followed: number }
let patternCache: Promise<Pattern[]> | null = null

/** Historical pattern comparison for the current primary signal (engine-log alarms, held-out period). */
function HistoricalPattern({ signal }: { signal?: string }) {
  const [rows, setRows] = useState<Pattern[] | null>(null)
  useEffect(() => { (patternCache ??= api.patterns().catch(() => [])).then(setRows) }, [])
  if (!signal || !rows) return null
  const r = rows.find((x) => x.signal === signal)
  return (
    <div className="mt-3 rounded border border-line bg-bg px-2.5 py-2 text-[11.5px]">
      <div className="label mb-0.5">Historical pattern</div>
      {r ? <span className="text-ink-2">In the held-out period, <b className="text-ink">{r.alarms}</b> alarms had <b className="text-ink">{signal}</b> as the primary signal;
        <b className="text-ink"> {r.followed_by_failure}</b> ({pct(r.share_followed)}) were followed by a simulated failure within 24 h.</span>
        : <span className="text-ink-3">No past alarm had “{signal}” as its primary signal.</span>}
    </div>
  )
}

/** "Why is risk changing?" — integrated-gradients attribution grouped by sensor. */
export function Explain({ rec }: { rec?: EngineRecord }) {
  const spec = useModelSpec()
  const at = (rec?.attribution ?? []).slice(0, 7)
  const tp = rec?.temporal_attribution
  const max = Math.max(0.0001, ...at.map((a) => a.share))
  return (
    <div>
      <ul className="space-y-1.5" aria-label="Model-attributed signals">
        {at.map((a) => (
          <li key={a.signal} className="grid grid-cols-[150px_1fr_40px] items-center gap-2 text-[11.5px]">
            <span className="truncate text-ink-2" title={a.signal}>{a.signal}</span>
            <div className="h-2 rounded bg-line/60"><motion.div className="h-2 rounded" style={{ background: a.direction === 'lowers risk' ? C.s[2] : C.s[1] }}
              animate={{ width: `${(a.share / max) * 100}%` }} transition={{ duration: 0.35 }} /></div>
            <span className="num text-right text-ink-2">{pct(a.share)}</span>
          </li>
        ))}
        {!at.length && <li className="text-[12px] text-ink-3">No attribution yet.</li>}
      </ul>
      {at.some((a) => a.direction) && <div className="mt-2 flex gap-3 text-[10.5px] text-ink-3">
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded" style={{ background: C.s[1] }} />pushes risk up</span>
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded" style={{ background: C.s[2] }} />pushes risk down</span></div>}
      {tp && tp.length > 0 && (
        <div className="mt-3">
          <div className="label mb-1">Where in the {spec.window} window the evidence sits</div>
          <svg viewBox={`0 0 ${tp.length} 20`} preserveAspectRatio="none" className="h-7 w-full" role="img" aria-label="Temporal attribution over the input window">
            {tp.map((v, i) => { const m = Math.max(...tp); return <rect key={i} x={i} y={20 - (v / m) * 20} width={0.85} height={(v / m) * 20} fill={C.accent} opacity={0.75} /> })}
          </svg>
          <div className="flex justify-between font-mono text-[9.5px] text-ink-3"><span>−{spec.window}</span><span>−{spec.windowHalf}</span><span>now</span></div>
        </div>
      )}
      <HistoricalPattern signal={rec?.signals?.[0]?.signal} />
      <p className="mt-2 text-[10.5px] leading-snug text-ink-3"><b>What the model relied on</b> (integrated gradients, grouped by sensor) — <b>not</b> what physically caused a failure. Attribution is associational.</p>
    </div>
  )
}
