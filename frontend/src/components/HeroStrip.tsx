import { dur, minutesBetween, pct, riskPct } from '../lib/format'
import { C } from '../lib/palette'
import { STATUS_META } from '../lib/status'
import type { EngineRecord, Status } from '../lib/types'
import Sparkline from './Sparkline'
import { useGrid, useModelSpec } from '../store/grid'

/** Top-level intelligence: aggregated over all assets at the shared cursor. */
export default function HeroStrip({ history, nAlerts }: { history: Record<string, EngineRecord[]>; nAlerts: number }) {
  const spec = useModelSpec()
  const assets = Object.keys(history)
  const last = assets.map((a) => history[a].at(-1)).filter(Boolean) as EngineRecord[]
  const series = (f: (r: EngineRecord) => number | null | undefined) => {
    const n = Math.min(...assets.map((a) => history[a].length))
    return Array.from({ length: Math.min(n, 96) }, (_, k) => {
      const vals = assets.map((a) => f(history[a][history[a].length - Math.min(n, 96) + k])).filter((x): x is number => x != null)
      return vals.length ? vals : null
    })
  }
  const healthS = series((r) => r.health?.score).map((v) => (v ? Math.min(...v) : null))
  const riskS = series((r) => r.risk).map((v) => (v ? Math.max(...v) : null))
  const confS = series((r) => r.confidence).map((v) => (v ? Math.min(...v) : null))
  const expS = series((r) => r.risk).map((v) => (v ? v.reduce((a, b) => a + b, 0) : null))
  const expected = last.length ? last.reduce((a, r) => a + (r.risk ?? 0), 0) : null
  const health = last.length ? Math.min(...last.map((r) => r.health?.score ?? 100)) : null
  const risk = last.length ? Math.max(...last.map((r) => r.risk ?? 0)) : null
  const conf = last.length ? Math.min(...last.map((r) => r.confidence ?? 1)) : null
  const warnRisk = useGrid((st) => st.system?.thresholds.warning_risk)
  // first step over the threshold: the alert policy needs 2 consecutive steps before raising WARNING
  const pending = last.some((r) => r.status === 'NORMAL' && warnRisk != null && (r.risk ?? 0) >= warnRisk)
  const worst: Status = last.reduce<Status>((w, r) => (STATUS_META[r.status].rank > STATUS_META[w].rank ? r.status : w), 'NORMAL')
  const warns = last.filter((r) => r.status === 'WARNING' || r.status === 'CRITICAL')
  const earliest = warns.filter((r) => r.warning_since).map((r) => minutesBetween(r.warning_since!, r.timestamp))
  const cells = [
    { k: 'Grid health', v: health == null ? '—' : Math.round(health).toString(), sub: 'min over assets · derived', spark: healthS, c: C.s[2], max: 100 },
    { k: 'Current risk', v: riskPct(risk, 1), sub: `highest asset · ${pending ? 'over threshold · confirming (2 steps)' : STATUS_META[worst].label}`, spark: riskS, c: C.s[0], max: 1, color: STATUS_META[worst].color },
    { k: 'Active warnings', v: String(warns.length), sub: `${nAlerts} open alert${nAlerts === 1 ? '' : 's'}`, color: warns.length ? C.warn : undefined },
    { k: `Predicted events ≤ ${spec.horizon}`, v: expected == null ? '—' : expected.toFixed(2), sub: 'Σ calibrated risk = expected failures', spark: expS, c: C.s[3] },
    { k: 'Earliest active warning', v: earliest.length ? dur(Math.max(...earliest)) : '—', sub: 'time since first alarm' },
    { k: 'Model confidence', v: pct(conf), sub: 'lowest asset · seed agreement', spark: confS, c: C.s[5], max: 1 },
  ]
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 xl:grid-cols-6" role="region" aria-label="Grid intelligence summary">
      {cells.map((c) => (
        <div key={c.k} className="bg-surface/95 px-3 py-2.5">
          <div className="label">{c.k}</div>
          <div className="mt-0.5 flex items-end justify-between gap-2">
            <span className="num text-[24px] font-semibold leading-none" style={{ color: c.color }}>{c.v}</span>
            {c.spark && <Sparkline values={c.spark} color={c.c} max={c.max} label={c.k} />}
          </div>
          <div className="mt-1 text-[10.5px] text-ink-3">{c.sub}</div>
        </div>
      ))}
    </div>
  )
}
