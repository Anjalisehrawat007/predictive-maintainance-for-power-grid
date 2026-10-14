import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { time, ts } from '../lib/format'
import { C, SENSOR_COLOR, SENSOR_NAME, tooltipStyle } from '../lib/palette'
import type { EngineRecord } from '../lib/types'

const axis = { stroke: C.line2, tick: { fill: C.ink3, fontSize: 10.5, fontFamily: 'IBM Plex Mono' }, tickLine: false }

interface Marker { index: number; label: string; color: string; dash?: string }

function offlineSpans(recs: EngineRecord[]) {
  const out: [string, string][] = []
  let s: string | null = null
  recs.forEach((r, i) => {
    if (r.offline && s == null) s = r.timestamp
    if ((!r.offline || i === recs.length - 1) && s != null) { out.push([s, r.timestamp]); s = null }
  })
  return out
}

/** Calibrated risk + raw score + thresholds + failure markers + predicted warning window. */
export function RiskChart({ recs, warn, crit, height = 220, markers = [], syncId, showRaw = true, showConf = false, predictedWindow }: {
  recs: EngineRecord[]; warn: number; crit: number; height?: number; markers?: Marker[]; syncId?: string; showRaw?: boolean; showConf?: boolean
  predictedWindow?: { from: string; to: string } | null
}) {
  const data: { t: string; risk: number | null; raw: number | null; conf: number | null; idx: number }[] = recs.map((r) => ({ t: r.timestamp, risk: r.risk == null ? null : r.risk * 100, raw: r.raw_score == null ? null : r.raw_score * 100,
    conf: r.confidence == null ? null : r.confidence * 100, idx: r.index }))
  let windowEnd: string | null = null
  if (predictedWindow && recs.length) {
    // empty future slots (no data) so the forecast window is visible to the right of "now"
    const t0 = new Date(predictedWindow.from.replace(' ', 'T') + 'Z').getTime()
    const n = Math.round((new Date(predictedWindow.to.replace(' ', 'T') + 'Z').getTime() - t0) / 900000)
    for (let k = 1; k <= n; k++) {
      const t = new Date(t0 + k * 900000).toISOString().replace('T', ' ').slice(0, 19)
      data.push({ t, risk: null, raw: null, conf: null, idx: -k })
      windowEnd = t
    }
  }
  const byIdx = new Map(recs.map((r) => [r.index, r.timestamp]))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} syncId={syncId} syncMethod="value" margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
        <defs>
          <linearGradient id="riskFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={C.s[0]} stopOpacity={0.35} /><stop offset="1" stopColor={C.s[0]} stopOpacity={0} /></linearGradient>
        </defs>
        <CartesianGrid stroke={C.line} vertical={false} />
        <XAxis dataKey="t" {...axis} tickFormatter={time} minTickGap={40} />
        <YAxis {...axis} domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickFormatter={(v) => `${v}%`} width={44} />
        <Tooltip {...tooltipStyle} labelFormatter={(l) => ts(String(l))}
          formatter={(v, n) => [v == null ? '—' : `${Number(v).toFixed(1)}%`, n === 'risk' ? 'Calibrated risk' : n === 'raw' ? 'Raw model score' : 'Model confidence']} />
        {offlineSpans(recs).map(([a, b]) => <ReferenceArea key={a} x1={a} x2={b} fill={C.off} fillOpacity={0.12} />)}
        {predictedWindow && windowEnd && <ReferenceArea x1={predictedWindow.from} x2={windowEnd} fill={C.warn} fillOpacity={0.08} stroke={C.warn} strokeOpacity={0.3} strokeDasharray="3 3"
          label={{ value: 'predicted window', position: 'insideTop', fill: C.warn, fontSize: 10 }} />}
        <ReferenceLine y={warn * 100} stroke={C.warn} strokeDasharray="4 3" label={{ value: `WARNING ${(warn * 100).toFixed(0)}%`, position: 'insideTopRight', fill: C.warn, fontSize: 10 }} />
        <ReferenceLine y={crit * 100} stroke={C.crit} strokeDasharray="4 3" label={{ value: `CRITICAL ${crit >= 0.995 ? '≥ 99' : (crit * 100).toFixed(0)}%`, position: 'insideTopLeft', fill: C.crit, fontSize: 10 }} />
        {markers.map((m) => byIdx.has(m.index) && <ReferenceLine key={`${m.label}${m.index}`} x={byIdx.get(m.index)} stroke={m.color} strokeDasharray={m.dash}
          label={{ value: m.label, position: 'insideTopLeft', fill: m.color, fontSize: 10 }} />)}
        <Area type="monotone" dataKey="risk" stroke={C.s[0]} strokeWidth={2} fill="url(#riskFill)" isAnimationActive={false} connectNulls={false} name="risk" />
        {showRaw && <Line type="monotone" dataKey="raw" stroke={C.ink3} strokeWidth={1} strokeDasharray="3 3" dot={false} isAnimationActive={false} name="raw" />}
        {showConf && <Line type="monotone" dataKey="conf" stroke={C.s[5]} strokeWidth={1.2} dot={false} isAnimationActive={false} name="conf" />}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

/** Sensor panel: each channel z-scored within the visible window so different units share one axis. */
export function SensorChart({ recs, sensors, height = 170, syncId, markers = [], futureUntil }: {
  recs: EngineRecord[]; sensors: string[]; height?: number; syncId?: string; markers?: Marker[]; futureUntil?: string | null
}) {
  const stats = Object.fromEntries(sensors.map((c) => {
    const v = recs.map((r) => r.sensors?.[c]).filter((x): x is number => x != null && !Number.isNaN(x))
    const m = v.reduce((a, b) => a + b, 0) / (v.length || 1)
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length || 1)) || 1
    return [c, { m, sd }]
  }))
  const data = recs.map((r) => {
    const o: Record<string, number | string | null> = { t: r.timestamp }
    for (const c of sensors) { const x = r.sensors?.[c]; o[c] = x == null ? null : (x - stats[c].m) / stats[c].sd; o[`${c}_v`] = x ?? null }
    return o
  })
  if (futureUntil && recs.length) {        // same empty future slots as the risk chart, so time columns align
    const t0 = new Date(recs.at(-1)!.timestamp.replace(' ', 'T') + 'Z').getTime()
    const n = Math.round((new Date(futureUntil.replace(' ', 'T') + 'Z').getTime() - t0) / 900000)
    for (let k = 1; k <= n; k++) data.push({ t: new Date(t0 + k * 900000).toISOString().replace('T', ' ').slice(0, 19) })
  }
  const byIdx = new Map(recs.map((r) => [r.index, r.timestamp]))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} syncId={syncId} syncMethod="value" margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
        <CartesianGrid stroke={C.line} vertical={false} />
        <XAxis dataKey="t" {...axis} tickFormatter={time} minTickGap={40} />
        <YAxis {...axis} width={44} tickFormatter={(v) => `${v}σ`} />
        <Tooltip {...tooltipStyle} labelFormatter={(l) => ts(String(l))}
          formatter={(v, n, item) => [`${(item?.payload?.[`${n}_v`] as number | null)?.toFixed?.(2) ?? '—'}  (${Number(v).toFixed(1)}σ)`, SENSOR_NAME[String(n)] ?? String(n)]} />
        {markers.map((m) => byIdx.has(m.index) && <ReferenceLine key={`${m.label}${m.index}`} x={byIdx.get(m.index)} stroke={m.color} strokeDasharray={m.dash} />)}
        {sensors.map((c) => <Line key={c} type="monotone" dataKey={c} stroke={SENSOR_COLOR[c]} strokeWidth={1.4} dot={false} isAnimationActive={false} connectNulls={false} />)}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function SensorLegend({ sensors, active, onToggle }: { sensors: string[]; active: string[]; onToggle: (c: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Sensor channels">
      {sensors.map((c) => {
        const on = active.includes(c)
        return (
          <button key={c} onClick={() => onToggle(c)} aria-pressed={on} title={SENSOR_NAME[c]}
            className={`flex items-center gap-1.5 rounded border px-1.5 py-0.5 font-mono text-[10.5px] ${on ? 'border-line-2 text-ink' : 'border-line text-ink-3'}`}>
            <span className="h-[2px] w-3 rounded" style={{ background: on ? SENSOR_COLOR[c] : C.line2 }} aria-hidden />{c}
          </button>
        )
      })}
    </div>
  )
}
