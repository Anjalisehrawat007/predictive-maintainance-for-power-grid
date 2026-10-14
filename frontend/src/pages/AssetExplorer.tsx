import { History, LocateFixed } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Explain, HealthBreakdown, IntelligenceCard, TrendIcon } from '../components/Intelligence'
import Panel, { Empty } from '../components/Panel'
import { api } from '../lib/api'
import { dur, riskPct, ts } from '../lib/format'
import { C, SENSOR_NAME, TYPE_COLOR, tooltipStyle } from '../lib/palette'
import StatusBadge from '../components/StatusBadge'
import type { Alert, Incident } from '../lib/types'
import { useGrid } from '../store/grid'

const axis = { stroke: C.line2, tick: { fill: C.ink3, fontSize: 10.5, fontFamily: 'IBM Plex Mono' }, tickLine: false }

/** Per-asset intelligence view: the *current* state comes from the shared clock (Command Center
 *  cursor); the long-range panels come from the engine log of the whole held-out period. */
export default function AssetExplorer() {
  const { id } = useParams()
  const nav = useNavigate()
  const g = useGrid()
  const asset = id ?? g.selected ?? g.assets[0]
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  useEffect(() => { api.incidents().then(setIncidents).catch(() => {}); api.alerts().then(setAlerts).catch(() => {}) }, [])
  useEffect(() => { if (asset && asset !== g.selected) g.select(asset) }, [asset]) // eslint-disable-line react-hooks/exhaustive-deps
  const labels = useMemo(() => Object.fromEntries((g.system?.dataset.assets ?? []).map((a) => [a.id, a.label])), [g.system])
  const tl = g.timeline
  const daily = useMemo(() => {
    if (!tl || !tl.assets[asset]) return []
    const a = tl.assets[asset]
    const out: { day: string; maxRisk: number; meanHealth: number; alarmShare: number; minConf: number; i0: number }[] = []
    for (let k = 0; k < a.index.length; k += 96) {
      const sl = (arr: (number | null)[]) => arr.slice(k, k + 96).filter((x): x is number => x != null)
      const r = sl(a.risk), h = sl(a.health), c = sl(a.confidence)
      const st = a.status.slice(k, k + 96)
      out.push({ day: tl.timestamps[k].slice(0, 10), maxRisk: r.length ? Math.max(...r) * 100 : 0, meanHealth: h.length ? h.reduce((x, y) => x + y, 0) / h.length : 0,
        alarmShare: (st.filter((s) => s === 2 || s === 3).length / st.length) * 100, minConf: c.length ? Math.min(...c) * 100 : 0, i0: a.index[k] })
    }
    return out
  }, [tl, asset])
  if (!g.system || !tl) return <Empty>Connecting…</Empty>
  const rec = g.history[asset]?.at(-1)
  const inc = incidents.filter((i) => i.asset === asset)
  const al = alerts.filter((a) => a.asset === asset && a.source.startsWith('engine log'))
  const leads = inc.filter((i) => i.engine_lead_time_min != null).map((i) => ({ t: ts(i.failure_time).slice(5, 10), lead: (i.engine_lead_time_min ?? 0) / 60, type: i.type }))
  const sensorsNow = rec?.sensors ?? {}
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="label">Asset explorer</div>
          <h1 className="flex items-center gap-3 text-[20px] font-semibold">Transformer {labels[asset]} <span className="text-[12px] font-normal text-ink-3">ETDataset {asset} · held-out period {g.system.dataset.test_period.map((x) => x.slice(0, 10)).join(' → ')}</span></h1>
        </div>
        <div className="flex items-center gap-2" role="group" aria-label="Choose asset">
          {g.assets.map((a) => <button key={a} onClick={() => nav(`/assets/${a}`)} aria-pressed={a === asset}
            className={`rounded border px-3 py-1 text-[12px] ${a === asset ? 'border-ink-3 bg-surface-2' : 'border-line text-ink-3'}`}>Transformer {labels[a]}</button>)}
          <Link to="/" className="flex items-center gap-1 rounded border border-line px-3 py-1 text-[12px] text-accent"><LocateFixed size={13} />Command center</Link>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
        <Panel className="xl:col-span-4" kicker={`At shared clock · ${ts(rec?.timestamp)}`} title="Current state" right={rec && <StatusBadge status={rec.status} />}>
          <IntelligenceCard rec={rec} label={labels[asset]} />
        </Panel>
        <Panel className="xl:col-span-4" kicker="Readings at shared clock" title="Sensors · temperature & load">
          <table className="w-full text-[12px]"><tbody>
            {Object.entries(sensorsNow).map(([k, v]) => (
              <tr key={k} className="border-b border-line/60"><td className="py-1.5 text-ink-2">{SENSOR_NAME[k]}</td><td className="num py-1.5 text-right">{v?.toFixed?.(2)}</td></tr>))}
          </tbody></table>
          <p className="mt-2 text-[10.5px] text-ink-3">Raw SCADA values (ETT units). Loads are active ("useful") and reactive ("useless") power at three voltage levels; OT is oil temperature.</p>
          <div className="mt-3 flex items-center gap-2 text-[12px]"><span className="label">Trend</span><TrendIcon trend={rec?.trend} /> {rec?.trend ?? '—'}</div>
        </Panel>
        <Panel className="xl:col-span-4" kicker="Derived" title="Health & anomaly"><HealthBreakdown rec={rec} /></Panel>
      </div>
      <Panel kicker="Engine log · daily aggregates" title="Risk, health and alarm history across the held-out period"
        right={<span className="text-[10.5px] text-ink-3">click a day to open it in the command center</span>}>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={daily} margin={{ top: 6, right: 12, left: -8, bottom: 0 }}
            onClick={(e) => { const p = (e as unknown as { activePayload?: { payload: { i0: number } }[] })?.activePayload?.[0]?.payload; if (p) { g.seek(p.i0 + 48); nav('/') } }}>
            <CartesianGrid stroke={C.line} vertical={false} />
            <XAxis dataKey="day" {...axis} minTickGap={30} tickFormatter={(d) => d.slice(5)} />
            <YAxis {...axis} domain={[0, 100]} width={40} />
            <Tooltip {...tooltipStyle} formatter={(v, n) => [`${Number(v).toFixed(1)}${n === 'meanHealth' ? '' : '%'}`, { maxRisk: 'Peak calibrated risk', meanHealth: 'Mean health (derived)', alarmShare: 'Time in alarm', minConf: 'Lowest confidence' }[String(n)] ?? String(n)]} />
            {inc.map((i) => <ReferenceLine key={i.event_id} x={i.failure_time.slice(0, 10)} stroke={TYPE_COLOR[i.type]} strokeOpacity={0.7} />)}
            <Bar dataKey="alarmShare" fill={C.warn} opacity={0.5} />
            <Line dataKey="maxRisk" stroke={C.s[0]} dot={false} strokeWidth={1.5} />
            <Line dataKey="meanHealth" stroke={C.s[2]} dot={false} strokeWidth={1.5} />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="mt-1 flex flex-wrap gap-4 text-[10.5px] text-ink-3">
          <span className="flex items-center gap-1"><span className="h-[2px] w-4" style={{ background: C.s[0] }} />peak calibrated risk (%)</span>
          <span className="flex items-center gap-1"><span className="h-[2px] w-4" style={{ background: C.s[2] }} />mean health (derived)</span>
          <span className="flex items-center gap-1"><span className="h-2 w-3" style={{ background: C.warn, opacity: .5 }} />% of day in alarm</span>
          <span>vertical lines = simulated failures (colour = type)</span>
        </div>
      </Panel>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
        <Panel className="xl:col-span-5" kicker="Held-out period" title={`Historical failures (${inc.length})`} bodyClass="p-0">
          <div className="scroll-thin max-h-[320px] overflow-y-auto">
            <table className="w-full text-[12px]">
              <thead className="sticky top-0 bg-surface text-left"><tr className="label">{['When', 'Type', 'Class', 'Warned', 'Lead', ''].map((h) => <th key={h} className="px-3 py-1.5 font-normal">{h}</th>)}</tr></thead>
              <tbody>{inc.map((i) => (
                <tr key={i.event_id} className="border-t border-line/60">
                  <td className="num px-3 py-1.5">{ts(i.failure_time)}</td>
                  <td className="px-3"><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: TYPE_COLOR[i.type] }} />{i.type.replace('_', ' ')}</td>
                  <td className="num px-3">{i.class}</td>
                  <td className="px-3">{i.engine_lead_time_min != null ? <span className="text-ok">✓ yes</span> : <span className="text-warn">✕ no</span>}</td>
                  <td className="num px-3">{dur(i.engine_lead_time_min)}</td>
                  <td className="px-3"><Link to={`/replay?asset=${i.asset}&event=${i.event_id}`} className="flex items-center gap-1 text-accent" aria-label={`Replay failure at ${ts(i.failure_time)}`}><History size={12} />replay</Link></td>
                </tr>))}</tbody>
            </table>
          </div>
        </Panel>
        <Panel className="xl:col-span-4" kicker="Engine log" title={`Warning history (${al.length} alarm episodes)`} bodyClass="p-0">
          <div className="scroll-thin max-h-[320px] overflow-y-auto">
            <table className="w-full text-[12px]">
              <thead className="sticky top-0 bg-surface text-left"><tr className="label">{['Opened', 'Severity', 'Peak risk', 'Outcome'].map((h) => <th key={h} className="px-3 py-1.5 font-normal">{h}</th>)}</tr></thead>
              <tbody>{al.map((a) => (
                <tr key={a.id} className="cursor-pointer border-t border-line/60 hover:bg-surface-2" onClick={() => { g.seek(a.index + 4); nav('/') }}>
                  <td className="num px-3 py-1.5">{ts(a.opened)}</td><td className="px-3"><StatusBadge status={a.severity} /></td>
                  <td className="num px-3">{riskPct(a.risk, 1)}</td><td className={`px-3 text-[11px] ${a.outcome?.startsWith('failure') ? 'text-ok' : 'text-ink-3'}`}>{a.outcome}</td>
                </tr>))}</tbody>
            </table>
          </div>
        </Panel>
        <Panel className="xl:col-span-3" kicker="Engine-measured" title="Lead-time history">
          {leads.length ? <ResponsiveContainer width="100%" height={260}>
            <BarChart data={leads} margin={{ top: 6, right: 8, left: -16, bottom: 0 }}>
              <CartesianGrid stroke={C.line} vertical={false} />
              <XAxis dataKey="t" {...axis} />
              <YAxis {...axis} tickFormatter={(v) => `${v}h`} />
              <Tooltip {...tooltipStyle} formatter={(v) => [`${Number(v).toFixed(2)} h`, 'lead time']} />
              <Bar dataKey="lead" radius={[3, 3, 0, 0]}>{leads.map((l, k) => <Cell key={k} fill={TYPE_COLOR[l.type]} />)}</Bar>
            </BarChart>
          </ResponsiveContainer> : <Empty>No warned failures for this asset.</Empty>}
        </Panel>
      </div>
      <Panel kicker="Why is risk changing? (at shared clock)" title="Model-attributed signals"><Explain rec={rec} /></Panel>
    </div>
  )
}
