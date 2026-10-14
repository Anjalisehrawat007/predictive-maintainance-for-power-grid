import { Check, History, RefreshCw, Search, Undo2 } from 'lucide-react'
import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Panel, { Empty } from '../components/Panel'
import { api } from '../lib/api'
import { dur, minutesBetween, pct, ts, riskPct } from '../lib/format'
import StatusBadge from '../components/StatusBadge'
import type { Alert, Incident } from '../lib/types'
import { useGrid } from '../store/grid'

type Filter = { severity: string; asset: string; status: string; outcome: string }

export default function AlertCenter() {
  const g = useGrid()
  const nav = useNavigate()
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string>()
  const [open, setOpen] = useState<string | null>(null)
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [f, setF] = useState<Filter>({ severity: 'all', asset: 'all', status: 'all', outcome: 'all' })
  const labels = useMemo(() => Object.fromEntries((g.system?.dataset.assets ?? []).map((a) => [a.id, a.label])), [g.system])
  const fetchAlerts = useCallback(() => api.alerts().then((a) => { setAlerts(a); setErr(undefined) }).catch((e) => setErr(e.message)).finally(() => setLoading(false)), [])
  const load = () => { setLoading(true); fetchAlerts() }
  useEffect(() => { fetchAlerts(); api.incidents().then(setIncidents).catch(() => {}) }, [fetchAlerts])

  const shown = alerts.filter((a) => (f.severity === 'all' || a.severity === f.severity) && (f.asset === 'all' || a.asset === f.asset)
    && (f.status === 'all' || a.status === f.status)
    && (f.outcome === 'all' || (f.outcome === 'failure' ? a.outcome?.startsWith('failure') : a.outcome && !a.outcome.startsWith('failure'))))
  const ack = async (a: Alert) => {
    const prev = alerts
    setAlerts((x) => x.map((y) => (y.id === a.id ? { ...y, status: a.status === 'acknowledged' ? 'resolved' : 'acknowledged' } : y)))   // optimistic
    try { if (a.status === 'acknowledged') await api.unack(a.id); else await api.ack(a.id); load() } catch (e) { setAlerts(prev); setErr((e as Error).message) }
  }
  const investigate = async (a: Alert) => { g.select(a.asset); await g.seek(a.index + 4); nav('/') }
  const replay = (a: Alert) => {
    // replay the failure that followed this alert (if any) — otherwise the asset's nearest later failure
    const later = incidents.filter((i) => i.asset === a.asset && i.failure_time >= a.opened).sort((x, y) => x.failure_time.localeCompare(y.failure_time))
    nav(later.length ? `/replay?asset=${a.asset}&event=${later[0].event_id}` : `/replay?asset=${a.asset}`)
  }
  const sel = (k: keyof Filter, opts: [string, string][]) => (
    <label className="flex items-center gap-1.5 text-[11.5px] text-ink-3">{k}
      <select value={f[k]} onChange={(e) => setF({ ...f, [k]: e.target.value })} className="rounded border border-line bg-bg px-1.5 py-1 text-ink">
        {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>)
  const stats = {
    total: alerts.length, open: alerts.filter((a) => a.status === 'open').length, ack: alerts.filter((a) => a.status === 'acknowledged').length,
    followed: alerts.filter((a) => a.outcome?.startsWith('failure')).length,
  }
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div><div className="label">Alert center</div><h1 className="text-[20px] font-semibold">Alerts raised by the engine</h1>
          <p className="text-[12.5px] text-ink-2">Engine-log alerts cover the whole held-out period; live-session alerts appear as you stream. "Outcome" uses simulated ground truth (failure within 24 h) and is shown for evaluation only. Each row is one engine alert lifecycle; the offline evaluation instead merges alarms with gaps ≤ 1 h into episodes and scores them against the 2 h horizon (see Failure Intelligence), so the counts differ by definition.</p></div>
        <button onClick={load} className="flex items-center gap-1.5 rounded border border-line-2 px-2.5 py-1.5 text-[12px]" aria-label="Refresh alerts"><RefreshCw size={13} />Refresh</button>
      </div>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-4">
        {[['Alerts', stats.total], ['Open', stats.open], ['Acknowledged', stats.ack], ['Followed by a failure ≤ 24 h', `${stats.followed} (${pct(stats.followed / (stats.total || 1))})`]].map(([k, v]) => (
          <div key={String(k)} className="bg-surface px-3 py-2.5"><div className="label">{k}</div><div className="num text-[22px] font-semibold">{v}</div></div>))}
      </div>
      <Panel kicker="Filter" title={`${shown.length} alerts`} right={<div className="flex flex-wrap gap-3">
        {sel('severity', [['all', 'all'], ['WARNING', 'warning'], ['CRITICAL', 'critical']])}
        {sel('asset', [['all', 'all'], ...g.assets.map((a) => [a, labels[a]] as [string, string])])}
        {sel('status', [['all', 'all'], ['open', 'open'], ['acknowledged', 'acknowledged'], ['resolved', 'resolved']])}
        {sel('outcome', [['all', 'all'], ['failure', 'failure followed'], ['none', 'no failure (false alarm)']])}
      </div>} bodyClass="p-0">
        {err && <div role="alert" className="m-3 rounded border border-crit/50 bg-crit/10 px-3 py-2 text-sm text-crit">{err}</div>}
        {loading && !alerts.length ? <Empty>Loading alerts…</Empty> : !shown.length ? <Empty>No alerts match these filters.</Empty> :
          <div className="scroll-thin max-h-[640px] overflow-y-auto">
            <table className="w-full text-[12px]">
              <thead className="sticky top-0 z-10 bg-surface text-left"><tr className="label">
                {['Severity', 'Asset', 'Opened', 'Duration', 'Peak risk', 'Confidence', 'Warning window', 'Model-attributed signals', 'Status', 'Actions'].map((h) => <th key={h} className="px-3 py-2 font-normal">{h}</th>)}</tr></thead>
              <tbody>{shown.map((a) => (
                <Fragment key={a.id}>
                  <tr className="border-t border-line/60 hover:bg-surface-2/60">
                    <td className="px-3 py-2"><StatusBadge status={a.severity} /></td>
                    <td className="px-3 font-mono">{labels[a.asset]}</td>
                    <td className="num px-3">{ts(a.opened)}</td>
                    <td className="num px-3">{a.closed ? dur(minutesBetween(a.opened, a.closed)) : 'active'}</td>
                    <td className="num px-3">{riskPct(a.risk, 1)}</td>
                    <td className="num px-3">{pct(a.confidence)}</td>
                    <td className="px-3 text-ink-2">{a.warning_window_min ? `≤ ${dur(a.warning_window_min)}` : '—'}</td>
                    <td className="px-3 text-ink-2"><button className="text-left hover:text-ink" onClick={() => setOpen(open === a.id ? null : a.id)} aria-expanded={open === a.id}>
                      {(a.signals ?? []).slice(0, 2).map((s) => s.signal).join(' · ') || '—'}</button></td>
                    <td className="px-3"><span className={`rounded px-1.5 py-0.5 font-mono text-[10.5px] ${a.status === 'open' ? 'bg-warn/15 text-warn' : a.status === 'acknowledged' ? 'bg-accent/15 text-accent' : 'bg-line text-ink-3'}`}>{a.status}</span></td>
                    <td className="px-3">
                      <div className="flex gap-1">
                        <button onClick={() => investigate(a)} className="flex items-center gap-1 rounded border border-line-2 px-1.5 py-1 text-[11px]" aria-label={`Investigate alert at ${ts(a.opened)}`}><Search size={12} />Investigate</button>
                        <button onClick={() => replay(a)} className="flex items-center gap-1 rounded border border-line-2 px-1.5 py-1 text-[11px]" aria-label="Replay"><History size={12} />Replay</button>
                        <button onClick={() => ack(a)} className="flex items-center gap-1 rounded border border-line-2 px-1.5 py-1 text-[11px]" aria-label={a.status === 'acknowledged' ? 'Undo acknowledge' : 'Acknowledge'}>
                          {a.status === 'acknowledged' ? <><Undo2 size={12} />Undo</> : <><Check size={12} />Ack</>}</button>
                      </div>
                    </td>
                  </tr>
                  {open === a.id && <tr className="bg-bg/60"><td colSpan={10} className="px-4 py-2 text-[11.5px] text-ink-2">
                    <b>Outcome (simulated ground truth):</b> {a.outcome ?? 'n/a'} · <b>closed by:</b> {a.closed_by ?? '—'} · <b>raw score:</b> {a.raw_score?.toFixed(3)} · <b>source:</b> {a.source}
                    {a.ack && <> · acknowledged by {a.ack.by} at {a.ack.at}</>}
                    <div className="mt-1">{(a.signals ?? []).map((s) => `${s.signal} ${pct(s.share)}`).join(' · ')}</div></td></tr>}
                </Fragment>))}</tbody>
            </table>
          </div>}
      </Panel>
    </div>
  )
}
