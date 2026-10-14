/* eslint-disable @typescript-eslint/no-explicit-any */
import { History, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Link, useNavigate } from 'react-router-dom'
import Panel, { Empty } from '../components/Panel'
import { api } from '../lib/api'
import { dur, pct, riskPct, ts } from '../lib/format'
import { C, TYPE_COLOR, tooltipStyle } from '../lib/palette'
import { useGrid } from '../store/grid'

const axis = { stroke: C.line2, tick: { fill: C.ink3, fontSize: 10.5, fontFamily: 'IBM Plex Mono' }, tickLine: false }
const TYPES = ['thermal_runaway', 'load_oscillation', 'intermittent_arcing', 'sudden']
const CLASS_DESC: Record<string, string> = { A: 'strong precursor', B: 'moderate', C: 'weak', D: 'no precursor (sudden)' }

export default function FailureIntel() {
  const [d, setD] = useState<Record<string, any>>()
  const [err, setErr] = useState<string>()
  const [faFilter, setFaFilter] = useState('all')
  const nav = useNavigate()
  const g = useGrid()
  useEffect(() => { api.failures().then(setD).catch((e) => setErr(e.message)) }, [])
  const tf: any[] = useMemo(() => d?.test_failures ?? [], [d])
  const byClass = useMemo(() => ['A', 'B', 'C', 'D'].map((c) => {
    const x = tf.filter((f) => f.class === c)
    return { cls: `${c} · ${CLASS_DESC[c]}`, warned: x.filter((f) => f.engine_lead_time_min != null).length, missed: x.filter((f) => f.engine_lead_time_min == null).length }
  }), [tf])
  const leadHist = useMemo(() => {
    const bins: Record<string, number | string>[] = Array.from({ length: 10 }, (_, h) => ({ bin: `${h}–${h + 1} h`, ...Object.fromEntries(TYPES.map((t) => [t, 0])) }))
    tf.filter((f) => f.engine_lead_time_min != null).forEach((f) => { const h = Math.min(9, Math.floor(f.engine_lead_time_min / 60)); bins[h][f.type] = Number(bins[h][f.type]) + 1 })
    return bins
  }, [tf])
  const episodes: any[] = useMemo(() => d?.false_alarm_episodes ?? [], [d])
  const faCat = useMemo(() => {
    const m: Record<string, number> = {}
    for (const f of episodes) m[f.category] = (m[f.category] ?? 0) + 1
    return Object.entries(m).map(([cat, n]) => ({ cat, n })).sort((a, b) => b.n - a.n)
  }, [episodes])
  const monthly = useMemo(() => (d?.monthly ?? []).map((m: any) => ({ ...m, key: `${m.month} ${m.asset === 'ETTm1' ? 'T1' : 'T2'}` })), [d])
  const perAsset = useMemo(() => ['ETTm1', 'ETTm2'].map((a) => {
    const x = tf.filter((f) => f.asset === a), det = x.filter((f) => f.class !== 'D')
    const fa = episodes.filter((f: any) => f.asset === a).length
    return { a: a === 'ETTm1' ? 'T1' : 'T2', failures: x.length, detectable: det.length, warned: x.filter((f) => f.engine_lead_time_min != null).length, fa }
  }), [tf, episodes])
  if (err) return <Empty><span role="alert">Could not load data from the ML backend ({err}). Is the API running on port 8000?</span></Empty>
  if (!d) return <Empty>Loading failure analytics…</Empty>
  const fin = d.final?.test ?? {}
  const op = d.operating_point?.new
  const fb = d.final?.false_alarms
  const missed = tf.filter((f) => f.engine_lead_time_min == null)
  const rate = (c: { warned: number; missed: number }) => c.warned / ((c.warned + c.missed) || 1)
  const [A, B, Cc] = byClass
  const monotonic = rate(A) >= rate(B) && rate(B) >= rate(Cc)
  const testFA = episodes.length
  const meaningfulShare = episodes.filter((e) => e.category !== 'model error').length / (testFA || 1)
  const shownFA = episodes.filter((e) => faFilter === 'all' || e.category === faFilter)
  const inspect = async (e: any) => { g.select(e.asset); await g.seek(e.index + 4); nav('/') }
  return (
    <div className="space-y-3">
      <div><div className="label">Failure intelligence</div><h1 className="text-[20px] font-semibold">What failed, what was predictable, and how early</h1>
        <p className="max-w-3xl text-[12.5px] text-ink-2">Held-out test period (2018-03-10 → 06-26), both transformers, deployed model at the final operating point{op ? ` (warning threshold ${op.warning.toFixed(3)}, selected on fresh validation simulations)` : ''}. Failures are <b>simulated</b>; detectability classes measure how much precursor signal the real sensors carried. Recall is always shown both over <b>all</b> failures and over <b>predictable</b> failures — sudden failures have no precursor in these sensors.</p></div>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-4 xl:grid-cols-7">
        {[['Test failures', fin.n_failures ?? tf.length], ['Predictable (A–C)', fin.n_detectable ?? '—'], ['Unpredictable (sudden)', (fin.n_failures ?? 0) - (fin.n_detectable ?? 0)],
          ['Recall · all failures', pct(fin.recall_all_failures, 1)], ['Recall · predictable', pct(fin.recall_predictable_failures, 1)],
          ['Median lead time', dur(fin.lead_time_median_min)], ['False alarms / transformer-day', fin.false_alarms_per_day?.toFixed(3) ?? '—']].map(([k, v]) => (
          <div key={String(k)} className="bg-surface px-3 py-2.5"><div className="label">{k}</div><div className="num text-[22px] font-semibold">{v}</div></div>))}
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <Panel kicker="Detectability" title="Warned vs missed by precursor strength">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={byClass} layout="vertical" margin={{ left: 60, right: 10 }}>
              <CartesianGrid stroke={C.line} horizontal={false} />
              <XAxis type="number" {...axis} allowDecimals={false} /><YAxis type="category" dataKey="cls" {...axis} width={140} />
              <Tooltip {...tooltipStyle} /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="warned" stackId="a" fill={C.ok} name="warned in time" /><Bar dataKey="missed" stackId="a" fill={C.crit} name="missed" />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-[10.5px] text-ink-3">{monotonic ? 'Recall increases with precursor strength.' : 'Recall is not monotonic in precursor strength — some strong-signal failures were missed, i.e. model/threshold errors rather than sensor limits.'}</p>
        </Panel>
        <Panel kicker="Engine-measured" title="Lead-time distribution (warned failures)">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={leadHist} margin={{ left: -16, right: 8 }}>
              <CartesianGrid stroke={C.line} vertical={false} />
              <XAxis dataKey="bin" {...axis} interval={0} angle={-30} textAnchor="end" height={40} /><YAxis {...axis} allowDecimals={false} />
              <Tooltip {...tooltipStyle} />
              {TYPES.slice(0, 3).map((t) => <Bar key={t} dataKey={t} stackId="a" fill={TYPE_COLOR[t]} name={t.replace('_', ' ')} />)}
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel kicker="All splits (263 simulated failures)" title="Failure mechanisms">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={TYPES.map((t) => ({ t: t.replace('_', ' '), n: d.failure_types_all?.[t] ?? 0, k: t }))} margin={{ left: -16, right: 8 }}>
              <CartesianGrid stroke={C.line} vertical={false} /><XAxis dataKey="t" {...axis} /><YAxis {...axis} />
              <Tooltip {...tooltipStyle} /><Bar dataKey="n" name="failures">{TYPES.map((t) => <Cell key={t} fill={TYPE_COLOR[t]} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <Panel kicker="False-alarm anatomy · final operating point" title={`${testFA} false-alarm episodes (test)`} bodyClass="p-0">
          <table className="w-full text-[12px]">
            <thead className="text-left"><tr className="label"><th className="px-3 py-2 font-normal">Category</th><th className="px-3 text-right font-normal">Episodes</th><th className="px-3 text-right font-normal">Share</th><th className="px-3 text-right font-normal">Base rate*</th></tr></thead>
            <tbody>{faCat.map((r) => (<tr key={r.cat} className="border-t border-line/60"><td className="px-3 py-1.5">{r.cat}</td><td className="num px-3 text-right">{r.n}</td>
              <td className="num px-3 text-right">{pct(r.n / (testFA || 1))}</td><td className="num px-3 text-right text-ink-3">{fb?.base_rates_event_free_windows?.[r.cat] != null ? pct(fb.base_rates_event_free_windows[r.cat]) : '—'}</td></tr>))}</tbody>
          </table>
          <p className="px-3 py-2 text-[10.5px] text-ink-3">{pct(meaningfulShare)} coincide with an identifiable anomaly (harmless injected disturbance, sensor noise, or unusual real operating data); the rest are model errors. *Share of random event-free windows meeting the same criterion — a high base rate means the category is common anyway.</p>
        </Panel>
        <Panel kicker="Failure cases" title={`Missed failures (${missed.length})`} bodyClass="p-0">
          <div className="scroll-thin max-h-[300px] overflow-y-auto">
            <table className="w-full text-[12px]">
              <thead className="sticky top-0 bg-surface text-left"><tr className="label">{['Asset', 'When', 'Type', 'Class', 'Signal', ''].map((h) => <th key={h} className="px-3 py-2 font-normal">{h}</th>)}</tr></thead>
              <tbody>{missed.map((f) => (
                <tr key={`${f.asset}${f.event_id}`} className="border-t border-line/60"><td className="px-3 py-1.5 font-mono">{f.label}</td><td className="num px-3">{ts(f.failure_time)}</td>
                  <td className="px-3">{f.type.replace('_', ' ')}</td><td className="num px-3">{f.class}</td><td className="num px-3 text-ink-2">{f.signal_strength == null ? '—' : Number(f.signal_strength).toFixed(1)}</td>
                  <td className="px-3"><Link className="flex items-center gap-1 text-accent" to={`/replay?asset=${f.asset}&event=${f.event_id}`}><History size={12} />replay</Link></td></tr>))}</tbody>
            </table>
          </div>
        </Panel>
      </div>
      <Panel kicker="False-alarm explorer" title="Inspect every false alarm of the deployed system"
        right={<label className="flex items-center gap-1.5 text-[11.5px] text-ink-3">category
          <select value={faFilter} onChange={(e) => setFaFilter(e.target.value)} className="rounded border border-line bg-bg px-1.5 py-1 text-ink">
            <option value="all">all ({episodes.length})</option>{faCat.map((c) => <option key={c.cat} value={c.cat}>{c.cat} ({c.n})</option>)}</select></label>} bodyClass="p-0">
        {shownFA.length ? <div className="scroll-thin max-h-[340px] overflow-y-auto">
          <table className="w-full text-[12px]">
            <thead className="sticky top-0 bg-surface text-left"><tr className="label">{['Asset', 'Start', 'Duration', 'Category', 'Injected event (sim)', 'Peak risk', 'Confidence', 'Real |z| max', ''].map((h) => <th key={h} className="px-3 py-2 font-normal">{h}</th>)}</tr></thead>
            <tbody>{shownFA.map((e) => (
              <tr key={`${e.asset}${e.start}`} className="border-t border-line/60">
                <td className="px-3 py-1.5 font-mono">{e.asset === 'ETTm1' ? 'T1' : 'T2'}</td><td className="num px-3">{ts(e.start)}</td><td className="num px-3">{dur(e.duration_h * 60)}</td>
                <td className="px-3">{e.category}</td><td className="px-3 text-ink-2">{e.injected_event ?? '—'}</td><td className="num px-3">{riskPct(e.peak_risk)}</td>
                <td className="num px-3">{pct(e.confidence_at_peak)}</td><td className="num px-3 text-ink-2">{e.max_real_abs_zscore?.toFixed(1)}</td>
                <td className="px-3"><button onClick={() => inspect(e)} className="flex items-center gap-1 text-accent" aria-label={`Inspect false alarm at ${ts(e.start)}`}><Search size={12} />inspect</button></td>
              </tr>))}</tbody>
          </table></div> : <Empty>No false alarms in this category.</Empty>}
      </Panel>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <Panel className="xl:col-span-2" kicker="Seasonal performance" title="Monthly PR-AUC and recall per transformer (test)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={monthly} margin={{ left: -16, right: 8 }}>
              <CartesianGrid stroke={C.line} vertical={false} /><XAxis dataKey="key" {...axis} /><YAxis {...axis} domain={[0, 1]} />
              <Tooltip {...tooltipStyle} formatter={(v) => Number(v).toFixed(3)} /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="pr_auc" fill={C.s[0]} name="PR-AUC" /><Bar dataKey="recall" fill={C.s[2]} name="recall @ threshold" />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel kicker="Asset performance" title="Per transformer (test)" bodyClass="p-0">
          <table className="w-full text-[12.5px]"><thead className="text-left"><tr className="label">{['Asset', 'Failures', 'With precursor', 'Warned', 'False alarms'].map((h) => <th key={h} className="px-3 py-2 font-normal">{h}</th>)}</tr></thead>
            <tbody>{perAsset.map((r) => <tr key={r.a} className="border-t border-line/60"><td className="px-3 py-2 font-mono">{r.a}</td><td className="num px-3">{r.failures}</td><td className="num px-3">{r.detectable}</td><td className="num px-3">{r.warned} ({pct(r.warned / (r.detectable || 1))})</td><td className="num px-3">{r.fa}</td></tr>)}</tbody></table>
        </Panel>
      </div>
    </div>
  )
}
