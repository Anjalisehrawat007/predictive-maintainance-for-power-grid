/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useState } from 'react'
import { Legend, Line, LineChart } from 'recharts'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, LabelList } from 'recharts'
import Panel, { Empty } from '../components/Panel'
import { api } from '../lib/api'
import { C, tooltipStyle } from '../lib/palette'

const axis = { stroke: C.line2, tick: { fill: C.ink3, fontSize: 10.5, fontFamily: 'IBM Plex Mono' }, tickLine: false }
type Row = { name: string; family: 'deep' | 'quantized' | 'runtime' | 'baseline' | 'ensemble'; selected?: boolean } & Record<string, any>
const METRICS: { key: string; label: string; better: 'high' | 'low'; fmt: (v: number) => string }[] = [
  { key: 'pr_auc', label: 'PR-AUC (test)', better: 'high', fmt: (v) => v.toFixed(3) },
  { key: 'val_pr_auc', label: 'PR-AUC (val)', better: 'high', fmt: (v) => v.toFixed(3) },
  { key: 'roc_auc', label: 'ROC-AUC', better: 'high', fmt: (v) => v.toFixed(3) },
  { key: 'precision', label: 'Precision', better: 'high', fmt: (v) => v.toFixed(3) },
  { key: 'recall', label: 'Recall', better: 'high', fmt: (v) => v.toFixed(3) },
  { key: 'f1', label: 'F1', better: 'high', fmt: (v) => v.toFixed(3) },
  { key: 'event_recall', label: 'Failures warned (detectable)', better: 'high', fmt: (v) => `${(v * 100).toFixed(0)}%` },
  { key: 'fa_day', label: 'False alarms / day', better: 'low', fmt: (v) => v.toFixed(3) },
  { key: 'lead', label: 'Median lead time (min)', better: 'high', fmt: (v) => v.toFixed(0) },
  { key: 'latency', label: 'CPU latency (ms)', better: 'low', fmt: (v) => v.toFixed(3) },
  { key: 'params', label: 'Parameters', better: 'low', fmt: (v) => v.toLocaleString() },
  { key: 'size', label: 'Model size (KB)', better: 'low', fmt: (v) => v.toFixed(1) },
]
const TICKS: Record<string, number[]> = { latency: [0.03, 0.1, 0.3, 1, 3, 10, 30], params: [10000, 20000, 40000, 100000], size: [10, 30, 100, 300, 1000, 10000, 100000] }
const FAM_COLOR: Record<string, string> = { deep: C.s[0], quantized: C.s[5], runtime: C.accent, baseline: C.s[3], ensemble: C.s[4] }
const NAMES: Record<string, string> = { best_attention_model: 'GRU + attention', best_lstm: 'LSTM', best_gru: 'GRU', best_lite_model: 'Compact GRU (h=32)', ensemble: 'Seed ensemble ×3' }

export default function ModelLab() {
  const [d, setD] = useState<Record<string, any>>()
  const [err, setErr] = useState<string>()
  const [metric, setMetric] = useState('pr_auc')
  const [xKey, setXKey] = useState<'latency' | 'params' | 'size'>('latency')
  useEffect(() => { api.models().then(setD).catch((e) => setErr(e.message)) }, [])
  const rows: Row[] = useMemo(() => {
    if (!d) return []
    const cand = Object.fromEntries((d.phase1_candidates ?? []).map((c: any) => [c.file, c]))
    const out: Row[] = []
    for (const e of d.efficiency ?? []) {
      const base = e.model.replace('_int8', '')
      const c = cand[e.model] ?? {}
      const q = e.model.endsWith('_int8')
      const dep = e.model === 'best_attention_model'
      const r: Row = { name: `${NAMES[base] ?? base}${q ? ' · int8' : ''}${dep ? ' · Phase-1 OP' : ''}`, family: q ? 'quantized' : base === 'ensemble' ? 'ensemble' : 'deep', selected: false,
        pr_auc: e.test_pr_auc, val_pr_auc: e.val_pr_auc, roc_auc: c.roc_auc, precision: c.precision, recall: e.test_recall, f1: e.test_f1,
        event_recall: e.event_recall, fa_day: e.fa_per_day, lead: e.lead_median_min, latency: e.cpu_latency_ms_b1, params: e.params, size: e.size_kb }
      const f = d.final_test
      if (dep && f) out.push({ ...r, name: 'GRU + attention · deployed OP', selected: true, pr_auc: f.pr_auc, roc_auc: f.roc_auc, precision: f.precision,
        recall: f.recall, f1: f.f1, event_recall: f.event_recall_detectable, fa_day: f.false_alarms_per_day, lead: f.lead_time_median_min })
      out.push(r)
      if (!q && e.onnx_cpu_latency_ms_b1) out.push({ ...r, name: `${r.name} · ONNX Runtime`, family: 'runtime', selected: false, latency: e.onnx_cpu_latency_ms_b1 })
    }
    for (const b of d.baselines ?? []) {
      if (!['stationary', 'raw_engineered'].includes(b.features)) continue
      out.push({ name: `${b.model.replace('_', ' ')} (${b.features === 'stationary' ? 'stat.' : 'raw+eng'})`, family: 'baseline', pr_auc: b.pr_auc, val_pr_auc: b.val_pr_auc,
        roc_auc: b.roc_auc, precision: b.precision, recall: b.recall, f1: b.f1, event_recall: b.event_recall_detectable, fa_day: b.false_alarms_per_day,
        lead: b.lead_time_median_min, latency: b.inference_latency_ms, params: null, size: b.model_size_kb })
    }
    return out
  }, [d])
  if (err) return <Empty><span role="alert">Could not load data from the ML backend ({err}). Is the API running on port 8000?</span></Empty>
  if (!d) return <Empty>Loading model results…</Empty>
  const M = METRICS.find((m) => m.key === metric)!
  const bars = rows.filter((r) => r[metric] != null).sort((a, b) => (M.better === 'high' ? b[metric] - a[metric] : a[metric] - b[metric]))
  const isPerf = !['latency', 'params', 'size'].includes(metric)
  const scatter = rows.filter((r) => r[xKey] != null && r[isPerf ? metric : 'pr_auc'] != null)
  const p2 = (d.phase2_summary ?? []) as any[]
  return (
    <div className="space-y-3">
      <div><div className="label">Model intelligence lab</div><h1 className="text-[20px] font-semibold">Accuracy, operations and computation — side by side</h1>
        <p className="max-w-3xl text-[12.5px] text-ink-2">All values are measured: test metrics from the held-out period (never used for selection), latency from single-thread CPU benchmarks on this machine. ONNX rows reuse the same weights with a different runtime. OP = operating point (alarm threshold). Comparison rows use each model's Phase-1 OP (chosen on the original validation events); the <b>deployed OP</b> row is the same GRU + attention at the final operating point (chosen on fresh validation simulations), which trades precision for recall and lead time. Threshold-free metrics (PR-AUC, ROC-AUC) are identical for both.</p></div>
      <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Metric">
        {METRICS.map((m) => <button key={m.key} role="radio" aria-checked={metric === m.key} onClick={() => setMetric(m.key)}
          className={`rounded border px-2 py-1 text-[11.5px] ${metric === m.key ? 'border-ink-3 bg-surface-2 text-ink' : 'border-line text-ink-3 hover:text-ink'}`}>{m.label}</button>)}
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <Panel kicker={M.better === 'high' ? 'higher is better' : 'lower is better'} title={M.label}>
          <ResponsiveContainer width="100%" height={Math.max(260, bars.length * 25)}>
            <BarChart data={bars} layout="vertical" margin={{ left: 40, right: 40 }}>
              <CartesianGrid stroke={C.line} horizontal={false} />
              <XAxis type="number" {...axis} /><YAxis type="category" dataKey="name" {...axis} width={235} />
              <Tooltip {...tooltipStyle} formatter={(v) => [M.fmt(Number(v)), M.label]} />
              <Bar dataKey={metric} radius={[0, 3, 3, 0]}>
                {bars.map((r) => <Cell key={r.name} fill={FAM_COLOR[r.family]} stroke={r.selected ? C.ink : undefined} />)}
                <LabelList dataKey={metric} position="right" formatter={(v: unknown) => M.fmt(Number(v))} style={{ fill: C.ink2, fontSize: 10.5, fontFamily: 'IBM Plex Mono' }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 text-[10.5px] text-ink-3">{Object.entries(FAM_COLOR).map(([k, c]) => <span key={k} className="flex items-center gap-1"><span className="h-2 w-3 rounded" style={{ background: c }} />{k}</span>)}<span>outlined = deployed model</span></div>
        </Panel>
        <Panel kicker="Performance vs computation" title={`${isPerf ? M.label : 'PR-AUC (test)'} vs ${xKey}`}
          right={<div className="flex gap-1" role="radiogroup" aria-label="Computation axis">{(['latency', 'params', 'size'] as const).map((k) => (
            <button key={k} role="radio" aria-checked={xKey === k} onClick={() => setXKey(k)} className={`rounded border px-2 py-0.5 text-[11px] ${xKey === k ? 'border-ink-3 text-ink' : 'border-line text-ink-3'}`}>{k}</button>))}</div>}>
          <ResponsiveContainer width="100%" height={360}>
            <ScatterChart margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
              <CartesianGrid stroke={C.line} />
              <XAxis type="number" dataKey={xKey} scale="log" domain={['dataMin', 'dataMax']} {...axis} name={xKey} ticks={TICKS[xKey]} allowDataOverflow
                tickFormatter={(v) => (xKey === 'params' ? `${(v / 1000).toFixed(0)}k` : v < 1 ? String(Number(v.toPrecision(2))) : String(Math.round(v)))} label={{ value: `${xKey}${xKey === 'latency' ? ' (ms, log)' : xKey === 'size' ? ' (KB, log)' : ' (log)'}`, fill: C.ink3, fontSize: 10.5, position: 'insideBottom', offset: -5 }} />
              <YAxis type="number" dataKey={isPerf ? metric : 'pr_auc'} {...axis} domain={['auto', 'auto']} />
              <ZAxis range={[70, 70]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => {
                const r = payload?.[0]?.payload as Row | undefined
                if (!r) return null
                const ym = METRICS.find((m) => m.key === (isPerf ? metric : 'pr_auc'))!
                return <div className="rounded border border-line-2 bg-surface px-2 py-1.5 font-mono text-[11.5px]"><div className="text-ink">{r.name}</div>
                  <div className="text-ink-2">{ym.label}: {ym.fmt(Number(r[ym.key]))}</div><div className="text-ink-2">{xKey}: {Number(r[xKey]).toPrecision(3)}</div></div>
              }} />
              {Object.keys(FAM_COLOR).map((fam) => <Scatter key={fam} name={fam} data={scatter.filter((r) => r.family === fam)} fill={FAM_COLOR[fam]}
                stroke={C.bg} strokeWidth={1.5} />)}
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-[10.5px] text-ink-3">Hover a point for the model name. Tree baselines have no parameter count (not comparable); their latency includes scikit-learn call overhead.</p>
        </Panel>
      </div>
      <Panel kicker="Full comparison" title="All measured models" bodyClass="p-0">
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[1000px] text-[12px]">
            <thead className="text-left"><tr className="label"><th className="px-3 py-2 font-normal">Model</th>{METRICS.map((m) => <th key={m.key} className="px-2 py-2 text-right font-normal">{m.label}</th>)}</tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.name} className={`border-t border-line/60 ${r.selected ? 'bg-accent/5' : ''}`}><td className="px-3 py-1.5">{r.name}{r.selected && <span className="ml-1 text-[10px] text-accent">deployed</span>}</td>
                {METRICS.map((m) => <td key={m.key} className="num px-2 text-right text-ink-2">{r[m.key] == null ? '—' : m.fmt(Number(r[m.key]))}</td>)}</tr>))}</tbody>
          </table>
        </div>
      </Panel>
      <Panel kicker="Final model-selection protocol · 3 seeds × 5 fresh validation simulations" title="Candidate architectures (mean ± std [min, max])" bodyClass="p-0">
        {(d.protocol ?? []).length ? <div className="scroll-thin overflow-x-auto"><table className="w-full min-w-[1000px] text-[12px]">
          <thead className="text-left"><tr className="label">{['Candidate', 'PR-AUC (fresh val)', 'Recall', 'Precision', 'F1', 'FA/day', 'Lead (min)', 'Δ vs incumbent', 'Replaces?', 'Params', 'Latency p50 / p99'].map((h) => <th key={h} className="px-2 py-2 font-normal">{h}</th>)}</tr></thead>
          <tbody>{[...d.protocol].sort((a: any, b: any) => b.pr_auc_mean - a.pr_auc_mean).map((r: any) => (
            <tr key={r.candidate} className={`border-t border-line/60 ${r.candidate.includes('incumbent') ? 'bg-accent/5' : ''}`}>
              <td className="px-2 py-1.5">{r.candidate}</td>
              <td className="num px-2">{r.pr_auc_mean.toFixed(3)} ± {r.pr_auc_std.toFixed(3)} <span className="text-ink-3">[{r.pr_auc_min.toFixed(2)}, {r.pr_auc_max.toFixed(2)}]</span></td>
              {['recall', 'precision', 'f1', 'false_alarms_per_day'].map((k) => <td key={k} className="num px-2">{r[`${k}_mean`].toFixed(3)} ± {r[`${k}_std`].toFixed(3)}</td>)}
              <td className="num px-2">{r.lead_time_median_min_mean.toFixed(0)}</td>
              <td className="num px-2">{r.diff_vs_incumbent >= 0 ? '+' : ''}{r.diff_vs_incumbent.toFixed(3)}</td>
              <td className="px-2">{r.replaces ? 'yes' : 'no'}</td>
              <td className="num px-2">{r.params.toLocaleString()}</td><td className="num px-2">{r.p50_ms.toFixed(2)} / {r.p99_ms.toFixed(2)} ms</td></tr>))}</tbody></table>
          <p className="px-3 py-2 text-[10.5px] text-ink-3">A challenger must beat the incumbent by more than max(0.01, 1 SE). None does → GRU + attention is kept. Test data was never used for this decision.</p></div>
          : <Empty>Run <code className="font-mono">python -m src.final.protocol</code>.</Empty>}
      </Panel>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <Panel kicker="Robustness · held-out test period" title="Detection under degraded fault signatures">
          {d.robustness?.summary ? <RobustnessChart rows={d.robustness.summary} /> : <Empty>Run <code className="font-mono">python -m src.final.robustness</code>.</Empty>}
        </Panel>
        <Panel kicker="Model confidence ≠ physical certainty" title="Are high-confidence alarms more often right?">
          {d.confidence_bins?.test ? <ConfidenceBins bins={d.confidence_bins} /> : <Empty>Run the operating-point evaluation.</Empty>}
        </Panel>
      </div>
      <Panel kicker="Phase 2 · 3 seeds · multi-realisation validation" title="Architecture & feature experiments (mean ± std)" bodyClass="p-0">
        {p2.length ? <div className="scroll-thin overflow-x-auto"><table className="w-full min-w-[900px] text-[12px]">
          <thead className="text-left"><tr className="label">{['Group', 'Candidate', 'Seeds', 'Val PR-AUC (fresh events)', 'Val PR-AUC (original)', 'Test PR-AUC', 'Test PR-AUC (re-sim)', 'F1 (test)', 'FA/day (val, fresh)', 'Lead (min)', 'Params', 'Latency ms'].map((h) => <th key={h} className="px-2 py-2 font-normal">{h}</th>)}</tr></thead>
          <tbody>{p2.map((r) => (
            <tr key={`${r.group}${r.candidate}`} className="border-t border-line/60"><td className="px-2 py-1.5 text-ink-3">{r.group}</td><td className="px-2">{r.candidate}</td><td className="num px-2">{r.n}</td>
              {['val_rs_pr_auc', 'val_pr_auc', 'test_pr_auc', 'test_rs_pr_auc', 'test_f1', 'val_rs_false_alarms_per_day', 'test_lead_time_median_min'].map((k) => (
                <td key={k} className="num px-2">{r[k] == null ? '—' : `${Number(r[k]).toFixed(k.includes('lead') ? 0 : 3)}${r[`${k}_std`] != null ? ` ± ${Number(r[`${k}_std`]).toFixed(k.includes('lead') ? 0 : 3)}` : ''}`}</td>))}
              <td className="num px-2">{r.parameters?.toLocaleString?.() ?? '—'}</td><td className="num px-2">{r.latency_ms?.toFixed?.(2) ?? '—'}</td></tr>))}</tbody></table></div>
          : <Empty>Phase-2 experiment summary not generated yet (run <code className="font-mono">python -m src.phase2.report</code>).</Empty>}
      </Panel>
    </div>
  )
}

const PROFILES = ['NORMAL', 'MILD', 'MODERATE', 'SEVERE']
function RobustnessChart({ rows }: { rows: any[] }) {
  const [metric, setMetric] = useState<'pr_auc' | 'event_recall_detectable' | 'false_alarms_per_day' | 'lead_time_median_min'>('event_recall_detectable')
  const variants = [...new Set(rows.map((r) => r.variant))]
  const data = PROFILES.map((p) => {
    const o: Record<string, any> = { profile: p }
    for (const v of variants) {
      const r = rows.find((x) => x.variant === v && x.profile === p && x.split === 'test')
      o[v] = r ? r[`${metric}_mean`] ?? null : null
    }
    return o
  })
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-1" role="radiogroup" aria-label="Robustness metric">
        {([['event_recall_detectable', 'predictable failures warned'], ['pr_auc', 'PR-AUC'], ['lead_time_median_min', 'lead time (min)'], ['false_alarms_per_day', 'false alarms / day']] as const).map(([k, l]) => (
          <button key={k} role="radio" aria-checked={metric === k} onClick={() => setMetric(k)} className={`rounded border px-2 py-0.5 text-[11px] ${metric === k ? 'border-ink-3 text-ink' : 'border-line text-ink-3'}`}>{l}</button>))}
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ left: -10, right: 16, top: 6 }}>
          <CartesianGrid stroke={C.line} /><XAxis dataKey="profile" {...axis} /><YAxis {...axis} domain={['auto', 'auto']} />
          <Tooltip {...tooltipStyle} formatter={(v) => Number(v).toFixed(3)} /><Legend wrapperStyle={{ fontSize: 11 }} />
          {variants.map((v, i) => <Line key={v} dataKey={v} stroke={C.s[i]} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
      <p className="text-[10.5px] text-ink-3">Mean over 5 re-simulations × 3 seeds per stress level (weaker, delayed, intermittent, irregular, asset-varied and overlapping signatures — see configs/failure_simulation.yaml).</p>
    </div>
  )
}

function ConfidenceBins({ bins }: { bins: any }) {
  const rows = (bins.test as any[])
  const agg = ['HIGH', 'MEDIUM', 'LOW'].map((b) => {
    const a = rows.reduce((s, r) => s + (r[`${b}_alarms`] ?? 0), 0)
    const hits = rows.reduce((s, r) => s + (r[`${b}_alarms`] ?? 0) * (r[`${b}_precision`] ?? 0), 0)
    return { bin: b, alarms: a, precision: a ? hits / a : null, val: bins.val?.[b]?.precision }
  })
  return (
    <div>
      <table className="w-full text-[12.5px]"><thead className="text-left"><tr className="label"><th className="py-1.5 font-normal">Confidence bin</th><th className="font-normal">Alarm-steps (test, 6 sets)</th><th className="font-normal">Precision · test</th><th className="font-normal">Precision · fresh val</th></tr></thead>
        <tbody>{agg.map((r) => <tr key={r.bin} className="border-t border-line/60"><td className="py-1.5 font-mono">{r.bin} <span className="text-ink-3">{r.bin === 'HIGH' ? '≥ 0.9' : r.bin === 'MEDIUM' ? '0.7–0.9' : '< 0.7'}</span></td>
          <td className="num">{r.alarms.toLocaleString()}</td><td className="num">{r.precision == null ? '—' : `${(r.precision * 100).toFixed(0)}%`}</td><td className="num">{r.val == null ? '—' : `${(r.val * 100).toFixed(0)}%`}</td></tr>)}</tbody></table>
      <p className="mt-2 text-[10.5px] text-ink-3">Confidence = agreement of three independently seeded models. HIGH-confidence alarms are clearly more often correct; MEDIUM and LOW are similar, so the useful distinction is HIGH vs not-HIGH. This is model confidence, not certainty about the physical asset.</p>
    </div>
  )
}
