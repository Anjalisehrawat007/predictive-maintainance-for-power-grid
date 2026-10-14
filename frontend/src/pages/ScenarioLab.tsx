/* eslint-disable @typescript-eslint/no-explicit-any */
import { Dice5, Info } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import Panel from '../components/Panel'
import SessionPlayer from '../components/SessionPlayer'
import { api } from '../lib/api'
import { PUBLIC } from '../lib/deploy'
import { TYPE_COLOR } from '../lib/palette'
import type { Fault } from '../lib/types'
import { useGrid } from '../store/grid'

const FAULTS: { id: Fault; label: string; what: string }[] = [
  { id: 'thermal_runaway', label: 'Thermal runaway', what: 'Cooling degradation: oil temperature rises above what the load explains.' },
  { id: 'load_oscillation', label: 'Load oscillation', what: 'Growing 1–3 h oscillation on HV/MV load with a lagged thermal response.' },
  { id: 'intermittent_arcing', label: 'Intermittent arcing', what: 'Reactive-load spike bursts whose rate rises towards the trip.' },
  { id: 'sudden', label: 'Sudden trip', what: 'No precursor at all — shows the limit of sensor-based prediction.' },
  { id: 'none', label: 'No fault', what: 'Normal operation only — any alarm is a false alarm.' },
]
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export default function ScenarioLab() {
  const system = useGrid((s) => s.system)
  const [asset, setAsset] = useState('ETTm1')
  const [month, setMonth] = useState(4)
  const [fault, setFault] = useState<Fault>('thermal_runaway')
  const [severity, setSeverity] = useState(1)
  const [precursor, setPrecursor] = useState(8)
  const [seed, setSeed] = useState(0)
  const [val, setVal] = useState<Record<string, any>>()
  useEffect(() => { api.scenarioValidation().then(setVal).catch(() => {}) }, [])
  const assets = system?.dataset.assets ?? [{ id: 'ETTm1', label: 'T1' }, { id: 'ETTm2', label: 'T2' }]
  const f = FAULTS.find((x) => x.id === fault)!
  const hasPrecursor = fault !== 'none' && fault !== 'sudden'
  const det = useMemo(() => (val?.detection ?? []).filter((d: any) => d.asset === asset && d.fault === fault), [val, asset, fault])
  const fa = useMemo(() => {
    const rows = (val?.false_alarms ?? []).filter((d: any) => d.asset === asset)
    return rows.length ? rows.reduce((a: number, d: any) => a + d.false_alarms_per_day, 0) / rows.length : null
  }, [val, asset])
  const label = assets.find((a) => a.id === asset)?.label ?? asset
  const req = { asset, month, fault, severity, precursor_hours: precursor, seed }

  return (
    <div className="space-y-3">
      <div>
        <div className="label">Scenario Lab · interactive</div>
        <h1 className="text-[20px] font-semibold tracking-tight">Inject a fault and watch the model react</h1>
        <p className="mt-0.5 max-w-3xl text-[12.5px] text-ink-2">Build a scenario on a <b>synthetic</b> transformer stream (generated from aggregate statistics of normal operation — no dataset rows), inject a failure precursor with the same simulator the model was evaluated on, and stream it through the real engine: GRU + attention, calibration, model confidence, attribution, alert policy. Every number comes from the backend as it runs.</p>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
        <div className="space-y-3 xl:col-span-3">
          <Panel kicker="1 · configure" title="Scenario">
            <div className="space-y-3 text-[12.5px]">
              <fieldset>
                <legend className="label mb-1">Transformer profile</legend>
                <div className="flex gap-1" role="radiogroup" aria-label="Transformer">
                  {assets.map((a) => <button key={a.id} role="radio" aria-checked={asset === a.id} onClick={() => setAsset(a.id)}
                    className={`flex-1 rounded border px-2 py-1 ${asset === a.id ? 'border-ink-3 bg-surface-2' : 'border-line text-ink-3'}`}>Transformer {a.label}</button>)}
                </div>
              </fieldset>
              <label className="block"><span className="label">Season</span>
                <select value={month} onChange={(e) => setMonth(Number(e.target.value))} className="mt-1 w-full rounded border border-line bg-bg px-2 py-1">
                  {MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
                </select></label>
              <fieldset>
                <legend className="label mb-1">Fault to inject</legend>
                <div className="space-y-1" role="radiogroup" aria-label="Fault type">
                  {FAULTS.map((x) => <button key={x.id} role="radio" aria-checked={fault === x.id} onClick={() => setFault(x.id)}
                    className={`flex w-full items-center gap-2 rounded border px-2 py-1.5 text-left ${fault === x.id ? 'border-ink-3 bg-surface-2' : 'border-line text-ink-2'}`}>
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: TYPE_COLOR[x.id] ?? 'var(--color-ok)' }} />{x.label}</button>)}
                </div>
                <p className="mt-1 text-[11px] text-ink-3">{f.what}</p>
              </fieldset>
              <label className={`block ${hasPrecursor ? '' : 'opacity-40'}`}><span className="label">Severity · {severity.toFixed(2)}×</span>
                <input type="range" min={0.25} max={2} step={0.25} value={severity} disabled={!hasPrecursor} onChange={(e) => setSeverity(Number(e.target.value))} className="w-full accent-[var(--color-accent)]" aria-label="Severity" />
                <span className="text-[11px] text-ink-3">1× = typical benchmark failure (mid-range magnitude)</span></label>
              <label className={`block ${hasPrecursor ? '' : 'opacity-40'}`}><span className="label">Precursor duration · {precursor} h</span>
                <input type="range" min={3} max={18} step={1} value={precursor} disabled={!hasPrecursor} onChange={(e) => setPrecursor(Number(e.target.value))} className="w-full accent-[var(--color-accent)]" aria-label="Precursor duration in hours" /></label>
              <div className="flex items-center justify-between">
                <span className="label">Baseline seed · <span className="num">{seed}</span></span>
                <button onClick={() => setSeed(Math.floor(Math.random() * 100000))} className="flex items-center gap-1 rounded border border-line-2 px-2 py-0.5 text-[11.5px]"><Dice5 size={13} />new baseline</button>
              </div>
            </div>
          </Panel>
          <Panel kicker="How much to trust this" title="Measured on synthetic data">
            <div className="space-y-2 text-[12px]">
              {fa != null && <p>Fault-free synthetic runs, Transformer {label}: <b className="num">{fa.toFixed(2)}</b> false alarms/day (real held-out data: 0.25/day).</p>}
              {det.length > 0 && <table className="w-full text-[11.5px]">
                <thead><tr className="text-left text-ink-3"><th className="font-normal">severity</th><th className="font-normal">warned in time</th><th className="font-normal">median lead</th></tr></thead>
                <tbody>{det.map((d: any) => <tr key={d.severity} className="border-t border-line/60"><td className="num">{d.severity}×</td><td className="num">{Math.round(d.detection_rate * 100)}%</td><td className="num">{d.lead_median_min == null ? '—' : `${Math.round(d.lead_median_min)} min`}</td></tr>)}</tbody>
              </table>}
              <p className="flex gap-1.5 text-[11px] text-ink-3"><Info size={13} className="mt-0.5 shrink-0" />{det.length ? `${det[0].n} runs per row, 8 h precursor (src/scenario/validate.py). ` : ''}The model was trained and evaluated on real transformer data with simulated faults; synthetic baselines are an approximation, so treat these runs as illustrative, not as the benchmark.</p>
            </div>
          </Panel>
        </div>
        <div className="xl:col-span-9">
          <SessionPlayer kind="scenario" ready title={`Transformer ${label} · ${f.label}${hasPrecursor ? ` · ${severity}× · ${precursor} h precursor` : ''} · ${MONTHS[month - 1]}`}
            {...(PUBLIC ? { run: () => api.runScenario(req) } : { create: () => api.scenario(req) })} />
        </div>
      </div>
    </div>
  )
}
