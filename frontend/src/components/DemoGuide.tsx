import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { dur, modelName, pct, ts } from '../lib/format'
import type { Incident } from '../lib/types'
import { pickDemoIncident } from '../lib/demo'
import { useGrid } from '../store/grid'

/** Guided recruiter demo. Every step drives the real UI and real engine; nothing here is scripted output. */
const scrollTo = (id: string) => setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 250)

export default function DemoGuide() {
  const g = useGrid()
  const nav = useNavigate()
  const [step, setStep] = useState(0)
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [test, setTest] = useState<Record<string, number>>()
  useEffect(() => {
    if (!g.demo) return
    api.incidents().then(setIncidents).catch(() => {})
    api.failures().then((f) => setTest(f.final?.test)).catch(() => {})
  }, [g.demo])
  const inc = useMemo(() => pickDemoIncident(incidents), [incidents])
  const label = (a?: string) => g.system?.dataset.assets.find((x) => x.id === a)?.label ?? a
  const m = g.system?.model
  const bench = g.system?.benchmark ?? {}

  if (!g.demo) return null
  const steps = [
    {
      title: 'System online',
      body: m ? `Backend ready: ${modelName(m.type)} (${m.parameters.toLocaleString()} parameters), ${m.n_features} causal features over a ${m.window_hours} h window, ${m.onnx_runtime ? 'ONNX Runtime' : 'PyTorch'} inference, ${m.calibration} calibration. Sensor data are real; failure events are simulated.` : 'Waiting for the backend…',
      run: () => nav('/'),
    },
    {
      title: 'Choose an example transformer',
      body: inc ? `Transformer ${label(inc.asset)}: a held-out ${inc.type.replace('_', ' ')} failure (simulated) at ${ts(inc.failure_time)}, detectability class ${inc.class}. Picked automatically as the typical warned failure.` : 'Loading held-out failures…',
      run: () => { if (inc) g.select(inc.asset); nav('/') },
    },
    {
      title: 'Start the live simulation',
      body: 'Streams the real held-out sensor data, starting 4 h before the first warning, through the live engine at 1× (2 observations per second = 30 min of grid time per second). Each tick is a POST to the backend; nothing is pre-rendered.',
      run: async () => {
        if (!inc) return
        g.select(inc.asset); nav('/')
        const lead = (inc.engine_lead_time_min ?? 0) / 15
        await g.seek(inc.failure_idx - lead - 16)
        g.setSpeed(1); await g.play(); scrollTo('cc-risk')
      },
    },
    {
      title: 'Watch the risk rise',
      body: 'Calibrated risk (solid) and raw model score (dashed). The status turns WARNING after 2 consecutive steps over the threshold chosen on validation data; the dotted marker is the simulator\'s ground-truth anomaly onset, which the model never sees.',
      run: () => { nav('/'); scrollTo('cc-risk') },
    },
    {
      title: 'Open the explanation',
      body: 'Integrated-gradients attribution grouped by sensor, and where in the window the evidence sits. This is what the model relied on: associational, not a physical cause.',
      run: () => { nav('/'); scrollTo('cc-explain') },
    },
    {
      title: 'Incident Replay',
      body: inc ? `Replay the same failure from 6 h before. Press REPLAY INCIDENT: the result card shows the engine-measured first reliable warning (offline evaluation for this failure: ${dur(inc.engine_lead_time_min)}).` : '',
      run: () => { g.pause(); if (inc) nav(`/replay?asset=${inc.asset}&event=${inc.event_id}`) },
    },
    {
      title: 'Model performance',
      body: test ? `Held-out test: PR-AUC ${test.pr_auc.toFixed(3)}, ROC-AUC ${test.roc_auc.toFixed(3)}, ${pct(test.recall_predictable_failures, 1)} of predictable failures warned, median lead ${dur(test.lead_time_median_min)}, ${test.false_alarms_per_day.toFixed(2)} false alarms per transformer-day. The protocol table shows why GRU + attention was kept.` : 'Loading metrics…',
      run: () => { g.pause(); nav('/models') },
    },
    {
      title: 'Latency',
      body: `Measured per-stage latency of the streaming engine. Benchmark p50: full engine step ${Number(bench.engine_step_ms_p50 ?? NaN).toFixed(1)} ms, ONNX inference ${Number(bench.onnx_ms_p50 ?? NaN).toFixed(3)} ms vs PyTorch ${Number(bench.pytorch_ms_p50 ?? NaN).toFixed(2)} ms (${bench.machine ?? 'this machine'}, 1 thread).`,
      run: () => { nav('/'); scrollTo('cc-pipeline') },
    },
  ]
  const s = steps[step]
  const go = (k: number) => { setStep(k); steps[k].run() }
  return (
    <aside aria-label="Demo mode" className="fixed bottom-4 right-4 z-40 w-[min(380px,calc(100vw-32px))] rounded-lg border border-accent/60 bg-surface shadow-2xl">
      <div className="flex items-center gap-2 border-b border-line px-3 py-2">
        <span className="font-mono text-[11px] font-semibold tracking-[0.14em] text-accent">DEMO MODE</span>
        <span className="font-mono text-[11px] text-ink-3">{step + 1} / {steps.length}</span>
        <button onClick={() => { g.pause(); g.setDemo(false) }} className="ml-auto text-ink-3 hover:text-ink" aria-label="Close demo mode"><X size={15} /></button>
      </div>
      <ol className="flex gap-1 px-3 pt-2" aria-hidden>{steps.map((_, k) => <li key={k} className={`h-1 flex-1 rounded ${k <= step ? 'bg-accent' : 'bg-line'}`} />)}</ol>
      <div className="px-3 py-2.5" aria-live="polite">
        <h2 className="text-[14px] font-semibold">{s.title}</h2>
        <p className="mt-1 text-[12.5px] leading-snug text-ink-2">{s.body}</p>
      </div>
      <div className="flex items-center gap-2 border-t border-line px-3 py-2">
        <button onClick={() => go(Math.max(0, step - 1))} disabled={step === 0} className="flex items-center gap-1 rounded border border-line-2 px-2 py-1 text-[12px] disabled:opacity-40"><ChevronLeft size={13} />Back</button>
        <button onClick={() => s.run()} className="rounded border border-line-2 px-2 py-1 text-[12px]">Show me</button>
        {step < steps.length - 1
          ? <button onClick={() => go(step + 1)} className="ml-auto flex items-center gap-1 rounded border border-accent/70 bg-accent/10 px-2.5 py-1 text-[12px] font-semibold text-accent">Next<ChevronRight size={13} /></button>
          : <button onClick={() => { g.pause(); g.setDemo(false) }} className="ml-auto rounded border border-accent/70 bg-accent/10 px-2.5 py-1 text-[12px] font-semibold text-accent">Finish</button>}
      </div>
    </aside>
  )
}
