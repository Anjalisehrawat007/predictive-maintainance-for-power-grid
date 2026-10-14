import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle2, Circle, Pause, Play, RotateCcw, StepForward, XCircle } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { RiskChart, SensorChart, SensorLegend } from '../components/Charts'
import EventFeed from '../components/EventFeed'
import { Explain, IntelligenceCard } from '../components/Intelligence'
import Panel, { Empty } from '../components/Panel'
import { api } from '../lib/api'
import { dur, liveRecord, minutesBetween, riskPct, ts } from '../lib/format'
import { C, TYPE_COLOR } from '../lib/palette'
import StatusBadge from '../components/StatusBadge'
import type { EngineRecord, FeedEvent, Incident, SessionSummary } from '../lib/types'
import { useGrid } from '../store/grid'

const ALL = ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
const KEY_SENSORS: Record<string, string[]> = {
  thermal_runaway: ['OT', 'HUFL'], load_oscillation: ['HUFL', 'MUFL', 'OT'], intermittent_arcing: ['HULL', 'MULL'], sudden: ['OT', 'HUFL', 'HULL'],
}
const ANOMALY_KINDS = new Set(['thermal_deviation', 'thermal_residual', 'load_instability', 'reactive_spikes'])
const STAGES = [
  { key: 'normal', label: 'Normal', rule: 'replay start' },
  { key: 'anomaly', label: 'Subtle anomaly', rule: 'first anomaly event from the feature monitors' },
  { key: 'rising', label: 'Risk increasing', rule: 'risk trend rising and ≥ 25 % of the warning level' },
  { key: 'warning', label: 'Warning', rule: 'WARNING status (2 consecutive steps over threshold)' },
  { key: 'critical', label: 'Critical', rule: 'CRITICAL status' },
  { key: 'failure', label: 'Failure', rule: 'simulated ground-truth trip' },
] as const
type StageKey = typeof STAGES[number]['key']
const SPEEDS = [1, 2, 4, 8]

export default function IncidentReplay() {
  const system = useGrid((s) => s.system)
  const [params, setParams] = useSearchParams()
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [asset, setAsset] = useState(params.get('asset') ?? 'ETTm1')
  const [eventId, setEventId] = useState<number | null>(params.get('event') ? Number(params.get('event')) : null)
  const [hours, setHours] = useState(6)
  const [speed, setSpeed] = useState(2)
  const [session, setSession] = useState<SessionSummary | null>(null)
  const [recs, setRecs] = useState<EngineRecord[]>([])
  const [events, setEvents] = useState<FeedEvent[]>([])
  const [stages, setStages] = useState<Partial<Record<StageKey, string>>>({})
  const [playing, setPlaying] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState<string>()
  const [sensorPick, setSensorPick] = useState<{ type: string; list: string[] } | null>(null)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)
  const busy = useRef(false)
  const state = useRef({ session, stages, recs })
  useEffect(() => { state.current = { session, stages, recs } }, [session, stages, recs])

  useEffect(() => { api.incidents().then((x) => { setIncidents(x); if (eventId == null) { const d = x.find((i) => i.asset === asset && (i.engine_lead_time_min ?? 0) > 60); if (d) setEventId(d.event_id) } }).catch((e) => setError(e.message)) }, []) // eslint-disable-line react-hooks/exhaustive-deps
  const list = incidents.filter((i) => i.asset === asset)
  const inc = incidents.find((i) => i.asset === asset && i.event_id === eventId)
  const labels = useMemo(() => Object.fromEntries((system?.dataset.assets ?? []).map((a) => [a.id, a.label])), [system])
  // default channels follow the failure mechanism until the user picks their own
  const sensors = sensorPick && sensorPick.type === inc?.type ? sensorPick.list : KEY_SENSORS[inc?.type ?? ''] ?? ['OT', 'HUFL']
  const setSensors = (f: (s: string[]) => string[]) => setSensorPick({ type: inc?.type ?? '', list: f(sensors) })

  const stop = useCallback(() => { if (timer.current) clearInterval(timer.current); timer.current = null; setPlaying(false) }, [])
  useEffect(() => () => { stop() }, [stop])

  const advance = useCallback(async (n: number) => {
    const s = state.current.session
    if (!s || busy.current) return
    busy.current = true
    try {
      const r = await api.step(s.id, n)
      const newRecs = (r.records[s.assets[0]] ?? []).map(liveRecord)
      const st = { ...state.current.stages }
      const warnRisk = system?.thresholds.warning_risk ?? 0.1
      for (const ev of r.events) if (ANOMALY_KINDS.has(ev.kind) && !st.anomaly) st.anomaly = ev.ts
      for (const x of newRecs) {
        if (!st.rising && x.trend === 'rising' && (x.risk ?? 0) >= 0.25 * warnRisk) st.rising = x.timestamp
        if (!st.warning && (x.status === 'WARNING' || x.status === 'CRITICAL')) st.warning = x.timestamp
        if (!st.critical && x.status === 'CRITICAL') st.critical = x.timestamp
        if (!st.failure && x.ground_truth?.failure_event) st.failure = x.timestamp
      }
      setStages(st)
      setRecs((p) => [...p, ...newRecs])
      setEvents((p) => [...[...r.events].reverse(), ...p])
      setSession({ ...s, cursor: r.cursor })
      if (r.done) { stop(); setDone(true) }
    } catch (e) { setError((e as Error).message); stop() } finally { busy.current = false }
  }, [stop, system])

  const start = async () => {
    if (!inc) return
    stop(); setRecs([]); setEvents([]); setDone(false); setError(undefined)
    try {
      const s = await api.replay(inc.asset, inc.event_id, hours)
      setSession(s); state.current.session = s
      const first = { normal: s.start_timestamp }
      setStages(first); state.current.stages = first
      setParams({ asset: inc.asset, event: String(inc.event_id) }, { replace: true })
      setPlaying(true)
      timer.current = setInterval(() => advance(speed), 400)
      advance(speed)
    } catch (e) { setError((e as Error).message) }
  }
  const resume = () => { if (!session || done) return; setPlaying(true); timer.current = setInterval(() => advance(speed), 400) }
  useEffect(() => { if (playing) { if (timer.current) clearInterval(timer.current); timer.current = setInterval(() => advance(speed), 400) } }, [speed]) // eslint-disable-line react-hooks/exhaustive-deps

  const last = recs.at(-1)
  const failRec = recs.find((r) => r.ground_truth?.failure_event)
  const failureTime = inc?.failure_time
  const rel = (t?: string) => (t && failureTime ? minutesBetween(failureTime, t) : null)
  const markers = [
    ...(inc ? recs.filter((r) => r.index === inc.onset_idx && inc.type !== 'sudden').map((r) => ({ index: r.index, label: 'anomaly begins (GT)', color: TYPE_COLOR[inc.type], dash: '2 3' })) : []),
    ...(stages.warning ? recs.filter((r) => r.timestamp === stages.warning).map((r) => ({ index: r.index, label: 'model warning', color: C.warn })) : []),
    ...(failRec ? [{ index: failRec.index, label: 'FAILURE (sim)', color: C.crit }] : []),
  ]
  const progress = session ? (session.cursor - session.start) / (session.stop - session.start) : 0
  const curStage = [...STAGES].reverse().find((s) => stages[s.key])?.key ?? 'normal'
  const th = system?.thresholds

  return (
    <div className="space-y-3">
      <div>
        <div className="label">Incident reconstruction · signature feature</div>
        <h1 className="text-[20px] font-semibold tracking-tight">Failure replay through the real streaming engine</h1>
        <p className="mt-0.5 max-w-3xl text-[12.5px] text-ink-2">Pick a held-out failure and replay the historical sensor stream observation-by-observation through the same engine used in production (features → GRU+attention → calibration → alert policy). Nothing is pre-rendered; the lead time shown at the end is what the engine measured. Failures are <b>simulated</b> on top of real transformer data.</p>
      </div>
      {error && <div role="alert" className="rounded border border-crit/50 bg-crit/10 px-3 py-2 text-sm text-crit">{error}</div>}
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
        <Panel className="xl:col-span-3" kicker="1 · select" title="Historical failure" bodyClass="p-0">
          <div className="space-y-2 border-b border-line p-3">
            <div className="flex gap-1" role="radiogroup" aria-label="Transformer">
              {(system?.dataset.assets ?? []).map((a) => (
                <button key={a.id} role="radio" aria-checked={asset === a.id} onClick={() => { setAsset(a.id); setEventId(null) }}
                  className={`flex-1 rounded border px-2 py-1 text-[12px] ${asset === a.id ? 'border-ink-3 bg-surface-2' : 'border-line text-ink-3'}`}>Transformer {a.label}</button>))}
            </div>
            <div className="flex items-center gap-2 text-[11.5px]">
              <label htmlFor="hb" className="text-ink-3">Start</label>
              <select id="hb" value={hours} onChange={(e) => setHours(Number(e.target.value))} className="rounded border border-line bg-bg px-1.5 py-1">
                {[3, 6, 12].map((h) => <option key={h} value={h}>{h} h before failure</option>)}
              </select>
            </div>
          </div>
          <ul className="scroll-thin max-h-[430px] overflow-y-auto" aria-label="Failures in the held-out period">
            {list.map((i) => (
              <li key={i.event_id}>
                <button onClick={() => setEventId(i.event_id)} aria-current={i.event_id === eventId}
                  className={`w-full border-b border-line/70 px-3 py-2 text-left text-[12px] hover:bg-surface-2 ${i.event_id === eventId ? 'bg-surface-2' : ''}`}>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: TYPE_COLOR[i.type] }} />{i.type.replace('_', ' ')}</span>
                    <span className="num text-[10.5px] text-ink-3">{ts(i.failure_time)}</span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-[10.5px] text-ink-3">
                    <span title="Detectability class (A strong … D no precursor)">class {i.class ?? '—'}</span>
                    {i.engine_lead_time_min != null ? <span className="text-ok">✓ warned {dur(i.engine_lead_time_min)} ahead</span> : <span className="text-warn">✕ not warned</span>}
                  </div>
                </button>
              </li>))}
          </ul>
        </Panel>
        <div className="space-y-3 xl:col-span-9">
          <Panel kicker="2 · replay" title={inc ? `Transformer ${labels[inc.asset]} · ${inc.type.replace('_', ' ')} · ${ts(inc.failure_time)}` : 'Select a failure'}
            right={<div className="flex flex-wrap items-center gap-2">
              <button onClick={start} disabled={!inc} className="flex items-center gap-1.5 rounded border border-accent/70 bg-accent/10 px-3 py-1.5 text-[12.5px] font-semibold text-accent disabled:opacity-40" aria-label="Replay incident through the live engine"><Play size={14} />REPLAY INCIDENT</button>
              {playing ? <button onClick={stop} className="rounded border border-line-2 px-2 py-1.5 text-[12px]" aria-label="Pause replay"><Pause size={14} /></button>
                : <button onClick={resume} disabled={!session || done} className="rounded border border-line-2 px-2 py-1.5 text-[12px] disabled:opacity-40" aria-label="Resume replay"><Play size={14} /></button>}
              <button onClick={() => { stop(); advance(1) }} disabled={!session || done} className="rounded border border-line-2 px-2 py-1.5 disabled:opacity-40" aria-label="Step one observation"><StepForward size={14} /></button>
              <button onClick={start} disabled={!inc} className="rounded border border-line-2 px-2 py-1.5 disabled:opacity-40" aria-label="Restart replay"><RotateCcw size={14} /></button>
              <div className="flex overflow-hidden rounded border border-line-2" role="radiogroup" aria-label="Replay speed">
                {SPEEDS.map((s) => <button key={s} role="radio" aria-checked={speed === s} onClick={() => setSpeed(s)} className={`px-2 py-1 font-mono text-[11.5px] ${speed === s ? 'bg-ink text-bg' : 'text-ink-2'}`}>{s}×</button>)}
              </div>
            </div>}>
            {/* story stepper */}
            <ol className="mb-3 grid grid-cols-3 gap-1.5 md:grid-cols-6" aria-label="Incident progression">
              {STAGES.map((s) => {
                const t = stages[s.key]
                const isCur = s.key === curStage && !!session
                const missed = done && !t
                const r = rel(t)
                return (
                  <li key={s.key} title={s.rule} className={`rounded border px-2 py-1.5 ${isCur ? 'border-accent bg-accent/10' : t ? 'border-line-2' : 'border-line'}`}>
                    <div className="flex items-center gap-1.5 text-[11.5px] font-semibold">
                      {t ? <CheckCircle2 size={13} className={s.key === 'failure' ? 'text-crit' : s.key === 'warning' || s.key === 'critical' ? 'text-warn' : 'text-ok'} aria-hidden />
                        : missed ? <XCircle size={13} className="text-ink-3" aria-hidden /> : <Circle size={13} className="text-ink-3" aria-hidden />}
                      {s.label.toUpperCase()}
                    </div>
                    <div className="num text-[10.5px] text-ink-3">{t ? (r != null ? (r === 0 ? 'T = failure' : `T ${r < 0 ? '−' : '+'}${dur(Math.abs(r))}`) : ts(t)) : missed ? 'not reached' : 'pending'}</div>
                  </li>)
              })}
            </ol>
            {/* countdown ruler */}
            {session && inc && <div className="mb-2">
              <div className="relative h-6 rounded border border-line bg-bg" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Replay progress">
                <motion.div className="absolute inset-y-0 left-0 bg-accent/15" animate={{ width: `${progress * 100}%` }} transition={{ duration: 0.3 }} />
                {Array.from({ length: hours + 2 }, (_, k) => k).map((k) => {
                  const minutes = -hours * 60 + k * 60
                  const pos = (k * 4) / (session.stop - session.start)
                  if (pos > 1) return null
                  return <span key={k} className={`absolute top-1 font-mono text-[9.5px] text-ink-3 ${k === 0 ? 'pl-1' : '-translate-x-1/2'}`} style={{ left: `${pos * 100}%` }}>{minutes === 0 ? 'FAILURE' : minutes > 0 ? `+${minutes / 60}h` : `${minutes / 60}h`}</span>
                })}
                {(() => { const p = (hours * 4 - 2) / (session.stop - session.start); return <span className="absolute top-1 -translate-x-1/2 font-mono text-[9.5px] text-ink-3" style={{ left: `${p * 100}%` }}>−30m</span> })()}
              </div>
            </div>}
            {recs.length && th ? <>
              <RiskChart recs={recs} warn={th.warning_risk} crit={th.critical_risk} markers={markers} syncId="rp" height={220} showConf />
              <div className="mt-1 flex flex-wrap items-center justify-between gap-2 px-1">
                <div className="flex flex-wrap gap-3 text-[10.5px] text-ink-3">
                  <span className="flex items-center gap-1"><span className="h-[2px] w-4" style={{ background: C.s[0] }} />calibrated risk</span>
                  <span className="flex items-center gap-1"><span className="w-4 border-t border-dashed" style={{ borderColor: C.ink3 }} />raw score</span>
                  <span className="flex items-center gap-1"><span className="h-[2px] w-4" style={{ background: C.s[5] }} />model confidence</span>
                  <span>dotted marker = anomaly onset from the simulator (ground truth, not visible to the model)</span>
                </div>
                <SensorLegend sensors={ALL} active={sensors} onToggle={(c) => setSensors((s) => (s.includes(c) ? s.filter((x) => x !== c) : [...s, c]))} />
              </div>
              <SensorChart recs={recs} sensors={sensors} syncId="rp" markers={markers} height={150} />
            </> : <Empty>{inc ? 'Press REPLAY INCIDENT to stream the historical data through the engine.' : 'Choose a failure on the left.'}</Empty>}
          </Panel>
          <AnimatePresence>
            {done && inc && <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="panel grid gap-3 p-4 md:grid-cols-[1.3fr_1fr]" role="status" aria-live="polite">
              <div>
                <div className="label">Result · measured by the engine during this replay</div>
                {failRec?.lead_time_min != null
                  ? <><div className="mt-1 font-mono text-[12px] font-semibold tracking-[0.14em] text-ok">FIRST RELIABLE WARNING</div>
                    <div className="num text-[34px] font-semibold leading-tight text-ok">{dur(failRec.lead_time_min)} <span className="text-[15px] font-semibold tracking-[0.08em] text-ink-2">BEFORE FAILURE</span></div>
                    {failRec.lead_time_min > hours * 60 && <p className="text-[11.5px] text-ink-3">The warning was already active when this {hours} h replay window started (the engine is warmed up on the preceding data).</p>}</>
                  : <><div className="mt-1 font-mono text-[12px] font-semibold tracking-[0.14em] text-warn">OUTCOME</div>
                    <div className="text-[24px] font-semibold leading-tight text-warn">NO RELIABLE WARNING DETECTED</div>
                    <p className="text-[12px] text-ink-2">{inc.type === 'sudden' ? 'Sudden failure: the simulator injects no precursor, so no sensor-based model can warn (detectability class D).'
                      : stages.warning ? `A warning was raised at T −${dur(Math.abs(rel(stages.warning) ?? 0))} but had cleared more than 1 h before the failure.`
                      : 'Risk never stayed above the warning threshold for 2 consecutive steps before the failure.'}</p></>}
                <p className="mt-1 text-[12px] text-ink-3">A warning is "reliable" if it is still active (gaps ≤ 1 h) when the failure occurs — the same definition used in the offline evaluation.</p>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[12px]">
                <dt className="text-ink-3">Failure mechanism (sim)</dt><dd>{inc.type.replace('_', ' ')}</dd>
                <dt className="text-ink-3">Detectability class</dt><dd>{inc.class ?? '—'} <span className="text-ink-3">({inc.signal_metric}: {inc.signal_strength?.toFixed?.(1) ?? '—'})</span></dd>
                <dt className="text-ink-3">Anomaly onset (ground truth)</dt><dd>{inc.type === 'sudden' ? 'none — sudden failure' : `${dur(minutesBetween(inc.onset_time, inc.failure_time))} before failure`}</dd>
                <dt className="text-ink-3">Offline evaluation (deployed threshold)</dt><dd>{inc.lead_time_min != null ? dur(inc.lead_time_min) : 'not warned'}{(inc.lead_time_min ?? null) === (failRec?.lead_time_min ?? null) && <span className="ml-1 text-ok" title="Replay and offline evaluation agree">✓ matches</span>}</dd>
                <dt className="text-ink-3">Peak calibrated risk</dt><dd className="num">{riskPct(Math.max(...recs.map((r) => r.risk ?? 0)))}</dd>
              </dl>
            </motion.div>}
          </AnimatePresence>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            <Panel kicker="Engine state" title="Live intelligence" right={last && <StatusBadge status={last.status} />}><IntelligenceCard rec={last} label={labels[asset]} /></Panel>
            <Panel kicker="Why is risk changing?" title="Model-attributed signals"><Explain rec={last} /></Panel>
            <Panel kicker="Backend events" title="Event feed" bodyClass="p-0 h-[420px]"><EventFeed events={events} labels={labels} showIngestion={false} /></Panel>
          </div>
        </div>
      </div>
    </div>
  )
}
