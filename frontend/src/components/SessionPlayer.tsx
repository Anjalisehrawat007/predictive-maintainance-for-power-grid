import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle2, Circle, Download, Pause, Play, StepForward, XCircle } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { dur, liveRecord, minutesBetween, riskPct, ts } from '../lib/format'
import { C, TYPE_COLOR } from '../lib/palette'
import type { EngineRecord, FeedEvent, RunResult, SessionSummary } from '../lib/types'
import { useGrid } from '../store/grid'
import { RiskChart, SensorChart, SensorLegend } from './Charts'
import EventFeed from './EventFeed'
import { Explain, IntelligenceCard } from './Intelligence'
import Panel, { Empty } from './Panel'
import StatusBadge from './StatusBadge'

const ALL = ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
const KEY_SENSORS: Record<string, string[]> = {
  thermal_runaway: ['OT', 'HUFL'], load_oscillation: ['HUFL', 'MUFL', 'OT'], intermittent_arcing: ['HULL', 'MULL'],
  sudden: ['OT', 'HUFL', 'HULL'], none: ['OT', 'HUFL'], upload: ['OT', 'HUFL', 'MUFL'],
}
const ANOMALY_KINDS = new Set(['thermal_deviation', 'thermal_residual', 'load_instability', 'reactive_spikes'])
const STAGES = [
  { key: 'normal', label: 'Normal', rule: 'stream start' },
  { key: 'anomaly', label: 'Subtle anomaly', rule: 'first anomaly event from the feature monitors' },
  { key: 'rising', label: 'Risk increasing', rule: 'risk trend rising and ≥ 25 % of the warning level' },
  { key: 'warning', label: 'Warning', rule: 'WARNING status (2 consecutive steps over threshold)' },
  { key: 'critical', label: 'Critical', rule: 'CRITICAL status' },
  { key: 'failure', label: 'Trip', rule: 'injected failure (ground truth of the scenario)' },
] as const
type StageKey = typeof STAGES[number]['key']
const SPEEDS = [1, 4, 16, 64]

/** Streams an engine session (synthetic scenario or uploaded data) through the REAL engine: stage progression,
 *  risk & sensor charts, attribution, events and a measured result.
 *  - ``create``: stateful session, stepped on the server tick by tick (local / long-running server).
 *  - ``run``: stateless run (serverless deployment) — the engine computes every step in ONE request, and the
 *    returned records are then played back here at the chosen speed. Nothing is generated client-side. */
export default function SessionPlayer({ create, run: runAll, ready, kind, title }: {
  create?: () => Promise<SessionSummary>; run?: () => Promise<RunResult>; ready: boolean; kind: 'scenario' | 'upload'; title: string
}) {
  const system = useGrid((s) => s.system)
  const [session, setSession] = useState<SessionSummary | null>(null)
  const [recs, setRecs] = useState<EngineRecord[]>([])
  const [events, setEvents] = useState<FeedEvent[]>([])
  const [stages, setStages] = useState<Partial<Record<StageKey, string>>>({})
  const [playing, setPlaying] = useState(false)
  const [done, setDone] = useState(false)
  const [busyStart, setBusyStart] = useState(false)
  const [error, setError] = useState<string>()
  const [speed, setSpeed] = useState(kind === 'upload' ? 16 : 4)
  const [pick, setPick] = useState<string[] | null>(null)
  const [computeMs, setComputeMs] = useState<number | null>(null)
  const queue = useRef<{ records: EngineRecord[]; events: FeedEvent[] } | null>(null)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)
  const busy = useRef(false)
  const state = useRef({ session, stages })
  useEffect(() => { state.current = { session, stages } }, [session, stages])

  const ctx = session?.incident
  const fault = (ctx?.fault ?? ctx?.type ?? 'upload') as string
  const hasFault = kind === 'scenario' && fault !== 'none'
  const sensors = pick ?? KEY_SENSORS[fault] ?? ['OT', 'HUFL']

  const stop = useCallback(() => { if (timer.current) clearInterval(timer.current); timer.current = null; setPlaying(false) }, [])
  useEffect(() => () => { stop() }, [stop])

  const advance = useCallback(async (n: number) => {
    const s = state.current.session
    if (!s || busy.current) return
    busy.current = true
    try {
      let r: { records: Record<string, EngineRecord[]>; events: FeedEvent[]; cursor: number; done: boolean }
      if (queue.current) {                       // playback of a stateless run (already computed by the engine)
        const q = queue.current
        const chunk = q.records.splice(0, n)
        const lastIdx = chunk.at(-1)?.index ?? -1
        const k = q.events.findIndex((e) => e.index > lastIdx)
        const evs = q.events.splice(0, k < 0 ? q.events.length : k)
        r = { records: { [s.assets[0]]: chunk }, events: evs, cursor: s.cursor + chunk.length, done: q.records.length === 0 }
      } else {
        r = await api.step(s.id, n)
      }
      const newRecs = (r.records[s.assets[0]] ?? []).map(liveRecord)
      const st = { ...state.current.stages }
      const warnRisk = system?.thresholds.warning_risk ?? 0.04
      for (const ev of r.events) if (ANOMALY_KINDS.has(ev.kind) && !st.anomaly) st.anomaly = ev.ts
      for (const x of newRecs) {
        if (!st.rising && x.trend === 'rising' && (x.risk ?? 0) >= 0.25 * warnRisk) st.rising = x.timestamp
        if (!st.warning && (x.status === 'WARNING' || x.status === 'CRITICAL')) st.warning = x.timestamp
        if (!st.critical && x.status === 'CRITICAL') st.critical = x.timestamp
        if (!st.failure && x.ground_truth?.failure_event) st.failure = x.timestamp
      }
      setStages(st)
      setRecs((p) => [...p, ...newRecs])
      setEvents((p) => [...[...r.events].reverse(), ...p].slice(0, 400))
      setSession({ ...s, cursor: r.cursor })
      if (r.done) { stop(); setDone(true) }
    } catch (e) { setError((e as Error).message); stop() } finally { busy.current = false }
  }, [stop, system])

  const run = (n: number) => { if (timer.current) clearInterval(timer.current); timer.current = setInterval(() => advance(n), 350); setPlaying(true) }
  const start = async () => {
    stop(); setRecs([]); setEvents([]); setDone(false); setError(undefined); setPick(null); setBusyStart(true)
    try {
      if (session && !queue.current) api.dropSession(session.id).catch(() => {})
      let s: SessionSummary
      if (runAll) {
        const res = await runAll()
        s = res.session
        queue.current = { records: [...res.records], events: [...res.events].sort((a, b) => a.index - b.index) }
        setComputeMs(res.compute_ms)
      } else {
        queue.current = null
        setComputeMs(null)
        s = await create!()
      }
      setSession(s); state.current.session = s
      const first = { normal: s.start_timestamp }
      setStages(first); state.current.stages = first
      run(speed); advance(speed)
    } catch (e) { setError((e as Error).message) } finally { setBusyStart(false) }
  }

  const last = recs.at(-1)
  const failRec = recs.find((r) => r.ground_truth?.failure_event)
  const rel = (t?: string) => (t && ctx?.failure_time ? minutesBetween(ctx.failure_time, t) : null)
  const onsetIdx = ctx?.onset_idx
  const markers = [
    ...(hasFault && onsetIdx != null && fault !== 'sudden' ? recs.filter((r) => r.index === onsetIdx).map((r) => ({ index: r.index, label: 'fault injected', color: TYPE_COLOR[fault] ?? C.warn, dash: '2 3' })) : []),
    ...(stages.warning ? recs.filter((r) => r.timestamp === stages.warning).map((r) => ({ index: r.index, label: 'model warning', color: C.warn })) : []),
    ...(failRec ? [{ index: failRec.index, label: 'TRIP (injected)', color: C.crit }] : []),
  ]
  const progress = session ? (session.cursor - session.start) / Math.max(1, session.stop - session.start) : 0
  const curStage = [...STAGES].reverse().find((s) => stages[s.key])?.key ?? 'normal'
  const th = system?.thresholds
  const alarmSteps = recs.filter((r) => r.status === 'WARNING' || r.status === 'CRITICAL').length
  const alarmEpisodes = recs.reduce((n, r, i) => n + ((r.status === 'WARNING' || r.status === 'CRITICAL') && !(i && (recs[i - 1].status === 'WARNING' || recs[i - 1].status === 'CRITICAL')) ? 1 : 0), 0)
  const peak = recs.length ? Math.max(...recs.map((r) => r.risk ?? 0)) : null

  const downloadCsv = () => {
    const head = 'timestamp,raw_score,calibrated_risk,model_confidence,status,primary_signal'
    const body = recs.map((r) => [r.timestamp, r.raw_score ?? '', r.risk ?? '', r.confidence ?? '', r.status, r.signals?.[0]?.signal ?? ''].join(','))
    const url = URL.createObjectURL(new Blob([[head, ...body].join('\n')], { type: 'text/csv' }))
    const a = document.createElement('a'); a.href = url; a.download = 'smartgrid_scored.csv'; a.click(); URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-3">
      <Panel kicker="2 · stream through the real engine" title={title}
        right={<div className="flex flex-wrap items-center gap-2">
          <button onClick={start} disabled={!ready || busyStart} className="flex items-center gap-1.5 rounded border border-accent/70 bg-accent/10 px-3 py-1.5 text-[12.5px] font-semibold text-accent disabled:opacity-40" aria-label="Run through the live engine">
            <Play size={14} />{busyStart ? 'STARTING…' : session ? 'RUN AGAIN' : 'RUN'}</button>
          {playing ? <button onClick={stop} className="rounded border border-line-2 px-2 py-1.5" aria-label="Pause"><Pause size={14} /></button>
            : <button onClick={() => run(speed)} disabled={!session || done} className="rounded border border-line-2 px-2 py-1.5 disabled:opacity-40" aria-label="Resume"><Play size={14} /></button>}
          <button onClick={() => { stop(); advance(1) }} disabled={!session || done} className="rounded border border-line-2 px-2 py-1.5 disabled:opacity-40" aria-label="Step one observation"><StepForward size={14} /></button>
          <div className="flex overflow-hidden rounded border border-line-2" role="radiogroup" aria-label="Speed (observations per tick)">
            {SPEEDS.map((s) => <button key={s} role="radio" aria-checked={speed === s} onClick={() => { setSpeed(s); if (playing) run(s) }} className={`px-2 py-1 font-mono text-[11.5px] ${speed === s ? 'bg-ink text-bg' : 'text-ink-2'}`}>{s}×</button>)}
          </div>
        </div>}>
        {error && <div role="alert" className="mb-2 rounded border border-crit/50 bg-crit/10 px-3 py-2 text-sm text-crit">{error}</div>}
        {hasFault && <ol className="mb-3 grid grid-cols-3 gap-1.5 md:grid-cols-6" aria-label="Progression">
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
                <div className="num text-[10.5px] text-ink-3">{t ? (r != null ? (r === 0 ? 'T = trip' : `T ${r < 0 ? '−' : '+'}${dur(Math.abs(r))}`) : ts(t)) : missed ? 'not reached' : 'pending'}</div>
              </li>)
          })}
        </ol>}
        {computeMs != null && <div className="mb-1 font-mono text-[10.5px] text-ink-3">engine computed {session ? session.stop - session.start : ''} steps in {computeMs.toFixed(0)} ms on the server · playback paced for reading</div>}
        {session && <div className="mb-2 h-1.5 overflow-hidden rounded bg-line" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Stream progress">
          <motion.div className="h-full bg-accent" animate={{ width: `${progress * 100}%` }} transition={{ duration: 0.3 }} /></div>}
        {recs.length && th ? <>
          <RiskChart recs={recs} warn={th.warning_risk} crit={th.critical_risk} markers={markers} syncId="sp" height={220} showConf />
          <div className="mt-1 flex flex-wrap items-center justify-between gap-2 px-1">
            <div className="flex flex-wrap gap-3 text-[10.5px] text-ink-3">
              <span className="flex items-center gap-1"><span className="h-[2px] w-4" style={{ background: C.s[0] }} />calibrated risk</span>
              <span className="flex items-center gap-1"><span className="w-4 border-t border-dashed" style={{ borderColor: C.ink3 }} />raw score</span>
              <span className="flex items-center gap-1"><span className="h-[2px] w-4" style={{ background: C.s[5] }} />model confidence</span>
              {hasFault && <span>dotted marker = fault injection starts (scenario ground truth, not visible to the model)</span>}
            </div>
            <SensorLegend sensors={ALL} active={sensors} onToggle={(c) => setPick((sensors.includes(c) ? sensors.filter((x) => x !== c) : [...sensors, c]))} />
          </div>
          <SensorChart recs={recs} sensors={sensors} syncId="sp" markers={markers} height={150} />
        </> : <Empty>{ready ? 'Press RUN to stream the data through the engine, one 15-minute observation at a time.' : 'Configure the input on the left.'}</Empty>}
      </Panel>

      <AnimatePresence>
        {done && <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="panel grid gap-3 p-4 md:grid-cols-[1.3fr_1fr]" role="status" aria-live="polite">
          <div>
            <div className="label">Result · measured by the engine during this run</div>
            {hasFault ? (failRec?.lead_time_min != null
              ? <><div className="mt-1 font-mono text-[12px] font-semibold tracking-[0.14em] text-ok">FIRST RELIABLE WARNING</div>
                <div className="num text-[34px] font-semibold leading-tight text-ok">{dur(failRec.lead_time_min)} <span className="text-[15px] font-semibold tracking-[0.08em] text-ink-2">BEFORE TRIP</span></div></>
              : <><div className="mt-1 font-mono text-[12px] font-semibold tracking-[0.14em] text-warn">OUTCOME</div>
                <div className="text-[24px] font-semibold leading-tight text-warn">NO RELIABLE WARNING DETECTED</div>
                <p className="text-[12px] text-ink-2">{fault === 'sudden' ? 'Sudden failure: no precursor is injected, so no sensor-based model can warn.'
                  : stages.warning ? 'A warning was raised but had cleared more than 1 h before the trip.' : 'Risk never stayed above the warning threshold for 2 consecutive steps before the trip — try a higher severity or a longer precursor.'}</p></>)
              : <><div className="mt-1 font-mono text-[12px] font-semibold tracking-[0.14em] text-ink-2">{kind === 'upload' ? 'YOUR DATA' : 'FAULT-FREE RUN'}</div>
                <div className="num text-[28px] font-semibold leading-tight">{alarmEpisodes} alarm episode{alarmEpisodes === 1 ? '' : 's'}</div>
                <p className="text-[12px] text-ink-2">{kind === 'upload' ? `${recs.length} observations scored; ${alarmSteps} in WARNING/CRITICAL.` : 'Any alarm here is a false alarm (no fault was injected).'}</p></>}
            <p className="mt-1 text-[12px] text-ink-3">A warning is "reliable" if it is still active (gaps ≤ 1 h) when the failure occurs — the definition used in the offline evaluation.</p>
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[12px]">
            <dt className="text-ink-3">Observations scored</dt><dd className="num">{recs.length}</dd>
            <dt className="text-ink-3">Peak calibrated risk</dt><dd className="num">{riskPct(peak)}</dd>
            <dt className="text-ink-3">Alarm steps</dt><dd className="num">{alarmSteps}</dd>
            {hasFault && <><dt className="text-ink-3">Injected fault</dt><dd>{fault.replace('_', ' ')} · severity {ctx?.severity}×</dd></>}
            <dt className="text-ink-3">Export</dt><dd><button onClick={downloadCsv} className="flex items-center gap-1 text-accent"><Download size={12} />scored CSV</button></dd>
          </dl>
        </motion.div>}
      </AnimatePresence>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <Panel kicker="Engine state" title="Live intelligence" right={last && <StatusBadge status={last.status} />}><IntelligenceCard rec={last} label={ctx?.label ?? ''} /></Panel>
        <Panel kicker="Why is risk changing?" title="Model-attributed signals"><Explain rec={last} /></Panel>
        <Panel kicker="Backend events" title="Event feed" bodyClass="p-0 h-[420px]"><EventFeed events={events} labels={{ ETTm1: 'T1', ETTm2: 'T2' }} showIngestion={false} /></Panel>
      </div>
    </div>
  )
}
