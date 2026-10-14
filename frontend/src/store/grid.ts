import { create } from 'zustand'
import { api } from '../lib/api'
import { fromLog, liveRecord } from '../lib/format'
import type { Alert, EngineRecord, FeedEvent, SessionSummary, SystemInfo, Timeline } from '../lib/types'

const WINDOW = 192          // 48 h of 15-min steps kept per asset for charts
const FEED_MAX = 400
let timer: ReturnType<typeof setInterval> | null = null
let seekToken = 0
let inflight = false

/** Lightweight client-side performance trace (read by the benchmark; bounded to the last 500 steps). */
type PerfSample = { rtt_ms: number; render_ms: number; n: number; painted: boolean }
declare global { interface Window { __sgPerf?: { steps: PerfSample[] } } }
function recordPerf(sample: Omit<PerfSample, 'render_ms' | 'painted'>, t0: number) {
  const push = (painted: boolean) => {
    const w = window as Window
    w.__sgPerf ??= { steps: [] }
    w.__sgPerf.steps.push({ ...sample, render_ms: performance.now() - t0, painted })
    if (w.__sgPerf.steps.length > 500) w.__sgPerf.steps.shift()
    // mirrored to the DOM so external tooling (benchmarks, browser automation) can read it without page globals
    document.documentElement.setAttribute('data-sg-perf', JSON.stringify(w.__sgPerf.steps))
  }
  // visible: two animation frames ≈ the frame in which the commit was painted.
  // hidden tab: frames are paused, so fall back to "after React's commit" and flag it as not painted.
  if (document.visibilityState === 'visible') requestAnimationFrame(() => requestAnimationFrame(() => push(true)))
  else setTimeout(() => push(false), 0)
}

export type Speed = 1 | 2 | 4
export interface GridState {
  system?: SystemInfo
  timeline?: Timeline
  assets: string[]
  cursor: number
  selected: string
  mode: 'log' | 'live'
  history: Record<string, EngineRecord[]>
  feed: FeedEvent[]
  liveAlerts: Alert[]
  activeLogAlerts: Alert[]
  session?: SessionSummary
  playing: boolean
  speed: Speed
  pulse: number
  rttMs?: number
  reducedMotion: boolean
  showIngestion: boolean
  demo: boolean
  loading: boolean
  error?: string
  init: () => Promise<void>
  seek: (index: number) => Promise<void>
  play: () => Promise<void>
  pause: () => void
  stepOnce: () => Promise<void>
  replayBack: () => Promise<void>
  setSpeed: (s: Speed) => void
  select: (a: string) => void
  toggleReduced: () => void
  toggleIngestion: () => void
  setDemo: (on: boolean) => void
}

export const useGrid = create<GridState>((set, get) => ({
  assets: [], cursor: 0, selected: '', mode: 'log', history: {}, feed: [], liveAlerts: [], activeLogAlerts: [],
  playing: false, speed: 1, pulse: 0, loading: true, showIngestion: false,
  demo: typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('demo'),
  reducedMotion: typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,

  init: async () => {
    if (get().system) return
    try {
      const [system, timeline, incidents] = await Promise.all([api.system(), api.timeline(), api.incidents()])
      const assets = system.dataset.assets.map((a) => a.id)
      // open shortly before a held-out failure that the engine warned about in time (≥ 1 h lead),
      // so the first screen shows the system doing its job; falls back to any precursor failure
      const lo0 = timeline.assets[assets[0]].index[0] + 192
      const f = incidents.find((x) => (x.engine_lead_time_min ?? 0) >= 60 && x.failure_idx > lo0)
        ?? incidents.find((x) => x.type !== 'sudden' && x.failure_idx > lo0)
      const start = f ? f.failure_idx - 5 : timeline.assets[assets[0]].index[200]
      set({ system, timeline, assets, selected: f?.asset ?? assets[0] })
      await get().seek(start)
    } catch (e) {
      set({ error: (e as Error).message, loading: false })
    }
  },

  seek: async (index) => {
    const { timeline, assets, session } = get()
    if (!timeline) return
    if (timer) { clearInterval(timer); timer = null }
    if (session) { api.dropSession(session.id).catch(() => {}) }
    const lo = timeline.assets[assets[0]].index[0]
    const hi = timeline.assets[assets[0]].index.at(-1)!
    const i = Math.max(lo, Math.min(hi, Math.round(index)))
    const token = ++seekToken
    set({ cursor: i, mode: 'log', playing: false, session: undefined, loading: true })
    try {
      const h = get().system?.model.horizon_min ?? null
      const [snap, ...logs] = await Promise.all([api.snapshot(i), ...assets.map((a) => api.log(a, Math.max(lo, i - WINDOW + 1), i + 1))])
      if (token !== seekToken) return              // a newer seek superseded this one
      const history: Record<string, EngineRecord[]> = {}
      assets.forEach((a, k) => { history[a] = logs[k].map((r) => fromLog(r, a, h)) })
      set({
        history, feed: [...snap.feed].reverse() as FeedEvent[], liveAlerts: [],
        activeLogAlerts: snap.active_alerts as Alert[], loading: false, error: undefined,
      })
    } catch (e) {
      if (token === seekToken) set({ error: (e as Error).message, loading: false })
    }
  },

  play: async () => {
    const s = get()
    if (s.playing || !s.timeline) return
    let session = s.session
    try {
      if (!session || session.cursor !== s.cursor + 1) {
        session = await api.createSession(s.cursor + 1, s.assets)
      }
      set({ session, mode: 'live', playing: true, error: undefined })
      const tick = async () => {
        if (inflight) return
        inflight = true
        try { await advance(get().speed) } finally { inflight = false }
      }
      timer = setInterval(tick, 500)
      tick()
    } catch (e) {
      set({ error: (e as Error).message, playing: false })
    }
  },

  pause: () => {
    if (timer) { clearInterval(timer); timer = null }
    set({ playing: false })
  },

  stepOnce: async () => {
    const s = get()
    get().pause()
    if (!s.session || s.session.cursor !== s.cursor + 1) {
      const session = await api.createSession(s.cursor + 1, s.assets)
      set({ session, mode: 'live' })
    }
    await advance(1)
  },

  replayBack: async () => {
    const s = get()
    get().pause()
    await get().seek(s.cursor - 24)               // rewind 6 h, then replay through the live engine
    await get().play()
  },

  setSpeed: (speed) => set({ speed }),
  select: (selected) => set({ selected }),
  toggleReduced: () => set((s) => ({ reducedMotion: !s.reducedMotion })),
  toggleIngestion: () => set((s) => ({ showIngestion: !s.showIngestion })),
  setDemo: (demo) => set({ demo }),
}))

/** Advance the live session n steps and merge the real engine outputs into the shared state. */
async function advance(n: number) {
  const s = useGrid.getState()
  if (!s.session) return
  try {
    const t0 = performance.now()
    const r = await api.step(s.session.id, n)
    const rttMs = performance.now() - t0
    const history = { ...s.history }
    const ingest: FeedEvent[] = []
    for (const [a, recs] of Object.entries(r.records)) {
      const live = recs.map(liveRecord)
      history[a] = [...(history[a] ?? []), ...live].slice(-WINDOW)
      for (const x of live) ingest.push({ ts: x.timestamp, asset: a, kind: 'ingest', severity: 'debug', index: x.index,
        message: `Sensor ingestion · 7 channels · ${x.timings_ms?.total?.toFixed(1) ?? '—'} ms pipeline` })
    }
    const events = [...ingest, ...r.events].sort((a, b) => b.index - a.index || (a.kind === 'ingest' ? 1 : -1))
    const alerts = [...s.liveAlerts]
    for (const al of r.alerts) {
      const id = `s-${r.session}-${al.id}`
      const k = alerts.findIndex((x) => x.id === id)
      const v = { ...al, id }
      if (k >= 0) alerts[k] = v; else alerts.unshift(v)
    }
    useGrid.setState({
      history, cursor: r.cursor - 1, session: { ...s.session, cursor: r.cursor }, liveAlerts: alerts,
      feed: [...events, ...s.feed].slice(0, FEED_MAX), pulse: s.pulse + 1, error: undefined, rttMs,
    })
    recordPerf({ rtt_ms: rttMs, n }, performance.now())
    if (r.done) useGrid.getState().pause()
  } catch (e) {
    useGrid.getState().pause()
    useGrid.setState({ error: (e as Error).message })
  }
}

/** Latest record per asset at the shared cursor. */
export const latest = (s: GridState, a: string) => s.history[a]?.at(-1)

const hours = (min: number) => `${+(min / 60).toFixed(1)} h`
/** Model window/horizon labels taken from the backend's model metadata (never hard-coded in the UI). */
export function useModelSpec() {
  const m = useGrid((s) => s.system?.model)
  return {
    horizon: m ? hours(m.horizon_min) : '—',
    window: m ? `${+m.window_hours.toFixed(1)} h` : '—',
    windowHalf: m ? `${+(m.window_hours / 2).toFixed(1)} h` : '—',
    seq: m ? `${m.seq_len} × 15 min` : '—',
  }
}
