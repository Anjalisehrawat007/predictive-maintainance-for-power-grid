import { useMemo, useRef, useState } from 'react'
import { ts } from '../lib/format'
import { C, STATUS_HEX, TYPE_COLOR } from '../lib/palette'
import type { Timeline } from '../lib/types'

const CODE_HEX: Record<number, string> = { 0: C.line, 1: '#13261a', 2: C.warn, 3: C.crit, 4: '#2a2f38' }

/** Global time scrubber over the held-out test period: per-asset status strips (engine log),
 *  failure markers, and a keyboard-accessible range input that drives the shared clock. */
export default function Scrubber({ tl, assets, labels, cursor, onSeek }: {
  tl: Timeline; assets: string[]; labels: Record<string, string>; cursor: number; onSeek: (i: number) => void
}) {
  const idx = tl.assets[assets[0]].index
  const lo = idx[0], hi = idx.at(-1)!
  const [drag, setDrag] = useState<number | null>(null)       // value while the user is dragging
  const val = drag ?? cursor
  const t = useRef<ReturnType<typeof setTimeout> | null>(null)
  const strips = useMemo(() => assets.map((a) => {
    const st = tl.assets[a].status, n = st.length, W = 1000
    const rects: { x: number; w: number; c: string }[] = []
    let s = 0
    for (let i = 1; i <= n; i++) {
      if (i === n || st[i] !== st[s]) { rects.push({ x: (s / n) * W, w: Math.max(((i - s) / n) * W, 0.6), c: CODE_HEX[st[s]] }); s = i }
    }
    return rects
  }), [tl, assets])
  const change = (v: number) => {
    setDrag(v)
    if (t.current) clearTimeout(t.current)
    t.current = setTimeout(() => { onSeek(v); setDrag(null) }, 120)   // debounce: one engine-log fetch per pause
  }
  const pos = ((val - lo) / (hi - lo)) * 100
  const months = useMemo(() => {
    const out: { p: number; l: string }[] = []
    let last = ''
    tl.timestamps.forEach((s, i) => { const m = s.slice(0, 7); if (m !== last) { out.push({ p: (i / tl.timestamps.length) * 100, l: m }); last = m } })
    return out
  }, [tl])
  return (
    <div className="select-none">
      <div className="mb-1 flex items-center justify-between">
        <span className="label">Held-out test period · engine log</span>
        <span className="num text-[11.5px] text-ink-2" aria-live="polite">{ts(tl.timestamps[val - lo])}</span>
      </div>
      <div className="relative">
        {assets.map((a, k) => (
          <div key={a} className="mb-[3px] flex items-center gap-2">
            <span className="w-6 font-mono text-[10px] text-ink-3">{labels[a]}</span>
            <svg viewBox="0 0 1000 10" preserveAspectRatio="none" className="h-2.5 flex-1" aria-hidden>
              {strips[k].map((r, i) => <rect key={i} x={r.x} width={r.w} height={10} fill={r.c} />)}
              {tl.failures.filter((f) => f.asset === a).map((f) => (
                <rect key={f.event_id} x={((f.failure_idx - lo) / (hi - lo)) * 1000 - 1} width={2.5} height={10} fill={TYPE_COLOR[f.type] ?? C.ink} />
              ))}
            </svg>
          </div>
        ))}
        <div className="pointer-events-none absolute -top-1 bottom-0 w-px bg-ink" style={{ left: `calc(2rem + (100% - 2rem) * ${pos / 100})` }} aria-hidden />
      </div>
      <div className="flex items-center gap-2">
        <span className="w-6" />
        <input className="scrub flex-1" type="range" min={lo} max={hi} step={1} value={val}
          onChange={(e) => change(Number(e.target.value))} aria-label="Timeline scrubber (15-minute steps)"
          aria-valuetext={ts(tl.timestamps[val - lo])} />
      </div>
      <div className="relative ml-8 h-3">
        {months.map((m) => <span key={m.l} className="absolute font-mono text-[9.5px] text-ink-3" style={{ left: `${m.p}%` }}>{m.l}</span>)}
      </div>
      <div className="mt-1 flex flex-wrap gap-3 font-mono text-[10px] text-ink-3">
        {[['NORMAL', '#13261a'], ['WARNING', STATUS_HEX.WARNING], ['CRITICAL', STATUS_HEX.CRITICAL], ['OFFLINE', '#2a2f38']].map(([l, c]) => (
          <span key={l} className="flex items-center gap-1"><span className="h-2 w-3 rounded-sm border border-line-2" style={{ background: c }} />{l}</span>))}
        <span className="flex items-center gap-1"><span className="h-2 w-[3px]" style={{ background: TYPE_COLOR.thermal_runaway }} />failure (simulated) — colour = type</span>
      </div>
    </div>
  )
}
