import { motion } from 'framer-motion'
import { ChevronRight } from 'lucide-react'
import { C } from '../lib/palette'
import type { SystemInfo } from '../lib/types'

const STAGES: { key: string; label: string; sub: string }[] = [
  { key: 'ingest', label: 'Sensor ingestion', sub: '7 SCADA channels' },
  { key: 'features', label: 'Causal features', sub: 'rolling · residual · ratios' },
  { key: 'sequence', label: 'Sequence buffer', sub: '' },
  { key: 'inference', label: 'Neural inference', sub: 'GRU + attention' },
  { key: 'confidence', label: 'Confidence', sub: '3-seed agreement' },
  { key: 'calibration', label: 'Calibration', sub: 'score → risk' },
  { key: 'explain', label: 'Attribution', sub: 'integrated gradients' },
  { key: 'risk_engine', label: 'Risk engine', sub: 'trend · health · status' },
  { key: 'alert_engine', label: 'Alert engine', sub: 'events · alerts' },
]

/** Real-time pipeline with MEASURED per-stage latency (latest live step, or engine-log median). */
export default function PipelineViz({ timings, source, pulse, system, rtt }: {
  timings?: Record<string, number>; source: 'live' | 'log'; pulse: number; system?: SystemInfo; rtt?: number
}) {
  const t = timings ?? {}
  const total = t.total ?? STAGES.reduce((a, s) => a + (t[s.key] ?? 0), 0)
  const onnx = system?.model.onnx_runtime
  const sub = (s: (typeof STAGES)[number]) => s.key === 'sequence' && system ? `${system.model.seq_len} × 15 min` : s.sub
  const b = system?.benchmark ?? {}
  return (
    <div className="space-y-3">
      <ol className="flex flex-wrap items-stretch gap-y-2" aria-label="Inference pipeline stages with measured latency">
        {STAGES.map((s, i) => {
          const ms = t[s.key]
          return (
            <li key={s.key} className="flex items-center">
              <div className="relative min-w-[92px] overflow-hidden rounded border border-line bg-bg px-2 py-1.5">
                <motion.div key={`${pulse}-${s.key}`} className="absolute inset-0 bg-accent/15" initial={{ opacity: 0 }}
                  animate={{ opacity: [0, 1, 0] }} transition={{ duration: 0.45, delay: i * 0.05 }} aria-hidden />
                <div className="relative text-[11px] font-semibold leading-tight">{s.label}{s.key === 'inference' && onnx ? <span className="ml-1 text-[9px] text-accent">ONNX</span> : null}</div>
                <div className="relative text-[9.5px] text-ink-3">{sub(s)}</div>
                <div className="relative num mt-0.5 text-[11.5px] text-accent">{ms == null ? '—' : ms < 0.1 ? `${(ms * 1000).toFixed(0)} µs` : `${ms.toFixed(2)} ms`}</div>
              </div>
              {i < STAGES.length - 1 && <ChevronRight size={12} className="mx-0.5 text-ink-3" aria-hidden />}
            </li>
          )
        })}
        <li className="flex items-center"><ChevronRight size={12} className="mx-0.5 text-ink-3" aria-hidden />
          <div className="rounded border border-line bg-bg px-2 py-1.5"><div className="text-[11px] font-semibold">Dashboard</div>
            <div className="text-[9.5px] text-ink-3">API round-trip</div><div className="num text-[11.5px] text-accent">{rtt ? `${rtt.toFixed(0)} ms` : '—'}</div></div></li>
      </ol>
      {/* processing timeline: width ∝ measured time */}
      <div>
        <div className="mb-1 flex items-baseline justify-between">
          <span className="label">Processing timeline · {source === 'live' ? 'latest live observation' : 'median over 20,904 engine-log steps'}</span>
          <span className="num text-[12px]">TOTAL <b className="text-ink">{total ? total.toFixed(2) : '—'} ms</b></span>
        </div>
        <div className="flex h-5 w-full overflow-hidden rounded border border-line" role="img" aria-label="Share of processing time per stage">
          {STAGES.map((s, i) => {
            const v = t[s.key] ?? 0
            if (!total || v / total < 0.004) return null
            return <div key={s.key} title={`${s.label}: ${v.toFixed(3)} ms`} style={{ width: `${(v / total) * 100}%`, background: C.s[i % C.s.length], opacity: 0.85 }}
              className="flex items-center justify-center overflow-hidden border-r border-bg text-[9.5px] font-mono text-bg">{v / total > 0.12 ? s.label : ''}</div>
          })}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        {[['PyTorch inference', b.pytorch_ms_p50], ['ONNX Runtime', b.onnx_ms_p50]].map(([l, v]) => (
          <div key={String(l)} className="rounded border border-line bg-bg px-2 py-1.5"><div className="label">{l}</div><div className="num text-[15px]">{v == null ? '—' : `${Number(v).toFixed(3)} ms`}</div><div className="text-[9.5px] text-ink-3">benchmark p50 · batch 1 · 1 CPU thread</div></div>
        ))}
        <div className="rounded border border-line bg-bg px-2 py-1.5"><div className="label">Full streaming step</div><div className="num text-[15px]">{system?.engine_timings.median_ms?.total?.toFixed(1) ?? '—'} ms</div><div className="text-[9.5px] text-ink-3">engine-log median incl. attribution</div></div>
      </div>
    </div>
  )
}
