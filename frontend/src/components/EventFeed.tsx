import { AnimatePresence, motion } from 'framer-motion'
import { Activity, AlertOctagon, AlertTriangle, Bolt, CheckCircle2, Clock, Flame, Gauge, Power, Radio, Waves } from 'lucide-react'
import { time } from '../lib/format'
import type { FeedEvent } from '../lib/types'

const KIND: Record<string, { Icon: typeof Activity; color: string; label: string }> = {
  ingest: { Icon: Radio, color: '#6b7685', label: 'ingest' },
  thermal_deviation: { Icon: Flame, color: '#d95926', label: 'thermal' },
  thermal_residual: { Icon: Flame, color: '#d95926', label: 'thermal' },
  load_instability: { Icon: Waves, color: '#3987e5', label: 'load' },
  reactive_spikes: { Icon: Bolt, color: '#199e70', label: 'spikes' },
  risk_update: { Icon: Gauge, color: '#a3adbb', label: 'risk' },
  warning: { Icon: AlertTriangle, color: '#fab219', label: 'WARNING' },
  critical: { Icon: AlertOctagon, color: '#d03b3b', label: 'CRITICAL' },
  cleared: { Icon: CheckCircle2, color: '#0ca30c', label: 'cleared' },
  offline: { Icon: Power, color: '#d03b3b', label: 'trip' },
  online: { Icon: Power, color: '#0ca30c', label: 'online' },
  failure: { Icon: AlertOctagon, color: '#d03b3b', label: 'FAILURE·GT' },
  lead_time: { Icon: Clock, color: '#3fb6d6', label: 'lead time' },
  missed: { Icon: AlertTriangle, color: '#fab219', label: 'missed' },
}

export default function EventFeed({ events, labels, showIngestion, onToggleIngestion, max = 120 }: {
  events: FeedEvent[]; labels: Record<string, string>; showIngestion: boolean; onToggleIngestion?: () => void; max?: number
}) {
  const shown = events.filter((e) => showIngestion || e.kind !== 'ingest').slice(0, max)
  return (
    <div className="flex h-full flex-col">
      {onToggleIngestion && (
        <label className="flex items-center gap-2 border-b border-line px-3 py-1.5 text-[11px] text-ink-3">
          <input type="checkbox" checked={showIngestion} onChange={onToggleIngestion} /> show per-observation ingestion events
        </label>
      )}
      <ol className="scroll-thin min-h-0 flex-1 overflow-y-auto" aria-live="polite" aria-label="Event feed">
        <AnimatePresence initial={false}>
          {shown.map((e) => {
            const k = KIND[e.kind] ?? { Icon: Activity, color: '#a3adbb', label: e.kind }
            return (
              <motion.li key={`${e.index}-${e.asset}-${e.kind}-${e.message}`} layout initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="grid grid-cols-[38px_26px_16px_1fr] items-start gap-1.5 border-b border-line/60 px-3 py-1.5 text-[11.5px]">
                <span className="num text-ink-3">{time(e.ts)}</span>
                <span className="font-mono text-[10.5px] text-ink-2">{labels[e.asset] ?? e.asset}</span>
                <k.Icon size={13} style={{ color: k.color }} aria-label={k.label} />
                <span className={e.kind === 'ingest' ? 'text-ink-3' : 'text-ink'}>{e.message}</span>
              </motion.li>
            )
          })}
        </AnimatePresence>
        {!shown.length && <li className="p-4 text-center text-[12px] text-ink-3">No events in this window. Press <b>Stream</b> to run the live engine.</li>}
      </ol>
    </div>
  )
}
