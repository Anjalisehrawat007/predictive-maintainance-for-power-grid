import { STATUS_META } from '../lib/status'
import type { Status } from '../lib/types'

/** Severity is always communicated with icon + text + colour (never colour alone). */
export default function StatusBadge({ status, size = 'sm' }: { status: Status; size?: 'sm' | 'lg' }) {
  const m = STATUS_META[status] ?? STATUS_META.WARMUP
  const big = size === 'lg'
  return (
    <span role="status" aria-label={`Status ${m.label}`}
      className={`inline-flex items-center gap-1.5 rounded border font-mono font-semibold tracking-wider ${big ? 'px-2.5 py-1 text-sm' : 'px-1.5 py-0.5 text-[10.5px]'}`}
      style={{ color: m.color, borderColor: `color-mix(in srgb, ${m.color} 55%, transparent)`, background: `color-mix(in srgb, ${m.color} 10%, transparent)` }}>
      <m.Icon size={big ? 16 : 12} aria-hidden />{m.label}
    </span>
  )
}
