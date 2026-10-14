import { AlertOctagon, AlertTriangle, CheckCircle2, Loader2, PauseCircle } from 'lucide-react'
import type { Status } from './types'

/** Severity is always communicated with icon + text + colour (never colour alone). */
export const STATUS_META: Record<Status, { label: string; color: string; Icon: typeof CheckCircle2; rank: number }> = {
  NORMAL: { label: 'NORMAL', color: 'var(--color-ok)', Icon: CheckCircle2, rank: 1 },
  WARNING: { label: 'WARNING', color: 'var(--color-warn)', Icon: AlertTriangle, rank: 2 },
  CRITICAL: { label: 'CRITICAL', color: 'var(--color-crit)', Icon: AlertOctagon, rank: 3 },
  OFFLINE: { label: 'OFFLINE', color: 'var(--color-off)', Icon: PauseCircle, rank: 0 },
  WARMUP: { label: 'WARM-UP', color: 'var(--color-off)', Icon: Loader2, rank: 0 },
}

