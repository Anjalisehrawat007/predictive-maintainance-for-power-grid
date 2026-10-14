import type { Incident } from './types'

/** Demo incident: chosen from the held-out failures as the warned, strong-signal (class A) failure whose
 *  engine-measured lead time is closest to the median lead time of all warned failures (real data, deterministic). */
export function pickDemoIncident(list: Incident[]): Incident | undefined {
  const warned = list.filter((i) => i.engine_lead_time_min != null)
  if (!warned.length) return list[0]
  const leads = warned.map((i) => i.engine_lead_time_min!).sort((a, b) => a - b)
  const median = leads[Math.floor(leads.length / 2)]
  const pool = warned.filter((i) => i.class === 'A' && i.type !== 'sudden')
  return (pool.length ? pool : warned).reduce((best, i) =>
    Math.abs(i.engine_lead_time_min! - median) < Math.abs(best.engine_lead_time_min! - median) ? i : best)
}
