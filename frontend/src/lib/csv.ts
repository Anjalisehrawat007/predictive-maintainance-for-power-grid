/** Client-side parsing and validation of an uploaded sensor CSV (the backend validates again). */
export const SENSOR_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'] as const
export const MIN_ROWS = 162          // feature look-back (114) + model window (48): backend `/api/uploads` enforces the same
export const MAX_ROWS = 5000

export type CsvRow = Record<string, string | number> & { timestamp: string }
export interface ParsedCsv { rows: CsvRow[]; error?: string; note?: string }

export function parseSensorCsv(text: string): ParsedCsv {
  const lines = text.replace(/^﻿/, '').split(/\r?\n/).filter((l) => l.trim())
  if (lines.length < 2) return { rows: [], error: 'The file is empty.' }
  const head = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''))
  const lower = head.map((h) => h.toLowerCase())
  const tsCol = lower.findIndex((h) => h === 'timestamp' || h === 'date' || h === 'datetime')
  if (tsCol < 0) return { rows: [], error: 'Missing a "timestamp" (or "date") column.' }
  const idx: Record<string, number> = {}
  for (const c of SENSOR_COLS) {
    const i = lower.indexOf(c.toLowerCase())
    if (i < 0) return { rows: [], error: `Missing column "${c}". Required: ${SENSOR_COLS.join(', ')}.` }
    idx[c] = i
  }
  let body = lines.slice(1)
  let note: string | undefined
  if (body.length > MAX_ROWS) { note = `using the last ${MAX_ROWS} of ${body.length} rows`; body = body.slice(-MAX_ROWS) }
  if (body.length < MIN_ROWS) return { rows: [], error: `Need at least ${MIN_ROWS} rows (about 40 h at 15 min); got ${body.length}.` }
  const rows: CsvRow[] = []
  let prev: number | null = null
  for (let k = 0; k < body.length; k++) {
    const cells = body[k].split(',').map((x) => x.trim().replace(/^"|"$/g, ''))
    const t = cells[tsCol]
    const ms = Date.parse(t.includes('T') ? t : t.replace(' ', 'T'))
    if (Number.isNaN(ms)) return { rows: [], error: `Row ${k + 2}: cannot read timestamp "${t}".` }
    if (prev != null && ms - prev !== 15 * 60 * 1000) return { rows: [], error: `Row ${k + 2}: readings must be consecutive 15-minute steps (got a ${Math.round((ms - prev) / 60000)} min gap).` }
    prev = ms
    const row: CsvRow = { timestamp: t.replace('T', ' ').slice(0, 19) }
    for (const c of SENSOR_COLS) {
      const v = Number(cells[idx[c]])
      if (!Number.isFinite(v)) return { rows: [], error: `Row ${k + 2}: "${c}" is not a number.` }
      row[c] = v
    }
    rows.push(row)
  }
  return { rows, note }
}
