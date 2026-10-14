import { describe, expect, it } from 'vitest'
import { MIN_ROWS, parseSensorCsv } from '../csv'

const COLS = 'HUFL,HULL,MUFL,MULL,LUFL,LULL,OT'
const csv = (n: number, header = `date,${COLS}`, stepMin = 15) => {
  const t0 = Date.UTC(2018, 3, 1)
  const rows = Array.from({ length: n }, (_, i) => {
    const t = new Date(t0 + i * stepMin * 60000).toISOString().slice(0, 19).replace('T', ' ')
    return `${t},5.1,2,1.4,0.4,4,1.2,30.5`
  })
  return [header, ...rows].join('\n')
}

describe('sensor CSV upload', () => {
  it('accepts ETDataset-style files ("date" column) of consecutive 15-min rows', () => {
    const p = parseSensorCsv(csv(MIN_ROWS))
    expect(p.error).toBeUndefined()
    expect(p.rows).toHaveLength(MIN_ROWS)
    expect(p.rows[0]).toMatchObject({ timestamp: '2018-04-01 00:00:00', HUFL: 5.1, OT: 30.5 })
  })
  it('rejects too few rows, missing channels, gaps and non-numbers with a clear message', () => {
    expect(parseSensorCsv(csv(MIN_ROWS - 1)).error).toMatch(/at least/)
    expect(parseSensorCsv(csv(MIN_ROWS, 'date,HUFL,HULL')).error).toMatch(/Missing column "MUFL"/)
    expect(parseSensorCsv(csv(MIN_ROWS, `date,${COLS}`, 30)).error).toMatch(/15-minute/)
    expect(parseSensorCsv(csv(MIN_ROWS).replace('30.5', 'abc')).error).toMatch(/not a number/)
  })
  it('keeps the most recent 5000 rows of a long file', () => {
    const p = parseSensorCsv(csv(5100))
    expect(p.rows).toHaveLength(5000)
    expect(p.note).toMatch(/last 5000/)
  })
})
