import { describe, expect, it } from 'vitest'
import { dur, fromLog, minutesBetween, pct, riskPct } from '../format'

describe('formatting', () => {
  it('never displays 100 % calibrated risk', () => {
    expect(riskPct(1)).toBe('≥ 99%')
    expect(riskPct(0.9951)).toBe('≥ 99%')
    expect(riskPct(0.123)).toBe('12.3%')
    expect(riskPct(null)).toBe('—')
  })
  it('formats durations and percentages', () => {
    expect(dur(165)).toBe('2h 45m')
    expect(dur(30)).toBe('30m')
    expect(dur(null)).toBe('—')
    expect(pct(0.5)).toBe('50%')
    expect(minutesBetween('2018-03-20 04:00:00', '2018-03-20 06:45:00')).toBe(165)
  })
})

describe('engine-log adapter', () => {
  const row = {
    index: 60000, timestamp: '2018-03-20 04:00:00', HUFL: 1, HULL: 2, MUFL: 3, MULL: 4, LUFL: 5, LULL: 6, OT: 30,
    offline: false, status: 'WARNING', raw_score: 0.7, risk: 0.2, confidence: 0.95, trend: 'rising',
    signal1: 'Oil temperature', signal1_share: 0.4, signal2: 'HV reactive load', signal2_share: 0.2,
    attribution: '{"Oil temperature":0.4,"HV reactive load":0.2}', health: 70, h_thermal_stability: 40,
    h_load_stability: 80, h_sensor_quality: 90, h_failure_risk: 80, h_model_confidence: 95,
    failure_event: 0, sim_phase: 'precursor', sim_event_type: 'thermal_runaway', lead_time_min: null, t_total_ms: 14,
  }
  it('maps flat rows to engine records with the deployed horizon', () => {
    const r = fromLog(row, 'ETTm1', 120)
    expect(r.sensors.OT).toBe(30)
    expect(r.warning_window_min).toBe(120)          // alarm state → horizon window
    expect(r.signals?.[0].signal).toBe('Oil temperature')
    expect(r.attribution?.[0].share).toBeCloseTo(0.4)
    expect(r.health?.components.thermal_stability).toBe(40)
    expect(r.source).toBe('log')
  })
  it('has no warning window when not alarming', () => {
    expect(fromLog({ ...row, status: 'NORMAL' }, 'ETTm1', 120).warning_window_min).toBeNull()
  })
})

describe('offline rows are never shown as scored', () => {
  it('masks risk, score and confidence of offline live records', async () => {
    const { liveRecord } = await import('../format')
    const base = { index: 1, timestamp: 't', asset: 'ETTm2', sensors: {}, status: 'OFFLINE', risk: 0.57, raw_score: 0.9, confidence: 0.8 }
    const off = liveRecord({ ...base, offline: true } as never)
    expect([off.risk, off.raw_score, off.confidence]).toEqual([null, null, null])
    expect(liveRecord({ ...base, offline: false, status: 'WARNING' } as never).risk).toBe(0.57)
  })
})
