import { describe, expect, it } from 'vitest'
import { pickDemoIncident } from '../demo'
import { confBin, modelName } from '../format'
import type { Incident } from '../types'

const inc = (id: number, lead: number | null, cls = 'A', type = 'thermal_runaway') =>
  ({ asset: 'ETTm1', label: 'T1', event_id: id, type, failure_idx: 0, onset_idx: 0, failure_time: '', onset_time: '', class: cls, engine_lead_time_min: lead }) as Incident

describe('confidence bins come from the API cut points', () => {
  const bins = { HIGH: 0.9, MEDIUM: 0.7, LOW: 0 }
  it('labels by lower bound', () => {
    expect(confBin(0.95, bins)).toBe('HIGH')
    expect(confBin(0.9, bins)).toBe('HIGH')
    expect(confBin(0.8, bins)).toBe('MEDIUM')
    expect(confBin(0.2, bins)).toBe('LOW')
  })
  it('shows nothing without data', () => {
    expect(confBin(null, bins)).toBe('')
    expect(confBin(0.95, undefined)).toBe('')
  })
})

describe('model names', () => {
  it('maps internal types to readable names', () => {
    expect(modelName('gru_attention')).toBe('GRU + temporal attention')
    expect(modelName('unknown_type')).toBe('unknown_type')
  })
})

describe('demo incident selection (real data, deterministic)', () => {
  it('picks the class-A warned failure closest to the median lead time', () => {
    const list = [inc(1, null), inc(2, 60), inc(3, 180), inc(4, 600), inc(5, 175, 'B'), inc(6, 200)]
    expect(pickDemoIncident(list)?.event_id).toBe(3)
  })
  it('never picks an unwarned failure when a warned one exists', () => {
    expect(pickDemoIncident([inc(1, null), inc(2, 90, 'C')])?.event_id).toBe(2)
  })
})
