/* eslint-disable @typescript-eslint/no-explicit-any */
import { AlertTriangle, CheckCircle2, ChevronDown, XCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import Panel, { Empty } from '../components/Panel'
import { api } from '../lib/api'
import { pct } from '../lib/format'
import { useGrid } from '../store/grid'

function stepDesc(m: { n_features: number; type: string; seq_len: number; window_hours: number; horizon_min: number; calibration: string; onnx_runtime: boolean; ensemble_members: number }): Record<string, string> {
  return {
    'Sensor ingestion': '7 real SCADA channels every 15 min: HV/MV/LV active & reactive load, oil temperature.',
    'Causal preprocessing': 'Per-transformer scaling with statistics fitted on the training period only; offline rows are masked, never filled from the future.',
    'Temporal features': `${m.n_features} features computed only from the past: differences, rolling std/range, 24 h z-scores, volatility ratios, spike counts, oil-temperature residual vs. load.`,
    [`${m.window_hours}-hour sequence`]: `Ring buffer of the last ${m.seq_len} feature vectors (${m.seq_len} × 15 min).`,
    'GRU + attention': `GRU (64 units) + additive temporal attention → one raw score for "failure within ${m.horizon_min / 60} h"${m.onnx_runtime ? '; served with ONNX Runtime (verified numerically identical to PyTorch at load)' : ''}.`,
    'Failure probability': 'The raw score from weighted-BCE training. It ranks risk but is not a probability.',
    Calibration: `${m.calibration} calibration fitted on validation data maps the raw score to a calibrated probability.`,
    Confidence: `Model confidence = 1 − 2·σ of ${m.ensemble_members} independently seeded models (agreement, not physical certainty).`,
    'Risk engine': 'Trend (1 h), derived health indicator and integrated-gradients attribution (what the model relied on, not causes).',
    Alert: 'WARNING after 2 consecutive steps above the validated threshold; CRITICAL above the 90 %-precision threshold; events on threshold crossings.',
  }
}

export default function Methodology() {
  const system = useGrid((s) => s.system)
  const [d, setD] = useState<Record<string, any>>()
  const [err, setErr] = useState<string>()
  useEffect(() => { api.methodology().then(setD).catch((e) => setErr(e.message)) }, [])
  if (err) return <Empty><span role="alert">Could not load data from the ML backend ({err}). Is the API running on port 8000?</span></Empty>
  if (!d || !system) return <Empty>Loading…</Empty>
  const shift = d.shift?.resim
  const STEP_DESC = stepDesc(system.model)
  const leak = Object.entries(d.leakage ?? {}) as [string, { passed: boolean; detail: string }][]
  const timings = system.engine_timings.median_ms ?? {}
  // stage → engine timing keys (median ms over the engine-log run); preprocessing is timed together with the features
  const tKey: Record<string, string[]> = { 'Sensor ingestion': ['ingest'], 'Temporal features': ['features'], [`${system.model.window_hours}-hour sequence`]: ['sequence'], 'GRU + attention': ['inference'],
    Calibration: ['calibration'], Confidence: ['confidence'], 'Risk engine': ['risk_engine', 'explain'], Alert: ['alert_engine'] }
  const stageMs = (s: string) => (tKey[s]?.every((k) => timings[k] != null) ? tKey[s].reduce((a, k) => a + timings[k], 0) : null)
  return (
    <div className="mx-auto max-w-6xl space-y-3">
      <div><div className="label">Model & methodology</div><h1 className="text-[22px] font-semibold">How the system works — and what it cannot claim</h1></div>
      <div role="note" className="flex gap-3 rounded-lg border border-warn/50 bg-warn/5 p-4">
        <AlertTriangle className="mt-0.5 shrink-0 text-warn" size={20} aria-hidden />
        <div><div className="font-semibold text-warn">Failure events are simulated because the source dataset does not contain real failure labels.</div>
          <p className="mt-1 text-[13px] text-ink-2">{d.disclosure} Every injected perturbation is stored separately from the real measurement and is never a model input (verified by automated leakage checks). Performance numbers describe detection of these simulated fault signatures inside real operating data — not real-world transformer failure rates.</p></div>
      </div>
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-4">
        {Object.entries(d.facts).map(([k, v]) => <div key={k} className="bg-surface px-3 py-2.5"><div className="label">{k.replace('_', ' ')}</div><div className="text-[13.5px] font-semibold">{String(v)}</div></div>)}
      </div>
      <Panel kicker="End-to-end pipeline" title="From sensor to alert (median measured latency per stage)">
        <ol className="grid gap-1">
          {d.pipeline.map((s: string, i: number) => (
            <li key={s}>
              <div className="grid grid-cols-[28px_200px_1fr_90px] items-center gap-3 rounded border border-line bg-bg px-3 py-2">
                <span className="num text-ink-3">{String(i + 1).padStart(2, '0')}</span>
                <span className="font-semibold">{s}</span>
                <span className="text-[12.5px] text-ink-2">{STEP_DESC[s]}</span>
                <span className="num text-right text-accent">{(() => { const v = stageMs(s); return v == null ? (s === 'Causal preprocessing' ? 'incl. in features' : '') : `${v.toFixed(v < 0.1 ? 3 : 2)} ms` })()}</span>
              </div>
              {i < d.pipeline.length - 1 && <ChevronDown size={14} className="mx-auto my-0.5 text-ink-3" aria-hidden />}
            </li>))}
        </ol>
      </Panel>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Panel kicker="Data integrity" title="Leakage checks (automated)">
          <ul className="space-y-1.5 text-[12.5px]">{leak.map(([k, v]) => (
            <li key={k} className="flex gap-2">{v.passed ? <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-ok" aria-label="passed" /> : <XCircle size={15} className="mt-0.5 shrink-0 text-crit" aria-label="failed" />}
              <div><b>{k.replace(/_/g, ' ')}</b> <span className="text-ink-3">— {v.detail}</span></div></li>))}</ul>
        </Panel>
        <Panel kicker="Reproducibility" title="Phase-1 results re-computed from saved artefacts">
          <table className="w-full text-[12px]"><thead className="text-left"><tr className="label"><th className="py-1 font-normal">Metric</th><th className="font-normal">Claimed</th><th className="font-normal">Reproduced</th><th className="font-normal">Status</th></tr></thead>
            <tbody>{(d.reproduction ?? []).map((r: any) => <tr key={r.metric} className="border-t border-line/60"><td className="py-1">{r.metric}</td><td className="num">{r.claimed}</td><td className="num">{typeof r.reproduced === 'number' ? r.reproduced.toFixed(4) : r.reproduced}</td><td className={r.status === 'MATCH' ? 'text-ok' : 'text-ink-3'}>{r.status}</td></tr>)}</tbody></table>
        </Panel>
      </div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Panel kicker="Phase-2 finding" title="Why test looked worse than validation">
          {shift ? <div className="space-y-2 text-[12.5px] text-ink-2">
            <p>The real sensor data were kept and the failure events were <b>re-simulated {shift.n} times</b>; the deployed model was evaluated unchanged.</p>
            <p>Mean PR-AUC on detectable positives: validation <b className="num text-ink">{shift.val_pr_auc_det_mean.toFixed(3)}</b> vs test <b className="num text-ink">{shift.test_pr_auc_det_mean.toFixed(3)}</b> (gap {shift.gap_mean >= 0 ? '+' : ''}{shift.gap_mean.toFixed(3)} ± {shift.gap_std.toFixed(3)}; test lower in {shift.test_lower_in}/{shift.n}).</p>
            <p>So the original validation → test gap was mainly the luck of the particular simulated events plus selection on the original validation events — <b>not</b> a systematic seasonal failure. A realistic expectation for this model on detectable positives is ≈ {((shift.val_pr_auc_det_mean + shift.test_pr_auc_det_mean) / 2).toFixed(2)} PR-AUC.</p>
          </div> : <Empty>Run the Phase-2 analysis.</Empty>}
        </Panel>
        <Panel kicker="Calibration & confidence" title="Raw score vs calibrated risk">
          <div className="space-y-2 text-[12.5px] text-ink-2">
            <p>Chosen calibrator: <b className="text-ink">{d.calibration?.chosen}</b> (selected by time-split cross-fitting inside validation; never fitted on test).</p>
            <table className="w-full text-[12px]"><thead className="text-left"><tr className="label"><th className="font-normal">Method</th><th className="font-normal">CV ECE</th><th className="font-normal">CV Brier</th></tr></thead>
              <tbody>{(d.calibration?.cv_selection ?? []).map((r: any) => <tr key={r.method} className="border-t border-line/60"><td className="py-1">{r.method}</td><td className="num">{r.cv_ece.toFixed(4)}</td><td className="num">{r.cv_brier.toFixed(4)}</td></tr>)}</tbody></table>
            <p>Model confidence = agreement of three independently-seeded models. At the deployed operating point on the held-out test period, HIGH-confidence alarm steps were correct {pct(d.final?.test?.confidence_bins?.HIGH?.precision)} of the time vs {pct(d.final?.test?.confidence_bins?.MEDIUM?.precision)} (MEDIUM) and {pct(d.final?.test?.confidence_bins?.LOW?.precision)} (LOW). It is <b>model confidence</b>, not physical certainty.</p>
          </div>
        </Panel>
      </div>
      <Panel kicker="Simulation layer" title="How failures and look-alike disturbances are generated">
        <div className="grid gap-3 text-[12.5px] text-ink-2 md:grid-cols-2">
          <div><div className="label mb-1">Failure mechanisms (probability)</div>
            <ul className="space-y-0.5">{Object.entries(d.simulation.failure_types).map(([k, v]) => <li key={k}><span className="text-ink">{k.replace('_', ' ')}</span> — {Number(v) * 100}%</li>)}</ul>
            <p className="mt-2">Precursor length {d.simulation.precursor_hours.join('–')} h; failures every {d.simulation.failure_gap_days.join('–')} days (accelerated for statistical power); outage {d.simulation.outage_hours.join('–')} h after each trip.</p></div>
          <div><div className="label mb-1">Benign look-alikes ({d.simulation.nuisance_per_day} per day)</div>
            <ul className="space-y-0.5">{Object.entries(d.simulation.nuisance_types).map(([k, v]) => <li key={k}><span className="text-ink">{k.replace(/_/g, ' ')}</span> — {Number(v) * 100}%</li>)}</ul>
            <p className="mt-2">These never lead to failure; they force the model to separate degradation from ordinary disturbances.</p></div>
        </div>
      </Panel>
      <Panel kicker="Honest limits" title="What this system does not show">
        <ul className="list-disc space-y-1 pl-5 text-[12.5px] text-ink-2">
          <li>Real failure behaviour: all failure labels are simulated; absolute numbers measure detection of the simulated signatures.</li>
          <li>Voltage, current, frequency and weather: not present in the dataset and not fabricated.</li>
          <li>Generalisation to new transformers is weak (train on one asset → test on the other drops PR-AUC sharply; see Model Lab / Phase-2 report).</li>
          <li>Sudden failures (no precursor) cannot be predicted from these sensors.</li>
          <li>The health score is a derived application indicator, not an industry reliability standard. Attributions are associational, not causal.</li>
          <li>This is a decision-support prototype; it does not guarantee detection or prevention of outages.</li>
        </ul>
      </Panel>
    </div>
  )
}
