import { ExternalLink } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { RiskChart, SensorChart, SensorLegend } from '../components/Charts'
import EventFeed from '../components/EventFeed'
import HeroStrip from '../components/HeroStrip'
import { Explain, HealthBreakdown, IntelligenceCard } from '../components/Intelligence'
import Panel, { Empty } from '../components/Panel'
import PipelineViz from '../components/PipelineViz'
import Scrubber from '../components/Scrubber'
import StreamControls from '../components/StreamControls'
import Topology from '../components/Topology'
import { ts } from '../lib/format'
import { C } from '../lib/palette'
import type { EngineRecord } from '../lib/types'
import { useGrid } from '../store/grid'

const ALL = ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']

export default function CommandCenter() {
  const g = useGrid()
  const [sensors, setSensors] = useState(['OT', 'HUFL', 'HULL'])
  const labels = useMemo(() => Object.fromEntries((g.system?.dataset.assets ?? []).map((a) => [a.id, a.label])), [g.system])
  if (!g.system || !g.timeline) return <Empty>{g.error ? 'Backend unavailable.' : 'Connecting to the ML backend…'}</Empty>
  const sel = g.selected
  const recs: EngineRecord[] = g.history[sel] ?? []
  const last = recs.at(-1)
  const th = g.system.thresholds
  const markers = recs.filter((r) => r.ground_truth?.failure_event).map((r) => ({ index: r.index, label: 'FAILURE (sim)', color: C.crit }))
  const predicted = last && (last.status === 'WARNING' || last.status === 'CRITICAL') && last.warning_window_min
    ? { from: last.timestamp, to: new Date(new Date(last.timestamp.replace(' ', 'T') + 'Z').getTime() + last.warning_window_min * 60000).toISOString().replace('T', ' ').slice(0, 19) } : null
  const nAlerts = g.mode === 'live' ? g.liveAlerts.filter((a) => a.status === 'open').length : g.activeLogAlerts.length
  const lastTimings = g.mode === 'live' ? last?.timings_ms : g.system.engine_timings.median_ms
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <div className="label">Live grid command center</div>
          <h1 className="text-[20px] font-semibold tracking-tight">Predictive reliability · {ts(last?.timestamp)}</h1>
        </div>
        <StreamControls />
      </div>
      <HeroStrip history={g.history} nAlerts={nAlerts} />
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
        <Panel className="xl:col-span-5" kicker="Live grid topology" title="Assets & feeders"
          right={<Link to={`/assets/${sel}`} className="flex items-center gap-1 text-[11.5px] text-accent">Open {labels[sel]} <ExternalLink size={12} /></Link>}>
          <div className="h-[300px]"><Topology pulse={g.pulse} selected={sel} onSelect={g.select}
            nodes={g.assets.map((a) => ({ id: a, label: labels[a], substation: a.endsWith('1') ? 'Substation A' : 'Substation B', rec: g.history[a]?.at(-1) }))} /></div>
          <p className="px-1 text-[10.5px] text-ink-3">Two transformers exist in the dataset; both are shown. Ring = calibrated risk. Click to focus, double-click (or press O) to open.</p>
        </Panel>
        <Panel id="cc-risk" className="xl:col-span-7" kicker={`Transformer ${labels[sel]} · ${g.mode === 'live' ? 'live engine' : 'engine log'}`} title="Failure risk & sensor stream"
          right={<div className="flex gap-1" role="group" aria-label="Select asset">{g.assets.map((a) => (
            <button key={a} onClick={() => g.select(a)} aria-pressed={a === sel}
              className={`rounded border px-2 py-0.5 font-mono text-[11px] ${a === sel ? 'border-ink-3 text-ink' : 'border-line text-ink-3'}`}>{labels[a]}</button>))}</div>}>
          {recs.length ? <>
            <RiskChart recs={recs} warn={th.warning_risk} crit={th.critical_risk} syncId="cc" markers={markers} predictedWindow={predicted} height={210} />
            <div className="mt-1 flex flex-wrap items-center justify-between gap-2 px-1">
              <div className="flex gap-3 text-[10.5px] text-ink-3">
                <span className="flex items-center gap-1"><span className="h-[2px] w-4" style={{ background: C.s[0] }} />calibrated risk</span>
                <span className="flex items-center gap-1"><span className="h-0 w-4 border-t border-dashed" style={{ borderColor: C.ink3 }} />raw model score</span>
                <span className="flex items-center gap-1"><span className="h-3 w-3" style={{ background: 'rgba(107,118,133,.25)' }} />offline</span>
              </div>
              <SensorLegend sensors={ALL} active={sensors} onToggle={(c) => setSensors((s) => (s.includes(c) ? s.filter((x) => x !== c) : [...s, c]))} />
            </div>
            <SensorChart recs={recs} sensors={sensors} syncId="cc" markers={markers} height={150} futureUntil={predicted?.to} />
          </> : <Empty>Loading engine output…</Empty>}
        </Panel>
      </div>
      <div className="panel px-3 py-2.5">
        <Scrubber tl={g.timeline} assets={g.assets} labels={labels} cursor={g.cursor} onSeek={g.seek} />
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-12">
        <Panel className="xl:col-span-3" kicker="Operational intelligence" title={`Transformer ${labels[sel]}`}><IntelligenceCard rec={last} label={labels[sel]} /></Panel>
        <Panel id="cc-explain" className="xl:col-span-3" kicker="Why is risk changing?" title="Model-attributed signals"><Explain rec={last} /></Panel>
        <Panel className="xl:col-span-3" kicker="Asset condition" title="Grid health"><HealthBreakdown rec={last} /></Panel>
        <Panel className="xl:col-span-3" kicker={g.mode === 'live' ? 'Backend events · live' : 'Backend events · engine log (last 48 h)'} title="Event feed" bodyClass="p-0 h-[360px]">
          <EventFeed events={g.feed} labels={labels} showIngestion={g.showIngestion} onToggleIngestion={g.toggleIngestion} />
        </Panel>
      </div>
      <Panel id="cc-pipeline" kicker="Real-time pipeline" title="Sensor → model → risk → alert, with measured latency">
        <PipelineViz timings={lastTimings} source={g.mode} pulse={g.pulse} system={g.system} rtt={g.rttMs} />
      </Panel>
    </div>
  )
}
