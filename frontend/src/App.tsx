import { MotionConfig } from 'framer-motion'
import { lazy, Suspense, useEffect } from 'react'
import { Route, Routes } from 'react-router-dom'
import Shell from './components/Shell'
import CommandCenter from './pages/CommandCenter'
import { useGrid } from './store/grid'
import LiveOnly from './components/LiveOnly'
import { STATIC } from './lib/staticMode'

const IncidentReplay = lazy(() => import('./pages/IncidentReplay'))
const AssetExplorer = lazy(() => import('./pages/AssetExplorer'))
const AlertCenter = lazy(() => import('./pages/AlertCenter'))
const FailureIntel = lazy(() => import('./pages/FailureIntel'))
const ModelLab = lazy(() => import('./pages/ModelLab'))
const Methodology = lazy(() => import('./pages/Methodology'))

export default function App() {
  const init = useGrid((s) => s.init)
  const reduced = useGrid((s) => s.reducedMotion)
  useEffect(() => { init() }, [init])
  useEffect(() => { document.documentElement.classList.toggle('reduce-motion', reduced) }, [reduced])
  return (
    <MotionConfig reducedMotion={reduced ? 'always' : 'user'}>
      <Shell>
        <Suspense fallback={<div className="p-8 label">Loading view…</div>}>
          <Routes>
            {STATIC ? <>
              <Route path="/" element={<LiveOnly title="Command Center" shot="/screenshots/command_center.png"
                what="Topology, calibrated risk, alerts, explanations and measured pipeline latency, synchronised to one clock while held-out sensor data stream through the real engine. Guided Demo Mode starts here." />} />
              <Route path="/replay" element={<LiveOnly title="Incident Replay" shot="/screenshots/incident_replay.png"
                what="Replay any held-out (simulated) failure through the real streaming engine and see when the first reliable warning was raised. The engine-measured lead time equals the offline evaluation for all 40 test failures." />} />
              <Route path="/assets/*" element={<LiveOnly title="Asset Explorer" shot="/screenshots/command_center.png"
                what="Per-transformer state, health breakdown, sensor statistics and the risk history across the held-out period." />} />
              <Route path="/alerts" element={<LiveOnly title="Alert Center" shot="/screenshots/command_center.png"
                what="Every alert raised by the engine over the held-out period, with investigate, replay and acknowledge actions." />} />
            </> : <>
            <Route path="/" element={<CommandCenter />} />
            <Route path="/replay" element={<IncidentReplay />} />
            <Route path="/assets/:id" element={<AssetExplorer />} />
            <Route path="/assets" element={<AssetExplorer />} />
            <Route path="/alerts" element={<AlertCenter />} />
            </>}
            <Route path="/failures" element={<FailureIntel />} />
            <Route path="/models" element={<ModelLab />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="*" element={<div className="p-8">Not found.</div>} />
          </Routes>
        </Suspense>
      </Shell>
    </MotionConfig>
  )
}
