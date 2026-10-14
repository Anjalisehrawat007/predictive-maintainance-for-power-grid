import { MotionConfig } from 'framer-motion'
import { lazy, Suspense, useEffect } from 'react'
import { Route, Routes } from 'react-router-dom'
import Shell from './components/Shell'
import CommandCenter from './pages/CommandCenter'
import { useGrid } from './store/grid'

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
            <Route path="/" element={<CommandCenter />} />
            <Route path="/replay" element={<IncidentReplay />} />
            <Route path="/assets/:id" element={<AssetExplorer />} />
            <Route path="/assets" element={<AssetExplorer />} />
            <Route path="/alerts" element={<AlertCenter />} />
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
