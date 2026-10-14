import { Activity, Bell, BookOpen, Boxes, FlaskConical, History, LayoutDashboard, PlayCircle, ShieldAlert, Zap } from 'lucide-react'
import type { ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { modelName } from '../lib/format'
import { useGrid } from '../store/grid'
import DemoGuide from './DemoGuide'
import { STATIC } from '../lib/staticMode'

const NAV = [
  { to: '/', label: 'Command Center', Icon: LayoutDashboard },
  { to: '/replay', label: 'Incident Replay', Icon: History },
  { to: '/assets', label: 'Asset Explorer', Icon: Boxes },
  { to: '/alerts', label: 'Alert Center', Icon: Bell },
  { to: '/failures', label: 'Failure Intelligence', Icon: ShieldAlert },
  { to: '/models', label: 'Model Lab', Icon: FlaskConical },
  { to: '/methodology', label: 'Model & Methodology', Icon: BookOpen },
]

export default function Shell({ children }: { children: ReactNode }) {
  const { system, error, mode, playing, reducedMotion, toggleReduced, liveAlerts, activeLogAlerts, demo, setDemo } = useGrid()
  const onReplay = useLocation().pathname.startsWith('/replay')
  const online = !!system && !error
  const nAlerts = mode === 'live' ? liveAlerts.filter((a) => a.status === 'open').length : activeLogAlerts.length
  return (
    <div className="flex h-full min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:m-2 focus:rounded focus:bg-surface-2 focus:px-3 focus:py-2">Skip to content</a>
      <nav aria-label="Primary" className="hidden w-56 shrink-0 flex-col border-r border-line bg-bg/80 md:flex">
        <div className="flex items-center gap-2.5 border-b border-line px-4 py-4">
          <div className="grid h-8 w-8 place-items-center rounded-md border border-line-2 bg-surface-2"><Zap size={16} className="text-accent" aria-hidden /></div>
          <div>
            <div className="text-[13px] font-bold tracking-[0.14em]">SMARTGRID AI</div>
            <div className="text-[10.5px] text-ink-3">Predictive Reliability Intelligence</div>
          </div>
        </div>
        <ul className="flex-1 space-y-0.5 p-2">
          {NAV.map(({ to, label, Icon }) => (
            <li key={to}>
              <NavLink to={to} end={to === '/'}
                className={({ isActive }) => `flex items-center gap-2.5 rounded-md px-3 py-2 text-[13px] transition-colors ${isActive ? 'bg-surface-2 text-ink border border-line-2' : 'text-ink-2 border border-transparent hover:bg-surface hover:text-ink'}`}>
                <Icon size={15} aria-hidden />{label}
                {to === '/alerts' && nAlerts > 0 && <span className="ml-auto rounded bg-warn/15 px-1.5 text-[10.5px] font-mono text-warn" aria-label={`${nAlerts} active alerts`}>{nAlerts}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
        <div className="space-y-2 border-t border-line p-3 text-[11px] text-ink-3">
          <label className="flex cursor-pointer items-center justify-between gap-2">
            <span>Reduced motion</span>
            <input type="checkbox" checked={reducedMotion} onChange={toggleReduced} className="accent-[var(--color-accent)]" aria-label="Toggle reduced motion" />
          </label>
          <p className="leading-snug">Failure events are <b className="text-ink-2">simulated</b>; sensor data are real (ETDataset).</p>
        </div>
      </nav>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-line bg-bg/70 px-5 py-2.5 backdrop-blur">
          <div className="md:hidden text-[13px] font-bold tracking-[0.14em]">SMARTGRID AI</div>
          <div className="flex items-center gap-2 text-[11.5px] font-mono" role="status" aria-live="polite">
            <span className={`h-2 w-2 rounded-full ${online ? 'bg-ok' : 'bg-crit'}`} aria-hidden />
            <span className={online ? 'text-ok' : 'text-crit'}>{STATIC ? (online ? 'STATIC RESULTS SNAPSHOT' : 'LOADING') : online ? 'SYSTEM ONLINE' : error ? 'BACKEND UNREACHABLE' : 'CONNECTING'}</span>
          </div>
          <div className="flex items-center gap-2 text-[11.5px] font-mono text-ink-2">
            <Activity size={13} aria-hidden className={playing ? 'text-accent' : ''} />
            {STATIC ? 'LIVE ENGINE · RUN LOCALLY' : onReplay ? 'INCIDENT REPLAY · LIVE ENGINE' : mode === 'live' ? (playing ? 'LIVE ENGINE · STREAMING' : 'LIVE ENGINE · PAUSED') : 'ENGINE LOG · SCRUB MODE'}
          </div>
          {system && <div className="hidden lg:block text-[11.5px] font-mono text-ink-3">
            MODEL {modelName(system.model.type).toUpperCase()} · {system.model.parameters.toLocaleString()} params · {system.model.onnx_runtime ? 'ONNX Runtime' : 'PyTorch'} · {system.model.calibration} calibration
          </div>}
          {!STATIC && <button onClick={() => setDemo(!demo)} aria-pressed={demo} disabled={!online}
            className="ml-auto flex items-center gap-1.5 rounded border border-accent/60 px-2 py-1 text-[11.5px] font-semibold text-accent hover:bg-accent/10 disabled:opacity-40"><PlayCircle size={13} aria-hidden />Demo mode</button>}
          <div className={`${STATIC ? 'ml-auto ' : ''}rounded border border-warn/40 bg-warn/5 px-2 py-1 text-[11px] text-warn`}>Simulated failure events · real sensor data</div>
        </header>
        <nav aria-label="Mobile" className="flex gap-1 overflow-x-auto border-b border-line px-2 py-1 md:hidden scroll-thin">
          {NAV.map(({ to, label }) => <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `whitespace-nowrap rounded px-2 py-1 text-xs ${isActive ? 'bg-surface-2 text-ink' : 'text-ink-2'}`}>{label}</NavLink>)}
        </nav>
        {error && !STATIC && <div role="alert" className="mx-5 mt-3 rounded border border-crit/50 bg-crit/10 px-3 py-2 text-sm text-crit">Backend error: {error}. Start it with <code className="font-mono">.venv/bin/uvicorn backend.app:app --port 8000</code></div>}
        <main id="main" className="min-w-0 flex-1 p-4 lg:p-5">{children}</main>
        {online && !STATIC && <DemoGuide />}
      </div>
    </div>
  )
}
