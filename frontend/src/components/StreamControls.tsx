import { Pause, Play, RotateCcw, StepForward } from 'lucide-react'
import { useGrid, type Speed } from '../store/grid'

export default function StreamControls() {
  const { playing, play, pause, stepOnce, replayBack, speed, setSpeed, mode, session } = useGrid()
  const btn = 'flex items-center gap-1.5 rounded border border-line-2 bg-surface-2 px-2.5 py-1.5 text-[12px] hover:border-ink-3 disabled:opacity-40'
  return (
    <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Stream controls">
      {playing
        ? <button className={btn} onClick={pause} aria-label="Pause live stream"><Pause size={14} />Pause</button>
        : <button className={`${btn} border-accent/60 text-accent`} onClick={play} aria-label="Start live streaming through the model"><Play size={14} />Stream</button>}
      <button className={btn} onClick={stepOnce} aria-label="Process one observation"><StepForward size={14} />Step</button>
      <button className={btn} onClick={replayBack} aria-label="Rewind 6 hours and replay through the live engine"><RotateCcw size={14} />Replay −6h</button>
      <div className="flex overflow-hidden rounded border border-line-2" role="radiogroup" aria-label="Playback speed">
        {([1, 2, 4] as Speed[]).map((s) => (
          <button key={s} role="radio" aria-checked={speed === s} onClick={() => setSpeed(s)}
            className={`px-2.5 py-1.5 font-mono text-[12px] ${speed === s ? 'bg-ink text-bg' : 'bg-surface-2 text-ink-2 hover:text-ink'}`}>{s}×</button>
        ))}
      </div>
      <span className="font-mono text-[10.5px] text-ink-3">{speed * 2} obs/s · {speed * 30} min of grid time per second{mode === 'live' && session ? ` · session ${session.id}` : ''}</span>
    </div>
  )
}
