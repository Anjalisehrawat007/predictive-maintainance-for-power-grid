import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { pct, riskPct } from '../lib/format'
import { C, STATUS_HEX } from '../lib/palette'
import { STATUS_META } from '../lib/status'
import type { EngineRecord } from '../lib/types'
import { useModelSpec } from '../store/grid'

const GLYPH: Record<string, string> = { NORMAL: '✓', WARNING: '▲', CRITICAL: '■', OFFLINE: '‖', WARMUP: '…' }

interface Node { id: string; label: string; substation: string; rec?: EngineRecord }

/** Live grid topology: GRID → substations → transformer assets.  Only the assets that exist
 *  in the dataset are drawn (two transformers); nothing is invented. */
export default function Topology({ nodes, selected, onSelect, pulse }: { nodes: Node[]; selected: string; onSelect: (id: string) => void; pulse: number }) {
  const spec = useModelSpec()
  const nav = useNavigate()
  const W = 560, H = 290
  const xs = nodes.map((_, i) => (W / (nodes.length + 1)) * (i + 1))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="group" aria-label="Live grid topology">
      <defs>
        <pattern id="dots" width="14" height="14" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.8" fill="#1c2430" /></pattern>
      </defs>
      <rect width={W} height={H} fill="url(#dots)" />
      {/* grid source */}
      <g transform={`translate(${W / 2},34)`}>
        <rect x={-62} y={-16} width={124} height={32} rx={4} fill={C.surface} stroke={C.line2} />
        <text textAnchor="middle" y={-1} fill={C.ink} fontSize={11} fontWeight={600} fontFamily="IBM Plex Mono">HV GRID FEED</text>
        <text textAnchor="middle" y={11} fill={C.ink3} fontSize={8.5} fontFamily="IBM Plex Mono">source bus</text>
      </g>
      <line x1={xs[0]} x2={xs.at(-1)} y1={92} y2={92} stroke={C.line2} strokeWidth={2} />
      <line x1={W / 2} x2={W / 2} y1={50} y2={92} stroke={C.line2} strokeWidth={2} />
      {nodes.map((n, i) => {
        const st = n.rec?.status ?? 'WARMUP'
        const col = STATUS_HEX[st]
        const r = n.rec?.risk ?? 0
        const isSel = n.id === selected
        const alarm = st === 'WARNING' || st === 'CRITICAL'
        const M = STATUS_META[st]
        return (
          <g key={n.id} transform={`translate(${xs[i]},0)`}>
            {/* energy flow along the feeder: animated only while the asset is online */}
            <line x1={0} x2={0} y1={92} y2={150} stroke={n.rec?.offline ? C.off : C.line2} strokeWidth={2} strokeDasharray={n.rec?.offline ? '4 4' : undefined} />
            {!n.rec?.offline && <motion.circle key={`p${pulse}`} cx={0} r={2.2} fill={C.accent}
              initial={{ cy: 92, opacity: 0.9 }} animate={{ cy: 150, opacity: 0 }} transition={{ duration: 0.6, ease: 'easeIn' }} />}
            <g transform="translate(0,124)">
              <rect x={-52} y={-11} width={104} height={20} rx={3} fill={C.bg} stroke={C.line} />
              <text textAnchor="middle" y={3} fill={C.ink3} fontSize={8.5} fontFamily="IBM Plex Mono">{n.substation.toUpperCase()}</text>
            </g>
            <g transform="translate(0,200)" role="button" tabIndex={0} aria-pressed={isSel}
              aria-label={`${n.label}: ${M.label}, risk ${pct(r)}. Press Enter to select, double-click to open asset view`}
              onClick={() => onSelect(n.id)} onDoubleClick={() => nav(`/assets/${n.id}`)}
              onKeyDown={(e) => { if (e.key === 'Enter') onSelect(n.id); if (e.key === 'o') nav(`/assets/${n.id}`) }} style={{ cursor: 'pointer', outline: 'none' }}>
              {alarm && <motion.circle r={44} fill="none" stroke={col} strokeWidth={1}
                animate={{ r: [40, 52], opacity: [0.6, 0] }} transition={{ duration: 1.8, repeat: Infinity }} />}
              <circle r={40} fill={C.surface} stroke={isSel ? C.ink : C.line2} strokeWidth={isSel ? 1.5 : 1} />
              {/* risk arc: calibrated risk 0–100 % */}
              <motion.circle r={34} fill="none" stroke={col} strokeWidth={4} strokeLinecap="round" transform="rotate(-90)"
                strokeDasharray={2 * Math.PI * 34} animate={{ strokeDashoffset: 2 * Math.PI * 34 * (1 - Math.min(1, Math.max(0.02, r))) }}
                transition={{ type: 'spring', stiffness: 80, damping: 18 }} />
              <circle r={34} fill="none" stroke={C.line} strokeWidth={4} opacity={0.4} />
              <text textAnchor="middle" y={-6} fill={C.ink} fontSize={15} fontWeight={700} fontFamily="IBM Plex Sans">{n.label}</text>
              <text textAnchor="middle" y={10} fill={col} fontSize={10.5} fontWeight={600} fontFamily="IBM Plex Mono">{riskPct(n.rec?.risk ?? null, 1)}</text>
              <text textAnchor="middle" y={22} fill={C.ink3} fontSize={7.5} fontFamily="IBM Plex Mono">RISK ≤ {spec.horizon}</text>
              <g transform="translate(0,58)">
                <rect x={-38} y={-10} width={76} height={18} rx={3} fill={C.bg} stroke={col} />
                <text textAnchor="middle" y={3} fill={col} fontSize={9.5} fontWeight={700} fontFamily="IBM Plex Mono">{GLYPH[st] ?? ''} {M.label}</text>
              </g>
            </g>
          </g>
        )
      })}
    </svg>
  )
}
