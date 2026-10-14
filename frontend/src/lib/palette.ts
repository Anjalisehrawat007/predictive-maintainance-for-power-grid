// Hex constants for SVG/Recharts (CSS variables are not reliable in SVG presentation attributes).
export const C = {
  bg: '#07090c', surface: '#0c1016', line: '#1c2430', line2: '#283242', ink: '#e6ebf2', ink2: '#a3adbb', ink3: '#6b7685',
  accent: '#3fb6d6', ok: '#0ca30c', warn: '#fab219', crit: '#d03b3b', off: '#6b7685',
  s: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#008300', '#e66767'],
}
export const SENSOR_COLOR: Record<string, string> = {
  OT: C.s[1], HUFL: C.s[0], HULL: C.s[2], MUFL: C.s[5], MULL: C.s[4], LUFL: C.s[3], LULL: '#7f8a99',
}
export const SENSOR_NAME: Record<string, string> = {
  OT: 'Oil temperature (°C)', HUFL: 'HV useful load', HULL: 'HV reactive load', MUFL: 'MV useful load',
  MULL: 'MV reactive load', LUFL: 'LV useful load', LULL: 'LV reactive load',
}
export const TYPE_COLOR: Record<string, string> = {
  thermal_runaway: C.s[0], load_oscillation: C.s[1], intermittent_arcing: C.s[2], sudden: C.s[3],
}
export const STATUS_HEX: Record<string, string> = { NORMAL: C.ok, WARNING: C.warn, CRITICAL: C.crit, OFFLINE: C.off, WARMUP: C.off }
export const tooltipStyle = {
  contentStyle: { background: '#0c1016', border: `1px solid ${C.line2}`, borderRadius: 6, fontSize: 12, fontFamily: 'IBM Plex Mono, monospace' },
  labelStyle: { color: C.ink2 }, itemStyle: { padding: 0 },
}
