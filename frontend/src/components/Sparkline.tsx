/** Minimal SVG sparkline (no axes); nulls break the line. */
export default function Sparkline({ values, color = '#3fb6d6', height = 22, width = 96, max, min = 0, label }: {
  values: (number | null | undefined)[]; color?: string; height?: number; width?: number; max?: number; min?: number; label: string
}) {
  const v = values.map((x) => (x == null || Number.isNaN(x) ? null : x))
  const nums = v.filter((x): x is number => x != null)
  if (nums.length < 2) return <svg width={width} height={height} role="img" aria-label={`${label}: no data`} />
  const hi = max ?? Math.max(...nums), lo = Math.min(min, ...nums)
  const sx = (i: number) => (i / (v.length - 1)) * width
  const sy = (y: number) => height - 2 - ((y - lo) / (hi - lo || 1)) * (height - 4)
  let d = '', pen = false
  v.forEach((y, i) => { if (y == null) { pen = false; return } d += `${pen ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(y).toFixed(1)}`; pen = true })
  return (
    <svg width={width} height={height} role="img" aria-label={`${label} trend, last value ${nums.at(-1)?.toFixed(2)}`}>
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
      <circle cx={sx(v.length - 1)} cy={sy(nums.at(-1)!)} r={2} fill={color} />
    </svg>
  )
}
