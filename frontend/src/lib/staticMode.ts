/** Static "results-only" build (deployed site, e.g. Vercel): no Python backend, no live engine.
 *  Built with VITE_STATIC=1; data come from /data/*.json exported by `backend/export_results.py`,
 *  which contains aggregate results only — never sensor rows (ETDataset is CC BY-ND 4.0). */
export const STATIC = import.meta.env.VITE_STATIC === '1'
export const REPO_URL = 'https://github.com/Anjalisehrawat007/predictive-maintainance-for-power-grid'

const STATIC_FILES: Record<string, string> = {
  '/api/system': 'system', '/api/models': 'models', '/api/analytics/failures': 'failures',
  '/api/methodology': 'methodology', '/api/incidents': 'incidents', '/api/patterns': 'patterns',
}
/** Map an API path to its exported JSON file, or null when it needs the live engine. */
export const staticUrl = (path: string) => {
  const f = STATIC_FILES[path.split('?')[0]]
  return f ? `/data/${f}.json` : null
}
