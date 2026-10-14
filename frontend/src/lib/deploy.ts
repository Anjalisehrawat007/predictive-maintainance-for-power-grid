/** Public deployment (Vercel frontend + hosted Python engine). Built with VITE_PUBLIC=1 and
 *  VITE_API_BASE=<engine URL>. The public site never shows ETDataset rows (CC BY-ND 4.0): visitors drive the
 *  real engine with synthetic scenarios or their own uploaded readings. */
export const PUBLIC = import.meta.env.VITE_PUBLIC === '1'
export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '')
export const REPO_URL = 'https://github.com/Anjalisehrawat007/predictive-maintainance-for-power-grid'
