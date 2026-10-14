# SmartGrid AI — operations frontend

React 19 + TypeScript + Vite + Tailwind CSS v4 + Framer Motion + Recharts + Zustand.
It is a thin client over the Python backend (`backend/app.py`); **no model logic lives here**
and every number shown comes from an API response (dataset, model output, measured benchmark,
or a labelled derived metric).

```bash
npm install
npm run dev        # http://localhost:5173  (proxies /api → http://127.0.0.1:8000)
npm run build      # production bundle in dist/ — served by the FastAPI app at http://127.0.0.1:8000
```

| Route | View |
|---|---|
| `/` | Command center — topology, hero intelligence, synchronized risk/sensor charts, global time scrubber, live stream (Stream/Step/Replay −6h, 1×/2×/4×), event feed, pipeline timing |
| `/replay` | Incident replay — replays a held-out failure through the real streaming engine; measured lead time |
| `/assets/:id` | Asset explorer — current state, sensors, health, risk history, failures, warning & lead-time history |
| `/alerts` | Alert center — filters; Investigate / Replay / Acknowledge (persisted by the backend) |
| `/failures` | Failure intelligence — detectability, lead times, false-alarm anatomy, misses, seasonal & asset performance |
| `/models` | Model lab — metric selector, ranked bars, performance-vs-computation scatter, Phase-2 multi-seed results |
| `/methodology` | Model & methodology — pipeline, leakage checks, reproduction, calibration, simulation disclosure, limits |

State: `src/store/grid.ts` holds one shared clock (cursor) for all synchronized views. In
*engine-log* mode the scrubber reads precomputed engine outputs; pressing **Stream**/**Step**
creates a backend session at the cursor and advances the real engine.
