---
title: SmartGrid AI Engine
emoji: ⚡
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Live engine (FastAPI) for the SmartGrid AI predictive-maintenance demo
---

# SmartGrid AI — live engine

FastAPI backend of [SmartGrid AI](https://github.com/Anjalisehrawat007/predictive-maintainance-for-power-grid):
GRU + temporal attention early-warning model, isotonic calibration, seed-ensemble confidence, integrated-gradients
attribution and the streaming alert engine. The web app lives at
<https://predictive-maintenance-power-grid.vercel.app>.

Runs in **public mode** (`SGPM_PUBLIC=1`). No ETDataset rows are on this server; visitors drive the engine with
synthetic fault scenarios (`POST /api/scenarios`) or their own readings (`POST /api/uploads`). Trained weights are
fetched at startup from a private model repository (Space secret `HF_TOKEN`, variable `SGPM_MODEL_REPO`).
Interactive API docs: `/docs`.
