PY ?= .venv/bin/python
# Python 3.11 is the tested version; falls back to python3 if a working python3.11 is not on PATH
PYTHON ?= $(shell python3.11 -c '' >/dev/null 2>&1 && echo python3.11 || echo python3)

# ---------------- quick start ----------------
.PHONY: setup data models demo api web test check deploy-engine deploy-web
setup:      ; $(PYTHON) -m venv .venv && $(PY) -m pip install -q -r requirements-dev.txt && cd frontend && npm ci
# data: download ETDataset (CC BY-ND 4.0, not redistributed here) → regenerate the seeded failure simulation →
#       train the deployed models if no weights are present (`make models`) → replay the held-out period through
#       the engine (engine log for the UI timeline). Deterministic.
data:       ; $(PY) -c "from src.utils.config import load_config; from src.data.dataset import build_processed; build_processed(load_config(), force=True)" \
              && { [ -f models/best_attention_model.pt ] || $(MAKE) models; } && $(PY) -m backend.precompute
# models: retrain the deployed GRU + attention, the 3-seed ensemble and the preprocessing from the committed
#         configuration (experiments/best_config.json; no hyper-parameter search), then export ONNX. ~20 min on a laptop CPU.
models:     ; $(PY) train.py --stage final && $(PY) train.py --stage ensemble \
              && $(PY) -c "from src.utils.config import load_config, resolve; from src.streaming.artifacts import load_artifacts; from src.evaluation.report import onnx_latency; cfg = load_config(); a = load_artifacts(cfg); assert onnx_latency(a.model, a.seq_len, len(a.feature_names), resolve(cfg, 'models_dir') / f\"{a.meta['active_model']}.onnx\") is not None, 'ONNX export failed'"
demo:       ; cd frontend && npm run build && cd .. && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
api:        ; .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
web:        ; cd frontend && npm run dev
# public deployment: engine → Hugging Face Space (private weights repo), web → Vercel. See README "Live website".
deploy-engine: ; $(PY) deploy/publish_engine.py
deploy-web:    ; cd frontend && VITE_PUBLIC=1 VITE_API_BASE=$(API_BASE) npx vite build --outDir ../.deploy-web --emptyOutDir \
                 && cp ../deploy/vercel.json ../.deploy-web/ && cd ../.deploy-web && vercel deploy --prod --yes
test:       ; $(PY) -m pytest -q tests && cd frontend && npm test
check:      ; $(PY) -m pytest -q tests && .venv/bin/ruff check . && .venv/bin/pip-audit -r requirements.txt \
              && cd frontend && npm run typecheck && npm run lint && npm test && npm run build && npm audit

# ---------------- research pipeline (optional; artefacts are committed) ----------------
.PHONY: eda train evaluate stream final precompute benchmark dashboard
eda:        ; $(PY) eda.py
train:      ; $(PY) train.py --stage all
evaluate:   ; $(PY) evaluate.py
stream:     ; $(PY) stream.py --asset ETTm1 && $(PY) stream.py --asset ETTm2
final:      ; $(PY) -m src.final.protocol && $(PY) -m src.final.operating_point && $(PY) -m src.final.final_eval \
              && $(PY) -m src.final.robustness && $(PY) -m src.final.cross_asset
precompute: ; $(PY) -m backend.precompute
benchmark:  ; $(PY) -m src.final.benchmark
dashboard:  ; .venv/bin/streamlit run app.py
