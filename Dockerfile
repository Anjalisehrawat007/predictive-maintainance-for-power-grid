# SmartGrid AI — one image: FastAPI backend serving the built React frontend.
# NOTE: written for reproducibility; Docker was not available on the development machine,
# so this file has not been built there (see reports/final_project_quality_report.md).

# ---------- 1. build the frontend ----------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- 2. Python runtime ----------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch wheel keeps the image small
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
COPY --from=web /web/dist ./frontend/dist
# Same as `make data`: download ETDataset (CC BY-ND 4.0, not redistributed in this repo), regenerate the seeded
# failure simulation, train the deployed models if no weights are present (~20 min), replay the held-out period.
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*
RUN make data PY=python
RUN useradd --create-home app && chown -R app /app/reports/phase2
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/ready').status==200 else 1)"
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
