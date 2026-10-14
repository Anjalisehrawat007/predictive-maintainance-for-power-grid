# SmartGrid AI engine for a Hugging Face Docker Space (public mode: no dataset rows on the server).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SGPM_PUBLIC=1 OMP_NUM_THREADS=2
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY --chown=user . .
USER user
EXPOSE 7860
# weights come from the private model repo at start-up (never baked into this public image)
CMD ["bash", "-c", "python -m backend.fetch_models && uvicorn backend.app:app --host 0.0.0.0 --port 7860"]
