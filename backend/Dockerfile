FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY . .
RUN mkdir -p /app/models /app/data /app/.cache/numba /app/.cache/huggingface /app/.cache/matplotlib \
    && chown -R 1000:1000 /app
USER 1000
ENV HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/app/.cache/numba \
    MPLCONFIGDIR=/app/.cache/matplotlib
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
