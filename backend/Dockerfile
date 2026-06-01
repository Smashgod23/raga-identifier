FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the Deno JS runtime so yt-dlp can run YouTube's EJS challenge-solver
# scripts. This Space runs on a datacenter IP, which YouTube blocks unless the
# JS signature / n-token challenges are solved, and solving them needs a JS
# runtime on PATH. Pinned for reproducible builds; bump DENO_VERSION to update.
ARG DENO_VERSION=v2.1.4
RUN curl -fsSL "https://github.com/denoland/deno/releases/download/${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip" -o /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno \
    && deno --version

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY . .
RUN mkdir -p /app/models /app/data /app/.cache/numba /app/.cache/huggingface /app/.cache/matplotlib /app/.cache/yt-dlp /app/.cache/deno \
    && chown -R 1000:1000 /app
USER 1000
# XDG_CACHE_HOME points yt-dlp's cache (downloaded EJS components) and Deno's
# runtime cache at a writable dir owned by the runtime user.
ENV HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/app/.cache/numba \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    XDG_CACHE_HOME=/app/.cache
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
