# ── Stage 1: builder (installs deps) ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into a clean layer
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY run.py .

# Default data/config/output paths (override at runtime via -v + --input etc.)
# Keeps the image self-contained for quick local testing with COPY below.
COPY data.csv .
COPY config.yaml .

# Run as non-root for security
RUN useradd -m jobuser
USER jobuser

# ─────────────────────────────────────────────────────────────────────────────
# Default entrypoint — all paths can be overridden at `docker run` time:
#   docker run --rm \
#     -v $(pwd)/data.csv:/app/data.csv:ro \
#     -v $(pwd)/config.yaml:/app/config.yaml:ro \
#     -v $(pwd)/out:/app/out \
#     mlops-job \
#     --input data.csv --config config.yaml \
#     --output out/metrics.json --log-file out/run.log
# ─────────────────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "run.py"]
CMD ["--input", "data.csv", "--config", "config.yaml", \
     "--output", "metrics.json", "--log-file", "run.log"]
