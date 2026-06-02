# MLOps Batch Job — Rolling Mean Signal Generator

A reproducible, observable, Dockerized batch pipeline that reads OHLCV data,
computes a rolling-mean signal, and emits structured metrics + detailed logs.

---

## Project layout

```
mlops_job/
├── run.py            # Main entry point
├── config.yaml       # Job configuration
├── data.csv          # Input OHLCV data (10 000 rows)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Two-stage Docker build
└── README.md
```

---

## Quick start — local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the job
python run.py \
  --input  data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log

# 3. Inspect results
cat metrics.json
cat run.log
```

---

## Quick start — Docker (one command)

```bash
# Build
docker build -t mlops-job .

# Run (outputs stay inside the container)
docker run --rm mlops-job

# Run with host-mounted output directory
mkdir -p out
docker run --rm \
  -v "$(pwd)/data.csv:/app/data.csv:ro" \
  -v "$(pwd)/config.yaml:/app/config.yaml:ro" \
  -v "$(pwd)/out:/app/out" \
  mlops-job \
  --input   data.csv \
  --config  config.yaml \
  --output  out/metrics.json \
  --log-file out/run.log
```

---

## Config (`config.yaml`)

| Key       | Type   | Description                              |
|-----------|--------|------------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility    |
| `window`  | int    | Rolling-mean window size (rows)          |
| `version` | string | Pipeline version tag (written to output) |

---

## Output — `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4990,
  "latency_ms": 127.34,
  "seed": 42,
  "status": "success"
}
```

> **`rows_processed`** counts only rows *after* the warm-up period
> (first `window − 1` rows produce NaN rolling-mean and are excluded).

---

## Error handling

The job exits with a non-zero code and writes an error record on:

| Condition                       | Exit code |
|---------------------------------|-----------|
| Missing input / config file     | 1         |
| Invalid CSV / missing `close`   | 1         |
| Empty dataset                   | 1         |
| Invalid config structure/types  | 1         |
| Unexpected runtime error        | 2         |

Error `metrics.json`:
```json
{ "status": "error", "error": "Input file not found: data.csv" }
```

---

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Warm-up rows | Excluded via `min_periods=window` | Avoids biasing `signal_rate` with partial-window means |
| Seed scope | `numpy.random.seed` at startup | Deterministic for any downstream stochastic ops |
| Log format | Timestamped, leveled, human-readable | Works with `grep`, log aggregators (Datadog, Loki) |
| Metrics format | Machine-readable JSON, exact keys | Drop-in compatible with dashboards / downstream jobs |
| Docker build | Two-stage (builder + slim runtime) | Minimal image size, no build tools in prod layer |
