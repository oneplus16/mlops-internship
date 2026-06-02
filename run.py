"""
MLOps Batch Job: Rolling Mean Signal Generator
Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    """Configure root logger to emit structured lines to both file and stdout."""
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler — always append so re-runs are traceable
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict[str, Any]:
    """Parse YAML config and validate required fields."""
    logger.info(f"Loading config from: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping.")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required key(s): {sorted(missing)}")

    # Type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"Config 'version' must be a non-empty string, got: {cfg['version']!r}")

    logger.info(
        f"Config loaded — version={cfg['version']}  seed={cfg['seed']}  window={cfg['window']}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Read CSV, validate structure and content."""
    logger.info(f"Loading dataset from: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (zero rows after header).")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    # Coerce 'close' to numeric; non-parseable values become NaN
    original_len = len(df)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    bad_rows = df["close"].isna().sum()
    if bad_rows:
        logger.warning(f"{bad_rows} row(s) with non-numeric 'close' values will be dropped.")
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    logger.info(
        f"Dataset loaded — {len(df)} usable rows "
        f"(dropped {original_len - len(df)} bad rows)"
    )
    logger.debug(f"Columns: {list(df.columns)}")
    logger.debug(f"close  min={df['close'].min():.4f}  max={df['close'].max():.4f}")
    return df


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_rolling_mean(series: pd.Series, window: int, logger: logging.Logger) -> pd.Series:
    """
    Compute rolling mean with min_periods=window so the first (window-1) rows
    produce NaN.  Those rows are excluded from signal computation downstream.
    """
    logger.info(f"Computing rolling mean — window={window}")
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    nan_count = rolling_mean.isna().sum()
    logger.debug(
        f"Rolling mean computed — {nan_count} warm-up NaN rows (excluded from signal)"
    )
    return rolling_mean


def compute_signal(close: pd.Series, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    Binary signal: 1 if close > rolling_mean else 0.
    Rows where rolling_mean is NaN (warm-up period) are set to NaN here;
    they are filled with 0 when computing signal_rate so all rows are counted.
    """
    logger.info("Computing binary signal (close > rolling_mean → 1, else 0)")
    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=close.index)
    signal[valid_mask] = (close[valid_mask] > rolling_mean[valid_mask]).astype(int)

    valid_signals = signal.dropna()
    ones = int(valid_signals.sum())
    zeros = len(valid_signals) - ones
    logger.debug(f"Signal distribution — 1s: {ones}  0s: {zeros}  NaN (warm-up): {(~valid_mask).sum()}")
    return signal


# ---------------------------------------------------------------------------
# Metrics writing
# ---------------------------------------------------------------------------

def write_metrics(
    output_path: str,
    version: str,
    rows_processed: int,
    signal_rate: float,
    latency_ms: float,
    seed: int,
    logger: logging.Logger,
) -> None:
    metrics = {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": round(latency_ms, 2),
        "seed": seed,
        "status": "success",
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics written to {output_path}: {metrics}")


def write_error_metrics(
    output_path: str,
    error_message: str,
    logger: logging.Logger,
) -> None:
    metrics = {
        "status": "error",
        "error": error_message,
    }
    try:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass  # Best-effort; don't mask the original error
    logger.error(f"Error metrics written to {output_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps batch job: rolling mean signal generator"
    )
    parser.add_argument("--input",    required=True, help="Path to input CSV (OHLCV)")
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument("--output",   required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path to log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("MLOps Batch Job — START")
    logger.info(f"Args: input={args.input}  config={args.config}  "
                f"output={args.output}  log_file={args.log_file}")

    t_start = time.perf_counter()

    try:
        # 1. Load & validate config
        cfg = load_config(args.config, logger)
        seed: int    = cfg["seed"]
        window: int  = cfg["window"]
        version: str = cfg["version"]

        # Set global random seed for reproducibility
        np.random.seed(seed)
        logger.info(f"Random seed set: {seed}")

        # 2. Load & validate dataset
        df = load_dataset(args.input, logger)

        # 3. Rolling mean (NaN for first window-1 rows)
        df["rolling_mean"] = compute_rolling_mean(df["close"], window, logger)

        # 4. Binary signal
        df["signal"] = compute_signal(df["close"], df["rolling_mean"], logger)

        # 5. Compute metrics
        # rows_processed = total rows in dataset (all rows, including warm-up)
        # signal_rate = mean over all rows; warm-up NaN signals treated as 0
        rows_processed = len(df)
        signal_filled  = df["signal"].fillna(0)
        signal_rate    = float(signal_filled.mean())

        t_end      = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        logger.info(
            f"Processing complete — rows_processed={rows_processed}  "
            f"signal_rate={signal_rate:.4f}  latency_ms={latency_ms:.2f}"
        )

        # 6. Write metrics
        write_metrics(
            output_path=args.output,
            version=version,
            rows_processed=rows_processed,
            signal_rate=signal_rate,
            latency_ms=latency_ms,
            seed=seed,
            logger=logger,
        )

    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Job failed: {exc}", exc_info=True)
        write_error_metrics(args.output, str(exc), logger)
        sys.exit(1)

    except Exception as exc:
        logger.critical(f"Unexpected error: {exc}", exc_info=True)
        write_error_metrics(args.output, f"Unexpected error: {exc}", logger)
        sys.exit(2)

    finally:
        logger.info("MLOps Batch Job — END")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
