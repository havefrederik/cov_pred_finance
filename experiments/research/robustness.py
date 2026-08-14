from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HAC_LAGS = (5, 21, 63)
DEFAULT_TRIM_FRACTIONS = (0.01, 0.05)


def newey_west_se(values, lag=21):
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 2:
        return np.nan
    centered = x - x.mean()
    long_run_variance = float(centered @ centered) / n
    max_lag = min(int(lag), n - 1)
    for j in range(1, max_lag + 1):
        covariance = float(centered[j:] @ centered[:-j]) / n
        weight = 1.0 - j / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    return float(np.sqrt(long_run_variance / n))


def trimmed_mean(values, fraction):
    x = np.sort(pd.Series(values, dtype=float).dropna().to_numpy())
    if len(x) == 0:
        return np.nan
    trim = int(np.floor(float(fraction) * len(x)))
    if trim == 0:
        return float(x.mean())
    if 2 * trim >= len(x):
        return np.nan
    return float(x[trim:-trim].mean())


def top_positive_contribution_fraction(values, top_n=10):
    x = pd.Series(values, dtype=float).dropna()
    total = float(x.sum())
    if total <= 0:
        return np.nan
    positives = x[x > 0].sort_values(ascending=False)
    return float(positives.iloc[:top_n].sum() / total)


def summarize_improvement(improvement, hac_lags=DEFAULT_HAC_LAGS):
    x = pd.Series(improvement, dtype=float).dropna()
    mean = float(x.mean())
    result = {
        "n_obs": int(len(x)),
        "mean": mean,
        "median": float(x.median()),
        "standard_deviation": float(x.std()),
        "positive_fraction": float((x > 0).mean()),
        "top_10_positive_contribution_fraction": (
            top_positive_contribution_fraction(x, top_n=10)
        ),
        "yearly_mean": {
            str(year): float(value)
            for year, value in x.groupby(x.index.year).mean().items()
        },
    }
    for fraction in DEFAULT_TRIM_FRACTIONS:
        result[f"trimmed_mean_{int(100 * fraction)}pct"] = trimmed_mean(
            x,
            fraction,
        )
    for lag in hac_lags:
        standard_error = newey_west_se(x, lag=lag)
        result[f"hac_se_{lag}"] = standard_error
        result[f"hac_t_{lag}"] = (
            float(mean / standard_error)
            if np.isfinite(standard_error) and standard_error > 0
            else np.nan
        )
    return result


def model_keys(daily):
    suffix = "__nll"
    return sorted(
        column[: -len(suffix)]
        for column in daily.columns
        if column.endswith(suffix)
    )


def analyze_walkforward(daily, summary):
    start = pd.Timestamp(summary["oos_start"])
    frame = daily.loc[start:].copy()
    baseline_nll = frame["johansson_nll"]
    baseline_gmv = frame["johansson_gmv_square"]

    models = {}
    for key in model_keys(frame):
        nll_improvement = baseline_nll - frame[f"{key}__nll"]
        gmv_improvement = baseline_gmv - frame[f"{key}__gmv_square"]
        entry = {
            "nll_improvement": summarize_improvement(nll_improvement),
            "gmv_square_improvement": summarize_improvement(gmv_improvement),
        }
        mean_baseline_gmv = float(baseline_gmv.mean())
        entry["gmv_square_percent_improvement"] = (
            100.0 * float(gmv_improvement.mean()) / mean_baseline_gmv
            if mean_baseline_gmv > 0
            else np.nan
        )
        models[key] = entry

    ablations = {}
    for model_name in ("ridge", "random_forest"):
        base_key = f"{model_name}__johansson"
        if base_key not in models:
            continue
        base_mean = models[base_key]["nll_improvement"]["mean"]
        ablations[model_name] = {}
        for feature_name in (
            "johansson_market",
            "johansson_spectral",
            "johansson_market_spectral",
            "all",
        ):
            key = f"{model_name}__{feature_name}"
            if key not in models:
                continue
            ablations[model_name][feature_name] = {
                "incremental_mean_nll_improvement_vs_johansson_features": (
                    models[key]["nll_improvement"]["mean"] - base_mean
                )
            }

    return {
        "oos_start": str(start.date()),
        "oos_end": str(frame.index.max().date()),
        "johansson_mean_nll": float(baseline_nll.mean()),
        "johansson_mean_gmv_square": float(baseline_gmv.mean()),
        "models": models,
        "feature_ablations": ablations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Robustness diagnostics for walk-forward covariance results"
    )
    parser.add_argument(
        "--daily",
        type=Path,
        default=Path("experiments/research/results/walkforward_daily.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/research/results/walkforward_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/research/results/walkforward_robustness.json"),
    )
    args = parser.parse_args()

    daily = pd.read_csv(args.daily, index_col=0, parse_dates=True)
    summary = json.loads(args.summary.read_text())
    result = analyze_walkforward(daily, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
