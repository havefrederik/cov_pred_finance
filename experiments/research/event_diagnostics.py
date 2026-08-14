from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.research.phase0 import (
    _load,
    build_cm_iewma,
    build_iewma_experts,
    run_phase0,
)
from experiments.research.walkforward import build_walkforward_features

MODELS = (
    "ridge__johansson_market",
    "random_forest__johansson",
    "random_forest__johansson_spectral",
    "random_forest__johansson_market_spectral",
)
FEATURES = (
    "cov_log_condition",
    "cov_leading_eigen_share",
    "cov_effective_rank",
    "cov_log_eigen_dispersion",
    "cov_top_subspace_rotation",
    "relgeom__mean_shape_loss",
    "relgeom__max_shape_loss",
)


def _next_state(returns):
    future = returns.shift(-1)
    out = pd.DataFrame(index=returns.index)
    out["next_market_return"] = future.mean(axis=1)
    out["next_abs_market_return"] = out["next_market_return"].abs()
    out["next_dispersion"] = future.std(axis=1)
    out["next_max_abs_stock_return"] = future.abs().max(axis=1)
    return out


def _event_records(frame, count=15):
    columns = [
        "nll_improvement",
        "gmv_improvement",
        "eta",
        "next_market_return",
        "next_abs_market_return",
        "next_dispersion",
        "next_max_abs_stock_return",
    ]
    columns += [column for column in FEATURES if column in frame]

    def records(sample):
        sample = sample[columns].copy()
        sample.insert(0, "forecast_date", sample.index.astype(str))
        return sample.replace({np.nan: None}).to_dict(orient="records")

    return {
        "largest_gains": records(frame.nlargest(count, "nll_improvement")),
        "largest_losses": records(frame.nsmallest(count, "nll_improvement")),
    }


def _quintiles(frame):
    output = {}
    for feature in FEATURES:
        if feature not in frame:
            continue
        valid = frame[[feature, "nll_improvement"]].dropna()
        buckets = pd.qcut(valid[feature], 5, labels=False, duplicates="drop")
        output[feature] = {
            str(int(bucket) + 1): {
                "n_obs": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
            }
            for bucket, values in valid.groupby(buckets)["nll_improvement"]
        }
    return output


def _outcome_checks(frame):
    output = {}
    for column in (
        "next_abs_market_return",
        "next_dispersion",
        "next_max_abs_stock_return",
    ):
        valid = frame[["nll_improvement", column]].dropna()
        output[f"corr_with_{column}"] = float(
            valid["nll_improvement"].corr(valid[column])
        )
        for quantile in (0.95, 0.99):
            cutoff = valid[column].quantile(quantile)
            calm = valid.loc[valid[column] <= cutoff, "nll_improvement"]
            key = f"mean_below_{int(100 * quantile)}pct_{column}"
            output[key] = float(calm.mean())
    return output


def run(returns, factors, daily):
    experts = build_iewma_experts(returns)
    cm, _ = build_cm_iewma(returns, experts)
    phase0, _ = run_phase0(returns, factors)
    features, _ = build_walkforward_features(phase0, experts, cm)
    next_state = _next_state(returns)

    report = {"models": {}, "incremental_spectral_value": {}}
    for model in MODELS:
        index = daily.index.intersection(features.index).intersection(next_state.index)
        frame = pd.DataFrame(index=index)
        frame["nll_improvement"] = (
            daily.loc[index, "johansson_nll"] - daily.loc[index, f"{model}__nll"]
        )
        frame["gmv_improvement"] = (
            daily.loc[index, "johansson_gmv_square"]
            - daily.loc[index, f"{model}__gmv_square"]
        )
        frame["eta"] = daily.loc[index, f"{model}__eta"]
        frame = frame.join(features[list(FEATURES)])
        frame = frame.join(next_state).dropna(subset=["nll_improvement"])
        report["models"][model] = {
            "events": _event_records(frame),
            "state_quintiles": _quintiles(frame),
            "outcome_checks": _outcome_checks(frame),
        }

    comparisons = {
        "spectral_vs_johansson": (
            "random_forest__johansson",
            "random_forest__johansson_spectral",
        ),
        "market_spectral_vs_market": (
            "random_forest__johansson_market",
            "random_forest__johansson_market_spectral",
        ),
    }
    for name, (base, augmented) in comparisons.items():
        delta = (daily[f"{base}__nll"] - daily[f"{augmented}__nll"]).dropna()
        report["incremental_spectral_value"][name] = {
            "mean": float(delta.mean()),
            "median": float(delta.median()),
            "positive_fraction": float((delta > 0).mean()),
            "top_dates": {
                str(date.date()): float(value)
                for date, value in delta.nlargest(10).items()
            },
            "bottom_dates": {
                str(date.date()): float(value)
                for date, value in delta.nsmallest(10).items()
            },
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--returns",
        type=Path,
        default=Path("experiments/data/SP500_top25_adjusted.csv"),
    )
    parser.add_argument(
        "--factors",
        type=Path,
        default=Path("experiments/data/ff5_no_rf.csv"),
    )
    parser.add_argument(
        "--daily",
        type=Path,
        default=Path("experiments/research/results/walkforward_daily.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/research/results/walkforward_event_diagnostics.json"
        ),
    )
    args = parser.parse_args()

    returns = _load(args.returns)
    factors = (
        _load(args.factors).reindex(returns.index).dropna()
        if args.factors.exists()
        else None
    )
    daily = pd.read_csv(args.daily, index_col=0, parse_dates=True)
    report = run(returns, factors, daily)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
