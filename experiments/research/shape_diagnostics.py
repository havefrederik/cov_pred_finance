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
from experiments.research.robustness import summarize_improvement
from experiments.research.walkforward_shape import build_walkforward_features

SHAPE_FEATURES = (
    "temprel__1d__log_shape_std",
    "temprel__1d__log_condition",
    "temprel__5d__log_shape_std",
    "temprel__5d__log_condition",
    "temprel__21d__log_shape_std",
    "temprel__21d__log_condition",
    "fastslow__log_shape_std",
    "fastslow__log_condition",
    "relgeom__mean_shape_loss",
    "relgeom__max_shape_loss",
    "cov_top_subspace_rotation",
)

COMPARISONS = {
    "ridge_relshape_vs_johansson": (
        "ridge__johansson",
        "ridge__johansson_relshape",
    ),
    "ridge_market_relshape_vs_market": (
        "ridge__johansson_market",
        "ridge__johansson_market_relshape",
    ),
    "random_forest_relshape_vs_johansson": (
        "random_forest__johansson",
        "random_forest__johansson_relshape",
    ),
    "random_forest_market_relshape_vs_market": (
        "random_forest__johansson_market",
        "random_forest__johansson_market_relshape",
    ),
}


def _next_state(returns):
    future = returns.shift(-1)
    out = pd.DataFrame(index=returns.index)
    out["next_market_return"] = future.mean(axis=1)
    out["next_abs_market_return"] = out["next_market_return"].abs()
    out["next_dispersion"] = future.std(axis=1)
    out["next_max_abs_stock_return"] = future.abs().max(axis=1)
    return out


def _quintiles(frame, target):
    output = {}
    for feature in SHAPE_FEATURES:
        if feature not in frame:
            continue
        valid = frame[[feature, target]].dropna()
        if valid.empty:
            continue
        buckets = pd.qcut(valid[feature], 5, labels=False, duplicates="drop")
        output[feature] = {
            str(int(bucket) + 1): {
                "n_obs": int(len(values)),
                "mean_incremental_nll_improvement": float(values.mean()),
                "median_incremental_nll_improvement": float(values.median()),
            }
            for bucket, values in valid.groupby(buckets, observed=True)[target]
        }
    return output


def _outcome_checks(frame, target):
    output = {}
    for column in (
        "next_abs_market_return",
        "next_dispersion",
        "next_max_abs_stock_return",
    ):
        valid = frame[[target, column]].dropna()
        output[f"corr_with_{column}"] = float(valid[target].corr(valid[column]))
        for quantile in (0.95, 0.99):
            cutoff = valid[column].quantile(quantile)
            calm = valid.loc[valid[column] <= cutoff, target]
            output[f"mean_below_{int(100 * quantile)}pct_{column}"] = float(
                calm.mean()
            )
    return output


def run(returns, factors, daily):
    experts = build_iewma_experts(returns)
    cm, _ = build_cm_iewma(returns, experts)
    phase0, _ = run_phase0(returns, factors)
    features, feature_sets = build_walkforward_features(phase0, experts, cm)
    next_state = _next_state(returns)

    report = {
        "shape_feature_count": int(
            len(feature_sets["johansson_relshape"])
            - len(feature_sets["johansson"])
        ),
        "comparisons": {},
    }

    for name, (base, augmented) in COMPARISONS.items():
        base_col = f"{base}__nll"
        aug_col = f"{augmented}__nll"
        if base_col not in daily or aug_col not in daily:
            continue
        delta = (daily[base_col] - daily[aug_col]).dropna()
        report["comparisons"][name] = summarize_improvement(delta)

    focus_name = "random_forest_market_relshape_vs_market"
    base, augmented = COMPARISONS[focus_name]
    delta = (daily[f"{base}__nll"] - daily[f"{augmented}__nll"]).rename(
        "incremental_nll_improvement"
    )
    index = delta.dropna().index.intersection(features.index).intersection(
        next_state.index
    )
    frame = pd.DataFrame(index=index)
    frame["incremental_nll_improvement"] = delta.reindex(index)
    frame = frame.join(
        features[[column for column in SHAPE_FEATURES if column in features]]
    )
    frame = frame.join(next_state)

    report["focus"] = {
        "comparison": focus_name,
        "shape_feature_quintiles": _quintiles(
            frame,
            "incremental_nll_improvement",
        ),
        "outcome_checks": _outcome_checks(
            frame,
            "incremental_nll_improvement",
        ),
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostics for scale-invariant relative-shape features"
    )
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
            "experiments/research/results/walkforward_shape_diagnostics.json"
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
