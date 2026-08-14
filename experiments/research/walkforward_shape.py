from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

import experiments.research.walkforward as base
from experiments.research.phase0 import _arr, _load

_BASE_BUILD_FEATURES = base.build_walkforward_features


def _relative_geometry(current, reference):
    """Return log-scale and scale-invariant shape statistics for two SPD matrices."""
    try:
        eta = eigh(_arr(current), _arr(reference), eigvals_only=True)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    if np.any(~np.isfinite(eta)) or np.any(eta <= 0):
        return np.nan, np.nan, np.nan
    log_eta = np.log(eta)
    return (
        float(log_eta.mean()),
        float(log_eta.std()),
        float(log_eta.max() - log_eta.min()),
    )


def temporal_relative_geometry_features(covs, horizons=(1, 5, 21)):
    """Measure covariance deformation relative to its own lagged geometry.

    If Sigma_t = c * Sigma_{t-h}, log_shape_std and log_condition are zero,
    so the shape measures are exactly invariant to a common volatility scaling.
    """
    times = sorted(covs)
    rows = {time: {} for time in times}
    for i, time in enumerate(times):
        for horizon in horizons:
            prefix = f"temprel__{horizon}d"
            if i < horizon:
                scale = shape_std = condition = np.nan
            else:
                scale, shape_std, condition = _relative_geometry(
                    covs[time],
                    covs[times[i - horizon]],
                )
            rows[time][f"{prefix}__log_scale"] = scale
            rows[time][f"{prefix}__log_shape_std"] = shape_std
            rows[time][f"{prefix}__log_condition"] = condition
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def fast_slow_relative_geometry_features(experts):
    """Directly compare the fastest and slowest IEWMA expert geometries."""
    fast_name = base.EXPERT_NAMES[0]
    slow_name = base.EXPERT_NAMES[-1]
    fast, slow = experts[fast_name], experts[slow_name]
    rows = {}
    for time in sorted(set(fast).intersection(slow)):
        scale, shape_std, condition = _relative_geometry(
            fast[time],
            slow[time],
        )
        rows[time] = {
            "fastslow__log_scale": scale,
            "fastslow__log_shape_std": shape_std,
            "fastslow__log_condition": condition,
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def build_walkforward_features(phase0, experts, cm):
    """Extend the run-5 features with a clean, scale-invariant shape block."""
    x, original_sets = _BASE_BUILD_FEATURES(phase0, experts, cm)

    temporal = temporal_relative_geometry_features(cm).reindex(x.index)
    x = x.join(temporal)
    temporal_shape = [
        column
        for column in temporal.columns
        if column.endswith("__log_shape_std")
        or column.endswith("__log_condition")
    ]
    temporal_changes = base._difference_features(
        x,
        temporal_shape,
        horizons=(1, 5),
    )
    x = x.join(temporal_changes)

    fastslow = fast_slow_relative_geometry_features(experts).reindex(x.index)
    x = x.join(fastslow)
    fastslow_shape = [
        "fastslow__log_shape_std",
        "fastslow__log_condition",
    ]
    fastslow_changes = base._difference_features(
        x,
        fastslow_shape,
        horizons=(1, 5),
    )
    x = x.join(fastslow_changes)

    # Existing expert-vs-CM relative geometry is also scale invariant once
    # log_scale terms are removed. Reuse it rather than duplicating work.
    expert_shape = [
        column
        for column in x.columns
        if column.startswith("relgeom__") and "log_scale" not in column
    ]

    rotation = [
        column
        for column in (
            "cov_top_subspace_rotation",
            "cov_top_subspace_rotation__change_1d",
            "cov_top_subspace_rotation__change_5d",
            "cov_top_subspace_rotation__change_21d",
        )
        if column in x
    ]

    relative_shape = list(
        dict.fromkeys(
            temporal_shape
            + list(temporal_changes.columns)
            + fastslow_shape
            + list(fastslow_changes.columns)
            + expert_shape
            + rotation
        )
    )

    feature_sets = OrderedDict(original_sets)
    feature_sets["johansson_relshape"] = (
        original_sets["johansson"] + relative_shape
    )
    feature_sets["johansson_market_relshape"] = (
        original_sets["johansson_market"] + relative_shape
    )

    return x.replace([np.inf, -np.inf], np.nan), feature_sets


def run_walkforward(returns, factors=None):
    """Run the existing engine after injecting the extended feature builder."""
    original = base.build_walkforward_features
    base.build_walkforward_features = build_walkforward_features
    try:
        return base.run_walkforward(returns, factors)
    finally:
        base.build_walkforward_features = original


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward CM-IEWMA with scale-invariant relative-shape features"
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
        "--output-dir",
        type=Path,
        default=Path("experiments/research/results"),
    )
    args = parser.parse_args()

    returns = _load(args.returns)
    factors = (
        _load(args.factors).reindex(returns.index).dropna()
        if args.factors.exists()
        else None
    )
    daily, summary = run_walkforward(returns, factors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "walkforward_daily.csv")
    (args.output_dir / "walkforward_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
