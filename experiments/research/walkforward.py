from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.research.phase0 import (
    IEWMA_PAIRS,
    _arr,
    _load,
    build_cm_iewma,
    build_iewma_experts,
    factor_features,
    gaussian_nll,
    run_phase0,
)

EXPERT_NAMES = tuple(f"{hv}-{hc}" for hv, hc in IEWMA_PAIRS)
REFIT_EVERY = 63
INITIAL_TRAIN = 756
VALIDATION_DAYS = 252
ETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def precision_cholesky(cov):
    sigma = _arr(cov)
    return np.linalg.cholesky(np.linalg.inv(sigma))


def combine_cholesky(choleskies, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return sum(
        weight * chol
        for weight, chol in zip(weights, choleskies, strict=True)
    )


def losses_from_cholesky(r, chol):
    logdet_sigma = -2.0 * np.log(np.diag(chol)).sum()
    z = chol.T @ r
    nll = 0.5 * (logdet_sigma + float(z @ z))
    precision = chol @ chol.T
    one = np.ones(len(r))
    x = precision @ one
    denom = float(one @ x)
    if not np.isfinite(denom) or abs(denom) <= 1e-14:
        return nll, np.nan, np.full_like(one, np.nan)
    weights = x / denom
    gmv_square = float((weights @ r) ** 2)
    return nll, gmv_square, weights


def tilted_weights(
    base_weights,
    predicted_relative_losses,
    eta,
    floor=1e-6,
):
    base = np.clip(np.asarray(base_weights, dtype=float), floor, None)
    base = base / base.sum()
    delta = np.asarray(predicted_relative_losses, dtype=float)
    logits = np.log(base) - float(eta) * delta
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def relative_geometry_features(experts, reference):
    rows = {}
    times = sorted(
        set(reference).intersection(*(set(path) for path in experts.values()))
    )
    for t in times:
        ref = _arr(reference[t])
        row = {}
        shape_losses = []
        for name in EXPERT_NAMES:
            try:
                eta = eigh(
                    ref,
                    _arr(experts[name][t]),
                    eigvals_only=True,
                )
            except np.linalg.LinAlgError:
                eta = np.full(ref.shape[0], np.nan)
            if np.any(~np.isfinite(eta)) or np.any(eta <= 0):
                row[f"relgeom__{name}__log_scale"] = np.nan
                row[f"relgeom__{name}__log_shape_std"] = np.nan
                row[f"relgeom__{name}__log_condition"] = np.nan
                continue
            log_eta = np.log(eta)
            log_scale = float(log_eta.mean())
            shape_std = float(log_eta.std())
            log_condition = float(log_eta.max() - log_eta.min())
            row[f"relgeom__{name}__log_scale"] = log_scale
            row[f"relgeom__{name}__log_shape_std"] = shape_std
            row[f"relgeom__{name}__log_condition"] = log_condition
            shape_losses.append(log_condition)
        row["relgeom__mean_shape_loss"] = (
            float(np.mean(shape_losses)) if shape_losses else np.nan
        )
        row["relgeom__max_shape_loss"] = (
            float(np.max(shape_losses)) if shape_losses else np.nan
        )
        rows[t] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _difference_features(frame, columns, horizons=(1, 5, 21)):
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        for horizon in horizons:
            out[f"{column}__change_{horizon}d"] = frame[column].diff(horizon)
    return out


def build_walkforward_features(phase0, experts, cm):
    x = pd.DataFrame(index=phase0.index)
    groups = OrderedDict()

    weight_cols = [f"cm_weight__{name}" for name in EXPERT_NAMES]
    weights = phase0[weight_cols].clip(lower=1e-12)
    x[weight_cols] = weights
    x["cm_weight_entropy"] = -(weights * np.log(weights)).sum(axis=1)
    x["cm_weight_max"] = weights.max(axis=1)
    vol_hl = np.array([pair[0] for pair in IEWMA_PAIRS], dtype=float)
    cor_hl = np.array([pair[1] for pair in IEWMA_PAIRS], dtype=float)
    x["cm_effective_vol_halflife"] = weights.to_numpy() @ vol_hl
    x["cm_effective_cor_halflife"] = weights.to_numpy() @ cor_hl
    for column in weight_cols:
        x[f"{column}__change_1d"] = phase0[column].diff()
        x[f"{column}__change_5d"] = phase0[column].diff(5)

    baseline_loss = phase0["loss_a_nll__CM-IEWMA"]
    for name in EXPERT_NAMES:
        relative = phase0[f"loss_a_nll__{name}"] - baseline_loss
        safe_relative = relative.shift(1)
        x[f"recent_rel_nll__{name}__1d"] = safe_relative
        for horizon in (5, 21, 63):
            x[f"recent_rel_nll__{name}__mean_{horizon}d"] = (
                safe_relative.rolling(horizon).mean()
            )

    groups["johansson"] = list(x.columns)

    x["cov_log_trace"] = np.log(
        phase0["cov_trace"].clip(lower=1e-18)
    )
    volatility_cols = ["cov_log_trace"]
    for column in (
        "factor_market_vol_fast",
        "factor_market_vol_slow",
        "factor_market_abs_return",
    ):
        if column in phase0:
            x[column] = phase0[column]
            volatility_cols.append(column)
    vol_changes = _difference_features(x, volatility_cols)
    x = x.join(vol_changes)
    volatility_cols += list(vol_changes.columns)
    groups["volatility"] = volatility_cols

    correlation_cols = ["cov_mean_corr", "cov_mean_abs_corr"]
    x[correlation_cols] = phase0[correlation_cols]
    corr_changes = _difference_features(x, correlation_cols)
    x = x.join(corr_changes)
    correlation_cols += list(corr_changes.columns)
    groups["correlation"] = correlation_cols

    spectral_cols = [
        "cov_log_condition",
        "cov_leading_eigen_share",
        "cov_effective_rank",
        "cov_log_eigen_dispersion",
        "cov_top_subspace_rotation",
    ]
    x[spectral_cols] = phase0[spectral_cols]
    spectral_changes = _difference_features(x, spectral_cols)
    x = x.join(spectral_changes)
    spectral_cols += list(spectral_changes.columns)

    relgeom = relative_geometry_features(experts, cm).reindex(x.index)
    x = x.join(relgeom)
    relgeom_cols = list(relgeom.columns)
    relgeom_changes = _difference_features(
        x,
        relgeom_cols,
        horizons=(1, 5),
    )
    x = x.join(relgeom_changes)
    spectral_cols += relgeom_cols + list(relgeom_changes.columns)
    groups["spectral_paleologo"] = spectral_cols

    factor_cols = [
        column
        for column in (
            "factor_fast_mean_corr",
            "factor_fast_mean_abs_corr",
            "factor_fast_leading_eigen_share",
            "factor_fast_effective_rank",
            "factor_slow_mean_corr",
            "factor_slow_mean_abs_corr",
            "factor_slow_leading_eigen_share",
            "factor_slow_effective_rank",
            "factor_corr_shift_fro",
        )
        if column in phase0
    ]
    if factor_cols:
        x[factor_cols] = phase0[factor_cols]
        factor_changes = _difference_features(
            x,
            factor_cols,
            horizons=(1, 5),
        )
        x = x.join(factor_changes)
        factor_cols += list(factor_changes.columns)
    groups["factor_geometry"] = factor_cols

    feature_sets = OrderedDict(
        johansson=groups["johansson"],
        johansson_market=(
            groups["johansson"]
            + groups["volatility"]
            + groups["correlation"]
        ),
        johansson_spectral=(
            groups["johansson"] + groups["spectral_paleologo"]
        ),
        johansson_market_spectral=(
            groups["johansson"]
            + groups["volatility"]
            + groups["correlation"]
            + groups["spectral_paleologo"]
        ),
        all=(
            groups["johansson"]
            + groups["volatility"]
            + groups["correlation"]
            + groups["spectral_paleologo"]
            + groups["factor_geometry"]
        ),
    )
    return x.replace([np.inf, -np.inf], np.nan), feature_sets


def target_relative_losses(phase0):
    baseline = phase0["loss_a_nll__CM-IEWMA"]
    return pd.DataFrame(
        {
            name: phase0[f"loss_a_nll__{name}"] - baseline
            for name in EXPERT_NAMES
        }
    )


def _model_templates():
    return OrderedDict(
        ridge=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        random_forest=RandomForestRegressor(
            n_estimators=64,
            max_depth=5,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=0,
            n_jobs=-1,
        ),
    )


def _impute(train, other):
    medians = train.median(axis=0)
    return (
        train.fillna(medians).fillna(0.0),
        other.fillna(medians).fillna(0.0),
    )


def _winsorize_targets(y, lower=0.01, upper=0.99):
    lo = y.quantile(lower)
    hi = y.quantile(upper)
    return y.clip(lower=lo, upper=hi, axis=1), lo, hi


def _predict_model(model_template, x_train, y_train, x_predict):
    x_train_i, x_predict_i = _impute(x_train, x_predict)
    y_clipped, lo, hi = _winsorize_targets(y_train)
    model = clone(model_template)
    model.fit(x_train_i, y_clipped)
    prediction = pd.DataFrame(
        model.predict(x_predict_i),
        index=x_predict.index,
        columns=y_train.columns,
    )
    return prediction.clip(lower=lo, upper=hi, axis=1)


def _precompute_choleskies(experts, times):
    result = {}
    for t in times:
        result[t] = [
            precision_cholesky(experts[name][t])
            for name in EXPERT_NAMES
        ]
    return result


def _evaluate_prediction_rows(
    times,
    predictions,
    eta,
    baseline_weights,
    choleskies,
    next_returns,
):
    rows = {}
    for t in times:
        base = baseline_weights.loc[t, list(EXPERT_NAMES)].to_numpy(float)
        pred = predictions.loc[t, list(EXPERT_NAMES)].to_numpy(float)
        weights = tilted_weights(base, pred, eta)
        chol = combine_cholesky(choleskies[t], weights)
        r = next_returns.loc[t].to_numpy(float)
        nll, gmv_square, gmv_weights = losses_from_cholesky(r, chol)
        row = {
            "nll": nll,
            "gmv_square": gmv_square,
            "eta": float(eta),
            "expert_weight_entropy": float(
                -(
                    weights
                    * np.log(np.clip(weights, 1e-12, None))
                ).sum()
            ),
        }
        for name, weight in zip(EXPERT_NAMES, weights, strict=True):
            row[f"weight__{name}"] = float(weight)
        for i, weight in enumerate(gmv_weights):
            row[f"gmv_weight__{i}"] = float(weight)
        rows[t] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _validation_eta(
    model_template,
    features,
    targets,
    train_index,
    baseline_weights,
    choleskies,
    next_returns,
):
    if len(train_index) <= VALIDATION_DAYS + 126:
        return 0.0
    validation_index = train_index[-VALIDATION_DAYS:]
    core_index = train_index[:-VALIDATION_DAYS]
    prediction = _predict_model(
        model_template,
        features.loc[core_index],
        targets.loc[core_index],
        features.loc[validation_index],
    )
    scores = {}
    for eta in ETA_GRID:
        evaluated = _evaluate_prediction_rows(
            validation_index,
            prediction,
            eta,
            baseline_weights,
            choleskies,
            next_returns,
        )
        scores[eta] = float(evaluated["nll"].mean())
    return min(scores, key=scores.get)


def walkforward_model(
    model_template,
    features,
    targets,
    baseline_weights,
    choleskies,
    next_returns,
    initial_train=INITIAL_TRAIN,
    refit_every=REFIT_EVERY,
):
    valid = (
        features.index.intersection(targets.dropna().index)
        .intersection(baseline_weights.dropna().index)
        .intersection(next_returns.dropna().index)
    )
    valid = pd.DatetimeIndex(sorted(valid))
    if len(valid) <= initial_train:
        raise ValueError(
            "Not enough observations for the requested walk-forward split"
        )

    oos = valid[initial_train:]
    chunks = [
        oos[i : i + refit_every]
        for i in range(0, len(oos), refit_every)
    ]
    results = []

    for block in chunks:
        block_start = block[0]
        train_index = valid[valid < block_start]
        eta = _validation_eta(
            model_template,
            features,
            targets,
            train_index,
            baseline_weights,
            choleskies,
            next_returns,
        )
        prediction = _predict_model(
            model_template,
            features.loc[train_index],
            targets.loc[train_index],
            features.loc[block],
        )
        evaluated = _evaluate_prediction_rows(
            block,
            prediction,
            eta,
            baseline_weights,
            choleskies,
            next_returns,
        )
        results.append(evaluated)

    return pd.concat(results).sort_index()


def _baseline_path(
    times,
    baseline_weights,
    choleskies,
    next_returns,
):
    zero_prediction = pd.DataFrame(
        0.0,
        index=times,
        columns=EXPERT_NAMES,
    )
    return _evaluate_prediction_rows(
        times,
        zero_prediction,
        0.0,
        baseline_weights,
        choleskies,
        next_returns,
    )


def _annualized_gmv_turnover(result):
    columns = [
        column
        for column in result
        if column.startswith("gmv_weight__")
    ]
    if not columns or len(result) < 2:
        return np.nan
    weights = result[columns].to_numpy(float)
    return float(
        252 * np.abs(np.diff(weights, axis=0)).sum(axis=1).mean()
    )


def _expert_weight_turnover(result):
    columns = [f"weight__{name}" for name in EXPERT_NAMES]
    if len(result) < 2:
        return np.nan
    weights = result[columns].to_numpy(float)
    return float(
        252 * np.abs(np.diff(weights, axis=0)).sum(axis=1).mean()
    )


def summarize_result(result, baseline):
    aligned = result.index.intersection(baseline.index)
    improvement = (
        baseline.loc[aligned, "nll"] - result.loc[aligned, "nll"]
    )
    gmv_improvement = (
        baseline.loc[aligned, "gmv_square"]
        - result.loc[aligned, "gmv_square"]
    )
    return {
        "n_obs": int(len(aligned)),
        "mean_nll": float(result.loc[aligned, "nll"].mean()),
        "mean_nll_improvement_vs_johansson": float(improvement.mean()),
        "median_nll_improvement_vs_johansson": float(
            improvement.median()
        ),
        "positive_nll_improvement_fraction": float(
            (improvement > 0).mean()
        ),
        "mean_gmv_square": float(
            result.loc[aligned, "gmv_square"].mean()
        ),
        "mean_gmv_square_improvement_vs_johansson": float(
            gmv_improvement.mean()
        ),
        "mean_eta": float(result.loc[aligned, "eta"].mean()),
        "eta_zero_fraction": float(
            (result.loc[aligned, "eta"] == 0).mean()
        ),
        "annualized_expert_weight_turnover": _expert_weight_turnover(
            result.loc[aligned]
        ),
        "annualized_gmv_weight_turnover": _annualized_gmv_turnover(
            result.loc[aligned]
        ),
    }


def run_walkforward(returns, factors=None):
    experts = build_iewma_experts(returns)
    cm, weights = build_cm_iewma(returns, experts)
    phase0, _ = run_phase0(returns, factors)

    features, feature_sets = build_walkforward_features(
        phase0,
        experts,
        cm,
    )
    targets = target_relative_losses(phase0)
    baseline_weights = weights.reindex(columns=EXPERT_NAMES)
    next_returns = returns.shift(-1)

    common_times = sorted(
        set(phase0.index)
        .intersection(*(set(experts[name]) for name in EXPERT_NAMES))
        .intersection(set(baseline_weights.dropna().index))
    )
    choleskies = _precompute_choleskies(experts, common_times)

    valid_target = targets.dropna().index
    valid_returns = next_returns.dropna().index
    model_times = pd.DatetimeIndex(common_times).intersection(valid_target)
    model_times = model_times.intersection(valid_returns)
    baseline = _baseline_path(
        model_times,
        baseline_weights,
        choleskies,
        next_returns,
    )

    phase0_cm = phase0["loss_a_nll__CM-IEWMA"].reindex(baseline.index)
    reconstruction_error = float(
        (baseline["nll"] - phase0_cm).abs().dropna().max()
    )

    daily = pd.DataFrame(index=baseline.index)
    daily["johansson_nll"] = baseline["nll"]
    daily["johansson_gmv_square"] = baseline["gmv_square"]
    for name in EXPERT_NAMES:
        daily[f"johansson_weight__{name}"] = baseline[f"weight__{name}"]

    summary = {
        "feature_sets": {
            name: columns
            for name, columns in feature_sets.items()
        },
        "baseline_reconstruction_max_abs_nll_diff": reconstruction_error,
        "models": {},
    }

    templates = _model_templates()
    first_oos = None
    for model_name, template in templates.items():
        summary["models"][model_name] = {}
        for feature_name, columns in feature_sets.items():
            result = walkforward_model(
                template,
                features[columns],
                targets,
                baseline_weights,
                choleskies,
                next_returns,
            )
            if first_oos is None or result.index.min() < first_oos:
                first_oos = result.index.min()
            key = f"{model_name}__{feature_name}"
            daily.loc[result.index, f"{key}__nll"] = result["nll"]
            daily.loc[result.index, f"{key}__gmv_square"] = result[
                "gmv_square"
            ]
            daily.loc[result.index, f"{key}__eta"] = result["eta"]
            for name in EXPERT_NAMES:
                daily.loc[
                    result.index,
                    f"{key}__weight__{name}",
                ] = result[f"weight__{name}"]
            summary["models"][model_name][feature_name] = (
                summarize_result(result, baseline)
            )

    if first_oos is not None:
        baseline_oos = baseline.loc[first_oos:]
        summary["oos_start"] = str(first_oos.date())
        summary["oos_end"] = str(baseline_oos.index.max().date())
        summary["oos_n_obs"] = int(len(baseline_oos))
        summary["johansson"] = {
            "mean_nll": float(baseline_oos["nll"].mean()),
            "mean_gmv_square": float(
                baseline_oos["gmv_square"].mean()
            ),
            "annualized_expert_weight_turnover": (
                _expert_weight_turnover(baseline_oos)
            ),
            "annualized_gmv_weight_turnover": (
                _annualized_gmv_turnover(baseline_oos)
            ),
        }

    return daily.sort_index(), summary


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward state-conditioned CM-IEWMA experiment"
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
