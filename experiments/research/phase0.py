from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from cvx.covariance.combination import from_sigmas
from cvx.covariance.ewma import iterated_ewma

IEWMA_PAIRS = ((10, 21), (21, 63), (63, 125), (125, 250), (250, 500))
CM_WINDOW = 10
MIN_PERIODS = 63
FAST_REGULARIZATION = 0.05
SMOOTHING_HALFLIVES = (1, 5, 10, 21, 63, 125)


def _arr(cov):
    x = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    return 0.5 * (x + x.T)


def _frame(x, like):
    return pd.DataFrame(x, index=like.index, columns=like.columns)


def gaussian_nll(r, cov):
    sigma = _arr(cov)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return np.nan
    try:
        quad = float(r @ np.linalg.solve(sigma, r))
    except np.linalg.LinAlgError:
        return np.nan
    return 0.5 * (logdet + quad)


def gmv_weights(cov):
    sigma = _arr(cov)
    one = np.ones(sigma.shape[0])
    try:
        x = np.linalg.solve(sigma, one)
    except np.linalg.LinAlgError:
        return np.full_like(one, np.nan)
    denom = float(one @ x)
    return x / denom if np.isfinite(denom) and abs(denom) > 1e-14 else np.full_like(one, np.nan)


def realized_gmv_square(r, cov):
    w = gmv_weights(cov)
    return np.nan if np.isnan(w).any() else float((w @ r) ** 2)


def paleologo_relative_shape_loss(forecast, reference):
    """log condition number of relative risk shape; zero for forecast=c*reference."""
    try:
        eta = eigh(_arr(reference), _arr(forecast), eigvals_only=True)
    except np.linalg.LinAlgError:
        return np.nan
    if np.any(eta <= 0):
        return np.nan
    return float(np.log(eta[-1] / eta[0]))


def build_iewma_experts(returns):
    experts = {}
    for hv, hc in IEWMA_PAIRS:
        name = f"{hv}-{hc}"
        results = iterated_ewma(
            returns,
            vola_halflife=hv,
            cov_halflife=hc,
            min_periods_vola=MIN_PERIODS,
            min_periods_cov=MIN_PERIODS,
        )
        experts[name] = {x.time: x.covariance for x in results}
    fast = f"{IEWMA_PAIRS[0][0]}-{IEWMA_PAIRS[0][1]}"
    experts[fast] = {
        t: _frame(_arr(s) + FAST_REGULARIZATION * np.diag(np.diag(_arr(s))), s)
        for t, s in experts[fast].items()
    }
    return experts


def build_cm_iewma(returns, experts):
    results = [x for x in from_sigmas(experts, returns).solve(window=CM_WINDOW) if x is not None]
    covs = {x.time: x.covariance for x in results}
    weights = pd.DataFrame({x.time: x.weights for x in results}).T.sort_index()
    return covs, weights


def smooth_path(covs, halflife):
    alpha = 1 - np.exp(-np.log(2) / halflife)
    out, prev = {}, None
    for t, cov in sorted(covs.items()):
        cur = _arr(cov)
        val = cur if prev is None else (1 - alpha) * prev + alpha * cur
        out[t] = _frame(0.5 * (val + val.T), cov)
        prev = val
    return out


def build_smoothing_experts(cm):
    out = OrderedDict(raw=cm)
    for h in SMOOTHING_HALFLIVES:
        out[f"smooth-{h}"] = smooth_path(cm, h)
    return out


def score_next_day(returns, paths):
    next_returns = returns.shift(-1)
    nll, gmv = {}, {}
    for name, path in paths.items():
        ln, lg = {}, {}
        for t, cov in path.items():
            if t not in next_returns.index or next_returns.loc[t].isna().any():
                continue
            r = next_returns.loc[t].to_numpy(float)
            ln[t] = gaussian_nll(r, cov)
            lg[t] = realized_gmv_square(r, cov)
        nll[name], gmv[name] = pd.Series(ln, dtype=float), pd.Series(lg, dtype=float)
    return pd.DataFrame(nll).sort_index(), pd.DataFrame(gmv).sort_index()


def oracle(losses, baseline, prefix):
    x = losses.dropna(how="all")
    out = pd.DataFrame(index=x.index)
    out[f"{prefix}_winner"] = x.idxmin(axis=1)
    out[f"{prefix}_best_loss"] = x.min(axis=1)
    out[f"{prefix}_baseline_loss"] = x[baseline]
    out[f"{prefix}_oracle_gap"] = out[f"{prefix}_baseline_loss"] - out[f"{prefix}_best_loss"]
    return out


def _participation_ratio(eigs):
    d = float(np.sum(eigs**2))
    return float(np.sum(eigs) ** 2 / d) if d > 0 else np.nan


def spectral_features(covs, subspace_rank=3):
    rows, prev_p = {}, None
    for t, cov in sorted(covs.items()):
        s = _arr(cov)
        eig, vec = np.linalg.eigh(s)
        if np.any(eig <= 0):
            continue
        sd = np.sqrt(np.clip(np.diag(s), 1e-18, None))
        corr = s / np.outer(sd, sd)
        off = corr[np.triu_indices_from(corr, 1)]
        k = min(subspace_rank, s.shape[0])
        u = vec[:, -k:]
        p = u @ u.T
        rot = np.nan if prev_p is None else np.linalg.norm(p - prev_p, "fro") / np.sqrt(2 * k)
        le = np.log(eig)
        rows[t] = {
            "cov_trace": float(np.trace(s)),
            "cov_logdet": float(le.sum()),
            "cov_log_condition": float(le[-1] - le[0]),
            "cov_leading_eigen_share": float(eig[-1] / eig.sum()),
            "cov_effective_rank": _participation_ratio(eig),
            "cov_log_eigen_dispersion": float(le.std()),
            "cov_mean_corr": float(off.mean()),
            "cov_mean_abs_corr": float(np.abs(off).mean()),
            "cov_top_subspace_rotation": float(rot),
        }
        prev_p = p
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def factor_features(factors, fast=21, slow=63):
    def stats(x):
        c = np.corrcoef(x, rowvar=False)
        e = np.linalg.eigvalsh(c)
        off = c[np.triu_indices_from(c, 1)]
        return float(off.mean()), float(np.abs(off).mean()), float(e[-1] / e.sum()), _participation_ratio(e), c

    rows = {}
    for i in range(slow - 1, len(factors)):
        t = factors.index[i]
        xf = factors.iloc[i - fast + 1 : i + 1].to_numpy(float)
        xs = factors.iloc[i - slow + 1 : i + 1].to_numpy(float)
        mf, maf, lf, ef, cf = stats(xf)
        ms, mas, ls, es, cs = stats(xs)
        rows[t] = {
            "factor_fast_mean_corr": mf,
            "factor_fast_mean_abs_corr": maf,
            "factor_fast_leading_eigen_share": lf,
            "factor_fast_effective_rank": ef,
            "factor_slow_mean_corr": ms,
            "factor_slow_mean_abs_corr": mas,
            "factor_slow_leading_eigen_share": ls,
            "factor_slow_effective_rank": es,
            "factor_corr_shift_fro": float(np.linalg.norm(cf - cs, "fro")),
            "factor_market_vol_fast": float(xf[:, 0].std()),
            "factor_market_vol_slow": float(xs[:, 0].std()),
            "factor_market_abs_return": float(abs(factors.iloc[i, 0])),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def add_reporting_blocks(df, columns, horizons=(5, 10, 21, 63)):
    out = df.copy()
    for h in horizons:
        for col in columns:
            out[f"{col}_mean_{h}d"] = out[col].rolling(h).mean()
    return out


def _freq(x):
    return {str(k): float(v) for k, v in x.dropna().value_counts(normalize=True).items()}


def run_phase0(returns, factors=None):
    experts = build_iewma_experts(returns)
    cm, weights = build_cm_iewma(returns, experts)

    a_paths = OrderedDict(experts)
    a_paths["CM-IEWMA"] = cm
    a_nll, a_gmv = score_next_day(returns, a_paths)
    a_on, a_og = oracle(a_nll, "CM-IEWMA", "a_nll"), oracle(a_gmv, "CM-IEWMA", "a_gmv")

    b_paths = build_smoothing_experts(cm)
    b_nll, b_gmv = score_next_day(returns, b_paths)
    b_on, b_og = oracle(b_nll, "raw", "b_nll"), oracle(b_gmv, "raw", "b_gmv")

    data = spectral_features(cm)
    if factors is not None:
        data = data.join(factor_features(factors), how="left")
    for frame in (
        a_nll.add_prefix("loss_a_nll__"), a_gmv.add_prefix("loss_a_gmv__"), a_on, a_og,
        b_nll.add_prefix("loss_b_nll__"), b_gmv.add_prefix("loss_b_gmv__"), b_on, b_og,
        weights.add_prefix("cm_weight__"),
    ):
        data = data.join(frame, how="left")
    data = add_reporting_blocks(data, ["a_nll_oracle_gap", "a_gmv_oracle_gap", "b_nll_oracle_gap", "b_gmv_oracle_gap"])

    summary = {
        "start": str(data.index.min().date()),
        "end": str(data.index.max().date()),
        "n_rows": int(len(data)),
        "candidate_a": {
            "nll_winner_frequency": _freq(data["a_nll_winner"]),
            "gmv_winner_frequency": _freq(data["a_gmv_winner"]),
            "mean_daily_nll_oracle_gap_vs_cm": float(data["a_nll_oracle_gap"].mean()),
            "mean_daily_gmv_oracle_gap_vs_cm": float(data["a_gmv_oracle_gap"].mean()),
            "positive_nll_oracle_gap_fraction": float((data["a_nll_oracle_gap"].dropna() > 0).mean()),
        },
        "candidate_b": {
            "nll_winner_frequency": _freq(data["b_nll_winner"]),
            "gmv_winner_frequency": _freq(data["b_gmv_winner"]),
            "mean_daily_nll_oracle_gap_vs_raw": float(data["b_nll_oracle_gap"].mean()),
            "mean_daily_gmv_oracle_gap_vs_raw": float(data["b_gmv_oracle_gap"].mean()),
            "positive_nll_oracle_gap_fraction": float((data["b_nll_oracle_gap"].dropna() > 0).mean()),
        },
    }
    return data, summary


def _load(path):
    x = pd.read_csv(path, index_col=0, parse_dates=True)
    x.index = pd.DatetimeIndex(x.index)
    return x.sort_index()


def main():
    p = argparse.ArgumentParser(description="Phase 0 adaptive CM-IEWMA diagnostics")
    p.add_argument("--returns", type=Path, default=Path("experiments/data/SP500_top25_adjusted.csv"))
    p.add_argument("--factors", type=Path, default=Path("experiments/data/ff5_no_rf.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/research/results"))
    args = p.parse_args()

    returns = _load(args.returns)
    factors = _load(args.factors).reindex(returns.index).dropna() if args.factors.exists() else None
    data, summary = run_phase0(returns, factors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "phase0_daily_diagnostics.csv")
    (args.output_dir / "phase0_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
