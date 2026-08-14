# Phase 0: adaptive covariance research

This directory contains the first controlled extension of Johansson, Ogut,
Pelger, Schmelzer, and Boyd's CM-IEWMA methodology.

The Phase 0 rule is deliberately conservative: **do not replace the authors'
covariance machinery before showing that there is exploitable variation in
how fast covariance should adapt.**

## Baseline

The stock experiment reproduces the five IEWMA volatility/correlation
half-life pairs used in the authors' stock notebook:

- `(10, 21)`
- `(21, 63)`
- `(63, 125)`
- `(125, 250)`
- `(250, 500)`

It uses 63 minimum observations for volatility and covariance estimation,
adds 5% diagonal regularization to the fastest `(10, 21)` expert, and
combines the experts with the original precision-Cholesky CM-IEWMA method
using a 10-day likelihood window.

The authors' timing convention is preserved: a covariance stored at date
`t` uses returns through `t` and forecasts the return covariance at `t+1`.

## Candidate A: predictive expert weighting

The original CM-IEWMA weights react to recent realized likelihood. Phase 0
first asks whether there is enough oracle room for a predictive
state-conditioned gate.

For every date and expert, we compute next-day Gaussian negative
log-likelihood, up to the common constant,

\[
L_{t+1}(\hat\Sigma_t)=\frac12\left[
\log\det\hat\Sigma_t+
r_{t+1}^\top\hat\Sigma_t^{-1}r_{t+1}
\right],
\]

and the next-day squared return of the unconstrained fully invested GMV
portfolio implied by the covariance forecast.

The ex-post winner is only a diagnostic. It is not a deployable strategy.
The purpose is to see whether the useful covariance memory scale varies
enough through time to justify machine learning.

The deployable extension does **not** hard-select one expert. It preserves
Johansson et al.'s precision-Cholesky combination and learns a
state-dependent tilt to the original CM-IEWMA mixture weights.

## Candidate B: state-dependent update intensity

The authors' smoothing extension uses a fixed smoothing rate. Phase 0
creates fixed smoothing experts around the raw CM-IEWMA path with
half-lives `1, 5, 10, 21, 63, 125` days.

If the best smoothing intensity varies systematically through time, the
later thesis model can replace the fixed rate by

\[
\gamma_t=g_\theta(X_t),
\]

with

\[
\tilde\Sigma_t=
(1-\gamma_t)\tilde\Sigma_{t-1}
+\gamma_t\hat\Sigma_t^{CM}.
\]

## Short horizons

This remains a **one-day-ahead covariance forecasting problem**. One future
daily return gives only the rank-one matrix `r r'`, but this does not
prevent valid next-day forecast scoring. The Johansson paper itself
evaluates daily covariance forecasts with rank-one MSE and daily Gaussian
likelihood.

We therefore do not lengthen the forecasting horizon merely to obtain a
full-rank ex-post covariance matrix. Five-, ten-, twenty-one-, and
sixty-three-day quantities in the output are rolling summaries of repeated
daily out-of-sample scores.

## Paleologo-inspired diagnostics

Paleologo's misspecification results motivate looking at covariance
**shape**, not only raw matrix distance. If an estimated covariance is a
scalar multiple of the reference covariance, portfolio directions are
unchanged even if Frobenius error is large.

`paleologo_relative_shape_loss()` implements

\[
\log\kappa\left(
\Sigma_{\rm ref}^{1/2}
\hat\Sigma^{-1}
\Sigma_{\rm ref}^{1/2}
\right)
\]

using a generalized eigenvalue problem. The loss is zero for
`forecast = c * reference`.

Because a one-day realized covariance is rank one, this full relative-shape
loss is not used as the one-day oracle label. Instead Phase 0 records
observable state variables motivated by the same geometry: covariance
condition number, leading eigenvalue share, effective rank, dispersion of
log eigenvalues, mean correlation, mean absolute correlation, and rotation
of the leading covariance subspace. The supplied Fama-French factors add
fast/slow factor-correlation geometry and market-volatility state.

The walk-forward extension also uses Paleologo-style relative geometry
between the observable expert forecasts and CM-IEWMA as ex-ante features.
This measures whether experts disagree mainly about overall scale or about
relative risk directions.

## Run

From the repository root:

```bash
python -m experiments.research.phase0
```

Outputs are written to `experiments/research/results/`:

- `phase0_daily_diagnostics.csv`
- `phase0_summary.json`

The `research-phase0.yml` GitHub Actions workflow runs unit tests and the
same Phase 0 experiment, then uploads the results as an artifact.

## Kill criteria before ML

Phase 0 is designed to answer five questions before we fit a more complex
machine-learning gate:

1. Does the best IEWMA memory scale switch materially through time?
2. Is oracle improvement over original CM-IEWMA economically nontrivial?
3. Does the best smoothing intensity vary materially through time?
4. Do likelihood and realized-GMV winners disagree?
5. Are oracle gaps associated with volatility, correlation geometry,
   factor state, conditioning, or eigenspace rotation?

If oracle room is tiny or one expert dominates almost everywhere, the ML
thesis direction should be killed early rather than rescued with a more
complicated model.
