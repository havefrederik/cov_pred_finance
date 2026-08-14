from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.research.phase0 import gaussian_nll
from experiments.research.walkforward import (
    EXPERT_NAMES,
    combine_cholesky,
    losses_from_cholesky,
    precision_cholesky,
    relative_geometry_features,
    tilted_weights,
)


def test_tilt_eta_zero_recovers_johansson_weights():
    base = np.array([0.10, 0.20, 0.30, 0.15, 0.25])
    predicted = np.array([-2.0, 1.0, 0.5, -0.3, 0.0])
    result = tilted_weights(base, predicted, eta=0.0)
    assert np.allclose(result, base / base.sum())


def test_tilt_rewards_lower_predicted_relative_loss():
    base = np.repeat(0.2, 5)
    predicted = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    result = tilted_weights(base, predicted, eta=1.0)
    assert result[0] > result[1]
    assert np.isclose(result.sum(), 1.0)


def test_cholesky_loss_matches_covariance_nll():
    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    returns = np.array([0.02, -0.01])
    chol = precision_cholesky(covariance)
    nll, _, weights = losses_from_cholesky(returns, chol)
    assert np.isclose(nll, gaussian_nll(returns, covariance))
    assert np.isclose(weights.sum(), 1.0)


def test_combining_identical_experts_leaves_cholesky_unchanged():
    covariance = np.array([[1.0, 0.2], [0.2, 1.5]])
    chol = precision_cholesky(covariance)
    combined = combine_cholesky([chol, chol], [0.25, 0.75])
    assert np.allclose(combined, chol)


def test_relative_geometry_separates_scale_from_shape():
    time = pd.Timestamp("2020-01-01")
    reference_matrix = np.array([[2.0, 0.4], [0.4, 1.0]])
    reference = {time: pd.DataFrame(reference_matrix)}
    experts = {
        name: {time: pd.DataFrame(3.0 * reference_matrix)}
        for name in EXPERT_NAMES
    }
    features = relative_geometry_features(experts, reference)
    assert np.isclose(
        features.loc[time, "relgeom__10-21__log_scale"],
        -np.log(3.0),
    )
    assert np.isclose(
        features.loc[time, "relgeom__10-21__log_condition"],
        0.0,
        atol=1e-12,
    )
