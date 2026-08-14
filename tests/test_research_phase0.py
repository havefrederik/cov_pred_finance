from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.research.phase0 import (
    gaussian_nll,
    gmv_weights,
    paleologo_relative_shape_loss,
    smooth_path,
    spectral_features,
)


def test_paleologo_shape_loss_is_scale_invariant():
    reference = np.array([[2.0, 0.4], [0.4, 1.0]])
    assert np.isclose(paleologo_relative_shape_loss(3.7 * reference, reference), 0.0)


def test_paleologo_shape_loss_detects_relative_distortion():
    reference = np.eye(2)
    forecast = np.diag([1.0, 2.0])
    assert np.isclose(
        paleologo_relative_shape_loss(forecast, reference),
        np.log(2.0),
    )


def test_gmv_weights_are_fully_invested():
    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    weights = gmv_weights(covariance)
    assert np.isclose(weights.sum(), 1.0)


def test_smoothing_path_stays_positive_definite():
    index = ["a", "b"]
    covariances = {
        pd.Timestamp("2020-01-01"): pd.DataFrame([[1.0, 0.2], [0.2, 1.5]], index=index, columns=index),
        pd.Timestamp("2020-01-02"): pd.DataFrame([[2.0, -0.1], [-0.1, 0.8]], index=index, columns=index),
    }
    path = smooth_path(covariances, halflife=5)
    for covariance in path.values():
        assert np.linalg.eigvalsh(covariance.values).min() > 0


def test_spectral_features_report_no_rotation_for_constant_path():
    index = ["a", "b", "c"]
    covariance = pd.DataFrame(
        [[1.0, 0.2, 0.1], [0.2, 1.3, 0.05], [0.1, 0.05, 0.8]],
        index=index,
        columns=index,
    )
    path = {
        pd.Timestamp("2020-01-01"): covariance,
        pd.Timestamp("2020-01-02"): covariance.copy(),
    }
    features = spectral_features(path, subspace_rank=2)
    assert np.isclose(features.iloc[1]["cov_top_subspace_rotation"], 0.0)


def test_gaussian_nll_is_finite_for_positive_definite_covariance():
    covariance = np.array([[1.0, 0.2], [0.2, 1.0]])
    value = gaussian_nll(np.array([0.01, -0.02]), covariance)
    assert np.isfinite(value)
