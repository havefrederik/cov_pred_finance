from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.research.walkforward import EXPERT_NAMES
from experiments.research.walkforward_shape import (
    fast_slow_relative_geometry_features,
    temporal_relative_geometry_features,
)


def test_temporal_relative_shape_is_zero_for_common_rescaling():
    times = pd.date_range("2020-01-01", periods=2, freq="D")
    sigma = np.array([[2.0, 0.4], [0.4, 1.0]])
    covs = {
        times[0]: pd.DataFrame(sigma),
        times[1]: pd.DataFrame(3.0 * sigma),
    }
    features = temporal_relative_geometry_features(covs, horizons=(1,))
    assert np.isclose(
        features.loc[times[1], "temprel__1d__log_scale"],
        np.log(3.0),
    )
    assert np.isclose(
        features.loc[times[1], "temprel__1d__log_shape_std"],
        0.0,
        atol=1e-12,
    )
    assert np.isclose(
        features.loc[times[1], "temprel__1d__log_condition"],
        0.0,
        atol=1e-12,
    )


def test_temporal_relative_shape_detects_nonproportional_deformation():
    times = pd.date_range("2020-01-01", periods=2, freq="D")
    covs = {
        times[0]: pd.DataFrame(np.eye(2)),
        times[1]: pd.DataFrame(np.diag([2.0, 0.5])),
    }
    features = temporal_relative_geometry_features(covs, horizons=(1,))
    assert features.loc[times[1], "temprel__1d__log_shape_std"] > 0
    assert np.isclose(
        features.loc[times[1], "temprel__1d__log_condition"],
        np.log(4.0),
    )


def test_fast_slow_shape_is_scale_invariant():
    time = pd.Timestamp("2020-01-01")
    sigma = np.array([[1.5, 0.2], [0.2, 0.8]])
    experts = {
        name: {time: pd.DataFrame(sigma)}
        for name in EXPERT_NAMES
    }
    experts[EXPERT_NAMES[0]][time] = pd.DataFrame(4.0 * sigma)
    features = fast_slow_relative_geometry_features(experts)
    assert np.isclose(
        features.loc[time, "fastslow__log_scale"],
        np.log(4.0),
    )
    assert np.isclose(
        features.loc[time, "fastslow__log_shape_std"],
        0.0,
        atol=1e-12,
    )
    assert np.isclose(
        features.loc[time, "fastslow__log_condition"],
        0.0,
        atol=1e-12,
    )
