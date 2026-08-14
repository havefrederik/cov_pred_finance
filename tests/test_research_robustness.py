from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.research.robustness import (
    newey_west_se,
    summarize_improvement,
    trimmed_mean,
)


def test_newey_west_se_is_positive_for_nonconstant_series():
    values = pd.Series([1.0, -0.5, 0.2, 1.1, -0.3, 0.7])
    assert newey_west_se(values, lag=2) > 0


def test_trimmed_mean_drops_extreme_tails():
    values = [0.0] * 98 + [-100.0, 100.0]
    assert np.isclose(trimmed_mean(values, 0.01), 0.0)


def test_summary_reports_yearly_means_and_hac_statistics():
    index = pd.date_range("2020-01-01", periods=8, freq="180D")
    values = pd.Series(np.arange(8, dtype=float), index=index)
    result = summarize_improvement(values, hac_lags=(1,))
    assert result["n_obs"] == 8
    assert "2020" in result["yearly_mean"]
    assert np.isfinite(result["hac_t_1"])
