import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.integrate import trapezoid

from credit_scorecard import metrics as m


def _brute_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


Y4 = np.array([0, 0, 1, 1])
R4 = np.array([0.1, 0.4, 0.35, 0.8])


def test_auc_classic_four_point_example():
    assert m.auc(Y4, R4) == pytest.approx(0.75)


def test_auc_perfect_inverted_and_tied():
    assert m.auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert m.auc([0, 0, 1, 1], [4, 3, 2, 1]) == 0.0
    assert m.auc([0, 1], [5, 5]) == 0.5


def test_auc_matches_brute_force_with_ties(rng):
    y = rng.integers(0, 2, 400)
    s = rng.integers(0, 15, 400).astype(float)  # many ties
    assert m.auc(y, s) == pytest.approx(_brute_auc(y, s))


def test_gini_is_two_auc_minus_one():
    assert m.gini(Y4, R4) == pytest.approx(0.5)
    assert m.gini([0, 1], [1, 2]) == 1.0


def test_binary_validation():
    with pytest.raises(ValueError):
        m.auc([0, 0, 0], [1, 2, 3])
    with pytest.raises(ValueError):
        m.auc([0, 2, 1], [1, 2, 3])


def test_ks_hand_computed():
    assert m.ks_statistic(Y4, R4) == pytest.approx(0.5)


def test_ks_perfect_and_degenerate():
    assert m.ks_statistic([0, 0, 1, 1], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert m.ks_statistic([0, 1], [7, 7]) == pytest.approx(0.0)


def test_ks_matches_scipy_two_sample(rng):
    y = rng.integers(0, 2, 2000)
    s = rng.normal(size=2000) + 0.7 * y
    expected = stats.ks_2samp(s[y == 1], s[y == 0]).statistic
    assert m.ks_statistic(y, s) == pytest.approx(expected)


def test_roc_curve_endpoints_and_area(rng):
    y = rng.integers(0, 2, 1000)
    s = rng.normal(size=1000) + y
    fpr, tpr = m.roc_curve(y, s)
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert trapezoid(tpr, fpr) == pytest.approx(m.auc(y, s))


def test_brier_hand_computed():
    assert m.brier([1, 0], [0.8, 0.4]) == pytest.approx((0.04 + 0.16) / 2)


def test_psi_identical_distributions_is_zero(rng):
    x = rng.normal(size=5000)
    assert m.psi(x, x.copy()) == pytest.approx(0.0, abs=1e-12)
    assert m.psi_from_proportions([0.2, 0.8], [0.2, 0.8]) == 0.0


def test_psi_hand_computed():
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert m.psi_from_proportions([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected)
    assert expected == pytest.approx(0.27465, abs=1e-5)


def test_psi_is_symmetric_and_non_negative():
    e, a = np.array([0.1, 0.4, 0.5]), np.array([0.3, 0.3, 0.4])
    assert m.psi_from_proportions(e, a) == pytest.approx(m.psi_from_proportions(a, e))
    assert m.psi_from_proportions(e, a) > 0


def test_psi_shifted_distribution_is_large(rng):
    x = rng.normal(0, 1, 20000)
    assert m.psi(x, rng.normal(1.0, 1, 20000)) > 0.25
    assert m.psi(x, rng.normal(0.05, 1, 20000)) < 0.02


def test_psi_empty_bin_is_finite_thanks_to_floor():
    assert np.isfinite(m.psi_from_counts([50, 50, 0], [0, 50, 50]))


def test_psi_from_counts_equals_proportions():
    assert m.psi_from_counts([20, 80], [40, 60]) == pytest.approx(
        m.psi_from_proportions([0.2, 0.8], [0.4, 0.6]))


@pytest.mark.parametrize("v,label", [(0.05, "stable"), (0.15, "moderate shift"),
                                     (0.30, "significant shift")])
def test_psi_labels(v, label):
    assert m.psi_label(v) == label


def test_calibration_table_accounting(rng):
    p = rng.uniform(0.01, 0.3, 5000)
    y = (rng.random(5000) < p).astype(int)
    t = m.calibration_table(y, p)
    assert t["n"].sum() == 5000 and t["bads"].sum() == y.sum() and len(t) == 10
    assert t["mean_pd"].is_monotonic_increasing
    assert abs(np.average(t["obs_rate"], weights=t["n"]) - y.mean()) < 1e-12
    assert t["ratio"].between(0.7, 1.4).all()  # well calibrated by construction


def test_hosmer_lemeshow_hand_computed():
    t = pd.DataFrame({"n": [100, 100, 100], "mean_pd": [0.1, 0.2, 0.3], "bads": [14, 20, 30]})
    chi2, p = m.hosmer_lemeshow(t)
    assert chi2 == pytest.approx(16 / 9)
    assert p == pytest.approx(stats.chi2.sf(16 / 9, 1))


def test_lift_table_hand_computed():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    risk = np.arange(10, 0, -1).astype(float)  # first rows are riskiest
    t = m.lift_table(y, risk, n_bands=2)
    assert t["bad_rate"].tolist() == [0.4, 0.0]
    assert t["lift"].tolist() == [2.0, 0.0]
    assert t["cum_pct_bads"].tolist() == [1.0, 1.0]
    assert t["cum_pct_pop"].tolist() == [0.5, 1.0]


def test_lift_cumulative_bads_reaches_one(rng):
    y = rng.integers(0, 2, 1000)
    t = m.lift_table(y, rng.normal(size=1000) + y)
    assert t["cum_pct_bads"].iloc[-1] == pytest.approx(1.0)
    assert t["lift"].iloc[0] > 1 > t["lift"].iloc[-1]


def test_bootstrap_auc_interval(rng):
    y = rng.integers(0, 2, 1500)
    s = rng.normal(size=1500) + 0.8 * y
    point, lo, hi = m.bootstrap_auc(y, s, n_boot=200, seed=1)
    assert lo < point < hi and hi - lo < 0.08
    assert m.bootstrap_auc(y, s, n_boot=200, seed=1) == (point, lo, hi)  # deterministic
    _, lo2, hi2 = m.bootstrap_auc(y[:300], s[:300], n_boot=200, seed=1)
    assert hi2 - lo2 > hi - lo  # smaller sample, wider interval


def test_bootstrap_auc_diff_of_identical_scores_is_zero(rng):
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + y
    assert m.bootstrap_auc_diff(y, s, s, n_boot=50) == (0.0, 0.0, 0.0)
