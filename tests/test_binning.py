import numpy as np
import pandas as pd
import pytest

from credit_scorecard.binning import (
    FeatureBinning,
    _merge_bins,
    bin_features,
    fit_binning,
    is_monotone,
    iv_table,
    woe_iv,
    woe_matrix,
)


def _y(counts_rates):
    """Exact 0/1 outcomes: [(n, n_bad), ...] concatenated."""
    return np.concatenate([np.r_[np.ones(b), np.zeros(n - b)] for n, b in counts_rates])


def test_pava_merge_hand_example():
    n, bad = [10, 10, 10, 10], [1, 3, 2, 5]  # rates .1 .3 .2 .5 : bins 1,2 violate
    spans = _merge_bins(np.array(n), np.array(bad), 1, 1, 0, 10)
    assert spans == [(0, 0), (1, 2), (3, 3)]


def test_pava_merge_decreasing_direction():
    spans = _merge_bins(np.array([10] * 4), np.array([5, 2, 3, 1]), -1, 1, 0, 10)
    assert spans == [(0, 0), (1, 2), (3, 3)]


def test_hand_computed_binning_table():
    x = np.repeat([1, 2, 3], 100)
    y = _y([(100, 5), (100, 10), (100, 20)])
    b = fit_binning(x, y, "x", min_bin_frac=0.01, min_class=1)
    assert b.edges == [1.0, 2.0]
    assert b.good[:3] == [95, 90, 80] and b.bad[:3] == [5, 10, 20]
    woe, terms = woe_iv([95, 90, 80], [5, 10, 20], smoothing=0.5)
    assert b.woe[:3] == pytest.approx(woe)
    assert b.iv == pytest.approx(terms.sum())


def test_edge_value_goes_to_left_bin():
    b = fit_binning(np.repeat([1, 2, 3], 100), _y([(100, 5), (100, 10), (100, 20)]), "x",
                    min_bin_frac=0.01, min_class=1)
    assert list(b.assign([1, 1.5, 2, 2.0001, 3, 99])) == [0, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("seed", range(6))
def test_monotone_invariant_on_non_monotone_truth(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=6000)
    p = 1 / (1 + np.exp(-(-2.2 + 0.5 * x + 0.6 * np.sin(3 * x))))
    y = (rng.random(6000) < p).astype(int)
    b = fit_binning(x, y, "x", min_bin_frac=0.03)
    t = b.table().iloc[:-1]
    assert is_monotone(t["bad_rate"], strict=True)
    assert (t["count"] >= int(np.ceil(0.03 * 6000))).all() or len(t) == 1


def test_u_shape_is_forced_monotone():
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, 8000)
    y = (rng.random(8000) < 1 / (1 + np.exp(-(-2.5 + 0.5 * x**2)))).astype(int)
    b = fit_binning(x, y, "x")
    assert is_monotone(b.table().iloc[:-1]["bad_rate"])


def test_direction_detected_for_decreasing_risk():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(-2 - 0.9 * x)))).astype(int)
    b = fit_binning(x, y, "x")
    assert b.direction == -1
    assert np.all(np.diff(b.table().iloc[:-1]["bad_rate"]) < 0)
    assert np.all(np.diff(b.woe[:-1]) > 0)  # WoE rises as risk falls


def test_max_bins_is_respected():
    rng = np.random.default_rng(3)
    x = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(-2.5 + 1.2 * x)))).astype(int)
    assert fit_binning(x, y, "x", max_bins=4).n_bins <= 4
    assert fit_binning(x, y, "x", max_bins=10).n_bins > 4


def test_min_bin_size_is_respected():
    rng = np.random.default_rng(4)
    x = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(-2 + 0.8 * x)))).astype(int)
    b = fit_binning(x, y, "x", min_bin_frac=0.10)
    assert (b.table().iloc[:-1]["count"] >= 400).all()


def test_merging_cannot_increase_iv():
    rng = np.random.default_rng(5)
    x = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(-2.5 + 0.9 * x)))).astype(int)
    fine = fit_binning(x, y, "x", max_bins=12, min_bin_frac=0.02)
    coarse = fit_binning(x, y, "x", max_bins=3, min_bin_frac=0.02)
    assert fine.iv >= coarse.iv - 1e-9


def test_missing_values_get_special_bin_with_own_woe():
    rng = np.random.default_rng(6)
    x = rng.normal(size=6000)
    y = (rng.random(6000) < 0.05).astype(int)
    miss = rng.random(6000) < 0.1
    y[miss] = (rng.random(miss.sum()) < 0.30).astype(int)  # informative missingness
    x[miss] = np.nan
    b = fit_binning(x, y, "x")
    idx = b.assign(x)
    assert (idx[miss] == b.special_index).all() and (idx[~miss] < b.special_index).all()
    assert b.woe[-1] < min(b.woe[:-1])  # missing rows are the riskiest group
    assert b.transform(np.array([np.nan]))[0] == pytest.approx(b.woe[-1])


def test_special_bin_is_neutral_when_no_missing_in_training():
    rng = np.random.default_rng(7)
    x = rng.normal(size=3000)
    y = (rng.random(3000) < 0.1).astype(int)
    b = fit_binning(x, y, "x")
    assert b.woe[-1] == 0.0
    assert b.transform(np.array([np.nan]))[0] == 0.0


def test_categorical_rare_level_merged_with_closest_risk_neighbour():
    levels = {"A": (500, 25), "B": (500, 50), "C": (500, 100), "D": (10, 5)}
    x = np.concatenate([[k] * n for k, (n, _) in levels.items()])
    y = _y(list(levels.values()))
    b = fit_binning(pd.Series(x), y, "cat", min_bin_frac=0.02)
    group_of = {lvl: i for i, g in enumerate(b.groups) for lvl in g}
    assert group_of["D"] == group_of["C"]  # D (50% bad) is closest to C (20% bad)
    rates = b.table().iloc[:-1]["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) > 0)


def test_categorical_unseen_level_goes_to_special_bin():
    x = np.array(["a"] * 100 + ["b"] * 100)
    y = _y([(100, 5), (100, 30)])
    b = fit_binning(pd.Series(x), y, "c", min_bin_frac=0.05)
    assert list(b.assign(pd.Series(["a", "b", "zzz", np.nan]))) == [0, 1, 2, 2]


def test_constant_feature_has_single_bin_and_zero_iv():
    b = fit_binning(np.ones(500), np.r_[np.ones(50), np.zeros(450)], "c")
    assert b.n_bins == 1 and b.iv == pytest.approx(0.0, abs=1e-9)


def test_table_proportions_sum_to_one():
    rng = np.random.default_rng(8)
    x = rng.normal(size=3000)
    y = (rng.random(3000) < 0.1).astype(int)
    t = fit_binning(x, y, "x").table()
    assert t["pct_good"].sum() == pytest.approx(1.0) and t["pct_bad"].sum() == pytest.approx(1.0)
    assert t["count"].sum() == 3000


def test_serialisation_round_trip():
    rng = np.random.default_rng(9)
    x = rng.normal(size=3000)
    y = (rng.random(3000) < 1 / (1 + np.exp(-(-2 + x)))).astype(int)
    b = fit_binning(x, y, "x")
    b2 = FeatureBinning.from_dict(b.to_dict())
    assert np.array_equal(b.assign(x), b2.assign(x))
    assert b2.iv == b.iv


def test_all_missing_feature_raises():
    with pytest.raises(ValueError):
        fit_binning(np.full(10, np.nan), np.r_[np.ones(5), np.zeros(5)], "x")


def test_bin_features_woe_matrix_and_iv_table(small_df):
    df = small_df.dropna(subset=["dti", "income"])
    feats = ["dti", "income", "purpose"]
    binnings = bin_features(df, df["default"].to_numpy(), feats)
    woe = woe_matrix(df, binnings)
    assert list(woe.columns) == feats and len(woe) == len(df)
    t = iv_table(binnings)
    assert t["iv"].is_monotonic_decreasing and set(t["feature"]) == set(feats)


def test_is_monotone_helper():
    assert is_monotone([1, 2, 3]) and is_monotone([3, 2, 1])
    assert not is_monotone([1, 3, 2])
    assert not is_monotone([1, 1, 2], strict=True) and is_monotone([1, 1, 2], strict=False)
    assert is_monotone([1, 2, 3], direction=1) and not is_monotone([1, 2, 3], direction=-1)
