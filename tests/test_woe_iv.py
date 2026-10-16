import numpy as np
import pytest

from credit_scorecard.binning import iv_strength, woe_iv


def test_woe_iv_hand_computed():
    # goods 80/20 of 100, bads 10/40 of 50: pg = .8/.2, pb = .2/.8
    woe, terms = woe_iv([80, 20], [10, 40])
    assert woe == pytest.approx([np.log(4), np.log(0.25)])
    assert terms.sum() == pytest.approx(0.6 * np.log(4) * 2)
    assert terms.sum() == pytest.approx(1.6636, abs=1e-4)


def test_woe_three_bins_hand_computed():
    good, bad = [50, 30, 20], [5, 10, 35]
    woe, terms = woe_iv(good, bad)
    pg = np.array(good) / 100
    pb = np.array(bad) / 50
    assert woe == pytest.approx(np.log(pg / pb))
    assert terms.sum() == pytest.approx(np.sum((pg - pb) * np.log(pg / pb)))


def test_iv_zero_when_distributions_match():
    woe, terms = woe_iv([10, 20, 30], [1, 2, 3])
    assert woe == pytest.approx(0.0)
    assert terms.sum() == pytest.approx(0.0)


def test_iv_is_non_negative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        good, bad = rng.integers(1, 500, 6), rng.integers(1, 500, 6)
        assert woe_iv(good, bad)[1].sum() >= 0.0


def test_zero_cell_is_infinite_without_smoothing():
    woe, _ = woe_iv([10, 10], [0, 5])
    assert np.isinf(woe[0]) and woe[0] > 0


def test_smoothing_keeps_woe_finite():
    woe, terms = woe_iv([10, 10], [0, 5], smoothing=0.5)
    assert np.isfinite(woe).all() and np.isfinite(terms).all()


def test_higher_bad_rate_means_lower_woe():
    good, bad = np.array([900, 800, 600, 300]), np.array([10, 40, 120, 300])
    woe, _ = woe_iv(good, bad)
    assert np.all(np.diff(woe) < 0)


def test_woe_sign_convention_positive_means_safer():
    woe, _ = woe_iv([90, 10], [5, 45])
    assert woe[0] > 0 > woe[1]


@pytest.mark.parametrize("iv,label", [(0.01, "not useful"), (0.05, "weak"), (0.2, "medium"),
                                      (0.4, "strong")])
def test_iv_strength_bands(iv, label):
    assert iv_strength(iv) == label


def test_iv_strength_flags_suspicious():
    assert "suspicious" in iv_strength(0.9)
