import numpy as np
import pandas as pd

from credit_scorecard.binning import bin_features, woe_matrix
from credit_scorecard.selection import select_features


def _frame(seed=0, n=8000):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "a_copy": a + rng.normal(0, 0.05, n),
                       "b": rng.normal(size=n), "noise": rng.normal(size=n)})
    p = 1 / (1 + np.exp(-(-2.5 + 1.1 * df["a"] + 0.5 * df["b"])))
    return df, (rng.random(n) < p).astype(int)


def test_weak_feature_dropped_by_iv():
    df, y = _frame()
    bins = bin_features(df, y, list(df.columns))
    sel = select_features(bins, woe_matrix(df, bins), min_iv=0.05)
    assert "noise" not in sel.selected
    assert any(f == "noise" and "IV" in why for f, why in sel.dropped)


def test_correlated_duplicate_is_pruned_keeping_higher_iv():
    df, y = _frame(1)
    bins = bin_features(df, y, list(df.columns))
    sel = select_features(bins, woe_matrix(df, bins), min_iv=0.02, max_corr=0.7)
    assert ("a" in sel.selected) != ("a_copy" in sel.selected)  # exactly one survives
    survivor = "a" if "a" in sel.selected else "a_copy"
    other = "a_copy" if survivor == "a" else "a"
    assert bins[survivor].iv >= bins[other].iv
    assert "b" in sel.selected


def test_suspiciously_high_iv_raises_warning():
    rng = np.random.default_rng(3)
    y = (rng.random(5000) < 0.1).astype(int)
    df = pd.DataFrame({"leak": 3 * y + rng.normal(0, 1.0, 5000), "ok": rng.normal(size=5000)})
    bins = bin_features(df, y, ["leak", "ok"])
    sel = select_features(bins, woe_matrix(df, bins))
    assert any("leak" in w for w in sel.warnings)
