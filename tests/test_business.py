import numpy as np
import pytest

from credit_scorecard import business as b


@pytest.fixture
def econ():
    return b.Economics()


def test_economics_hand_computed(econ):
    assert econ.revenue(10_000, 36) == pytest.approx(10_000 * 0.055 * 0.55 * 3)
    assert econ.revenue(10_000, 36) == pytest.approx(907.5)
    assert econ.ead(10_000) == pytest.approx(8_500)
    assert econ.loss_if_default(10_000) == pytest.approx(5_950)


def test_cutoff_analysis_hand_computed(econ):
    score, y = np.array([500.0, 600.0, 700.0]), np.array([1, 0, 0])
    amount, term = np.full(3, 1000.0), np.full(3, 12)
    t = b.cutoff_analysis(score, y, amount, term, econ, step=100.0).set_index("cutoff")
    good, loss = 1000 * 0.055 * 0.55, 1000 * 0.85 * 0.7  # 30.25 and 595
    assert t.loc[500, "profit"] == pytest.approx(2 * good - loss)
    assert t.loc[600, "profit"] == pytest.approx(2 * good)
    assert t.loc[700, "profit"] == pytest.approx(good)
    assert t.loc[500, "approval_rate"] == 1.0
    assert t.loc[600, "approval_rate"] == pytest.approx(2 / 3)
    assert t.loc[500, "bad_rate_approved"] == pytest.approx(1 / 3)
    assert t.loc[600, "bad_rate_approved"] == 0.0
    assert b.optimal_cutoff(t.reset_index()) == 600.0


def test_cutoff_above_all_scores_approves_nobody(econ):
    t = b.cutoff_analysis(np.array([500.0, 510.0]), np.array([0, 1]), np.full(2, 1000.0),
                          np.full(2, 36), econ, step=100.0)
    assert t.iloc[-1]["approval_rate"] == 0.0 and t.iloc[-1]["profit"] == 0.0


def test_evaluate_cutoff_agrees_with_table(econ, rng):
    score = rng.normal(600, 40, 2000)
    y = (rng.random(2000) < 1 / (1 + np.exp((score - 520) / 25))).astype(int)
    amount, term = rng.uniform(2000, 20000, 2000), rng.choice([36, 60], 2000)
    t = b.cutoff_analysis(score, y, amount, term, econ, step=10.0)
    row = t.iloc[len(t) // 2]
    ev = b.evaluate_cutoff(score, y, amount, term, econ, row["cutoff"])
    assert ev["profit"] == pytest.approx(row["profit"])
    assert ev["approval_rate"] == pytest.approx(row["approval_rate"])


def test_profit_is_unimodal_enough_to_have_interior_optimum(econ, rng):
    score = rng.normal(600, 50, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp((score - 540) / 30))).astype(int)
    t = b.cutoff_analysis(score, y, np.full(20000, 5000.0), np.full(20000, 36), econ)
    best = b.optimal_cutoff(t)
    assert t["cutoff"].min() < best < t["cutoff"].max()


def test_expected_loss_by_band_accounting(econ, rng):
    n = 3000
    score = rng.normal(600, 40, n)
    p = 1 / (1 + np.exp((score - 520) / 25))
    y = (rng.random(n) < p).astype(int)
    amount = rng.uniform(2000, 20000, n)
    t = b.expected_loss_by_band(score, p, y, amount, econ, b.band_edges(score, 5))
    body, total = t.iloc[:-1], t.iloc[-1]
    assert len(body) == 5 and body["n"].sum() == n == total["n"]
    assert total["expected_loss"] == pytest.approx(np.sum(p * 0.70 * 0.85 * amount))
    assert total["ead"] == pytest.approx(0.85 * amount.sum())
    assert body["avg_pd"].is_monotonic_decreasing  # first band is the lowest scores
    assert total["el_rate"] == pytest.approx(total["expected_loss"] / total["ead"])


def test_ifrs9_staging_invariants(econ, rng):
    n = 4000
    p0 = rng.uniform(0.01, 0.2, n)
    default = (rng.random(n) < p0).astype(int)
    amount, term = rng.uniform(2000, 20000, n), rng.choice([36, 60], n)
    out = b.ifrs9_illustration(p0, default, amount, term, econ, seed=3)
    t = out["table"].set_index("stage")
    assert t["share"].sum() == pytest.approx(1.0) and t["n"].sum() == n
    assert t.loc[3, "n"] == default.sum()  # every defaulted loan is stage 3
    assert t.loc[3, "ecl"] == pytest.approx(0.70 * econ.ead(amount)[default == 1].sum())
    assert t.loc[1, "coverage"] < t.loc[2, "coverage"] < t.loc[3, "coverage"]
    assert out["total_ecl"] == pytest.approx(t["ecl"].sum())


def test_ifrs9_is_deterministic_by_seed(econ, rng):
    p0 = rng.uniform(0.01, 0.2, 500)
    args = (p0, np.zeros(500), np.full(500, 5000.0), np.full(500, 36), econ)
    a, c = b.ifrs9_illustration(*args, seed=1), b.ifrs9_illustration(*args, seed=1)
    assert a["total_ecl"] == c["total_ecl"]
    assert b.ifrs9_illustration(*args, seed=2)["total_ecl"] != a["total_ecl"]


def test_adverse_impact_ratio_hand_computed():
    group = np.array(["a"] * 10 + ["b"] * 10)
    approved = np.array([True] * 8 + [False] * 2 + [True] * 4 + [False] * 6)
    t = b.adverse_impact_screen(group, approved).set_index("group")
    assert t.loc["a", "impact_ratio"] == 1.0
    assert t.loc["b", "impact_ratio"] == pytest.approx(0.5)
    assert t.loc["b", "approval_rate"] == pytest.approx(0.4)
