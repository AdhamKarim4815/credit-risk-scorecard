import numpy as np
import pandas as pd
import pytest

from credit_scorecard import pipeline
from credit_scorecard.binning import FeatureBinning
from credit_scorecard.scorecard import REASON_TEXT, Scaling, Scorecard


@pytest.fixture(scope="module")
def card_and_data(mini_run):
    cfg, _ = mini_run
    return Scorecard.load(cfg.artifacts / "scorecard.json"), pipeline.prepare(cfg).oot


@pytest.fixture
def hand_card():
    """A two-characteristic scorecard small enough to verify by hand."""
    x = FeatureBinning(name="x", kind="numeric", edges=[0.0], good=[50, 50, 0], bad=[10, 40, 0],
                       woe=[0.5, -0.5, 0.0])
    c = FeatureBinning(name="c", kind="categorical", groups=[["a"], ["b"]], good=[50, 50, 0],
                       bad=[10, 40, 0], woe=[0.8, -0.8, 0.0])
    return Scorecard({"x": x, "c": c}, intercept=-2.0, coefs={"x": -1.0, "c": -0.5})


def test_scaling_base_point_and_doubling_identity():
    s = Scaling(600, 30, 20)
    assert s.score_from_pd(1 / 31) == pytest.approx(600)  # odds 30:1 -> 600 points
    assert s.score_from_pd(1 / 61) == pytest.approx(620)  # odds 60:1 -> +PDO
    assert s.score_from_pd(1 / 16) == pytest.approx(580)  # odds 15:1 -> -PDO


@pytest.mark.parametrize("pdo", [10, 20, 50])
def test_odds_double_every_pdo_points(pdo):
    s = Scaling(600, 30, pdo)
    scores = np.array([500.0, 550.0, 600.0, 640.0])
    odds = (1 - s.pd_from_score(scores)) / s.pd_from_score(scores)
    odds_plus = (1 - s.pd_from_score(scores + pdo)) / s.pd_from_score(scores + pdo)
    assert odds_plus == pytest.approx(2 * odds)


def test_score_pd_round_trip():
    s = Scaling()
    p = np.array([0.001, 0.02, 0.2, 0.7])
    assert s.pd_from_score(s.score_from_pd(p)) == pytest.approx(p)


def test_factor_and_offset_formulas():
    s = Scaling(600, 30, 20)
    assert s.factor == pytest.approx(20 / np.log(2))
    assert s.offset == pytest.approx(600 - s.factor * np.log(30))


def test_hand_computed_score_and_pd(hand_card):
    df = pd.DataFrame({"x": [1.0, -1.0], "c": ["a", "b"]})
    s = hand_card.scaling
    eta = np.array([-2 + 0.5 - 0.4, -2 - 0.5 + 0.4])  # logit(PD) by hand
    assert hand_card.predict_pd(df) == pytest.approx(1 / (1 + np.exp(-eta)))
    assert hand_card.score(df) == pytest.approx(s.offset + s.factor * (-eta))


def test_points_sum_to_score_and_match_pd_scaling(card_and_data):
    card, df = card_and_data
    assert card.points_matrix(df).sum(axis=1).to_numpy() == pytest.approx(card.score(df))
    assert card.scaling.score_from_pd(card.predict_pd(df)) == pytest.approx(card.score(df))


def test_pairwise_odds_ratio_follows_pdo(card_and_data):
    card, df = card_and_data
    s, p = card.score(df.head(50)), card.predict_pd(df.head(50))
    odds = (1 - p) / p
    assert odds[1:] / odds[0] == pytest.approx(2 ** ((s[1:] - s[0]) / card.scaling.pdo))


def test_higher_score_means_lower_pd(card_and_data):
    card, df = card_and_data
    assert np.corrcoef(card.score(df), card.predict_pd(df))[0, 1] < -0.8
    assert card.coefs and all(c < 0 for c in card.coefs.values())  # WoE sign convention


def test_points_table_structure(card_and_data):
    card, _ = card_and_data
    t = card.points_table()
    assert set(t["characteristic"]) == set(card.features)
    assert (t.groupby("characteristic")["attribute"].last() == "Missing / unseen").all()
    reg = t[t["attribute"] != "Missing / unseen"]
    for _, g in reg.groupby("characteristic"):  # WoE and points move together (coef < 0)
        assert np.corrcoef(g["woe"], g["points"])[0, 1] > 0.99 or len(g) < 3


def test_score_applicant_from_dict_with_missing_and_unseen(card_and_data):
    card, _ = card_and_data
    row = dict.fromkeys(card.features, np.nan)
    row["purpose"] = "never_seen_before"
    out = card.score_frame(pd.DataFrame([row]))
    assert np.isfinite(out["score"].iloc[0]) and 0 < out["pd"].iloc[0] < 1


def test_reason_codes_hand_computed(hand_card):
    df = pd.DataFrame({"x": [1.0], "c": ["a"]})
    reasons = hand_card.reason_codes(df)[0]
    assert [r["characteristic"] for r in reasons] == ["x"]  # c is already at its best bin
    assert reasons[0]["points_lost"] == pytest.approx(hand_card.scaling.factor, abs=0.06)


def test_reason_codes_top_three_ranked(card_and_data):
    card, df = card_and_data
    sample = df.head(40)
    pts, best = card.points_matrix(sample), pd.Series(card.max_points())
    for i, reasons in enumerate(card.reason_codes(sample)):
        assert 1 <= len(reasons) <= 3
        lost = [r["points_lost"] for r in reasons]
        assert lost == sorted(lost, reverse=True)
        top = (best - pts.iloc[i]).clip(lower=0).sort_values(ascending=False).head(3)
        assert [r["characteristic"] for r in reasons][0] == top.index[0]
        assert all(r["code"].startswith("R") for r in reasons)


def test_reason_text_covers_model_features(card_and_data):
    card, _ = card_and_data
    assert set(card.features) <= set(REASON_TEXT)


def test_score_frame_columns(card_and_data):
    card, df = card_and_data
    out = card.score_frame(df.head(5))
    assert list(out.columns) == ["score", "pd", "reason_1", "reason_2", "reason_3"]
    assert out["reason_1"].str.startswith("R").all()


def test_save_load_round_trip(card_and_data, tmp_path):
    card, df = card_and_data
    card.save(tmp_path / "c.json")
    again = Scorecard.load(tmp_path / "c.json")
    assert again.score(df) == pytest.approx(card.score(df))
    assert again.reason_codes(df.head(10)) == card.reason_codes(df.head(10))
