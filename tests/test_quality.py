import pandas as pd
import pytest

from credit_scorecard import generate, quality


@pytest.fixture(scope="module")
def report(small_df):
    return quality.run_checks(small_df)


def test_finds_duplicate_ids(report):
    assert report.counts["duplicate_ids"] == generate.DQ_PLAN["duplicates"]
    assert report.by_check("duplicates")[0].severity == "critical"


def test_flags_the_leaky_post_origination_column(report):
    flagged = report.leakage.loc[report.leakage["flagged"], "column"].tolist()
    assert flagged == ["max_dpd_post_orig"]
    assert report.leakage.iloc[0]["single_var_auc"] > 0.95


def test_legitimate_features_are_not_flagged_as_leaky(report):
    ok = report.leakage[~report.leakage["flagged"]]
    assert ok["single_var_auc"].max() < 0.85 and len(ok) >= 8


def test_finds_impossible_values_and_sentinels(report):
    plan = generate.DQ_PLAN
    assert report.counts["range:age"] == plan["bad_age"]
    assert report.counts["range:revol_util"] == plan["util_sentinel"]
    assert report.counts["range:income"] == plan["negative_income"]
    assert report.counts["sentinel:income"] == plan["income_sentinel"]
    assert report.counts["sentinel:revol_util"] == plan["util_sentinel"]
    assert report.counts["invalid_term"] == plan["bad_term"]


def test_finds_inconsistent_employment_length(report):
    assert report.counts["emp_gt_age"] >= generate.DQ_PLAN["emp_gt_age"]


def test_detects_feed_outage_month(report):
    flagged = report.vintages.loc[report.vintages["volume_flag"], "month"].tolist()
    assert flagged == ["2022-03"]
    assert len(report.vintages) == 36


def test_target_rate_by_vintage_table(report):
    v = report.vintages
    assert {"month", "n", "bad_rate"} <= set(v.columns)
    assert v["bad_rate"].iloc[-6:].mean() > v["bad_rate"].iloc[:6].mean()


def test_missingness_flagged_as_informative():
    df = generate.generate_loans(n=30000, seed=3, with_dq_issues=False)
    rep = quality.QualityReport(n_rows=len(df))
    quality.check_missingness(df, rep)
    mt = rep.missingness.set_index("column")
    assert bool(mt.loc["revol_util", "informative"]) and bool(mt.loc["emp_length", "informative"])
    assert mt.loc["revol_util", "bad_rate_missing"] > mt.loc["revol_util", "bad_rate_present"]
    assert any("informative" in i.message for i in rep.by_check("missingness"))


def test_clean_data_applies_remediation(small_df, report):
    clean, actions = quality.clean_data(small_df, report)
    assert clean["loan_id"].is_unique
    assert "max_dpd_post_orig" not in clean
    assert clean["term"].isin([36, 60]).all()
    assert (clean["income"].dropna() > 0).all() and (clean["revol_util"].dropna() <= 3).all()
    assert clean["age"].dropna().between(18, 100).all()
    assert len(actions) >= 6 and len(small_df) - len(clean) == 30 + 6


def test_clean_frame_passes_range_checks(small_df, report):
    clean, _ = quality.clean_data(small_df, report)
    again = quality.QualityReport(n_rows=len(clean))
    quality.check_ranges(clean, again)
    quality.check_duplicates(clean, again)
    assert not [i for i in again.issues if i.severity == "critical"]


def test_dq_free_data_only_trips_the_leakage_check():
    df = generate.generate_loans(n=4000, seed=21, with_dq_issues=False)
    rep = quality.run_checks(df)
    assert {i.check for i in rep.issues if i.severity == "critical"} == {"leakage"}


def test_leakage_check_catches_any_near_perfect_feature():
    df = pd.DataFrame({"default": [0, 1] * 200, "perfect": [0.0, 1.0] * 200,
                       "noise": range(400)})
    rep = quality.QualityReport(n_rows=400)
    quality.check_leakage(df, rep)
    assert rep.leakage.set_index("column").loc["perfect", "flagged"]
