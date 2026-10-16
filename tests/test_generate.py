import numpy as np
import pandas as pd
import pytest

from credit_scorecard import generate


def test_generator_is_deterministic():
    a = generate.generate_loans(n=3000, seed=42)
    b = generate.generate_loans(n=3000, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = generate.generate_loans(n=3000, seed=1)
    b = generate.generate_loans(n=3000, seed=2)
    assert not a["income"].equals(b["income"])


def test_columns_and_row_count(small_df):
    assert list(small_df.columns) == generate.COLUMNS
    assert len(small_df) == 6000 + generate.DQ_PLAN["duplicates"]


def test_no_personal_identifiers_in_columns(small_df):
    forbidden = {"name", "email", "phone", "address", "sin", "ssn", "sex", "gender", "race"}
    assert not forbidden & set(small_df.columns)


def test_three_years_of_vintages(small_df):
    months = pd.to_datetime(small_df["origination_date"]).dt.to_period("M").nunique()
    assert months == 36
    assert small_df["origination_date"].min() == "2021-01-01"
    assert small_df["origination_date"].max() == "2023-12-01"


def test_overall_bad_rate_near_target_and_rises_over_time():
    df = generate.generate_loans(n=30000, seed=3, with_dq_issues=False)
    assert 0.065 < df["default"].mean() < 0.095
    by_year = df.groupby(pd.to_datetime(df["origination_date"]).dt.year)["default"].mean()
    assert by_year.is_monotonic_increasing and by_year.iloc[-1] > 1.5 * by_year.iloc[0]


def test_applicant_mix_drifts_between_vintages():
    df = generate.generate_loans(n=30000, seed=3, with_dq_issues=False)
    year = pd.to_datetime(df["origination_date"]).dt.year
    assert df.loc[year == 2023, "revol_util"].mean() > df.loc[year == 2021, "revol_util"].mean()
    assert df.loc[year == 2023, "income"].median() < df.loc[year == 2021, "income"].median()


def test_missingness_is_present_and_informative():
    df = generate.generate_loans(n=30000, seed=3, with_dq_issues=False)
    for col in ("emp_length", "revol_util", "credit_file_age_months"):
        miss = df[col].isna()
        assert 0.005 < miss.mean() < 0.12
        assert df.loc[miss, "default"].mean() > df.loc[~miss, "default"].mean()


def test_planted_data_quality_problems_present(small_df):
    plan = generate.DQ_PLAN
    assert small_df.duplicated("loan_id").sum() == plan["duplicates"]
    assert (small_df["revol_util"] == 9.99).sum() == plan["util_sentinel"]
    assert (small_df["income"] == 999_999).sum() == plan["income_sentinel"]
    assert (small_df["income"] < 0).sum() == plan["negative_income"]
    assert (~small_df["term"].isin([36, 60])).sum() == plan["bad_term"]


def test_clean_generation_has_no_planted_problems():
    df = generate.generate_loans(n=3000, seed=4, with_dq_issues=False)
    assert len(df) == 3000 and df["loan_id"].is_unique
    assert df["term"].isin([36, 60]).all() and (df["income"] > 0).all()


def test_leaky_column_tracks_the_target(small_df):
    assert (small_df.loc[small_df["default"] == 1, "max_dpd_post_orig"] >= 90).mean() > 0.9
    assert (small_df.loc[small_df["default"] == 0, "max_dpd_post_orig"] >= 90).sum() == 0


def test_oracle_truth_aligned_and_informative():
    df, truth = generate.generate_loans(n=20000, seed=5, with_dq_issues=False, return_truth=True)
    from credit_scorecard.metrics import auc

    t = truth.reindex(df["loan_id"]).to_numpy()
    assert auc(df["default"], t) > 0.75
    assert abs(t.mean() - df["default"].mean()) < 0.01


def _base():
    n = 2
    return {"income": np.full(n, 52_000.0), "emp_length": np.full(n, 5.0),
            "dti": np.full(n, 0.3), "revol_util": np.full(n, 0.5),
            "delinq_24m": np.zeros(n), "credit_file_age_months": np.full(n, 84.0),
            "loan_amount": np.full(n, 8_000.0), "term": np.full(n, 36),
            "purpose": np.array(["auto"] * n), "home_ownership": np.array(["MORTGAGE"] * n),
            "region": np.array(["Ontario"] * n), "emp_missing": np.zeros(n),
            "util_missing": np.zeros(n), "file_missing": np.zeros(n),
            "unemployment": np.full(n, 5.5)}


def test_planted_logit_directions():
    base = generate.planted_logit(_base())[0]
    for col, worse in (("revol_util", 0.95), ("dti", 0.6), ("delinq_24m", 3.0),
                       ("income", 30_000.0), ("loan_amount", 25_000.0)):
        f = _base()
        f[col] = np.full(2, worse)
        assert generate.planted_logit(f)[0] > base, col


def test_planted_logit_has_interaction_and_hinge():
    f = _base()
    lo = generate.planted_logit(f)[0]
    f1, f2, f3 = _base(), _base(), _base()
    f1["revol_util"][:] = 0.9
    f2["dti"][:] = 0.5
    f3["revol_util"][:], f3["dti"][:] = 0.9, 0.5
    joint = generate.planted_logit(f3)[0] - lo
    separate = (generate.planted_logit(f1)[0] - lo) + (generate.planted_logit(f2)[0] - lo)
    assert joint > separate + 1.0  # interaction adds a clear premium
    f4, f5 = _base(), _base()
    f4["dti"][:], f5["dti"][:] = 0.2, 0.28
    # flat below the hinge at DTI 0.28
    assert generate.planted_logit(f4)[0] == pytest.approx(generate.planted_logit(f5)[0])


def test_planted_process_ignores_age():
    import inspect

    assert "f[\"age\"]" not in inspect.getsource(generate.planted_logit)


def test_write_dataset_round_trip_and_size(tmp_path):
    df = generate.write_dataset(tmp_path / "d.csv", n=2500, seed=9)
    back = pd.read_csv(tmp_path / "d.csv")
    assert len(back) == len(df) and list(back.columns) == generate.COLUMNS
