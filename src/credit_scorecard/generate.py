"""Synthetic consumer loan generator.

ALL DATA PRODUCED HERE IS SYNTHETIC. No real person, lender or portfolio is
represented. The default process is a KNOWN, planted logistic model (see
``planted_logit``) so that modelling results can be checked against the truth.

Features of the simulated portfolio:

* 36 monthly origination vintages (2021-01 to 2023-12) with a macro shift
  (unemployment rises through 2023) and an applicant-mix drift, so that the
  population stability index between development and out-of-time samples is
  meaningful.
* Non-linear effects (hinge on DTI, convex utilisation, saturating
  delinquency effect) and interactions (high utilisation x high DTI,
  long term x high loan-to-income, delinquencies x renting).
* Missing values that are not at random: high utilisation is more likely to be
  missing, thin files are missing credit file age, self-employed style
  borrowers miss employment length.
* A handful of planted data quality problems for the checker to find
  (duplicates, impossible values, sentinels, an invalid term, a leaky
  post-origination column, and a feed outage month).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

N_DEFAULT = 60_000
SEED = 20260101
N_MONTHS = 36
TARGET_BAD_RATE = 0.08
NOISE_SD = 0.65  # std dev of the unobserved log-odds component
START_DATE = "2021-01-01"
VOLUME_GAP_MONTH = 14  # 2022-03: simulated data feed outage

PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "auto", "medical",
            "small_business", "other"]
PURPOSE_EFFECT = {"debt_consolidation": 0.10, "credit_card": -0.10, "home_improvement": -0.15,
                  "auto": -0.20, "medical": 0.30, "small_business": 0.45, "other": 0.05}
HOME_EFFECT = {"RENT": 0.25, "MORTGAGE": 0.0, "OWN": -0.20}
REGIONS = ["Atlantic", "Quebec", "Ontario", "Prairies", "Alberta", "BC"]
REGION_EFFECT = {"Atlantic": 0.10, "Quebec": -0.05, "Ontario": 0.0, "Prairies": 0.05,
                 "Alberta": 0.08, "BC": -0.05}

COLUMNS = ["loan_id", "origination_date", "age", "income", "emp_length", "dti", "revol_util",
           "delinq_24m", "credit_file_age_months", "loan_amount", "term", "purpose",
           "home_ownership", "region", "max_dpd_post_orig", "default"]

# Planted data quality problems (counts), used by the tests and documented in the report.
DQ_PLAN = {"duplicates": 30, "bad_age": 10, "negative_income": 8, "income_sentinel": 15,
           "util_sentinel": 25, "emp_gt_age": 20, "bad_term": 6}


def unemployment_path(n_months: int = N_MONTHS) -> np.ndarray:
    """Synthetic unemployment rate (percent) by origination month."""
    return np.interp(np.arange(n_months), [0, 12, 24, 35], [5.0, 5.2, 5.9, 7.6])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def planted_logit(f: dict[str, np.ndarray]) -> np.ndarray:
    """The KNOWN log-odds of default (without intercept), before the noise term.

    ``f`` holds the true, uncorrupted feature values plus the missingness
    indicators ``emp_missing``, ``util_missing`` and ``file_missing``.
    Protected attributes (sex, race, religion, marital status, national origin)
    are not in the data and not in this function. Age is not used directly; it
    acts only through employment length and credit file age.
    """
    lti = f["loan_amount"] / f["income"]
    z = 0.9 * np.log1p(f["delinq_24m"])  # saturating
    z = z + 2.6 * np.maximum(f["revol_util"] - 0.35, 0.0) ** 1.4  # convex
    z = z + 3.2 * np.maximum(f["dti"] - 0.28, 0.0)  # hinge
    z = z - 0.5 * np.log(f["income"] / 52_000.0)
    z = z - 0.07 * np.minimum(f["emp_length"], 12.0)
    z = z - 0.45 * np.log(np.maximum(f["credit_file_age_months"], 6.0) / 84.0)
    z = z + 1.4 * lti
    term60 = (f["term"] == 60).astype(float)
    z = z + 0.25 * term60
    z = z + 0.80 * term60 * (lti > 0.30)  # interaction
    z = z + 1.20 * ((f["revol_util"] > 0.75) & (f["dti"] > 0.38))  # interaction
    z = z + 0.50 * ((f["delinq_24m"] >= 2) & (f["home_ownership"] == "RENT"))  # interaction
    z = z + np.vectorize(PURPOSE_EFFECT.get)(f["purpose"])
    z = z + np.vectorize(HOME_EFFECT.get)(f["home_ownership"])
    z = z + np.vectorize(REGION_EFFECT.get)(f["region"])
    z = z + 0.30 * f["emp_missing"] + 0.35 * f["file_missing"]
    z = z + 0.30 * (f["unemployment"] - 5.5)  # macro
    return z


def _simulate(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = 1.0 + 0.5 * np.arange(N_MONTHS) / (N_MONTHS - 1)
    w[VOLUME_GAP_MONTH] *= 0.15
    month = np.sort(rng.choice(N_MONTHS, size=n, p=w / w.sum()))
    v = month / (N_MONTHS - 1)  # 0 -> 1 drift driver
    unemp = unemployment_path()[month]

    age = np.clip(np.round(rng.normal(41 - 1.5 * v, 12.5, n)), 21, 75)
    income = np.exp(rng.normal(np.log(52_000) + 0.010 * (age - 40) - 0.20 * v, 0.55, n))
    income = np.round(np.clip(income, 14_000, 350_000), -2)
    emp = np.minimum(rng.gamma(1.6, 4.0, n), age - 18)
    emp = np.round(emp, 1)
    dti = rng.beta(2.2, 5.5, n) * 0.9 - 0.12 * np.log(income / 52_000) + 0.10 * v
    dti = np.round(np.clip(dti, 0.02, 0.85), 4)
    util = rng.beta(1.8, 2.6, n) * 1.05 + 0.19 * v
    util = np.round(np.clip(util, 0.0, 1.2), 4)
    frailty = rng.normal(0, 1, n)
    lam = np.exp(-1.7 + 0.5 * frailty + 0.8 * (util - 0.5) + 0.6 * v)
    delinq = np.minimum(rng.poisson(lam), 8)
    file_age = np.maximum((age - 18) * 12 * rng.uniform(0.3, 0.9, n), 6.0)
    file_age = np.round(file_age)
    amount = np.exp(rng.normal(np.log(9_000) + 0.25 * np.log(income / 52_000), 0.6, n))
    amount = np.round(np.clip(amount, 1_000, 50_000), -1)
    term = np.where(rng.random(n) < _sigmoid(-1.2 + 0.00007 * (amount - 9_000)), 60, 36)

    p_purpose = np.array([0.34, 0.22, 0.12, 0.10, 0.05, 0.05, 0.12])
    drift = np.array([0.06, -0.03, -0.01, -0.01, 0.0, 0.01, -0.02])
    cum = np.cumsum(p_purpose[None, :] + v[:, None] * drift[None, :], axis=1)
    purpose = np.array(PURPOSES)[(rng.random(n)[:, None] > cum[:, :-1]).sum(axis=1)]
    own_logit = -0.8 + 0.035 * (age - 40)
    u = rng.random(n)
    home = np.where(u < _sigmoid(own_logit - 0.4), "MORTGAGE", "RENT")
    home = np.where((u > 0.80) & (age > 35), "OWN", home)
    region = rng.choice(REGIONS, size=n, p=[0.07, 0.22, 0.38, 0.07, 0.12, 0.14])

    emp_missing = rng.random(n) < _sigmoid(
        -3.3 + 1.1 * (purpose == "small_business") - 0.5 * np.log(income / 52_000))
    util_missing = rng.random(n) < _sigmoid(-4.6 + 3.2 * util)  # MNAR: depends on util itself
    file_missing = rng.random(n) < _sigmoid(-3.6 - 0.05 * (age - 25))  # thin files
    true = {"income": income, "emp_length": emp, "dti": dti, "revol_util": util,
            "delinq_24m": delinq, "credit_file_age_months": file_age, "loan_amount": amount,
            "term": term, "purpose": purpose, "home_ownership": home, "region": region,
            "emp_missing": emp_missing.astype(float), "util_missing": util_missing.astype(float),
            "file_missing": file_missing.astype(float), "unemployment": unemp}
    z0 = planted_logit(true)
    z = z0 + NOISE_SD * rng.normal(0, 1, n)  # unobserved heterogeneity
    b0 = brentq(lambda b: _sigmoid(z + b).mean() - TARGET_BAD_RATE, -15, 5)
    default = (rng.random(n) < _sigmoid(z + b0)).astype(int)
    # Oracle: PD given the true features, integrating out the unobservable noise
    # (Gauss-Hermite). The best any model could do with these inputs; it also sees values
    # that missingness hides from a real model.
    nodes, weights = np.polynomial.hermite_e.hermegauss(30)
    true_pd = sum(w * _sigmoid(z0 + b0 + NOISE_SD * x)
                  for x, w in zip(nodes, weights, strict=True)) / np.sqrt(2 * np.pi)

    dpd = np.where(default == 1, rng.choice([90, 120, 150, 180], size=n), 0)
    flip = rng.random(n)
    dpd = np.where((default == 1) & (flip < 0.03), rng.integers(0, 60, n), dpd)
    dpd = np.where((default == 0) & (flip < 0.08), rng.integers(1, 31, n), dpd)
    dpd = np.where((default == 0) & (flip > 0.98), rng.integers(31, 61, n), dpd)

    dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(1, n + 1)],
        "origination_date": np.asarray(dates)[month],
        "age": age.astype(int), "income": income,
        "emp_length": np.where(emp_missing, np.nan, emp), "dti": dti,
        "revol_util": np.where(util_missing, np.nan, util), "delinq_24m": delinq,
        "credit_file_age_months": np.where(file_missing, np.nan, file_age),
        "loan_amount": amount, "term": term, "purpose": purpose, "home_ownership": home,
        "region": region, "max_dpd_post_orig": dpd, "default": default,
    })
    return df, true_pd


def _plant_data_quality_problems(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    pool = rng.permutation(len(df))
    cursor = 0

    def take(k: int) -> np.ndarray:
        nonlocal cursor
        idx = pool[cursor:cursor + k]
        cursor += k
        return idx

    df.loc[df.index[take(DQ_PLAN["bad_age"])], "age"] = rng.choice([0, 5, 150, 199],
                                                                    DQ_PLAN["bad_age"])
    df.loc[df.index[take(DQ_PLAN["negative_income"])], "income"] = -rng.integers(
        1_000, 90_000, DQ_PLAN["negative_income"]).astype(float)
    df.loc[df.index[take(DQ_PLAN["income_sentinel"])], "income"] = 999_999.0
    df.loc[df.index[take(DQ_PLAN["util_sentinel"])], "revol_util"] = 9.99
    idx = df.index[take(DQ_PLAN["emp_gt_age"])]
    df.loc[idx, "emp_length"] = (df.loc[idx, "age"] - 5).astype(float)
    df.loc[df.index[take(DQ_PLAN["bad_term"])], "term"] = 48
    dups = df.iloc[take(DQ_PLAN["duplicates"])]
    df = pd.concat([df, dups], ignore_index=True)
    return df.sort_values(["origination_date", "loan_id"], kind="stable").reset_index(drop=True)


def generate_loans(n: int = N_DEFAULT, seed: int = SEED, with_dq_issues: bool = True,
                   return_truth: bool = False):
    """Generate the synthetic loan table (deterministic for a given ``n`` and ``seed``).

    With ``return_truth`` also returns a Series of the true default probability
    indexed by ``loan_id`` (the oracle, never written to the shipped CSV).
    """
    df, true_pd = _simulate(n, seed)
    truth = pd.Series(true_pd, index=df["loan_id"].to_numpy(), name="true_pd")
    if with_dq_issues and n >= 2_000:
        df = _plant_data_quality_problems(df, seed)
    df = df[COLUMNS]
    return (df, truth) if return_truth else df


def write_dataset(path, n: int = N_DEFAULT, seed: int = SEED) -> pd.DataFrame:
    """Generate and write the CSV (``float_format`` keeps the file small)."""
    df = generate_loans(n=n, seed=seed)
    df.to_csv(path, index=False, float_format="%.4f")
    return df
