"""Data quality checks: missingness, outliers, duplicates, leakage, target rate by vintage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from credit_scorecard.metrics import auc

TARGET = "default"
ID_COLS = ("loan_id",)
DATE_COL = "origination_date"
# Domain rules: column -> (min, max) inclusive. Anything outside is an impossible value.
RANGE_RULES = {"age": (18, 100), "income": (1, 5_000_000), "emp_length": (0, 60),
               "dti": (0, 1.5), "revol_util": (0, 3), "delinq_24m": (0, 50),
               "credit_file_age_months": (0, 900), "loan_amount": (100, 1_000_000)}
VALID_TERMS = (36, 60)  # the product offers 36 and 60 month terms only
SENTINELS = (-1, 999, 9999, 99999, 999999, 9999999, 9.99, 99.9)
LEAKY_NAME = re.compile(r"post|after|collect|charge|recover|write_?off|dpd|outcome|status", re.I)


@dataclass
class Issue:
    severity: str  # "critical", "warning" or "info"
    check: str
    message: str


@dataclass
class QualityReport:
    n_rows: int
    issues: list[Issue] = field(default_factory=list)
    missingness: pd.DataFrame | None = None
    outliers: pd.DataFrame | None = None
    leakage: pd.DataFrame | None = None
    vintages: pd.DataFrame | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, severity: str, check: str, message: str) -> None:
        self.issues.append(Issue(severity, check, message))

    def by_check(self, check: str) -> list[Issue]:
        return [i for i in self.issues if i.check == check]

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(i) for i in self.issues],
                            columns=["severity", "check", "message"])


def check_missingness(df: pd.DataFrame, report: QualityReport, target: str = TARGET) -> None:
    rows = []
    for col in df.columns:
        m = df[col].isna()
        if not m.any():
            continue
        br_m, br_p = df.loc[m, target].mean(), df.loc[~m, target].mean()
        p = (m.sum() * br_m + (~m).sum() * br_p) / len(df)
        se = np.sqrt(p * (1 - p) * (1 / m.sum() + 1 / (~m).sum()))
        z = (br_m - br_p) / se if se > 0 else 0.0
        rows.append({"column": col, "pct_missing": m.mean(), "bad_rate_missing": br_m,
                     "bad_rate_present": br_p, "z": z, "informative": bool(abs(z) > 3)})
        if abs(z) > 3:
            report.add("info", "missingness",
                       f"{col}: {m.mean():.1%} missing and the bad rate differs "
                       f"({br_m:.1%} vs {br_p:.1%}); missingness is informative (not random), "
                       f"so it gets its own scorecard bin")
    report.missingness = pd.DataFrame(rows)


def check_ranges(df: pd.DataFrame, report: QualityReport) -> None:
    """Impossible values, sentinel codes, IQR outliers and cross-field consistency."""
    rows = []
    for col, (lo, hi) in RANGE_RULES.items():
        if col not in df:
            continue
        x = df[col]
        bad = int(((x < lo) | (x > hi)).sum())
        sent = int(x.isin(SENTINELS).sum())
        q1, q3 = x.quantile([0.25, 0.75])
        fence = 3 * (q3 - q1)
        iqr_out = int(((x < q1 - fence) | (x > q3 + fence)).sum())
        rows.append({"column": col, "out_of_range": bad, "sentinel_values": sent,
                     "iqr_outliers_3x": iqr_out})
        report.counts[f"range:{col}"] = bad
        report.counts[f"sentinel:{col}"] = sent
        if bad:
            report.add("critical", "range", f"{col}: {bad} values outside [{lo}, {hi}]")
        if sent:
            report.add("warning", "sentinel", f"{col}: {sent} rows hold sentinel-like codes "
                                              f"{[s for s in SENTINELS if (x == s).any()]}")
    report.outliers = pd.DataFrame(rows)
    if "term" in df:
        bad_term = int((~df["term"].isin(VALID_TERMS)).sum())
        report.counts["invalid_term"] = bad_term
        if bad_term:
            report.add("critical", "domain", f"term: {bad_term} rows with a non-standard term "
                                             f"{sorted(set(df['term']) - set(VALID_TERMS))}")
    if {"emp_length", "age"} <= set(df):
        n = int((df["emp_length"] > df["age"] - 14).sum())
        report.counts["emp_gt_age"] = n
        if n:
            report.add("warning", "consistency", f"{n} rows where employment length exceeds "
                                                 f"age minus 14 years")


def check_duplicates(df: pd.DataFrame, report: QualityReport, id_col: str = "loan_id") -> None:
    n = int(df.duplicated(id_col).sum())
    report.counts["duplicate_ids"] = n
    if n:
        report.add("critical", "duplicates", f"{n} duplicate {id_col} rows "
                                             f"({int(df.duplicated().sum())} exact duplicates)")


def check_leakage(df: pd.DataFrame, report: QualityReport, target: str = TARGET,
                  auc_flag: float = 0.85) -> None:
    """Single-variable AUC for every column plus a name screen for post-origination fields.

    A feature that alone separates defaults almost perfectly, or whose name suggests
    it is recorded after the loan was booked, is a leakage suspect.
    """
    y = df[target].to_numpy()
    rows = []
    for col in df.columns:
        if col in (target, *ID_COLS, DATE_COL):
            continue
        s = df[col]
        ok = s.notna().to_numpy()
        if is_numeric_dtype(s):
            score = s.to_numpy(dtype=float)[ok]
        elif s.nunique() <= 50:  # target-encode low-cardinality categoricals (in-sample)
            score = s[ok].map(df.loc[ok].groupby(col)[target].mean()).to_numpy(dtype=float)
        else:
            continue
        if len(np.unique(y[ok])) < 2:
            continue
        a = auc(y[ok], score)
        a = max(a, 1 - a)
        by_name = bool(LEAKY_NAME.search(col))
        flagged = a >= auc_flag or (by_name and a >= 0.60)
        rows.append({"column": col, "single_var_auc": a, "name_suspect": by_name,
                     "flagged": flagged})
        if flagged:
            report.add("critical", "leakage",
                       f"{col}: single-variable AUC {a:.3f}"
                       f"{' and a post-origination style name' if by_name else ''}; "
                       f"treated as leakage and excluded from modelling")
    report.leakage = pd.DataFrame(rows).sort_values("single_var_auc", ascending=False)
    report.counts["leaky_columns"] = int(report.leakage["flagged"].sum())


def check_vintages(df: pd.DataFrame, report: QualityReport, target: str = TARGET) -> None:
    """Volume and target rate by origination month, with anomaly flags."""
    dates = pd.to_datetime(df[DATE_COL])
    g = df.groupby(dates.dt.to_period("M").astype(str))[target].agg(["size", "mean"])
    g.columns = ["n", "bad_rate"]
    g.index.name = "month"
    med = g["n"].median()
    g["volume_flag"] = g["n"] < 0.5 * med
    smooth = g["bad_rate"].rolling(5, center=True, min_periods=3).median()
    z = (g["bad_rate"] - smooth).abs() / np.sqrt(smooth * (1 - smooth) / g["n"])
    g["rate_flag"] = z > 4
    report.vintages = g.reset_index()
    for m, r in g[g["volume_flag"]].iterrows():
        report.add("warning", "vintage",
                   f"{m}: only {int(r['n'])} loans vs median {med:.0f} (possible feed gap)")
    for m in g.index[g["rate_flag"]]:
        report.add("warning", "vintage", f"{m}: bad rate {g.loc[m, 'bad_rate']:.1%} is an "
                                         f"outlier against neighbouring months")
    yearly = df.groupby(dates.dt.year)[target].mean()
    if len(yearly) > 1 and yearly.iloc[-1] > 1.3 * yearly.iloc[0]:
        report.add("info", "vintage",
                   f"bad rate rises from {yearly.iloc[0]:.1%} ({yearly.index[0]}) to "
                   f"{yearly.iloc[-1]:.1%} ({yearly.index[-1]}); expect calibration drift")
    report.counts["volume_flag_months"] = int(g["volume_flag"].sum())


def run_checks(df: pd.DataFrame, target: str = TARGET) -> QualityReport:
    report = QualityReport(n_rows=len(df))
    check_duplicates(df, report)
    check_missingness(df, report, target)
    check_ranges(df, report)
    check_leakage(df, report, target)
    check_vintages(df, report, target)
    return report


def clean_data(df: pd.DataFrame, report: QualityReport) -> tuple[pd.DataFrame, list[str]]:
    """Apply the documented remediation and return (clean frame, action log).

    Impossible values become missing (and fall into the special WoE bin), rows with an
    invalid term and duplicate ids are removed, leakage columns are dropped.
    """
    out = df.drop_duplicates("loan_id", keep="first").copy()
    actions = [f"dropped {len(df) - len(out)} duplicate loan_id rows"]
    for col, (lo, hi) in RANGE_RULES.items():
        if col in out:
            bad = (out[col] < lo) | (out[col] > hi) | out[col].isin(SENTINELS)
            if bad.any():
                out.loc[bad, col] = np.nan
                actions.append(
                    f"set {int(bad.sum())} impossible or sentinel {col} values to missing")
    inconsistent = out["emp_length"] > out["age"] - 14
    if inconsistent.any():
        out.loc[inconsistent, "emp_length"] = np.nan
        actions.append(f"set {int(inconsistent.sum())} inconsistent emp_length values to missing")
    bad_term = ~out["term"].isin(VALID_TERMS)
    if bad_term.any():
        out = out[~bad_term]
        actions.append(f"dropped {int(bad_term.sum())} rows with a non-standard term")
    leaky = report.leakage.loc[report.leakage["flagged"], "column"].tolist()
    if leaky:
        out = out.drop(columns=leaky)
        actions.append(f"dropped leakage columns: {', '.join(leaky)}")
    return out.reset_index(drop=True), actions
