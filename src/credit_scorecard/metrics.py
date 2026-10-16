"""Discrimination, calibration, lift and stability metrics (hand implemented).

Convention: ``y`` is 1 for bad (default). ``score`` arguments named ``risk``
are oriented so that a HIGHER value means MORE likely to default. Scorecard
points run the other way, so pass ``-points`` where a risk orientation is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PSI_EPS = 1e-4


def _check_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.size == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError("y must be a non-empty array of 0/1 labels")
    if y.min() == y.max():
        raise ValueError("y must contain both classes")
    return y.astype(int)


def auc(y, risk) -> float:
    """ROC AUC via the Mann-Whitney U statistic with mid-ranks for ties."""
    y = _check_binary(y)
    r = stats.rankdata(np.asarray(risk, dtype=float))
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gini(y, risk) -> float:
    """Gini coefficient (Somers' D) = 2 * AUC - 1."""
    return 2.0 * auc(y, risk) - 1.0


def _cum_curves(y, risk) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative bad and good fractions at each distinct threshold (risk descending)."""
    y = _check_binary(y)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(-risk, kind="mergesort")
    s, yy = risk[order], y[order]
    cum_bad = np.cumsum(yy) / yy.sum()
    cum_good = np.cumsum(1 - yy) / (1 - yy).sum()
    last = np.r_[s[1:] != s[:-1], True]  # one point per tie group
    return s[last], cum_bad[last], cum_good[last]


def ks_statistic(y, risk) -> float:
    """Kolmogorov-Smirnov: max distance between cumulative bad and good distributions."""
    _, cb, cg = _cum_curves(y, risk)
    return float(np.max(np.abs(cb - cg)))


def roc_curve(y, risk) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates (bad = positive), starting at (0, 0)."""
    _, cb, cg = _cum_curves(y, risk)
    return np.r_[0.0, cg], np.r_[0.0, cb]


def ks_curve(y, risk) -> pd.DataFrame:
    thr, cb, cg = _cum_curves(y, risk)
    return pd.DataFrame({"threshold": thr, "cum_bad": cb, "cum_good": cg})


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def _bands_by_rank(values: np.ndarray, n_bands: int) -> np.ndarray:
    """Equal-count band index 0..n_bands-1 (ascending value), ties broken by order."""
    ranks = stats.rankdata(values, method="ordinal")
    return np.minimum(((ranks - 1) * n_bands // len(values)).astype(int), n_bands - 1)


def calibration_table(y, pd_pred, n_bands: int = 10) -> pd.DataFrame:
    """Decile calibration table: mean predicted PD vs observed bad rate per band."""
    y = _check_binary(y)
    p = np.asarray(pd_pred, dtype=float)
    band = _bands_by_rank(p, n_bands)
    rows = []
    for b in range(n_bands):
        m = band == b
        rows.append({"band": b + 1, "n": int(m.sum()), "bads": int(y[m].sum()),
                     "mean_pd": float(p[m].mean()), "obs_rate": float(y[m].mean())})
    out = pd.DataFrame(rows)
    out["ratio"] = out["obs_rate"] / out["mean_pd"]
    return out


def hosmer_lemeshow(table: pd.DataFrame) -> tuple[float, float]:
    """Hosmer-Lemeshow chi-square and p-value (G - 2 degrees of freedom)."""
    n, e, o = table["n"].to_numpy(), table["mean_pd"].to_numpy(), table["bads"].to_numpy()
    chi2 = float(np.sum((o - n * e) ** 2 / (n * e * (1 - e))))
    return chi2, float(stats.chi2.sf(chi2, len(table) - 2))


def lift_table(y, risk, n_bands: int = 10) -> pd.DataFrame:
    """Gains and lift by risk decile (band 1 = riskiest 10%)."""
    y = _check_binary(y)
    band = n_bands - 1 - _bands_by_rank(np.asarray(risk, dtype=float), n_bands)
    overall = y.mean()
    rows = []
    cum_n = cum_b = 0
    for b in range(n_bands):
        m = band == b
        cum_n += int(m.sum())
        cum_b += int(y[m].sum())
        rate = y[m].mean()
        rows.append({"band": b + 1, "n": int(m.sum()), "bads": int(y[m].sum()),
                     "bad_rate": float(rate), "lift": float(rate / overall),
                     "cum_pct_pop": cum_n / len(y), "cum_pct_bads": cum_b / y.sum(),
                     "cum_lift": float((cum_b / cum_n) / overall)})
    return pd.DataFrame(rows)


def psi_from_proportions(expected, actual, eps: float = PSI_EPS) -> float:
    """PSI = sum((a - e) * ln(a / e)), zero proportions floored at ``eps``."""
    e = np.maximum(np.asarray(expected, dtype=float), eps)
    a = np.maximum(np.asarray(actual, dtype=float), eps)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_from_counts(expected_counts, actual_counts) -> float:
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    return psi_from_proportions(e / e.sum(), a / a.sum())


def psi(expected, actual, n_bins: int = 10) -> float:
    """PSI of a continuous variable using quantile bins taken from ``expected``."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e_cnt = np.bincount(np.searchsorted(edges, expected, side="left"), minlength=len(edges) + 1)
    a_cnt = np.bincount(np.searchsorted(edges, actual, side="left"), minlength=len(edges) + 1)
    return psi_from_counts(e_cnt, a_cnt)


def psi_label(value: float) -> str:
    """Conventional industry reading of a PSI value."""
    if value < 0.10:
        return "stable"
    return "moderate shift" if value < 0.25 else "significant shift"


def bootstrap_auc(y, risk, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for AUC. Returns (auc, low, high)."""
    y = _check_binary(y)
    risk = np.asarray(risk, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(y):
            draws.append(auc(y[idx], risk[idx]))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return auc(y, risk), float(lo), float(hi)


def bootstrap_auc_diff(y, risk_a, risk_b, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap CI for AUC(a) - AUC(b). Returns (diff, low, high)."""
    y = _check_binary(y)
    ra, rb = np.asarray(risk_a, dtype=float), np.asarray(risk_b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(y):
            draws.append(auc(y[idx], ra[idx]) - auc(y[idx], rb[idx]))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return auc(y, ra) - auc(y, rb), float(lo), float(hi)
