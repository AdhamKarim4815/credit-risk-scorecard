"""Business layer: cutoff economics, expected loss by score band, IFRS 9 style staging.

Everything here rests on STATED, SIMPLIFIED assumptions (see ``Economics``). The
numbers are illustrations of the method on synthetic data, not forecasts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Economics:
    """Simplified loan economics, all expressed per unit of original loan amount."""

    net_margin_pa: float = 0.055  # annual net interest margin after funding cost and opex
    avg_balance_factor: float = 0.55  # average outstanding balance as a share of the amount
    lgd: float = 0.70  # loss given default (unsecured consumer credit)
    ead_factor: float = 0.85  # exposure at default as a share of the original amount

    def revenue(self, amount, term_months):
        """Margin earned over the life of a loan that performs."""
        return (np.asarray(amount, dtype=float) * self.net_margin_pa * self.avg_balance_factor
                * np.asarray(term_months, dtype=float) / 12.0)

    def ead(self, amount):
        return np.asarray(amount, dtype=float) * self.ead_factor

    def loss_if_default(self, amount):
        return self.ead(amount) * self.lgd

    def to_dict(self) -> dict:
        return asdict(self)


def cutoff_analysis(score, y, amount, term, econ: Economics, step: float = 5.0) -> pd.DataFrame:
    """Approval rate, bad rate and realised profit for each candidate cutoff.

    A loan is approved when ``score >= cutoff``. A performing loan earns ``revenue``; a
    defaulted loan loses ``EAD * LGD`` (earlier margin on defaulted loans is ignored, a
    conservative simplification).
    """
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    profit = np.where(y == 0, econ.revenue(amount, term), -econ.loss_if_default(amount))
    order = np.argsort(-score, kind="mergesort")
    s, yy, pp = score[order], y[order], profit[order]
    cum_n, cum_b, cum_p = np.arange(1, len(s) + 1), np.cumsum(yy), np.cumsum(pp)
    cutoffs = np.arange(np.floor(score.min() / step) * step, score.max() + step, step)
    # number approved at cutoff c = count of scores >= c
    k = np.searchsorted(-s, -cutoffs, side="right")
    rows = []
    for c, m in zip(cutoffs, k, strict=True):
        if m == 0:
            rows.append({"cutoff": c, "approval_rate": 0.0, "bad_rate_approved": np.nan,
                         "profit": 0.0, "profit_per_applicant": 0.0})
            continue
        rows.append({"cutoff": float(c), "approval_rate": m / len(s),
                     "bad_rate_approved": cum_b[m - 1] / cum_n[m - 1],
                     "profit": float(cum_p[m - 1]), "profit_per_applicant": cum_p[m - 1] / len(s)})
    return pd.DataFrame(rows)


def optimal_cutoff(table: pd.DataFrame) -> float:
    """Cutoff with the highest total profit (first one on ties)."""
    return float(table.loc[table["profit"].idxmax(), "cutoff"])


def evaluate_cutoff(score, y, amount, term, econ: Economics, cutoff: float) -> dict:
    """Outcome of applying a fixed cutoff to a sample (for out-of-time evaluation)."""
    score, y = np.asarray(score, dtype=float), np.asarray(y, dtype=int)
    approved = score >= cutoff
    profit = np.where(y == 0, econ.revenue(amount, term), -econ.loss_if_default(amount))
    return {"cutoff": float(cutoff), "approval_rate": float(approved.mean()),
            "bad_rate_approved": float(y[approved].mean()) if approved.any() else float("nan"),
            "profit": float(profit[approved].sum()),
            "profit_all_approved": float(profit.sum()),
            "profit_per_applicant": float(profit[approved].sum() / len(y))}


def band_edges(score, n_bands: int = 10) -> np.ndarray:
    """Score band edges (interior cut points) from quantiles of a reference sample."""
    return np.unique(np.quantile(np.asarray(score, dtype=float),
                                 np.linspace(0, 1, n_bands + 1)[1:-1]))


def expected_loss_by_band(score, pd_pred, y, amount, econ: Economics,
                          edges: np.ndarray) -> pd.DataFrame:
    """Portfolio expected loss EL = PD x LGD x EAD per score band (riskiest band first)."""
    score = np.asarray(score, dtype=float)
    band = np.searchsorted(edges, score, side="right")
    ead = econ.ead(amount)
    pd_pred, y = np.asarray(pd_pred, dtype=float), np.asarray(y, dtype=int)
    rows = []
    for b in range(len(edges) + 1):
        m = band == b
        if not m.any():
            continue
        lo = "min" if b == 0 else f"{edges[b - 1]:.0f}"
        hi = "max" if b == len(edges) else f"{edges[b]:.0f}"
        el = float(np.sum(pd_pred[m] * econ.lgd * ead[m]))
        rows.append({"band": f"{lo} to {hi}", "n": int(m.sum()), "avg_pd": float(pd_pred[m].mean()),
                     "obs_bad_rate": float(y[m].mean()), "ead": float(ead[m].sum()),
                     "expected_loss": el, "el_rate": el / float(ead[m].sum()),
                     "observed_loss": float(np.sum(y[m] * econ.lgd * ead[m]))})
    out = pd.DataFrame(rows)
    total = {"band": "Total", "n": int(out["n"].sum()), "avg_pd": float(pd_pred.mean()),
             "obs_bad_rate": float(y.mean()), "ead": float(out["ead"].sum()),
             "expected_loss": float(out["expected_loss"].sum()),
             "observed_loss": float(out["observed_loss"].sum())}
    total["el_rate"] = total["expected_loss"] / total["ead"]
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def ifrs9_illustration(pd_origination, defaulted, amount, term, econ: Economics,
                       seed: int = 7, sicr_ratio: float = 2.0, sicr_floor: float = 0.01) -> dict:
    """ILLUSTRATIVE staging and ECL, not an IFRS 9 implementation.

    A reporting-date PD is simulated as the origination PD times a seeded log-normal
    migration factor. Stage 3: defaulted. Stage 2: significant increase in credit risk
    (PD at least ``sicr_ratio`` times origination and at least ``sicr_floor`` higher in
    absolute terms) or a simulated 30 days past due backstop. Stage 1: the rest.
    Stage 1 ECL uses the 12-month PD; stage 2 uses a lifetime PD with a constant hazard
    over half the contractual term (mid-life); stage 3 uses PD = 100%. No discounting.
    """
    rng = np.random.default_rng(seed)
    p0 = np.asarray(pd_origination, dtype=float)
    p1 = np.clip(p0 * np.exp(rng.normal(0.0, 0.55, len(p0))), 1e-4, 0.99)
    dpd30 = rng.random(len(p0)) < 0.015
    defaulted = np.asarray(defaulted, dtype=int) == 1
    sicr = ((p1 >= sicr_ratio * p0) & (p1 - p0 >= sicr_floor)) | dpd30
    stage = np.where(defaulted, 3, np.where(sicr, 2, 1))
    ead = econ.ead(amount)
    years = np.asarray(term, dtype=float) / 12.0 / 2.0
    lifetime = 1.0 - (1.0 - p1) ** np.maximum(years, 1.0)
    pd_used = np.where(stage == 1, p1, np.where(stage == 2, lifetime, 1.0))
    ecl = pd_used * econ.lgd * ead
    rows = []
    for s in (1, 2, 3):
        m = stage == s
        rows.append({"stage": s, "n": int(m.sum()), "share": float(m.mean()),
                     "ead": float(ead[m].sum()), "ecl": float(ecl[m].sum()),
                     "coverage": float(ecl[m].sum() / ead[m].sum()) if m.any() else 0.0})
    return {"table": pd.DataFrame(rows), "total_ecl": float(ecl.sum()),
            "total_ead": float(ead.sum())}


def adverse_impact_screen(group, approved) -> pd.DataFrame:
    """Approval rate by group and the adverse impact ratio against the best treated group
    (the 'four-fifths' screen: a ratio below 0.80 prompts investigation, it is not a verdict)."""
    df = pd.DataFrame({"group": np.asarray(group), "approved": np.asarray(approved, dtype=bool)})
    g = df.groupby("group", observed=True)["approved"].agg(["size", "mean"])
    g.columns = ["n", "approval_rate"]
    g["impact_ratio"] = g["approval_rate"] / g["approval_rate"].max()
    return g.reset_index()
