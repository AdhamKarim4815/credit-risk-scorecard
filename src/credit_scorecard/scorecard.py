"""Scorecard scaling, points table, scoring of new applicants and reason codes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scorecard.binning import FeatureBinning
from credit_scorecard.model import LogitFit

# Adverse action style reason codes: feature -> (code, text shown to the applicant).
REASON_TEXT = {
    "delinq_24m": ("R01", "Recent delinquencies on credit accounts"),
    "revol_util": ("R02", "High utilisation of revolving credit"),
    "dti": ("R03", "High debt relative to income"),
    "income": ("R04", "Income level"),
    "emp_length": ("R05", "Short length of employment"),
    "credit_file_age_months": ("R06", "Short credit history"),
    "loan_amount": ("R07", "Requested loan amount"),
    "loan_to_income": ("R08", "Requested amount high relative to income"),
    "term": ("R09", "Requested loan term"),
    "purpose": ("R10", "Purpose of the loan"),
    "home_ownership": ("R11", "Housing status"),
    "region": ("R12", "Geographic region"),
}


@dataclass
class Scaling:
    """Points scaling: ``base_score`` points at ``base_odds`` good:bad, ``pdo`` points per doubling.

    score = offset + factor * ln(odds_good), factor = pdo / ln 2,
    offset = base_score - factor * ln(base_odds).
    """

    base_score: float = 600.0
    base_odds: float = 30.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_pd(self, pd_: np.ndarray | float) -> np.ndarray | float:
        pd_ = np.asarray(pd_, dtype=float)
        return self.offset + self.factor * np.log((1.0 - pd_) / pd_)

    def pd_from_score(self, score: np.ndarray | float) -> np.ndarray | float:
        odds = np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)
        return 1.0 / (1.0 + odds)


class Scorecard:
    """A points-based scorecard built from WoE binnings and a logistic fit.

    The model is ``logit(PD) = b0 + sum(b_j * WoE_j)``, so ``ln(odds_good) = -logit(PD)``
    and ``score = offset + factor * (-b0 - sum(b_j * WoE_j))``. The intercept part is split
    evenly across characteristics so every attribute carries a whole share of the points.
    """

    def __init__(self, binnings: dict[str, FeatureBinning], intercept: float,
                 coefs: dict[str, float], scaling: Scaling | None = None):
        self.scaling = scaling or Scaling()
        self.intercept = float(intercept)
        self.coefs = dict(coefs)
        self.features = list(coefs)
        self.binnings = {f: binnings[f] for f in self.features}
        self._points = {f: self._attribute_points(f) for f in self.features}

    @classmethod
    def from_fit(cls, binnings: dict[str, FeatureBinning], fit: LogitFit,
                 scaling: Scaling | None = None) -> Scorecard:
        coefs = dict(zip(fit.names[1:], fit.coef[1:].tolist(), strict=True))
        return cls(binnings, float(fit.coef[0]), coefs, scaling)

    def _attribute_points(self, f: str) -> np.ndarray:
        s = self.scaling
        base_share = (s.offset - s.factor * self.intercept) / len(self.features)
        return -s.factor * self.coefs[f] * np.asarray(self.binnings[f].woe) + base_share

    def points_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Points earned by each applicant on each characteristic."""
        return pd.DataFrame({f: self._points[f][self.binnings[f].assign(df[f])]
                             for f in self.features}, index=df.index)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Total scorecard points (higher = lower risk)."""
        return self.points_matrix(df).sum(axis=1).to_numpy()

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        """Model PD, computed from the WoE logit directly (not from rounded points)."""
        eta = np.full(len(df), self.intercept)
        for f in self.features:
            eta = eta + self.coefs[f] * self.binnings[f].transform(df[f])
        return 1.0 / (1.0 + np.exp(-eta))

    def points_table(self) -> pd.DataFrame:
        rows = []
        for f in self.features:
            t = self.binnings[f].table()
            for i, r in t.iterrows():
                rows.append({"characteristic": f, "attribute": r["bin"], "count": r["count"],
                             "bad_rate": r["bad_rate"], "woe": r["woe"],
                             "points": self._points[f][i]})
        return pd.DataFrame(rows)

    def max_points(self) -> dict[str, float]:
        """Best attainable points per characteristic (regular bins only)."""
        return {f: float(self._points[f][:-1].max()) for f in self.features}

    def reason_codes(self, df: pd.DataFrame, n: int = 3) -> list[list[dict]]:
        """Top ``n`` adverse action reasons per applicant: the characteristics where the
        applicant lost the most points relative to the best attainable attribute."""
        pts = self.points_matrix(df)
        shortfall = (pd.Series(self.max_points()) - pts).clip(lower=0.0).to_numpy()
        order = np.argsort(-shortfall, axis=1, kind="stable")[:, :n]
        out = []
        for i in range(len(shortfall)):
            row = []
            for j in order[i]:
                lost = shortfall[i, j]
                if lost > 0:
                    f = self.features[j]
                    code, text = REASON_TEXT.get(f, ("R99", f))
                    row.append({"characteristic": f, "code": code, "reason": text,
                                "points_lost": round(float(lost), 1)})
            out.append(row)
        return out

    def score_frame(self, df: pd.DataFrame, n_reasons: int = 3) -> pd.DataFrame:
        """Score, PD and reason codes for new applicants (what the ``score`` command writes)."""
        out = pd.DataFrame({"score": np.round(self.score(df), 1),
                            "pd": np.round(self.predict_pd(df), 5)}, index=df.index)
        codes = self.reason_codes(df, n_reasons)
        for k in range(n_reasons):
            out[f"reason_{k + 1}"] = [
                f"{r[k]['code']} {r[k]['reason']} (-{r[k]['points_lost']:.0f} pts)"
                if len(r) > k else "" for r in codes]
        return out

    def to_dict(self) -> dict:
        return {"scaling": asdict(self.scaling), "intercept": self.intercept,
                "coefs": self.coefs, "binnings": {f: b.to_dict() for f, b in self.binnings.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> Scorecard:
        binnings = {f: FeatureBinning.from_dict(b) for f, b in d["binnings"].items()}
        return cls(binnings, d["intercept"], d["coefs"], Scaling(**d["scaling"]))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Scorecard:
        return cls.from_dict(json.loads(Path(path).read_text()))
