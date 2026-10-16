"""Weight-of-evidence (WoE) and information value (IV) with monotonic binning.

Conventions
-----------
* ``y`` is 1 for bad (default), 0 for good.
* ``WoE = ln(%good / %bad)``: positive WoE means lower risk than average.
* ``IV = sum((%good - %bad) * WoE)``.

Binning algorithm (numeric features)
------------------------------------
1. Start from fine pre-bins (every distinct value if there are few, otherwise
   quantile cut points that are actual data values).
2. Determine the risk direction (bad rate rising or falling with the feature)
   from the weighted trend of the pre-bin bad rates.
3. Repeat until stable: pool adjacent violators of the monotone trend (a
   pool-adjacent-violators / isotonic step), merge bins that are too small or
   have too few goods or bads into the neighbour with the closest bad rate, and
   finally, if there are still too many bins, merge the adjacent pair that
   loses the least IV.
4. Missing values always get their own special bin whose WoE is estimated from
   the missing rows, so informative missingness is kept as a signal.

Categorical features are ordered by bad rate and then merged with the same
routine, so rare levels are pooled with their closest neighbour in risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

SMOOTHING = 0.5
IV_BANDS = [(0.02, "not useful"), (0.10, "weak"), (0.30, "medium"), (0.50, "strong")]


def woe_iv(good, bad, smoothing: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """WoE and per-bin IV terms from good/bad counts.

    With ``smoothing > 0`` that constant is added to every count, which keeps
    WoE finite when a bin has no goods or no bads. With ``smoothing == 0`` an
    empty cell gives an infinite WoE (the textbook definition).
    """
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    pg = (good + smoothing) / (good.sum() + smoothing * len(good))
    pb = (bad + smoothing) / (bad.sum() + smoothing * len(bad))
    with np.errstate(divide="ignore", invalid="ignore"):
        woe = np.log(pg / pb)
    return woe, (pg - pb) * woe


def iv_strength(iv: float) -> str:
    """Conventional label for the predictive strength implied by an IV."""
    for upper, label in IV_BANDS:
        if iv < upper:
            return label
    return "suspiciously strong (check leakage)"


def _iv_term(g: float, b: float, G: float, B: float) -> float:
    pg, pb = (g + SMOOTHING) / (G + SMOOTHING), (b + SMOOTHING) / (B + SMOOTHING)
    return (pg - pb) * np.log(pg / pb)


def _merge_bins(n, bad, direction, min_n, min_cls, max_bins):
    """Merge adjacent pre-bins; returns a list of inclusive (start, end) index pairs."""
    bins = [[float(n[i]), float(bad[i]), i, i] for i in range(len(n))]
    G = float(sum(b[0] - b[1] for b in bins))
    B = float(sum(b[1] for b in bins))

    def rate(b):
        return b[1] / b[0]

    def merge(i):
        a, b = bins[i], bins[i + 1]
        bins[i:i + 2] = [[a[0] + b[0], a[1] + b[1], a[2], b[3]]]

    while len(bins) > 1:
        viol = next((i for i in range(len(bins) - 1)
                     if direction * (rate(bins[i + 1]) - rate(bins[i])) <= 0), None)
        if viol is not None:  # pool adjacent violators of the strict monotone trend
            merge(viol)
            continue
        small = [i for i, b in enumerate(bins)
                 if b[0] < min_n or b[1] < min_cls or b[0] - b[1] < min_cls]
        if small:
            i = min(small, key=lambda k: bins[k][0])
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(bins)]
            j = min(nbrs, key=lambda k: (abs(rate(bins[k]) - rate(bins[i])), bins[k][0]))
            merge(min(i, j))
            continue
        if len(bins) > max_bins:
            def loss(i):
                a, b = bins[i], bins[i + 1]
                return (_iv_term(a[0] - a[1], a[1], G, B) + _iv_term(b[0] - b[1], b[1], G, B)
                        - _iv_term(a[0] + b[0] - a[1] - b[1], a[1] + b[1], G, B))
            merge(min(range(len(bins) - 1), key=loss))
            continue
        break
    return [(b[2], b[3]) for b in bins]


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


@dataclass
class FeatureBinning:
    """A fitted binning: regular bins plus one special bin (missing / unseen) at the end."""

    name: str
    kind: str  # "numeric" or "categorical"
    edges: list[float] = field(default_factory=list)
    groups: list[list[str]] = field(default_factory=list)
    good: list[int] = field(default_factory=list)
    bad: list[int] = field(default_factory=list)
    woe: list[float] = field(default_factory=list)
    iv: float = 0.0
    direction: int = 1

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1 if self.kind == "numeric" else len(self.groups)

    @property
    def special_index(self) -> int:
        return self.n_bins

    def assign(self, values) -> np.ndarray:
        """Bin index per row; the special bin index for missing or unseen values."""
        if self.kind == "numeric":
            x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
            idx = np.searchsorted(self.edges, x, side="left")
            idx[np.isnan(x)] = self.special_index
            return idx
        lookup = {lvl: i for i, g in enumerate(self.groups) for lvl in g}
        s = pd.Series(values).astype(object)
        return np.array([lookup.get(v, self.special_index) if isinstance(v, str)
                         else self.special_index for v in s], dtype=int)

    def transform(self, values) -> np.ndarray:
        return np.asarray(self.woe)[self.assign(values)]

    def labels(self) -> list[str]:
        if self.kind == "numeric":
            e = self.edges
            if not e:
                out = ["all values"]
            else:
                out = [f"<= {_fmt(e[0])}"]
                out += [f"({_fmt(e[i - 1])}, {_fmt(e[i])}]" for i in range(1, len(e))]
                out.append(f"> {_fmt(e[-1])}")
        else:
            out = [" / ".join(g) for g in self.groups]
        return [*out, "Missing / unseen"]

    def table(self) -> pd.DataFrame:
        good, bad = np.asarray(self.good, float), np.asarray(self.bad, float)
        total = good + bad
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(total > 0, bad / total, np.nan)
        pg = good / good.sum() if good.sum() else good
        pb = bad / bad.sum() if bad.sum() else bad
        return pd.DataFrame({"bin": self.labels(), "count": total.astype(int),
                             "pct_total": total / total.sum(), "good": good.astype(int),
                             "bad": bad.astype(int), "bad_rate": rate,
                             "pct_good": pg, "pct_bad": pb, "woe": self.woe})

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "edges": [float(e) for e in self.edges],
                "groups": self.groups, "good": [int(v) for v in self.good],
                "bad": [int(v) for v in self.bad], "woe": [float(v) for v in self.woe],
                "iv": float(self.iv), "direction": int(self.direction)}

    @classmethod
    def from_dict(cls, d: dict) -> FeatureBinning:
        return cls(**d)


def _finalize(name, kind, edges, groups, good, bad, direction, min_special) -> FeatureBinning:
    """Attach the special bin and compute WoE / IV for all bins."""
    good, bad = np.asarray(good, dtype=float), np.asarray(bad, dtype=float)
    special_n = good[-1] + bad[-1]
    keep = np.ones(len(good), dtype=bool)
    if special_n < min_special:
        keep[-1] = False  # too few rows to estimate a WoE: neutral (0)
    woe = np.zeros(len(good))
    terms = np.zeros(len(good))
    woe[keep], terms[keep] = woe_iv(good[keep], bad[keep], smoothing=SMOOTHING)
    return FeatureBinning(name=name, kind=kind, edges=list(edges), groups=groups,
                          good=good.astype(int).tolist(), bad=bad.astype(int).tolist(),
                          woe=woe.tolist(), iv=float(terms.sum()), direction=direction)


def _trend_direction(n: np.ndarray, bad: np.ndarray) -> int:
    """+1 if the bad rate tends to rise with the feature, else -1 (weighted OLS slope)."""
    if len(n) < 2:
        return 1
    x = np.arange(len(n), dtype=float)
    r = bad / n
    xm, rm = np.average(x, weights=n), np.average(r, weights=n)
    slope = np.sum(n * (x - xm) * (r - rm))
    return -1 if slope < 0 else 1


def fit_binning(x, y, name: str = "x", kind: str | None = None, min_bin_frac: float = 0.03,
                min_class: int = 5, max_bins: int = 6, max_prebins: int = 20,
                min_special: int = 30, direction: int | None = None) -> FeatureBinning:
    """Fit a monotonic (numeric) or risk-ordered (categorical) WoE binning.

    ``min_bin_frac`` is the minimum share of the non-missing rows per bin and
    ``min_class`` the minimum number of goods and of bads per bin. ``direction``
    forces +1 (bad rate rises with x) or -1 (falls); by default it is inferred.
    """
    series = pd.Series(x).reset_index(drop=True)
    y = np.asarray(y, dtype=int)
    if kind is None:
        kind = "numeric" if is_numeric_dtype(series) else "categorical"
    if kind == "numeric":
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        miss = np.isnan(vals)
    else:
        vals = series.astype(object).to_numpy()
        miss = np.array([not isinstance(v, str) for v in vals])
    ok = ~miss
    n_ok = int(ok.sum())
    if n_ok == 0:
        raise ValueError(f"feature {name!r} has no non-missing values")
    min_n = int(np.ceil(min_bin_frac * n_ok))
    spec_good, spec_bad = int((y[miss] == 0).sum()), int(y[miss].sum())

    if kind == "numeric":
        xv, yv = vals[ok], y[ok]
        uniq = np.unique(xv)
        if len(uniq) == 1:
            cuts = np.array([])
        elif len(uniq) <= max_prebins:
            cuts = uniq[:-1]
        else:
            q = np.quantile(xv, np.linspace(0, 1, max_prebins + 1)[1:-1], method="lower")
            cuts = np.unique(q)
            cuts = cuts[cuts < uniq[-1]]
        idx = np.searchsorted(cuts, xv, side="left")
        n = np.bincount(idx, minlength=len(cuts) + 1).astype(float)
        bad = np.bincount(idx, weights=yv, minlength=len(cuts) + 1)
        d = direction or _trend_direction(n, bad)
        spans = _merge_bins(n, bad, d, min_n, min_class, max_bins)
        edges = [float(cuts[e]) for _, e in spans[:-1]]
        good = [float(n[s:e + 1].sum() - bad[s:e + 1].sum()) for s, e in spans]
        bads = [float(bad[s:e + 1].sum()) for s, e in spans]
        return _finalize(name, kind, edges, [], [*good, spec_good], [*bads, spec_bad], d,
                         min_special)

    levels, inv = np.unique(vals[ok].astype(str), return_inverse=True)
    n = np.bincount(inv, minlength=len(levels)).astype(float)
    bad = np.bincount(inv, weights=y[ok], minlength=len(levels))
    order = np.argsort(bad / n, kind="mergesort")
    n, bad, levels = n[order], bad[order], levels[order]
    spans = _merge_bins(n, bad, 1, min_n, min_class, max_bins)
    groups = [[str(v) for v in levels[s:e + 1]] for s, e in spans]
    good = [float(n[s:e + 1].sum() - bad[s:e + 1].sum()) for s, e in spans]
    bads = [float(bad[s:e + 1].sum()) for s, e in spans]
    return _finalize(name, kind, [], groups, [*good, spec_good], [*bads, spec_bad], 1,
                     min_special)


def bin_features(df: pd.DataFrame, y, features: list[str], **kwargs) -> dict[str, FeatureBinning]:
    """Fit a binning for each feature (on the development sample)."""
    return {f: fit_binning(df[f], y, name=f, **kwargs) for f in features}


def woe_matrix(df: pd.DataFrame, binnings: dict[str, FeatureBinning],
               features: list[str] | None = None) -> pd.DataFrame:
    """Replace each characteristic by its WoE (missing / unseen use the special bin)."""
    feats = features or list(binnings)
    return pd.DataFrame({f: binnings[f].transform(df[f]) for f in feats}, index=df.index)


def iv_table(binnings: dict[str, FeatureBinning]) -> pd.DataFrame:
    rows = [{"feature": b.name, "iv": b.iv, "strength": iv_strength(b.iv), "n_bins": b.n_bins,
             "kind": b.kind, "direction": b.direction if b.kind == "numeric" else 0}
            for b in binnings.values()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def is_monotone(bad_rates, direction: int = 0, strict: bool = True) -> bool:
    """True if the bad rates are monotone (in ``direction`` if given, else either way)."""
    d = np.diff(np.asarray(bad_rates, dtype=float))
    up = np.all(d > 0) if strict else np.all(d >= 0)
    down = np.all(d < 0) if strict else np.all(d <= 0)
    if direction > 0:
        return bool(up)
    if direction < 0:
        return bool(down)
    return bool(up or down)
