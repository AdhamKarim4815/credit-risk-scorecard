"""IV based feature selection with correlation pruning on the WoE scale."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from credit_scorecard.binning import FeatureBinning


@dataclass
class SelectionResult:
    selected: list[str]
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (feature, reason)
    warnings: list[str] = field(default_factory=list)


def select_features(binnings: dict[str, FeatureBinning], woe_df: pd.DataFrame,
                    min_iv: float = 0.02, max_corr: float = 0.70,
                    suspicious_iv: float = 0.50) -> SelectionResult:
    """Greedy selection: rank by IV, drop weak features, then drop any feature whose
    absolute WoE correlation with an already kept (higher IV) feature exceeds ``max_corr``."""
    ranked = sorted(binnings.values(), key=lambda b: b.iv, reverse=True)
    corr = woe_df.corr().abs()
    result = SelectionResult(selected=[])
    for b in ranked:
        if b.iv > suspicious_iv:
            result.warnings.append(f"{b.name}: IV {b.iv:.2f} is suspiciously high, check leakage")
        if b.iv < min_iv:
            result.dropped.append((b.name, f"IV {b.iv:.3f} below {min_iv}"))
            continue
        clash = next((k for k in result.selected if corr.loc[b.name, k] > max_corr), None)
        if clash:
            result.dropped.append(
                (b.name, f"|corr| {corr.loc[b.name, clash]:.2f} with {clash} above {max_corr}"))
            continue
        result.selected.append(b.name)
    return result
