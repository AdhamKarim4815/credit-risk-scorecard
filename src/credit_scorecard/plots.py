"""Matplotlib charts for the validation pack (PNG, no seaborn)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from credit_scorecard import metrics  # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY = "#1f5f99", "#d9731a", "#2a8f5b", "#c0392b", "#6b7280"
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 130, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "legend.frameon": False})
SYN = "Synthetic data"


def _save(fig, path: Path) -> None:
    fig.text(0.995, 0.005, SYN, ha="right", va="bottom", fontsize=7, color=GREY)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def roc_plot(path, y, score, ch_pred=None) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for key, color, label in (("test", BLUE, "Scorecard, in-time test"),
                              ("oot", ORANGE, "Scorecard, out-of-time")):
        fpr, tpr = metrics.roc_curve(y[key], -score[key])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC {metrics.auc(y[key], -score[key]):.3f})")
    if ch_pred is not None:
        fpr, tpr = metrics.roc_curve(y["oot"], ch_pred["oot"])
        a = metrics.auc(y["oot"], ch_pred["oot"])
        ax.plot(fpr, tpr, color=GREEN, lw=1.5, ls="--",
                label=f"Gradient boosting, out-of-time (AUC {a:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=":")
    ax.set(xlabel="False positive rate (goods flagged)", ylabel="True positive rate (bads caught)",
           title="ROC curve", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def ks_plot(path, y, score) -> None:
    c = metrics.ks_curve(y, -score)
    pts = -c["threshold"].to_numpy()  # back to points, ascending
    k = (c["cum_bad"] - c["cum_good"]).abs().to_numpy()
    i = int(np.argmax(k))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(pts, c["cum_bad"], color=RED, lw=2, label="Cumulative bads")
    ax.plot(pts, c["cum_good"], color=BLUE, lw=2, label="Cumulative goods")
    ax.vlines(pts[i], c["cum_good"].iloc[i], c["cum_bad"].iloc[i], color="black", lw=1.5)
    ax.annotate(f"KS = {k[i]:.3f} at {pts[i]:.0f} pts", (pts[i], (c['cum_bad'].iloc[i]
                + c['cum_good'].iloc[i]) / 2), xytext=(10, 0), textcoords="offset points",
                fontsize=9, bbox={"fc": "white", "ec": "none", "alpha": 0.85})
    ax.set(xlabel="Score (points, riskiest on the left)", ylabel="Cumulative share",
           title="Kolmogorov-Smirnov, out-of-time sample")
    ax.legend(loc="lower right")
    _save(fig, path)


def calibration_plot(path, cal_test, cal_oot) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    hi = max(cal_oot["obs_rate"].max(), cal_oot["mean_pd"].max(), cal_test["obs_rate"].max()) * 1.1
    ax.plot([0, hi], [0, hi], color=GREY, ls=":", label="Perfect calibration")
    ax.plot(cal_test["mean_pd"], cal_test["obs_rate"], "o-", color=BLUE, label="In-time test")
    ax.plot(cal_oot["mean_pd"], cal_oot["obs_rate"], "s-", color=ORANGE, label="Out-of-time")
    ax.set(xlabel="Mean predicted PD (by decile)", ylabel="Observed bad rate",
           title="Calibration by decile", xlim=(0, hi), ylim=(0, hi))
    ax.legend(loc="upper left")
    _save(fig, path)


def score_distribution_plot(path, y, score) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 40)
    ax.hist(score[y == 0], bins=bins, density=True, alpha=0.65, color=BLUE, label="Good")
    ax.hist(score[y == 1], bins=bins, density=True, alpha=0.65, color=RED, label="Bad (default)")
    ax.set(xlabel="Scorecard points (higher = lower risk)", ylabel="Density",
           title="Score distribution, goods vs bads (out-of-time)")
    ax.legend()
    _save(fig, path)


def psi_plot(path, psi_res) -> None:
    rows = [(c["characteristic"], c["psi"]) for c in psi_res["characteristics"]]
    rows.append(("SCORE", psi_res["score_train_vs_oot"]))
    rows.sort(key=lambda r: r[1])
    names, vals = zip(*rows, strict=True)
    colors = [RED if v >= 0.25 else ORANGE if v >= 0.10 else BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(rows) + 1.6))
    ax.barh(names, vals, color=colors)
    for x, label in ((0.10, "0.10 moderate"), (0.25, "0.25 significant")):
        ax.axvline(x, color=GREY, ls="--", lw=1)
        ax.text(x, len(rows) - 0.4, label, fontsize=7, color=GREY, ha="left", va="bottom")
    ax.set(xlabel="PSI (development vs out-of-time)", title="Population stability index")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def cutoff_plot(path, cut_test, cut_oot, cutoff) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(cut_test["cutoff"], cut_test["profit_per_applicant"], color=BLUE, lw=2,
            label="Profit per applicant, in-time test")
    ax.plot(cut_oot["cutoff"], cut_oot["profit_per_applicant"], color=ORANGE, lw=2,
            label="Profit per applicant, out-of-time")
    ax.axvline(cutoff, color="black", ls="--", lw=1)
    ax.text(cutoff, ax.get_ylim()[0], f" chosen cutoff {cutoff:.0f}", fontsize=8, va="bottom")
    ax.set(xlabel="Cutoff (approve if score is at least this)",
           ylabel="Profit per applicant (currency units)", title="Cutoff vs expected profit")
    ax2 = ax.twinx()
    ax2.plot(cut_oot["cutoff"], cut_oot["approval_rate"], color=GREY, ls=":", lw=1.5,
             label="Approval rate, out-of-time")
    ax2.set(ylabel="Approval rate", ylim=(0, 1.02))
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    _save(fig, path)


def woe_plot(path, binning) -> None:
    t = binning.table()
    t = t[t["count"] > 0]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(range(len(t)), t["bad_rate"], color=BLUE)
    ax.set_xticks(range(len(t)), t["bin"], rotation=30, ha="right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(range(len(t)), t["woe"], "o-", color=ORANGE)
    ax2.set(ylabel="WoE")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set(ylabel="Bad rate", title=f"Monotonic WoE binning: {binning.name}")
    _save(fig, path)


def make_all(docs: Path, sc, score, y, res, ch_pred) -> list[str]:
    docs.mkdir(parents=True, exist_ok=True)
    roc_plot(docs / "roc.png", y, score, ch_pred)
    ks_plot(docs / "ks.png", y["oot"], score["oot"])
    calibration_plot(docs / "calibration.png", res["calibration"]["test"],
                     res["calibration"]["oot"])
    score_distribution_plot(docs / "score_distribution.png", y["oot"], score["oot"])
    psi_plot(docs / "psi.png", res["psi"])
    b = res["business"]
    cutoff_plot(docs / "cutoff_profit.png", b["cutoff_table_test"], b["cutoff_table_oot"],
                b["chosen_cutoff"])
    top = max(sc.binnings.values(), key=lambda x: x.iv)
    woe_plot(docs / "woe_binning.png", top)
    return ["roc", "ks", "calibration", "score_distribution", "psi", "cutoff_profit",
            "woe_binning"]

