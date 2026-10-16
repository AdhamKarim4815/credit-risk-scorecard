"""Pipeline stages: generate, train, validate, report. Each stage persists its output."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from credit_scorecard import business, generate, metrics, plots, quality
from credit_scorecard.binning import bin_features, iv_table, woe_matrix
from credit_scorecard.model import fit_logit_scipy, fit_with_pruning, vif
from credit_scorecard.scorecard import Scaling, Scorecard
from credit_scorecard.selection import select_features

TARGET = "default"
NUMERIC = ["income", "emp_length", "dti", "revol_util", "delinq_24m", "credit_file_age_months",
           "loan_amount", "term", "loan_to_income"]
CATEGORICAL = ["purpose", "home_ownership", "region"]
FEATURES = NUMERIC + CATEGORICAL
# Regulated attribute: collected and monitored (adverse impact screen) but never a model input.
EXCLUDED_FROM_MODEL = {"age": "regulated attribute, used for monitoring only"}
OOT_START = "2023-01-01"
TEST_FRACTION = 0.30
N_BOOT = 400


@dataclass
class Config:
    workdir: Path
    n: int = generate.N_DEFAULT
    seed: int = generate.SEED

    def __post_init__(self):
        self.workdir = Path(self.workdir)

    @property
    def data_path(self) -> Path:
        return self.workdir / "data" / "synthetic_loans.csv"

    @property
    def meta_path(self) -> Path:
        return self.workdir / "data" / "synthetic_loans.meta.json"

    @property
    def artifacts(self) -> Path:
        return self.workdir / "artifacts"

    @property
    def docs(self) -> Path:
        return self.workdir / "docs"


@dataclass
class Prepared:
    raw: pd.DataFrame
    clean: pd.DataFrame
    quality: quality.QualityReport
    actions: list[str]
    train: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def frame(records) -> pd.DataFrame:
    return pd.DataFrame(records)


def stage_generate(cfg: Config) -> pd.DataFrame:
    cfg.data_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate.write_dataset(cfg.data_path, n=cfg.n, seed=cfg.seed)
    save_json({"synthetic": True, "n_base": cfg.n, "seed": cfg.seed, "rows_written": len(df),
               "oot_start": OOT_START}, cfg.meta_path)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["loan_to_income"] = out["loan_amount"] / out["income"]
    return out


def split_data(df: pd.DataFrame, seed: int):
    """Development = origination before OOT_START, split at random into train and test;
    out-of-time = origination from OOT_START onward."""
    oot_mask = pd.to_datetime(df["origination_date"]) >= pd.Timestamp(OOT_START)
    dev, oot = df[~oot_mask], df[oot_mask]
    perm = np.random.default_rng(seed + 2).permutation(len(dev))
    n_test = int(round(TEST_FRACTION * len(dev)))
    test = dev.iloc[perm[:n_test]].sort_index()
    train = dev.iloc[perm[n_test:]].sort_index()
    return train.reset_index(drop=True), test.reset_index(drop=True), oot.reset_index(drop=True)


def prepare(cfg: Config) -> Prepared:
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"{cfg.data_path} not found; run `credit-scorecard generate` first")
    raw = pd.read_csv(cfg.data_path)
    report = quality.run_checks(raw)
    clean, actions = quality.clean_data(raw, report)
    train, test, oot = split_data(add_features(clean), cfg.seed)
    return Prepared(raw, clean, report, actions, train, test, oot)


# ----------------------------------------------------------------------------- training

def _sklearn_logit_check(X: np.ndarray, y: np.ndarray, ours: np.ndarray):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    lr = LogisticRegression(C=1e10, max_iter=1000, tol=1e-10).fit(X, y)
    theirs = np.r_[lr.intercept_, lr.coef_.ravel()]
    return float(np.max(np.abs(theirs - ours)))


def stage_train(cfg: Config, prep: Prepared | None = None):
    prep = prep or prepare(cfg)
    tr = prep.train
    y = tr[TARGET].to_numpy()
    binnings = bin_features(tr, y, FEATURES)
    woe = woe_matrix(tr, binnings)
    sel = select_features(binnings, woe, min_iv=0.02, max_corr=0.70)
    fit, kept, drop_log = fit_with_pruning(woe[sel.selected], y)
    sc = Scorecard.from_fit({k: binnings[k] for k in kept}, fit, Scaling())
    sc.save(cfg.artifacts / "scorecard.json")

    X = woe[kept].to_numpy()
    info = {
        "data": {"n_raw": len(prep.raw), "n_clean": len(prep.clean),
                 "splits": {name: {"n": len(d), "bad_rate": float(d[TARGET].mean()),
                                   "start": d["origination_date"].min(),
                                   "end": d["origination_date"].max()}
                            for name, d in (("train", prep.train), ("test", prep.test),
                                            ("oot", prep.oot))}},
        "quality": {"issues": prep.quality.issues_frame(), "counts": prep.quality.counts,
                    "missingness": prep.quality.missingness, "outliers": prep.quality.outliers,
                    "leakage": prep.quality.leakage, "vintages": prep.quality.vintages,
                    "actions": prep.actions},
        "iv_table": iv_table(binnings), "selection_dropped": sel.dropped,
        "selection_warnings": sel.warnings, "selected": sel.selected, "pruning_log": drop_log,
        "excluded_from_model": EXCLUDED_FROM_MODEL,
        "model": {"coef_table": fit.table(), "vif": vif(woe[kept]).to_dict(),
                  "mcfadden_r2": fit.mcfadden_r2, "lr_pvalue": fit.lr_pvalue,
                  "n_iter": fit.n_iter, "converged": fit.converged, "n_obs": fit.n_obs,
                  "max_abs_diff_vs_scipy": float(np.max(np.abs(
                      fit_logit_scipy(X, y) - fit.coef))),
                  "max_abs_diff_vs_sklearn": _sklearn_logit_check(X, y, fit.coef)},
        "scaling": {"base_score": sc.scaling.base_score, "base_odds": sc.scaling.base_odds,
                    "pdo": sc.scaling.pdo, "factor": sc.scaling.factor,
                    "offset": sc.scaling.offset},
        "points_table": sc.points_table(),
    }
    save_json(info, cfg.artifacts / "training.json")
    return sc, info


# --------------------------------------------------------------------------- validation

def _fit_challenger(prep: Prepared, seed: int):
    """Gradient boosting on the raw features, only if scikit-learn is installed."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return None
    levels = {"purpose": generate.PURPOSES, "home_ownership": list(generate.HOME_EFFECT),
              "region": generate.REGIONS}

    def encode(d: pd.DataFrame) -> np.ndarray:
        cols = [d[c].to_numpy(dtype=float) for c in NUMERIC]
        for c in CATEGORICAL:
            codes = pd.Categorical(d[c], categories=levels[c]).codes.astype(float)
            cols.append(np.where(codes < 0, np.nan, codes))
        return np.column_stack(cols)

    mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=10, min_samples_leaf=150,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, categorical_features=mask, random_state=seed)
    clf.fit(encode(prep.train), prep.train[TARGET])
    return lambda d: clf.predict_proba(encode(d))[:, 1], int(clf.n_iter_)


def _sample_metrics(d: pd.DataFrame, score: np.ndarray, pd_pred: np.ndarray) -> dict:
    y, risk = d[TARGET].to_numpy(), -score
    return {"n": len(d), "bad_rate": float(y.mean()), "mean_pd": float(pd_pred.mean()),
            "auc": metrics.auc(y, risk), "gini": metrics.gini(y, risk),
            "ks": metrics.ks_statistic(y, risk), "brier": metrics.brier(y, pd_pred)}


def _recalibration_shift(y: np.ndarray, p: np.ndarray) -> float:
    """Logit shift delta so that the mean recalibrated PD equals the observed bad rate."""
    z = logit(np.clip(p, 1e-9, 1 - 1e-9))
    return float(brentq(lambda d: expit(z + d).mean() - y.mean(), -5, 5))


def stage_validate(cfg: Config, prep: Prepared | None = None, sc: Scorecard | None = None):
    prep = prep or prepare(cfg)
    sc = sc or Scorecard.load(cfg.artifacts / "scorecard.json")
    econ = business.Economics()
    samples = {"train": prep.train, "test": prep.test, "oot": prep.oot}
    score = {k: sc.score(d) for k, d in samples.items()}
    pdp = {k: sc.predict_pd(d) for k, d in samples.items()}
    y = {k: d[TARGET].to_numpy() for k, d in samples.items()}
    res: dict = {"metrics": {k: _sample_metrics(d, score[k], pdp[k]) for k, d in samples.items()}}

    lo, hi = metrics.bootstrap_auc(y["oot"], -score["oot"], N_BOOT, cfg.seed)[1:]
    res["auc_ci_oot"] = {"low": lo, "high": hi, "n_boot": N_BOOT}
    lo, hi = metrics.bootstrap_auc(y["test"], -score["test"], N_BOOT, cfg.seed)[1:]
    res["auc_ci_test"] = {"low": lo, "high": hi, "n_boot": N_BOOT}

    cal = {k: metrics.calibration_table(y[k], pdp[k]) for k in ("test", "oot")}
    res["calibration"] = dict(cal)
    res["hosmer_lemeshow"] = {k: dict(zip(("chi2", "p_value"), metrics.hosmer_lemeshow(t),
                                          strict=True)) for k, t in cal.items()}
    shift = _recalibration_shift(y["oot"], pdp["oot"])
    p_rc = expit(logit(pdp["oot"]) + shift)
    res["recalibration"] = {"logit_shift": shift,
                            "oot_brier_before": res["metrics"]["oot"]["brier"],
                            "oot_brier_after": metrics.brier(y["oot"], p_rc),
                            "oot_hl_after": metrics.hosmer_lemeshow(
                                metrics.calibration_table(y["oot"], p_rc))[0]}
    res["lift"] = {k: metrics.lift_table(y[k], -score[k]) for k in ("test", "oot")}

    # Stability: population stability of the score and of each characteristic.
    chars = []
    for f in sc.features:
        b = sc.binnings[f]
        e = np.bincount(b.assign(prep.train[f]), minlength=b.n_bins + 1)
        a = np.bincount(b.assign(prep.oot[f]), minlength=b.n_bins + 1)
        v = metrics.psi_from_counts(e, a)
        chars.append({"characteristic": f, "psi": v, "reading": metrics.psi_label(v)})
    score_psi = metrics.psi(score["train"], score["oot"])
    res["psi"] = {"score_train_vs_oot": score_psi, "score_label": metrics.psi_label(score_psi),
                  "score_train_vs_test": metrics.psi(score["train"], score["test"]),
                  "characteristics": chars}

    challenger = _fit_challenger(prep, cfg.seed)
    ch_pred = None
    if challenger is not None:
        predict, n_trees = challenger
        ch = ch_pred = {k: predict(d) for k, d in samples.items()}
        res["challenger"] = {
            "model": "HistGradientBoostingClassifier (raw features, early stopping)",
            "n_iter": n_trees,
            "metrics": {k: {"auc": metrics.auc(y[k], ch[k]), "gini": metrics.gini(y[k], ch[k]),
                            "ks": metrics.ks_statistic(y[k], ch[k])} for k in samples},
            "auc_diff_oot": dict(zip(("diff", "low", "high"), metrics.bootstrap_auc_diff(
                y["oot"], ch["oot"], -score["oot"], N_BOOT, cfg.seed), strict=True))}
    else:
        res["challenger"] = None

    if cfg.meta_path.exists():
        meta = load_json(cfg.meta_path)
        _, truth = generate.generate_loans(meta["n_base"], meta["seed"], return_truth=True)
        t = truth.reindex(prep.oot["loan_id"]).to_numpy()
        res["oracle"] = {"oot_auc_true_pd": metrics.auc(y["oot"], t),
                         "oot_ks_true_pd": metrics.ks_statistic(y["oot"], t),
                         "oot_mean_true_pd": float(t.mean())}

    # Business layer: the cutoff is chosen on the in-time test sample, judged out-of-time.
    def args(k):
        d = samples[k]
        return score[k], y[k], d["loan_amount"].to_numpy(), d["term"].to_numpy(), econ

    cut_test = business.cutoff_analysis(*args("test"))
    cut_oot = business.cutoff_analysis(*args("oot"))
    cutoff = business.optimal_cutoff(cut_test)
    res["business"] = {
        "economics": econ.to_dict(), "chosen_cutoff": cutoff,
        "cutoff_table_test": cut_test, "cutoff_table_oot": cut_oot,
        "oot_at_cutoff": business.evaluate_cutoff(*args("oot"), cutoff),
        "test_at_cutoff": business.evaluate_cutoff(*args("test"), cutoff),
        "el_by_band": business.expected_loss_by_band(
            score["oot"], pdp["oot"], y["oot"], prep.oot["loan_amount"], econ,
            business.band_edges(score["train"])),
        "ifrs9": None}
    ifrs = business.ifrs9_illustration(pdp["oot"], y["oot"], prep.oot["loan_amount"],
                                       prep.oot["term"], econ, seed=cfg.seed)
    res["business"]["ifrs9"] = {"table": ifrs["table"], "total_ecl": ifrs["total_ecl"],
                                "total_ead": ifrs["total_ead"]}
    age = pd.cut(prep.oot["age"], [0, 29, 44, 61, 200], labels=["under 30", "30 to 44",
                                                                "45 to 61", "62 and over"])
    res["business"]["adverse_impact_age"] = business.adverse_impact_screen(
        age[age.notna()], (score["oot"] >= cutoff)[age.notna().to_numpy()])

    save_json(res, cfg.artifacts / "validation.json")
    plots.make_all(cfg.docs, sc, score, y, res, ch_pred)
    return res


def stage_report(cfg: Config) -> Path:
    from credit_scorecard import report
    return report.write_report(cfg)


def run_all(cfg: Config, log=print) -> dict:
    t0 = time.time()
    if not cfg.data_path.exists() or not cfg.meta_path.exists() \
            or load_json(cfg.meta_path).get("n_base") != cfg.n \
            or load_json(cfg.meta_path).get("seed") != cfg.seed:
        stage_generate(cfg)
        log(f"[1/4] generated synthetic dataset ({cfg.data_path})")
    else:
        log(f"[1/4] reusing dataset {cfg.data_path}")
    prep = prepare(cfg)
    sc, _ = stage_train(cfg, prep)
    log(f"[2/4] trained scorecard with {len(sc.features)} characteristics")
    res = stage_validate(cfg, prep, sc)
    m = res["metrics"]["oot"]
    log(f"[3/4] validated: OOT AUC {m['auc']:.3f}, KS {m['ks']:.3f}, "
        f"score PSI {res['psi']['score_train_vs_oot']:.3f}")
    path = stage_report(cfg)
    log(f"[4/4] wrote {path} ({time.time() - t0:.0f}s total)")
    return res
