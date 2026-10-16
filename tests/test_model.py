import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

from credit_scorecard.model import fit_logit, fit_logit_scipy, fit_with_pruning, vif


def _sim(n=3000, beta=(-1.0, 0.8, -0.5, 0.3), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta) - 1))
    y = (rng.random(n) < expit(beta[0] + X @ np.array(beta[1:]))).astype(float)
    return X, y


def test_irls_matches_scipy_bfgs():
    X, y = _sim()
    fit = fit_logit(X, y)
    assert fit.coef == pytest.approx(fit_logit_scipy(X, y), abs=1e-5)
    assert fit.converged


def test_irls_saturated_two_by_two_table_closed_form():
    x = np.r_[np.zeros(100), np.ones(100)]
    y = np.r_[np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(40), np.zeros(60)]]
    fit = fit_logit(x.reshape(-1, 1), y)
    assert fit.coef[0] == pytest.approx(np.log(20 / 80))
    assert fit.coef[1] == pytest.approx(np.log((40 / 60) / (20 / 80)))
    assert fit.se[0] == pytest.approx(np.sqrt(1 / 80 + 1 / 20))
    assert fit.se[1] == pytest.approx(np.sqrt(1 / 80 + 1 / 20 + 1 / 60 + 1 / 40))


def test_intercept_only_model():
    y = np.r_[np.ones(30), np.zeros(170)]
    fit = fit_logit(np.empty((200, 0)), y)
    p = 0.15
    assert fit.coef[0] == pytest.approx(np.log(p / (1 - p)))
    assert fit.se[0] == pytest.approx(1 / np.sqrt(200 * p * (1 - p)))


def test_recovers_known_coefficients_within_standard_errors():
    beta = (-1.0, 0.8, -0.5, 0.3)
    X, y = _sim(n=60000, beta=beta, seed=4)
    fit = fit_logit(X, y)
    assert np.all(np.abs(fit.coef - np.array(beta)) < 4 * fit.se)


def test_wald_statistics_are_consistent():
    X, y = _sim(seed=2)
    fit = fit_logit(X, y, names=["a", "b", "c"])
    assert fit.z == pytest.approx(fit.coef / fit.se)
    assert fit.p_values == pytest.approx(2 * stats.norm.sf(np.abs(fit.z)))
    assert fit.names == ["intercept", "a", "b", "c"]
    assert np.allclose(fit.cov, fit.cov.T)
    assert list(fit.table().columns) == ["term", "coef", "std_err", "z", "p_value"]


def test_standard_errors_match_numerical_hessian():
    X, y = _sim(n=1500, seed=6)
    fit = fit_logit(X, y)
    D = np.column_stack([np.ones(len(X)), X])
    p = expit(D @ fit.coef)
    info = (D * (p * (1 - p))[:, None]).T @ D
    assert fit.se == pytest.approx(np.sqrt(np.diag(np.linalg.inv(info))), rel=1e-6)


def test_likelihood_and_fit_statistics():
    X, y = _sim()
    fit = fit_logit(X, y)
    assert fit.loglik > fit.null_loglik
    assert 0 < fit.mcfadden_r2 < 1
    assert fit.lr_pvalue < 1e-6


def test_predict_proba_matches_expit():
    X, y = _sim(n=500)
    fit = fit_logit(X, y)
    D = np.column_stack([np.ones(len(X)), X])
    assert fit.predict_proba(X) == pytest.approx(expit(D @ fit.coef))


def test_ridge_shrinks_coefficients():
    X, y = _sim(n=800, seed=8)
    assert np.linalg.norm(fit_logit(X, y, ridge=50.0).coef[1:]) < \
        np.linalg.norm(fit_logit(X, y).coef[1:])


def test_perfect_separation_with_ridge_stays_finite():
    x = np.r_[-np.ones(50), np.ones(50)].reshape(-1, 1)
    y = (x.ravel() > 0).astype(float)
    fit = fit_logit(x, y, ridge=1e-3)
    assert np.isfinite(fit.coef).all() and fit.coef[1] > 1


def test_input_validation():
    with pytest.raises(ValueError):
        fit_logit(np.ones((5, 2)), np.ones(4))


def test_without_intercept():
    X, y = _sim(n=2000, beta=(0.0, 0.7, -0.4, 0.2), seed=9)
    fit = fit_logit(X, y, add_intercept=False)
    assert len(fit.coef) == 3 and fit.coef == pytest.approx(
        fit_logit_scipy(X, y, add_intercept=False), abs=1e-5)


def test_pruning_drops_wrong_sign_feature():
    rng = np.random.default_rng(10)
    n = 6000
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < expit(-2.5 - 1.0 * a + 0.5 * b)).astype(float)  # b raises risk
    fit, kept, log = fit_with_pruning(pd.DataFrame({"a": a, "b": b}), y)
    assert kept == ["a"] and log[0][0] == "b" and "wrong sign" in log[0][1]
    assert fit.coef[1] < 0


def test_pruning_drops_insignificant_feature():
    rng = np.random.default_rng(11)
    n = 6000
    a, junk = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < expit(-2.5 - 1.0 * a - 0.001 * junk)).astype(float)
    _, kept, log = fit_with_pruning(pd.DataFrame({"a": a, "junk": junk}), y)
    assert "a" in kept
    assert all(why.startswith(("wrong sign", "not significant")) for _, why in log)


def test_vif_independent_and_collinear():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=4000), rng.normal(size=4000)
    low = vif(pd.DataFrame({"a": a, "b": b}))
    high = vif(pd.DataFrame({"a": a, "b": b, "c": a + b + rng.normal(0, 0.05, 4000)}))
    assert low.max() < 1.1 and high.max() > 100


def test_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.linear_model")
    X, y = _sim(n=4000, seed=13)
    lr = sk.LogisticRegression(C=1e10, max_iter=2000, tol=1e-10).fit(X, y)
    ours = fit_logit(X, y).coef
    assert np.r_[lr.intercept_, lr.coef_.ravel()] == pytest.approx(ours, abs=1e-4)
