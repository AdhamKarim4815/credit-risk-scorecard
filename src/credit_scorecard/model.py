"""Logistic regression by iteratively reweighted least squares (IRLS), with Wald inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit


@dataclass
class LogitFit:
    names: list[str]  # first entry is "intercept" when one was fitted
    coef: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    loglik: float
    null_loglik: float
    n_obs: int
    n_iter: int
    converged: bool
    has_intercept: bool = True

    @property
    def z(self) -> np.ndarray:
        return self.coef / self.se

    @property
    def p_values(self) -> np.ndarray:
        return 2.0 * stats.norm.sf(np.abs(self.z))

    @property
    def mcfadden_r2(self) -> float:
        return 1.0 - self.loglik / self.null_loglik

    @property
    def lr_pvalue(self) -> float:
        df = len(self.coef) - int(self.has_intercept)
        return float(stats.chi2.sf(2.0 * (self.loglik - self.null_loglik), df))

    def design(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X]) if self.has_intercept else X

    def predict_proba(self, X) -> np.ndarray:
        return expit(self.design(X) @ self.coef)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names, "coef": self.coef, "std_err": self.se,
                             "z": self.z, "p_value": self.p_values})


def _loglik(eta: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def fit_logit(X, y, names: list[str] | None = None, add_intercept: bool = True,
              max_iter: int = 100, tol: float = 1e-10, ridge: float = 0.0) -> LogitFit:
    """Maximum likelihood logistic regression via IRLS (Newton-Raphson) with step halving.

    Standard errors come from the inverse observed information X'WX at the solution.
    ``ridge`` adds an optional L2 penalty (never on the intercept) for separated data.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D with the same number of rows as y")
    k = X.shape[1]
    names = list(names) if names is not None else [f"x{i}" for i in range(k)]
    D = np.column_stack([np.ones(len(X)), X]) if add_intercept else X
    all_names = (["intercept"] if add_intercept else []) + names
    penalty = np.full(D.shape[1], ridge)
    if add_intercept:
        penalty[0] = 0.0

    beta = np.zeros(D.shape[1])
    if add_intercept:
        ybar = float(np.clip(y.mean(), 1e-9, 1 - 1e-9))
        beta[0] = np.log(ybar / (1 - ybar))
    ll = _loglik(D @ beta, y) - 0.5 * float(penalty @ beta**2)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007
        p = expit(D @ beta)
        w = np.maximum(p * (1 - p), 1e-12)
        grad = D.T @ (y - p) - penalty * beta
        hess = (D * w[:, None]).T @ D + np.diag(penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        scale = 1.0
        for _ in range(30):  # step halving keeps the likelihood non-decreasing
            cand = beta + scale * step
            cand_ll = _loglik(D @ cand, y) - 0.5 * float(penalty @ cand**2)
            if cand_ll >= ll - 1e-12:
                break
            scale *= 0.5
        if cand_ll < ll - 1e-12:  # no improving step exists: stop at the current point
            converged = np.max(np.abs(grad)) < 1e-6 * max(1.0, len(y))
            break
        beta, ll = cand, cand_ll
        if np.max(np.abs(scale * step)) < tol:
            converged = True
            break

    p = expit(D @ beta)
    w = np.maximum(p * (1 - p), 1e-12)
    info = (D * w[:, None]).T @ D + np.diag(penalty)
    cov = np.linalg.pinv(info)
    ybar = y.mean()
    null_ll = float(len(y) * (ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))) \
        if 0 < ybar < 1 else 0.0
    if not add_intercept:
        null_ll = float(len(y) * np.log(0.5))
    return LogitFit(names=all_names, coef=beta, se=np.sqrt(np.diag(cov)), cov=cov, loglik=_loglik(
        D @ beta, y), null_loglik=null_ll, n_obs=len(y), n_iter=n_iter, converged=converged,
        has_intercept=add_intercept)


def fit_logit_scipy(X, y, add_intercept: bool = True) -> np.ndarray:
    """Independent reference solution: BFGS on the negative log-likelihood (for cross-checks)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    D = np.column_stack([np.ones(len(X)), X]) if add_intercept else X

    def fun(b):
        eta = D @ b
        return -np.sum(y * eta - np.logaddexp(0.0, eta)), -(D.T @ (y - expit(eta)))

    res = optimize.minimize(fun, np.zeros(D.shape[1]), jac=True, method="BFGS",
                            options={"gtol": 1e-9, "maxiter": 500})
    return res.x


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factors: 1 / (1 - R^2) from regressing each column on the rest."""
    out = {}
    A = X.to_numpy(dtype=float)
    for j, col in enumerate(X.columns):
        others = np.column_stack([np.ones(len(A)), np.delete(A, j, axis=1)])
        resid = A[:, j] - others @ np.linalg.lstsq(others, A[:, j], rcond=None)[0]
        r2 = 1.0 - resid.var() / A[:, j].var()
        out[col] = 1.0 / max(1.0 - r2, 1e-12)
    return pd.Series(out)


def fit_with_pruning(X: pd.DataFrame, y, expected_sign: int = -1, max_p: float = 0.05):
    """Fit, then iteratively drop features with the wrong sign (worst first), then the
    least significant feature while its Wald p-value exceeds ``max_p``.

    With WoE = ln(good/bad) and a model for the log-odds of BAD, every coefficient
    should be negative (``expected_sign=-1``). Returns (fit, kept_columns, drop_log).
    """
    cols = list(X.columns)
    log: list[tuple[str, str]] = []
    while True:
        fit = fit_logit(X[cols].to_numpy(), y, names=cols)
        coef, p = fit.coef[1:], fit.p_values[1:]
        wrong = [i for i, c in enumerate(coef) if np.sign(c) != expected_sign]
        if wrong:
            i = max(wrong, key=lambda j: abs(fit.z[1:][j]))
            log.append((cols[i], f"wrong sign (coef {coef[i]:+.3f})"))
            cols.pop(i)
        elif len(cols) > 1 and p.max() > max_p:
            i = int(np.argmax(p))
            log.append((cols[i], f"not significant (p = {p[i]:.3f})"))
            cols.pop(i)
        else:
            return fit, cols, log
