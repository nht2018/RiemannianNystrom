import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator


@dataclass
class CubicSolveInfo:
    iterations: int
    method: str
    psi: float
    res_stat: float
    res_sec: float
    converged: bool
    lam_lo: float
    lam_hi: float
    # history is attached at return time (list of dicts per-iteration)


def _as_dense_symmetric(H: Union[np.ndarray, LinearOperator]) -> np.ndarray:
    if isinstance(H, np.ndarray):
        Hs = 0.5 * (H + H.T)
        return np.asarray(Hs, dtype=float)
    if isinstance(H, LinearOperator):
        n, m = H.shape
        if n != m:
            raise ValueError("H must be square")
        # Materialize dense small operator via basis applications
        I = np.eye(n)
        cols = [H @ I[:, i] for i in range(n)]
        Hd = np.column_stack(cols)
        Hd = 0.5 * (Hd + Hd.T)
        return np.asarray(Hd, dtype=float)
    raise TypeError("H must be a numpy array or scipy.sparse.linalg.LinearOperator")


def _eigen_decompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # eigh returns ascending eigenvalues
    w, U = eigh(H, overwrite_a=True, check_finite=False)
    return w, U


def _psi_eval(
    mu: np.ndarray,
    alpha: np.ndarray,
    alpha_sq: np.ndarray,
    sigma: float,
    lam: float,
    tiny: float,
) -> Tuple[float, float, np.ndarray, float, np.ndarray]:
    denom = mu + lam
    denom = np.maximum(denom, tiny)
    y = alpha / denom
    nv_sq = float(np.dot(y, y))
    nv = math.sqrt(nv_sq)
    if nv == 0.0:
        dnorm = 0.0
    else:
        dnorm = -float(np.sum(alpha_sq / (denom ** 3))) / nv
    psi = 0.5 * sigma * nv - lam
    dpsi = 0.5 * sigma * dnorm - 1.0
    return psi, dpsi, y, nv, denom


try:  # pragma: no cover - optional line_profiler support
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def profile(func):  # type: ignore
        return func


@profile
def solve_cubic_subproblem(
    H: Union[np.ndarray, LinearOperator],
    h: ArrayLike,
    sigma: float,
    tol_stat: float = 1e-10,
    tol_sec: float = 1e-10,
    max_iter: int = 100,
    method: str = "auto",
    bracket_scale: float = 2.0,
    return_info: bool = True,
    record_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, Dict[str, float]]]:
    """
    Solve the cubic-regularized subproblem:
        min_v  h^T v + 0.5 v^T H v + (sigma/6) ||v||^3

    KKT: (H + lam I)v = -h, lam = (sigma/2) ||v||, lam >= 0.

    Parameters
    ----------
    H : ndarray or LinearOperator (symmetric)
        Reduced Hessian in the sketch subspace.
    h : array_like (shape (l,))
        Reduced gradient in the sketch subspace.
    sigma : float (>0)
        Cubic regularization parameter.
    tol_stat : float
        Tolerance on stationarity residual ||(H+lam I)v + h||.
    tol_sec : float
        Tolerance on secular equation |(sigma/2)*||v|| - lam|.
    max_iter : int
        Maximum root-finding iterations.
    method : {"auto", "bisection", "newton"}
        Root-finding strategy for the secular equation.
    bracket_scale : float
        Factor to expand the upper bracket when searching for a sign change.
    return_info : bool
        If True, return (v, lam, info_dict); else return (v, lam).
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if h.ndim != 1:
        raise ValueError("h must be a 1-D array")

    # Trivial case
    if float(np.linalg.norm(h)) == 0.0:
        v = np.zeros_like(h)
        lam = 0.0
        info = CubicSolveInfo(
            iterations=0,
            method="trivial",
            psi=0.0,
            res_stat=0.0,
            res_sec=0.0,
            converged=True,
            lam_lo=0.0,
            lam_hi=0.0,
        )
        return (v, lam, info.__dict__) if return_info else (v, lam)

    Hd = _as_dense_symmetric(H)
    mu, U = _eigen_decompose(Hd)
    alpha = U.T @ h
    alpha_sq = alpha ** 2
    mu_norm_inf = np.linalg.norm(mu, ord=np.inf)

    # Lower bracket: ensure (H + lam I) is SPD and avoid singularity
    mu_min = float(np.min(mu))
    lam_lo = max(0.0, -mu_min)
    eps = 10 * np.finfo(float).eps * (mu_norm_inf + 1.0)
    tiny = eps
    if lam_lo > 0:
        lam_lo += eps

    # Ensure Psi(lam_lo) >= 0 (monotonicity guarantees it except for numeric issues)
    psi_lo, _, _, _, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam_lo, tiny)
    backoff_tries = 0
    while psi_lo < 0 and backoff_tries < 5:
        lam_lo = max(0.0, lam_lo * 0.5)
        if lam_lo == 0.0:
            lam_lo += eps
        psi_lo, _, _, _, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam_lo, tiny)
        backoff_tries += 1

    # Upper bracket: start from ||h||/sigma and expand until Psi <= 0
    lam_hi = max(lam_lo * bracket_scale, 2.0 * float(np.linalg.norm(alpha)) / sigma)
    psi_hi, _, _, _, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam_hi, tiny)
    expand_iter = 0
    while psi_hi > 0 and expand_iter < 64:
        lam_hi *= bracket_scale
        psi_hi, _, _, _, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam_hi, tiny)
        expand_iter += 1

    if psi_lo < 0 or psi_hi > 0:
        # As a last resort, widen aggressively
        lam_lo = max(0.0, lam_lo - bracket_scale * (eps + 1.0)) + eps
        lam_hi = lam_hi + bracket_scale * (eps + 1.0)

    # Root-finding
    use_newton = method in ("newton", "auto")
    lam = max(lam_lo, min(lam_hi, 0.5 * (lam_lo + lam_hi)))

    iterations = 0
    converged = False
    last_method = "bisection"
    history: List[Dict[str, float]] = []

    for it in range(max_iter):
        iterations = it + 1
        psi, dpsi, y, nv, denom = _psi_eval(mu, alpha, alpha_sq, sigma, lam, tiny)
        v = -(U @ y)
        res_stat_iter = float(np.linalg.norm(alpha - denom * y))
        if record_history:
            history.append(
                {
                    "iter": iterations,
                    "lam": float(lam),
                    "psi": float(psi),
                    "dpsi": float(dpsi),
                    "res_stat": res_stat_iter,
                    "lam_lo": float(lam_lo),
                    "lam_hi": float(lam_hi),
                }
            )

        if abs(psi) <= tol_sec * (1.0 + lam):
            converged = True
            last_method = "newton" if use_newton else "bisection"
            break

        # Update bracket
        if psi > 0:
            lam_lo = lam
        else:
            lam_hi = lam

        if use_newton and dpsi != 0.0 and nv > 0:
            lam_new = lam - psi / dpsi
            if lam_lo < lam_new < lam_hi:
                lam = lam_new
                last_method = "newton"
                continue

        lam = 0.5 * (lam_lo + lam_hi)
        last_method = "bisection"

    # Final values
    psi, _, y, nv, denom = _psi_eval(mu, alpha, alpha_sq, sigma, lam, tiny)
    v = -(U @ y)
    res_stat = float(np.linalg.norm(alpha - denom * y))
    res_sec = abs(psi)
    info = CubicSolveInfo(
        iterations=iterations,
        method=last_method,
        psi=float(psi),
        res_stat=res_stat,
        res_sec=res_sec,
        converged=converged,
        lam_lo=float(lam_lo),
        lam_hi=float(lam_hi),
    )
    info_dict = info.__dict__.copy()
    if record_history:
        info_dict["history"] = history

    return (v, float(lam), info_dict) if return_info else (v, float(lam))
