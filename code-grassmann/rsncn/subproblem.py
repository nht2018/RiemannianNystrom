import math
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List

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


def _as_dense_symmetric(H: Union[np.ndarray, LinearOperator]) -> np.ndarray:
    if isinstance(H, np.ndarray):
        Hs = 0.5 * (H + H.T)
        return np.asarray(Hs, dtype=float)
    if isinstance(H, LinearOperator):
        n, m = H.shape
        if n != m:
            raise ValueError("H must be square")
        I = np.eye(n)
        cols = [H @ I[:, i] for i in range(n)]
        Hd = np.column_stack(cols)
        Hd = 0.5 * (Hd + Hd.T)
        return np.asarray(Hd, dtype=float)
    raise TypeError("H must be a numpy array or scipy.sparse.linalg.LinearOperator")


def _eigen_decompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    denom = np.maximum(mu + lam, tiny)
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
):
    h = np.asarray(h, dtype=float).reshape(-1)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    Hs = _as_dense_symmetric(H)
    n = Hs.shape[0]
    if n == 0:
        empty = np.zeros(0)
        if return_info:
            return empty, 0.0, {"iterations": 0, "method": "empty"}
        return empty, 0.0

    mu, U = _eigen_decompose(Hs)
    alpha = U.T @ (-h)
    alpha_sq = alpha * alpha
    tiny = 1e-14

    lam_lo = max(0.0, -float(mu.min()) + tiny)
    lam_hi = max(lam_lo + 1.0, lam_lo * 2.0)
    psi_lo, _, _, _, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam_lo, tiny)
    psi_hi, _, _, _, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam_hi, tiny)
    it_bracket = 0
    while psi_lo * psi_hi > 0 and it_bracket < 50:
        lam_hi = lam_hi * bracket_scale + 1.0
        psi_hi, _, _, _, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam_hi, tiny)
        it_bracket += 1

    if psi_lo * psi_hi > 0:
        lam = max(lam_lo, lam_hi)
        denom = np.maximum(mu + lam, tiny)
        v = U @ (alpha / denom)
        if return_info:
            info = {
                "iterations": 0,
                "method": "bracket_fail",
                "psi": float(psi_hi),
                "res_stat": float(np.linalg.norm(Hs @ v + h + lam * v)),
                "res_sec": float(abs(0.5 * sigma * np.linalg.norm(v) - lam)),
                "converged": False,
                "lam_lo": float(lam_lo),
                "lam_hi": float(lam_hi),
            }
            return v, lam, info
        return v, lam

    hist: List[Dict[str, float]] = []
    lam = max(lam_lo, min(lam_hi, 0.5 * (lam_lo + lam_hi)))
    for it in range(1, max_iter + 1):
        psi, dpsi, y, nv, _ = _psi_eval(mu, alpha, alpha_sq, sigma, lam, tiny)
        v = U @ y
        res_stat = np.linalg.norm(Hs @ v + h + lam * v)
        res_sec = abs(0.5 * sigma * nv - lam)
        if record_history:
            hist.append({"iter": float(it), "lam": float(lam), "psi": float(psi), "res_stat": float(res_stat), "res_sec": float(res_sec)})
        if res_stat <= tol_stat and res_sec <= tol_sec:
            info = {
                "iterations": it,
                "method": method,
                "psi": float(psi),
                "res_stat": float(res_stat),
                "res_sec": float(res_sec),
                "converged": True,
                "lam_lo": float(lam_lo),
                "lam_hi": float(lam_hi),
            }
            if record_history:
                info["history"] = hist
            if return_info:
                return v, lam, info
            return v, lam
        if method == "newton" or (method == "auto" and dpsi < 0):
            lam_new = lam - psi / dpsi
        else:
            lam_new = 0.5 * (lam_lo + lam_hi)
        if psi > 0:
            lam_lo = lam
        else:
            lam_hi = lam
        if not (lam_lo <= lam_new <= lam_hi):
            lam_new = 0.5 * (lam_lo + lam_hi)
        lam = lam_new

    info = {
        "iterations": max_iter,
        "method": method,
        "psi": float(psi),
        "res_stat": float(res_stat),
        "res_sec": float(res_sec),
        "converged": False,
        "lam_lo": float(lam_lo),
        "lam_hi": float(lam_hi),
    }
    if record_history:
        info["history"] = hist
    if return_info:
        return v, lam, info
    return v, lam
