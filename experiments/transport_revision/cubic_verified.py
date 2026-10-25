"""Verified small cubic solver for the transport-revision experiments.

This is an optional experimental backend, not a modification of the upstream
solver. It minimizes h.T v + v.T H v / 2 + sigma * ||v||**3 / 6.
The upstream two-versus-three return-value convention is preserved.

Numerical acceptance requires all cubic global-optimality conditions to the
requested tolerances: actual cubic stationarity, the secular equation, and
positive semidefiniteness of both the returned-shift and actual-shift Hessians.
Tolerances are absolute-plus-relative, with the scales reported in info.
No failed solve is silently returned as a successful step.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import eigh, norm
from scipy.sparse.linalg import LinearOperator


def _dense_symmetric(H: Any) -> np.ndarray:
    if isinstance(H, LinearOperator):
        n, m = H.shape
        if n != m:
            raise ValueError("H must be square")
        H = H @ np.eye(n)
    array = np.asarray(H)
    if np.iscomplexobj(array):
        raise ValueError("This backend requires a real matrix")
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("H must be square")
    if not np.all(np.isfinite(array)):
        raise ValueError("H must be finite")
    return 0.5 * (array + array.T)


def solve_cubic_subproblem(
    H: Any,
    h: Any,
    sigma: float,
    tol_stat: float = 1e-10,
    tol_sec: float = 1e-10,
    max_iter: int = 100,
    method: str = "auto",
    bracket_scale: float = 2.0,
    return_info: bool = True,
    record_history: bool = False,
):
    """Return (v, lambda, info), or (v, lambda) when return_info is false.

    The global conditions are (H+lambda*I)v=-h, lambda=sigma*||v||/2,
    and H+lambda*I >= 0. An indefinite hard case adds a component in the
    lowest eigenspace; solving only the regular secular equation is insufficient.
    A RuntimeError explicitly marks a numerical failure.
    """
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma must be finite and positive")
    if not (math.isfinite(tol_stat) and math.isfinite(tol_sec) and tol_stat > 0 and tol_sec > 0):
        raise ValueError("tol_stat and tol_sec must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if method not in ("auto", "newton", "bisection"):
        raise ValueError("method must be auto, newton, or bisection")
    if not (math.isfinite(bracket_scale) and bracket_scale > 1):
        raise ValueError("bracket_scale must exceed one")
    Hs = _dense_symmetric(H)
    if np.iscomplexobj(h):
        raise ValueError("This backend requires a real gradient")
    h = np.asarray(h, dtype=float).reshape(-1)
    if h.size != Hs.shape[0] or not np.all(np.isfinite(h)):
        raise ValueError("h must be finite and match H")
    gamma = sigma / 2.0
    history = []
    if h.size == 0:
        info = {
            "iterations": 0, "method": "empty", "psi": 0.0,
            "res_stat": 0.0, "res_sec": 0.0, "min_shifted_eig": 0.0,
            "actual_min_shifted_eig": 0.0, "converged": True,
            "lam_lo": 0.0, "lam_hi": 0.0, "hard_case": False,
        }
        if record_history:
            info["history"] = []
        return (h.copy(), 0.0, info) if return_info else (h.copy(), 0.0)

    mu, U = eigh(Hs, check_finite=False)
    alpha = U.T @ (-h)
    h_norm = float(norm(h))
    spectral_norm = float(np.max(np.abs(mu)))
    lower = max(0.0, -float(mu[0]))
    shifted = mu + lower
    machine = np.finfo(float).eps
    eig_tolerance = 32.0 * machine * max(1.0, spectral_norm)
    gradient_tolerance = 32.0 * machine * max(1.0, h_norm)
    kernel = shifted <= eig_tolerance
    base = np.zeros_like(alpha)
    base[~kernel] = alpha[~kernel] / shifted[~kernel]
    base_norm = float(norm(base))
    hard_case = False
    iterations = 0
    lo = hi = 0.0

    if h_norm == 0.0 and lower == 0.0:
        coeff = np.zeros_like(alpha)
        lam = 0.0
        solve_method = "zero_gradient_psd"
    elif (
        lower > 0.0 and np.any(kernel)
        and float(norm(alpha[kernel])) <= gradient_tolerance
        and base_norm <= lower / gamma
    ):
        # A tiny null-space gradient may be treated as zero only if the final
        # actual KKT check still passes. The true lowest eigenvector is used.
        coeff = base.copy()
        radius = lower / gamma
        remainder = math.sqrt(max(0.0, (radius - base_norm) * (radius + base_norm)))
        coeff[0] = math.copysign(remainder, alpha[0]) if alpha[0] != 0.0 else remainder
        lam = lower
        hard_case = True
        solve_method = "hard_case"
    else:
        # Work with tau=lambda-lower to avoid subtracting nearly equal large
        # eigenvalues in the regular problem near an indefinite hard case.
        def secular(tau: float):
            denominator = shifted + tau
            if np.any(denominator <= 0):
                return math.inf, -math.inf, np.zeros_like(alpha), math.inf
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                vector = alpha / denominator
                vector_norm = float(norm(vector, check_finite=False))
                if not np.isfinite(vector_norm):
                    return math.inf, -math.inf, vector, vector_norm
                if vector_norm == 0:
                    derivative = -1.0
                else:
                    # Normalize before squaring to reduce avoidable overflow.
                    unit = vector / vector_norm
                    derivative = -gamma * vector_norm * float(np.sum(unit * unit / denominator)) - 1.0
                return gamma * vector_norm - (lower + tau), derivative, vector, vector_norm

        lo = 0.0
        hi = max(np.finfo(float).tiny, math.sqrt(gamma) * math.sqrt(h_norm))
        if not math.isfinite(hi):
            raise RuntimeError("Cubic root scale exceeds floating-point range")
        for _ in range(100):
            if secular(hi)[0] <= 0:
                break
            hi *= bracket_scale
            if not math.isfinite(hi):
                raise RuntimeError("Cubic secular bracketing overflow")
        else:
            raise RuntimeError("Could not bracket the regular cubic secular root")
        tau = 0.5 * hi
        root_converged = False
        # Use a stricter secular tolerance internally so the reconstructed
        # actual stationarity check is not defeated by scalar-root error.
        root_tol = min(tol_sec, tol_stat) * 0.02
        for iterations in range(1, max_iter + 1):
            psi, derivative, coeff, vector_norm = secular(tau)
            lam = lower + tau
            if record_history:
                history.append({
                    "iter": iterations, "lam": float(lam), "tau": float(tau),
                    "psi": float(psi), "res_sec": float(abs(psi)),
                })
            if abs(psi) <= root_tol * (1.0 + lam + gamma * vector_norm):
                root_converged = True
                break
            if psi > 0:
                lo = tau
            else:
                hi = tau
            candidate = math.nan
            if method != "bisection" and math.isfinite(derivative) and derivative < 0:
                candidate = tau - psi / derivative
            if not math.isfinite(candidate) or not lo < candidate < hi:
                candidate = lo + 0.5 * (hi - lo)
            if candidate == tau:
                break
            tau = candidate
        if not root_converged:
            raise RuntimeError(
                f"Cubic secular solve did not converge in {iterations} iterations: "
                f"lambda={lower + tau!r}, psi={secular(tau)[0]!r}"
            )
        lam = lower + tau
        solve_method = "secular_bisection" if method == "bisection" else "secular_newton"

    v = U @ coeff
    v_norm = float(norm(v))
    actual_shift = gamma * v_norm
    actual_residual = Hs @ v + h + actual_shift * v
    res_stat = float(norm(actual_residual))
    res_sec = abs(lam - actual_shift)
    min_shifted = float(mu[0] + lam)
    actual_min_shifted = float(mu[0] + actual_shift)
    stat_scale = 1.0 + h_norm + spectral_norm * v_norm + actual_shift * v_norm
    sec_scale = 1.0 + abs(lam) + actual_shift
    psd_scale = 1.0 + spectral_norm + max(abs(lam), actual_shift)
    converged = bool(
        np.all(np.isfinite(v))
        and all(math.isfinite(value) for value in (lam, res_stat, res_sec, stat_scale, sec_scale, psd_scale))
        and res_stat <= tol_stat * stat_scale
        and res_sec <= tol_sec * sec_scale
        and min_shifted >= -tol_stat * psd_scale
        and actual_min_shifted >= -tol_stat * psd_scale
    )
    info = {
        "iterations": iterations, "method": solve_method,
        "psi": float(actual_shift - lam), "res_stat": res_stat, "res_sec": res_sec,
        "min_shifted_eig": min_shifted, "actual_min_shifted_eig": actual_min_shifted,
        "res_stat_relative": res_stat / stat_scale,
        "res_sec_relative": res_sec / sec_scale,
        "stat_scale": stat_scale, "sec_scale": sec_scale, "psd_scale": psd_scale,
        "converged": converged, "hard_case": hard_case,
        "lam_lo": float(lower + lo), "lam_hi": float(lower + hi),
    }
    if record_history:
        info["history"] = history
    if not converged:
        raise RuntimeError(f"Cubic global-optimality verification failed: {info}")
    return (v, float(lam), info) if return_info else (v, float(lam))
