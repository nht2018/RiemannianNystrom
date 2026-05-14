from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spd import (
    assemble_reduced,
    full_tangent_basis,
    full_tangent_basis_pair,
    geometry_terms,
    inner_product,
    objective,
    retraction_exp,
    sym,
    tangent_norm,
)
from .subproblem import solve_cubic_subproblem


@dataclass
class FullCubicOptions:
    max_iter: int = 100
    tol_grad: float = 1e-6
    tol_pred: float = 1e-12
    tol_step: float = 1e-14
    sigma0: float = 1.0
    sigma_min: float = 1e-6
    sigma_max: Optional[float] = None
    eta1: float = 0.1
    eta2: float = 0.9
    gamma1: float = 0.5
    gamma2: float = 2.0
    eta_accept: Optional[float] = None
    gamma_inc: Optional[float] = None
    gamma_dec: Optional[float] = None
    ls_c1: float = 1e-4
    ls_beta: float = 0.5
    ls_max_steps: int = 20
    ridge: float = 1e-12
    display_progress: bool = False
    time_limit: Optional[float] = None
    dist_weight: float = 1.0
    inv_linear_weight: float = 1.0
    inv_quadratic_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.eta_accept is not None:
            self.eta1 = self.eta_accept
        if self.gamma_inc is not None:
            self.gamma2 = self.gamma_inc
        if self.gamma_dec is not None:
            self.gamma1 = self.gamma_dec
        if not (0.0 < self.eta1 <= self.eta2 < 1.0):
            raise ValueError("Require 0 < eta1 <= eta2 < 1 for ARC.")
        if not (0.0 < self.gamma1 < 1.0 < self.gamma2):
            raise ValueError("Require 0 < gamma1 < 1 < gamma2 for ARC.")
        if self.sigma_max is not None and self.sigma_max <= 0.0:
            raise ValueError("sigma_max must be positive when provided.")


def _make_spd(X: np.ndarray) -> np.ndarray:
    X = sym(X)
    w, Q = np.linalg.eigh(X)
    eps = 1e-8
    w = np.maximum(w, eps)
    return (Q * w) @ Q.T


def solve_spd_full_cubic(
    A: np.ndarray,
    B: np.ndarray,
    X0: np.ndarray,
    opts: Optional[FullCubicOptions] = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if opts is None:
        opts = FullCubicOptions()

    objective_kwargs = {
        "dist_weight": opts.dist_weight,
        "inv_linear_weight": opts.inv_linear_weight,
        "inv_quadratic_weight": opts.inv_quadratic_weight,
    }
    X = _make_spd(X0)
    geom_state, f = geometry_terms(A, B, X, **objective_kwargs)
    grad = geom_state.grad_riemannian
    gn = tangent_norm(geom_state.X_inv, grad)

    sigma = opts.sigma0
    hist: Dict[str, List[float]] = {
        "f": [f],
        "grad_norm": [gn],
        "sigma": [sigma],
        "rho": [],
        "accepted": [],
        "time": [0.0],
        "subprob_grad_norm": [],
        "v_norm_ratio": [],
        "v_norm": [],
        "eta_norm": [],
        "X_norm": [float(np.linalg.norm(X, ord="fro"))],
    }
    t0 = time.perf_counter()

    for it in range(1, opts.max_iter + 1):
        basis, basis_tilde = full_tangent_basis_pair(
            geom_state.X_inv,
            geom_state.distance_state.X_sqrt,
        )
        hS, HS = assemble_reduced(
            A,
            B,
            X,
            basis,
            basis_tilde=basis_tilde,
            geom_state=geom_state,
            **objective_kwargs,
        )
        if sigma == 0.0:
            HSd = HS + (opts.ridge * np.eye(HS.shape[0]) if opts.ridge > 0 else 0)
            try:
                v = -np.linalg.solve(HSd, hS)
            except np.linalg.LinAlgError:
                v = -np.linalg.lstsq(HSd, hS, rcond=None)[0]
        else:
            v, _ = solve_cubic_subproblem(HS, hS, sigma, return_info=False)

        HSv = HS @ v
        v_norm = float(np.linalg.norm(v))
        phi_v = float(hS @ v + 0.5 * v @ HSv + (sigma / 6.0) * (v_norm ** 3))
        subgrad = hS + HSv + (0.5 * sigma * v_norm) * v
        subgrad_norm = float(np.linalg.norm(subgrad))
        try:
            v_newton = -np.linalg.solve(HS, hS)
        except np.linalg.LinAlgError:
            v_newton = -np.linalg.lstsq(HS, hS, rcond=None)[0]
        v_newton_norm = float(np.linalg.norm(v_newton))
        if v_newton_norm == 0.0:
            v_newton_norm = 1.0
        v_norm_ratio = v_norm / v_newton_norm

        eta = np.tensordot(v, basis, axes=(0, 0))
        eta_norm = float(np.linalg.norm(eta, ord="fro"))
        X_trial = retraction_exp(X, eta)
        f_trial = objective(A, B, X_trial, **objective_kwargs)

        pred = -phi_v
        ared = f - f_trial
        rho = ared / pred if pred > 0 else float("nan")
        accepted = (pred > 0) and np.isfinite(rho) and rho >= opts.eta1

        if accepted:
            X = _make_spd(X_trial)
            geom_state, f = geometry_terms(A, B, X, **objective_kwargs)
            grad = geom_state.grad_riemannian
            gn = tangent_norm(geom_state.X_inv, grad)
        if sigma > 0:
            if np.isfinite(rho) and rho >= opts.eta2:
                sigma = max(opts.sigma_min, sigma * opts.gamma1)
            elif (not np.isfinite(rho)) or rho < opts.eta1:
                sigma = sigma * opts.gamma2
                if opts.sigma_max is not None:
                    sigma = min(opts.sigma_max, sigma)

        current_time = time.perf_counter() - t0
        hist["f"].append(f)
        hist["grad_norm"].append(gn)
        hist["sigma"].append(sigma)
        hist["rho"].append(float(rho))
        hist["time"].append(current_time)
        hist["accepted"].append(1.0 if accepted else 0.0)
        hist["subprob_grad_norm"].append(subgrad_norm)
        hist["v_norm_ratio"].append(v_norm_ratio)
        hist["v_norm"].append(v_norm)
        hist["eta_norm"].append(eta_norm)
        hist["X_norm"].append(float(np.linalg.norm(X, ord="fro")))

        if opts.display_progress:
            msg = (
                f"[spd-full-cubic] iter {it}/{opts.max_iter} "
                f"f={f:.6e} ||grad||={gn:.3e} sigma={sigma:.2e} accepted={int(accepted)}"
            )
            print(msg, flush=True)

        if opts.time_limit is not None and current_time >= opts.time_limit:
            hist["time"][-1] = opts.time_limit
            break

        pred_tiny = abs(pred) <= opts.tol_pred * max(1.0, abs(f))
        step_tiny = eta_norm <= opts.tol_step * max(1.0, hist["X_norm"][-1])
        if gn <= opts.tol_grad or pred_tiny or step_tiny:
            break

    return X, hist
