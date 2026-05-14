from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh

from .spd import (
    GeometryState,
    assemble_reduced,
    basis_from_candidates,
    full_tangent_basis,
    full_tangent_basis_pair,
    geometry_terms,
    inner_product,
    objective,
    retraction_exp,
    sym,
    tangent_norm,
)


@dataclass
class VanillaNewtonOptions:
    max_iter: int = 100
    tol_grad: float = 1e-6
    ls_c1: float = 1e-4
    ls_beta: float = 0.5
    ls_max_steps: int = 20
    ridge: float = 1e-12
    display_progress: bool = False
    time_limit: Optional[float] = None
    dist_weight: float = 1.0
    inv_linear_weight: float = 1.0
    inv_quadratic_weight: float = 1.0


def _make_spd(X: np.ndarray) -> np.ndarray:
    X = sym(X)
    w, Q = eigh(X)
    eps = 1e-8
    w = np.maximum(w, eps)
    return (Q * w) @ Q.T


def solve_spd_vanilla_newton(
    A: np.ndarray,
    B: np.ndarray,
    X0: np.ndarray,
    opts: Optional[VanillaNewtonOptions] = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if opts is None:
        opts = VanillaNewtonOptions()

    objective_kwargs = {
        "dist_weight": opts.dist_weight,
        "inv_linear_weight": opts.inv_linear_weight,
        "inv_quadratic_weight": opts.inv_quadratic_weight,
    }
    X = _make_spd(X0)
    geom_state, f = geometry_terms(A, B, X, **objective_kwargs)
    gn = tangent_norm(geom_state.X_inv, geom_state.grad_riemannian)

    hist: Dict[str, List[float]] = {"f": [f], "grad_norm": [gn], "alpha": [], "accepted": [], "time": [0.0]}
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
        if opts.ridge > 0:
            HS = HS + opts.ridge * np.eye(HS.shape[0])
        try:
            v = -np.linalg.solve(HS, hS)
        except np.linalg.LinAlgError:
            v = -np.linalg.lstsq(HS, hS, rcond=None)[0]

        eta = np.tensordot(v, basis, axes=(0, 0))
        g_dot_eta = inner_product(geom_state.X_inv, geom_state.grad_riemannian, eta)
        if g_dot_eta >= 0.0:
            eta = -eta
            g_dot_eta = -g_dot_eta

        alpha = 1.0
        accepted = False
        for _ in range(opts.ls_max_steps):
            X_try = retraction_exp(X, alpha * eta)
            f_try = objective(A, B, X_try, **objective_kwargs)
            if f_try <= f + opts.ls_c1 * alpha * g_dot_eta:
                X = _make_spd(X_try)
                geom_state, f = geometry_terms(A, B, X, **objective_kwargs)
                gn = tangent_norm(geom_state.X_inv, geom_state.grad_riemannian)
                accepted = True
                break
            alpha *= opts.ls_beta

        current_time = time.perf_counter() - t0
        hist["alpha"].append(alpha if accepted else 0.0)
        hist["accepted"].append(1.0 if accepted else 0.0)
        hist["f"].append(f)
        hist["grad_norm"].append(gn)
        hist["time"].append(current_time)

        if opts.display_progress:
            msg = (
                f"[spd-vanilla] iter {it}/{opts.max_iter} "
                f"f={f:.6e} ||grad||={gn:.3e} alpha={hist['alpha'][-1]:.2e} accepted={int(accepted)}"
            )
            print(msg, flush=True)

        if opts.time_limit is not None and current_time >= opts.time_limit:
            hist["time"][-1] = opts.time_limit
            break

        if opts.time_limit is None and gn <= opts.tol_grad:
            break

    return X, hist
