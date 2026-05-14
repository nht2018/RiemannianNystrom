from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spd import (
    assemble_reduced,
    basis_from_candidates,
    full_tangent_basis,
    geometry_terms,
    inner_product,
    objective,
    random_tangent_metric_gaussian,
    random_tangent,
    retraction_exp,
    sym,
    tangent_norm,
)


@dataclass
class NewtonOptions:
    max_iter: int = 200
    tol_grad: float = 1e-6
    sketch_size: int = 20
    include_gradient_in_sketch: bool = True
    sketch_distribution: str = "ambient_symmetric"
    adaptive_sketch: bool = False
    ls_c1: float = 1e-4
    ls_beta: float = 0.5
    ls_max_steps: int = 20
    ridge: float = 1e-12
    random_state: Optional[int] = None
    display_progress: bool = False
    time_limit: Optional[float] = None
    dist_weight: float = 1.0
    inv_linear_weight: float = 1.0
    inv_quadratic_weight: float = 1.0


def _make_spd(X: np.ndarray) -> np.ndarray:
    X = sym(X)
    w, Q = np.linalg.eigh(X)
    eps = 1e-8
    w = np.maximum(w, eps)
    return (Q * w) @ Q.T


def solve_spd_nystrom_newton(
    A: np.ndarray,
    B: np.ndarray,
    X0: np.ndarray,
    opts: Optional[NewtonOptions] = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if opts is None:
        opts = NewtonOptions()

    rng = np.random.default_rng(opts.random_state)
    objective_kwargs = {
        "dist_weight": opts.dist_weight,
        "inv_linear_weight": opts.inv_linear_weight,
        "inv_quadratic_weight": opts.inv_quadratic_weight,
    }
    X = _make_spd(X0)
    geom_state, f = geometry_terms(A, B, X, **objective_kwargs)
    grad = geom_state.grad_riemannian
    gn = tangent_norm(geom_state.X_inv, grad)

    hist: Dict[str, List[float]] = {"f": [f], "grad_norm": [gn], "alpha": [], "accepted": [], "time": [0.0]}
    t0 = time.perf_counter()

    d = X.shape[0]
    tangent_dim = d * (d + 1) // 2
    curr_ell = max(1, min(opts.sketch_size, tangent_dim))
    stagnation = 0
    prev_grad_norm = gn

    for it in range(1, opts.max_iter + 1):
        if opts.adaptive_sketch and it > 1 and curr_ell < tangent_dim:
            if prev_grad_norm > 0 and gn >= prev_grad_norm * 0.9:
                stagnation += 1
            else:
                stagnation = 0
            if stagnation >= 3:
                curr_ell = min(tangent_dim, max(curr_ell + 5, int(curr_ell * 1.5)))
                stagnation = 0

        ell = curr_ell
        if ell >= tangent_dim:
            basis = full_tangent_basis(geom_state.X_inv, geom_state.distance_state.X_sqrt)
            ell = len(basis)
        else:
            candidate_count = max(ell - (1 if opts.include_gradient_in_sketch else 0), 0)
            sketch_grad = grad if opts.include_gradient_in_sketch else None
            if opts.sketch_distribution == "ambient_symmetric":
                candidates = [random_tangent(rng, d) for _ in range(candidate_count)]
                basis = basis_from_candidates(
                    geom_state.X_inv,
                    sketch_grad,
                    candidates,
                    ell,
                    X_sqrt=geom_state.distance_state.X_sqrt,
                    X_inv_sqrt=geom_state.distance_state.X_inv_sqrt,
                )
            elif opts.sketch_distribution == "metric_gaussian":
                X_sqrt = geom_state.distance_state.X_sqrt
                candidates = [random_tangent_metric_gaussian(rng, X_sqrt) for _ in range(candidate_count)]
                basis = basis_from_candidates(
                    geom_state.X_inv,
                    sketch_grad,
                    candidates,
                    ell,
                    X_sqrt=geom_state.distance_state.X_sqrt,
                    X_inv_sqrt=geom_state.distance_state.X_inv_sqrt,
                )
            else:
                raise ValueError(f"Unknown sketch_distribution={opts.sketch_distribution!r}")
            if len(basis) < ell:
                extra = full_tangent_basis(geom_state.X_inv, geom_state.distance_state.X_sqrt)
                basis = basis_from_candidates(
                    geom_state.X_inv,
                    sketch_grad,
                    list(extra),
                    ell,
                    X_sqrt=geom_state.distance_state.X_sqrt,
                    X_inv_sqrt=geom_state.distance_state.X_inv_sqrt,
                )

        hS, HS = assemble_reduced(
            A,
            B,
            X,
            basis,
            geom_state=geom_state,
            **objective_kwargs,
        )
        HSd = HS + (opts.ridge * np.eye(HS.shape[0]) if opts.ridge > 0 else 0)
        try:
            v = -np.linalg.solve(HSd, hS)
        except np.linalg.LinAlgError:
            v = -np.linalg.lstsq(HSd, hS, rcond=None)[0]

        eta = np.tensordot(v, basis, axes=(0, 0))
        g_dot_eta = inner_product(geom_state.X_inv, grad, eta)
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
                grad = geom_state.grad_riemannian
                gn = tangent_norm(geom_state.X_inv, grad)
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
                f"[spd-nystrom] iter {it}/{opts.max_iter} "
                f"f={f:.6e} ||grad||={gn:.3e} ell={len(basis)}"
            )
            print(msg, flush=True)

        if opts.time_limit is not None and current_time >= opts.time_limit:
            hist["time"][-1] = opts.time_limit
            break

        if opts.time_limit is None and gn <= opts.tol_grad:
            break

        prev_grad_norm = gn

    return X, hist
