from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spd import (
    assemble_reduced,
    basis_from_candidates,
    full_tangent_basis,
    geometry_terms,
    hess_action,
    objective,
    random_tangent_metric_gaussian,
    random_tangent,
    retraction_exp,
    spd_sqrt_and_inv,
    sym,
    tangent_norm,
    transport_tangent_basis,
)
from .subproblem import solve_cubic_subproblem


@lru_cache(maxsize=None)
def _symvec_helpers(dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_idx, col_idx, scale) for symmetric vectorization under Frobenius."""
    row_idx, col_idx = np.triu_indices(dim)
    scale = np.ones_like(row_idx, dtype=float)
    scale[row_idx < col_idx] = np.sqrt(2.0)
    return row_idx, col_idx, scale


def _symvec_fro(mat: np.ndarray) -> np.ndarray:
    """Vectorize a symmetric matrix with sqrt(2) scaling on off-diagonals."""
    dim = mat.shape[0]
    row_idx, col_idx, scale = _symvec_helpers(dim)
    vec = mat[row_idx, col_idx] * scale
    return np.asarray(vec, dtype=float)


@lru_cache(maxsize=None)
def _frobenius_sym_basis(dim: int) -> np.ndarray:
    """Frobenius-orthonormal basis for symmetric dim×dim matrices.

    Ordering matches ``np.triu_indices(dim)`` / ``_symvec_fro``.
    """
    m = dim * (dim + 1) // 2
    basis = np.zeros((m, dim, dim), dtype=float)
    row_idx, col_idx = np.triu_indices(dim)
    for idx, (i, j) in enumerate(zip(row_idx, col_idx)):
        if i == j:
            basis[idx, i, j] = 1.0
        else:
            inv_sqrt2 = 1.0 / np.sqrt(2.0)
            basis[idx, i, j] = inv_sqrt2
            basis[idx, j, i] = inv_sqrt2
    return basis


def _op_norm_symmetric(mat: np.ndarray) -> float:
    mat = 0.5 * (mat + mat.T)
    w = np.linalg.eigvalsh(mat)
    return float(max(abs(float(w[0])), abs(float(w[-1]))))


def _gaussian_nystrom_upper_bound(eigvals_desc: np.ndarray, ell: int) -> float:
    """Return the Gaussian Nyström operator-norm upper bound from the corollary."""
    if ell < 4:
        return float("nan")
    eigvals_desc = np.asarray(eigvals_desc, dtype=float)
    if eigvals_desc.size == 0:
        return float("nan")

    best = float("inf")
    for p in range(2, ell - 1):
        k = ell - p
        if k < 0 or k >= eigvals_desc.size:
            continue
        lam = eigvals_desc[k]
        tail = float(np.sum(eigvals_desc[k:]))
        candidate = (1.0 + 2.0 * k / (p - 1)) * lam
        candidate += (2.0 * np.e**2 * ell / (p**2 - 1)) * tail
        best = min(best, float(candidate))
    return best if np.isfinite(best) else float("nan")


def _hess_nystrom_error_operator_norm(
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    geom_state,
    sketch_basis: List[np.ndarray],
    *,
    ridge: float = 0.0,
    dist_weight: float = 1.0,
    inv_linear_weight: float = 1.0,
    inv_quadratic_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """Return (abs_err, rel_err, gaussian_bound) for ||H - H_nys||_op at (A, X).

    The operator norm is induced by the affine-invariant metric. We compute it by
    representing operators in a g_X-orthonormal frame induced by the isometry
        U -> X^{-1/2} U X^{-1/2},
    so that the operator norm becomes the spectral norm of the symmetric matrix
    representation.
    """
    dim = X.shape[0]
    tangent_dim = dim * (dim + 1) // 2
    if tangent_dim == 0:
        return 0.0, 0.0, float("nan")
    if len(sketch_basis) == 0:
        return 0.0, 0.0, float("nan")

    X_sqrt, X_inv_sqrt = spd_sqrt_and_inv(X)
    fro_basis = _frobenius_sym_basis(dim)

    # Build the full Hessian matrix in the metric-orthonormal coordinates.
    H_full = np.empty((tangent_dim, tangent_dim), dtype=float)
    for j in range(tangent_dim):
        U = X_sqrt @ fro_basis[j] @ X_sqrt
        HU = hess_action(
            A,
            B,
            X,
            U,
            geom_state,
            dist_weight=dist_weight,
            inv_linear_weight=inv_linear_weight,
            inv_quadratic_weight=inv_quadratic_weight,
        )
        HU_tilde = X_inv_sqrt @ HU @ X_inv_sqrt
        H_full[:, j] = _symvec_fro(sym(HU_tilde))
    H_full = 0.5 * (H_full + H_full.T)

    # Sketch matrix P: columns are sketch vectors expressed in the same coordinates.
    ell = len(sketch_basis)
    P = np.empty((tangent_dim, ell), dtype=float)
    for j, Bj in enumerate(sketch_basis):
        Bj_tilde = X_inv_sqrt @ Bj @ X_inv_sqrt
        P[:, j] = _symvec_fro(sym(Bj_tilde))

    C = H_full @ P
    core = P.T @ C
    core = 0.5 * (core + core.T)
    if ridge > 0:
        core = core + ridge * np.eye(core.shape[0], dtype=float)

    core_pinv = np.linalg.pinv(core)
    H_nys = C @ core_pinv @ C.T
    H_nys = 0.5 * (H_nys + H_nys.T)

    err = H_full - H_nys
    abs_err = _op_norm_symmetric(err)
    denom = max(_op_norm_symmetric(H_full), 1e-16)
    rel_err = abs_err / denom
    eigvals_desc = np.linalg.eigvalsh(H_full)[::-1]
    eigvals_desc = np.maximum(eigvals_desc, 0.0)
    gaussian_bound = _gaussian_nystrom_upper_bound(eigvals_desc, ell)
    return abs_err, rel_err, gaussian_bound


@dataclass
class SolverOptions:
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
    sketch_size: int = 10
    include_gradient_in_sketch: bool = True
    sketch_distribution: str = "ambient_symmetric"
    transport_sketch: bool = False
    transport_refresh_every: int = 0
    transport_reset_on_reject: bool = False
    transport_disable_grad_tol: Optional[float] = None
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
    track_hess_nystrom_error: bool = False
    hess_error_every: int = 1
    hess_error_max_dim: Optional[int] = 400
    hess_nystrom_ridge: float = 0.0

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
        if self.transport_refresh_every < 0:
            raise ValueError("transport_refresh_every must be nonnegative.")
        if self.transport_disable_grad_tol is not None and self.transport_disable_grad_tol < 0.0:
            raise ValueError("transport_disable_grad_tol must be nonnegative when provided.")


def _make_spd(X: np.ndarray) -> np.ndarray:
    X = sym(X)
    w, Q = np.linalg.eigh(X)
    eps = 1e-8
    w = np.maximum(w, eps)
    return (Q * w) @ Q.T


def _build_sketch_basis(
    rng: np.random.Generator,
    X: np.ndarray,
    grad: np.ndarray,
    geom_state,
    ell: int,
    opts: SolverOptions,
    it: int,
    transport_active: bool,
    cached_basis: Optional[np.ndarray],
) -> np.ndarray:
    tangent_dim = X.shape[0] * (X.shape[0] + 1) // 2
    if ell >= tangent_dim:
        basis = full_tangent_basis(geom_state.X_inv, geom_state.distance_state.X_sqrt)
        return basis[:ell]

    should_refresh = (
        cached_basis is None
        or len(cached_basis) < ell
        or not transport_active
        or (opts.transport_refresh_every > 0 and ((it - 1) % opts.transport_refresh_every) == 0)
    )
    if not should_refresh:
        return np.asarray(cached_basis[:ell], dtype=float)

    candidate_count = max(ell - (1 if opts.include_gradient_in_sketch else 0), 0)
    sketch_grad = grad if opts.include_gradient_in_sketch else None
    if opts.sketch_distribution == "ambient_symmetric":
        candidates = [random_tangent(rng, X.shape[0]) for _ in range(candidate_count)]
    elif opts.sketch_distribution == "metric_gaussian":
        X_sqrt = geom_state.distance_state.X_sqrt
        candidates = [random_tangent_metric_gaussian(rng, X_sqrt) for _ in range(candidate_count)]
    else:
        raise ValueError(f"Unknown sketch_distribution={opts.sketch_distribution!r}")

    basis = basis_from_candidates(
        geom_state.X_inv,
        sketch_grad,
        candidates,
        ell,
        X_sqrt=geom_state.distance_state.X_sqrt,
        X_inv_sqrt=geom_state.distance_state.X_inv_sqrt,
    )
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
    return basis


def solve_spd_nystrom_cubic(
    A: np.ndarray,
    B: np.ndarray,
    X0: np.ndarray,
    opts: Optional[SolverOptions] = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if opts is None:
        opts = SolverOptions()

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
        "ell": [],
        "hess_nystrom_err_op": [],
        "hess_nystrom_err_rel": [],
        "hess_nystrom_gaussian_bound_op": [],
    }
    t0 = time.perf_counter()

    d = X.shape[0]
    tangent_dim = d * (d + 1) // 2
    curr_ell = max(1, min(opts.sketch_size, tangent_dim))
    stagnation = 0
    prev_grad_norm = gn
    cached_basis: Optional[np.ndarray] = None

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
        transport_active = opts.transport_sketch and (
            opts.transport_disable_grad_tol is None or gn > opts.transport_disable_grad_tol
        )
        if not transport_active:
            cached_basis = None
        basis = _build_sketch_basis(rng, X, grad, geom_state, ell, opts, it, transport_active, cached_basis)
        ell = len(basis)

        hS, HS = assemble_reduced(
            A,
            B,
            X,
            basis,
            geom_state=geom_state,
            **objective_kwargs,
        )
        if opts.track_hess_nystrom_error:
            should_compute = True
            if opts.hess_error_every > 1 and ((it - 1) % opts.hess_error_every) != 0:
                should_compute = False
            if opts.hess_error_max_dim is not None and tangent_dim > opts.hess_error_max_dim:
                should_compute = False
            if should_compute:
                abs_err, rel_err, gaussian_bound = _hess_nystrom_error_operator_norm(
                    A,
                    B,
                    X,
                    geom_state,
                    basis,
                    ridge=float(opts.hess_nystrom_ridge),
                    **objective_kwargs,
                )
            else:
                abs_err, rel_err, gaussian_bound = float("nan"), float("nan"), float("nan")
            hist["hess_nystrom_err_op"].append(float(abs_err))
            hist["hess_nystrom_err_rel"].append(float(rel_err))
            hist["hess_nystrom_gaussian_bound_op"].append(float(gaussian_bound))
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

        prev_grad_norm = gn
        X_prev = X
        if accepted:
            X = _make_spd(X_trial)
            geom_state, f = geometry_terms(A, B, X, **objective_kwargs)
            grad = geom_state.grad_riemannian
            gn = tangent_norm(geom_state.X_inv, grad)
            next_transport_active = opts.transport_sketch and (
                opts.transport_disable_grad_tol is None or gn > opts.transport_disable_grad_tol
            )
            if next_transport_active and ell < tangent_dim:
                cached_basis = transport_tangent_basis(
                    X_prev,
                    X,
                    basis,
                    X_new_sqrt=geom_state.distance_state.X_sqrt,
                    X_new_inv_sqrt=geom_state.distance_state.X_inv_sqrt,
                    ell=ell,
                )
            else:
                cached_basis = None
        elif transport_active and ell < tangent_dim:
            cached_basis = None if opts.transport_reset_on_reject else np.asarray(basis, dtype=float)
        else:
            cached_basis = None
        if sigma > 0:
            if np.isfinite(rho) and rho >= opts.eta2:
                sigma = max(opts.sigma_min, sigma * opts.gamma1)
            elif (not np.isfinite(rho)) or rho < opts.eta1:
                sigma = sigma * opts.gamma2
                if opts.sigma_max is not None:
                    sigma = min(opts.sigma_max, sigma)

        hist["f"].append(f)
        hist["grad_norm"].append(gn)
        hist["sigma"].append(sigma)
        hist["rho"].append(float(rho))
        current_time = time.perf_counter() - t0
        hist["time"].append(current_time)
        hist["accepted"].append(1.0 if accepted else 0.0)
        hist["subprob_grad_norm"].append(subgrad_norm)
        hist["v_norm_ratio"].append(v_norm_ratio)
        hist["v_norm"].append(v_norm)
        hist["eta_norm"].append(eta_norm)
        hist["X_norm"].append(float(np.linalg.norm(X, ord="fro")))
        hist["ell"].append(float(ell))

        if opts.time_limit is not None and current_time >= opts.time_limit:
            hist["time"][-1] = opts.time_limit
            break

        if opts.display_progress:
            msg = (
                f"[spd-nystrom-cubic] iter {it}/{opts.max_iter} f={f:.6e} "
                f"||grad||={gn:.3e} sigma={sigma:.2e} ell={ell} accepted={int(accepted)}"
            )
            print(msg, flush=True)

        pred_tiny = abs(pred) <= opts.tol_pred * max(1.0, abs(f))
        step_tiny = eta_norm <= opts.tol_step * max(1.0, hist["X_norm"][-1])
        if gn <= opts.tol_grad or pred_tiny or step_tiny:
            break

    return X, hist
