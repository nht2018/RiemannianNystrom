"""Geometry helpers for the affine-invariant metric on the SPD manifold."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import inv
from scipy.linalg import eigh, expm


def sym(M: np.ndarray) -> np.ndarray:
    """Return the symmetric part of ``M``."""
    return 0.5 * (M + M.T)


def spd_sqrt_and_inv(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the matrix square root and its inverse for SPD ``X``."""
    w, Q = eigh(X)
    eps = np.finfo(float).eps
    w = np.maximum(w, eps)
    sqrt_w = np.sqrt(w)
    inv_sqrt_w = 1.0 / sqrt_w
    X_sqrt = (Q * sqrt_w) @ Q.T
    X_inv_sqrt = (Q * inv_sqrt_w) @ Q.T
    return X_sqrt, X_inv_sqrt


def spd_logm(X: np.ndarray) -> np.ndarray:
    """Return the matrix logarithm for an SPD matrix."""
    w, Q = eigh(sym(X))
    eps = np.finfo(float).eps
    w = np.maximum(w, eps)
    return sym((Q * np.log(w)) @ Q.T)


def log_frechet_divided_differences(eigvals: np.ndarray) -> np.ndarray:
    """Return the divided-difference matrix for the SPD matrix logarithm."""
    eigvals = np.asarray(eigvals, dtype=float)
    log_eigvals = np.log(eigvals)
    eigval_diff = eigvals[:, None] - eigvals[None, :]
    log_diff = log_eigvals[:, None] - log_eigvals[None, :]
    near_equal = np.isclose(eigval_diff, 0.0, rtol=1e-12, atol=1e-15)
    safe_diff = np.where(near_equal, 1.0, eigval_diff)
    divdiff = log_diff / safe_diff
    divdiff[near_equal] = 2.0 / (eigvals[:, None] + eigvals[None, :])[near_equal]
    diag_idx = np.diag_indices_from(divdiff)
    divdiff[diag_idx] = 1.0 / eigvals
    return divdiff


def log_frechet_spd(
    eigvecs: np.ndarray,
    log_divdiff: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Return the exact Fréchet derivative of ``log`` on an SPD matrix."""
    rotated = eigvecs.T @ sym(H) @ eigvecs
    return sym(eigvecs @ (log_divdiff * rotated) @ eigvecs.T)


def retraction_exp(X: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Affine-invariant exponential-map retraction."""
    X_sqrt, X_inv_sqrt = spd_sqrt_and_inv(X)
    H = X_inv_sqrt @ sym(eta) @ X_inv_sqrt
    Y = X_sqrt @ expm(H) @ X_sqrt
    return sym(Y)


def inner_product(X_inv: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    """Affine-invariant inner product at ``X``."""
    return float(np.trace((X_inv @ U) @ (X_inv @ V)))


def tangent_norm(X_inv: np.ndarray, U: np.ndarray) -> float:
    """Affine-invariant norm of a tangent vector at ``X``."""
    return float(np.sqrt(max(inner_product(X_inv, U, U), 0.0)))


@lru_cache(maxsize=None)
def _frobenius_sym_basis(dim: int) -> np.ndarray:
    """Cached Frobenius-orthonormal basis for symmetric dim×dim matrices."""
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


@lru_cache(maxsize=None)
def _symmetric_gaussian_indices(dim: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    return np.diag_indices(dim), np.triu_indices(dim, 1)


def _metric_sqrt_pair_from_inverse(X_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_sqrt, X_inv_sqrt) from X^{-1} without reconstructing X."""
    w, Q = eigh(sym(X_inv))
    eps = np.finfo(float).eps
    w = np.maximum(w, eps)
    sqrt_w = np.sqrt(w)
    inv_sqrt_w = 1.0 / sqrt_w
    X_inv_sqrt = (Q * sqrt_w) @ Q.T
    X_sqrt = (Q * inv_sqrt_w) @ Q.T
    return sym(X_sqrt), sym(X_inv_sqrt)


def normalize_under_metric(X_inv: np.ndarray, U: np.ndarray, tol: float = 1e-12) -> Optional[np.ndarray]:
    """Return the normalized version of ``U`` under the affine-invariant metric."""
    v = sym(U)
    nrm_sq = inner_product(X_inv, v, v)
    if nrm_sq <= tol:
        return None
    return v / np.sqrt(nrm_sq)


def gram_schmidt_metric(
    X_inv: np.ndarray,
    mats: Iterable[np.ndarray],
    ell: Optional[int] = None,
    *,
    X_sqrt: Optional[np.ndarray] = None,
    X_inv_sqrt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Orthonormalize the provided symmetric directions under the affine-invariant metric."""
    if X_sqrt is None or X_inv_sqrt is None:
        X_sqrt, X_inv_sqrt = _metric_sqrt_pair_from_inverse(X_inv)

    if ell is not None:
        d = X_inv.shape[0]
        basis_tilde = np.empty((ell, d, d), dtype=float)
        basis = np.empty((ell, d, d), dtype=float)
        count = 0
        for M in mats:
            vcand_tilde = sym(X_inv_sqrt @ sym(M) @ X_inv_sqrt)
            if count:
                coeffs = np.tensordot(
                    basis_tilde[:count],
                    vcand_tilde,
                    axes=([1, 2], [0, 1]),
                )
                vcand_tilde = vcand_tilde - np.tensordot(coeffs, basis_tilde[:count], axes=(0, 0))
            nrm_sq = float(np.tensordot(vcand_tilde, vcand_tilde, axes=([0, 1], [0, 1])))
            if nrm_sq <= 1e-12:
                continue
            vcand_tilde /= np.sqrt(nrm_sq)
            basis_tilde[count] = vcand_tilde
            candidate = X_sqrt @ vcand_tilde @ X_sqrt
            basis[count] = sym(candidate)
            count += 1
            if count >= ell:
                break
        return basis[:count].copy()

    basis_list: List[np.ndarray] = []
    basis_tilde_list: List[np.ndarray] = []
    for M in mats:
        vcand_tilde = sym(X_inv_sqrt @ sym(M) @ X_inv_sqrt)
        if basis_tilde_list:
            stacked = np.asarray(basis_tilde_list, dtype=float)
            coeffs = np.tensordot(stacked, vcand_tilde, axes=([1, 2], [0, 1]))
            vcand_tilde = vcand_tilde - np.tensordot(coeffs, stacked, axes=(0, 0))
        nrm_sq = float(np.tensordot(vcand_tilde, vcand_tilde, axes=([0, 1], [0, 1])))
        if nrm_sq <= 1e-12:
            continue
        vcand_tilde /= np.sqrt(nrm_sq)
        basis_tilde_list.append(vcand_tilde)
        basis_list.append(sym(X_sqrt @ vcand_tilde @ X_sqrt))
    if not basis_list:
        d = X_inv.shape[0]
        return np.zeros((0, d, d), dtype=float)
    return np.asarray(basis_list, dtype=float)


def basis_from_candidates(
    X_inv: np.ndarray,
    grad: Optional[np.ndarray],
    candidates: Iterable[np.ndarray],
    ell: int,
    *,
    X_sqrt: Optional[np.ndarray] = None,
    X_inv_sqrt: Optional[np.ndarray] = None,
) -> np.ndarray:
    ordered: List[np.ndarray] = []
    if grad is not None:
        ordered.append(sym(grad))
    ordered.extend(candidates)
    return gram_schmidt_metric(X_inv, ordered, ell, X_sqrt=X_sqrt, X_inv_sqrt=X_inv_sqrt)


@dataclass
class DistanceState:
    X_sqrt: np.ndarray
    X_inv_sqrt: np.ndarray
    X_eigvecs: np.ndarray
    X_sqrt_eigvals: np.ndarray
    X_inv_sqrt_eigvals: np.ndarray
    log_whitened: np.ndarray
    log_map: np.ndarray
    whitened_eigvecs: np.ndarray
    log_divdiff: np.ndarray
    dist_sq: float


@dataclass
class GeometryState:
    grad_riemannian: np.ndarray
    X_inv: np.ndarray
    B_metric: np.ndarray
    BX_inv: np.ndarray
    X_inv_B: np.ndarray
    GX_inv: np.ndarray
    X_inv_G: np.ndarray
    trace_B_X_inv: float
    distance_state: DistanceState


def _distance_state(X: np.ndarray, A: np.ndarray) -> DistanceState:
    x_eigvals, x_eigvecs = eigh(X)
    eps = np.finfo(float).eps
    x_eigvals = np.maximum(x_eigvals, eps)
    x_sqrt_eigvals = np.sqrt(x_eigvals)
    x_inv_sqrt_eigvals = 1.0 / x_sqrt_eigvals
    X_sqrt = (x_eigvecs * x_sqrt_eigvals) @ x_eigvecs.T
    X_inv_sqrt = (x_eigvecs * x_inv_sqrt_eigvals) @ x_eigvecs.T
    whitened = sym(X_inv_sqrt @ A @ X_inv_sqrt)
    eigvals, eigvecs = eigh(whitened)
    eigvals = np.maximum(eigvals, eps)
    log_whitened = sym((eigvecs * np.log(eigvals)) @ eigvecs.T)
    log_map = sym(X_sqrt @ log_whitened @ X_sqrt)
    dist_sq = float(np.linalg.norm(log_whitened, ord="fro") ** 2)
    log_divdiff = log_frechet_divided_differences(eigvals)
    return DistanceState(
        X_sqrt=X_sqrt,
        X_inv_sqrt=X_inv_sqrt,
        X_eigvecs=x_eigvecs,
        X_sqrt_eigvals=x_sqrt_eigvals,
        X_inv_sqrt_eigvals=x_inv_sqrt_eigvals,
        log_whitened=log_whitened,
        log_map=log_map,
        whitened_eigvecs=eigvecs,
        log_divdiff=log_divdiff,
        dist_sq=dist_sq,
    )


def _objective_and_grad(
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    *,
    dist_weight: float,
    inv_linear_weight: float,
    inv_quadratic_weight: float,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    X_inv = inv(X)
    dist_state = _distance_state(X, A)
    trace_B_X_inv = float(np.trace(B @ X_inv))
    grad_r = -2.0 * dist_weight * dist_state.log_map
    grad_r = grad_r - (inv_linear_weight + 2.0 * inv_quadratic_weight * trace_B_X_inv) * B
    fval = dist_weight * dist_state.dist_sq
    fval += inv_linear_weight * trace_B_X_inv
    fval += inv_quadratic_weight * (trace_B_X_inv ** 2)
    return float(fval), sym(grad_r), X_inv, trace_B_X_inv


def geometry_terms(
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    *,
    dist_weight: float = 1.0,
    inv_linear_weight: float = 1.0,
    inv_quadratic_weight: float = 1.0,
) -> Tuple[GeometryState, float]:
    dist_state = _distance_state(X, A)
    X_inv = inv(X)
    trace_B_X_inv = float(np.trace(B @ X_inv))
    grad_r = -2.0 * dist_weight * dist_state.log_map
    grad_r = grad_r - (inv_linear_weight + 2.0 * inv_quadratic_weight * trace_B_X_inv) * B
    grad_r = sym(grad_r)
    B_metric = sym(dist_state.X_inv_sqrt @ B @ dist_state.X_inv_sqrt)
    BX_inv = B @ X_inv
    X_inv_B = X_inv @ B
    GX_inv = dist_state.log_map @ X_inv
    X_inv_G = X_inv @ dist_state.log_map
    fval = dist_weight * dist_state.dist_sq
    fval += inv_linear_weight * trace_B_X_inv
    fval += inv_quadratic_weight * (trace_B_X_inv ** 2)
    return GeometryState(
        grad_r,
        X_inv,
        B_metric,
        BX_inv,
        X_inv_B,
        GX_inv,
        X_inv_G,
        trace_B_X_inv,
        dist_state,
    ), float(fval)


def _symmetrize_batch(mats: np.ndarray) -> np.ndarray:
    return 0.5 * (mats + mats.transpose(0, 2, 1))


def _as_matrix_batch(mats: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(mats, dtype=float)
    is_single = arr.ndim == 2
    if is_single:
        arr = arr[None, ...]
    return arr, is_single


def _log_frechet_spd_batch(
    eigvecs: np.ndarray,
    log_divdiff: np.ndarray,
    H_batch: np.ndarray,
) -> np.ndarray:
    H_arr, is_single = _as_matrix_batch(H_batch)
    rotated = np.einsum("ba,kbc,cd->kad", eigvecs, _symmetrize_batch(H_arr), eigvecs, optimize=True)
    result = np.einsum(
        "ab,kbc,dc->kad",
        eigvecs,
        log_divdiff[None, :, :] * rotated,
        eigvecs,
        optimize=True,
    )
    result = _symmetrize_batch(result)
    return result[0] if is_single else result


def _solve_sylvester_spd_batch(state: DistanceState, U_batch: np.ndarray) -> np.ndarray:
    U_arr, is_single = _as_matrix_batch(U_batch)
    eigvecs = state.X_eigvecs
    rotated = np.einsum("ba,kbc,cd->kad", eigvecs, _symmetrize_batch(U_arr), eigvecs, optimize=True)
    denom = state.X_sqrt_eigvals[:, None] + state.X_sqrt_eigvals[None, :]
    solved = rotated / denom[None, :, :]
    result = np.einsum("ab,kbc,dc->kad", eigvecs, solved, eigvecs, optimize=True)
    result = _symmetrize_batch(result)
    return result[0] if is_single else result


def hess_action_batch(
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    U_batch: np.ndarray,
    state: GeometryState,
    *,
    dist_weight: float = 1.0,
    inv_linear_weight: float = 1.0,
    inv_quadratic_weight: float = 1.0,
) -> np.ndarray:
    U_arr, is_single = _as_matrix_batch(U_batch)
    U_arr = _symmetrize_batch(U_arr)
    X_inv = state.X_inv
    dist_state = state.distance_state

    R = dist_state.X_sqrt
    S = dist_state.X_inv_sqrt
    G = dist_state.log_map
    L = dist_state.log_whitened

    d_X_sqrt = _solve_sylvester_spd_batch(dist_state, U_arr)
    d_X_inv_sqrt = -np.einsum(
        "ab,kbc,cd->kad",
        S,
        d_X_sqrt,
        S,
        optimize=True,
    )
    d_whitened = _symmetrize_batch(
        np.einsum("kab,bc,cd->kad", d_X_inv_sqrt, A, S, optimize=True)
        + np.einsum("ab,bc,kcd->kad", S, A, d_X_inv_sqrt, optimize=True)
    )
    d_log_whitened = _log_frechet_spd_batch(
        dist_state.whitened_eigvecs,
        dist_state.log_divdiff,
        d_whitened,
    )
    d_log_map = _symmetrize_batch(
        np.einsum("kab,bc,cd->kad", d_X_sqrt, L, R, optimize=True)
        + np.einsum("ab,kbc,cd->kad", R, d_log_whitened, R, optimize=True)
        + np.einsum("ab,bc,kcd->kad", R, L, d_X_sqrt, optimize=True)
    )
    correction = _symmetrize_batch(
        np.einsum("kab,bc->kac", U_arr, state.X_inv_G, optimize=True)
        + np.einsum("ab,kbc->kac", state.GX_inv, U_arr, optimize=True)
    )
    hess_dist = -2.0 * dist_weight * (d_log_map - 0.5 * correction)

    trace_B_X_inv = state.trace_B_X_inv
    scale = inv_linear_weight + 2.0 * inv_quadratic_weight * trace_B_X_inv
    U_metric = np.einsum("ab,kbc,cd->kad", S, U_arr, S, optimize=True)
    metric_B_U = np.tensordot(U_metric, state.B_metric, axes=([1, 2], [0, 1]))
    hess_inv = 0.5 * scale * _symmetrize_batch(
        np.einsum("kab,bc->kac", U_arr, state.X_inv_B, optimize=True)
        + np.einsum("ab,kbc->kac", state.BX_inv, U_arr, optimize=True)
    )
    hess_inv = hess_inv + 2.0 * inv_quadratic_weight * metric_B_U[:, None, None] * B

    result = _symmetrize_batch(hess_dist + hess_inv)
    return result[0] if is_single else result


def hess_action(
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    state: GeometryState,
    *,
    dist_weight: float = 1.0,
    inv_linear_weight: float = 1.0,
    inv_quadratic_weight: float = 1.0,
) -> np.ndarray:
    return hess_action_batch(
        A,
        B,
        X,
        U,
        state,
        dist_weight=dist_weight,
        inv_linear_weight=inv_linear_weight,
        inv_quadratic_weight=inv_quadratic_weight,
    )


def objective(
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    *,
    dist_weight: float = 1.0,
    inv_linear_weight: float = 1.0,
    inv_quadratic_weight: float = 1.0,
) -> float:
    fval, _, _, _ = _objective_and_grad(
        A,
        B,
        X,
        dist_weight=dist_weight,
        inv_linear_weight=inv_linear_weight,
        inv_quadratic_weight=inv_quadratic_weight,
    )
    return fval


def random_tangent(rng: np.random.Generator, d: int) -> np.ndarray:
    M = rng.standard_normal((d, d))
    return sym(M)


def random_tangent_metric_gaussian(rng: np.random.Generator, X_sqrt: np.ndarray) -> np.ndarray:
    d = X_sqrt.shape[0]
    symmetric_gaussian = np.zeros((d, d), dtype=float)
    diag_idx, upper_idx = _symmetric_gaussian_indices(d)
    symmetric_gaussian[diag_idx] = rng.standard_normal(d)
    upper = rng.standard_normal(len(upper_idx[0])) / np.sqrt(2.0)
    symmetric_gaussian[upper_idx] = upper
    symmetric_gaussian[(upper_idx[1], upper_idx[0])] = upper
    return sym(X_sqrt @ symmetric_gaussian @ X_sqrt)


def transport_tangent_basis(
    X_old: np.ndarray,
    X_new: np.ndarray,
    basis: np.ndarray,
    *,
    X_new_sqrt: Optional[np.ndarray] = None,
    X_new_inv_sqrt: Optional[np.ndarray] = None,
    ell: Optional[int] = None,
) -> np.ndarray:
    """Parallel transport SPD tangent basis from ``X_old`` to ``X_new``.

    For the affine-invariant metric, parallel transport along the geodesic from
    ``X_old`` to ``X_new`` is ``U -> E U E^T`` with
    ``E = X_new^{1/2} (X_new^{-1/2} X_old X_new^{-1/2})^{-1/2} X_new^{-1/2}``.
    """
    basis_arr = np.asarray(basis, dtype=float)
    if basis_arr.size == 0:
        dim = X_old.shape[0]
        return np.zeros((0, dim, dim), dtype=float)
    if basis_arr.ndim == 2:
        basis_arr = basis_arr[None, ...]
    if ell is not None:
        basis_arr = basis_arr[:ell]
    if X_new_sqrt is None or X_new_inv_sqrt is None:
        X_new_sqrt, X_new_inv_sqrt = spd_sqrt_and_inv(X_new)
    bridge = sym(X_new_inv_sqrt @ X_old @ X_new_inv_sqrt)
    _, bridge_inv_sqrt = spd_sqrt_and_inv(bridge)
    E = X_new_sqrt @ bridge_inv_sqrt @ X_new_inv_sqrt
    transported = np.einsum("ab,kbc,dc->kad", E, basis_arr, E, optimize=True)
    return _symmetrize_batch(transported)


def full_tangent_basis_pair(
    X_inv: np.ndarray,
    X_sqrt: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    d = X_inv.shape[0]
    if X_sqrt is None:
        X_sqrt, _ = _metric_sqrt_pair_from_inverse(X_inv)
    canonical = _frobenius_sym_basis(d)
    basis = np.einsum("ab,ibc,cd->iad", X_sqrt, canonical, X_sqrt, optimize=True)
    return _symmetrize_batch(basis), canonical


def full_tangent_basis(X_inv: np.ndarray, X_sqrt: Optional[np.ndarray] = None) -> np.ndarray:
    basis, _ = full_tangent_basis_pair(X_inv, X_sqrt)
    return basis


def assemble_reduced(
    A: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    basis: Sequence[np.ndarray],
    basis_tilde: Optional[np.ndarray] = None,
    *,
    geom_state: Optional[GeometryState] = None,
    dist_weight: float = 1.0,
    inv_linear_weight: float = 1.0,
    inv_quadratic_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if geom_state is None:
        geom_state, _ = geometry_terms(
            A,
            B,
            X,
            dist_weight=dist_weight,
            inv_linear_weight=inv_linear_weight,
            inv_quadratic_weight=inv_quadratic_weight,
        )

    X_inv = geom_state.X_inv
    X_inv_sqrt = geom_state.distance_state.X_inv_sqrt
    grad = geom_state.grad_riemannian
    basis_arr = np.asarray(basis, dtype=float)
    if basis_arr.ndim == 2:
        basis_arr = basis_arr[None, ...]
    ell = basis_arr.shape[0]
    if ell == 0:
        return np.zeros(0), np.zeros((0, 0))

    grad_tilde = sym(X_inv_sqrt @ grad @ X_inv_sqrt)
    if basis_tilde is None:
        basis_tilde_arr = np.einsum("ab,ibc,cd->iad", X_inv_sqrt, basis_arr, X_inv_sqrt, optimize=True)
    else:
        basis_tilde_arr = np.asarray(basis_tilde, dtype=float)
        if basis_tilde_arr.ndim == 2:
            basis_tilde_arr = basis_tilde_arr[None, ...]
    hS = np.tensordot(basis_tilde_arr, grad_tilde, axes=([1, 2], [0, 1])).astype(float, copy=False)

    H_batch = hess_action_batch(
        A,
        B,
        X,
        basis_arr,
        geom_state,
        dist_weight=dist_weight,
        inv_linear_weight=inv_linear_weight,
        inv_quadratic_weight=inv_quadratic_weight,
    )
    H_batch_tilde = np.einsum("ab,kbc,cd->kad", X_inv_sqrt, H_batch, X_inv_sqrt, optimize=True)
    HS = np.tensordot(basis_tilde_arr, H_batch_tilde, axes=([1, 2], [1, 2]))
    HS = 0.5 * (HS + HS.T)
    return hS, HS
