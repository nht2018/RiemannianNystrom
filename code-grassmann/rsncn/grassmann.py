"""Minimal Grassmann geometry for dominant-eigenspace experiments."""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np


def proj_Y(Y: np.ndarray, G: np.ndarray) -> np.ndarray:
    return G - Y @ (Y.T @ G)


def _proj_Y_batch(Y: np.ndarray, G: np.ndarray) -> np.ndarray:
    if G.ndim != 3:
        raise ValueError("G must have shape (ell, n, p)")
    YT_G = np.matmul(Y.T[None, :, :], G)
    return G - np.matmul(Y[None, :, :], YT_G)


def retraction_qr(Y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(Y + eta, mode="reduced")
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q @ np.diag(s)


def tangent_norm(U: np.ndarray) -> float:
    return float(np.linalg.norm(U, ord="fro"))


def objective(a_diag: np.ndarray, Y: np.ndarray) -> float:
    return -float(np.sum(a_diag[:, None] * (Y * Y)))


@dataclass
class GeometryState:
    AY: np.ndarray
    S: np.ndarray
    grad: np.ndarray
    grad_euclidean: np.ndarray


def geometry_terms(a_diag: np.ndarray, Y: np.ndarray) -> Tuple[GeometryState, float]:
    AY = a_diag[:, None] * Y
    S = Y.T @ AY
    grad_e = -2.0 * AY
    grad = proj_Y(Y, grad_e)
    fval = objective(a_diag, Y)
    return GeometryState(AY=AY, S=S, grad=grad, grad_euclidean=grad_e), fval


def hess_action_batch(
    a_diag: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    *,
    S: np.ndarray,
) -> np.ndarray:
    AU = a_diag[None, :, None] * U
    return _proj_Y_batch(Y, -2.0 * AU + 2.0 * np.matmul(U, S))


def random_tangent_metric_gaussian(
    rng: np.random.Generator,
    Y: np.ndarray,
    ell: int,
) -> np.ndarray:
    G = rng.standard_normal((ell, *Y.shape))
    return _proj_Y_batch(Y, G)


def _as_batch(mats: Union[np.ndarray, Iterable[np.ndarray]], n: int, p: int) -> np.ndarray:
    if isinstance(mats, np.ndarray):
        mats_arr = np.asarray(mats, dtype=float)
    else:
        mats_arr = np.asarray(list(mats), dtype=float)
    if mats_arr.size == 0:
        return np.zeros((0, n, p), dtype=float)
    if mats_arr.ndim == 2:
        mats_arr = mats_arr[None, ...]
    return mats_arr


def _orthonormalize_flat(flat: np.ndarray, ell: Optional[int] = None, tol: float = 1e-12) -> np.ndarray:
    if flat.size == 0:
        return flat.reshape(0, flat.shape[-1] if flat.ndim == 2 else 0)
    gram = flat @ flat.T
    gram = 0.5 * (gram + gram.T)
    evals, evecs = np.linalg.eigh(gram)
    keep = evals > max(tol, tol * float(np.max(evals)))
    if not np.any(keep):
        return np.zeros((0, flat.shape[1]), dtype=float)
    evals_keep = evals[keep]
    evecs_keep = evecs[:, keep]
    q_flat = ((evecs_keep / np.sqrt(evals_keep)).T @ flat)
    if ell is not None:
        q_flat = q_flat[: min(int(ell), q_flat.shape[0])]
    return q_flat


def _reorthonormalize_flat_near_identity(
    flat: np.ndarray,
    *,
    ell: Optional[int] = None,
    tol: float = 1e-12,
) -> np.ndarray:
    if flat.size == 0:
        return flat.reshape(0, flat.shape[-1] if flat.ndim == 2 else 0)
    gram = flat @ flat.T
    gram = 0.5 * (gram + gram.T)
    try:
        chol = np.linalg.cholesky(gram)
        q_flat = np.linalg.solve(chol, flat)
    except np.linalg.LinAlgError:
        q_flat = _orthonormalize_flat(flat, ell=ell, tol=tol)
    else:
        if ell is not None:
            q_flat = q_flat[: min(int(ell), q_flat.shape[0])]
    return q_flat


def orthonormalize_tangent_candidates(
    Y: np.ndarray,
    candidates: Union[np.ndarray, Iterable[np.ndarray]],
    ell: Optional[int] = None,
    *,
    first_direction: Optional[np.ndarray] = None,
) -> np.ndarray:
    n, p = Y.shape
    cand_arr = _as_batch(candidates, n, p)
    tangent_candidates = _proj_Y_batch(Y, cand_arr)
    q_blocks = []
    remaining = None if ell is None else int(ell)

    q_first_flat = None
    if first_direction is not None:
        first = proj_Y(Y, np.asarray(first_direction, dtype=float))
        first_flat = first.reshape(-1)
        first_norm = float(np.linalg.norm(first_flat))
        if first_norm > 1e-12:
            q_first_flat = first_flat / first_norm
            q_blocks.append(q_first_flat.reshape(n, p))
            if remaining is not None:
                remaining = max(0, remaining - 1)

    if tangent_candidates.size == 0 or remaining == 0:
        if q_blocks:
            return np.stack(q_blocks, axis=0)
        return np.zeros((0, n, p), dtype=float)

    flat = tangent_candidates.reshape(tangent_candidates.shape[0], -1)
    if q_first_flat is not None:
        coeff = flat @ q_first_flat
        flat = flat - coeff[:, None] * q_first_flat[None, :]

    q_rest_flat = _orthonormalize_flat(flat, remaining)
    if q_rest_flat.size:
        q_blocks.extend(q_rest_flat.reshape(q_rest_flat.shape[0], n, p))

    if q_blocks:
        return np.stack(q_blocks, axis=0)
    return np.zeros((0, n, p), dtype=float)


def gram_schmidt_tangent(Y: np.ndarray, mats: Iterable[np.ndarray], ell: Optional[int] = None) -> np.ndarray:
    return orthonormalize_tangent_candidates(Y, mats, ell=ell)


def basis_from_candidates(Y: np.ndarray, grad: Optional[np.ndarray], candidates: np.ndarray, ell: int) -> np.ndarray:
    return orthonormalize_tangent_candidates(Y, candidates, ell=ell, first_direction=grad)


def _append_orthonormal_block(
    q_blocks: list[np.ndarray],
    flat: np.ndarray,
    *,
    ell_remaining: Optional[int],
    use_fast_reorth: bool,
) -> tuple[list[np.ndarray], Optional[int]]:
    if flat.size == 0 or ell_remaining == 0:
        return q_blocks, ell_remaining
    if q_blocks:
        existing = np.stack(q_blocks, axis=0)
        coeff = flat @ existing.T
        flat = flat - coeff @ existing
    q_new = (
        _reorthonormalize_flat_near_identity(flat, ell=ell_remaining)
        if use_fast_reorth
        else _orthonormalize_flat(flat, ell=ell_remaining)
    )
    if q_new.size:
        q_blocks.extend(q_new)
        if ell_remaining is not None:
            ell_remaining = max(0, ell_remaining - q_new.shape[0])
    return q_blocks, ell_remaining


def build_transport_sketch_basis(
    Y: np.ndarray,
    *,
    ell: int,
    grad: Optional[np.ndarray],
    transported: Optional[np.ndarray] = None,
    fresh_candidates: Optional[np.ndarray] = None,
    skip_transport_reorth: bool = False,
) -> np.ndarray:
    n, p = Y.shape
    q_blocks: list[np.ndarray] = []
    remaining: Optional[int] = int(ell)

    if grad is not None:
        grad_tan = proj_Y(Y, np.asarray(grad, dtype=float))
        grad_flat = grad_tan.reshape(-1)
        grad_norm = float(np.linalg.norm(grad_flat))
        if grad_norm > 1e-12:
            q_blocks.append(grad_flat / grad_norm)
            remaining = max(0, remaining - 1)

    trans_arr = _as_batch(transported if transported is not None else np.zeros((0, n, p)), n, p)
    if trans_arr.size and remaining != 0:
        trans_flat = trans_arr.reshape(trans_arr.shape[0], -1)
        if skip_transport_reorth:
            if q_blocks:
                existing = np.stack(q_blocks, axis=0)
                coeff = trans_flat @ existing.T
                trans_flat = trans_flat - coeff @ existing
            if remaining is not None:
                trans_flat = trans_flat[:remaining]
            if trans_flat.size:
                q_blocks.extend(trans_flat)
                if remaining is not None:
                    remaining = max(0, remaining - trans_flat.shape[0])
        else:
            q_blocks, remaining = _append_orthonormal_block(
                q_blocks,
                trans_flat,
                ell_remaining=remaining,
                use_fast_reorth=True,
            )

    fresh_arr = _as_batch(fresh_candidates if fresh_candidates is not None else np.zeros((0, n, p)), n, p)
    if fresh_arr.size and remaining != 0:
        fresh_flat = _proj_Y_batch(Y, fresh_arr).reshape(fresh_arr.shape[0], -1)
        q_blocks, remaining = _append_orthonormal_block(
            q_blocks,
            fresh_flat,
            ell_remaining=remaining,
            use_fast_reorth=False,
        )

    if not q_blocks:
        return np.zeros((0, n, p), dtype=float)
    basis_flat = np.stack(q_blocks, axis=0)
    return basis_flat.reshape(basis_flat.shape[0], n, p)


def transport_tangent_basis(
    Y_new: np.ndarray,
    basis: np.ndarray,
    ell: Optional[int] = None,
    *,
    return_orth_error: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    basis_arr = np.asarray(basis, dtype=float)
    if basis_arr.size == 0:
        n, p = Y_new.shape
        empty = np.zeros((0, n, p), dtype=float)
        return (empty, 0.0) if return_orth_error else empty
    if basis_arr.ndim == 2:
        basis_arr = basis_arr[None, ...]
    projected = _proj_Y_batch(Y_new, basis_arr)
    flat = projected.reshape(projected.shape[0], -1)
    gram = flat @ flat.T
    gram = 0.5 * (gram + gram.T)
    orth_err = float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord=2))
    q_flat = _reorthonormalize_flat_near_identity(flat, ell=ell)
    q = q_flat.reshape(q_flat.shape[0], *Y_new.shape)
    if return_orth_error:
        return q, orth_err
    return q
