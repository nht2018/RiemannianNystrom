"""Explicit CURRENT reduced-Hessian cores for the diagonal Grassmann problem.

These functions return the symmetric ell-by-ell CORE, never a tangent Hessian
action. In particular, unprojected W must not be exposed as a public HVP.
All current-point quantities are recomputed; no past Hessian is reused.

Write W_i=-2 A B_i+2 B_i S, E_i=Y.T B_i, F_i=Y.T W_i. For arbitrary B,

    sym(<B_i,P_Y W_j>) = sym(<B_i,W_j> - <E_i,F_j>).

For positive diagonal A and L L.T=sym(S), the first term is

    -2 Gram(sqrt(A) B) + 2 Gram(B L).

The factor is B@L, NOT B@L.T. Using sym(S) is valid because the skew part
contributes only a skew core, including in the small projection correction.
Indefinite reduced cores are preserved; no eigenvalue clipping is performed.
Nonpositive a or failed Cholesky falls back to the generic contraction route.

The algebraic identities do not imply bitwise equality. Gram differences and
the projection correction may cancel. A conservative sqrt(eps) relative
cancellation guard falls back to the ORIGINAL projected-HVP core; error
checks should use an absolute term-scale floor, not relative core error alone
when the true core is nearly zero. These guards are not convergence claims.

Optional ``info`` counters distinguish ``exact_current_core_columns`` (ell
for each returned core) from ``projected_hvp_columns_materialized`` and
``unprojected_action_columns_materialized`` (actual full arrays, including
fallback work). ``gram_factor_columns_processed`` counts two sets of ell
weighted basis columns, not Hessian-vector products.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg.blas import dsyrk

from rsncn import grassmann as original
from geometry_isometric_fused import GeometryHooks as BufferedGeometry


_CANCELLATION_RTOL = np.sqrt(np.finfo(float).eps)


def _inputs(a_diag, Y, B, S):
    a_diag = np.asarray(a_diag, dtype=float)
    Y = np.asarray(Y, dtype=float)
    B = np.asarray(B, dtype=float)
    S = np.asarray(S, dtype=float)
    if Y.ndim != 2 or B.ndim != 3:
        raise ValueError("Y and B must have shapes (n,p) and (ell,n,p)")
    n, p = Y.shape
    if a_diag.shape != (n,) or B.shape[1:] != (n, p) or S.shape != (p, p):
        raise ValueError("Inconsistent diagonal, point, frame or S shape")
    return a_diag, Y, B, S


def _count(info, name, amount=1):
    if info is not None:
        info[name] = info.get(name, 0) + amount


def _symmetric_finite(matrix):
    result = np.ascontiguousarray(0.5 * (matrix + matrix.T))
    if not np.all(np.isfinite(result)):
        raise ValueError("Reduced Hessian core is nonfinite")
    return result


def core_original(a_diag, Y, B, *, S, info=None):
    """Oracle: upstream projected HVP followed by the original full GEMM."""
    a_diag, Y, B, S = _inputs(a_diag, Y, B, S)
    H = original.hess_action_batch(a_diag, Y, B, S=S)
    _count(info, "projected_hvp_columns_materialized", len(B))
    flat = B.reshape(len(B), Y.size)
    _count(info, "original")
    result = _symmetric_finite(flat @ H.reshape(len(B), Y.size).T)
    _count(info, "exact_current_core_columns", len(B))
    return result


def core_buffered(a_diag, Y, B, *, S, info=None):
    """Oracle: buffered CURRENT projected HVP, still followed by full GEMM."""
    a_diag, Y, B, S = _inputs(a_diag, Y, B, S)
    H = BufferedGeometry.hess_action_batch(a_diag, Y, B, S=S)
    _count(info, "projected_hvp_columns_materialized", len(B))
    flat = B.reshape(len(B), Y.size)
    _count(info, "buffered")
    result = _symmetric_finite(flat @ H.reshape(len(B), Y.size).T)
    _count(info, "exact_current_core_columns", len(B))
    return result


def _cancels(core, scale):
    return scale > 0.0 and np.linalg.norm(core, ord="fro") <= _CANCELLATION_RTOL * scale


def core_unprojected(a_diag, Y, B, *, S, info=None):
    """Unprojected current action contracted with the exact small correction."""
    a_diag, Y, B, S = _inputs(a_diag, Y, B, S)
    W = np.matmul(B, S)
    work = np.multiply(a_diag[None, :, None], B)
    np.subtract(W, work, out=W)
    np.multiply(W, 2.0, out=W)
    _count(info, "unprojected_action_columns_materialized", len(B))
    flat = B.reshape(len(B), Y.size)
    raw = flat @ W.reshape(len(B), Y.size).T
    E = np.matmul(Y.T[None, :, :], B).reshape(len(B), Y.shape[1] ** 2)
    F = np.matmul(Y.T[None, :, :], W).reshape(E.shape)
    correction = E @ F.T
    result = _symmetric_finite(raw - correction)
    scale = float(np.linalg.norm(raw, ord="fro") + np.linalg.norm(correction, ord="fro"))
    if _cancels(result, scale):
        _count(info, "projection_cancellation_fallback")
        return core_original(a_diag, Y, B, S=S, info=info)
    _count(info, "unprojected")
    _count(info, "exact_current_core_columns", len(B))
    return result


def _full_lower(matrix):
    lower = np.tril(matrix)
    return lower + np.tril(lower, -1).T


def core_syrk(a_diag, Y, B, *, S, info=None):
    """Two symmetric rank-k products plus the generic projection correction."""
    a_diag, Y, B, S = _inputs(a_diag, Y, B, S)
    if np.any(a_diag <= 0.0):
        _count(info, "nonpositive_diagonal_fallback")
        return core_unprojected(a_diag, Y, B, S=S, info=info)
    symmetric_S = 0.5 * (S + S.T)
    try:
        L = np.linalg.cholesky(symmetric_S)
    except np.linalg.LinAlgError:
        _count(info, "cholesky_fallback")
        return core_unprojected(a_diag, Y, B, S=S, info=info)
    if not len(B):
        _count(info, "exact_current_core_columns", 0)
        return np.zeros((0, 0))
    work = np.empty(B.shape, dtype=float, order="C")
    np.multiply(np.sqrt(a_diag)[None, :, None], B, out=work)
    # flat.T is F-contiguous: dsyrk(trans=1) forms an ell-by-ell Gram
    # without materializing a full GEMM or copying the tall factor.
    first = dsyrk(-2.0, work.reshape(len(B), Y.size).T, trans=1, lower=1)
    np.matmul(B, L, out=work)
    second = dsyrk(2.0, work.reshape(len(B), Y.size).T, trans=1, lower=1)
    _count(info, "gram_factor_columns_processed", 2 * len(B))
    first = _full_lower(first)
    second = _full_lower(second)
    E = np.matmul(Y.T[None, :, :], B)
    # F=2(E S-Y.T A B), avoiding a full W or full A B allocation.
    weighted_YT = (a_diag[:, None] * Y).T
    F = np.matmul(E, symmetric_S)
    F -= np.matmul(weighted_YT[None, :, :], B)
    F *= 2.0
    small_shape = (len(B), Y.shape[1] ** 2)
    correction = E.reshape(small_shape) @ F.reshape(small_shape).T
    result = _symmetric_finite(first + second - correction)
    scale = float(np.linalg.norm(first, ord="fro") + np.linalg.norm(second, ord="fro")
                  + np.linalg.norm(correction, ord="fro"))
    if _cancels(result, scale):
        _count(info, "gram_cancellation_fallback")
        return core_original(a_diag, Y, B, S=S, info=info)
    _count(info, "syrk")
    _count(info, "exact_current_core_columns", len(B))
    return result


KERNELS = {
    "original": core_original,
    "buffered": core_buffered,
    "unprojected": core_unprojected,
    "syrk": core_syrk,
}
