"""General dense-symmetric Grassmann oracles; no diagonal/spectral shortcut.

The optimization interface receives only a dense ``A``. Every objective,
gradient and Hessian evaluation multiplies that current array; no eigenvectors,
eigenvalues, factorization, or previous-point curvature are cached here.

For an orthonormal point Y, S=Y.T A Y, and horizontal U,
    grad f = -2 (I-Y Y.T) A Y,
    Hess f[U] = (I-Y Y.T) (-2 A U + 2 U S).
For an arbitrary supplied frame B, ``core`` retains the exact projection
correction in sym(<B_i, P_Y(-2 A B_j + 2 B_j S)>). Calling this expression a
Riemannian Hessian requires horizontal inputs; nonhorizontal inputs are kept
only as a well-defined extension matching the original solver's HVP formula.

Dense products pack one chunk as n-by-(chunk*p). The workspace budget covers
a conservative estimate of explicit temporary chunk/correction buffers. It
excludes caller inputs, returned arrays, dtype conversions, and opaque BLAS
packing/scratch. This is not an RSS bound. Use native float64 inputs to avoid
input-conversion copies. Matrix symmetry can be checked once, outside timings,
using ``validate_matrix``; expensive whole-A checks are not repeated per call.
"""

from __future__ import annotations

import operator

import numpy as np

from rsncn.grassmann import GeometryState


DEFAULT_WORKSPACE_BYTES = 128 * 1024**2


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return np.asarray(value, dtype=float)


def _base_inputs(A, Y):
    A = _real_array(A, "A")
    Y = _real_array(Y, "Y")
    if Y.ndim != 2 or min(Y.shape) <= 0 or Y.shape[1] > Y.shape[0]:
        raise ValueError("Y must have shape (n,p), with 0 < p <= n")
    if A.ndim != 2 or A.shape != (Y.shape[0], Y.shape[0]):
        raise ValueError("A must be a dense square (n,n) matrix; diagonal vectors are not accepted")
    return A, Y


def validate_matrix(A, *, rtol=1e-12, atol=1e-12):
    """Optional one-time, untimed validation of the dense symmetric input."""
    A = _real_array(A, "A")
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not len(A):
        raise ValueError("A must be a nonempty dense square matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("A is nonfinite")
    if not np.allclose(A, A.T, rtol=rtol, atol=atol):
        raise ValueError("A must be symmetric")
    return A


def _add(mapping, name, amount=1):
    if mapping is not None:
        mapping[name] = mapping.get(name, 0) + amount


class DenseOracle:
    """State-free-in-curvature dense hooks with transparent work counters.

    ``chunk_size`` counts frame directions, not individual matrix RHS columns.
    Its effective value is reduced when necessary to respect the explicit
    workspace estimate. Use the same configuration for every comparison arm.
    ``stats`` contains counters only, never cached matrices or eigeninformation.
    """

    def __init__(self, chunk_size=16, workspace_bytes=DEFAULT_WORKSPACE_BYTES):
        if chunk_size is not None:
            chunk_size = operator.index(chunk_size)
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive or None")
        workspace_bytes = operator.index(workspace_bytes)
        if workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be positive")
        self.chunk_size = chunk_size
        self.workspace_bytes = workspace_bytes
        self.stats = {
            "kernel": "general_dense_current_hessian",
            "requested_chunk_size": chunk_size,
            "workspace_budget_bytes": workspace_bytes,
            "peak_explicit_workspace_estimate_bytes": 0,
            "maximum_effective_chunk_size": 0,
            "workspace_scope": "Explicit transient buffers only; excludes inputs, returned arrays, dtype conversions and opaque BLAS scratch.",
        }

    def _multiply(self, A, block, purpose, info=None):
        result = A @ block
        _add(self.stats, "dense_A_matmul_calls")
        _add(self.stats, "dense_A_rhs_columns", block.shape[1])
        _add(self.stats, f"{purpose}_dense_A_matmul_calls")
        _add(self.stats, f"{purpose}_dense_A_rhs_columns", block.shape[1])
        _add(info, "dense_A_matmul_calls")
        _add(info, "dense_A_rhs_columns", block.shape[1])
        return result

    def objective(self, A, Y):
        A, Y = _base_inputs(A, Y)
        AY = self._multiply(A, Y, "objective")
        value = -float(np.einsum("ij,ij->", Y, AY))
        if not np.isfinite(value):
            raise ValueError("Objective is nonfinite")
        _add(self.stats, "objective_calls")
        return value

    def geometry_terms(self, A, Y):
        A, Y = _base_inputs(A, Y)
        AY = self._multiply(A, Y, "geometry")
        S = Y.T @ AY
        grad_e = -2.0 * AY
        grad = grad_e + 2.0 * (Y @ S)
        value = -float(np.trace(S))
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            raise ValueError("Geometry terms are nonfinite")
        _add(self.stats, "geometry_calls")
        return GeometryState(AY=AY, S=S, grad=grad, grad_euclidean=grad_e), value

    def _frame_inputs(self, A, Y, B, S):
        A, Y = _base_inputs(A, Y)
        B = _real_array(B, "B")
        if B.ndim != 3 or B.shape[1:] != Y.shape:
            raise ValueError("Frame must have shape (ell,n,p)")
        if S is None:
            S = Y.T @ self._multiply(A, Y, "current_S")
        S = _real_array(S, "S")
        if S.shape != (Y.shape[1], Y.shape[1]) or not np.all(np.isfinite(S)):
            raise ValueError("S must be a finite (p,p) matrix")
        return A, Y, B, S

    def _chunk_limit(self, ell, n, p, *, core):
        if ell == 0:
            return 0
        # Four n*p buffers cover packing/product/action plus either a strided
        # left-frame copy or a projection temporary. Small E is full-frame;
        # F, contraction/correction blocks, and row symmetrization are covered.
        fixed = 8 * ell * p * p if core else 0
        per_direction = 8 * (4 * n * p + 4 * p * p + 4 * ell)
        available = (self.workspace_bytes - fixed) // per_direction
        if available < 1:
            raise ValueError("workspace_bytes is too small for one frame direction")
        chunk = min(ell, available, self.chunk_size or ell)
        estimate = fixed + per_direction * chunk
        self.stats["peak_explicit_workspace_estimate_bytes"] = max(
            self.stats["peak_explicit_workspace_estimate_bytes"], estimate)
        self.stats["maximum_effective_chunk_size"] = max(
            self.stats["maximum_effective_chunk_size"], chunk)
        return chunk

    def _unprojected_chunks(self, A, Y, B, S, chunk, purpose, info=None):
        n, p = Y.shape
        for start in range(0, len(B), chunk):
            stop = min(start + chunk, len(B))
            block = B[start:stop]
            packed = np.ascontiguousarray(block.transpose(1, 0, 2)).reshape(n, (stop-start)*p)
            product = self._multiply(A, packed, purpose, info)
            W = np.matmul(block, S)
            W -= product.reshape(n, stop-start, p).transpose(1, 0, 2)
            W *= 2.0
            del packed, product
            if not np.all(np.isfinite(W)):
                raise ValueError("Current Hessian action is nonfinite")
            _add(self.stats, f"{purpose}_chunks")
            _add(info, "dense_core_chunks")
            yield start, stop, W

    def hess_action_batch(self, A, Y, U, *, S=None):
        """Materialize the exact current projected dense HVP for each U."""
        A, Y, U, S = self._frame_inputs(A, Y, U, S)
        chunk = self._chunk_limit(len(U), *Y.shape, core=False)
        result = np.empty(U.shape, dtype=float)
        if chunk:
            for start, stop, W in self._unprojected_chunks(A, Y, U, S, chunk, "hvp"):
                small = np.matmul(Y.T[None], W)
                W -= np.matmul(Y[None], small)
                if not np.all(np.isfinite(W)):
                    raise ValueError("Projected Hessian action is nonfinite")
                result[start:stop] = W
        _add(self.stats, "hvp_calls")
        _add(self.stats, "projected_hvp_directions", len(U))
        return result

    def core(self, A, Y, B, *, S=None, info=None):
        """Exact symmetric current core, including nonhorizontal correction."""
        A, Y, B, S = self._frame_inputs(A, Y, B, S)
        ell, n, p = B.shape
        chunk = self._chunk_limit(ell, n, p, core=True)
        result = np.empty((ell, ell), dtype=float)
        if chunk:
            E = np.matmul(Y.T[None], B).reshape(ell, p*p)
            flat = B.reshape(ell, n*p) if B.flags.c_contiguous else None
            for start, stop, W in self._unprojected_chunks(A, Y, B, S, chunk, "core", info):
                right = W.reshape(stop-start, n*p).T
                if flat is not None:
                    np.matmul(flat, right, out=result[:, start:stop])
                else:
                    # Avoid a hidden copy of the entire noncontiguous frame.
                    for left in range(0, ell, chunk):
                        end = min(left+chunk, ell)
                        left_flat = np.ascontiguousarray(B[left:end]).reshape(end-left, n*p)
                        np.matmul(left_flat, right, out=result[left:end, start:stop])
                F = np.matmul(Y.T[None], W).reshape(stop-start, p*p)
                result[:, start:stop] -= E @ F.T
                _add(info, "unprojected_action_columns_materialized", stop-start)
                _add(info, "projection_correction_columns", stop-start)
            # In-place symmetrization avoids a second full ell-by-ell array.
            for i in range(ell):
                row = 0.5 * (result[i, i+1:] + result[i+1:, i])
                result[i, i+1:] = row
                result[i+1:, i] = row
                if not np.all(np.isfinite(result[i])):
                    raise ValueError("Reduced Hessian core is nonfinite")
        _add(self.stats, "current_core_calls")
        _add(self.stats, "exact_current_core_columns", ell)
        _add(info, "dense_current_core")
        _add(info, "exact_current_core_columns", ell)
        return result


def objective(A, Y):
    return DenseOracle().objective(A, Y)


def geometry_terms(A, Y):
    return DenseOracle().geometry_terms(A, Y)


def hess_action_batch(A, Y, U, *, S=None, chunk_size=16,
                      workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    return DenseOracle(chunk_size, workspace_bytes).hess_action_batch(A, Y, U, S=S)


def core(A, Y, B, *, S=None, info=None, chunk_size=16,
         workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    return DenseOracle(chunk_size, workspace_bytes).core(A, Y, B, S=S, info=info)


core_dense = core
