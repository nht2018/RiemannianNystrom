"""Allocation-fused implementation of ``geometry_isometric`` (v1 preserved).

The full-polar map, native eigen fresh whitening, Gaussian stream, rank-loss
threshold and fallback decisions are unchanged. Both comparator arms must use
these hooks, so fresh deflation's allocation improvement applies equally to
the every-step-fresh arm and the periodic refreshes of the transport arm.

Large broadcast products are written directly into their destination arrays.
The transport input finite scan is omitted ONLY for the exact immutable
hook-generated frame/slice provenance recognized by v1: the same array was
already verified finite when cached, and mutation between paired calls is
prohibited. Generic inputs retain all v1 finite checks. Each newly computed
transport and each final frame is verified finite; the rank-one intermediate
is the final frame's row view, so it no longer needs a duplicate full scan.

The sixth hook buffers the CURRENT Hessian action using the same formula
for arbitrary input matrices (no horizontality shortcut). It changes only
temporary-array allocation, is applied to ALL comparison arms, and does not
reuse a Hessian or reduce the number of Hessian-vector products. This module
changes neither the transport mathematics nor Hessian work. Its timing
benefit is an empirical question; no speedup is guaranteed here.
"""

from __future__ import annotations

import weakref

import numpy as np

from rsncn import grassmann as upstream
from geometry_isometric import GeometryHooks as ReferenceHooks, HOOK_NAMES as REFERENCE_HOOK_NAMES


HOOK_NAMES = REFERENCE_HOOK_NAMES + ("hess_action_batch",)


def polar_transport_batch(X, eta, basis):
    """Identical full-polar map; avoid a full-sized correction temporary."""
    gram = eta.T @ eta
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    d = np.sqrt(1.0 + np.maximum(values, 0.0))
    inv_d = (vectors * (1.0 / d)) @ vectors.T
    correction = (vectors * (1.0 / (d * (d + 1.0)))) @ vectors.T
    A = X @ inv_d + eta @ correction
    coefficients = np.matmul(eta.T[None, :, :], basis)
    result = np.empty(basis.shape, dtype=float, order="C")
    np.matmul(A[None, :, :], coefficients, out=result)
    np.subtract(basis, result, out=result)
    return result


class GeometryHooks(ReferenceHooks):
    """v1 contracts and guards, with reduced temporary-array allocation."""

    def __init__(self, *, skip_transport_reorth=False):
        super().__init__(skip_transport_reorth=skip_transport_reorth)
        self.stats.update({
            "trusted_finite_scan_skipped": 0,
            "fused_fresh_deflations": 0,
            "fused_rank_one_updates": 0,
        })

    def _remember_frame(self, Y, array, *, finite_verified=False):
        array = np.ascontiguousarray(array)
        if not finite_verified and not np.all(np.isfinite(array)):
            self.stats["nonfinite_errors"] += 1
            raise ValueError("Cannot cache a nonfinite tangent frame")
        self._frame = (array, Y)
        return array

    @staticmethod
    def hess_action_batch(a_diag, Y, U, *, S):
        """Buffered CURRENT Hessian action, identical work for every arm.

        P_Y(-2 A U + 2 U S) is evaluated for arbitrary U. Both contractions
        and the diagonal multiplication are recomputed on every call. No
        prior-point Hessian data or transported Hessian values are reused.
        """
        result = np.matmul(U, S)
        work = np.multiply(a_diag[None, :, None], U)
        np.subtract(result, work, out=result)
        np.multiply(result, 2.0, out=result)
        coefficients = np.matmul(Y.T[None, :, :], result)
        np.matmul(Y[None, :, :], coefficients, out=work)
        np.subtract(result, work, out=result)
        return np.ascontiguousarray(result)

    def basis_from_candidates(self, Y, grad, candidates, ell):
        n, p = Y.shape
        tangent = upstream._as_batch(candidates, n, p)
        if self._matches(self._fresh, candidates, Y):
            self.stats["fresh_projection_skipped"] += 1
        else:
            tangent = upstream._proj_Y_batch(Y, tangent)
        flat = tangent.reshape(tangent.shape[0], n * p)
        first = self._gradient_direction(Y, grad)
        remaining = max(0, int(ell) - int(first is not None))
        if first is not None:
            coefficients = flat @ first
            deflated = np.empty(flat.shape, dtype=float, order="C")
            np.multiply(coefficients[:, None], first[None, :], out=deflated)
            np.subtract(flat, deflated, out=deflated)
            flat = deflated
            self.stats["fused_fresh_deflations"] += 1
        self.stats["fresh_eigen_whitenings"] += int(flat.size > 0 and remaining > 0)
        rest = upstream._orthonormalize_flat(flat, remaining) if remaining else np.zeros((0, n * p))
        result = np.vstack((first, rest)) if first is not None else rest
        return self._remember_frame(Y, result.reshape(result.shape[0], n, p))

    def transport_tangent_basis(self, Y_new, basis, ell=None, *, return_orth_error=False):
        self._transported = None
        n, p = Y_new.shape
        batch = upstream._as_batch(basis, n, p)
        if not batch.size:
            empty = np.zeros((0, n, p))
            return (empty, 0.0) if return_orth_error else empty
        if self._move is None or (Y_new is not self._move[0] and Y_new is not self._move[2]):
            self.stats["missing_context_errors"] += 1
            raise ValueError("Full-polar transport requires its paired QR trial source/destination")
        X, eta, destination = self._move
        known = self._known_frame_slice(batch, X)
        # Exact provenance + no mutation means cached finiteness is still
        # valid. Generic inputs are NEVER granted this shortcut.
        if (not known and not np.all(np.isfinite(batch))) or not np.all(np.isfinite(eta)):
            self.stats["nonfinite_errors"] += 1
            raise ValueError("Full-polar transport received nonfinite vectors or step")
        self.stats["trusted_finite_scan_skipped"] += int(known)
        if np.linalg.norm(X.T @ eta) > 1e-10 * max(1.0, float(np.linalg.norm(eta))):
            raise ValueError("Full-polar transport requires a horizontal retraction step")
        if not known:
            batch = upstream._proj_Y_batch(X, batch)
        if Y_new is destination:
            result = polar_transport_batch(X, eta, batch)
            self.stats["polar_transports"] += 1
        else:
            result = np.ascontiguousarray(batch.copy())
            self.stats["rejected_identity_transports"] += 1
        if not known or (ell is not None and int(ell) < result.shape[0]):
            self.stats["transport_fallback"] += 1
            flat = upstream._proj_Y_batch(Y_new, result).reshape(result.shape[0], n * p)
            flat = upstream._orthonormalize_flat(flat, ell)
            result = np.ascontiguousarray(flat.reshape(flat.shape[0], n, p))
        if not np.all(np.isfinite(result)):
            self.stats["nonfinite_errors"] += 1
            raise ValueError("Full-polar transport produced nonfinite vectors")
        self._transported = (weakref.ref(result), weakref.ref(Y_new))
        # Same analytic-only defect convention as v1, never a measured Gram.
        return (result, 0.0) if return_orth_error else result

    def build_transport_sketch_basis(
        self, Y, *, ell, grad, transported=None, fresh_candidates=None,
        skip_transport_reorth=False,
    ):
        if skip_transport_reorth:
            raise ValueError("Full-polar hooks do not support transport_skip_reorth=True")
        n, p = Y.shape
        trans = upstream._as_batch(transported if transported is not None else [], n, p)
        fresh = upstream._as_batch(fresh_candidates if fresh_candidates is not None else [], n, p)
        first = self._gradient_direction(Y, grad)
        remaining = max(0, int(ell) - int(first is not None))
        paired = self._matches(self._transported, transported, Y)

        def fallback():
            self.stats["build_fallback"] += 1
            candidates = np.concatenate((trans, fresh), axis=0)
            return self.basis_from_candidates(Y, grad, candidates, ell)

        if not paired or fresh.size or remaining < trans.shape[0]:
            return fallback()
        if remaining == 0 or not trans.size:
            flat = first[None, :] if first is not None else np.zeros((0, n * p))
            return self._remember_frame(Y, flat.reshape(flat.shape[0], n, p))
        flat = trans.reshape(trans.shape[0], n * p)
        if first is None:
            # Exactly paired transported array was checked finite above.
            return self._remember_frame(Y, trans, finite_verified=True)
        a = flat @ first
        s_squared = 1.0 - float(a @ a)
        if not np.isfinite(s_squared) or s_squared <= 1e-6:
            self.stats["rank_loss_fallback"] += 1
            return fallback()
        s = np.sqrt(s_squared)
        update = (a @ flat) / (s * (1.0 + s)) - first / s
        result = np.empty((len(flat) + 1, n * p), dtype=float, order="C")
        result[0] = first
        # These operations keep the same multiply-then-add floating-point
        # ordering as v1, without outer-product/rest/vstack temporaries.
        np.multiply(a[:, None], update[None, :], out=result[1:])
        np.add(flat, result[1:], out=result[1:])
        if not np.all(np.isfinite(result)):
            return fallback()
        self.stats["gradient_rank_one_whitenings"] += 1
        self.stats["fused_rank_one_updates"] += 1
        return self._remember_frame(Y, result.reshape(result.shape[0], n, p), finite_verified=True)


def make_hooks(*, skip_transport_reorth=False):
    return GeometryHooks(skip_transport_reorth=skip_transport_reorth)
