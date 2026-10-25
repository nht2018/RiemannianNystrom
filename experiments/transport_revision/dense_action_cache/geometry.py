"""Fused full-polar geometry with shared-coefficient fixed-A action updates.

The two guarded methods below preserve the fused geometry's operations,
thresholds and fallback decisions. Extra action calculations use the SAME
M,C,u,a,s arrays, not independently recomputed eigen/rank-one coefficients.
Action instability guards only discard a cache; they never change geometry.
"""
from __future__ import annotations

import weakref
import numpy as np

from rsncn import grassmann as upstream
from geometry_isometric_fused import GeometryHooks as FusedHooks
from dense.kernels import _add


class ActionCacheGeometry(FusedHooks):
    def __init__(self, oracle, *, skip_transport_reorth=False):
        super().__init__(skip_transport_reorth=skip_transport_reorth)
        self.oracle = oracle
        self._basis_rebase_reason = None

    def _remember_frame(self, Y, array, *, finite_verified=False):
        result = super()._remember_frame(Y, array, finite_verified=finite_verified)
        self.oracle.trust_frame(Y, result)
        return result

    def basis_from_candidates(self, Y, grad, candidates, ell):
        result = super().basis_from_candidates(Y, grad, candidates, ell)
        self.oracle.invalidate(self._basis_rebase_reason or "fresh_frame")
        return result

    def transport_tangent_basis(self, Y_new, basis, ell=None, *, return_orth_error=False):
        self._transported = None
        n, p = Y_new.shape
        batch = upstream._as_batch(basis, n, p)
        if not batch.size:
            self.oracle.invalidate("empty_transport")
            empty = np.zeros((0, n, p))
            return (empty, 0.0) if return_orth_error else empty
        if self._move is None or (Y_new is not self._move[0] and Y_new is not self._move[2]):
            self.stats["missing_context_errors"] += 1
            raise ValueError("Full-polar transport requires its paired QR trial source/destination")
        X, eta, destination = self._move
        known = self._known_frame_slice(batch, X)
        if (not known and not np.all(np.isfinite(batch))) or not np.all(np.isfinite(eta)):
            self.stats["nonfinite_errors"] += 1
            raise ValueError("Full-polar transport received nonfinite vectors or step")
        self.stats["trusted_finite_scan_skipped"] += int(known)
        if np.linalg.norm(X.T @ eta) > 1e-10*max(1.0, float(np.linalg.norm(eta))):
            raise ValueError("Full-polar transport requires a horizontal retraction step")
        actions = self.oracle.lookup_actions(X, batch, allow_leading_slice=True) if known else None
        if not known:
            batch = upstream._proj_Y_batch(X, batch)
        if Y_new is destination:
            gram = eta.T @ eta
            values, vectors = np.linalg.eigh(0.5*(gram+gram.T))
            d = np.sqrt(1.0+np.maximum(values, 0.0))
            inv_d = (vectors*(1.0/d)) @ vectors.T
            correction = (vectors*(1.0/(d*(d+1.0)))) @ vectors.T
            M = X @ inv_d+eta @ correction
            coefficients = np.matmul(eta.T[None, :, :], batch)
            result = np.empty(batch.shape, dtype=float, order="C")
            np.matmul(M[None, :, :], coefficients, out=result)
            np.subtract(batch, result, out=result)
            updated_actions = self.oracle.propagate_polar(actions, M, coefficients) if actions is not None else None
            self.stats["polar_transports"] += 1
        else:
            result = np.ascontiguousarray(batch.copy())
            updated_actions = actions.copy(order="C") if actions is not None else None
            if updated_actions is not None:
                _add(self.oracle.stats, "rejected_identity_action_updates")
            self.stats["rejected_identity_transports"] += 1
        fallback = not known or (ell is not None and int(ell) < result.shape[0])
        if fallback:
            self.stats["transport_fallback"] += 1
            flat = upstream._proj_Y_batch(Y_new, result).reshape(result.shape[0], n*p)
            flat = upstream._orthonormalize_flat(flat, ell)
            result = np.ascontiguousarray(flat.reshape(flat.shape[0], n, p))
            updated_actions = None
        if not np.all(np.isfinite(result)):
            self.stats["nonfinite_errors"] += 1
            raise ValueError("Full-polar transport produced nonfinite vectors")
        self._transported = (weakref.ref(result), weakref.ref(Y_new))
        self.oracle.trust_frame(Y_new, result)
        if updated_actions is not None:
            self.oracle.store_actions(Y_new, result, updated_actions, origin="polar_or_identity_update")
        else:
            self.oracle.invalidate("transport_fallback" if fallback else "transport_without_matching_actions")
        return (result, 0.0) if return_orth_error else result

    def build_transport_sketch_basis(self, Y, *, ell, grad, transported=None,
                                    fresh_candidates=None, skip_transport_reorth=False):
        if skip_transport_reorth:
            raise ValueError("Full-polar hooks do not support transport_skip_reorth=True")
        n, p = Y.shape
        trans = upstream._as_batch(transported if transported is not None else [], n, p)
        fresh = upstream._as_batch(fresh_candidates if fresh_candidates is not None else [], n, p)
        first = self._gradient_direction(Y, grad)
        remaining = max(0, int(ell)-int(first is not None))
        paired = self._matches(self._transported, transported, Y)
        actions = self.oracle.lookup_actions(Y, trans) if paired else None

        def fallback(reason="mixed_or_unpaired_build"):
            self.stats["build_fallback"] += 1
            candidates = np.concatenate((trans, fresh), axis=0)
            self._basis_rebase_reason = reason
            try:
                return self.basis_from_candidates(Y, grad, candidates, ell)
            finally:
                self._basis_rebase_reason = None

        if not paired or fresh.size or remaining < trans.shape[0]:
            return fallback()
        if remaining == 0 or not trans.size:
            flat = first[None, :] if first is not None else np.zeros((0, n*p))
            result = self._remember_frame(Y, flat.reshape(flat.shape[0], n, p))
            self.oracle.invalidate("gradient_only_or_empty_frame")
            return result
        flat = trans.reshape(trans.shape[0], n*p)
        if first is None:
            result = self._remember_frame(Y, trans, finite_verified=True)
            if actions is None:
                self.oracle.invalidate("gradient_free_without_matching_actions")
            return result
        a = flat @ first
        s_squared = 1.0-float(a @ a)
        if not np.isfinite(s_squared) or s_squared <= 1e-6:
            self.stats["rank_loss_fallback"] += 1
            return fallback("rank_loss_geometry_fallback")
        s = np.sqrt(s_squared)
        update = (a @ flat)/(s*(1.0+s))-first/s
        result = np.empty((len(flat)+1, n*p), dtype=float, order="C")
        result[0] = first
        np.multiply(a[:, None], update[None, :], out=result[1:])
        np.add(flat, result[1:], out=result[1:])
        if not np.all(np.isfinite(result)):
            return fallback("nonfinite_rank_one_geometry_fallback")
        self.stats["gradient_rank_one_whitenings"] += 1
        self.stats["fused_rank_one_updates"] += 1
        result = self._remember_frame(Y, result.reshape(result.shape[0], n, p), finite_verified=True)
        self.oracle.stats["max_rank_one_amplification_considered"] = max(
            self.oracle.stats["max_rank_one_amplification_considered"], 1.0+float(a @ a)/(s*(1.0+s)))
        if actions is not None and s_squared >= 0.25:
            updated = self.oracle.propagate_gradient(actions, first, a, s)
            self.oracle.store_actions(Y, result, updated, origin="gradient_rank_one_update")
        else:
            reason = "rank_one_amplification_rebase" if s_squared < 0.25 else "rank_one_without_matching_actions"
            self.oracle.invalidate(reason)
            if s_squared < 0.25:
                _add(self.oracle.stats, "low_s_squared_rebase_requests")
        return result
