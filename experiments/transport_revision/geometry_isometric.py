"""Full-polar isometric transport, with solver-local geometry hooks.

This is a NEW transport variant, not an optimization equivalent to upstream
project-and-whiten transport and not differentiated QR/polar retraction.
For horizontal eta at X, D=(I+eta.T@eta)**(1/2), the map is

    U(Z) = (I-Y@Y.T) [Z + eta@(D+I)^(-1)@eta.T@Z]
         = Z - (X + eta@(D+I)^(-1))@D^(-1)@eta.T@Z.

Here Y is the QR retraction; its column projector equals the polar
retraction's projector. U is the full polar factor of tangent projection
restricted to X-perp, hence a Frobenius isometry on ALL horizontal vectors,
not just on the cached sketch. The fast equality assumes X.T@Z=0.

For orthonormal transported rows Q and normalized gradient q, set a=Q@q,
s=sqrt(1-a.T@a). Gradient deflation has Gram I-a@a.T, whose inverse square
root gives Q + a[:,None] * ((a@Q)/(s*(1+s)) - q/s)[None,:]. This avoids an
ell-by-ell whitening only when s is safely bounded away from zero. Near
rank loss, generic/mixed/truncated inputs use explicit eigen whitening.

Patch ALL HOOK_NAMES together for one sequential solver call. Exact object
identity links retraction inputs to the accepted destination (or the rejected
source). Frames/candidates must not be mutated between paired calls. Only
known hook-generated frames or their exact leading-row-removed views use the
fast frame path. The upstream solver's rejection policy remains untouched.

The returned transport defect 0 is ANALYTIC, not a measured floating-point
Gram norm. Full Gram/tangency diagnostics belong in separate untimed runs;
this file does not silently present an omitted diagnostic as a measurement.
Both fresh and periodically refreshed arms use the SAME native eigen
whitening, with only the provably redundant paired projection omitted.
Current Hessian actions are still required at every iterate.
"""

from __future__ import annotations

import weakref

import numpy as np

from rsncn import grassmann as upstream


HOOK_NAMES = (
    "random_tangent_metric_gaussian", "basis_from_candidates",
    "transport_tangent_basis", "build_transport_sketch_basis", "retraction_qr",
)


def polar_transport_batch(X, eta, basis):
    """Apply the full isometry to horizontal rows; no frame whitening.

    Preconditions: X.T@X=I, X.T@eta=0 and X.T@basis[j]=0. Caller performs
    pairing/shape/finite guards. This helper supports arbitrary, not necessarily
    orthonormal, horizontal vectors and is directly checked against the full
    projection formula in the tests.
    """
    gram = eta.T @ eta
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    d = np.sqrt(1.0 + np.maximum(values, 0.0))
    inv_d = (vectors * (1.0 / d)) @ vectors.T
    correction = (vectors * (1.0 / (d * (d + 1.0)))) @ vectors.T
    A = X @ inv_d + eta @ correction
    coefficients = np.matmul(eta.T[None, :, :], basis)
    return np.ascontiguousarray(basis - np.matmul(A[None, :, :], coefficients))


class GeometryHooks:
    """Private state for one sequential solver call; never shared by threads."""

    def __init__(self, *, skip_transport_reorth=False):
        if skip_transport_reorth:
            raise ValueError("Full-polar hooks do not support transport_skip_reorth=True")
        self._fresh = None
        self._frame = None
        self._move = None
        self._transported = None
        self.stats = {
            "fresh_projection_skipped": 0,
            "fresh_eigen_whitenings": 0,
            "polar_transports": 0,
            "rejected_identity_transports": 0,
            "transport_fallback": 0,
            "gradient_rank_one_whitenings": 0,
            "build_fallback": 0,
            "rank_loss_fallback": 0,
            "missing_context_errors": 0,
            "nonfinite_errors": 0,
        }

    @staticmethod
    def _matches(record, array, point):
        return record is not None and record[0]() is array and record[1]() is point

    def _remember_frame(self, Y, array):
        array = np.ascontiguousarray(array)
        if not np.all(np.isfinite(array)):
            self.stats["nonfinite_errors"] += 1
            raise ValueError("Cannot cache a nonfinite tangent frame")
        self._frame = (array, Y)
        return array

    def _known_frame_slice(self, basis, point):
        if self._frame is None or self._frame[1] is not point:
            return False
        frame = self._frame[0]
        if basis.ndim != 3 or basis.shape[1:] != frame.shape[1:]:
            return False
        # Upstream caches either the entire frame or basis[1:]. No copies,
        # reordered rows or unrelated arrays are trusted merely by shape.
        for start in (0, 1):
            candidate = frame[start:]
            if (basis.shape == candidate.shape and basis.strides == candidate.strides
                    and basis.ctypes.data == candidate.ctypes.data):
                return True
        return False

    def random_tangent_metric_gaussian(self, rng, Y, ell):
        result = np.ascontiguousarray(upstream.random_tangent_metric_gaussian(rng, Y, ell))
        self._fresh = (weakref.ref(result), weakref.ref(Y))
        return result

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
            flat = flat - (flat @ first)[:, None] * first[None, :]
        self.stats["fresh_eigen_whitenings"] += int(flat.size > 0 and remaining > 0)
        rest = upstream._orthonormalize_flat(flat, remaining) if remaining else np.zeros((0, n * p))
        result = np.vstack((first, rest)) if first is not None else rest
        return self._remember_frame(Y, result.reshape(result.shape[0], n, p))

    @staticmethod
    def _gradient_direction(Y, grad):
        if grad is None:
            return None
        flat = upstream.proj_Y(Y, np.asarray(grad, dtype=float)).reshape(-1)
        norm = float(np.linalg.norm(flat))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite gradient cannot define a sketch frame")
        return flat / norm if norm > 1e-12 else None

    def retraction_qr(self, X, eta):
        Y = np.ascontiguousarray(upstream.retraction_qr(X, eta))
        # Keep source/eta alive until the next trial. Recording a trial does
        # NOT make it accepted: transport checks exact destination identity.
        self._move = (X, eta, Y)
        return Y

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
        if not np.all(np.isfinite(batch)) or not np.all(np.isfinite(eta)):
            self.stats["nonfinite_errors"] += 1
            raise ValueError("Full-polar transport received nonfinite vectors or step")
        if np.linalg.norm(X.T @ eta) > 1e-10 * max(1.0, float(np.linalg.norm(eta))):
            raise ValueError("Full-polar transport requires a horizontal retraction step")
        if not known:
            # Generic inputs have no trusted frame invariant. Project at the
            # KNOWN source, then explicitly orthonormalize after the map.
            batch = upstream._proj_Y_batch(X, batch)
        if Y_new is destination:
            result = polar_transport_batch(X, eta, batch)
            self.stats["polar_transports"] += 1
        else:
            # A rejected trial with cache retention leaves the point at X.
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
        # Analytic preservation of the input frame Gram, NOT a numerical test.
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
            # Full eigen whitening handles rank loss robustly. Calling this
            # path also restores tangent projection for unpaired inputs.
            return self.basis_from_candidates(Y, grad, candidates, ell)

        if not paired or fresh.size or remaining < trans.shape[0]:
            return fallback()
        if remaining == 0 or not trans.size:
            flat = first[None, :] if first is not None else np.zeros((0, n * p))
            return self._remember_frame(Y, flat.reshape(flat.shape[0], n, p))
        flat = trans.reshape(trans.shape[0], n * p)
        if first is None:
            return self._remember_frame(Y, flat.reshape(flat.shape[0], n, p))
        a = flat @ first
        s_squared = 1.0 - float(a @ a)
        if not np.isfinite(s_squared) or s_squared <= 1e-6:
            self.stats["rank_loss_fallback"] += 1
            return fallback()
        s = np.sqrt(s_squared)
        update = (a @ flat) / (s * (1.0 + s)) - first / s
        rest = flat + a[:, None] * update[None, :]
        if not np.all(np.isfinite(rest)):
            return fallback()
        self.stats["gradient_rank_one_whitenings"] += 1
        result = np.vstack((first, rest))
        return self._remember_frame(Y, result.reshape(result.shape[0], n, p))


def make_hooks(*, skip_transport_reorth=False):
    return GeometryHooks(skip_transport_reorth=skip_transport_reorth)
