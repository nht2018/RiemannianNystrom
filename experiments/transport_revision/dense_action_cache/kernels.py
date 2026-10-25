"""Exact-in-real-arithmetic fixed-A action updates, not Hessian transport.

The cache contains D_j=A B_j for one private, immutable frame and one fixed
read-only dense float64 A. Current S=Y.T A Y and current tangent projection
are still evaluated at every trial. Floating-point updates are not exact:
diagnostic runs measure direct-action/core discrepancies on every trial.

All comparator arms use this oracle. enable_updates=False computes current
actions directly for every core, providing the same-oracle direct ablation.
No generating eigenvectors, spectrum, old Hessian, or old S is stored.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from dense.kernels import DEFAULT_WORKSPACE_BYTES, _add
from dense_opt.kernels import DenseBlockOracle


@dataclass
class ActionRecord:
    frame: np.ndarray
    point: np.ndarray
    actions: np.ndarray
    origin: str


class ActionCacheOracle(DenseBlockOracle):
    """One sequential solve, one immutable dense A, private immutable frames.

    Read-only ndarray flags are a contract, not tamper-proof protection against
    mutation through another owner. The runner hashes A before/after each solve;
    callers must not mutate A or frame owners through aliases during a solve.
    The cache is invalid outside this contract and is not for variable A(Y).
    """

    def __init__(self, chunk_size=None, workspace_bytes=DEFAULT_WORKSPACE_BYTES,
                 *, enable_updates=True, diagnostic=False):
        super().__init__(chunk_size=chunk_size, workspace_bytes=workspace_bytes)
        self.enable_updates = bool(enable_updates)
        self.diagnostic = bool(diagnostic)
        self._fixed_A = None
        self._trusted_frame = None
        self._cache = None
        self._pending_rebase_reason = "cold_start"
        self.stats.update(
            kernel="fixed_dense_A_linear_action_cache_current_hessian",
            action_updates_enabled=self.enable_updates,
            fixed_A_contract="One read-only float64 dense A by object identity; no external alias mutations; runner verifies hashes.",
            cached_quantity="A times the private current sketch frame, NOT a previous Hessian or previous S",
            floating_point_claim="Algebraically exact update; no claim of bitwise equality or a uniform roundoff bound.",
            persistent_action_cache_bytes=0, persistent_action_cache_peak_bytes=0,
            cache_transition_action_bytes_peak=0,
            persistent_storage_scope="Action arrays only, including cache-disabled direct arms; these arrays are excluded from the transient core workspace budget. Held frame/point references, caller inputs and BLAS storage are not included. Transient old+new action storage is reported separately; diagnostic comparison arrays are outside performance workspace scope.",
            workspace_scope="Explicit transient core/packing buffers and chunked finite masks only; excludes cached/direct action arrays (reported separately), caller inputs, returned arrays, dtype/noncontiguous input conversions, diagnostic-only arrays and opaque BLAS scratch.",
            rank_one_amplification_semantics="Computed coefficient norm factor 1+float(a@a)/(s*(1+s)); equals 1/s in exact arithmetic under s^2=1-a@a. This records a computed local coefficient, not a proved floating-point stability bound.",
            max_rank_one_amplification_considered=1.0,
            max_rank_one_amplification_applied=1.0,
            action_rebase_reasons={},
            diagnostic_action_checks=0, diagnostic_max_action_relative_error=0.0,
            diagnostic_max_core_relative_error=0.0, diagnostic_seconds=0.0,
            diagnostic_error_history=[],
        )

    def _bind(self, A):
        if not isinstance(A, np.ndarray) or A.dtype != np.float64 or A.ndim != 2 or A.flags.writeable:
            raise ValueError("Action cache requires one read-only float64 dense ndarray A")
        if self._fixed_A is None:
            self._fixed_A = A
        elif A is not self._fixed_A:
            raise ValueError("Action cache cannot change its fixed A object during a solve")
        return A

    def _multiply(self, A, block, purpose, info=None):
        self._bind(A)
        return super()._multiply(A, block, purpose, info)

    def trust_frame(self, Y, B):
        """Only called for finished, hook-owned arrays; never generic inputs."""
        if B.dtype != np.float64 or not B.flags.c_contiguous:
            raise ValueError("Private action-cache frames must be contiguous float64 arrays")
        B.setflags(write=False)
        self._trusted_frame = (B, Y)

    def invalidate(self, reason):
        self._cache = None
        self._pending_rebase_reason = str(reason)
        self.stats["persistent_action_cache_bytes"] = 0
        _add(self.stats, "action_cache_invalidations")

    def _trusted(self, Y, B):
        return (self._trusted_frame is not None and self._trusted_frame[0] is B
                and self._trusted_frame[1] is Y and not B.flags.writeable)

    def lookup_actions(self, Y, B, *, allow_leading_slice=False):
        if not self.enable_updates or self._cache is None:
            return None
        cache = self._cache
        if cache.point is not Y or cache.frame.flags.writeable or B.flags.writeable:
            return None
        if B is cache.frame:
            return cache.actions
        if allow_leading_slice:
            for start in (0, 1):
                candidate = cache.frame[start:]
                if (B.shape == candidate.shape and B.strides == candidate.strides
                        and B.ctypes.data == candidate.ctypes.data):
                    return cache.actions[start:]
        return None

    def store_actions(self, Y, B, actions, *, origin):
        if not self._trusted(Y, B):
            raise ValueError("Can only attach actions to the exact finished hook-owned frame")
        if actions.shape != B.shape or actions.dtype != np.float64 or not actions.flags.c_contiguous:
            raise ValueError("Cached action array must match its contiguous frame")
        if not self._finite_actions(actions):
            self.invalidate("nonfinite_action_update")
            _add(self.stats, "nonfinite_action_update_rebases")
            return False
        old_bytes = self._cache.actions.nbytes if self._cache is not None else 0
        actions.setflags(write=False)
        self._cache = ActionRecord(B, Y, actions, str(origin))
        self.stats["persistent_action_cache_bytes"] = int(actions.nbytes)
        self.stats["persistent_action_cache_peak_bytes"] = max(self.stats["persistent_action_cache_peak_bytes"], int(actions.nbytes))
        self.stats["cache_transition_action_bytes_peak"] = max(
            self.stats["cache_transition_action_bytes_peak"], old_bytes+int(actions.nbytes))
        self._pending_rebase_reason = None
        return True

    def _finite_actions(self, actions):
        """Chunk masks so a tiny budget does not imply a full-frame mask."""
        ell, n, p = actions.shape
        chunk = self._core_chunk_limit(ell, n, p)
        return all(np.all(np.isfinite(actions[start:start+chunk]))
                   for start in range(0, ell, chunk)) if chunk else True

    def dense_actions(self, A, B, *, purpose, info=None):
        """Direct actual dense A-products, without any spectrum/factorization."""
        self._bind(A)
        ell, n, p = B.shape
        result = np.empty(B.shape, dtype=float, order="C")
        chunk = self._core_chunk_limit(ell, n, p)
        if chunk:
            for start in range(0, ell, chunk):
                stop = min(start+chunk, ell)
                packed = np.ascontiguousarray(B[start:stop].transpose(1, 0, 2)).reshape(n, (stop-start)*p)
                product = self._multiply(A, packed, purpose, info)
                result[start:stop] = product.reshape(n, stop-start, p).transpose(1, 0, 2)
                del packed, product
        if not self._finite_actions(result):
            raise ValueError("Direct dense A-actions are nonfinite")
        return result

    def propagate_polar(self, actions, M, coefficients):
        """The SAME M,C used in B'=B-MC give D'=D-(AM)C."""
        if self._fixed_A is None:
            raise ValueError("No fixed A has been registered")
        AM = self._multiply(self._fixed_A, M, "action_transport")
        result = np.empty(actions.shape, dtype=float, order="C")
        np.matmul(AM[None], coefficients, out=result)
        np.subtract(actions, result, out=result)
        _add(self.stats, "polar_action_updates")
        _add(self.stats, "polar_action_updated_directions", len(actions))
        return result

    def propagate_gradient(self, actions, first, a, s):
        """The SAME u,a,s used in the frame update act linearly on A B."""
        n, p = actions.shape[1:]
        Au = self._multiply(self._fixed_A, first.reshape(n, p), "action_gradient")
        flat = actions.reshape(len(actions), n*p)
        update = (a @ flat)/(s*(1.0+s))-Au.reshape(-1)/s
        result = np.empty((len(actions)+1, n*p), dtype=float, order="C")
        result[0] = Au.reshape(-1)
        np.multiply(a[:, None], update[None], out=result[1:])
        np.add(flat, result[1:], out=result[1:])
        _add(self.stats, "gradient_action_updates")
        _add(self.stats, "gradient_action_updated_directions", len(actions)+1)
        self.stats["max_rank_one_amplification_applied"] = max(
            self.stats["max_rank_one_amplification_applied"], 1.0+float(a @ a)/(s*(1.0+s)))
        return result.reshape(len(actions)+1, n, p)

    def core_from_actions(self, Y, B, S, actions, *, info=None):
        """Current core and current projection, for both direct/cache arms."""
        ell, n, p = B.shape
        result = np.empty((ell, ell), dtype=float)
        chunk = self._core_chunk_limit(ell, n, p)
        if chunk:
            E = np.matmul(Y.T[None], B).reshape(ell, p*p)
            flat = B.reshape(ell, n*p)
            for start in range(0, ell, chunk):
                stop = min(start+chunk, ell)
                W = np.matmul(B[start:stop], S)
                W -= actions[start:stop]
                W *= 2.0
                right = W.reshape(stop-start, n*p).T
                np.matmul(flat, right, out=result[:, start:stop])
                F = np.matmul(Y.T[None], W).reshape(stop-start, p*p)
                result[:, start:stop] -= E @ F.T
                _add(info, "unprojected_action_columns_materialized", stop-start)
                _add(info, "projection_correction_columns", stop-start)
                _add(info, "current_core_assembly_chunks")
                del W, right, F
            for i in range(ell):
                row = 0.5*(result[i, i+1:]+result[i+1:, i])
                result[i, i+1:] = row
                result[i+1:, i] = row
                if not np.all(np.isfinite(result[i])):
                    raise ValueError("Current action-cache reduced core is nonfinite")
        return result

    def core(self, A, Y, B, *, S=None, info=None):
        self._bind(A)
        A, Y, B, S = self._frame_inputs(A, Y, B, S)
        if not B.flags.c_contiguous:
            # Generic external arrays are never trusted as cache provenance.
            # A contiguous input copy is explicitly not covered by the inherited
            # transient budget, exactly like dtype-conversion copies.
            self.invalidate("generic_noncontiguous_core")
            B = np.ascontiguousarray(B)
        actions = self.lookup_actions(Y, B)
        hit = actions is not None
        if hit:
            _add(self.stats, "action_cache_core_hits")
            _add(info, "action_cache_core_hits")
        else:
            reason = self._pending_rebase_reason or ("direct_ablation" if not self.enable_updates else "unmatched_frame_or_point")
            actions = self.dense_actions(A, B, purpose="action_rebase", info=info)
            _add(self.stats, "direct_action_rebase_calls")
            _add(self.stats, "direct_action_rebase_directions", len(B))
            _add(self.stats["action_rebase_reasons"], reason)
            _add(info, "direct_action_rebase_columns", len(B))
            if self._trusted(Y, B):
                self.store_actions(Y, B, actions, origin="direct_rebase")
            else:
                self.invalidate("untrusted_external_core")
        result = self.core_from_actions(Y, B, S, actions, info=info)
        if self.diagnostic:
            start = time.perf_counter()
            direct = self.dense_actions(A, B, purpose="diagnostic_direct_action")
            reference = self.core_from_actions(Y, B, S, direct)
            action_error = float(np.linalg.norm(actions-direct)/max(1.0, np.linalg.norm(direct)))
            core_error = float(np.linalg.norm(result-reference)/max(1.0, np.linalg.norm(reference)))
            self.stats["diagnostic_seconds"] += time.perf_counter()-start
            self.stats["diagnostic_max_action_relative_error"] = max(self.stats["diagnostic_max_action_relative_error"], action_error)
            self.stats["diagnostic_max_core_relative_error"] = max(self.stats["diagnostic_max_core_relative_error"], core_error)
            self.stats["diagnostic_error_history"].append({"cache_hit": hit, "action_relative_error": action_error, "core_relative_error": core_error})
            _add(self.stats, "diagnostic_action_checks")
            if not np.isfinite(action_error+core_error) or action_error > 1e-9 or core_error > 1e-9:
                raise ArithmeticError(f"Fixed-A action update diagnostic failed: action={action_error}, core={core_error}")
        _add(self.stats, "current_core_calls")
        _add(self.stats, "exact_current_core_columns", len(B))
        _add(info, "dense_current_core")
        _add(info, "exact_current_core_columns", len(B))
        return result
