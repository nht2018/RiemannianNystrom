"""Large-block implementation of the same general dense current-Hessian core.

Only contraction scheduling changes relative to ``dense.kernels``. Every
direction is multiplied by the actual current dense A; no previous action,
curvature, generating eigenvector, or spectrum is retained. The exact
nonhorizontal projection correction is included for every supplied frame.

The explicit workspace estimate excludes input arrays, returned arrays,
dtype-conversion copies and opaque BLAS packing/scratch, as in the original
dense oracle. It is not an RSS limit. Native float64 inputs are recommended.
"""

from __future__ import annotations

import numpy as np

from dense.kernels import DEFAULT_WORKSPACE_BYTES, DenseOracle, _add


class DenseBlockOracle(DenseOracle):
    """Use the largest allowed core block, normally the whole supplied frame.

    A larger GEMM avoids rescanning/repacking the full left frame once per
    sixteen right columns. The algorithm and arithmetic formula are unchanged.
    ``chunk_size`` is still an optional upper bound in frame directions; use
    None (or a sufficiently large integer) for budget-selected blocking.
    Non-C-contiguous frames safely use the original bounded-memory algorithm.
    Objective, gradient and public HVP hooks are inherited without changes.
    """

    def __init__(self, chunk_size=None, workspace_bytes=DEFAULT_WORKSPACE_BYTES):
        super().__init__(chunk_size=chunk_size, workspace_bytes=workspace_bytes)
        self.stats["kernel"] = "general_dense_current_hessian_large_block"
        self.stats["core_formula"] = "sym(Bflat@Wflat.T-E@F.T), W=2*(B*S-A*B), E=Y.T*B, F=Y.T*W"

    def _core_chunk_limit(self, ell, n, p):
        if ell == 0:
            return 0
        # Peak n*p buffers: packed B, A@packed B, and W=B@S. Packing and
        # product are deleted before contractions/projection intermediates.
        # E is the fixed full-frame correction. Four p*p/ell buffers per
        # direction conservatively cover F, correction/contraction blocks,
        # finite-check masks and row symmetrization (the latter are smaller).
        fixed = 8 * ell * p * p
        per_direction = 8 * (3 * n * p + 4 * p * p + 4 * ell)
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

    def _chunk_limit(self, ell, n, p, *, core):
        if not core:
            return super()._chunk_limit(ell, n, p, core=False)
        if ell == 0:
            return 0
        # The inherited noncontiguous fallback retains its previous yielded W
        # and final left-frame packing while constructing the next W. Reserve
        # six n*p buffers, including room for the temporary finiteness mask;
        # the optimized C-contiguous path uses _core_chunk_limit instead.
        fixed = 8 * ell * p * p
        per_direction = 8 * (6 * n * p + 4 * p * p + 4 * ell)
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

    def core(self, A, Y, B, *, S=None, info=None):
        """Assemble exactly the current dense core, including projection."""
        A, Y, B, S = self._frame_inputs(A, Y, B, S)
        if not B.flags.c_contiguous:
            # Do not silently allocate an additional full-frame flatten copy.
            _add(self.stats, "noncontiguous_core_fallbacks")
            return super().core(A, Y, B, S=S, info=info)

        ell, n, p = B.shape
        chunk = self._core_chunk_limit(ell, n, p)
        result = np.empty((ell, ell), dtype=float)
        if chunk:
            E = np.matmul(Y.T[None], B).reshape(ell, p*p)
            flat = B.reshape(ell, n*p)
            for start in range(0, ell, chunk):
                stop = min(start+chunk, ell)
                block = B[start:stop]
                packed = np.ascontiguousarray(block.transpose(1, 0, 2)).reshape(n, (stop-start)*p)
                product = self._multiply(A, packed, "core", info)
                W = np.matmul(block, S)
                W -= product.reshape(n, stop-start, p).transpose(1, 0, 2)
                W *= 2.0
                del packed, product
                if not np.all(np.isfinite(W)):
                    raise ValueError("Current Hessian action is nonfinite")
                _add(self.stats, "core_chunks")
                _add(info, "dense_core_chunks")
                right = W.reshape(stop-start, n*p).T
                np.matmul(flat, right, out=result[:, start:stop])
                F = np.matmul(Y.T[None], W).reshape(stop-start, p*p)
                result[:, start:stop] -= E @ F.T
                _add(info, "unprojected_action_columns_materialized", stop-start)
                _add(info, "projection_correction_columns", stop-start)
                # Release all views retaining W before packing the next block;
                # otherwise a previous generator yield would add a fourth
                # live n*p buffer and invalidate the three-buffer estimate.
                del right, F, W
            # Preserve the exact symmetry of the returned array without a
            # second full ell-by-ell result array or ell-by-ell Boolean mask.
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


def core(A, Y, B, *, S=None, info=None, chunk_size=None,
         workspace_bytes=DEFAULT_WORKSPACE_BYTES):
    return DenseBlockOracle(chunk_size, workspace_bytes).core(A, Y, B, S=S, info=info)
