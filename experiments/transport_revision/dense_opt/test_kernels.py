"""Exact-formula and finite-difference tests, intentionally not a benchmark."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code-grassmann"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense.kernels import DenseOracle
from dense import test_kernels as original_tests
from dense_opt.kernels import DenseBlockOracle, core


class DenseBlockTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(59013)
        A = rng.normal(size=(37, 37))
        self.A = (A+A.T)/2
        self.Y, _ = np.linalg.qr(rng.normal(size=(37, 4)))
        self.B = rng.normal(size=(13, 37, 4))
        self.S = self.Y.T @ self.A @ self.Y

    def test_old_oracle_and_direct_hvp_agree_for_all_block_sizes(self):
        for frame in (self.B, self.B-self.Y[None] @ (self.Y.T[None] @ self.B)):
            old = DenseOracle(chunk_size=3).core(self.A, self.Y, frame, S=self.S)
            direct = original_tests.direct_core(self.A, self.Y, frame, self.S)
            for chunk in (1, 3, 8, 13, 100, None):
                oracle = DenseBlockOracle(chunk_size=chunk)
                info = {}
                actual = oracle.core(self.A, self.Y, frame, S=self.S, info=info)
                np.testing.assert_allclose(actual, old, atol=5e-12, rtol=3e-13)
                np.testing.assert_allclose(actual, direct, atol=5e-12, rtol=3e-13)
                np.testing.assert_array_equal(actual, actual.T)
                self.assertEqual(info["exact_current_core_columns"], len(frame))
                self.assertEqual(info["projection_correction_columns"], len(frame))
                self.assertEqual(info["dense_A_rhs_columns"], len(frame)*frame.shape[-1])
                if chunk is None or chunk >= len(frame):
                    self.assertEqual(info["dense_A_matmul_calls"], 1)

    def test_fallback_noncontiguous_frames(self):
        for frame in (self.B[::2], self.B[:, ::-1], np.asfortranarray(self.B)):
            oracle = DenseBlockOracle(chunk_size=None)
            before = [x.copy() for x in (self.A, self.Y, frame, self.S)]
            actual = oracle.core(self.A, self.Y, frame, S=self.S)
            expected = original_tests.direct_core(self.A, self.Y, frame, self.S)
            np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=3e-13)
            self.assertEqual(oracle.stats["noncontiguous_core_fallbacks"], 1)
            for old, current in zip(before, (self.A, self.Y, frame, self.S)):
                np.testing.assert_array_equal(current, old)

    def test_small_workspace_forces_multiple_blocks(self):
        ell, n, p = self.B.shape
        fixed = 8*ell*p*p
        per_direction = 8*(3*n*p+4*p*p+4*ell)
        oracle = DenseBlockOracle(workspace_bytes=fixed+2*per_direction)
        info = {}
        actual = oracle.core(self.A, self.Y, self.B, S=self.S, info=info)
        expected = original_tests.direct_core(self.A, self.Y, self.B, self.S)
        np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=3e-13)
        self.assertEqual(oracle.stats["maximum_effective_chunk_size"], 2)
        self.assertEqual(info["dense_core_chunks"], 7)
        self.assertLessEqual(oracle.stats["peak_explicit_workspace_estimate_bytes"], oracle.workspace_bytes)

    def test_nominal_paper_shape_fits_one_block_without_allocating_data(self):
        oracle = DenseBlockOracle()
        self.assertEqual(oracle._core_chunk_limit(320, 768, 20), 320)
        self.assertLessEqual(oracle.stats["peak_explicit_workspace_estimate_bytes"], 128*1024**2)

    def test_noncontiguous_workspace_accounts_for_retained_buffers(self):
        frame = np.asfortranarray(self.B)
        ell, n, p = frame.shape
        fixed = 8*ell*p*p
        per_direction = 8*(6*n*p+4*p*p+4*ell)
        budget = fixed+2*per_direction
        oracle = DenseBlockOracle(workspace_bytes=budget)
        info = {}
        actual = oracle.core(self.A, self.Y, frame, S=self.S, info=info)
        expected = original_tests.direct_core(self.A, self.Y, frame, self.S)
        np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=3e-13)
        self.assertEqual(oracle.stats["maximum_effective_chunk_size"], 2)
        self.assertEqual(info["dense_core_chunks"], 7)
        self.assertLessEqual(oracle.stats["peak_explicit_workspace_estimate_bytes"], budget)
        with self.assertRaisesRegex(ValueError, "too small"):
            DenseBlockOracle(workspace_bytes=fixed+per_direction-1).core(
                self.A, self.Y, frame, S=self.S)

    def test_reuse_original_exact_and_differential_validations(self):
        # Re-run the existing hand-built rational SymPy reference, current-A
        # perturbation, orthogonal invariance and finite-difference checks with
        # only the core implementation replaced. Public HVP remains unchanged.
        methods = (
            "test_sympy_exact_gradient_hessian_and_projection_identity",
            "test_projection_correction_is_not_omitted",
            "test_central_differences_along_qr_retraction",
            "test_tangency_and_hessian_self_adjointness",
            "test_coordinate_permutation_and_orthogonal_rotation",
            "test_offdiagonal_entries_and_current_A_are_used",
            "test_indefinite_curvature_is_preserved",
            "test_empty_inputs_validation_and_nonfinite_errors",
        )
        with patch.object(original_tests, "DenseOracle", DenseBlockOracle), patch.object(original_tests, "core", core):
            for name in methods:
                with self.subTest(original_test=name):
                    case = original_tests.DenseKernelTests(methodName=name)
                    case.setUp()
                    getattr(case, name)()
                    case.tearDown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
