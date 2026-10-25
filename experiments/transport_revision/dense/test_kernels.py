"""Small exact/numerical checks for general dense kernels; no benchmarks."""

from pathlib import Path
import sys
import unittest

import numpy as np
import sympy as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code-grassmann"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsncn import grassmann as geometry
from dense.kernels import DenseOracle, core, geometry_terms, hess_action_batch, objective, validate_matrix


def direct_hvp(A, Y, B, S):
    actions = []
    for U in B:
        W = -2.0 * (A @ U) + 2.0 * (U @ S)
        actions.append(W - Y @ (Y.T @ W))
    return np.array(actions).reshape(B.shape)


def direct_core(A, Y, B, S):
    H = direct_hvp(A, Y, B, S)
    flat = B.reshape(len(B), Y.size)
    result = flat @ H.reshape(len(B), Y.size).T
    return 0.5 * (result + result.T)


class DenseKernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(45019)
        matrix = rng.normal(size=(19, 19))
        self.A = 0.5 * (matrix + matrix.T)
        self.Y, _ = np.linalg.qr(rng.normal(size=(19, 3)))
        self.B = rng.normal(size=(7, 19, 3))
        self.U = self.B - np.matmul(self.Y[None], np.matmul(self.Y.T[None], self.B))
        self.S = self.Y.T @ self.A @ self.Y

    def test_sympy_exact_gradient_hessian_and_projection_identity(self):
        Y = sp.Matrix([[sp.Rational(3, 5), 0], [sp.Rational(4, 5), 0],
                       [0, sp.Rational(5, 13)], [0, sp.Rational(12, 13)]])
        A = sp.Matrix([[4, 1, 2, 3], [1, -2, 1, -1], [2, 1, 3, 2], [3, -1, 2, 1]])
        P = sp.eye(4)-Y*Y.T
        raw = [sp.Matrix([[1, 2], [0, 1], [2, -1], [3, 2]]),
               sp.Matrix([[2, -1], [1, 3], [-2, 0], [1, 1]])]
        S = Y.T*A*Y
        grad = -2*P*A*Y
        U, V = (P*B for B in raw)
        t = sp.symbols('t', real=True)
        Z = Y+t*U
        f = -sp.trace(Z.T*A*Z)
        inner = lambda L, R: sp.trace(L.T*R)
        self.assertEqual(sp.simplify(sp.diff(f, t).subs(t, 0)-inner(grad, U)), 0)
        grad_curve = -2*(sp.eye(4)-Z*Z.T)*A*Z
        hess = P*(-2*A*U+2*U*S)
        self.assertEqual(sp.simplify(P*grad_curve.diff(t).subs(t, 0)-hess), sp.zeros(4, 2))
        hess_v = P*(-2*A*V+2*V*S)
        self.assertEqual(sp.simplify(inner(U, hess_v)-inner(hess, V)), 0)
        exact_core = sp.zeros(2, 2)
        for i, Bi in enumerate(raw):
            for j, Bj in enumerate(raw):
                Wj = -2*A*Bj+2*Bj*S
                correction = inner(Y.T*Bi, Y.T*Wj)
                self.assertEqual(sp.simplify(inner(Bi, P*Wj)-(inner(Bi, Wj)-correction)), 0)
                exact_core[i, j] = inner(Bi, P*Wj)
        exact_core = (exact_core+exact_core.T)/2
        to_np = lambda M: np.array(M, dtype=float)
        state, value = geometry_terms(to_np(A), to_np(Y))
        self.assertIsInstance(state, geometry.GeometryState)
        np.testing.assert_allclose(state.grad, to_np(grad), atol=1e-14, rtol=1e-14)
        self.assertAlmostEqual(value, float(-sp.trace(S)), places=13)
        np.testing.assert_allclose(hess_action_batch(to_np(A), to_np(Y), to_np(U)[None], S=to_np(S)),
                                   to_np(hess)[None], atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(core(to_np(A), to_np(Y), np.stack([to_np(B) for B in raw]), S=to_np(S), chunk_size=1),
                                   to_np(exact_core), atol=1e-12, rtol=2e-14)

    def test_dense_direct_formula_chunks_and_arbitrary_frames(self):
        for B in (self.B, self.U):
            expected_hvp = direct_hvp(self.A, self.Y, B, self.S)
            expected_core = direct_core(self.A, self.Y, B, self.S)
            before = [x.copy() for x in (self.A, self.Y, B, self.S)]
            for chunk in (1, 2, 4, 20, None):
                oracle = DenseOracle(chunk)
                H = oracle.hess_action_batch(self.A, self.Y, B, S=self.S)
                C = oracle.core(self.A, self.Y, B, S=self.S)
                np.testing.assert_allclose(H, expected_hvp, atol=2e-13, rtol=3e-13)
                np.testing.assert_allclose(C, expected_core, atol=2e-12, rtol=3e-13)
                np.testing.assert_array_equal(C, C.T)
                self.assertTrue(C.flags.c_contiguous)
                for original, current in zip(before, (self.A, self.Y, B, self.S)):
                    np.testing.assert_array_equal(original, current)

    def test_projection_correction_is_not_omitted(self):
        W = -2*np.matmul(self.A[None], self.B)+2*np.matmul(self.B, self.S)
        flat = self.B.reshape(len(self.B), -1)
        wrong = flat @ W.reshape(len(self.B), -1).T
        wrong = 0.5*(wrong+wrong.T)
        correct = core(self.A, self.Y, self.B, S=self.S, chunk_size=2)
        self.assertGreater(np.linalg.norm(correct-wrong), 1.0)
        np.testing.assert_allclose(correct, direct_core(self.A, self.Y, self.B, self.S), atol=2e-12)

    def test_central_differences_along_qr_retraction(self):
        direction = self.U[0] / np.linalg.norm(self.U[0])
        state, _ = geometry_terms(self.A, self.Y)
        predicted_d1 = float(np.sum(state.grad*direction))
        predicted_hvp = hess_action_batch(self.A, self.Y, direction[None], S=state.S)[0]
        errors = []
        for step in (1e-3, 3e-4, 1e-4):
            plus = geometry.retraction_qr(self.Y, step*direction)
            minus = geometry.retraction_qr(self.Y, -step*direction)
            d1 = (objective(self.A, plus)-objective(self.A, minus))/(2*step)
            gp = geometry_terms(self.A, plus)[0].grad
            gm = geometry_terms(self.A, minus)[0].grad
            dgrad = geometry.proj_Y(self.Y, (gp-gm)/(2*step))
            errors.append(np.linalg.norm(dgrad-predicted_hvp))
            self.assertLess(abs(d1-predicted_d1), 2e-5*step/1e-3)
        self.assertLess(errors[-1], 2e-7)
        self.assertLess(errors[-1], errors[0]/20)

    def test_tangency_and_hessian_self_adjointness(self):
        state, _ = geometry_terms(self.A, self.Y)
        H = hess_action_batch(self.A, self.Y, self.U, S=state.S, chunk_size=2)
        np.testing.assert_allclose(self.Y.T @ state.grad, 0, atol=2e-14)
        np.testing.assert_allclose(np.matmul(self.Y.T[None], H), 0, atol=5e-14)
        raw = self.U.reshape(len(self.U), -1) @ H.reshape(len(H), -1).T
        np.testing.assert_allclose(raw, raw.T, atol=3e-12, rtol=2e-13)

    def test_coordinate_permutation_and_orthogonal_rotation(self):
        rng = np.random.default_rng(2026)
        Q, _ = np.linalg.qr(rng.normal(size=self.A.shape))
        perm = rng.permutation(len(self.A))
        transforms = [np.eye(len(self.A))[perm], Q]
        expected = core(self.A, self.Y, self.B, S=self.S)
        old = geometry_terms(self.A, self.Y)
        H = hess_action_batch(self.A, self.Y, self.U, S=self.S)
        for transform in transforms:
            A = transform @ self.A @ transform.T
            Y = transform @ self.Y
            B = np.matmul(transform[None], self.B)
            U = np.matmul(transform[None], self.U)
            state, f = geometry_terms(A, Y)
            self.assertAlmostEqual(f, old[1], places=12)
            np.testing.assert_allclose(state.grad, transform @ old[0].grad, atol=8e-14)
            np.testing.assert_allclose(core(A, Y, B, S=state.S, chunk_size=2), expected, atol=3e-12)
            np.testing.assert_allclose(hess_action_batch(A, Y, U, S=state.S), np.matmul(transform[None], H), atol=2e-13)

    def test_offdiagonal_entries_and_current_A_are_used(self):
        oracle = DenseOracle(2)
        diagonal = np.diag(np.diag(self.A))
        full_state, f_full = oracle.geometry_terms(self.A, self.Y)
        diagonal_state, f_diagonal = oracle.geometry_terms(diagonal, self.Y)
        self.assertGreater(abs(f_full-f_diagonal), 1e-3)
        self.assertGreater(np.linalg.norm(full_state.grad-diagonal_state.grad), 1.0)
        dense_core = oracle.core(self.A, self.Y, self.U, S=full_state.S)
        diagonal_core = oracle.core(diagonal, self.Y, self.U, S=diagonal_state.S)
        self.assertGreater(np.linalg.norm(dense_core-diagonal_core), 1.0)
        changed = self.A.copy()
        oracle.geometry_terms(changed, self.Y)
        changed[0, 1] += 3
        changed[1, 0] += 3
        state, _ = oracle.geometry_terms(changed, self.Y)
        np.testing.assert_allclose(oracle.core(changed, self.Y, self.U, S=state.S), direct_core(changed, self.Y, self.U, state.S), atol=2e-12)

    def test_noncontiguous_inputs_and_optional_current_S(self):
        larger = np.stack([self.B, self.B], axis=1).reshape(14, 19, 3)
        for B in (larger[::2], self.B[:, ::-1], np.asfortranarray(self.B)):
            np.testing.assert_allclose(core(self.A, self.Y, B, S=self.S, chunk_size=2), direct_core(self.A, self.Y, B, self.S), atol=2e-12)
        np.testing.assert_allclose(core(self.A, self.Y, self.B, chunk_size=2), direct_core(self.A, self.Y, self.B, self.S), atol=2e-12)
        np.testing.assert_allclose(hess_action_batch(self.A, self.Y, self.U), direct_hvp(self.A, self.Y, self.U, self.S), atol=2e-13)

    def test_indefinite_curvature_is_preserved(self):
        angle = 0.37
        Q = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        A = Q @ np.diag([1., 4., 7.]) @ Q.T
        Y = Q[:, 1:2]
        B = np.stack([Q[:, 0:1], Q[:, 2:3]])
        np.testing.assert_allclose(core(A, Y, B), np.diag([6., -6.]), atol=1e-13)

    def test_memory_policy_and_counters(self):
        ell, n, p = self.B.shape
        budget = 8*ell*p*p + 8*(4*n*p+4*p*p+4*ell)*2
        oracle = DenseOracle(chunk_size=99, workspace_bytes=budget)
        info = {}
        oracle.core(self.A, self.Y, self.B, S=self.S, info=info)
        self.assertEqual(oracle.stats['maximum_effective_chunk_size'], 2)
        self.assertLessEqual(oracle.stats['peak_explicit_workspace_estimate_bytes'], budget)
        self.assertEqual(info['dense_core_chunks'], 4)
        self.assertEqual(info['dense_A_matmul_calls'], 4)
        self.assertEqual(info['dense_A_rhs_columns'], ell*p)
        self.assertEqual(info['exact_current_core_columns'], ell)
        self.assertEqual(info['projection_correction_columns'], ell)
        self.assertEqual(info.get('projected_hvp_columns_materialized', 0), 0)
        with self.assertRaisesRegex(ValueError, 'too small'):
            DenseOracle(workspace_bytes=1).core(self.A, self.Y, self.B, S=self.S)

    def test_empty_inputs_validation_and_nonfinite_errors(self):
        np.testing.assert_array_equal(core(self.A, self.Y, np.zeros((0, 19, 3)), S=self.S), np.zeros((0, 0)))
        self.assertEqual(hess_action_batch(self.A, self.Y, np.zeros((0, 19, 3)), S=self.S).shape, (0, 19, 3))
        self.assertIs(validate_matrix(self.A), self.A)
        with self.assertRaisesRegex(ValueError, 'dense square'):
            objective(np.diag(self.A), self.Y)
        asymmetric = self.A.copy()
        asymmetric[0, 1] += 1
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            validate_matrix(asymmetric)
        with self.assertRaisesRegex(ValueError, 'real'):
            objective(self.A.astype(complex), self.Y)
        bad = self.B.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            core(self.A, self.Y, bad, S=self.S)
        with self.assertRaisesRegex(ValueError, 'shape'):
            core(self.A, self.Y, self.B[:, :-1], S=self.S)


if __name__ == '__main__':
    unittest.main(verbosity=2)
