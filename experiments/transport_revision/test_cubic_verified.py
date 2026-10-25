"""Small deterministic tests; no optimization benchmarks or timing claims."""

import math
import unittest

import numpy as np
from scipy.sparse.linalg import aslinearoperator

from cubic_verified import solve_cubic_subproblem


def model(H, h, sigma, v):
    return float(h @ v + 0.5 * v @ H @ v + sigma * np.linalg.norm(v) ** 3 / 6.0)


class CubicVerifiedTests(unittest.TestCase):
    def assert_kkt(self, H, h, sigma, v, lam, info):
        actual_shift = 0.5 * sigma * np.linalg.norm(v)
        scale = 1 + np.linalg.norm(h) + np.linalg.norm(H, 2) * np.linalg.norm(v) + actual_shift * np.linalg.norm(v)
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.linalg.norm(H @ v + h + actual_shift * v), 2e-9 * scale)
        self.assertLessEqual(abs(lam - actual_shift), 2e-9 * (1 + lam + actual_shift))
        self.assertGreaterEqual(np.linalg.eigvalsh(H + actual_shift * np.eye(len(h)))[0],
                                -2e-9 * (1 + np.linalg.norm(H, 2) + actual_shift))
        self.assertLessEqual(model(H, h, sigma, v), 2e-10 * scale)

    def test_known_indefinite_hard_case(self):
        H, h, sigma = np.diag([-2.0, 3.0]), np.array([0.0, 1.0]), 2.0
        v, lam, info = solve_cubic_subproblem(H, h, sigma)
        self.assertTrue(info["hard_case"])
        self.assertAlmostEqual(lam, 2.0, places=12)
        self.assertAlmostEqual(v[1], -1.0 / 5.0, places=12)
        self.assertAlmostEqual(abs(v[0]), math.sqrt(99.0) / 5.0, places=12)
        self.assertAlmostEqual(model(H, h, sigma, v), -43.0 / 30.0, places=12)
        self.assert_kkt(H, h, sigma, v, lam, info)

    def test_zero_gradient_and_singular_psd(self):
        cases = [
            (np.diag([1.0, 3.0]), np.zeros(2), 2.0),
            (np.diag([0.0, 3.0]), np.zeros(2), 2.0),
            (np.zeros((2, 2)), np.zeros(2), 2.0),
            (np.diag([-2.0, 3.0]), np.zeros(2), 2.0),
            (np.diag([0.0, 2.0]), np.array([1.0, 0.0]), 2.0),
            (np.diag([0.0, 2.0]), np.array([0.0, 1.0]), 2.0),
            (np.zeros((2, 2)), np.array([1.0, -2.0]), 0.1),
        ]
        for H, h, sigma in cases:
            with self.subTest(H=H.tolist(), h=h.tolist()):
                result = solve_cubic_subproblem(H, h, sigma)
                self.assert_kkt(H, h, sigma, *result)
        v, _, _ = solve_cubic_subproblem(np.diag([-2.0, 3.0]), np.zeros(2), 2.0)
        self.assertAlmostEqual(np.linalg.norm(v), 2.0, places=12)
        v, _, _ = solve_cubic_subproblem(np.diag([0.0, 2.0]), [1.0, 0.0], 2.0)
        np.testing.assert_allclose(v, [-1.0, 0.0], atol=1e-11)

    def test_multiple_lowest_and_rotated_hard_case(self):
        rng = np.random.default_rng(921)
        D, h = np.diag([-2.0, -2.0, 3.0]), np.array([0.0, 0.0, 1.0])
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        for H, g in ((D, h), (Q @ D @ Q.T, Q @ h)):
            v, lam, info = solve_cubic_subproblem(H, g, 2.0)
            self.assertTrue(info["hard_case"])
            self.assert_kkt(H, g, 2.0, v, lam, info)
            self.assertAlmostEqual(model(H, g, 2.0, v), -43.0 / 30.0, places=11)

    def test_regular_near_hard_case_and_scaling(self):
        H, h = np.diag([-2.0, 3.0]), np.array([1e-8, 1.0])
        v, lam, info = solve_cubic_subproblem(H, h, 2.0)
        self.assertFalse(info["hard_case"])
        self.assertGreater(lam, 2.0)
        self.assert_kkt(H, h, 2.0, v, lam, info)
        for scale in (1e-6, 1e6):
            vv, ll, ii = solve_cubic_subproblem(scale * H, scale * h, scale * 2.0)
            self.assert_kkt(scale * H, scale * h, scale * 2.0, vv, ll, ii)
            np.testing.assert_allclose(vv, v, atol=2e-7, rtol=2e-7)

    def test_generic_global_model_comparisons(self):
        rng = np.random.default_rng(318)
        for method in ("auto", "bisection"):
            for _ in range(6):
                matrix = rng.normal(size=(4, 4))
                H = 0.5 * (matrix + matrix.T)
                h = rng.normal(size=4)
                sigma = float(rng.uniform(0.2, 4.0))
                v, lam, info = solve_cubic_subproblem(H, h, sigma, method=method)
                self.assert_kkt(H, h, sigma, v, lam, info)
                competitors = rng.normal(size=(80, 4))
                competitors *= rng.uniform(0.0, 3.0 * max(1.0, np.linalg.norm(v)), size=(80, 1))
                competitors = np.vstack((competitors, -v, 0.5 * v, 1.5 * v, np.zeros(4)))
                best = model(H, h, sigma, v)
                self.assertTrue(all(best <= model(H, h, sigma, w) + 1e-8 for w in competitors))

    def test_api_linear_operator_and_explicit_failure(self):
        H, h = np.diag([1.0, 3.0]), np.array([2.0, -1.0])
        pair = solve_cubic_subproblem(H, h, 1.0, return_info=False)
        triple = solve_cubic_subproblem(aslinearoperator(H), h, 1.0, record_history=True)
        self.assertEqual(len(pair), 2)
        self.assertEqual(len(triple), 3)
        np.testing.assert_allclose(pair[0], triple[0])
        self.assertTrue(triple[2]["history"])
        with self.assertRaises(RuntimeError):
            solve_cubic_subproblem(H, h, 1.0, method="bisection", max_iter=1)
        for bad_sigma in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                solve_cubic_subproblem(H, h, bad_sigma)
        empty = solve_cubic_subproblem(np.empty((0, 0)), [], 1.0)
        self.assertEqual(empty[0].size, 0)
        self.assertTrue(empty[2]["converged"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
