"""Small exact/differential/provenance checks, not performance benchmarks."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import unittest
from unittest.mock import patch

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(REPO/"code-grassmann"))

import numpy as np
import sympy as sp

from dense_action_cache.kernels import ActionCacheOracle
from dense_action_cache.geometry import ActionCacheGeometry
from dense_opt.kernels import DenseBlockOracle
from geometry_isometric_fused import GeometryHooks as ReferenceGeometry
from rsncn import grassmann


class ActionCacheTests(unittest.TestCase):
    diagnostic_evidence = []
    def setUp(self):
        self.rng = np.random.default_rng(382010)
        matrix = self.rng.normal(size=(23, 23))
        self.A = np.ascontiguousarray((matrix+matrix.T)/2)
        self.A.setflags(write=False)
        self.X, _ = np.linalg.qr(self.rng.normal(size=(23, 3)))
        self.oracle = ActionCacheOracle(chunk_size=20, diagnostic=True)
        self.hooks = ActionCacheGeometry(self.oracle)
        self.reference = ReferenceGeometry()
        self.geom, _ = self.oracle.geometry_terms(self.A, self.X)

    def fresh(self):
        left = self.hooks.random_tangent_metric_gaussian(np.random.default_rng(71), self.X, 5)
        right = self.reference.random_tangent_metric_gaussian(np.random.default_rng(71), self.X, 5)
        B = self.hooks.basis_from_candidates(self.X, self.geom.grad, left, 6)
        expected = self.reference.basis_from_candidates(self.X, self.geom.grad, right, 6)
        np.testing.assert_array_equal(B, expected)
        self.oracle.core(self.A, self.X, B, S=self.geom.S)
        return B, expected

    def moved(self):
        B, expected = self.fresh()
        eta = -0.03*self.geom.grad/np.linalg.norm(self.geom.grad)
        Y = self.hooks.retraction_qr(self.X, eta)
        expected_Y = self.reference.retraction_qr(self.X, eta)
        np.testing.assert_array_equal(Y, expected_Y)
        trans = self.hooks.transport_tangent_basis(Y, B[1:], ell=5)
        expected_trans = self.reference.transport_tangent_basis(expected_Y, expected[1:], ell=5)
        np.testing.assert_array_equal(trans, expected_trans)
        return B, Y, trans, expected_Y, expected_trans

    def check_core(self, Y, B):
        state, _ = self.oracle.geometry_terms(self.A, Y)
        actual = self.oracle.core(self.A, Y, B, S=state.S)
        expected = DenseBlockOracle().core(self.A, Y, B, S=state.S)
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=5e-12)
        actions = self.oracle.lookup_actions(Y, B)
        if actions is not None:
            np.testing.assert_allclose(actions, np.matmul(self.A[None], B), atol=3e-12, rtol=5e-12)
        return actual

    def test_sympy_fixed_linear_action_identities(self):
        A = sp.Matrix([[2, 1, -1], [1, 3, 2], [-1, 2, 5]])
        B = [sp.Matrix([[1, 2], [0, 1], [2, -1]]), sp.Matrix([[2, 0], [1, -2], [3, 1]])]
        M = sp.Matrix([[1, 2], [-1, 0], [2, 1]])/5
        C = [sp.Matrix([[1, 2], [0, 1]])/3, sp.Matrix([[2, 0], [1, -1]])/7]
        for frame, coeff in zip(B, C):
            self.assertEqual(A*(frame-M*coeff)-(A*frame-(A*M)*coeff), sp.zeros(3, 2))
        a = sp.Matrix([sp.Rational(3, 5), 0])
        s = sp.Rational(4, 5)
        u = sp.Matrix([[1, -2], [2, 1], [0, 1]])/3
        weighted = sum((a[i]*B[i] for i in range(2)), sp.zeros(3, 2))
        update = weighted/(s*(1+s))-u/s
        weighted_action = sum((a[i]*(A*B[i]) for i in range(2)), sp.zeros(3, 2))
        action_update = weighted_action/(s*(1+s))-(A*u)/s
        for i in range(2):
            self.assertEqual(A*(B[i]+a[i]*update)-(A*B[i]+a[i]*action_update), sp.zeros(3, 2))
        R = sp.eye(2)+a*a.T/(s*(1+s))
        self.assertEqual(R*a-a/s, sp.zeros(2, 1))

    def test_fixed_readonly_A_and_private_frame_contract(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            ActionCacheOracle().geometry_terms(self.A.copy(), self.X)
        copy = self.A.copy()
        copy.setflags(write=False)
        with self.assertRaisesRegex(ValueError, "fixed A"):
            self.oracle.geometry_terms(copy, self.X)
        B, _ = self.fresh()
        self.assertFalse(B.flags.writeable)
        self.assertIsNotNone(self.oracle.lookup_actions(self.X, B))
        self.assertIsNotNone(self.oracle.lookup_actions(self.X, B[1:], allow_leading_slice=True))
        self.assertIsNone(self.oracle.lookup_actions(self.X, B.copy()))
        self.assertIsNone(self.oracle.lookup_actions(self.X.copy(), B))
        self.assertIsNone(self.oracle.lookup_actions(self.X, B[::-1], allow_leading_slice=True))

    def test_shared_polar_and_rank_one_match_reference_geometry(self):
        _, Y, trans, expected_Y, expected_trans = self.moved()
        state, _ = self.oracle.geometry_terms(self.A, Y)
        B = self.hooks.build_transport_sketch_basis(Y, ell=6, grad=state.grad, transported=trans)
        expected = self.reference.build_transport_sketch_basis(expected_Y, ell=6, grad=state.grad, transported=expected_trans)
        np.testing.assert_array_equal(B, expected)
        self.check_core(Y, B)
        self.assertEqual(self.oracle.stats["polar_action_updates"], 1)
        self.assertEqual(self.oracle.stats["gradient_action_updates"], 1)
        self.assertEqual(self.oracle.stats["action_transport_dense_A_rhs_columns"], 3)
        self.assertEqual(self.oracle.stats["action_gradient_dense_A_rhs_columns"], 3)
        self.assertEqual(self.oracle.stats["action_cache_core_hits"], 1)
        self.assertLess(self.oracle.stats["diagnostic_max_action_relative_error"], 1e-12)

    def test_low_s_rebases_without_changing_geometry(self):
        _, Y, trans, expected_Y, expected_trans = self.moved()
        # Construct a horizontal normalized gradient with 0.9 energy in the
        # transported frame and 0.1 outside it, giving s^2 approximately 0.1.
        z = grassmann.proj_Y(Y, self.rng.normal(size=Y.shape)).reshape(-1)
        flat = trans.reshape(len(trans), -1)
        z -= (flat @ z) @ flat
        z /= np.linalg.norm(z)
        grad = (np.sqrt(0.9)*flat[0]+np.sqrt(0.1)*z).reshape(Y.shape)
        B = self.hooks.build_transport_sketch_basis(Y, ell=6, grad=grad, transported=trans)
        expected = self.reference.build_transport_sketch_basis(expected_Y, ell=6, grad=grad, transported=expected_trans)
        np.testing.assert_array_equal(B, expected)
        self.assertIsNone(self.oracle.lookup_actions(Y, B))
        self.check_core(Y, B)
        self.assertEqual(self.oracle.stats["action_rebase_reasons"]["rank_one_amplification_rebase"], 1)
        self.assertGreater(self.oracle.stats["max_rank_one_amplification_considered"], 2)
        self.assertEqual(self.oracle.stats["max_rank_one_amplification_applied"], 1)

    def test_rank_loss_and_truncated_or_unpaired_transport_rebase(self):
        _, Y, trans, expected_Y, expected_trans = self.moved()
        B = self.hooks.build_transport_sketch_basis(Y, ell=6, grad=trans[0], transported=trans)
        expected = self.reference.build_transport_sketch_basis(expected_Y, ell=6, grad=expected_trans[0], transported=expected_trans)
        np.testing.assert_array_equal(B, expected)
        self.check_core(Y, B)
        self.assertEqual(self.hooks.stats["rank_loss_fallback"], 1)
        self.assertEqual(self.oracle.stats["action_rebase_reasons"]["rank_loss_geometry_fallback"], 1)
        # Fresh establish then truncate; source is known but eigen-whitening
        # means old actions must not be propagated to the resulting rows.
        self.setUp()
        B, _ = self.fresh()
        Y = self.hooks.retraction_qr(self.X, -0.02*self.geom.grad)
        trans = self.hooks.transport_tangent_basis(Y, B[1:], ell=3)
        self.assertIsNone(self.oracle.lookup_actions(Y, trans))
        self.check_core(Y, trans)
        self.assertEqual(self.oracle.stats["action_rebase_reasons"]["transport_fallback"], 1)

    def test_rejected_identity_and_fresh_reset(self):
        B, _ = self.fresh()
        self.hooks.retraction_qr(self.X, -0.02*self.geom.grad)
        trans = self.hooks.transport_tangent_basis(self.X, B[1:], ell=5)
        np.testing.assert_array_equal(trans, B[1:])
        self.assertEqual(self.oracle.stats["rejected_identity_action_updates"], 1)
        built = self.hooks.build_transport_sketch_basis(self.X, ell=6, grad=self.geom.grad, transported=trans)
        self.check_core(self.X, built)
        candidates = self.hooks.random_tangent_metric_gaussian(self.rng, self.X, 5)
        fresh = self.hooks.basis_from_candidates(self.X, self.geom.grad, candidates, 6)
        self.assertIsNone(self.oracle.lookup_actions(self.X, fresh))
        self.check_core(self.X, fresh)
        self.assertEqual(self.oracle.stats["action_rebase_reasons"]["fresh_frame"], 2)

    def test_current_S_and_nonhorizontal_projection_are_not_cached(self):
        raw = self.rng.normal(size=(5, 23, 3))
        self.oracle.trust_frame(self.X, raw)
        first = self.oracle.core(self.A, self.X, raw, S=self.geom.S)
        altered_S = self.geom.S+np.eye(3)
        second = self.oracle.core(self.A, self.X, raw, S=altered_S)
        expected = DenseBlockOracle().core(self.A, self.X, raw, S=altered_S)
        np.testing.assert_allclose(second, expected, atol=3e-12, rtol=5e-12)
        self.assertGreater(np.linalg.norm(second-first), 1)

    def test_gradient_free_mixed_unknown_and_foreign_provenance(self):
        _, Y, trans, _, _ = self.moved()
        B = self.hooks.build_transport_sketch_basis(Y, ell=5, grad=None, transported=trans)
        self.assertIs(B, trans)
        self.check_core(Y, B)
        fresh = self.hooks.random_tangent_metric_gaussian(self.rng, Y, 1)
        mixed = self.hooks.build_transport_sketch_basis(Y, ell=6, grad=None, transported=trans, fresh_candidates=fresh)
        self.assertIsNone(self.oracle.lookup_actions(Y, mixed))
        self.check_core(Y, mixed)
        self.assertEqual(self.oracle.stats["action_rebase_reasons"]["mixed_or_unpaired_build"], 1)
        with self.assertRaisesRegex(ValueError, "paired QR"):
            self.hooks.transport_tangent_basis(Y.copy(), mixed, ell=6)
        self.setUp()
        B, _ = self.fresh()
        Y = self.hooks.retraction_qr(self.X, -0.01*self.geom.grad)
        unknown = self.hooks.transport_tangent_basis(Y, B[1:][::-1], ell=5)
        self.assertIsNone(self.oracle.lookup_actions(Y, unknown))
        self.check_core(Y, unknown)
        self.assertEqual(self.oracle.stats["action_rebase_reasons"]["transport_fallback"], 1)

    def test_noncontiguous_generic_core_and_direct_arm_zero_hits(self):
        oracle = ActionCacheOracle(chunk_size=2, enable_updates=False, diagnostic=True)
        raw = self.rng.normal(size=(5, 23, 3))[:, ::-1]
        old = raw.copy()
        for _ in range(2):
            actual = oracle.core(self.A, self.X, raw, S=self.geom.S)
            expected = DenseBlockOracle().core(self.A, self.X, raw, S=self.geom.S)
            np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=5e-12)
        np.testing.assert_array_equal(raw, old)
        self.assertEqual(oracle.stats.get("action_cache_core_hits", 0), 0)
        self.assertEqual(oracle.stats["direct_action_rebase_calls"], 2)

    def test_solver_rejection_retention_and_poor_rho_reset(self):
        from dense_action_cache import run_action_cache_experiments as wrapper
        solver = wrapper.protocol.solver
        for rho, reset, expect_identity in ((0.0, True, False), (0.0, False, True), (0.2, True, False)):
            opts = solver.SolverOptions(max_iter=4, sketch_size=6, random_state=903,
                                        transport_sketch=True, transport_refresh_every=8,
                                        include_gradient_in_sketch=True, transport_keep_gradient_direction=True,
                                        transport_reset_on_reject=reset,
                                        transport_bad_rho_tol=0.0 if expect_identity else 0.5)
            with wrapper.instrument_dense(validate=True, chunk_size=20) as (_, counts, checks, oracle):
                with patch.object(solver, "_stable_rho", return_value=(rho, True)):
                    _, history = solver.solve_grassmann_nystrom_cubic(self.A, self.X, opts)
            identity = checks["geometry_optimization"]["rejected_identity_transports"]
            if expect_identity:
                self.assertGreater(identity, 0)
                self.assertGreater(oracle.stats.get("action_cache_core_hits", 0), 0)
            else:
                self.assertEqual(identity, 0)
                self.assertEqual(oracle.stats.get("action_cache_core_hits", 0), 0)
                self.assertEqual(counts["fresh_basis"], len(history["f"])-1)
            self.diagnostic_evidence.append({"test": "controlled_acceptance_branch", "rho": rho,
                                             "reset_on_reject": reset, "expect_identity": expect_identity,
                                             "history": history, "counts": dict(counts), "oracle_stats": oracle.stats})

    def test_small_solver_ablations_and_fresh_enabled_disabled_equivalence(self):
        from dense_action_cache import run_action_cache_experiments as wrapper
        case = dict(name="diagnostic_small", n=32, p=3, ell=6,
                    matrix="dense_rotated_gapped", spectrum="gapped", tail_top=18,
                    max_iter=12, time_limit=60, targets=[0.01, 0.0001, 0.000001])
        problem = wrapper.protocol.build_problem(case, 901)
        runs = {}
        for T, enabled, control in ((1, False, False), (1, True, False),
                                    (8, False, False), (8, True, False), (1, False, True)):
            run_case = dict(case, action_cache_enabled=enabled, optimized_direct_control=control)
            runs[(T, enabled, control)] = wrapper.run_one(problem, run_case, 901, T, validate=True, chunk_size=20)
        np.testing.assert_array_equal(runs[(1, False, False)]["history"]["f"], runs[(1, True, False)]["history"]["f"])
        for key in ((1, True, False), (1, False, True)):
            np.testing.assert_allclose(runs[(1, False, False)]["history"]["f"], runs[key]["history"]["f"], atol=2e-11, rtol=2e-12)
        np.testing.assert_allclose(runs[(8, False, False)]["history"]["f"], runs[(8, True, False)]["history"]["f"], atol=2e-10, rtol=2e-11)
        cached = runs[(8, True, False)]["dense_oracle_stats"]
        self.assertGreater(cached.get("action_cache_core_hits", 0), 0)
        self.assertEqual(cached["diagnostic_action_checks"], runs[(8, True, False)]["iterations"])
        self.assertLess(cached["diagnostic_max_action_relative_error"], 1e-11)
        self.assertEqual(runs[(1, False, True)]["core"], "dense_large_block_current")
        self.assertEqual(runs[(1, False, True)]["dense_oracle_stats"]["kernel"], "general_dense_current_hessian_large_block")
        self.diagnostic_evidence.append({"test": "small_solver_ablation", "seed": 901,
                                         "runs": {str(key): value for key, value in runs.items()},
                                         "cache_direct_T8_max_abs_objective_difference": float(np.max(np.abs(
                                             np.asarray(runs[(8, False, False)]["history"]["f"])-
                                             np.asarray(runs[(8, True, False)]["history"]["f"]))))})
        with self.assertRaisesRegex(ValueError, "requires T1"):
            wrapper.run_one(problem, dict(case, optimized_direct_control=True), 901, 8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"diagnostics"/"unit_checks.json")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ActionCacheTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable, *sys.argv], "python": sys.version,
               "performance_evidence": False, "benchmark_seeds_used": False,
               "tests_run": result.testsRun, "successful": result.wasSuccessful(),
               "errors": [(str(case), detail) for case, detail in result.errors],
               "failures": [(str(case), detail) for case, detail in result.failures],
               "local_python_source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in sorted(HERE.glob("*.py"))},
               "evidence": ActionCacheTests.diagnostic_evidence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
