"""Correctness/equivalence tests for fused allocations; no speed assertions."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

import numpy as np

import test_geometry_isometric as reference_tests
from geometry_isometric import make_hooks as make_reference_hooks
from geometry_isometric import polar_transport_batch as reference_polar_transport
from geometry_isometric_fused import HOOK_NAMES, make_hooks, polar_transport_batch


@contextmanager
def patched_geometry(hooks, diagnostics=None):
    """Patch all six names; diagnostics call the SELECTED Hessian hook."""
    solver = reference_tests.solver
    saved = {name: getattr(solver, name) for name in HOOK_NAMES}
    cubic = solver.solve_cubic_subproblem
    try:
        for name in HOOK_NAMES:
            # The preserved v1 reference has no buffered Hessian hook.
            setattr(solver, name, getattr(hooks, name, saved[name]))
        solver.solve_cubic_subproblem = reference_tests.solve_cubic_subproblem
        selected_hess = solver.hess_action_batch
        if diagnostics is not None:
            def checked(a, Y, basis, **kwargs):
                flat = basis.reshape(len(basis), -1)
                diagnostics.append((np.linalg.norm(flat @ flat.T - np.eye(len(flat)), 2),
                                    np.linalg.norm(np.matmul(Y.T[None], basis))))
                return selected_hess(a, Y, basis, **kwargs)
            solver.hess_action_batch = checked
        yield
    finally:
        for name, function in saved.items():
            setattr(solver, name, function)
        solver.solve_cubic_subproblem = cubic


class FusedGeometryTests(reference_tests.IsometricGeometryTests):
    """Run all v1 invariant/fallback tests against the new implementation."""

    def setUp(self):
        super().setUp()
        self.addCleanup(patch.stopall)
        patch.object(reference_tests, "make_hooks", make_hooks).start()
        patch.object(reference_tests, "polar_transport_batch", polar_transport_batch).start()
        patch.object(reference_tests, "patched_geometry", patched_geometry).start()

    def test_polar_and_gradient_updates_match_v1_without_mutation(self):
        v1, fused = make_reference_hooks(), make_hooks()
        candidates = reference_tests.original.random_tangent_metric_gaussian(self.rng, self.X, 5)
        candidate_copy = candidates.copy()
        old1 = v1.basis_from_candidates(self.X, self.g, candidates, 6)
        old2 = fused.basis_from_candidates(self.X, self.g, candidates, 6)
        np.testing.assert_array_equal(old1, old2)
        np.testing.assert_array_equal(candidates, candidate_copy)
        eta = 0.2 * self.g
        Y1 = v1.retraction_qr(self.X, eta)
        Y2 = fused.retraction_qr(self.X, eta)
        np.testing.assert_array_equal(Y1, Y2)
        old_copy = old2.copy()
        trans1 = v1.transport_tangent_basis(Y1, old1[1:], ell=5)
        trans2 = fused.transport_tangent_basis(Y2, old2[1:], ell=5)
        np.testing.assert_array_equal(trans1, trans2)
        np.testing.assert_array_equal(old2, old_copy)
        gradient = reference_tests.original.proj_Y(Y1, self.rng.normal(size=Y1.shape))
        trans_copy = trans2.copy()
        result1 = v1.build_transport_sketch_basis(Y1, ell=6, grad=gradient, transported=trans1)
        result2 = fused.build_transport_sketch_basis(Y2, ell=6, grad=gradient, transported=trans2)
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(trans2, trans_copy)
        self.assert_frame(Y2, result2)
        self.assertEqual(fused.stats["trusted_finite_scan_skipped"], 1)
        self.assertEqual(fused.stats["fused_rank_one_updates"], 1)

    def test_generic_inputs_keep_finite_checks(self):
        hooks = make_hooks()
        old = self.initial(hooks)[1:]
        Y = hooks.retraction_qr(self.X, 0.1 * self.g)
        invalid = old.copy()
        invalid[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            hooks.transport_tangent_basis(Y, invalid)
        self.assertEqual(hooks.stats["trusted_finite_scan_skipped"], 0)

    def test_fixed_seed_solver_trajectories_match_v1(self):
        a = np.concatenate((np.linspace(20, 19, 3), np.linspace(5, 1, 37)))
        for period in (1, 3, 8):
            options = reference_tests.solver.SolverOptions(
                max_iter=15, sketch_size=6, random_state=17,
                include_gradient_in_sketch=True, sigma0=2,
                tol_grad=0, tol_pred=0, tol_step=0,
                transport_sketch=period > 1, transport_refresh_every=period,
                transport_bad_rho_tol=0, transport_orth_tol=0.5,
            )
            checks1, checks2 = [], []
            with reference_tests.patched_geometry(make_reference_hooks(), checks1):
                Y1, history1 = reference_tests.solver.solve_grassmann_nystrom_cubic(a, self.X, options)
            with reference_tests.patched_geometry(make_hooks(), checks2):
                Y2, history2 = reference_tests.solver.solve_grassmann_nystrom_cubic(a, self.X, options)
            np.testing.assert_allclose(Y1, Y2, rtol=2e-11, atol=2e-12)
            for key in history1:
                if key != "time":
                    np.testing.assert_allclose(history1[key], history2[key], rtol=2e-11, atol=2e-12)
            np.testing.assert_array_equal(history1["accepted"], history2["accepted"])
            np.testing.assert_array_equal(history1["ell"], history2["ell"])
            self.assertLess(max(error[0] for error in checks2), 1e-9)
            self.assertLess(max(error[1] for error in checks2), 1e-9)

    def test_direct_polar_map_matches_v1(self):
        basis = reference_tests.original.random_tangent_metric_gaussian(self.rng, self.X, 7)
        eta = 0.4 * self.g
        np.testing.assert_array_equal(polar_transport_batch(self.X, eta, basis),
                                      reference_polar_transport(self.X, eta, basis))

    def test_buffered_current_hessian_matches_generic_and_horizontal_inputs(self):
        hooks = make_hooks()
        a = np.linspace(0.25, 4, self.X.shape[0])
        S = self.X.T @ (a[:, None] * self.X)
        raw = self.rng.normal(size=(7, *self.X.shape))
        horizontal = reference_tests.original._proj_Y_batch(self.X, raw)
        for U in (raw, horizontal, np.asfortranarray(raw)):
            before = U.copy()
            expected = reference_tests.original.hess_action_batch(a, self.X, U, S=S)
            got = hooks.hess_action_batch(a, self.X, U, S=S)
            self.assertTrue(got.flags.c_contiguous)
            np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-14)
            np.testing.assert_array_equal(U, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
