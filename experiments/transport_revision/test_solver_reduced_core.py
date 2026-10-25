"""Static scope and tiny exact-reference checks for the solver adapter."""

import hashlib
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code-grassmann"))
from rsncn import grassmann
from rsncn import nystrom_cubic_newton as original
from cubic_verified import solve_cubic_subproblem
import solver_reduced_core as adapter


class ReducedCoreSolverTests(unittest.TestCase):
    def test_upstream_identity_and_shared_options(self):
        original_path = Path(inspect.getsourcefile(original))
        self.assertEqual(hashlib.sha256(original_path.read_bytes()).hexdigest(),
                         adapter.UPSTREAM_SOURCE_SHA256)
        self.assertIs(adapter.SolverOptions, original.SolverOptions)
        self.assertIs(adapter._stable_rho, original._stable_rho)

    def test_only_three_line_core_block_differs_in_solve_function(self):
        old = ("        Hv = hess_action_batch(a_diag, Y, basis, S=geom_state.S)\n"
               "        HS = basis_flat @ Hv.reshape(ell, -1).T\n"
               "        HS = 0.5 * (HS + HS.T)\n")
        new = "        HS = reduced_hessian_core(a_diag, Y, basis, S=geom_state.S)\n"
        upstream_source = inspect.getsource(original.solve_grassmann_nystrom_cubic)
        self.assertEqual(upstream_source.count(old), 1)
        self.assertEqual(inspect.getsource(adapter.solve_grassmann_nystrom_cubic),
                         upstream_source.replace(old, new))

    def test_default_core_matches_original_block_for_generic_inputs(self):
        rng = np.random.default_rng(716)
        Y, _ = np.linalg.qr(rng.standard_normal((31, 3)))
        a = np.linspace(0.25, 4.0, len(Y))
        S = Y.T @ (a[:, None] * Y)
        for basis in (rng.standard_normal((5, *Y.shape)),
                      grassmann.random_tangent_metric_gaussian(rng, Y, 5)):
            flat = basis.reshape(len(basis), -1)
            hess = grassmann.hess_action_batch(a, Y, basis, S=S)
            expected = flat @ hess.reshape(len(basis), -1).T
            expected = 0.5 * (expected + expected.T)
            np.testing.assert_array_equal(adapter.reduced_hessian_core(a, Y, basis, S=S), expected)

    def test_original_reference_trajectories_and_one_core_call_per_trial(self):
        rng = np.random.default_rng(917)
        Y0, _ = np.linalg.qr(rng.standard_normal((40, 3)))
        a = np.concatenate((np.linspace(20.0, 19.0, 3), np.linspace(5.0, 1.0, 37)))
        for period, sigma in ((1, 2.0), (3, 2.0), (8, 2.0), (3, 0.0)):
            with self.subTest(period=period, sigma=sigma):
                calls = []

                def reference_core(a_diag, Y, basis, *, S):
                    calls.append(len(basis))
                    flat = basis.reshape(len(basis), -1)
                    Hv = grassmann.hess_action_batch(a_diag, Y, basis, S=S)
                    HS = flat @ Hv.reshape(len(basis), -1).T
                    return 0.5 * (HS + HS.T)

                options = original.SolverOptions(
                    max_iter=12, sketch_size=6, random_state=37,
                    include_gradient_in_sketch=True, sigma0=sigma, ridge=1e-3,
                    tol_grad=0.0, tol_pred=0.0, tol_step=0.0,
                    transport_sketch=period > 1, transport_refresh_every=period,
                    transport_bad_rho_tol=0.0, transport_orth_tol=0.5,
                )
                with patch.object(original, "solve_cubic_subproblem", solve_cubic_subproblem):
                    Yref, histref = original.solve_grassmann_nystrom_cubic(a, Y0, options)
                with patch.object(adapter, "solve_cubic_subproblem", solve_cubic_subproblem), \
                        patch.object(adapter, "reduced_hessian_core", reference_core):
                    Ygot, histgot = adapter.solve_grassmann_nystrom_cubic(a, Y0, options)
                np.testing.assert_array_equal(Ygot, Yref)
                self.assertEqual(histgot.keys(), histref.keys())
                for key in histref:
                    if key != "time":
                        np.testing.assert_array_equal(histgot[key], histref[key])
                self.assertEqual(len(calls), len(histgot["accepted"]))
                np.testing.assert_array_equal(calls, histgot["ell"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
