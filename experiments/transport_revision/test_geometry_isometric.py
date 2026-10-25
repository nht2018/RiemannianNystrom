"""Small correctness checks only; never run concurrently with benchmarks."""

from contextlib import contextmanager
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code-grassmann"))
from rsncn import grassmann as original
from rsncn import nystrom_cubic_newton as solver
from cubic_verified import solve_cubic_subproblem
from geometry_isometric import HOOK_NAMES, make_hooks, polar_transport_batch


@contextmanager
def patched_geometry(hooks, diagnostics=None):
    saved = {name: getattr(solver, name) for name in HOOK_NAMES}
    cubic = solver.solve_cubic_subproblem
    hess = solver.hess_action_batch
    try:
        for name in HOOK_NAMES:
            setattr(solver, name, getattr(hooks, name))
        solver.solve_cubic_subproblem = solve_cubic_subproblem
        if diagnostics is not None:
            def checked(a, Y, basis, **kwargs):
                flat = basis.reshape(basis.shape[0], -1)
                diagnostics.append((np.linalg.norm(flat @ flat.T - np.eye(len(flat)), 2),
                                    np.linalg.norm(np.matmul(Y.T[None], basis))))
                return hess(a, Y, basis, **kwargs)
            solver.hess_action_batch = checked
        yield
    finally:
        for name, function in saved.items():
            setattr(solver, name, function)
        solver.solve_cubic_subproblem = cubic
        solver.hess_action_batch = hess


def projector(basis):
    flat = basis.reshape(basis.shape[0], -1)
    return flat.T @ flat


class IsometricGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(914)
        self.X, _ = np.linalg.qr(self.rng.normal(size=(40, 3)))
        self.g = original.proj_Y(self.X, self.rng.normal(size=self.X.shape))

    def assert_frame(self, Y, basis, tol=1e-10):
        self.assertTrue(basis.flags.c_contiguous)
        flat = basis.reshape(basis.shape[0], -1)
        np.testing.assert_allclose(flat @ flat.T, np.eye(len(flat)), atol=tol)
        np.testing.assert_allclose(np.matmul(Y.T[None], basis), 0, atol=tol)

    def initial(self, hooks, ell=6):
        candidates = hooks.random_tangent_metric_gaussian(self.rng, self.X, ell - 1)
        return hooks.basis_from_candidates(self.X, self.g, candidates, ell)

    def test_full_space_isometry_matches_projection_expression(self):
        Z = original.random_tangent_metric_gaussian(self.rng, self.X, 7)
        flat = Z.reshape(len(Z), -1)
        for size in (0.0, 1e-12, 0.1, 3.0):
            eta = size * self.g / np.linalg.norm(self.g)
            Y = original.retraction_qr(self.X, eta)
            got = polar_transport_batch(self.X, eta, Z)
            vals, vecs = np.linalg.eigh(np.eye(3) + eta.T @ eta)
            inverse = (vecs * (1 / (np.sqrt(vals) + 1))) @ vecs.T
            full = original._proj_Y_batch(Y, Z + np.matmul((eta @ inverse)[None], np.matmul(eta.T[None], Z)))
            np.testing.assert_allclose(got, full, atol=2e-13)
            got_flat = got.reshape(len(got), -1)
            np.testing.assert_allclose(got_flat @ got_flat.T, flat @ flat.T, atol=2e-12)
            np.testing.assert_allclose(np.matmul(Y.T[None], got), 0, atol=2e-13)
            self.assertTrue(got.flags.c_contiguous)

    def test_fresh_native_eigen_frame_and_generic_projection(self):
        hooks = make_hooks()
        candidates = hooks.random_tangent_metric_gaussian(self.rng, self.X, 5)
        got = hooks.basis_from_candidates(self.X, self.g, candidates, 6)
        ref = original.basis_from_candidates(self.X, self.g, candidates, 6)
        self.assert_frame(self.X, got)
        np.testing.assert_allclose(projector(got), projector(ref), atol=2e-12)
        self.assertEqual(hooks.stats["fresh_projection_skipped"], 1)
        candidates = self.rng.normal(size=(5, *self.X.shape))
        got = hooks.basis_from_candidates(self.X, self.g, candidates, 6)
        ref = original.basis_from_candidates(self.X, self.g, candidates, 6)
        np.testing.assert_array_equal(got, ref)

    def test_rank_one_whitening_matches_symmetric_whitening(self):
        hooks = make_hooks()
        old = self.initial(hooks)[1:]
        Y = hooks.retraction_qr(self.X, 0.15 * self.g / np.linalg.norm(self.g))
        Q, defect = hooks.transport_tangent_basis(Y, old, ell=5, return_orth_error=True)
        grad = original.proj_Y(Y, self.rng.normal(size=Y.shape))
        q = grad.reshape(-1) / np.linalg.norm(grad)
        flat = Q.reshape(len(Q), -1)
        deflated = flat - (flat @ q)[:, None] * q
        gram = deflated @ deflated.T
        values, vectors = np.linalg.eigh(gram)
        symmetric = ((vectors * (1 / np.sqrt(values))) @ vectors.T) @ deflated
        got = hooks.build_transport_sketch_basis(Y, ell=6, grad=grad, transported=Q)
        self.assert_frame(Y, got)
        self.assertEqual(defect, 0.0)
        np.testing.assert_allclose(got[1:].reshape(5, -1), symmetric, atol=2e-12)
        self.assertEqual(hooks.stats["gradient_rank_one_whitenings"], 1)
        self.assertEqual(hooks.stats["transport_fallback"], 0)

    def test_context_rejection_and_nonhorizontal_errors(self):
        hooks = make_hooks()
        old = self.initial(hooks)[1:]
        with self.assertRaisesRegex(ValueError, "paired QR"):
            hooks.transport_tangent_basis(self.X, old)
        Y = hooks.retraction_qr(self.X, 0.1 * self.g)
        with self.assertRaisesRegex(ValueError, "paired QR"):
            hooks.transport_tangent_basis(Y.copy(), old)
        got = hooks.transport_tangent_basis(self.X, old)
        np.testing.assert_array_equal(got, old)
        self.assertEqual(hooks.stats["rejected_identity_transports"], 1)
        hooks.retraction_qr(self.X, self.X)
        with self.assertRaisesRegex(ValueError, "horizontal"):
            hooks.transport_tangent_basis(self.X, old)
        with self.assertRaisesRegex(ValueError, "skip_reorth"):
            make_hooks(skip_transport_reorth=True)

    def test_rank_loss_mixed_truncated_and_unpaired_fallbacks(self):
        for mode in ("rank_loss", "mixed", "truncated", "unpaired"):
            hooks = make_hooks()
            old = self.initial(hooks)[1:]
            Y = hooks.retraction_qr(self.X, np.zeros_like(self.X))
            trans = hooks.transport_tangent_basis(Y, old, ell=5)
            grad = trans[0] if mode == "rank_loss" else original.proj_Y(Y, self.g)
            fresh = hooks.random_tangent_metric_gaussian(self.rng, Y, 1) if mode == "mixed" else None
            if mode == "unpaired":
                trans = trans.copy()
            ell = 3 if mode == "truncated" else 6
            got = hooks.build_transport_sketch_basis(Y, ell=ell, grad=grad, transported=trans, fresh_candidates=fresh)
            self.assert_frame(Y, got)
            self.assertGreater(hooks.stats["build_fallback"], 0)
            if mode == "rank_loss":
                self.assertEqual(len(got), 5)
                self.assertEqual(hooks.stats["rank_loss_fallback"], 1)

    def test_generic_transport_and_nonfinite_guard(self):
        hooks = make_hooks()
        old = self.initial(hooks)[1:]
        Y = hooks.retraction_qr(self.X, 0.1 * self.g)
        got = hooks.transport_tangent_basis(Y, old.copy(), ell=3)
        self.assertEqual(len(got), 3)
        self.assert_frame(Y, got)
        self.assertEqual(hooks.stats["transport_fallback"], 1)
        invalid = old.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            hooks.transport_tangent_basis(Y, invalid)

    def test_ten_step_gapped_solver_frame_invariants(self):
        a = np.concatenate((np.linspace(20, 19, 3), np.linspace(5, 1, 37)))
        for period in (1, 3, 8):
            hooks = make_hooks()
            checks = []
            options = solver.SolverOptions(
                max_iter=10, sketch_size=6, random_state=11,
                include_gradient_in_sketch=True, sigma0=2.0,
                tol_grad=0, tol_pred=0, tol_step=0,
                transport_sketch=period > 1, transport_refresh_every=period,
                transport_bad_rho_tol=0, transport_orth_tol=0.5,
            )
            with patched_geometry(hooks, checks):
                Y, history = solver.solve_grassmann_nystrom_cubic(a, self.X, options)
            self.assertEqual(len(history["accepted"]), 10)
            self.assertEqual(len(checks), 10)
            self.assertLess(max(v[0] for v in checks), 1e-9)
            self.assertLess(max(v[1] for v in checks), 1e-9)
            self.assertTrue(np.all(np.isfinite(Y)))
            if period > 1:
                self.assertGreater(hooks.stats["polar_transports"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
