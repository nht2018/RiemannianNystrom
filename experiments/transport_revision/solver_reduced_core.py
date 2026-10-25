"""Static, auditable adapter exposing an explicit current reduced-core hook.

The solve function is copied verbatim from code-grassmann/rsncn/
nystrom_cubic_newton.py at repository commit
91bdbf437d1906f961c4aa034c1ab99a07bbf0f4, source SHA-256
8e37f142b2118485cef60d5956c5d0b0d08353445c3f07cf357ee9c73a35b58f,
except its three-line current-HVP/core block is replaced by a single explicit
``reduced_hessian_core`` call. A source-text equality test enforces that scope.

SolverOptions and _stable_rho are imported, not independently reimplemented.
Geometry/cubic globals intentionally remain instrumentable just like upstream.
The default core is the shared upstream projected-current-HVP reference path;
alternative exact current-core kernels can replace ONLY that module global.
This adapter neither transports a Hessian nor exposes an unprojected product
under the public Hessian-action name. The original repository is untouched.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from rsncn.grassmann import (
    basis_from_candidates,
    build_transport_sketch_basis,
    geometry_terms,
    hess_action_batch,
    objective,
    random_tangent_metric_gaussian,
    retraction_qr,
    tangent_norm,
    transport_tangent_basis,
)
from rsncn.nystrom_cubic_newton import SolverOptions, _stable_rho
from rsncn.subproblem import solve_cubic_subproblem
from reduced_core import core_original as reduced_hessian_core


UPSTREAM_COMMIT = "91bdbf437d1906f961c4aa034c1ab99a07bbf0f4"
UPSTREAM_SOURCE_SHA256 = "8e37f142b2118485cef60d5956c5d0b0d08353445c3f07cf357ee9c73a35b58f"


def solve_grassmann_nystrom_cubic(
    a_diag: np.ndarray,
    Y0: np.ndarray,
    opts: Optional[SolverOptions] = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if opts is None:
        opts = SolverOptions()

    rng = np.random.default_rng(opts.random_state)
    Y, _ = np.linalg.qr(Y0, mode="reduced")
    geom_state, f = geometry_terms(a_diag, Y)
    g = geom_state.grad
    gn = tangent_norm(g)
    sigma = opts.sigma0
    cached_random_basis: Optional[np.ndarray] = None

    hist: Dict[str, List[float]] = {
        "f": [f],
        "grad_norm": [gn],
        "sigma": [sigma],
        "rho": [],
        "accepted": [],
        "time": [0.0],
        "ell": [],
    }
    t0 = time.perf_counter()

    def _build_basis(it: int) -> np.ndarray:
        grad_for_basis = g if (opts.include_gradient_in_sketch or (opts.transport_sketch and opts.transport_keep_gradient_direction)) else None
        target_random = max(0, opts.sketch_size - (1 if grad_for_basis is not None else 0))

        scheduled_refresh = (
            opts.transport_refresh_every > 0
            and it > 1
            and ((it - 1) % opts.transport_refresh_every == 0)
        )
        reuse_transport = (
            opts.transport_sketch
            and cached_random_basis is not None
            and cached_random_basis.size > 0
            and not scheduled_refresh
            and target_random > 0
        )

        transported = np.zeros((0, *Y.shape), dtype=float)
        if reuse_transport:
            transported, orth_err = transport_tangent_basis(
                Y,
                cached_random_basis,
                ell=target_random,
                return_orth_error=True,
            )
            if orth_err > opts.transport_orth_tol:
                transported = np.zeros((0, *Y.shape), dtype=float)

        need_random = max(0, target_random - transported.shape[0])
        fresh_random = (
            random_tangent_metric_gaussian(rng, Y, need_random)
            if need_random > 0
            else np.zeros((0, *Y.shape), dtype=float)
        )
        if transported.size and fresh_random.size:
            candidates = np.concatenate([transported, fresh_random], axis=0)
        elif transported.size:
            candidates = transported
        else:
            candidates = fresh_random

        if opts.transport_sketch and transported.size:
            basis = build_transport_sketch_basis(
                Y,
                ell=opts.sketch_size,
                grad=grad_for_basis,
                transported=transported,
                fresh_candidates=fresh_random,
                skip_transport_reorth=opts.transport_skip_reorth,
            )
        else:
            basis = basis_from_candidates(
                Y,
                grad=grad_for_basis,
                candidates=candidates,
                ell=opts.sketch_size,
            )
        if basis.shape[0] == 0:
            raise RuntimeError("Sketch basis generation failed")
        return basis

    for it in range(1, opts.max_iter + 1):
        basis = _build_basis(it)
        ell = int(basis.shape[0])
        basis_flat = basis.reshape(ell, -1)
        hS = basis_flat @ g.reshape(-1)
        HS = reduced_hessian_core(a_diag, Y, basis, S=geom_state.S)
        if sigma == 0.0:
            HSd = HS + (opts.ridge * np.eye(HS.shape[0]) if opts.ridge > 0 else 0.0)
            try:
                v = -np.linalg.solve(HSd, hS)
            except np.linalg.LinAlgError:
                v = -np.linalg.lstsq(HSd, hS, rcond=None)[0]
        else:
            v, _ = solve_cubic_subproblem(HS, hS, sigma, return_info=False)
        HSv = HS @ v
        v_norm = float(np.linalg.norm(v))
        phi_v = float(hS @ v + 0.5 * v @ HSv + (sigma / 6.0) * (v_norm ** 3))
        eta = np.tensordot(v, basis, axes=(0, 0))
        Y_trial = retraction_qr(Y, eta)
        f_trial = objective(a_diag, Y_trial)
        pred = -phi_v
        ared = f - f_trial
        rho, pred_is_usable = _stable_rho(ared, pred, opts.rho_stab_tol)
        accepted = pred_is_usable and np.isfinite(rho) and rho >= opts.eta1
        keep_transport = np.isfinite(rho) and rho >= opts.transport_bad_rho_tol
        if accepted:
            Y = Y_trial
            geom_state, f = geometry_terms(a_diag, Y)
            g = geom_state.grad
            gn = tangent_norm(g)
            if opts.transport_sketch:
                if opts.transport_keep_gradient_direction or opts.include_gradient_in_sketch:
                    random_part = basis[1:] if basis.shape[0] > 1 else np.zeros((0, *Y.shape), dtype=float)
                else:
                    random_part = basis
                cached_random_basis = random_part if keep_transport else None
        else:
            if opts.transport_sketch:
                if opts.transport_reset_on_reject or not keep_transport:
                    cached_random_basis = None
                else:
                    if opts.transport_keep_gradient_direction or opts.include_gradient_in_sketch:
                        cached_random_basis = basis[1:] if basis.shape[0] > 1 else np.zeros((0, *Y.shape), dtype=float)
                    else:
                        cached_random_basis = basis
        if sigma > 0:
            if np.isfinite(rho) and rho >= opts.eta2:
                sigma = max(opts.sigma_min, sigma * opts.gamma1)
            elif (not pred_is_usable) or (not np.isfinite(rho)) or rho < opts.eta1:
                sigma = sigma * opts.gamma2
                if opts.sigma_max is not None:
                    sigma = min(opts.sigma_max, sigma)

        hist["f"].append(f)
        hist["grad_norm"].append(gn)
        hist["sigma"].append(sigma)
        hist["rho"].append(float(rho))
        hist["accepted"].append(1.0 if accepted else 0.0)
        hist["time"].append(time.perf_counter() - t0)
        hist["ell"].append(float(ell))

        if opts.display_progress:
            print(
                f"[grassmann] iter {it}/{opts.max_iter} f={f:.6e} ||grad||={gn:.3e} sigma={sigma:.2e} accepted={int(accepted)}",
                flush=True,
            )
        pred_tiny = abs(pred) <= opts.tol_pred * max(1.0, abs(f))
        step_tiny = tangent_norm(eta) <= opts.tol_step * max(1.0, float(np.linalg.norm(Y, ord='fro')))
        if gn <= opts.tol_grad or pred_tiny or step_tiny:
            break
        if opts.time_limit is not None and hist["time"][-1] >= opts.time_limit:
            break

    return Y, hist
