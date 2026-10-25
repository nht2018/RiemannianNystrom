"""Serial dense-matrix Grassmann experiments with explicit provenance.

The original algorithm adapter is unchanged. A new context installs dense
objective/gradient/current-Hessian kernels and only the five matrix-independent
full-polar geometry hooks. Neither the diagonal HVP hook nor a diagonal reduced
core is ever installed. Eigenvalue references belong solely to the evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

# Request a common thread policy before numerical libraries load. Accelerate
# may not be recognized by threadpoolctl; actual thread count is not certified.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_info, threadpool_limits

DENSE = Path(__file__).resolve().parent
EXPERIMENT = DENSE.parent
REPO = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(REPO / "code-grassmann"))

import solver_reduced_core as solver
from cubic_verified import solve_cubic_subproblem as verified_cubic
from geometry_isometric_fused import make_hooks
from dense.kernels import DenseOracle

BACKEND = "verified"
GEOMETRY = "isometric_fused_dense"
CORE = "dense_chunked_current"
RANDOM_STREAMS = "spawned3"
GEOMETRY_HOOKS = ("random_tangent_metric_gaussian", "basis_from_candidates",
                  "transport_tangent_basis", "build_transport_sketch_basis", "retraction_qr")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    """Hash the actual C-order float64 input bytes, not its seed or generator."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def source_map():
    paths = (list((REPO / "code-grassmann/rsncn").glob("*.py"))
             + list(EXPERIMENT.glob("*.py")) + list(DENSE.glob("*.py")))
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(paths)}


def assert_sources_unchanged(expected):
    current = source_map()
    if current != expected:
        changed = sorted(name for name in set(current) | set(expected) if current.get(name) != expected.get(name))
        raise RuntimeError(f"Frozen execution source files changed: {changed}")


def provenance(plan_path):
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "repository": subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=REPO, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": source_map(), "plan_sha256": sha(plan_path),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "threadpoolctl")},
        "threadpools": threadpool_info(),
        "thread_policy": "Single-thread environment requested; actual Accelerate thread count is not certified when threadpools is empty.",
        "timing_scope": "Recorded solver loop includes current sketch/core/cubic/retraction/objective/accepted gradient. Problem generation, eigenvalue reference, initial point QR, solver initial QR and initial geometry are excluded. External solve wall time includes solver initial QR/geometry. No parallel benchmark jobs.",
        "objective_reference": "Numerical reference: minus sum of the largest p eigenvalues of the actual materialized symmetric A, independently computed with scipy.linalg.eigvalsh outside solver timing. Design eigenvalues are a cross-check, not an exact claim for the floating-point stored matrix.",
        "solver_input_contract": "The solver receives only the immutable dense ndarray A, Y0 and options; the generating Q, design spectrum and eigenvalue reference are never passed to solver kernels.",
    }


def stream_seeds(seed):
    streams = np.random.SeedSequence(int(seed)).spawn(3)
    names = ("matrix_seed", "initial_point_seed", "sketch_seed")
    return {name: int(stream.generate_state(1, dtype=np.uint64)[0]) for name, stream in zip(names, streams)}


@dataclass
class Problem:
    A: np.ndarray
    Y0: np.ndarray
    reference: float
    metadata: dict
    seeds: dict


def problem_key(case, seed):
    return (int(case["n"]), int(case["p"]), int(seed), case.get("matrix", "dense_rotated_gapped"),
            case.get("spectrum", "gapped"), float(case.get("tail_top", 18.0)))


def build_problem(case, seed):
    """Build once per size/spectrum/seed, outside all solve timing scopes."""
    n, p = int(case["n"]), int(case["p"])
    if case.get("matrix", "dense_rotated_gapped") != "dense_rotated_gapped" or case.get("spectrum", "gapped") != "gapped":
        raise ValueError("This runner supports only materialized dense_rotated_gapped inputs")
    tail_top = float(case.get("tail_top", 18.0))
    if not (0 < p < n and 1.0 <= tail_top < 19.0):
        raise ValueError("Require 0<p<n and 1<=tail_top<19 for this ordered spectrum")
    seeds = stream_seeds(seed)
    start = time.perf_counter()
    gaussian = np.random.default_rng(seeds["matrix_seed"]).standard_normal((n, n))
    Q, R = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q *= signs[None, :]
    design_values = np.concatenate((np.linspace(20.0, 19.0, p), np.linspace(tail_top, 1.0, n-p)))
    product = (Q * design_values[None, :]) @ Q.T
    A = np.ascontiguousarray(0.5 * (product + product.T), dtype=np.float64)
    generation_s = time.perf_counter() - start
    if not np.all(np.isfinite(A)):
        raise ValueError("Nonfinite materialized matrix")

    # These numerical checks and reference computations are never visible to
    # the solver. A floating-point eigensolve is not an interval certificate.
    start = time.perf_counter()
    reference_eigenvalues = eigvalsh(A, subset_by_index=(n-p, n-1), driver="evr", check_finite=True)
    reference = -float(np.sum(reference_eigenvalues))
    reference_s = time.perf_counter() - start
    if not np.all(np.isfinite(reference_eigenvalues)):
        raise ValueError("Nonfinite eigenvalue reference")
    start = time.perf_counter()
    A_norm = float(np.linalg.norm(A, ord="fro"))
    off_diagonal = A.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    design_reference = -float(np.sum(design_values[:p]))
    metadata = {
        "matrix": "dense_rotated_gapped", "n": n, "p": p, "dtype": str(A.dtype),
        "shape": list(A.shape), "matrix_sha256": array_sha(A),
        "matrix_generation_s": generation_s, "reference_eigenvalue_s": reference_s,
        "matrix_frobenius_norm": A_norm,
        "offdiagonal_frobenius_fraction": float(np.linalg.norm(off_diagonal, ord="fro")) / A_norm,
        "symmetry_frobenius_error": float(np.linalg.norm(A - A.T, ord="fro")),
        "generator_Q_orthogonality_frobenius_error": float(np.linalg.norm(Q.T @ Q - np.eye(n), ord="fro")),
        "generator_eigenpair_residual_frobenius_norm": float(np.linalg.norm(A @ Q - Q * design_values[None, :], ord="fro")),
        "reference_top_eigenvalues": reference_eigenvalues.tolist(),
        "design_top_eigenvalues": design_values[:p].tolist(),
        "reference_vs_design_top_max_absolute_error": float(np.max(np.abs(reference_eigenvalues - np.sort(design_values[:p])))),
        "f_star_numerical_reference": reference, "f_star_design_spectrum": design_reference,
        "reference_vs_design_objective_absolute_difference": abs(reference - design_reference),
        "reference_kind": "independent scipy.linalg.eigvalsh of actual materialized A; numerical, not an interval-certified exact value",
        "reference_passed_to_solver": False, "generator_Q_passed_to_solver": False,
        "design_eigenvalues_passed_to_solver": False,
        "matrix_seed": seeds["matrix_seed"],
        "setup_reuse": "Same immutable A and initial Y0 reused across all T, sketch-size controls and timing replicates with this size/spectrum/seed.",
    }
    metadata["matrix_validation_s"] = time.perf_counter() - start
    # The problem object intentionally does not retain Q or the design spectrum.
    del Q, R, gaussian, product, off_diagonal, design_values
    start = time.perf_counter()
    Y0, _ = np.linalg.qr(np.random.default_rng(seeds["initial_point_seed"]).standard_normal((n, p)), mode="reduced")
    Y0 = np.ascontiguousarray(Y0)
    metadata["initial_point_generation_s"] = time.perf_counter() - start
    metadata["initial_point_sha256"] = array_sha(Y0)
    A.setflags(write=False)
    Y0.setflags(write=False)
    return Problem(A=A, Y0=Y0, reference=reference, metadata=metadata, seeds=seeds)


def independent_projected_core(A, Y, basis, S):
    """Small diagnostic reference, independently materializing each dense HVP."""
    actions = []
    for direction in basis:
        W = -2.0 * (A @ direction) + 2.0 * (direction @ S)
        actions.append(W - Y @ (Y.T @ W))
    actions = np.asarray(actions, dtype=float)
    flat = basis.reshape(len(basis), Y.size)
    core = flat @ actions.reshape(len(basis), Y.size).T
    return 0.5 * (core + core.T)


@contextmanager
def instrument_dense(*, validate=False, chunk_size=16, workspace_bytes=128 * 1024 * 1024):
    """Install all dense hooks explicitly; never call old instrument()."""
    oracle = DenseOracle(chunk_size=chunk_size, workspace_bytes=workspace_bytes)
    hooks = make_hooks(skip_transport_reorth=False)
    elapsed, counts = defaultdict(float), Counter()
    checks = {"max_frame_gram_error": 0.0, "max_frame_tangent_error": 0.0,
              "max_pre_reorth_defect": 0.0, "max_current_core_absolute_error": 0.0,
              "max_current_core_scaled_error": 0.0, "max_cubic_stationarity": 0.0,
              "max_cubic_secular": 0.0, "cubic_failures": 0, "cubic_methods": Counter(),
              "hessian_columns": 0, "current_core_columns": 0, "current_core_calls": 0,
              "reduced_core_backend": CORE, "reduced_core_dispatch": {},
              "geometry_optimization": hooks.stats, "coordinate_tail_step_fractions": [],
              "diagnostic_checks_measured": bool(validate),
              "transport_defect_semantics": "The fast hook reports analytic zero for the full polar map, not a measured frame norm. Actual frame Gram/tangency checks are performed only when validate=true.",
              "hessian_implementation": "Same chunked dense CURRENT Hessian core for every arm, with actual dense A products every trial and full projection correction. No diagonal/spectral-majority optimization, old Hessian reuse, or materialized full np-by-np Hessian.",
              "hessian_count_semantics": "current_core_columns counts each returned reduced-core basis column; hessian_columns counts only the public solver HVP hook. DenseOracle counters separately expose dense products internal to the core; hessian_columns=0 does not mean zero curvature work."}
    implementations = {name: getattr(hooks, name) for name in GEOMETRY_HOOKS}
    implementations.update(objective=oracle.objective, geometry_terms=oracle.geometry_terms,
                           hess_action_batch=oracle.hess_action_batch, reduced_hessian_core=oracle.core)
    phases = {"random_tangent_metric_gaussian": "gaussian_and_projection",
              "basis_from_candidates": "fresh_basis", "transport_tangent_basis": "transport_and_reorth",
              "build_transport_sketch_basis": "transport_gradient_reorth", "retraction_qr": "retraction",
              "objective": "trial_objective", "geometry_terms": "objective_gradient",
              "hess_action_batch": "hessian_actions", "reduced_hessian_core": "current_reduced_core"}
    originals = {name: getattr(solver, name) for name in (*implementations, "solve_cubic_subproblem")}

    def wrap(name, implementation):
        def timed(*args, **kwargs):
            phase = phases[name]
            if name == "geometry_terms" and counts["initial_objective_gradient"] == 0:
                phase = "initial_objective_gradient"
            if name == "reduced_hessian_core":
                kwargs["info"] = checks["reduced_core_dispatch"]
            start = time.perf_counter()
            value = implementation(*args, **kwargs)
            elapsed[phase] += time.perf_counter() - start
            counts[phase] += 1
            if name == "transport_tangent_basis":
                checks["max_pre_reorth_defect"] = max(checks["max_pre_reorth_defect"], float(value[1]))
            if name in ("hess_action_batch", "reduced_hessian_core"):
                A, Y, basis = args[:3]
                counter = "current_core_columns" if name == "reduced_hessian_core" else "hessian_columns"
                checks[counter] += len(basis)
                if name == "reduced_hessian_core":
                    checks["current_core_calls"] += 1
                if validate:
                    start = time.perf_counter()
                    flat = basis.reshape(len(basis), Y.size)
                    checks["max_frame_gram_error"] = max(checks["max_frame_gram_error"], float(np.linalg.norm(flat @ flat.T - np.eye(len(basis)), ord="fro")))
                    checks["max_frame_tangent_error"] = max(checks["max_frame_tangent_error"], float(np.linalg.norm(np.matmul(Y.T[None, :, :], basis))))
                    if name == "reduced_hessian_core":
                        reference = independent_projected_core(A, Y, basis, kwargs["S"])
                        error = float(np.linalg.norm(value - reference, ord="fro"))
                        scaled = error / max(1.0, float(np.linalg.norm(reference, ord="fro")))
                        checks["max_current_core_absolute_error"] = max(checks["max_current_core_absolute_error"], error)
                        checks["max_current_core_scaled_error"] = max(checks["max_current_core_scaled_error"], scaled)
                        if not math.isfinite(scaled) or scaled > 1e-9:
                            raise ArithmeticError(f"Dense current-core diagnostic failed: scaled error {scaled}")
                    elapsed["diagnostic_validation"] += time.perf_counter() - start
                    counts["diagnostic_validation"] += 1
            return value
        return timed

    def cubic(*args, **kwargs):
        want_info = kwargs.pop("return_info", True)
        start = time.perf_counter()
        vector, multiplier, info = verified_cubic(*args, **kwargs, return_info=True)
        elapsed["cubic_subproblem"] += time.perf_counter() - start
        counts["cubic_subproblem"] += 1
        checks["cubic_methods"][info.get("method", "unknown")] += 1
        checks["cubic_failures"] += int(not info.get("converged", True))
        checks["max_cubic_stationarity"] = max(checks["max_cubic_stationarity"], float(info.get("res_stat", 0.0)))
        checks["max_cubic_secular"] = max(checks["max_cubic_secular"], float(info.get("res_sec", 0.0)))
        norm = float(np.linalg.norm(vector))
        checks["coordinate_tail_step_fractions"].append(float(np.linalg.norm(vector[1:])) / norm if norm else 0.0)
        return (vector, multiplier, info) if want_info else (vector, multiplier)

    try:
        for name, implementation in implementations.items():
            setattr(solver, name, wrap(name, implementation))
        solver.solve_cubic_subproblem = cubic
        yield elapsed, counts, checks, oracle
    finally:
        for name, implementation in originals.items():
            setattr(solver, name, implementation)


def builtin(value):
    if isinstance(value, dict):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_one(problem, case, seed, T, *, replicate=0, validate=False, chunk_size=16, workspace_bytes=128 * 1024 * 1024):
    if case.get("method_seed_offsets") and any(case["method_seed_offsets"].values()):
        raise ValueError("This paired dense comparison does not permit method-specific seed offsets")
    opts = solver.SolverOptions(
        max_iter=int(case.get("max_iter", 100)), tol_grad=float(case.get("tol_grad", 0.0)),
        tol_pred=0.0, tol_step=0.0, sigma0=1.0, sigma_min=1e-16,
        eta1=0.1, eta2=0.75, gamma1=0.5, gamma2=2.0,
        sketch_size=int(case["ell"]), include_gradient_in_sketch=True,
        random_state=problem.seeds["sketch_seed"], transport_sketch=T != 1,
        transport_refresh_every=T if T != 1 else 0, transport_keep_gradient_direction=True,
        transport_reset_on_reject=True, transport_bad_rho_tol=0.5,
        transport_orth_tol=0.01, transport_skip_reorth=False,
        time_limit=float(case.get("time_limit", 300.0)),
    )
    with instrument_dense(validate=validate, chunk_size=chunk_size, workspace_bytes=workspace_bytes) as (phases, counts, checks, oracle):
        start = time.perf_counter()
        Y, history = solver.solve_grassmann_nystrom_cubic(problem.A, problem.Y0, opts)
        wall = time.perf_counter() - start
        oracle_stats = builtin(dict(oracle.stats))
    if not np.all(np.isfinite(Y)) or any(not np.all(np.isfinite(history[name])) for name in ("f", "grad_norm", "time")):
        raise ArithmeticError("Solver produced nonfinite point/objective/gradient/time history")
    if array_sha(problem.A) != problem.metadata["matrix_sha256"] or array_sha(problem.Y0) != problem.metadata["initial_point_sha256"]:
        raise RuntimeError("Solver mutated a shared immutable input")
    gap = np.asarray(history["f"], dtype=float) - problem.reference
    if not math.isfinite(float(gap[0])) or gap[0] <= 0:
        raise ArithmeticError("Invalid nonpositive initial objective gap")
    point_error = float(np.linalg.norm(Y.T @ Y - np.eye(Y.shape[1]), ord="fro"))
    gap_tolerance = 100.0 * np.finfo(float).eps * len(problem.A) * max(1.0, abs(problem.reference))
    checks["final_point_orth_error"] = point_error
    checks["minimum_objective_gap"] = float(np.min(gap))
    checks["negative_gap_numerical_tolerance"] = gap_tolerance
    checks["reference_vs_design_relative_initial_gap"] = problem.metadata["reference_vs_design_objective_absolute_difference"] / float(gap[0])
    checks["reference_discrepancy_over_smallest_target"] = checks["reference_vs_design_relative_initial_gap"] / min(case.get("targets", (1e-2, 1e-4, 1e-6)))
    if point_error > 1e-8 or float(np.min(gap)) < -gap_tolerance:
        raise ArithmeticError("Final point orthogonality or numerical objective-reference sanity check failed")
    if checks["reference_discrepancy_over_smallest_target"] > 0.01:
        raise ArithmeticError("Observed reference/design discrepancy is not small compared with requested targets")
    relative_gap = gap / gap[0]
    history["objective_gap"] = gap.tolist()
    history["relative_objective_gap"] = relative_gap.tolist()
    initial_gradient = float(history["grad_norm"][0])
    history["relative_grad_norm"] = (np.asarray(history["grad_norm"]) / initial_gradient).tolist() if initial_gradient > 0 else [0.0] * len(history["grad_norm"])
    targets = {}
    for target in case.get("targets", (1e-2, 1e-4, 1e-6)):
        hits = np.flatnonzero(relative_gap <= target)
        hit = int(hits[0]) if hits.size else None
        targets[str(target)] = {"reached": hit is not None, "iteration": hit,
                                "time_s": history["time"][hit] if hit is not None else None,
                                "grad_norm": history["grad_norm"][hit] if hit is not None else None}
    iterations = len(history["f"]) - 1
    if checks["current_core_calls"] != iterations or counts["cubic_subproblem"] != iterations:
        raise RuntimeError("Every trial must have exactly one current dense core and one cubic solve")
    if counts["initial_objective_gradient"] != 1 or counts["objective_gradient"] != int(sum(history["accepted"])):
        raise RuntimeError("Initialization/current-gradient phase counters are inconsistent")
    return builtin({
        "case": case, "seed": int(seed), "T": int(T), "replicate": int(replicate),
        "backend": BACKEND, "geometry": GEOMETRY, "core": CORE, "random_streams": RANDOM_STREAMS,
        "diagnostic_run": bool(validate), "options": asdict(opts), **problem.seeds,
        "problem": problem.metadata, "matrix_sha256": problem.metadata["matrix_sha256"],
        "initial_point_sha256": problem.metadata["initial_point_sha256"],
        "chunk_size": int(chunk_size), "workspace_bytes": int(workspace_bytes),
        "f_star": problem.reference, "f_star_reference_kind": problem.metadata["reference_kind"],
        "initial_gap": float(gap[0]), "targets": targets,
        "iterations": iterations, "accepted_steps": int(sum(history["accepted"])),
        "elapsed_s": history["time"][-1], "external_wall_s": wall,
        "external_wall_scope": "One solve, including its initial point QR and initial dense objective/gradient; excluding shared matrix/reference/Y0 preparation.",
        "final_relative_gap": float(relative_gap[-1]), "final_grad_norm": history["grad_norm"][-1],
        "final_point_orth_error": point_error,
        "timings_s": dict(phases), "counts": dict(counts), "checks": checks,
        "dense_oracle_stats": oracle_stats, "history": history,
    })


def validate_plan(plan):
    for name, expected in (("backend", BACKEND), ("geometry", GEOMETRY), ("core", CORE), ("random_streams", RANDOM_STREAMS)):
        if plan.get(name, expected) != expected:
            raise ValueError(f"This runner requires {name}={expected}")
    if not plan.get("seed_role") or not plan.get("seeds") or not plan.get("cases"):
        raise ValueError("Plan must identify seed role, seeds and cases")
    if len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("Duplicate seeds")
    if int(plan.get("replicates", 1)) < 1:
        raise ValueError("Replicates must be positive")
    names = set()
    for case in plan["cases"]:
        name = case["name"]
        if not name or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError("Case names must be unique simple filename components")
        names.add(name)
        n, p, ell = int(case["n"]), int(case["p"]), int(case["ell"])
        if not 0 < p < n or not 0 < ell <= p * (n-p):
            raise ValueError(f"Invalid Grassmann/sketch dimensions in {name}")
        Ts = case.get("Ts", (1, 2, 4, 8))
        if not Ts or len(set(Ts)) != len(Ts) or any(not isinstance(T, int) or T < 1 for T in Ts):
            raise ValueError("Periods must be distinct positive integers")
        if any(not math.isfinite(float(target)) or float(target) <= 0 for target in case.get("targets", (1e-2, 1e-4, 1e-6))):
            raise ValueError("Targets must be finite positive relative gaps")
        if int(case.get("max_iter", 100)) < 1 or float(case.get("time_limit", 300)) <= 0:
            raise ValueError("Positive iteration/time caps required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    jobs = [(case, seed, T, rep) for case in plan["cases"] for seed in plan["seeds"]
            for rep in range(int(plan.get("replicates", 1))) for T in case.get("Ts", (1, 2, 4, 8))]
    order_rng = np.random.default_rng(int(plan.get("order_seed", 731)))
    if plan.get("randomize_order", True):
        order_rng.shuffle(jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    for case, seed, T, rep in jobs:
        path = args.out / f"{case['name']}_seed{seed}_T{T}_rep{rep}.json"
        if path.exists():
            raise FileExistsError(f"Raw data are never overwritten: {path}")
    with threadpool_limits(limits=1):
        metadata = provenance(args.plan)
        chunk_size = int(plan.get("chunk_size", 16))
        workspace_bytes = int(plan.get("workspace_bytes", 128 * 1024 * 1024))
        warm_case = {"name": "warmup_dense", "n": 32, "p": 3, "ell": 4,
                     "matrix": "dense_rotated_gapped", "max_iter": 3,
                     "time_limit": 60.0, "targets": [1e-2]}
        warm_problem = build_problem(warm_case, 902)
        for T in (1, 4):
            run_one(warm_problem, warm_case, 902, T, chunk_size=chunk_size, workspace_bytes=workspace_bytes)
        del warm_problem
        cache = {}
        for index, (case, seed, T, rep) in enumerate(jobs):
            assert_sources_unchanged(metadata["source_sha256"])
            if sha(args.plan) != metadata["plan_sha256"]:
                raise RuntimeError("Frozen plan changed during execution")
            name = f"{case['name']}_seed{seed}_T{T}_rep{rep}.json"
            path = args.out / name
            print(f"START {index+1}/{len(jobs)} {name}", flush=True)
            key = problem_key(case, seed)
            try:
                if key not in cache:
                    try:
                        cache[key] = build_problem(case, seed)
                    except Exception:
                        cache[key] = {"problem_generation_error": traceback.format_exc()}
                problem = cache[key]
                if isinstance(problem, dict):
                    raise RuntimeError(problem["problem_generation_error"])
                result = run_one(problem, case, seed, T, replicate=rep, validate=bool(plan.get("validate", False)),
                                 chunk_size=int(case.get("chunk_size", chunk_size)),
                                 workspace_bytes=int(case.get("workspace_bytes", workspace_bytes)))
                # Fail visibly rather than serializing invalid performance data.
                json.dumps(result, allow_nan=False)
                payload = {"metadata": metadata, "seed_role": plan["seed_role"], "result": result}
                print(f"DONE {result['elapsed_s']:.3f}s, iter={result['iterations']}, rel_gap={result['final_relative_gap']:.3g}, cubic_fail={result['checks']['cubic_failures']}, reused={result['counts'].get('transport_gradient_reorth', 0)}", flush=True)
            except Exception:
                payload = {"metadata": metadata, "seed_role": plan["seed_role"], "failed": True,
                           "case": case, "seed": int(seed), "T": int(T), "replicate": int(rep),
                           "backend": BACKEND, "geometry": GEOMETRY, "core": CORE,
                           "random_streams": RANDOM_STREAMS, "diagnostic_run": bool(plan.get("validate", False)),
                           "error": traceback.format_exc()}
                cached = cache.get(key)
                if isinstance(cached, Problem):
                    payload["problem"] = cached.metadata
                    payload.update(cached.seeds)
                print(payload["error"], flush=True)
            with path.open("x") as stream:
                json.dump(builtin(payload), stream, indent=2, allow_nan=False)
                stream.write("\n")
        assert_sources_unchanged(metadata["source_sha256"])
        if sha(args.plan) != metadata["plan_sha256"]:
            raise RuntimeError("Frozen plan changed during execution")


if __name__ == "__main__":
    main()
