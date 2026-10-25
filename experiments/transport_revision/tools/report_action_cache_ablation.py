"""Evidence-preserving report of the frozen four-arm fixed-A ablation.

This module never imports a solver or runs an experiment. Execute only after
the serial timing batch completes. Decimal timings are converted to exact
SymPy rationals before repeat medians, paired ratios and seed summaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from sympy import Rational

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
CACHE_CORE = "dense_fixed_A_action_cache_current"
DIRECT_CORE = "dense_large_block_current"
ARMS = ("common_fresh", "optimized_fresh_control",
        "common_transport_direct", "transport_action_cache")
LABELS = {
    "common_fresh": "Fresh (common oracle)",
    "optimized_fresh_control": "Fresh (original optimized)",
    "common_transport_direct": "Transport, direct actions (T=8)",
    "transport_action_cache": "Transport + action cache (T=8)",
}
COMPARISONS = (
    ("ordinary_vs_common_fresh", "common_fresh", "common_transport_direct"),
    ("ordinary_vs_optimized_fresh", "optimized_fresh_control", "common_transport_direct"),
    ("ordinary_vs_best_tested_fresh_implementation", "best_fresh", "common_transport_direct"),
    ("cache_vs_common_fresh", "common_fresh", "transport_action_cache"),
    ("cache_vs_optimized_fresh", "optimized_fresh_control", "transport_action_cache"),
    ("cache_vs_best_tested_fresh_implementation", "best_fresh", "transport_action_cache"),
    ("cache_vs_ordinary_transport", "common_transport_direct", "transport_action_cache"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(value):
    return Rational(str(value))


def stats(values):
    ordered = sorted(exact(value) for value in values)
    if not ordered:
        return {"count": 0, "median": None, "min": None, "max": None, "exact": {}}
    n = len(ordered)
    middle = ordered[n//2] if n % 2 else (ordered[n//2-1]+ordered[n//2])/2
    answer = {"median": middle, "min": ordered[0], "max": ordered[-1]}
    return {"count": n, **{key: float(value) for key, value in answer.items()},
            "exact": {key: str(value) for key, value in answer.items()}}


def identity(run):
    return (run["case"]["name"], run["seed"], run["T"], run.get("replicate", 0))


def expected_identities(plan):
    return {(case["name"], seed, T, rep) for case in plan["cases"]
            for seed in plan["seeds"] for T in case["Ts"]
            for rep in range(plan["replicates"])}


def arm_for(case, T):
    if case.get("optimized_direct_control", False):
        return "optimized_fresh_control" if T == 1 and not case.get("action_cache_enabled") else None
    if case.get("action_cache_enabled", False):
        return "transport_action_cache" if T == 8 else None
    return {1: "common_fresh", 8: "common_transport_direct"}.get(T)


def group_for(case, plan):
    for group in plan["ablation_groups"]:
        names = (group["common_direct_case"], group["cached_case"],
                 group["optimized_fresh_control_case"])
        if case["name"] in names:
            return group["name"]
    return None


def first_hit(run, target):
    for index, value in enumerate(run.get("history", {}).get("relative_objective_gap", [])):
        if isinstance(value, (int, float)) and math.isfinite(value) and value <= target:
            return index
    return None


def run_issues(run, planned_case, plan):
    """Audit recorded semantics, without re-running numerical computations."""
    issues = []
    if planned_case is None or run["case"] != planned_case:
        issues.append("case differs from frozen plan")
    if run["arm"] is None or run["group"] is None:
        issues.append("unknown/invalid ablation arm or group")
    if run["exception_failure"]:
        return issues
    if run.get("diagnostic_run") or run.get("checks", {}).get("action_diagnostics_measured"):
        issues.append("diagnostic run is not a timing run")
    if run.get("backend") != "verified" or run.get("geometry") != "isometric_fused_dense":
        issues.append("unexpected backend or geometry tag")
    if run.get("random_streams") != "spawned3":
        issues.append("unexpected random stream protocol")
    for name in ("chunk_size", "workspace_bytes"):
        expected = (planned_case or {}).get(name, plan[name])
        if run.get(name) != expected:
            issues.append(name+" differs from frozen plan")
    control = run["arm"] == "optimized_fresh_control"
    enabled = run["arm"] == "transport_action_cache"
    expected_core = DIRECT_CORE if control else CACHE_CORE
    if run.get("core") != expected_core:
        issues.append("actual core tag does not match labeled arm")
    if run.get("action_cache_enabled") is not enabled or run.get("optimized_direct_control") is not control:
        issues.append("recorded cache/control switches do not match labeled arm")
    problem = run.get("problem", {})
    checks, counts = run.get("checks", {}), run.get("counts", {})
    oracle, hist = run.get("dense_oracle_stats", {}), run.get("history", {})
    expected_kernel = ("general_dense_current_hessian_large_block" if control
                       else "fixed_dense_A_linear_action_cache_current_hessian")
    if oracle.get("kernel") != expected_kernel:
        issues.append("oracle kernel discriminator differs from labeled arm")
    if checks.get("diagnostic_checks_measured") is not False:
        issues.append("timed run diagnostic flag missing or not false")
    if checks.get("reduced_core_backend") != expected_core:
        issues.append("instrumented backend differs from labeled core")
    for flag in ("reference_passed_to_solver", "generator_Q_passed_to_solver",
                 "design_eigenvalues_passed_to_solver"):
        if problem.get(flag) is not False:
            issues.append("missing/invalid evaluator-only flag: " + flag)
    if not 0 < problem.get("offdiagonal_frobenius_fraction", 0) <= 1:
        issues.append("missing/nonpositive offdiagonal matrix fraction")
    for field in ("matrix_sha256", "initial_point_sha256"):
        if not run.get(field) or run[field] != problem.get(field):
            issues.append("missing/inconsistent " + field)
    if run.get("f_star") != problem.get("f_star_numerical_reference"):
        issues.append("reference differs from actual dense-matrix eigensolve")
    required = ("f", "time", "grad_norm", "objective_gap", "relative_objective_gap")
    length = len(hist.get("f", []))
    if not length or any(len(hist.get(field, [])) != length for field in required):
        return issues + ["missing/inconsistent histories"]
    if any(not math.isfinite(x) for field in required for x in hist[field]):
        return issues + ["nonfinite history"]
    if hist["time"][0] != 0 or any(b < a for a, b in zip(hist["time"], hist["time"][1:])):
        issues.append("loop time must start at zero and be nondecreasing")
    initial_gap = run.get("initial_gap", 0)
    if not math.isfinite(initial_gap) or initial_gap <= 0:
        return issues + ["invalid initial objective gap"]
    for i in range(length):
        gap = hist["f"][i] - run["f_star"]
        if not math.isclose(hist["objective_gap"][i], gap, rel_tol=1e-13, abs_tol=1e-13):
            issues.append("absolute gap history mismatch")
            break
        if not math.isclose(hist["relative_objective_gap"][i], gap/initial_gap,
                            rel_tol=1e-13, abs_tol=1e-15):
            issues.append("relative gap history mismatch")
            break
    if not math.isclose(hist["objective_gap"][0], initial_gap, rel_tol=1e-13, abs_tol=1e-13):
        issues.append("initial objective gap mismatch")
    point_error = checks.get("final_point_orth_error")
    if point_error is None or not math.isfinite(point_error) or point_error > 1e-8:
        issues.append("final point orthogonality sanity check failed/missing")
    discrepancy = checks.get("reference_discrepancy_over_smallest_target")
    if discrepancy is None or not math.isfinite(discrepancy) or discrepancy > 0.01:
        issues.append("reference discrepancy sanity check failed/missing")
    gap_tolerance = checks.get("negative_gap_numerical_tolerance")
    if gap_tolerance is None or not math.isfinite(gap_tolerance) or gap_tolerance < 0:
        issues.append("negative gap tolerance missing/invalid")
    elif min(hist["objective_gap"]) < -gap_tolerance:
        issues.append("recorded objective violates reference gap tolerance")
    for target in run["case"]["targets"]:
        stored = next((value for key, value in run.get("targets", {}).items()
                       if float(key) == float(target)), None)
        hit = first_hit(run, target)
        if stored is None:
            issues.append(f"missing target record {target}")
        elif stored.get("reached") != (hit is not None) or stored.get("iteration") != hit:
            issues.append(f"first-hit record mismatch {target}")
        elif hit is not None and (stored.get("time_s") != hist["time"][hit]
                                  or stored.get("grad_norm") != hist["grad_norm"][hit]):
            issues.append(f"first-hit time/gradient mismatch {target}")
        if hit is not None and hist["time"][hit] <= 0:
            issues.append(f"nonpositive first-hit time {target}")
    trials = length - 1
    columns = sum(hist.get("ell", []))
    if run.get("iterations") != trials or len(hist.get("ell", [])) != trials:
        issues.append("trial/frame-width history mismatch")
    for label, count in (
        ("instrumented core", checks.get("current_core_calls")),
        ("timed core", counts.get("current_reduced_core")),
        ("oracle core", oracle.get("current_core_calls")),
        ("cubic", counts.get("cubic_subproblem")),
        ("retraction", counts.get("retraction")),
        ("trial objective", counts.get("trial_objective")),
    ):
        if count != trials:
            issues.append(f"{label} call count differs from trial count")
    if checks.get("current_core_columns") != columns or oracle.get("exact_current_core_columns") != columns:
        issues.append("current core direction count mismatch")
    if counts.get("initial_objective_gradient") != 1:
        issues.append("initial gradient phase count is not one")
    if counts.get("objective_gradient") != sum(hist.get("accepted", [])):
        issues.append("accepted-gradient counter mismatch")
    if any(len(hist.get(field, [])) != trials for field in ("accepted", "rho", "ell")):
        issues.append("per-trial acceptance/rho/frame histories mismatch")
    if len(checks.get("coordinate_tail_step_fractions", [])) != trials:
        issues.append("non-gradient step fraction trace length differs from trials")
    if checks.get("cubic_failures", 0):
        issues.append("one or more cubic failures")
    p = run["case"]["p"]
    if control:
        if oracle.get("core_dense_A_rhs_columns") != columns*p:
            issues.append("optimized direct control dense RHS count mismatch")
    else:
        if oracle.get("action_updates_enabled") is not enabled:
            issues.append("oracle update switch mismatch")
        hits = oracle.get("action_cache_core_hits", 0)
        rebases = oracle.get("direct_action_rebase_calls", 0)
        if hits + rebases != trials:
            issues.append("cache-hit plus rebase count differs from trial count")
        if not enabled and (hits or oracle.get("polar_action_updates", 0)
                            or oracle.get("gradient_action_updates", 0)):
            issues.append("cache-disabled arm contains action updates or hits")
        for purpose, counter in (
            ("action_rebase", "direct_action_rebase_directions"),
            ("action_transport", "polar_action_updates"),
            ("action_gradient", "gradient_action_updates"),
        ):
            if oracle.get(purpose+"_dense_A_rhs_columns", 0) != p*oracle.get(counter, 0):
                issues.append(purpose+" dense RHS count mismatch")
        if oracle.get("diagnostic_action_checks", 0) or oracle.get("diagnostic_seconds", 0):
            issues.append("measured cache diagnostics contaminated timing run")
    for suffix in ("dense_A_rhs_columns", "dense_A_matmul_calls"):
        purposes = [value for key, value in oracle.items()
                    if key.endswith("_"+suffix) and key != suffix]
        if not purposes or sum(purposes) != oracle.get(suffix):
            issues.append("sum of purpose counters differs from " + suffix)
    if oracle.get("peak_explicit_workspace_estimate_bytes", 0) > run.get("workspace_bytes", 0):
        issues.append("explicit transient workspace estimate exceeds budget")
    return issues


def read_runs(directory, plan):
    cases = {case["name"]: case for case in plan["cases"]}
    runs, duplicates = [], []
    seen = set()
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if "metadata" not in payload or not ("result" in payload or payload.get("failed")):
            raise ValueError(f"Not a raw experiment record: {path}")
        run = dict(payload.get("result", payload))
        run.update(
            source=str(path.resolve()), source_sha256=sha(path),
            metadata=payload["metadata"], seed_role=payload.get("seed_role"),
            exception_failure=bool(payload.get("failed")),
            replicate=run.get("replicate", 0),
        )
        run["arm"] = arm_for(run["case"], run["T"])
        run["group"] = group_for(run["case"], plan)
        run["issues"] = run_issues(run, cases.get(run["case"]["name"]), plan)
        key = identity(run)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        runs.append(run)
    if not runs:
        raise ValueError("No raw run records found")
    for run in runs:
        if identity(run) in duplicates:
            run["issues"].append("duplicate run identity")
    return runs, duplicates


def audit_provenance(runs, duplicates, plan, plan_path):
    expected, observed = expected_identities(plan), {identity(run) for run in runs}
    metadata = runs[0]["metadata"]
    plan_hash = sha(plan_path)
    issues = []
    if len(expected) != 120:
        issues.append("frozen four-arm plan does not contain120 expected jobs")
    if any(run["metadata"] != metadata for run in runs):
        issues.append("mixed invocation/source/environment metadata")
    if any(run["seed_role"] != plan["seed_role"] for run in runs):
        issues.append("recorded seed role differs from plan")
    if any(run["metadata"].get("plan_sha256") != plan_hash for run in runs):
        issues.append("recorded plan hash differs from supplied frozen plan")
    hashes = metadata.get("source_sha256", {})
    required_paths = (list((REPO/"code-grassmann/rsncn").glob("*.py"))
                      + list(HERE.glob("*.py")) + list((HERE/"dense").glob("*.py"))
                      + list((HERE/"dense_opt").glob("*.py"))
                      + list((HERE/"dense_action_cache").glob("*.py")))
    required_sources = {str(path.relative_to(REPO)) for path in required_paths}
    missing_source_keys = sorted(required_sources-set(hashes))
    unexpected_source_keys = sorted(set(hashes)-required_sources)
    if missing_source_keys or unexpected_source_keys:
        issues.append("execution-source inventory differs from required original/dense/optimized/cache modules")
    changed = [name for name, value in hashes.items()
               if not (REPO/name).is_file() or sha(REPO/name) != value]
    if not hashes or changed:
        issues.append("recorded execution sources missing or changed")
    if observed-expected:
        issues.append("unexpected run identities")
    if duplicates:
        issues.append("duplicate run identities")
    for run in runs:
        run["issues"].extend("global provenance: "+issue for issue in issues)
    return {
        "expected_run_count": len(expected), "observed_record_count": len(runs),
        "missing_jobs": sorted(expected-observed), "unexpected_jobs": sorted(observed-expected),
        "duplicate_jobs": duplicates, "complete_expected_job_set": expected == observed and not duplicates,
        "global_issues": issues, "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan_hash, "recorded_sources_match_current": bool(hashes) and not changed,
        "changed_or_missing_sources": changed, "metadata": metadata,
        "required_source_count": len(required_sources), "recorded_source_count": len(hashes),
        "missing_source_keys": missing_source_keys, "unexpected_source_keys": unexpected_source_keys,
        "report_script": str(Path(__file__).resolve()), "report_script_sha256": sha(__file__),
    }


def check_input_pairing(runs):
    groups = defaultdict(list)
    for run in runs:
        if not run["exception_failure"]:
            groups[(run["group"], run["seed"])].append(run)
    answer = []
    fields = ("matrix_sha256", "initial_point_sha256", "matrix_seed",
              "initial_point_seed", "sketch_seed", "f_star", "initial_gap",
              "backend", "geometry", "random_streams", "chunk_size", "workspace_bytes")
    for (group, seed), members in sorted(groups.items(), key=lambda item: str(item[0])):
        issues = []
        reference = members[0]
        for field in fields:
            if reference.get(field) is None or any(run.get(field) != reference.get(field) for run in members):
                issues.append(field)
        options = []
        for run in members:
            values = dict(run.get("options", {}))
            for field in ("transport_sketch", "transport_refresh_every"):
                values.pop(field, None)
            options.append(values)
        if any(value != options[0] for value in options):
            issues.append("solver options apart from intended period/transport switches")
        if issues:
            for run in members:
                run["issues"].append("input/options pairing mismatch: "+", ".join(issues))
        answer.append({"group": group, "seed": seed, "issues": issues,
                       "successful_records_checked": len(members),
                       **{field: reference.get(field) for field in fields}})
    return answer


def make_target_rows(runs, plan):
    rows = []
    for run in runs:
        for target in sorted(set(run["case"].get("targets", [])) | {plan["primary_target"]}):
            hit = first_hit(run, target)
            valid = not run["exception_failure"] and not run["issues"]
            hist = run.get("history", {})
            if hit is not None and any(len(hist.get(field, [])) <= hit for field in
                                       ("f", "time", "grad_norm")):
                hit = None
            usable = valid and hit is not None
            relative_grad = hist.get("relative_grad_norm", [])
            rows.append({
                "group": run["group"], "case": run["case"]["name"], "n": run["case"]["n"],
                "ell": run["case"]["ell"], "arm": run["arm"], "seed": run["seed"],
                "T": run["T"], "replicate": run["replicate"], "target": target,
                "recorded_hit": hit is not None, "valid_run": valid, "usable_hit": usable,
                "first_hit_trial": hit,
                "first_hit_time_s": hist["time"][hit] if hit is not None else None,
                "first_hit_grad_norm": hist["grad_norm"][hit] if hit is not None else None,
                "first_hit_relative_grad_norm": relative_grad[hit]
                if hit is not None and len(relative_grad) > hit else None,
                "exception_failure": run["exception_failure"], "issues": " | ".join(run["issues"]),
                "exception_text": run.get("error", ""),
                "iterations": run.get("iterations"), "full_loop_time_s": run.get("elapsed_s"),
                "full_external_wall_s": run.get("external_wall_s"),
                "matrix_sha256": run.get("matrix_sha256"),
                "initial_point_sha256": run.get("initial_point_sha256"),
                "source": run["source"], "source_sha256": run["source_sha256"],
            })
    return rows


def repeated_histories(runs):
    groups = defaultdict(list)
    for run in runs:
        if not run["exception_failure"]:
            groups[(run["group"], run["arm"], run["seed"])].append(run)
    answer = []
    fields = ("f", "grad_norm", "sigma", "rho", "accepted", "ell",
              "objective_gap", "relative_objective_gap")
    for (group, arm, seed), members in sorted(groups.items(), key=lambda item: str(item[0])):
        baseline = min(members, key=lambda run: run["replicate"])
        mismatches = []
        for run in members:
            changed = [field for field in fields
                       if run.get("history", {}).get(field) != baseline.get("history", {}).get(field)]
            if changed:
                mismatches.append({"replicate": run["replicate"], "fields": changed})
        answer.append({"group": group, "arm": arm, "seed": seed,
                       "observed_replicates": sorted(run["replicate"] for run in members),
                       "exact_history_consistency": not mismatches, "mismatches": mismatches,
                       "policy": "A mismatch is reported, not used to discard a slow timing."})
    return answer


def aggregate_repeats(rows, plan):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["group"], row["arm"], row["seed"], row["target"])].append(row)
    result = []
    targets = sorted({target for case in plan["cases"] for target in case["targets"]})
    for group in plan["ablation_groups"]:
        for arm in ARMS:
            for seed in plan["seeds"]:
                for target in targets:
                    members = groups.get((group["name"], arm, seed, target), [])
                    reasons = []
                    if sorted(row["replicate"] for row in members) != list(range(plan["replicates"])):
                        reasons.append("missing/duplicate timing repeats")
                    if any(not row["usable_hit"] for row in members):
                        reasons.append("at least one failed, invalid or missed-target repeat")
                    valid = not reasons
                    times = stats(row["first_hit_time_s"] for row in members) if valid else stats([])
                    result.append({
                        "group": group["name"], "n": group["n"], "arm": arm,
                        "seed": seed, "target": target, "valid": valid, "exclusion_reasons": reasons,
                        "time_s": times,
                        "first_hit_trial": stats(row["first_hit_trial"] for row in members) if valid else stats([]),
                        "first_hit_grad_norm": stats(row["first_hit_grad_norm"] for row in members) if valid else stats([]),
                        "first_hit_iteration_consistent": len({row["first_hit_trial"] for row in members}) <= 1,
                        "repeats": members,
                    })
    return result


def paired_summaries(cells, plan):
    table = {(cell["group"], cell["arm"], cell["seed"], cell["target"]): cell for cell in cells}
    summaries, pair_rows = [], []
    targets = sorted({cell["target"] for cell in cells})
    for group in plan["ablation_groups"]:
        for target in targets:
            for name, numerator, denominator in COMPARISONS:
                pairs, excluded = [], []
                for seed in plan["seeds"]:
                    right = table[(group["name"], denominator, seed, target)]
                    left_arms = ("common_fresh", "optimized_fresh_control") if numerator == "best_fresh" else (numerator,)
                    lefts = [table[(group["name"], arm, seed, target)] for arm in left_arms]
                    invalid = [cell["arm"] for cell in [*lefts, right] if not cell["valid"]]
                    if invalid:
                        excluded.append({"seed": seed, "reason": "incomplete/invalid3-repeat cells: "+", ".join(invalid)})
                        continue
                    left = min(lefts, key=lambda cell: exact(cell["time_s"]["exact"]["median"]))
                    a, b = exact(left["time_s"]["exact"]["median"]), exact(right["time_s"]["exact"]["median"])
                    if a <= 0 or b <= 0:
                        excluded.append({"seed": seed, "reason": "nonpositive aggregated target time"})
                        continue
                    value = a/b
                    pair = {"group": group["name"], "n": group["n"], "target": target,
                            "comparison": name, "seed": seed, "speedup": float(value),
                            "exact_speedup": str(value), "baseline_arm": left["arm"],
                            "comparison_arm": denominator, "baseline_time_s": float(a),
                            "comparison_time_s": float(b), "exact_baseline_time_s": str(a),
                            "exact_comparison_time_s": str(b)}
                    pairs.append(pair)
                    pair_rows.append(pair)
                summaries.append({
                    "group": group["name"], "n": group["n"], "target": target, "comparison": name,
                    "definition": numerator+" time / "+denominator+" time; values >1 favor "+denominator,
                    "paired_speedup": stats(pair["exact_speedup"] for pair in pairs),
                    "pair_count": len(pairs), "comparison_faster_pairs": sum(bool(exact(pair["exact_speedup"]) > 1) for pair in pairs),
                    "baseline_faster_pairs": sum(bool(exact(pair["exact_speedup"]) < 1) for pair in pairs),
                    "pairs": pairs, "excluded_pairs": excluded,
                })
    return summaries, pair_rows


def cross_variant_trajectories(runs, plan):
    """Compare direct/cache T8 through their accuracy targets, not bitwise."""
    rows = []
    for group in plan["ablation_groups"]:
        for seed in plan["seeds"]:
            for rep in range(plan["replicates"]):
                pair = []
                for arm in ("common_transport_direct", "transport_action_cache"):
                    matches = [run for run in runs if run["group"] == group["name"]
                               and run["seed"] == seed and run["replicate"] == rep and run["arm"] == arm]
                    pair.append(matches[0] if len(matches) == 1 else None)
                row = {"group": group["name"], "seed": seed, "replicate": rep,
                       "policy": "Different variants need not be bitwise identical. Prefix ends at the later first primary-target hit, limited to their common recorded history; late roundoff/acceptance differences are not a run-invalidating test."}
                if any(run is None or run["exception_failure"] or run["issues"] for run in pair):
                    row["available"] = False
                    row["reason"] = "missing, failed or invalid member"
                    rows.append(row)
                    continue
                direct, cached = pair
                hd, hc = first_hit(direct, plan["primary_target"]), first_hit(cached, plan["primary_target"])
                row.update(available=True, direct_first_hit_trial=hd, cache_first_hit_trial=hc,
                           first_hit_trial_difference=hc-hd if hd is not None and hc is not None else None,
                           direct_source=direct["source"], cache_source=cached["source"])
                common = min(len(direct["history"]["f"]), len(cached["history"]["f"]))
                end = min(common, max(hd, hc)+1) if hd is not None and hc is not None else common
                prefix_errors = [abs(exact(a)-exact(b))
                                 for a, b in zip(direct["history"]["f"][:end], cached["history"]["f"][:end])]
                all_errors = [abs(exact(a)-exact(b))
                              for a, b in zip(direct["history"]["f"][:common], cached["history"]["f"][:common])]
                maximum = max(prefix_errors, default=Rational(0))
                row.update(prefix_last_trial=end-1,
                           prefix_max_absolute_objective_difference=float(maximum),
                           exact_prefix_max_absolute_objective_difference=str(maximum),
                           full_common_history_max_absolute_objective_difference=float(max(all_errors, default=Rational(0))),
                           prefix_acceptance_differences=sum(
                               a != b for a, b in zip(direct["history"]["accepted"][:max(0, end-1)],
                                                     cached["history"]["accepted"][:max(0, end-1)])),
                           full_common_history_acceptance_differences=sum(
                               a != b for a, b in zip(direct["history"]["accepted"], cached["history"]["accepted"])))
                rows.append(row)
    return rows


def workload_records(runs):
    return [{
        "group": run["group"], "arm": run["arm"], "seed": run["seed"], "replicate": run["replicate"],
        "source": run["source"], "exception_failure": run["exception_failure"], "issues": run["issues"],
        "iterations": run.get("iterations"), "accepted_steps": run.get("accepted_steps"),
        "full_loop_time_s": run.get("elapsed_s"), "full_external_wall_s": run.get("external_wall_s"),
        "counts": run.get("counts", {}), "timings_s": run.get("timings_s", {}),
        "dense_oracle_stats": run.get("dense_oracle_stats", {}), "checks": run.get("checks", {}),
        "problem": run.get("problem", {}),
        "action_cache_hits_observed": run.get("dense_oracle_stats", {}).get("action_cache_core_hits", 0),
        "cache_enabled_arm_had_positive_hits": (
            run.get("dense_oracle_stats", {}).get("action_cache_core_hits", 0) > 0
            if run["arm"] == "transport_action_cache" and not run["exception_failure"] else None),
        "scope": "Whole-run measurements, not prefix-to-target work. Persistent action storage excludes held frame/point references, caller arrays and opaque BLAS scratch; it is not RSS.",
    } for run in runs]


def measured_validation(paths):
    result = []
    for path in paths:
        text = path.read_text()
        try:
            content = json.loads(text)
        except json.JSONDecodeError:
            content = text
        item = {"source": str(path.resolve()), "source_sha256": sha(path), "content": content,
                "scope": "Separately supplied validation evidence, not timed performance data. Preserve its own measured/pass/failure semantics."}
        if isinstance(content, dict):
            comparisons = content.get("comparisons", [])
            item["summary"] = {
                "tests_run": content.get("tests_run"), "successful": content.get("successful"),
                "diagnostic_runs": len(content.get("diagnostic_runs", [])),
                "trajectory_pairs": len(comparisons),
                "max_action_relative_error": stats(entry["measured_max_action_relative_error"]
                                                   for entry in comparisons if "measured_max_action_relative_error" in entry)["max"],
                "max_core_relative_error": stats(entry["measured_max_core_relative_error"]
                                                 for entry in comparisons if "measured_max_core_relative_error" in entry)["max"],
                "max_absolute_objective_difference": stats(entry["max_objective_difference"]
                                                           for entry in comparisons if "max_objective_difference" in entry)["max"],
                "purpose": content.get("purpose"),
                "source_hashes_match_current": all((REPO/name).is_file() and sha(REPO/name) == digest
                                                  for name, digest in content.get("source_sha256",
                                                      content.get("local_python_source_sha256", {})).items()),
            }
        result.append(item)
    return result


def plots(runs, plan, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14, "lines.linewidth": 1.5, "axes.linewidth": 0.8,
                         "axes.facecolor": "white", "figure.facecolor": "white",
                         "axes.grid": False, "axes.spines.top": True, "axes.spines.right": True,
                         "xtick.direction": "out", "ytick.direction": "out"})
    colors = dict(zip(ARMS, ("#f6c7c4", "#f17c7c", "#c9443e", "#8c1c13")))
    styles = dict(zip(ARMS, ("-", (0, (5, 2)), (0, (3, 1.5)), (0, (1.5, 1.2)))))
    short = {"common_fresh": "Fresh (common)", "optimized_fresh_control": "Fresh (optimized)",
             "common_transport_direct": r"Transport ($T=8$)",
             "transport_action_cache": r"Transport + cache ($T=8$)"}
    manifests = []
    for group in plan["ablation_groups"]:
        members = [run for run in runs if run["group"] == group["name"]
                   and run["seed"] == plan["representative_seed"]
                   and run["replicate"] == plan["representative_replicate"]]
        omissions, curves, files = {}, {}, []

        def draw(ax, timed):
            for arm in ARMS:
                matches = [run for run in members if run["arm"] == arm]
                if len(matches) != 1:
                    omissions[arm] = "Missing or duplicated preselected trajectory; no replacement selected."
                    continue
                run = matches[0]
                if run["exception_failure"] or run["issues"]:
                    omissions[arm] = "Failed or invalid raw trajectory; retained in report, not plotted as valid."
                    continue
                hit = first_hit(run, plan["primary_target"])
                hist = run["history"]
                end = hit+1 if hit is not None else len(hist["f"])
                gaps = hist["objective_gap"][:end]
                invalid = sum(value <= 0 or not math.isfinite(value) for value in gaps)
                if invalid:
                    omissions[arm] = f"{invalid} nonpositive/nonfinite logarithmic points omitted without flooring."
                y = [value if value > 0 and math.isfinite(value) else float("nan") for value in gaps]
                x = hist["time"][:end] if timed else list(range(end))
                ax.plot(x, y, color=colors[arm], linestyle=styles[arm], label=short[arm])
                curves[arm] = {"source": run["source"], "source_sha256": run["source_sha256"],
                               "points": end, "first_hit_trial": hit,
                               "first_hit_time_s": hist["time"][hit] if hit is not None else None,
                               "transformation": "Raw curve to first recorded relative-gap target, or full curve for miss; no smoothing, interpolation, aggregation or time rescaling."}
            ax.set(xlabel="time [s]" if timed else "iteration",
                   ylabel=r"$f-f_{\mathrm{ref}}$", yscale="log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.legend(loc="upper right", frameon=False, labelspacing=0.25)
            ax.grid(False)

        prefix = group["name"]+f"_four_arm_seed{plan['representative_seed']}_rep{plan['representative_replicate']}"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, timed in zip(axes, (False, True)):
            draw(ax, timed)
        fig.tight_layout()
        path = output/(prefix+".png")
        fig.savefig(path, dpi=200)
        files.append(str(path.resolve()))
        plt.close(fig)
        for timed, name in ((False, "iteration"), (True, "time")):
            fig, ax = plt.subplots(figsize=(6, 4))
            draw(ax, timed)
            fig.tight_layout()
            for extension in ("png", "eps"):
                path = output/(prefix+"_"+name+"."+extension)
                fig.savefig(path, dpi=200)
                files.append(str(path.resolve()))
            plt.close(fig)
        manifests.append({
            "group": group["name"], "n": group["n"], "seed": plan["representative_seed"],
            "replicate": plan["representative_replicate"], "target": plan["primary_target"],
            "curves": curves, "omissions": omissions, "files": files,
            "caption": "Four preselected individual dense-matrix trajectories at fixed ell40: fresh/common oracle, fresh/original optimized implementation, ordinary transported sketch T8/direct actions, and transported sketch T8/action reuse. Absolute objective gap uses a numerical dense eigensolver reference. Curves end at first recorded q<=1e-6 hit, or remain full-length on a miss. Time is recorded loop time, excluding initialization. No T2/T4 curves, smoothing, averaging, interpolation, or fastest-seed selection.",
            "style": {"colors": colors, "line_styles": styles, "legend_fontsize": 14,
                      "combined_inches": [12, 4], "single_panel_inches": [6, 4]},
        })
    return manifests


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def number(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_report(plan, audit, runs, cells, summaries, pairing, repeats, cross_variant, manifest, validation):
    primary = plan["primary_target"]
    lines = [
        "# Fixed-A action-cache ablation",
        "",
        "This is a new fixed-dense-A implementation ablation, separate from the completed ordinary-transport search. It is not a theorem about general manifold Hessians or finite-precision exactness.",
        "",
        f"Expected/observed raw jobs: {audit['expected_run_count']}/{audit['observed_record_count']}. Complete expected job set: {audit['complete_expected_job_set']}.",
        f"Plan SHA256: {audit['plan_sha256']}. Execution sources still match: {audit['recorded_sources_match_current']}.",
        f"Global provenance issues: {audit['global_issues']}. Missing jobs: {audit['missing_jobs']}. Unexpected jobs: {audit['unexpected_jobs']}.",
        f"Exception records: {sum(run['exception_failure'] for run in runs)}; records with integrity issues: {sum(bool(run['issues']) for run in runs)}; pairing groups with issues: {sum(bool(item['issues']) for item in pairing)}.",
        "",
        "## Measurement and limits",
        "",
        "First take each arm/seed's median of all3 valid target-reaching repeats, then compute within-seed ratios, then report median/min–max over5 seeds. Failed, invalid, incomplete and missed-target cells are excluded explicitly, never silently replaced with successful repeats. Timing repeats are not independent problem instances; ranges are not confidence intervals or significance tests.",
        "",
        "Ratios above1 favor the named comparison arm. The fastest-tested fresh implementation is the per-seed minimum of the two fresh implementations' repeat medians at the SAME ell40. It is an oracle implementation choice, not a globally best fresh optimizer or a sketch-size sweep. Earlier ell80/320 results are not new-seed matched controls.",
        "",
        "Primary time is first recorded loop time to q=(f-f_reference)/(f_initial-f_reference)<=1e-6. All curves and tables use the independent numerical eigensolve of actual dense A, not an interval-certified exact reference. Objective gap alone does not certify stationarity. Matrix construction/reference evaluation and initial solver QR/geometry are excluded. Full external solve wall time includes solver initialization and is retained separately.",
        "",
        "## Results at every recorded target",
        "",
    ]
    for group in plan["ablation_groups"]:
        lines.extend([f"### {group['name']}", ""])
        for target in sorted({cell["target"] for cell in cells}):
            lines.extend([f"Target q <= {target:g}", "",
                          "| Arm | Complete seed cells | Median target time [s] | Seed-median min–max [s] | First-hit trial range |",
                          "| --- | --- | --- | --- | --- |"])
            for arm in ARMS:
                selected = [cell for cell in cells if cell["group"] == group["name"]
                            and cell["arm"] == arm and cell["target"] == target and cell["valid"]]
                values = stats(cell["time_s"]["exact"]["median"] for cell in selected)
                trials = [row["first_hit_trial"] for cell in selected for row in cell["repeats"]]
                trial_range = f"{min(trials)}–{max(trials)}" if trials else "—"
                lines.append(f"| {LABELS[arm]} | {len(selected)}/{len(plan['seeds'])} | {number(values['median'])} | {number(values['min'])}–{number(values['max'])} | {trial_range} |")
            lines.extend(["", "| Comparison | Paired seeds | Median speedup | Min–max | Faster pairs |",
                          "| --- | --- | --- | --- | --- |"])
            for summary in summaries:
                if summary["group"] == group["name"] and summary["target"] == target:
                    value = summary["paired_speedup"]
                    lines.append(f"| {summary['comparison']} | {summary['pair_count']}/{len(plan['seeds'])} | {number(value['median'])} | {number(value['min'])}–{number(value['max'])} | {summary['comparison_faster_pairs']} |")
            lines.append("")
    incomplete = [cell for cell in cells if not cell["valid"]]
    lines.extend([
        "## Failures, misses and diagnostics", "",
        f"Incomplete/invalid arm–seed–target cells: {len(incomplete)}. All reasons and repeats are retained in repeat_cells.json and raw_target_rows.csv; every paired exclusion is retained in summary.json.",
        f"Repeat-history consistency warnings: {sum(not item['exact_history_consistency'] for item in repeats)}. These warnings do not discard slow timings.",
        "Direct-T8 versus cached-T8 first-hit iterations, objective differences through the target, and later acceptance differences are separately preserved in cross_variant_trajectories.json. The variants are not required to be bitwise identical; late near-stationarity differences do not invalidate performance records.",
        f"Separately supplied validation files: {len(validation)}. Timed-run analytic/zero placeholders are NOT new residual measurements; measured validation evidence is kept separately in measured_validation.json.",
        *[f"Validation evidence {Path(item['source']).name}, SHA256 {item['source_sha256']}: {item.get('summary', 'see original content')}. Initial test-configuration failure records, if supplied, remain test evidence and are not timed algorithm failures."
          for item in validation],
        "",
        "## Whole-run oracle work and memory", "",
        "These are complete-run counts and storage, NOT work accumulated only to the accuracy target. Nested phases must not be summed as disjoint totals. Core calls are checked against trial count in the instrumentation, timing counts, and oracle. Actual dense A-products can decrease under action reuse; they are not required to equal the number of core directions.",
        "",
        "| Group / arm | Valid runs | Trial range | Dense A RHS range | Cache-hit range | Rebase range | Peak action bytes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for group in plan["ablation_groups"]:
        for arm in ARMS:
            members = [run for run in runs if run["group"] == group["name"] and run["arm"] == arm
                       and not run["exception_failure"] and not run["issues"]]
            def extent(name, oracle=False):
                values = [run.get("dense_oracle_stats", {}).get(name) if oracle else run.get(name) for run in members]
                values = [value for value in values if value is not None]
                return "n/a" if not values else f"{min(values)}–{max(values)}"
            memory = [run.get("dense_oracle_stats", {}).get("persistent_action_cache_peak_bytes") for run in members]
            memory = [value for value in memory if value is not None]
            lines.append(f"| {group['n']} / {LABELS[arm]} | {len(members)} | {extent('iterations')} | {extent('dense_A_rhs_columns', True)} | {extent('action_cache_core_hits', True)} | {extent('direct_action_rebase_calls', True)} | {max(memory) if memory else 'n/a'} |")
    lines.extend([
        "",
        "Persistent cached-action storage excludes held frame/point references, caller inputs and opaque BLAS scratch; temporary old+new action arrays are recorded separately. The128MiB transient workspace estimate is not a total memory/RSS limit. Full oracle/check/phase records remain in whole_run_integrity.json.",
        "",
        "## Figures and evidence", "",
        f"Figures use preselected seed{plan['representative_seed']}, replicate{plan['representative_replicate']} only. Four explicit arms are shown; no invented T2/T4 curves. Every curve uses raw points without averaging, smoothing, interpolation or time rescaling.",
        "",
    ])
    for item in manifest:
        lines.append(f"- {item['group']}: "+", ".join(Path(path).name for path in item["files"]) +
                     f"; omissions: {item['omissions']}.")
    lines.extend(["", "All raw evidence is read-only. This reporter changes only its new report directory and does not edit the manuscript, response, old reports or figures.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=HERE/"plans/dense_action_cache_ablation.json")
    parser.add_argument("--out", type=Path, default=HERE/"report/dense_action_cache_ablation")
    parser.add_argument("--validation", type=Path, action="append", default=[])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to((HERE/"report").resolve()) or output == (HERE/"report").resolve():
        raise ValueError("Reports must go in a new child directory of experiments/transport_revision/report")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty report directory; earlier report artifacts are never overwritten")
    plan = json.loads(args.plan.read_text())
    runs, duplicates = read_runs(args.input, plan)
    audit = audit_provenance(runs, duplicates, plan, args.plan)
    pairing = check_input_pairing(runs)
    rows = make_target_rows(runs, plan)
    histories = repeated_histories(runs)
    cross_variant = cross_variant_trajectories(runs, plan)
    cells = aggregate_repeats(rows, plan)
    summaries, pairs = paired_summaries(cells, plan)
    validation = measured_validation(args.validation)
    output.mkdir(parents=True, exist_ok=True)
    manifest = [] if args.no_plots else plots(runs, plan, output)
    write_csv(output/"raw_target_rows.csv", rows)
    write_csv(output/"paired_speedups.csv", pairs)
    write_json(output/"repeat_cells.json", cells)
    write_json(output/"repeat_history_consistency.json", histories)
    write_json(output/"cross_variant_trajectories.json", cross_variant)
    write_csv(output/"cross_variant_trajectories.csv", cross_variant)
    write_json(output/"input_pairing.json", pairing)
    write_json(output/"whole_run_integrity.json", workload_records(runs))
    write_json(output/"measured_validation.json", validation)
    write_json(output/"plot_manifest.json", manifest)
    write_json(output/"summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": audit, "plan": plan, "summaries": summaries,
        "raw_files": [{"source": run["source"], "source_sha256": run["source_sha256"],
                       "exception_failure": run["exception_failure"], "issues": run["issues"]} for run in runs],
        "plot_manifests": manifest,
        "measurement_scope": "Repeat medians then within-seed ratios then5-seed summaries; whole-run phases/counters not target-prefix; oracle-tuned fresh implementation baseline only at ell40.",
    })
    write_csv(output/"summary.csv", [{
        "group": item["group"], "n": item["n"], "target": item["target"],
        "comparison": item["comparison"], "pairs": item["pair_count"],
        "faster_pairs": item["comparison_faster_pairs"],
        **{key: item["paired_speedup"][key] for key in ("median", "min", "max")},
        **{"exact_"+key: item["paired_speedup"]["exact"].get(key) for key in ("median", "min", "max")},
        "excluded_pairs": item["excluded_pairs"],
    } for item in summaries])
    (output/"REPORT.md").write_text(markdown_report(
        plan, audit, runs, cells, summaries, pairing, histories, cross_variant, manifest, validation))
    print(json.dumps({
        "report_directory": str(output), "expected": audit["expected_run_count"],
        "observed": len(runs), "complete": audit["complete_expected_job_set"],
        "global_issues": audit["global_issues"], "run_integrity_issue_count": sum(bool(run["issues"]) for run in runs),
        "primary": [{"group": item["group"], "comparison": item["comparison"],
                     "speedup": item["paired_speedup"], "pairs": item["pair_count"]}
                    for item in summaries if item["target"] == plan["primary_target"]],
    }, indent=2))


if __name__ == "__main__":
    main()
