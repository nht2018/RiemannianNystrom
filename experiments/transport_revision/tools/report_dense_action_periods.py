"""Report the completed n1024 optimized-fresh/cached-transport period sweep.

No solver imports or experiments. All raw evidence and earlier reports are
read-only. Timings use complete three-repeat cells, then paired-seed ratios.
The historical runner's ablation wording is retained as provenance, NOT used
to mislabel this optimized-fresh T1 versus cached-transport T2/4/8 sweep.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess

from sympy import Rational

import report_action_cache_ablation as audit

HERE, REPO = audit.HERE, audit.REPO
PERIODS = (1, 2, 4, 8)
COLORS = {1: "#f6c7c4", 2: "#f17c7c", 4: "#c9443e", 8: "#8c1c13"}
STYLES = {1: "-", 2: (0, (5, 2)), 4: (0, (3, 1.5)), 8: (0, (1.5, 1.2))}


def load_runs(directory, plan):
    cases = {case["name"]: case for case in plan["cases"]}
    expected = audit.expected_identities(plan)
    runs, seen = [], set()
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if "metadata" not in payload or not ("result" in payload or payload.get("failed")):
            raise ValueError(f"Not a raw experiment record: {path}")
        run = dict(payload.get("result", payload))
        run.update(source=str(path.resolve()), source_sha256=audit.sha(path),
                   metadata=payload["metadata"], seed_role=payload.get("seed_role"),
                   exception_failure=bool(payload.get("failed")), replicate=run.get("replicate", 0))
        key = audit.identity(run)
        if key in seen:
            raise ValueError(f"Duplicate raw identity; no favorable repeat selection: {key}")
        seen.add(key)
        # The existing auditor's algebra/counter checks are valid for any T;
        # the period-to-case mapping is checked explicitly below.
        run["group"] = plan["display_case"]
        run["arm"] = "optimized_fresh_control" if run["T"] == 1 else "transport_action_cache"
        run["issues"] = audit.run_issues(run, cases.get(run["case"]["name"]), plan)
        if key not in expected:
            run["issues"].append("unexpected job outside frozen period plan")
        correct_case = plan["baseline_case"] if run["T"] == 1 else plan["transport_case"]
        if run["T"] not in PERIODS or run["case"]["name"] != correct_case:
            run["issues"].append("period/case mapping is not optimized fresh T1 versus cached T2/4/8")
        if not run["exception_failure"]:
            opts = run["options"]
            if opts.get("transport_sketch") != (run["T"] != 1):
                run["issues"].append("transport option differs from period label")
            if opts.get("transport_refresh_every") != (run["T"] if run["T"] != 1 else 0):
                run["issues"].append("refresh option differs from period label")
        runs.append(run)
    if not runs:
        raise ValueError(f"No raw records in {directory}")
    return runs


def provenance(runs, plan, plan_path):
    expected = audit.expected_identities(plan)
    observed = {audit.identity(run) for run in runs}
    metadata = runs[0]["metadata"]
    issues = []
    if len(expected) != 60 or plan["seeds"] != [90, 91, 92, 93, 94] or plan["replicates"] != 3:
        raise ValueError("This scoped report requires the declared sixty-run reused-seed sweep")
    if any((case["n"], case["p"], case["ell"], case["tail_top"]) != (1024, 20, 40, 18)
           for case in plan["cases"]):
        raise ValueError("Unexpected problem configuration")
    if any(run["metadata"] != metadata for run in runs):
        issues.append("mixed invocation/source/environment metadata")
    if any(run["seed_role"] != plan["seed_role"] for run in runs):
        issues.append("seed role differs from plan")
    if metadata.get("plan_sha256") != audit.sha(plan_path):
        issues.append("supplied plan differs from recorded plan hash")
    paths = (list((REPO/"code-grassmann/rsncn").glob("*.py")) + list(HERE.glob("*.py"))
             + list((HERE/"dense").glob("*.py")) + list((HERE/"dense_opt").glob("*.py"))
             + list((HERE/"dense_action_cache").glob("*.py")))
    required = {str(path.relative_to(REPO)) for path in paths}
    hashes = metadata.get("source_sha256", {})
    changed = [name for name, digest in hashes.items()
               if not (REPO/name).is_file() or audit.sha(REPO/name) != digest]
    if set(hashes) != required or changed:
        issues.append("execution source inventory/hash mismatch")
    if metadata.get("core_variant") != "fixed_dense_A_linear_action_cache_current_hessian":
        issues.append("wrong runner specialization metadata")
    if observed-expected:
        issues.append("unplanned raw records present")
    for run in runs:
        run["issues"].extend("global provenance: "+value for value in issues)
    return {"metadata": metadata, "plan_path": str(plan_path.resolve()), "plan_sha256": audit.sha(plan_path),
            "expected_runs": len(expected), "observed_runs": len(runs),
            "complete_plan": expected == observed, "missing_jobs": sorted(expected-observed),
            "unexpected_jobs": sorted(observed-expected), "global_issues": issues,
            "required_source_count": len(required), "recorded_source_count": len(hashes),
            "changed_sources": changed, "source_inventory_matches": set(hashes) == required,
            "reporter_sha256": audit.sha(__file__), "integrity_helper_sha256": audit.sha(audit.__file__),
            "current_method_interpretation": "Optimized fresh T1; fixed-A cached transport T2/T4/T8. Only T varies between transport curves.",
            "historical_runner_wording_note": "The frozen generic runner metadata mentions its earlier three-arm ablation. It describes runner capabilities, not the arm selection in this new plan; plan.case switches and actual per-run kernels define these four period curves.",
            "seed_reuse_note": "The frozen plan reuses the predeclared seeds90–94 from the earlier action-cache ablation, not new independent confirmation seeds. Historical input-file equivalence is checked only when --previous-results is supplied; all reported timings come only from the supplied sweep."}


def previous_input_audit(runs, directory=None):
    """Optionally compare historical inputs; current-run pairing is separate."""
    if directory is None:
        return {"status": "not_requested", "requested": False,
                "all_seeds_checked": False, "directory": None, "evidence": []}
    directory = Path(directory)
    evidence = []
    for seed in sorted({run["seed"] for run in runs}):
        path = directory/f"dense_action_n1024_p20_l40_optimized_fresh_seed{seed}_T1_rep0.json"
        item = {"seed": seed, "previous_source": str(path.resolve()),
                "checked": False, "issues": []}
        if not path.is_file():
            item["issues"].append("previous same-seed input evidence missing")
        else:
            old = json.loads(path.read_text())["result"]
            item["checked"] = True
            item["previous_source_sha256"] = audit.sha(path)
            for run in [run for run in runs if run["seed"] == seed and not run["exception_failure"]]:
                fields = ("matrix_sha256", "initial_point_sha256", "matrix_seed", "initial_point_seed", "sketch_seed")
                different = [name for name in fields if run.get(name) != old.get(name)]
                if different:
                    item["issues"].append({"source": run["source"], "different_fields": different})
                    run["issues"].append("input hashes/streams differ from previous same-seed problem")
            item.update(matrix_sha256=old["matrix_sha256"], initial_point_sha256=old["initial_point_sha256"])
        evidence.append(item)
    complete = bool(evidence) and all(item["checked"] for item in evidence)
    status = "incomplete" if not complete else ("mismatch" if any(item["issues"] for item in evidence) else "checked")
    return {"status": status, "requested": True, "all_seeds_checked": complete,
            "directory": str(directory.resolve()), "evidence": evidence}


def target_rows(runs, plan):
    rows = audit.make_target_rows(runs, plan)
    return rows


def repeat_cells(runs, rows, plan):
    grouped_rows, grouped_runs = defaultdict(list), defaultdict(list)
    for row in rows:
        grouped_rows[(row["seed"], row["T"], row["target"])].append(row)
    for run in runs:
        grouped_runs[(run["seed"], run["T"])].append(run)
    cells, diagnostics = [], []
    for seed in plan["seeds"]:
        for T in PERIODS:
            records = grouped_runs.get((seed, T), [])
            good_histories = [run for run in records if run.get("history")]
            baseline = good_histories[0] if good_histories else None
            differences = []
            if baseline is not None:
                reference = {key: value for key, value in baseline["history"].items() if key != "time"}
                for run in good_histories[1:]:
                    other = {key: value for key, value in run["history"].items() if key != "time"}
                    changed = [key for key in set(reference)|set(other) if reference.get(key) != other.get(key)]
                    if changed:
                        differences.append({"replicate": run["replicate"], "fields": sorted(changed)})
            diagnostic = {"seed": seed, "T": T, "present_repeats": sorted(run["replicate"] for run in records),
                          "all_histories_present": len(good_histories) == plan["replicates"],
                          "histories_identical": len(good_histories) == plan["replicates"] and not differences,
                          "differences": differences, "policy": "Diagnostic only; never drop a slower valid repeat."}
            diagnostics.append(diagnostic)
            for target in (1e-2, 1e-4, 1e-6):
                members = grouped_rows.get((seed, T, target), [])
                reasons = []
                if sorted(row["replicate"] for row in members) != list(range(plan["replicates"])):
                    reasons.append("missing or duplicate timing repeats")
                if any(not row["usable_hit"] for row in members):
                    reasons.append("one or more invalid, failed or target-missed repeats")
                complete = not reasons
                cells.append({"seed": seed, "T": T, "target": target, "complete": complete,
                              "reasons": reasons, "declared_repeats": plan["replicates"], "observed_repeats": len(members),
                              "time_s": audit.stats(row["first_hit_time_s"] for row in members) if complete else audit.stats([]),
                              "first_hit_trials": [row["first_hit_trial"] for row in members],
                              "first_hit_gradients": [row["first_hit_grad_norm"] for row in members],
                              "repeat_history_identical": diagnostic["histories_identical"], "repeats": members})
    return cells, diagnostics


def summarize(cells, plan):
    lookup = {(cell["seed"], cell["T"], cell["target"]): cell for cell in cells}
    summaries = []
    for target in (1e-2, 1e-4, 1e-6):
        for T in PERIODS:
            pairs, excluded = [], []
            for seed in plan["seeds"]:
                baseline, comparison = lookup[(seed, 1, target)], lookup[(seed, T, target)]
                if not baseline["complete"] or not comparison["complete"]:
                    excluded.append({"seed": seed, "reason": "baseline/comparison has an incomplete three-repeat cell"})
                    continue
                left = Rational(baseline["time_s"]["exact"]["median"])
                right = Rational(comparison["time_s"]["exact"]["median"])
                ratio = left/right
                pairs.append({"seed": seed, "speedup": float(ratio), "exact_speedup": str(ratio),
                              "fresh_median_time_s": float(left), "transport_median_time_s": float(right),
                              "exact_fresh_median_time_s": str(left), "exact_transport_median_time_s": str(right),
                              "fresh_trials": baseline["first_hit_trials"], "transport_trials": comparison["first_hit_trials"]})
            summaries.append({"T": T, "target": target, "pair_count": len(pairs), "declared_seeds": len(plan["seeds"]),
                              "paired_speedup": audit.stats(pair["exact_speedup"] for pair in pairs),
                              "faster_pairs": sum(bool(Rational(pair["exact_speedup"]) > 1) for pair in pairs),
                              "pairs": pairs, "excluded_pairs": excluded})
    return summaries


def render(runs, plan, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14, "lines.linewidth": 1.5, "axes.linewidth": 0.8,
                         "axes.facecolor": "white", "figure.facecolor": "white", "axes.grid": False,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "xtick.direction": "out", "ytick.direction": "out"})
    selected = [run for run in runs if run["seed"] == plan["representative_seed"]
                and run["replicate"] == plan["representative_replicate"]]
    omissions, curves, files = {}, {}, []

    def draw(ax, timed):
        for T in PERIODS:
            matches = [run for run in selected if run["T"] == T]
            if len(matches) != 1:
                omissions[str(T)] = "Missing/duplicate declared trajectory; no replacement chosen."
                continue
            run = matches[0]
            if run["issues"] or run["exception_failure"]:
                omissions[str(T)] = "Invalid/failed run retained in tables but not plotted as valid."
                continue
            hit = audit.first_hit(run, plan["primary_target"])
            hist = run["history"]
            end = hit+1 if hit is not None else len(hist["f"])
            gaps = hist["objective_gap"][:end]
            invalid = sum(not math.isfinite(value) or value <= 0 for value in gaps)
            if invalid:
                omissions[str(T)] = f"{invalid} nonpositive/nonfinite log points omitted without clipping."
            y = [value if math.isfinite(value) and value > 0 else float("nan") for value in gaps]
            x = hist["time"][:end] if timed else list(range(end))
            label = f"T={T}"
            ax.plot(x, y, color=COLORS[T], linestyle=STYLES[T], label=label)
            curves[str(T)] = {"source": run["source"], "source_sha256": run["source_sha256"],
                              "points": end, "first_hit_trial": hit,
                              "first_hit_time_s": hist["time"][hit] if hit is not None else None,
                              "method": "optimized fresh" if T == 1 else "fixed-A action-cached transport",
                              "transformation": "None; individual raw points, no smoothing, averaging, interpolation or time rescaling."}
        ax.set(xlabel="time [s]" if timed else "iteration", ylabel=r"$f-f^\star$", yscale="log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.legend(loc="upper right", frameon=False)
        ax.grid(False)

    prefix = plan["display_case"]+f"_T2_T4_T8_seed{plan['representative_seed']}_rep{plan['representative_replicate']}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, timed in zip(axes, (False, True)):
        draw(ax, timed)
    fig.tight_layout()
    path = out/(prefix+".png")
    fig.savefig(path, dpi=200)
    files.append(str(path.resolve()))
    plt.close(fig)
    for timed, panel in ((False, "iteration"), (True, "time")):
        fig, ax = plt.subplots(figsize=(6, 4))
        draw(ax, timed)
        fig.tight_layout()
        for extension in ("eps", "png"):
            path = out/(prefix+"_"+panel+"."+extension)
            fig.savefig(path, dpi=200)
            files.append(str(path.resolve()))
        plt.close(fig)
    return {"files": files, "omissions": omissions, "curves": curves,
            "representative_seed": plan["representative_seed"], "replicate": plan["representative_replicate"],
            "style": {"colors": COLORS, "line_styles": STYLES, "legend_fontsize": 14,
                      "label_fontsize": 16, "tick_fontsize": 14, "combined_inches": [12, 4], "single_inches": [6, 4]}}


def hardware_snapshot():
    answer = {"system": platform.system(), "release": platform.release(),
              "version": platform.version(), "machine": platform.machine(),
              "logical_cpu_count": os.cpu_count(),
              "macos_probes": "not_applicable", "optional_probe_errors": []}
    if answer["system"] == "Darwin":
        probes = [(key, ["sysctl", "-n", key]) for key in
                  ("machdep.cpu.brand_string", "hw.model", "hw.physicalcpu", "hw.logicalcpu")]
        probes += [("macos_product_version", ["sw_vers", "-productVersion"]),
                   ("macos_build", ["sw_vers", "-buildVersion"])]
        for key, command in probes:
            try:
                answer[key] = subprocess.check_output(
                    command, text=True, stderr=subprocess.PIPE, timeout=5).strip()
            except (OSError, subprocess.SubprocessError) as error:
                answer[key] = None
                answer["optional_probe_errors"].append(
                    {"field": key, "command": command, "error": type(error).__name__, "detail": str(error)})
        answer["macos_probes"] = "partial" if answer["optional_probe_errors"] else "complete"
    answer["scope"] = "Read-only host observation at reporting; raw metadata separately records Python/platform/thread policy."
    return answer


def markdown(report):
    plan, provenance = report["plan"], report["provenance"]
    previous = report["previous_input_audit"]
    if previous["status"] == "not_requested":
        historical_note = ("Historical input hashes and RNG streams were not checked: no --previous-results "
                           "directory was supplied. The plan's reused-seed declaration does not establish "
                           "bytewise equivalence to unavailable historical input records.")
    else:
        checked = sum(item["checked"] for item in previous["evidence"])
        historical_note = (f"Optional historical input comparison: status={previous['status']}; "
                           f"{checked}/{len(previous['evidence'])} seed records checked. "
                           "Missing historical records and any input-hash/RNG-stream mismatches are retained "
                           "in summary.json under previous_input_audit.")
    lines = ["# Dense n=1024 cached-transport refresh-period sweep", "",
             "Four curves only: original optimized fresh Nyström T=1, and fixed-A action-cached transport T=2,4,8. "
             "Only the refresh period varies among the three transport curves. This is NOT the four-implementation ablation "
             "and is NOT a pure-direction-reuse experiment.", "",
             "Problem: materialized non-diagonal dense A, n=1024, p=20, ell=40, with the prescribed spectrum and paired "
             "problem seeds90–94. The frozen plan reuses the earlier ablation's seed values, not new independent "
             "confirmation seeds. Only timings from the supplied sixty-run sweep are used; earlier timings are never pooled here.", "",
             "## Measurement", "",
             "At each q=(f-f_reference)/(f_initial-f_reference) target, take the first recorded hit without interpolation. "
             "An arm/seed timing median requires ALL3 valid repeats to reach. Form fresh/transport ratios within each seed "
             "after repeat medians, then summarize over five seeds using exact decimal-rational arithmetic. Min–max is not "
             "a confidence interval; no statistical significance or globally optimal T is claimed. Repeat-history differences "
             "are diagnostics, never grounds for dropping slow successful repeats.", "",
             "Loop timing excludes initial solver QR/geometry, matrix construction and the independent numerical eigenvalue "
             "reference. This reference is not an exact interval certificate, and objective gap alone is not a stationarity certificate.", ""]
    for target in (1e-2, 1e-4, 1e-6):
        lines += [f"## Target q <= {target:g}", "", "| T | Pairs | Median fresh/transport | Min–max | Faster pairs |",
                  "| --- | --- | --- | --- | --- |"]
        for item in report["summaries"]:
            if item["target"] == target:
                s = item["paired_speedup"]
                lines.append(f"| {item['T']} | {item['pair_count']}/5 | {audit.number(s['median'])} | {audit.number(s['min'])}–{audit.number(s['max'])} | {item['faster_pairs']} |")
        lines.append("")
    lines += ["## Primary target per seed", "", "| Seed | T | Complete repeats | Median seconds | First-hit trials |",
              "| --- | --- | --- | --- | --- |"]
    for cell in report["repeat_cells"]:
        if cell["target"] == plan["primary_target"]:
            lines.append(f"| {cell['seed']} | {cell['T']} | {cell['complete']} | {audit.number(cell['time_s']['median'], 9)} | {cell['first_hit_trials']} |")
    lines += ["", "## Evidence and limitations", "",
              f"Raw records: {provenance['observed_runs']}/{provenance['expected_runs']}; complete plan: {provenance['complete_plan']}.",
              f"Global issues: {provenance['global_issues']}; missing jobs: {provenance['missing_jobs']}; unexpected jobs: {provenance['unexpected_jobs']}.",
              f"Exceptions: {report['exception_count']}; integrity failures: {report['integrity_count']}; incomplete target cells: {sum(not c['complete'] for c in report['repeat_cells'])}.",
              f"Nonidentical/missing repeat histories: {sum(not c['histories_identical'] for c in report['repeat_history_checks'])}.",
              f"Plan SHA256: `{provenance['plan_sha256']}`. Required/recorded execution sources: {provenance['required_source_count']}/{provenance['recorded_source_count']}; changed: {provenance['changed_sources']}.", "",
              "Input hashes and RNG streams are checked across all current arms/repeats. " + historical_note + " "
              "The old generic runner metadata retains its historical ablation wording; the frozen new plan, "
              "actual kernel switches and this report explicitly define the current four-period comparison.", "",
              "Action caching reuses fixed-A products through shared transport/gradient coefficients; every current Hessian core "
              "still uses the current S, frame and tangent projection. It is not reuse of an old Hessian, a BFGS update, "
              "a generic manifold convergence theorem, or finite-precision exactness. Persistent action storage and full-run "
              "dense-product/core counters are retained; they are not target-prefix profiles or an RSS bound. Supporting "
              "derivation/diagnostic artifacts are linked only when present. This reporter runs no numerical diagnostics; "
              "an absent diagnostics file is not evidence of a completed check.", "",
              "## Host and figure", "", f"Observed host: `{report['hardware']}`.",
              f"Recorded Python/platform: `{provenance['metadata'].get('python')}` / `{provenance['metadata'].get('platform')}`.",
              "Single-thread BLAS controls are requested by the unchanged runner; actual Accelerate thread count is not independently certified.", "",
              report["plot"]["caption"], "",
              "No raw records, earlier reports, manuscript files or installed manuscript figures are edited by this reporter.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--previous-results", type=Path,
                        help="Optional earlier action-cache ablation records for historical input-hash comparison")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new report directory; earlier artifacts are never overwritten")
    plan = json.loads(args.plan.read_text())
    runs = load_runs(args.results, plan)
    source_audit = provenance(runs, plan, args.plan)
    input_pairing = audit.check_input_pairing(runs)
    previous_inputs = previous_input_audit(runs, args.previous_results)
    rows = target_rows(runs, plan)
    cells, repeat_checks = repeat_cells(runs, rows, plan)
    summaries = summarize(cells, plan)
    args.out.mkdir(parents=True)
    plot = render(runs, plan, args.out)
    plot["caption"] = ("Dense n=1024, p=20, ell=40; preselected seed90, replicate0. Nyström T=1 is the original optimized "
                       "fresh implementation; Transport T=2,4,8 all use the same fixed-A action-cache specialization. "
                       "The ordinate f-f_ref uses the independently computed numerical eigenvalue reference, not a rigorously exact certificate. "
                       "Individual raw trajectories stop at the first recorded relative-gap q<=1e-6 hit when reached; no averaging, "
                       "smoothing, interpolation, rescaling or fastest-seed/repeat selection. Loop times exclude initialization, "
                       "matrix generation and reference eigensolve. The figure keeps the established red-family fonts/line styles. "
                       "The plan reuses predeclared seeds90–94; historical input-file comparison is optional, "
                       "and only timings from the supplied sweep are plotted.")
    validation_paths = [HERE/"dense_action_cache/DERIVATION.md", HERE/"dense_action_cache/diagnostics/unit_checks.json",
                        HERE/"dense_action_cache/diagnostics/deployment_checks.json"]
    validation = [{"path": str(path.resolve()), "sha256": audit.sha(path),
                   "scope": "Existing derivation or separately measured numerical validation, not a new timed-run residual measurement."}
                  for path in validation_paths if path.is_file()]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan": plan, "provenance": source_audit,
              "hardware": hardware_snapshot(), "input_pairing": input_pairing, "previous_input_audit": previous_inputs,
              "repeat_cells": cells, "repeat_history_checks": repeat_checks, "summaries": summaries,
              "exception_count": sum(run["exception_failure"] for run in runs),
              "integrity_count": sum(bool(run["issues"]) for run in runs), "plot": plot,
              "validation_links": validation,
              "raw_sources": [{"path": run["source"], "sha256": run["source_sha256"]} for run in runs]}
    audit.write_json(args.out/"summary.json", report)
    audit.write_json(args.out/"plot_manifest.json", plot | {"provenance": source_audit, "raw_sources": report["raw_sources"]})
    audit.write_json(args.out/"whole_run_integrity.json", audit.workload_records(runs))
    audit.write_json(args.out/"repeat_cells.json", cells)
    audit.write_json(args.out/"repeat_history_checks.json", repeat_checks)
    audit.write_csv(args.out/"runs.csv", rows)
    audit.write_csv(args.out/"paired_speedups.csv", [{"T": item["T"], "target": item["target"], **pair}
                                                   for item in summaries for pair in item["pairs"]])
    audit.write_csv(args.out/"summary.csv", [{"T": item["T"], "target": item["target"], "pairs": item["pair_count"],
                 **{key: item["paired_speedup"][key] for key in ("median", "min", "max")},
                 "exact_median": item["paired_speedup"]["exact"].get("median"), "excluded_pairs": item["excluded_pairs"]}
                 for item in summaries])
    (args.out/"REPORT.md").write_text(markdown(report))
    (args.out/"figure_caption.txt").write_text(plot["caption"]+"\n")
    if any(audit.sha(run["source"]) != run["source_sha256"] for run in runs):
        raise RuntimeError("Raw evidence changed during reporting")
    print(json.dumps({"output": str(args.out.resolve()), "runs": len(runs), "complete": source_audit["complete_plan"],
                      "exceptions": report["exception_count"], "integrity_failures": report["integrity_count"],
                      "primary": [item for item in summaries if item["target"] == plan["primary_target"]],
                      "files": plot["files"]}, indent=2))


if __name__ == "__main__":
    main()
