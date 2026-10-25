"""Portable report metadata checks; no solver imports or benchmark runs."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_dense_action_periods as report


def run_fixture():
    return {"seed": 90, "exception_failure": False, "source": "current.json", "issues": [],
            "matrix_sha256": "matrix-hash", "initial_point_sha256": "point-hash",
            "matrix_seed": 1, "initial_point_seed": 2, "sketch_seed": 3}


def report_fixture(previous_inputs):
    return {"plan": {"primary_target": 1e-6}, "previous_input_audit": previous_inputs,
            "summaries": [], "repeat_cells": [], "repeat_history_checks": [],
            "exception_count": 0, "integrity_count": 0, "hardware": {}, "plot": {"caption": "fixture"},
            "provenance": {"observed_runs": 0, "expected_runs": 60, "complete_plan": False,
                           "global_issues": [], "missing_jobs": [], "unexpected_jobs": [],
                           "plan_sha256": "plan-hash", "required_source_count": 0,
                           "recorded_source_count": 0, "changed_sources": [], "metadata": {}}}


class ReportPortabilityTests(unittest.TestCase):
    def test_non_macos_hosts_do_not_run_macos_commands(self):
        for system in ("Linux", "Windows"):
            with self.subTest(system=system), patch.object(report.platform, "system", return_value=system), \
                    patch.object(report.subprocess, "check_output") as command:
                snapshot = report.hardware_snapshot()
            command.assert_not_called()
            self.assertEqual(snapshot["system"], system)
            self.assertEqual(snapshot["macos_probes"], "not_applicable")
            self.assertEqual(snapshot["optional_probe_errors"], [])
            for key in ("release", "version", "machine", "logical_cpu_count", "scope"):
                self.assertIn(key, snapshot)

    def test_macos_optional_probe_failures_are_recorded(self):
        outcomes = [FileNotFoundError("sysctl unavailable"),
                    subprocess.CalledProcessError(1, ["sysctl", "-n", "hw.model"]),
                    "4\n", "8\n", subprocess.TimeoutExpired(["sw_vers"], 5), "build\n"]
        with patch.object(report.platform, "system", return_value="Darwin"), \
                patch.object(report.subprocess, "check_output", side_effect=outcomes) as command:
            snapshot = report.hardware_snapshot()
        self.assertEqual(command.call_count, 6)
        self.assertEqual(snapshot["macos_probes"], "partial")
        self.assertEqual(snapshot["hw.logicalcpu"], "8")
        self.assertIsNone(snapshot["macos_product_version"])
        self.assertEqual([item["error"] for item in snapshot["optional_probe_errors"]],
                         ["FileNotFoundError", "CalledProcessError", "TimeoutExpired"])

    def test_historical_audit_is_not_requested_by_default(self):
        run = run_fixture()
        with patch.object(Path, "is_file", side_effect=AssertionError("must not access historical files")):
            result = report.previous_input_audit([run])
        self.assertEqual(result["status"], "not_requested")
        self.assertFalse(result["requested"])
        self.assertFalse(result["all_seeds_checked"])
        self.assertEqual(result["evidence"], [])
        self.assertEqual(run["issues"], [])

    def test_report_discloses_unrequested_history_and_absent_diagnostics(self):
        previous_inputs = report.previous_input_audit([run_fixture()])
        text = report.markdown(report_fixture(previous_inputs))
        self.assertIn("Historical input hashes and RNG streams were not checked", text)
        self.assertIn("Input hashes and RNG streams are checked across all current arms/repeats", text)
        self.assertIn("an absent diagnostics file is not evidence of a completed check", text)
        self.assertNotIn("checked against the previous", text)

    def test_requested_historical_audit_uses_explicit_directory(self):
        run = run_fixture()
        directory = Path("previous-input-fixtures")
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "read_text", return_value=json.dumps({"result": run})), \
                patch.object(report.audit, "sha", return_value="historical-file-hash") as sha:
            result = report.previous_input_audit([run], directory)
        expected = directory/"dense_action_n1024_p20_l40_optimized_fresh_seed90_T1_rep0.json"
        sha.assert_called_once_with(expected)
        self.assertEqual(result["status"], "checked")
        self.assertTrue(result["all_seeds_checked"])
        self.assertEqual(result["evidence"][0]["previous_source"], str(expected.resolve()))
        self.assertEqual(run["issues"], [])

    def test_missing_requested_historical_record_is_explicitly_unchecked(self):
        run = run_fixture()
        with patch.object(Path, "is_file", return_value=False):
            result = report.previous_input_audit([run], Path("missing-history"))
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["all_seeds_checked"])
        self.assertFalse(result["evidence"][0]["checked"])
        self.assertEqual(result["evidence"][0]["issues"], ["previous same-seed input evidence missing"])
        text = report.markdown(report_fixture(result))
        self.assertIn("status=incomplete; 0/1 seed records checked", text)

    def test_requested_historical_mismatch_still_invalidates_current_run(self):
        run = run_fixture()
        old = dict(run, matrix_sha256="different-matrix")
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "read_text", return_value=json.dumps({"result": old})), \
                patch.object(report.audit, "sha", return_value="historical-file-hash"):
            result = report.previous_input_audit([run], Path("previous-input-fixtures"))
        self.assertEqual(result["status"], "mismatch")
        self.assertTrue(result["all_seeds_checked"])
        self.assertEqual(run["issues"], ["input hashes/streams differ from previous same-seed problem"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
