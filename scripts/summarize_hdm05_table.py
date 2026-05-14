#!/usr/bin/env python3
"""Summarize the HDM05 PGA outputs into the paper table columns."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, Optional


METHOD_ORDER = [
    ("Exact", None, "exact"),
    ("Nystrom (ell=20)", 20, "nystrom(ell=20)"),
    ("Nystrom (ell=40)", 40, "nystrom(ell=40)"),
    ("Nystrom (ell=80)", 80, "nystrom(ell=80)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--logreg-dir", type=Path, default=None)
    parser.add_argument("--svm-dir", type=Path, default=None)
    parser.add_argument("--mlp-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def accuracy(csv_path: Path, method: str, k: int) -> Optional[float]:
    if not csv_path.exists():
        return None
    for row in read_rows(csv_path):
        if row.get("method") == method and int(float(row.get("k", "nan"))) == k:
            value = row.get("acc_mean")
            return None if value in (None, "") else float(value)
    return None


def median_pairwise_t2(csv_path: Path, method: str, k: int) -> Optional[float]:
    if not csv_path.exists():
        return None
    vals = []
    for row in read_rows(csv_path):
        if row.get("method") == method and int(float(row.get("k", "nan"))) == k:
            vals.append(float(row["t2"]))
    if not vals:
        return None
    return statistics.median(vals) / 1000.0


def memory_mib(summary_path: Path, ell: Optional[int]) -> Optional[float]:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    storage = summary.get("storage") or {}
    if ell is None:
        covariance = storage.get("covariance") or {}
        value = covariance.get("rss_delta_bytes") or covariance.get("dense_bytes")
        return None if value is None else float(value) / 1024.0 / 1024.0
    run = (summary.get("nystrom") or {}).get(str(ell)) or {}
    run_storage = run.get("storage") or {}
    value = run_storage.get("rss_delta_bytes")
    return None if value is None else float(value) / 1024.0 / 1024.0


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    logreg_dir = args.logreg_dir or results_dir / "hdm05_table_logreg_pairwise_storage"
    svm_dir = args.svm_dir or results_dir / "hdm05_table_svm_tuned"
    mlp_dir = args.mlp_dir or results_dir / "hdm05_table_mlp"
    output = args.output or results_dir / "hdm05_table_summary.csv"

    logreg_csv = logreg_dir / "classification_multiclass_compare.csv"
    svm_csv = svm_dir / "classification_multiclass_compare.csv"
    mlp_csv = mlp_dir / "classification_multiclass_compare.csv"
    pairwise_csv = logreg_dir / "hotelling_t2_pairwise_compare.csv"
    storage_json = logreg_dir / "summary.json"

    rows = []
    for label, ell, method in METHOD_ORDER:
        rows.append(
            {
                "method": label,
                "acc_logreg": fmt(accuracy(logreg_csv, method, args.k)),
                "acc_svm": fmt(accuracy(svm_csv, method, args.k)),
                "acc_mlp": fmt(accuracy(mlp_csv, method, args.k)),
                "memory_mib": fmt(memory_mib(storage_json, ell), digits=2),
                "median_pairwise_t2_x1e3": fmt(median_pairwise_t2(pairwise_csv, method, args.k), digits=3),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    print()
    print("| Method | Acc logreg | Acc SVM | Acc MLP | Memory MiB | T2 x 1e3 |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['method']} | {row['acc_logreg']} | {row['acc_svm']} | "
            f"{row['acc_mlp']} | {row['memory_mib']} | {row['median_pairwise_t2_x1e3']} |"
        )


if __name__ == "__main__":
    main()
