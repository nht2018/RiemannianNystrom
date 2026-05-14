"""
Downstream tasks on the HDM05 SPD dataset using PGA scores (exact vs Nyström).

This script:
1) Loads all HDM05 matrices from a directory of .npy files named like `id_length_label.npy`.
2) Computes log-Euclidean tangent vectors v_i = vec_sym(log(X_i)) and centers them.
3) Computes a rank-ell Nyström approximation of the tangent covariance, without forming it.
4) Optionally computes the "exact" PGA directions via SVD of the centered tangent data.
5) Uses the resulting PGA score features to run:
   A) Two-sample Hotelling's T^2 test between two labels.
   B) Binary classification between the same two labels (logreg, SVM, MLP, or XGBoost).

The choice of labels can be specified explicitly or selected automatically from the most frequent
labels to ensure reasonable sample counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import psutil
import scipy.linalg as la
from scipy.stats import f
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from pga_utils import vec_sym


HDM05_NAME_RE = re.compile(r"^(?P<id>\d+)_(?P<length>\d+)_(?P<label>\d+)\.npy$")


@dataclass(frozen=True)
class HDM05SampleInfo:
    sample_id: int
    length: int
    label: int
    path: Path


@dataclass(frozen=True)
class ClassifierConfig:
    model: str
    svm_tune: bool
    svm_c_grid: List[float]
    svm_gamma_grid: List[float]
    svm_cv: int
    mlp_hidden_layers: Tuple[int, ...]
    mlp_alpha: float
    mlp_max_iter: int
    mlp_early_stopping: bool
    mlp_learning_rate_init: float
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float


def parse_int_list(raw: str | None) -> List[int]:
    if raw is None or raw.strip() == "":
        return []
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = [p.strip() for p in token.split("-", maxsplit=1)]
            if len(parts) != 2 or parts[0] == "" or parts[1] == "":
                raise ValueError(f"Invalid integer range token: {token!r}")
            start = int(parts[0])
            end = int(parts[1])
            if end < start:
                raise ValueError(f"Invalid decreasing range: {token!r}")
            values.extend(list(range(start, end + 1)))
        else:
            values.append(int(token))
    return values


def parse_float_list(raw: str | None) -> List[float]:
    if raw is None or raw.strip() == "":
        return []
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def parse_layer_sizes(raw: str | None) -> List[int]:
    if raw is None or raw.strip() == "":
        return []
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            raise ValueError(f"Invalid layer size token: {token!r} (use comma-separated integers)")
        values.append(int(token))
    return values


def parse_group_labels(raw: str | None) -> Optional[Tuple[int, int]]:
    if raw is None or raw.strip() == "":
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--group-labels must be like 'a,b'")
    return int(parts[0]), int(parts[1])


def adjust_pvalues(pvals: List[float], method: str) -> List[float]:
    values = np.asarray(pvals, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return adjusted.tolist()
    vals = values[mask]
    m = vals.size
    if method == "bonferroni":
        adjusted_vals = np.clip(vals * float(m), 0.0, 1.0)
        adjusted[mask] = adjusted_vals
        return adjusted.tolist()
    order = np.argsort(vals)
    adj_sorted = np.empty_like(vals)
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        adj_val = (m - rank + 1) * vals[idx]
        adj_val = max(adj_val, prev)
        adj_val = min(adj_val, 1.0)
        adj_sorted[idx] = adj_val
        prev = adj_val
    adjusted[mask] = adj_sorted
    return adjusted.tolist()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downstream tasks on HDM05 via PGA scores (Nyström vs exact).")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/hdm05_extracted/hdm05"),
        help="Directory containing HDM05 .npy matrices named id_length_label.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hdm05_full_ell100"),
        help="Output directory.",
    )
    parser.add_argument("--ell", type=int, default=100, help="Nyström sketch size ℓ.")
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20,50",
        help="Component counts k for downstream tasks (e.g. '5,10,20,50' or range '0-40').",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--log-eps", type=float, default=1e-6, help="Eigenvalue floor for SPD regularization/log.")
    parser.add_argument(
        "--group-labels",
        type=str,
        default=None,
        help="Two labels 'a,b' for group test/classification. If omitted, chooses automatically.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=200,
        help="Permutations for Hotelling T^2 p-value (0 to skip).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files (debug).",
    )
    parser.add_argument(
        "--ells",
        type=str,
        default=None,
        help="Comma-separated Nyström sketch sizes (overrides --ell if provided).",
    )
    parser.add_argument(
        "--clf-eval",
        type=str,
        choices=("holdout", "cv"),
        default="holdout",
        help="Classification evaluation mode: 80/20 holdout split or cross-validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout test fraction (used when --clf-eval=holdout).",
    )
    parser.add_argument(
        "--clf-task",
        type=str,
        choices=("binary", "multiclass"),
        default="binary",
        help="Classification task: binary (two labels) or multiclass (many labels).",
    )
    parser.add_argument(
        "--pairwise-t2",
        action="store_true",
        help="For multiclass, compute pairwise Hotelling T^2 across classes with correction.",
    )
    parser.add_argument(
        "--pairwise-t2-correction",
        type=str,
        choices=("holm", "bonferroni"),
        default="holm",
        help="Correction method for pairwise Hotelling T^2 p-values.",
    )
    parser.add_argument(
        "--svm-tune",
        action="store_true",
        help="If set with --clf-model=svm, run a small grid search over C and gamma.",
    )
    parser.add_argument(
        "--svm-c-grid",
        type=str,
        default="0.1,1,10,100",
        help="Comma-separated SVM C values for tuning (used with --svm-tune).",
    )
    parser.add_argument(
        "--svm-gamma-grid",
        type=str,
        default="0.001,0.01,0.1,1",
        help="Comma-separated SVM gamma values for tuning (used with --svm-tune).",
    )
    parser.add_argument(
        "--clf-model",
        type=str,
        choices=("logreg", "svm", "mlp", "xgb"),
        default="logreg",
        help="Classifier for downstream metrics: logreg (default), svm (RBF SVC), mlp, or xgb.",
    )
    parser.add_argument(
        "--svm-cv",
        type=int,
        default=3,
        help="Inner CV splits for SVM tuning (used with --svm-tune).",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        type=str,
        default="100,50",
        help="Comma-separated hidden layer sizes for MLP (e.g. '100,50').",
    )
    parser.add_argument(
        "--mlp-alpha",
        type=float,
        default=1e-4,
        help="L2 regularization for MLP.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=2000,
        help="Max iterations for MLP training.",
    )
    parser.add_argument(
        "--mlp-learning-rate-init",
        type=float,
        default=1e-3,
        help="Initial learning rate for MLP.",
    )
    parser.add_argument(
        "--mlp-early-stopping",
        action="store_true",
        help="Enable early stopping for MLP.",
    )
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=500,
        help="Number of trees for XGBoost.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=5,
        help="Max tree depth for XGBoost.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for XGBoost.",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.9,
        help="Subsample ratio for XGBoost.",
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.9,
        help="Column sample ratio for XGBoost.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=4,
        help="Minimum per-class count to include (used when --clf-task=multiclass).",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="If set, keep only the top-K most frequent classes (used when --clf-task=multiclass).",
    )
    parser.add_argument(
        "--compare-exact",
        action="store_true",
        help="If set, also compute exact PGA directions via SVD and produce comparison plots.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only (re)plot figures from existing CSVs in --output-dir (skips any computation).",
    )
    parser.add_argument(
        "--plot-ks",
        type=str,
        default=None,
        help="When using --plot-only, optionally restrict plots to these k values (e.g. '0,10,20,30,40').",
    )
    parser.add_argument(
        "--report-storage",
        action="store_true",
        help="Print and record real runtime storage/RSS numbers for covariance vs Nyström.",
    )
    parser.add_argument(
        "--materialize-covariance",
        action="store_true",
        help="If set with --report-storage, allocates a dense d×d covariance array to measure RSS impact.",
    )
    return parser.parse_args(argv)


def list_hdm05_samples(raw_dir: Path, max_files: Optional[int]) -> List[HDM05SampleInfo]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir does not exist: {raw_dir}")
    paths = sorted(raw_dir.glob("*.npy"))
    if max_files is not None:
        paths = paths[:max_files]
    samples: List[HDM05SampleInfo] = []
    for p in paths:
        m = HDM05_NAME_RE.match(p.name)
        if m is None:
            continue
        samples.append(
            HDM05SampleInfo(
                sample_id=int(m.group("id")),
                length=int(m.group("length")),
                label=int(m.group("label")),
                path=p,
            )
        )
    if not samples:
        raise ValueError(f"No valid HDM05 files found under {raw_dir}")
    return samples


def regularized_log_vec_sym(mat: np.ndarray, eps: float) -> Tuple[np.ndarray, float, float]:
    """
    Compute vec_sym(log(X_reg)) where X_reg is a symmetrized/clipped SPD version of mat.

    Returns (v_log, shift, min_eig_before_shift).
    """
    sym = 0.5 * (mat + mat.T)
    w, u = la.eigh(sym)
    min_eig = float(w.min())
    shift = max(eps - min_eig, 0.0)
    w_shifted = w + shift
    w_clipped = np.clip(w_shifted, eps, None)
    log_w = np.log(w_clipped)
    logm = (u * log_w) @ u.T
    logm = 0.5 * (logm + logm.T)
    return vec_sym(logm), float(shift), min_eig


def choose_label_pair(labels: np.ndarray) -> Tuple[int, int]:
    vals, cnts = np.unique(labels, return_counts=True)
    order = np.argsort(cnts)[::-1]
    vals = vals[order]
    cnts = cnts[order]
    top = min(10, vals.size)
    best_pair: Optional[Tuple[int, int]] = None
    best_score = (-1, -1)  # (min_count, total_count)
    for i in range(top):
        for j in range(i + 1, top):
            a, b = int(vals[i]), int(vals[j])
            ca, cb = int(cnts[i]), int(cnts[j])
            score = (min(ca, cb), ca + cb)
            if score > best_score:
                best_score = score
                best_pair = (a, b)
    if best_pair is None:
        # Fallback: just pick the two most frequent labels.
        return int(vals[0]), int(vals[1]) if vals.size > 1 else (int(vals[0]), int(vals[0]))
    return best_pair


def select_multiclass_subset(
    labels: np.ndarray, min_class_count: int, max_classes: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    vals, cnts = np.unique(labels, return_counts=True)
    keep_mask = cnts >= min_class_count
    vals_keep = vals[keep_mask]
    cnts_keep = cnts[keep_mask]
    if vals_keep.size == 0:
        raise ValueError(f"No classes have at least min_class_count={min_class_count}.")

    order = np.argsort(cnts_keep)[::-1]
    vals_keep = vals_keep[order]
    cnts_keep = cnts_keep[order]

    if max_classes is not None:
        if max_classes <= 1:
            raise ValueError("--max-classes must be >= 2 when set.")
        vals_keep = vals_keep[:max_classes]
        cnts_keep = cnts_keep[:max_classes]

    idx = np.where(np.isin(labels, vals_keep))[0]
    y_orig = labels[idx].astype(int)

    label_to_class = {int(label): int(i) for i, label in enumerate(vals_keep.tolist())}
    y = np.array([label_to_class[int(label)] for label in y_orig], dtype=int)

    label_counts = {int(label): int(count) for label, count in zip(vals_keep.tolist(), cnts_keep.tolist())}
    return idx, y, label_to_class, label_counts


def stratified_holdout_split(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified holdout split that guarantees at least one sample per class in both train and test.

    This is stricter than StratifiedShuffleSplit, which may allocate 0 test samples for rare
    classes when test_size * count < 1.
    """
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("--test-size must be in (0,1).")
    y = np.asarray(y).astype(int)
    n = int(y.size)
    if n == 0:
        raise ValueError("Cannot split an empty label array.")

    classes, counts = np.unique(y, return_counts=True)
    if np.any(counts < 2):
        bad = classes[counts < 2]
        raise ValueError(f"Need at least 2 samples per class for holdout split; bad classes: {bad.tolist()}")

    rng = np.random.default_rng(seed)
    n_test_target = int(round(float(test_size) * float(n)))
    n_test_target = max(int(classes.size), min(n - int(classes.size), n_test_target))

    n_test = np.floor(counts * float(test_size)).astype(int)
    n_test = np.clip(n_test, 1, counts - 1)

    diff = int(n_test_target - int(n_test.sum()))
    if diff > 0:
        cap = (counts - 1) - n_test
        order = np.argsort(cap)[::-1]
        i = 0
        while diff > 0 and cap.sum() > 0:
            c = int(order[i % order.size])
            if cap[c] > 0:
                n_test[c] += 1
                cap[c] -= 1
                diff -= 1
            i += 1
    elif diff < 0:
        need = -diff
        cap = n_test - 1
        order = np.argsort(cap)[::-1]
        i = 0
        while need > 0 and cap.sum() > 0:
            c = int(order[i % order.size])
            if cap[c] > 0:
                n_test[c] -= 1
                cap[c] -= 1
                need -= 1
            i += 1

    test_idx_parts: List[np.ndarray] = []
    train_idx_parts: List[np.ndarray] = []
    for cls, n_test_cls in zip(classes, n_test):
        idx_cls = np.where(y == cls)[0]
        rng.shuffle(idx_cls)
        test_idx_parts.append(idx_cls[: int(n_test_cls)])
        train_idx_parts.append(idx_cls[int(n_test_cls) :])

    test_idx = np.concatenate(test_idx_parts, axis=0)
    train_idx = np.concatenate(train_idx_parts, axis=0)
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return train_idx, test_idx


def nystrom_pga_from_centered_data(
    v_centered: np.ndarray, ell: int, k_max: int, rng: np.random.Generator, report_storage: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, int]]]:
    """
    Nyström approximation for the covariance C = (1/N) V^T V, without forming C.

    Inputs:
      v_centered: V in R^{N x d}, rows are centered tangent vectors.
      ell: sketch size.
      k_max: number of leading components to return (k_max <= ell recommended).

    Returns:
      eigvals_hat: shape (m,)
      eigvecs_hat: shape (d, m)
      storage_bytes: if report_storage, dict of intermediate/output array sizes in bytes.
    """
    n, d = v_centered.shape
    if ell > d:
        raise ValueError(f"ell={ell} exceeds dimension d={d}.")
    if k_max <= 0:
        raise ValueError("k_max must be positive.")

    storage: Optional[Dict[str, int]] = {} if report_storage else None

    p = rng.standard_normal((d, ell))
    y = v_centered @ p  # (n, ell)
    b = (v_centered.T @ y) / float(n)  # (d, ell)  equals C @ P
    w = p.T @ b  # (ell, ell) equals P^T C P
    w = 0.5 * (w + w.T)

    if storage is not None:
        storage.update(
            {
                "P_bytes": int(p.nbytes),
                "Y_bytes": int(y.nbytes),
                "B_bytes": int(b.nbytes),
                "W_bytes": int(w.nbytes),
            }
        )

    w_eigs, w_vecs = la.eigh(w)
    w_eigs = np.clip(w_eigs, 0.0, None)
    tol = float(w_eigs.max()) * 1e-12 + 1e-15
    inv_sqrt = np.zeros_like(w_eigs)
    mask = w_eigs > tol
    inv_sqrt[mask] = 1.0 / np.sqrt(w_eigs[mask])
    w_inv_sqrt = (w_vecs * inv_sqrt) @ w_vecs.T  # (ell, ell)

    z = b @ w_inv_sqrt  # (d, ell) so that C_hat = Z Z^T (rank <= ell)
    g = z.T @ z
    g = 0.5 * (g + g.T)

    if storage is not None:
        storage.update(
            {
                "W_vecs_bytes": int(w_vecs.nbytes),
                "W_inv_sqrt_bytes": int(w_inv_sqrt.nbytes),
                "Z_bytes": int(z.nbytes),
                "G_bytes": int(g.nbytes),
            }
        )

    eigs, vecs = la.eigh(g)
    order = eigs.argsort()[::-1]
    eigs = eigs[order]
    vecs = vecs[:, order]

    eigs = np.clip(eigs, 0.0, None)
    tol2 = float(eigs.max()) * 1e-12 + 1e-15
    keep = np.where(eigs > tol2)[0]
    if keep.size == 0:
        raise RuntimeError("Nyström produced no positive eigenvalues; try a different seed or ell.")
    m = int(min(k_max, keep.size))
    eigvals_hat = eigs[:m]
    vecs_m = vecs[:, :m]
    eigvecs_hat = z @ (vecs_m / np.sqrt(eigvals_hat))
    # Orthonormalize defensively.
    eigvecs_hat, _ = np.linalg.qr(eigvecs_hat)
    if storage is not None:
        storage.update(
            {
                "eigvals_hat_bytes": int(eigvals_hat.nbytes),
                "eigvecs_hat_bytes": int(eigvecs_hat.nbytes),
            }
        )
    return eigvals_hat, eigvecs_hat, storage


def format_bytes(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0 or unit == "TiB":
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} TiB"


def get_rss_bytes() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def exact_pga_from_centered_data(v_centered: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact PGA via SVD of the centered tangent data matrix.

    With V in R^{N x d} (rows are centered samples), define C = (1/N) V^T V.
    If (1/sqrt(N)) V = U S V^T, then C has eigenvalues S^2 and eigenvectors given by V.
    """
    n, d = v_centered.shape
    if k_max <= 0:
        raise ValueError("k_max must be positive.")
    if k_max > min(n, d):
        raise ValueError(f"k_max={k_max} exceeds min(n,d)={min(n, d)}.")
    scaled = v_centered / math.sqrt(float(n))
    _, s, vt = la.svd(scaled, full_matrices=False)
    eigvals = s * s
    eigvecs = vt.T
    return eigvals[:k_max], eigvecs[:, :k_max]


def hotelling_t2(a0: np.ndarray, a1: np.ndarray, ridge: float = 1e-8) -> Dict[str, float]:
    n0, p = a0.shape
    n1, _ = a1.shape
    if n0 < 2 or n1 < 2:
        raise ValueError("Need at least 2 samples per group for Hotelling's T^2.")
    if p == 0:
        return {
            "t2": 0.0,
            "f_stat": float("nan"),
            "df1": 0.0,
            "df2": float(n0 + n1 - 1),
            "p_analytic": float("nan"),
            "ridge_gamma": 0.0,
        }
    mean0 = a0.mean(axis=0)
    mean1 = a1.mean(axis=0)
    x0 = a0 - mean0
    x1 = a1 - mean1
    s0 = (x0.T @ x0) / float(n0 - 1)
    s1 = (x1.T @ x1) / float(n1 - 1)
    sp = ((n0 - 1) * s0 + (n1 - 1) * s1) / float(n0 + n1 - 2)
    gamma = ridge * float(np.trace(sp)) / float(max(p, 1))
    sp = sp + gamma * np.eye(p)
    diff = mean0 - mean1
    sol = la.solve(sp, diff, assume_a="pos")
    t2 = float((n0 * n1) / float(n0 + n1) * (diff @ sol))

    df1 = int(p)
    df2 = int(n0 + n1 - p - 1)
    f_stat = float("nan")
    p_analytic = float("nan")
    if df2 > 0:
        f_stat = float((df2 * t2) / (df1 * float(n0 + n1 - 2)))
        p_analytic = float(f.sf(f_stat, df1, df2))

    return {
        "t2": t2,
        "f_stat": f_stat,
        "df1": float(df1),
        "df2": float(df2),
        "p_analytic": p_analytic,
        "ridge_gamma": float(gamma),
    }


def pairwise_hotelling_t2(
    scores: np.ndarray,
    labels: np.ndarray,
    label_map: Optional[Dict[int, int]],
    *,
    ridge: float = 1e-8,
    correction: str = "holm",
) -> List[Dict[str, float]]:
    classes = np.unique(labels)
    if classes.size < 2:
        return []
    inv_map = {v: k for k, v in label_map.items()} if label_map is not None else {}
    rows: List[Dict[str, float]] = []
    pvals: List[float] = []
    for i, a in enumerate(classes):
        idx_a = labels == a
        for b in classes[i + 1 :]:
            idx_b = labels == b
            a0 = scores[idx_a]
            a1 = scores[idx_b]
            try:
                stats = hotelling_t2(a0, a1, ridge=ridge)
            except ValueError:
                stats = {
                    "t2": float("nan"),
                    "f_stat": float("nan"),
                    "df1": float(scores.shape[1]),
                    "df2": float("nan"),
                    "p_analytic": float("nan"),
                    "ridge_gamma": float("nan"),
                }
            pvals.append(stats["p_analytic"])
            rows.append(
                {
                    "class_a": float(a),
                    "class_b": float(b),
                    "label_a": float(inv_map.get(int(a), int(a))),
                    "label_b": float(inv_map.get(int(b), int(b))),
                    "n_a": float(a0.shape[0]),
                    "n_b": float(a1.shape[0]),
                    **stats,
                }
            )
    p_adjusted = adjust_pvalues(pvals, correction)
    for row, p_adj in zip(rows, p_adjusted):
        row["p_adjusted"] = float(p_adj) if p_adj == p_adj else float("nan")
        row["p_adjustment"] = str(correction)
    return rows


def hotelling_permutation_pvalue(
    a: np.ndarray, y: np.ndarray, n_perm: int, seed: int, ridge: float = 1e-8
) -> float:
    if n_perm <= 0:
        return float("nan")
    if a.shape[1] == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0 = int(idx0.size)
    n1 = int(idx1.size)
    if n0 < 2 or n1 < 2:
        return float("nan")
    obs = hotelling_t2(a[idx0], a[idx1], ridge=ridge)["t2"]
    n = int(a.shape[0])
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        p0 = perm[:n0]
        p1 = perm[n0 : n0 + n1]
        t2 = hotelling_t2(a[p0], a[p1], ridge=ridge)["t2"]
        if t2 >= obs:
            ge += 1
    return float((ge + 1) / float(n_perm + 1))


def _majority_class(y: np.ndarray) -> int:
    y = np.asarray(y).astype(int)
    if y.size == 0:
        return 0
    counts = np.bincount(y)
    return int(counts.argmax())


def _load_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "xgboost is required for --clf-model xgb. Install with: pip install xgboost"
        ) from exc
    return xgb


def _make_xgb_classifier(cfg: ClassifierConfig, task: str, labels: np.ndarray, seed: int) -> object:
    xgb = _load_xgboost()
    params = {
        "n_estimators": cfg.xgb_n_estimators,
        "max_depth": cfg.xgb_max_depth,
        "learning_rate": cfg.xgb_learning_rate,
        "subsample": cfg.xgb_subsample,
        "colsample_bytree": cfg.xgb_colsample_bytree,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }
    if task == "multiclass":
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = int(np.unique(labels).size)
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    return xgb.XGBClassifier(**params)


def _make_classifier(cfg: ClassifierConfig, task: str, seed: int, labels: Optional[np.ndarray]) -> object:
    if cfg.model == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        )
    if cfg.model == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=cfg.mlp_hidden_layers,
                alpha=cfg.mlp_alpha,
                max_iter=cfg.mlp_max_iter,
                early_stopping=cfg.mlp_early_stopping,
                learning_rate_init=cfg.mlp_learning_rate_init,
                random_state=seed,
            ),
        )
    if cfg.model == "xgb":
        if labels is None:
            raise ValueError("Labels are required to configure XGBoost classifier.")
        return _make_xgb_classifier(cfg, task, labels, seed)
    if task == "multiclass":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, multi_class="multinomial", solver="lbfgs"),
        )
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def _tune_svm_classifier(
    labels: np.ndarray, seed: int, svm_c_grid: List[float], svm_gamma_grid: List[float], svm_cv: int
) -> Optional[GridSearchCV]:
    if not svm_c_grid or not svm_gamma_grid:
        return None
    y = labels.astype(int)
    min_class = int(np.bincount(y).min()) if y.size > 0 else 0
    if min_class < 2:
        return None
    n_splits = min(max(2, svm_cv), min_class)
    base = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
    )
    param_grid = {
        "svc__C": svm_c_grid,
        "svc__gamma": svm_gamma_grid,
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GridSearchCV(base, param_grid=param_grid, scoring="accuracy", cv=cv)


def _build_classifier(task: str, labels: np.ndarray, seed: int, cfg: ClassifierConfig) -> object:
    if cfg.model == "svm" and cfg.svm_tune:
        tuned = _tune_svm_classifier(labels, seed, cfg.svm_c_grid, cfg.svm_gamma_grid, cfg.svm_cv)
        if tuned is not None:
            return tuned
    return _make_classifier(cfg, task, seed, labels)


def _binary_scores(clf: object, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(feats)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return prob, pred
    if hasattr(clf, "decision_function"):
        score = clf.decision_function(feats)
        pred = (score >= 0.0).astype(int)
        return score, pred
    pred = clf.predict(feats)
    return pred.astype(float), pred.astype(int)


def binary_classification_cv(
    scores: np.ndarray, labels01: np.ndarray, seed: int, clf_cfg: ClassifierConfig
) -> Dict[str, float]:
    y = labels01.astype(int)
    min_class = int(np.bincount(y).min()) if y.size > 0 else 0
    if min_class < 2:
        return {}
    n_splits = min(5, min_class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: List[float] = []
    accs: List[float] = []
    for train_idx, test_idx in skf.split(scores, y):
        if scores.shape[1] == 0:
            p = float(y[train_idx].mean())
            prob = np.full(test_idx.size, p, dtype=float)
            pred = np.full(test_idx.size, _majority_class(y[train_idx]), dtype=int)
            aucs.append(roc_auc_score(y[test_idx], prob))
            accs.append(accuracy_score(y[test_idx], pred))
        else:
            clf = _build_classifier("binary", y[train_idx], seed=seed, cfg=clf_cfg)
            clf.fit(scores[train_idx], y[train_idx])
            score, pred = _binary_scores(clf, scores[test_idx])
            aucs.append(roc_auc_score(y[test_idx], score))
            accs.append(accuracy_score(y[test_idx], pred))
    return {
        "split": "cv",
        "num_classes": 2.0,
        "n_splits": float(n_splits),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
    }


def binary_classification_holdout(
    scores: np.ndarray,
    labels01: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    clf_cfg: ClassifierConfig,
) -> Dict[str, float]:
    y = labels01.astype(int)
    if train_idx.size == 0 or test_idx.size == 0:
        return {}
    if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
        return {}

    if scores.shape[1] == 0:
        p = float(y[train_idx].mean())
        prob_test = np.full(test_idx.size, p, dtype=float)
        prob_train = np.full(train_idx.size, p, dtype=float)
        maj = _majority_class(y[train_idx])
        pred_test = np.full(test_idx.size, maj, dtype=int)
        pred_train = np.full(train_idx.size, maj, dtype=int)
        auc_test = roc_auc_score(y[test_idx], prob_test)
        acc_test = accuracy_score(y[test_idx], pred_test)
        auc_train = roc_auc_score(y[train_idx], prob_train)
        acc_train = accuracy_score(y[train_idx], pred_train)
    else:
        clf = _build_classifier("binary", y[train_idx], seed=seed, cfg=clf_cfg)
        clf.fit(scores[train_idx], y[train_idx])

        score_test, pred_test = _binary_scores(clf, scores[test_idx])
        auc_test = roc_auc_score(y[test_idx], score_test)
        acc_test = accuracy_score(y[test_idx], pred_test)

        score_train, pred_train = _binary_scores(clf, scores[train_idx])
        auc_train = roc_auc_score(y[train_idx], score_train)
        acc_train = accuracy_score(y[train_idx], pred_train)

    return {
        "split": "holdout",
        "num_classes": 2.0,
        "n_train": float(train_idx.size),
        "n_test": float(test_idx.size),
        "acc_train": float(acc_train),
        "auc_train": float(auc_train),
        "acc_mean": float(acc_test),
        "acc_std": 0.0,
        "auc_mean": float(auc_test),
        "auc_std": 0.0,
    }


def multiclass_classification_cv(
    scores: np.ndarray, labels_mc: np.ndarray, seed: int, clf_cfg: ClassifierConfig
) -> Dict[str, float]:
    y = labels_mc.astype(int)
    num_classes = int(np.unique(y).size)
    if num_classes < 2:
        return {}
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        return {}
    n_splits = min(5, min_class)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs: List[float] = []
    f1s: List[float] = []
    for train_idx, test_idx in skf.split(scores, y):
        if scores.shape[1] == 0:
            maj = _majority_class(y[train_idx])
            pred = np.full(test_idx.size, maj, dtype=int)
            accs.append(accuracy_score(y[test_idx], pred))
            f1s.append(f1_score(y[test_idx], pred, average="macro"))
        else:
            clf = _build_classifier("multiclass", y[train_idx], seed=seed, cfg=clf_cfg)
            clf.fit(scores[train_idx], y[train_idx])
            pred = clf.predict(scores[test_idx])
            accs.append(accuracy_score(y[test_idx], pred))
            f1s.append(f1_score(y[test_idx], pred, average="macro"))
    return {
        "split": "cv",
        "num_classes": float(num_classes),
        "n_splits": float(n_splits),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
    }


def multiclass_classification_holdout(
    scores: np.ndarray,
    labels_mc: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    clf_cfg: ClassifierConfig,
) -> Dict[str, float]:
    y = labels_mc.astype(int)
    num_classes = int(np.unique(y).size)
    if train_idx.size == 0 or test_idx.size == 0:
        return {}
    if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
        return {}

    if scores.shape[1] == 0:
        maj = _majority_class(y[train_idx])
        pred_test = np.full(test_idx.size, maj, dtype=int)
        pred_train = np.full(train_idx.size, maj, dtype=int)
    else:
        clf = _build_classifier("multiclass", y[train_idx], seed=seed, cfg=clf_cfg)
        clf.fit(scores[train_idx], y[train_idx])

        pred_test = clf.predict(scores[test_idx])
        pred_train = clf.predict(scores[train_idx])

    acc_test = accuracy_score(y[test_idx], pred_test)
    acc_train = accuracy_score(y[train_idx], pred_train)
    f1_test = f1_score(y[test_idx], pred_test, average="macro")
    f1_train = f1_score(y[train_idx], pred_train, average="macro")

    return {
        "split": "holdout",
        "num_classes": float(num_classes),
        "n_train": float(train_idx.size),
        "n_test": float(test_idx.size),
        "acc_train": float(acc_train),
        "f1_macro_train": float(f1_train),
        "acc_mean": float(acc_test),
        "acc_std": 0.0,
        "f1_macro_mean": float(f1_test),
        "f1_macro_std": 0.0,
    }


def load_csv_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for raw in reader:
            row: Dict[str, object] = {}
            for key, val in raw.items():
                if key == "method":
                    row[key] = "" if val is None else str(val)
                    continue
                if val is None or val == "":
                    row[key] = float("nan")
                    continue
                try:
                    row[key] = float(val)
                except ValueError:
                    row[key] = str(val)
            rows.append(row)
    return rows


def plot_from_csvs(output_dir: Path, ks_filter: Optional[Sequence[int]] = None) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    ks_keep: Optional[set[int]] = None
    if ks_filter is not None:
        ks_keep = {int(k) for k in ks_filter}

    clf_candidates = [
        "classification_multiclass_compare.csv",
        "classification_compare.csv",
        "classification_multiclass.csv",
        "classification.csv",
    ]
    clf_csv = next((output_dir / name for name in clf_candidates if (output_dir / name).exists()), None)
    if clf_csv is not None:
        clf_rows = load_csv_rows(clf_csv)
        if ks_keep is not None:
            filtered: List[Dict[str, object]] = []
            for row in clf_rows:
                k_raw = row.get("k", float("nan"))
                try:
                    k_val = float(k_raw)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(k_val) and int(k_val) in ks_keep:
                    filtered.append(row)
            clf_rows = filtered
        clf_png = clf_csv.with_suffix(".png").name
        out = plot_classification_compare(clf_rows, output_dir, filename=clf_png)
        if out is not None:
            outputs["classification_png"] = str(out)

    group_candidates = [
        "hotelling_t2_compare.csv",
        "hotelling_t2.csv",
    ]
    group_csv = next((output_dir / name for name in group_candidates if (output_dir / name).exists()), None)
    if group_csv is not None:
        group_rows = load_csv_rows(group_csv)
        if ks_keep is not None:
            filtered = []
            for row in group_rows:
                k_raw = row.get("k", float("nan"))
                try:
                    k_val = float(k_raw)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(k_val) and int(k_val) in ks_keep:
                    filtered.append(row)
            group_rows = filtered
        group_png = group_csv.with_suffix(".png").name
        out = plot_hotelling_compare(group_rows, output_dir, filename=group_png)
        if out is not None:
            outputs["hotelling_png"] = str(out)

    if not outputs:
        raise FileNotFoundError(f"No CSVs found to plot under {output_dir}")
    return outputs


def plot_classification_compare(rows: List[Dict[str, float]], output_dir: Path, filename: str) -> Optional[Path]:
    if not rows:
        return None
    method_set = {row["method"] for row in rows}
    all_ks = sorted({int(row["k"]) for row in rows})
    ells = sorted({int(row["ell"]) for row in rows if not np.isnan(row.get("ell", float("nan")))})

    start_col = np.array([1.0, 0.7, 0.7])  # light red
    end_col = np.array([0.6, 0.0, 0.0])  # deep red
    n_ells = len(ells)
    ell_colors: Dict[int, np.ndarray] = {}
    for idx, ell in enumerate(ells):
        t = 1.0 if n_ells <= 1 else float(idx) / float(n_ells - 1)
        ell_colors[ell] = (1 - t) * start_col + t * end_col

    method_entries: List[Tuple[Tuple[int, int], str, Optional[int]]] = []
    for method in method_set:
        ell_val = None
        for row in rows:
            if row["method"] != method:
                continue
            ell_raw = float(row.get("ell", float("nan")))
            if not np.isnan(ell_raw):
                ell_val = int(ell_raw)
                break
        sort_key = (0, 0) if ell_val is None else (1, ell_val)
        method_entries.append((sort_key, method, ell_val))
    method_entries.sort(key=lambda t: t[0])

    has_auc = any("auc_mean" in row and not np.isnan(row["auc_mean"]) for row in rows)
    has_f1 = any("f1_macro_mean" in row and not np.isnan(row["f1_macro_mean"]) for row in rows)
    if has_auc:
        metric1 = ("AUC", "auc_mean", "auc_std")
        metric2 = ("Accuracy", "acc_mean", "acc_std")
    elif has_f1:
        metric1 = ("Accuracy", "acc_mean", "acc_std")
        metric2 = ("Macro F1", "f1_macro_mean", "f1_macro_std")
    else:
        metric1 = ("Accuracy", "acc_mean", "acc_std")
        metric2 = None

    if metric2 is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 4.0), sharex=True)
        axes = (ax1,)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10.0, 4.0), sharex=True)

    for _, method, ell_val in method_entries:
        method_rows = [r for r in rows if r["method"] == method]
        by_k = {int(r["k"]): r for r in method_rows}
        if not by_k:
            continue
        max_k_avail = max(by_k.keys())
        ks_plot = list(all_ks)

        if ell_val is None:
            color = "tab:blue"
            linestyle = "-"
            linewidth = 2.0
            label = "Exact PGA"
        else:
            color = ell_colors.get(ell_val, end_col)
            linestyle = "--"
            linewidth = 2.0
            label = rf"Nyström ($\ell={ell_val}$)"

        name1, key1, std1 = metric1
        rows1: List[Optional[Dict[str, float]]] = []
        for k in ks_plot:
            row_k = by_k.get(int(k))
            if row_k is None and ell_val is not None and int(k) > int(max_k_avail):
                row_k = by_k.get(int(max_k_avail))
            rows1.append(row_k)
        y1 = np.array(
            [r.get(key1, float("nan")) if r is not None else float("nan") for r in rows1],
            dtype=float,
        )
        y1_std = np.array([r.get(std1, 0.0) if r is not None else 0.0 for r in rows1], dtype=float)
        keep1 = np.isfinite(y1)
        axes[0].errorbar(
            np.asarray(ks_plot, dtype=int)[keep1],
            y1[keep1],
            yerr=y1_std[keep1],
            capsize=3.0,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
        )
        if metric2 is not None:
            name2, key2, std2 = metric2
            y2 = np.array(
                [r.get(key2, float("nan")) if r is not None else float("nan") for r in rows1],
                dtype=float,
            )
            y2_std = np.array([r.get(std2, 0.0) if r is not None else 0.0 for r in rows1], dtype=float)
            keep2 = np.isfinite(y2)
            axes[1].errorbar(
                np.asarray(ks_plot, dtype=int)[keep2],
                y2[keep2],
                yerr=y2_std[keep2],
                capsize=3.0,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker="o",
                markersize=4.0,
                markerfacecolor=color,
                markeredgecolor=color,
                label=label,
            )

    axes[0].set_title(f"Classification {metric1[0]}")
    axes[0].set_xlabel("Components k")
    axes[0].set_ylabel(metric1[0])
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(all_ks)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    if metric2 is not None:
        axes[1].set_title(f"Classification {metric2[0]}")
        axes[1].set_xlabel("Components k")
        axes[1].set_ylabel(metric2[0])
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_xticks(all_ks)
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_hotelling_compare(rows: List[Dict[str, float]], output_dir: Path, filename: str = "hotelling_t2_compare.png") -> Optional[Path]:
    if not rows:
        return None
    method_set = {row["method"] for row in rows}
    all_ks = sorted({int(row["k"]) for row in rows})
    ells = sorted({int(row["ell"]) for row in rows if not np.isnan(row.get("ell", float("nan")))})

    start_col = np.array([1.0, 0.7, 0.7])  # light red
    end_col = np.array([0.6, 0.0, 0.0])  # deep red
    n_ells = len(ells)
    ell_colors: Dict[int, np.ndarray] = {}
    for idx, ell in enumerate(ells):
        t = 1.0 if n_ells <= 1 else float(idx) / float(n_ells - 1)
        ell_colors[ell] = (1 - t) * start_col + t * end_col

    method_entries: List[Tuple[Tuple[int, int], str, Optional[int]]] = []
    for method in method_set:
        ell_val = None
        for row in rows:
            if row["method"] != method:
                continue
            ell_raw = float(row.get("ell", float("nan")))
            if not np.isnan(ell_raw):
                ell_val = int(ell_raw)
                break
        sort_key = (0, 0) if ell_val is None else (1, ell_val)
        method_entries.append((sort_key, method, ell_val))
    method_entries.sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for _, method, ell_val in method_entries:
        method_rows = [r for r in rows if r["method"] == method]
        by_k = {int(r["k"]): r for r in method_rows}
        if not by_k:
            continue
        max_k_avail = max(by_k.keys())
        ks_plot = list(all_ks)

        if ell_val is None:
            color = "tab:blue"
            linestyle = "-"
            linewidth = 2.0
            label = "Exact PGA"
        else:
            color = ell_colors.get(ell_val, end_col)
            linestyle = "--"
            linewidth = 2.0
            label = rf"Nyström ($\ell={ell_val}$)"

        t2_vals: List[float] = []
        for k in ks_plot:
            row_k = by_k.get(int(k))
            if row_k is None and ell_val is not None and int(k) > int(max_k_avail):
                row_k = by_k.get(int(max_k_avail))
            t2_vals.append(float(row_k["t2"]) if row_k is not None else float("nan"))
        t2 = np.array(t2_vals, dtype=float)
        keep = np.isfinite(t2) & (t2 > 0.0)
        ax.plot(
            np.asarray(ks_plot, dtype=int)[keep],
            t2[keep],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Components k")
    ax.set_ylabel(r"Hotelling $T^2$")
    ax.set_title(r"Group difference (Hotelling $T^2$)")
    ax.set_xticks(all_ks)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        ks_filter = parse_int_list(args.plot_ks) if args.plot_ks is not None else None
        outs = plot_from_csvs(output_dir, ks_filter=ks_filter)
        for k, v in outs.items():
            print(f"Wrote {k}: {v}")
        return

    t0 = time.time()

    rss_snapshots: Dict[str, int] = {}

    def snapshot_rss(tag: str) -> None:
        rss = get_rss_bytes()
        rss_snapshots[tag] = rss
        if args.report_storage:
            print(f"[rss] {tag}: {format_bytes(rss)}")

    samples = list_hdm05_samples(args.raw_dir, max_files=args.max_files)
    n = len(samples)
    first = np.load(samples[0].path)
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError(f"Expected square matrix in {samples[0].path}")
    dim = int(first.shape[0])
    d = dim * (dim + 1) // 2
    print(f"Found {n} HDM05 matrices of shape {dim}x{dim} (tangent dim d={d}).")
    snapshot_rss("after_load_first_matrix")

    labels = np.array([s.label for s in samples], dtype=int)
    lengths = np.array([s.length for s in samples], dtype=int)
    ids = np.array([s.sample_id for s in samples], dtype=int)

    v_log = np.empty((n, d), dtype=float)
    shifts = np.empty(n, dtype=float)
    min_eigs = np.empty(n, dtype=float)
    if args.report_storage:
        print(f"[storage] v_log bytes: {format_bytes(int(v_log.nbytes))}")
        print(f"[storage] shifts bytes: {format_bytes(int(shifts.nbytes))}")
        print(f"[storage] min_eigs bytes: {format_bytes(int(min_eigs.nbytes))}")
    snapshot_rss("after_allocate_tangent_arrays")
    for i, s in enumerate(tqdm(samples, desc="log-map")):
        mat = np.load(s.path)
        if mat.shape != (dim, dim):
            raise ValueError(f"Inconsistent shape {mat.shape} in {s.path} (expected {(dim, dim)})")
        v_i, shift, min_eig = regularized_log_vec_sym(mat, eps=args.log_eps)
        v_log[i] = v_i
        shifts[i] = shift
        min_eigs[i] = min_eig

    mean_v = v_log.mean(axis=0)
    v_log -= mean_v  # centered tangent vectors V
    v_centered = v_log
    if args.report_storage:
        print(f"[storage] mean_v bytes: {format_bytes(int(mean_v.nbytes))}")
        print(f"[storage] v_centered bytes: {format_bytes(int(v_centered.nbytes))}")
    snapshot_rss("after_log_map_and_center")

    rng = np.random.default_rng(args.seed)
    k_list = sorted(set(parse_int_list(args.ks)))
    ell_list = parse_int_list(args.ells) if args.ells is not None else [int(args.ell)]
    if not ell_list:
        raise ValueError("No sketch sizes provided (empty --ells).")
    ell_list = sorted(set(int(e) for e in ell_list))

    k_max_global = max(k_list) if k_list else min(20, max(ell_list))
    svm_c_grid = parse_float_list(args.svm_c_grid)
    svm_gamma_grid = parse_float_list(args.svm_gamma_grid)
    if args.clf_model == "svm" and args.svm_tune:
        if not svm_c_grid or not svm_gamma_grid:
            raise ValueError("--svm-tune requires non-empty --svm-c-grid and --svm-gamma-grid.")
    mlp_layers = parse_layer_sizes(args.mlp_hidden_layers)
    if not mlp_layers:
        mlp_layers = [100]
    clf_cfg = ClassifierConfig(
        model=args.clf_model,
        svm_tune=bool(args.svm_tune),
        svm_c_grid=svm_c_grid,
        svm_gamma_grid=svm_gamma_grid,
        svm_cv=int(args.svm_cv),
        mlp_hidden_layers=tuple(mlp_layers),
        mlp_alpha=float(args.mlp_alpha),
        mlp_max_iter=int(args.mlp_max_iter),
        mlp_early_stopping=bool(args.mlp_early_stopping),
        mlp_learning_rate_init=float(args.mlp_learning_rate_init),
        xgb_n_estimators=int(args.xgb_n_estimators),
        xgb_max_depth=int(args.xgb_max_depth),
        xgb_learning_rate=float(args.xgb_learning_rate),
        xgb_subsample=float(args.xgb_subsample),
        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
    )

    covariance_storage: Optional[Dict[str, object]] = None
    if args.report_storage:
        covariance_storage = {
            "d": int(d),
            "dtype": str(v_centered.dtype),
            "dense_bytes": int(d * d * v_centered.dtype.itemsize),
        }
        if args.materialize_covariance:
            snapshot_rss("before_materialize_covariance")
            cov_tmp = np.empty((d, d), dtype=v_centered.dtype)
            cov_tmp.fill(1.0)
            snapshot_rss("after_materialize_covariance")
            covariance_storage["dense_bytes_materialized"] = int(cov_tmp.nbytes)
            covariance_storage["rss_delta_bytes"] = int(
                rss_snapshots["after_materialize_covariance"] - rss_snapshots["before_materialize_covariance"]
            )
            if args.report_storage:
                print(
                    "[storage] covariance dense bytes: "
                    f"{format_bytes(int(cov_tmp.nbytes))} (RSS Δ {format_bytes(int(covariance_storage['rss_delta_bytes']))})"
                )
            del cov_tmp
            snapshot_rss("after_free_covariance")
        else:
            if args.report_storage:
                print(
                    "[storage] covariance dense bytes (not materialized): "
                    f"{format_bytes(int(covariance_storage['dense_bytes']))}"
                )

    nystrom_runs: Dict[int, Dict[str, object]] = {}
    scores_nys_by_ell: Dict[int, np.ndarray] = {}
    for ell in ell_list:
        k_max = min(k_max_global, int(ell))
        if k_max <= 0:
            continue
        t_nys = time.time()
        rng_ell = np.random.default_rng(args.seed + 1000 * int(ell))
        snapshot_rss(f"before_nystrom_ell{int(ell)}")
        eigvals_hat, eigvecs_hat, storage = nystrom_pga_from_centered_data(
            v_centered, ell=int(ell), k_max=k_max, rng=rng_ell, report_storage=args.report_storage
        )
        scores_nys_by_ell[int(ell)] = v_centered @ eigvecs_hat
        snapshot_rss(f"after_nystrom_ell{int(ell)}")
        nystrom_runs[int(ell)] = {
            "ell": int(ell),
            "k_max": int(scores_nys_by_ell[int(ell)].shape[1]),
            "eigvals_hat": eigvals_hat.tolist(),
            "runtime_sec": float(time.time() - t_nys),
        }
        if storage is not None:
            storage = dict(storage)
            storage["scores_bytes"] = int(scores_nys_by_ell[int(ell)].nbytes)
            storage["rss_before_bytes"] = int(rss_snapshots[f"before_nystrom_ell{int(ell)}"])
            storage["rss_after_bytes"] = int(rss_snapshots[f"after_nystrom_ell{int(ell)}"])
            storage["rss_delta_bytes"] = int(storage["rss_after_bytes"] - storage["rss_before_bytes"])
            nystrom_runs[int(ell)]["storage"] = storage
            if args.report_storage:
                print(
                    f"[storage] Nyström ell={int(ell)}: eigvecs {format_bytes(int(storage['eigvecs_hat_bytes']))}, "
                    f"scores {format_bytes(int(storage['scores_bytes']))}, RSS Δ {format_bytes(int(storage['rss_delta_bytes']))}"
                )

    eigvals_exact = None
    scores_exact = None
    exact_sec = None
    if args.compare_exact:
        snapshot_rss("before_exact_svd")
        t_ex = time.time()
        eigvals_exact, eigvecs_exact = exact_pga_from_centered_data(v_centered, k_max=k_max_global)
        scores_exact = v_centered @ eigvecs_exact
        exact_sec = float(time.time() - t_ex)
        snapshot_rss("after_exact_svd")
        if args.report_storage:
            print(f"[storage] exact eigvecs bytes: {format_bytes(int(eigvecs_exact.nbytes))}")
            print(f"[storage] exact scores bytes: {format_bytes(int(scores_exact.nbytes))}")
            print(
                f"[storage] exact RSS Δ: {format_bytes(int(rss_snapshots['after_exact_svd'] - rss_snapshots['before_exact_svd']))}"
            )

    chosen = parse_group_labels(args.group_labels)
    if chosen is None:
        label_a, label_b = choose_label_pair(labels)
    else:
        label_a, label_b = chosen
    keep = (labels == label_a) | (labels == label_b)
    idx = np.where(keep)[0]
    y_bin = (labels[idx] == label_b).astype(int)
    n0 = int((y_bin == 0).sum())
    n1 = int((y_bin == 1).sum())
    print(f"Binary downstream labels: {label_a} (n={n0}) vs {label_b} (n={n1}).")

    cls_idx: np.ndarray
    y_cls: np.ndarray
    cls_context: Dict[str, float | str]
    multiclass_label_map = None
    multiclass_counts = None
    if args.clf_task == "binary":
        cls_idx = idx
        y_cls = y_bin
        cls_context = {
            "classification_task": "binary",
            "clf_model": args.clf_model,
            "svm_tune": float(args.svm_tune) if args.clf_model == "svm" else 0.0,
            "mlp_hidden_layers": ",".join(str(x) for x in clf_cfg.mlp_hidden_layers)
            if args.clf_model == "mlp"
            else "",
            "mlp_alpha": float(clf_cfg.mlp_alpha) if args.clf_model == "mlp" else float("nan"),
            "mlp_max_iter": float(clf_cfg.mlp_max_iter) if args.clf_model == "mlp" else float("nan"),
            "mlp_early_stopping": float(args.mlp_early_stopping) if args.clf_model == "mlp" else float("nan"),
            "mlp_learning_rate_init": float(clf_cfg.mlp_learning_rate_init)
            if args.clf_model == "mlp"
            else float("nan"),
            "xgb_n_estimators": float(clf_cfg.xgb_n_estimators) if args.clf_model == "xgb" else float("nan"),
            "xgb_max_depth": float(clf_cfg.xgb_max_depth) if args.clf_model == "xgb" else float("nan"),
            "xgb_learning_rate": float(clf_cfg.xgb_learning_rate) if args.clf_model == "xgb" else float("nan"),
            "xgb_subsample": float(clf_cfg.xgb_subsample) if args.clf_model == "xgb" else float("nan"),
            "xgb_colsample_bytree": float(clf_cfg.xgb_colsample_bytree)
            if args.clf_model == "xgb"
            else float("nan"),
            "label0": float(label_a),
            "label1": float(label_b),
            "n0": float(n0),
            "n1": float(n1),
        }
    else:
        cls_idx, y_cls, multiclass_label_map, multiclass_counts = select_multiclass_subset(
            labels, min_class_count=args.min_class_count, max_classes=args.max_classes
        )
        cls_context = {
            "classification_task": "multiclass",
            "clf_model": args.clf_model,
            "svm_tune": float(args.svm_tune) if args.clf_model == "svm" else 0.0,
            "mlp_hidden_layers": ",".join(str(x) for x in clf_cfg.mlp_hidden_layers)
            if args.clf_model == "mlp"
            else "",
            "mlp_alpha": float(clf_cfg.mlp_alpha) if args.clf_model == "mlp" else float("nan"),
            "mlp_max_iter": float(clf_cfg.mlp_max_iter) if args.clf_model == "mlp" else float("nan"),
            "mlp_early_stopping": float(args.mlp_early_stopping) if args.clf_model == "mlp" else float("nan"),
            "mlp_learning_rate_init": float(clf_cfg.mlp_learning_rate_init)
            if args.clf_model == "mlp"
            else float("nan"),
            "xgb_n_estimators": float(clf_cfg.xgb_n_estimators) if args.clf_model == "xgb" else float("nan"),
            "xgb_max_depth": float(clf_cfg.xgb_max_depth) if args.clf_model == "xgb" else float("nan"),
            "xgb_learning_rate": float(clf_cfg.xgb_learning_rate) if args.clf_model == "xgb" else float("nan"),
            "xgb_subsample": float(clf_cfg.xgb_subsample) if args.clf_model == "xgb" else float("nan"),
            "xgb_colsample_bytree": float(clf_cfg.xgb_colsample_bytree)
            if args.clf_model == "xgb"
            else float("nan"),
            "min_class_count": float(args.min_class_count),
            "max_classes": float(args.max_classes) if args.max_classes is not None else float("nan"),
            "num_classes": float(int(np.unique(y_cls).size)),
            "num_samples": float(int(cls_idx.size)),
        }

    clf_train_idx = None
    clf_test_idx = None
    if args.clf_eval == "holdout":
        clf_train_idx, clf_test_idx = stratified_holdout_split(y_cls, test_size=args.test_size, seed=args.seed)

    group_rows: List[Dict[str, float]] = []
    clf_rows: List[Dict[str, float]] = []
    pairwise_rows: List[Dict[str, float]] = []

    def eval_method(method: str, scores: np.ndarray, ell_val: Optional[int]) -> None:
        for k in k_list:
            if k > scores.shape[1]:
                continue
            feats_group = scores[idx, :k]
            a0 = feats_group[y_bin == 0]
            a1 = feats_group[y_bin == 1]
            t2_stats = hotelling_t2(a0, a1, ridge=1e-8)
            p_perm = hotelling_permutation_pvalue(feats_group, y_bin, n_perm=args.n_perm, seed=args.seed + 1234 + k)
            group_rows.append(
                {
                    "method": method,
                    "ell": float(ell_val) if ell_val is not None else float("nan"),
                    "k": float(k),
                    "label0": float(label_a),
                    "label1": float(label_b),
                    "n0": float(n0),
                    "n1": float(n1),
                    **t2_stats,
                    "p_perm": float(p_perm),
                }
            )

            feats_cls = scores[cls_idx, :k]
            if args.clf_task == "binary":
                if args.clf_eval == "holdout":
                    clf = binary_classification_holdout(
                        feats_cls,
                        y_cls,
                        train_idx=clf_train_idx,
                        test_idx=clf_test_idx,
                        seed=args.seed + 9000 + k,
                        clf_cfg=clf_cfg,
                    )
                else:
                    clf = binary_classification_cv(
                        feats_cls, y_cls, seed=args.seed + 9000 + k, clf_cfg=clf_cfg
                    )
            else:
                if args.clf_eval == "holdout":
                    clf = multiclass_classification_holdout(
                        feats_cls,
                        y_cls,
                        train_idx=clf_train_idx,
                        test_idx=clf_test_idx,
                        seed=args.seed + 9000 + k,
                        clf_cfg=clf_cfg,
                    )
                else:
                    clf = multiclass_classification_cv(
                        feats_cls, y_cls, seed=args.seed + 9000 + k, clf_cfg=clf_cfg
                    )

            clf_rows.append(
                {
                    "method": method,
                    "ell": float(ell_val) if ell_val is not None else float("nan"),
                    "k": float(k),
                    **cls_context,
                    **clf,
                }
            )
            if args.clf_task == "multiclass" and args.pairwise_t2:
                feats_pair = scores[cls_idx, :k]
                pair_rows = pairwise_hotelling_t2(
                    feats_pair,
                    y_cls,
                    multiclass_label_map,
                    ridge=1e-8,
                    correction=args.pairwise_t2_correction,
                )
                for row in pair_rows:
                    pairwise_rows.append(
                        {
                            "method": method,
                            "ell": float(ell_val) if ell_val is not None else float("nan"),
                            "k": float(k),
                            **row,
                        }
                    )

    if args.compare_exact and scores_exact is not None:
        eval_method("exact", scores_exact, ell_val=None)
    for ell in ell_list:
        scores = scores_nys_by_ell.get(int(ell))
        if scores is None:
            continue
        eval_method(f"nystrom(ell={int(ell)})", scores, ell_val=int(ell))

    group_csv = output_dir / ("hotelling_t2_compare.csv" if args.compare_exact else "hotelling_t2.csv")
    if group_rows:
        with group_csv.open("w", newline="") as f_out:
            header = sorted(group_rows[0].keys(), key=lambda k: (k not in {"method", "ell", "k"}, k))
            writer = csv.DictWriter(f_out, fieldnames=header)
            writer.writeheader()
            writer.writerows(group_rows)

    if args.clf_task == "multiclass":
        clf_basename = "classification_multiclass"
    else:
        clf_basename = "classification"
    clf_csv = output_dir / (f"{clf_basename}_compare.csv" if args.compare_exact else f"{clf_basename}.csv")
    if clf_rows:
        with clf_csv.open("w", newline="") as f_out:
            keys: set[str] = set()
            for row in clf_rows:
                keys.update(row.keys())
            priority = [
                "method",
                "ell",
                "k",
                "classification_task",
                "clf_model",
                "svm_tune",
                "mlp_hidden_layers",
                "mlp_alpha",
                "mlp_max_iter",
                "mlp_early_stopping",
                "mlp_learning_rate_init",
                "xgb_n_estimators",
                "xgb_max_depth",
                "xgb_learning_rate",
                "xgb_subsample",
                "xgb_colsample_bytree",
                "split",
                "num_classes",
                "n_train",
                "n_test",
            ]
            header = [k for k in priority if k in keys] + sorted([k for k in keys if k not in priority])
            writer = csv.DictWriter(f_out, fieldnames=header)
            writer.writeheader()
            writer.writerows(clf_rows)

    pairwise_csv = None
    if pairwise_rows:
        pairwise_csv = output_dir / (
            "hotelling_t2_pairwise_compare.csv" if args.compare_exact else "hotelling_t2_pairwise.csv"
        )
        with pairwise_csv.open("w", newline="") as f_out:
            header = sorted(pairwise_rows[0].keys(), key=lambda k: (k not in {"method", "ell", "k"}, k))
            writer = csv.DictWriter(f_out, fieldnames=header)
            writer.writeheader()
            writer.writerows(pairwise_rows)

    compare_plots: Dict[str, str] = {}
    if args.compare_exact:
        clf_png_name = f"{clf_basename}_compare.png"
        p1 = plot_classification_compare(clf_rows, output_dir, filename=clf_png_name)
        p2 = plot_hotelling_compare(group_rows, output_dir)
        if p1 is not None:
            compare_plots[f"{clf_basename}_compare_png"] = str(p1)
        if p2 is not None:
            compare_plots["hotelling_t2_compare_png"] = str(p2)

    counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    runtime = float(time.time() - t0)
    summary = {
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        },
        "storage": {
            "rss_bytes": rss_snapshots,
            "covariance": covariance_storage,
        }
        if args.report_storage
        else None,
        "dataset": {
            "num_samples": int(n),
            "matrix_dim": int(dim),
            "tangent_dim": int(d),
            "num_classes": int(len(counts)),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "min_label_count": int(min(counts.values())),
            "max_label_count": int(max(counts.values())),
            "label_counts": counts,
            "shift_min": float(shifts.min()),
            "shift_max": float(shifts.max()),
            "min_eig_min": float(min_eigs.min()),
            "min_eig_max": float(min_eigs.max()),
        },
        "nystrom": nystrom_runs,
        "exact": {
            "eigvals": eigvals_exact.tolist() if eigvals_exact is not None else None,
            "runtime_sec": exact_sec,
        }
        if args.compare_exact
        else None,
        "binary_task": {
            "label0": int(label_a),
            "label1": int(label_b),
            "n0": int(n0),
            "n1": int(n1),
            "k_list": k_list,
        },
        "classification_task": {
            "task": args.clf_task,
            "model": args.clf_model,
            "svm_tune": bool(args.svm_tune) if args.clf_model == "svm" else False,
            "svm_c_grid": svm_c_grid if args.clf_model == "svm" and args.svm_tune else None,
            "svm_gamma_grid": svm_gamma_grid if args.clf_model == "svm" and args.svm_tune else None,
            "svm_cv": int(args.svm_cv) if args.clf_model == "svm" and args.svm_tune else None,
            "mlp_hidden_layers": list(clf_cfg.mlp_hidden_layers) if args.clf_model == "mlp" else None,
            "mlp_alpha": float(clf_cfg.mlp_alpha) if args.clf_model == "mlp" else None,
            "mlp_max_iter": int(clf_cfg.mlp_max_iter) if args.clf_model == "mlp" else None,
            "mlp_early_stopping": bool(args.mlp_early_stopping) if args.clf_model == "mlp" else None,
            "mlp_learning_rate_init": float(clf_cfg.mlp_learning_rate_init) if args.clf_model == "mlp" else None,
            "xgb_n_estimators": int(clf_cfg.xgb_n_estimators) if args.clf_model == "xgb" else None,
            "xgb_max_depth": int(clf_cfg.xgb_max_depth) if args.clf_model == "xgb" else None,
            "xgb_learning_rate": float(clf_cfg.xgb_learning_rate) if args.clf_model == "xgb" else None,
            "xgb_subsample": float(clf_cfg.xgb_subsample) if args.clf_model == "xgb" else None,
            "xgb_colsample_bytree": float(clf_cfg.xgb_colsample_bytree) if args.clf_model == "xgb" else None,
            "eval": args.clf_eval,
            "test_size": args.test_size if args.clf_eval == "holdout" else None,
            "min_class_count": args.min_class_count if args.clf_task == "multiclass" else None,
            "max_classes": args.max_classes if args.clf_task == "multiclass" else None,
            "num_classes": int(np.unique(y_cls).size),
            "num_samples": int(cls_idx.size),
            "label_map": multiclass_label_map,
            "label_counts": multiclass_counts,
        },
        "outputs": {
            "hotelling_t2_csv": str(group_csv),
            "classification_csv": str(clf_csv),
            "pairwise_t2_csv": str(pairwise_csv) if pairwise_csv is not None else None,
            **compare_plots,
        },
        "runtime_sec": runtime,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Outputs in {output_dir} (runtime {runtime:.1f}s).")


if __name__ == "__main__":
    main()
