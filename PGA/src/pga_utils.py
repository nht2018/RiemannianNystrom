"""
Utilities for SPD matrix handling, log-Euclidean PGA geometry, and Nyström sketches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.linalg as la

SQRT2 = math.sqrt(2.0)


def vec_sym(mat: np.ndarray) -> np.ndarray:
    """
    Vectorize a symmetric matrix with sqrt(2) scaling on off-diagonals to preserve Frobenius norms.
    """
    n = mat.shape[0]
    vec = np.empty(n * (n + 1) // 2, dtype=float)
    idx = 0
    for i in range(n):
        vec[idx] = mat[i, i]
        idx += 1
        for j in range(i + 1, n):
            vec[idx] = SQRT2 * mat[i, j]
            idx += 1
    return vec


def mat_from_vec(vec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of vec_sym."""
    mat = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        mat[i, i] = vec[idx]
        idx += 1
        for j in range(i + 1, n):
            val = vec[idx] / SQRT2
            mat[i, j] = val
            mat[j, i] = val
            idx += 1
    return mat


def regularize_to_spd(mat: np.ndarray, eps: float = 1e-4) -> Tuple[np.ndarray, float]:
    """Symmetrize and clip eigenvalues to ensure SPD; returns (spd_matrix, shift_used)."""
    sym = 0.5 * (mat + mat.T)
    w, u = la.eigh(sym)
    shift = max(eps - w.min(), 0.0)
    w_shifted = w + shift
    w_clipped = np.clip(w_shifted, eps, None)
    spd = (u * w_clipped) @ u.T
    spd = 0.5 * (spd + spd.T)
    return spd, float(shift)


def sym_logm(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log of an SPD matrix via eigendecomposition with clipping for stability."""
    w, u = la.eigh(mat)
    w_clipped = np.clip(w, eps, None)
    log_w = np.log(w_clipped)
    return (u * log_w) @ u.T


def sym_expm(mat: np.ndarray) -> np.ndarray:
    """Exp of a symmetric matrix via eigendecomposition."""
    w, u = la.eigh(mat)
    exp_w = np.exp(w)
    return (u * exp_w) @ u.T


def log_euclidean_mean(mats: Sequence[np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """Compute the log-Euclidean Fréchet mean."""
    logs = [sym_logm(m, eps) for m in mats]
    mean_log = np.mean(logs, axis=0)
    return sym_expm(mean_log)


def centered_log_maps(mats: Sequence[np.ndarray], eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_matrix, centered log maps) using log-Euclidean geometry."""
    logs = [sym_logm(m, eps) for m in mats]
    mean_log = np.mean(logs, axis=0)
    centered = [log - mean_log for log in logs]
    return sym_expm(mean_log), np.stack(centered, axis=0)


def covariance_matrix(z_flat: np.ndarray) -> np.ndarray:
    return z_flat.T @ z_flat / z_flat.shape[0]


def nystrom_covariance(cov: np.ndarray, ell: int, rng: np.random.Generator) -> np.ndarray:
    """Nyström sketch of a covariance matrix."""
    d = cov.shape[0]
    if ell > d:
        raise ValueError(f"Sketch size ℓ={ell} exceeds dimension d={d}.")
    p = rng.standard_normal((d, ell))
    b = cov @ p
    w = p.T @ b
    w_pinv = np.linalg.pinv(w)
    cov_hat = b @ w_pinv @ b.T
    cov_hat = 0.5 * (cov_hat + cov_hat.T)
    return cov_hat


def eigh_desc(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w, u = la.eigh(mat)
    idx = w.argsort()[::-1]
    return w[idx], u[:, idx]


def projector(vecs: np.ndarray, k: int) -> np.ndarray:
    return vecs[:, :k] @ vecs[:, :k].T


def reconstruction_error(z_flat: np.ndarray, proj: np.ndarray) -> float:
    residual = z_flat - z_flat @ proj
    return float(np.mean(np.sum(residual * residual, axis=1)))


def compute_metrics(
    z_flat: np.ndarray,
    cov: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    cov_hat: np.ndarray,
    eigvals_hat: np.ndarray,
    eigvecs_hat: np.ndarray,
    k_list: Iterable[int],
) -> Dict[str, float]:
    """Same metric set as synthetic PGA: spectral error, variance ratio, subspace gap, recon ratio."""
    metrics: Dict[str, float] = {}
    spectral_error = la.norm(cov - cov_hat, 2)
    spectral_error_rel = spectral_error / max(la.norm(cov, 2), 1e-12)
    metrics["spectral_error"] = spectral_error
    metrics["spectral_error_rel"] = spectral_error_rel
    total_var = float(np.sum(eigvals))
    total_var_hat = float(np.sum(eigvals_hat))
    for k in k_list:
        proj = projector(eigvecs, k)
        proj_hat = projector(eigvecs_hat, k)
        metrics[f"subspace_fro_k{k}"] = la.norm(proj - proj_hat, "fro")
        v_exact = float(np.sum(eigvals[:k]) / total_var)
        v_hat = float(np.sum(eigvals_hat[:k]) / max(total_var_hat, 1e-12))
        metrics[f"variance_ratio_k{k}"] = v_hat / max(v_exact, 1e-12)
        recon_exact = reconstruction_error(z_flat, proj)
        recon_hat = reconstruction_error(z_flat, proj_hat)
        metrics[f"recon_ratio_k{k}"] = recon_hat / max(recon_exact, 1e-12)
    return metrics


def aggregate_metrics(rows: List[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k != "ell"]
    agg: Dict[str, float] = {}
    agg["ell"] = float(rows[0].get("ell", 0.0))
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=float)
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return agg


@dataclass
class PGASpectrum:
    eigvals: np.ndarray
    eigvecs: np.ndarray
    mean_matrix: np.ndarray
    z_flat: np.ndarray
