

import math
from typing import Dict

import numpy as np

from models import Array
from solver import sym

def sqrtm_spd(A: Array, *, jitter: float = 1e-18) -> Array:
    """Matrix square-root for SPD/SPSD matrices via eigen-decomposition."""
    A = sym(np.asarray(A, dtype=float))
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, float(jitter))
    return V @ np.diag(np.sqrt(w)) @ V.T

def gaussian_w2_sq(m1: Array, C1: Array, m2: Array, C2: Array, *, jitter: float = 1e-18) -> float:
    """Squared 2-Wasserstein distance between Gaussians N(m1,C1) and N(m2,C2).

    W2^2 = ||m1-m2||^2 + tr(C1 + C2 - 2 (C2^{1/2} C1 C2^{1/2})^{1/2})
    """
    m1 = np.asarray(m1, dtype=float).reshape(-1)
    m2 = np.asarray(m2, dtype=float).reshape(-1)
    C1 = sym(np.asarray(C1, dtype=float))
    C2 = sym(np.asarray(C2, dtype=float))

    dm = m1 - m2
    mean_term = float(dm @ dm)

    sqrtC2 = sqrtm_spd(C2, jitter=jitter)
    M = sym(sqrtC2 @ C1 @ sqrtC2)
    sqrtM = sqrtm_spd(M, jitter=jitter)

    tr_term = float(np.trace(C1 + C2 - 2.0 * sqrtM))
    # Numerical guard: tr_term can get tiny negative due to rounding.
    if tr_term < 0.0 and tr_term > -1e-10:
        tr_term = 0.0
    return mean_term + tr_term

def trajectory_gaussian_w2(
    mean: Array,
    cov: Array,
    mean_ref: Array,
    cov_ref: Array,
) -> Array:
    """Per-time W2 distances between (mean,cov) and reference (mean_ref,cov_ref)."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    mean_ref = np.asarray(mean_ref, dtype=float)
    cov_ref = np.asarray(cov_ref, dtype=float)

    Tn = mean.shape[0]
    w2 = np.zeros((Tn,), dtype=float)
    for k in range(Tn):
        w2_sq = gaussian_w2_sq(mean[k], cov[k], mean_ref[k], cov_ref[k])
        w2[k] = math.sqrt(max(float(w2_sq), 0.0))
    return w2

def reduce_w2(w2_t: Array, *, mode: str = "rms") -> float:
    w2_t = np.asarray(w2_t, dtype=float).reshape(-1)
    m = str(mode).lower()
    return {
        "rms": float(np.sqrt(np.mean(w2_t ** 2))),
        "mean": float(np.mean(w2_t)),
        "max": float(np.max(w2_t)),
    }[m]

def w2_rms(mean: Array, cov: Array, mean_ref: Array, cov_ref: Array) -> float:
    """W2rms score."""
    w2_t = trajectory_gaussian_w2(mean, cov, mean_ref, cov_ref)
    return float(np.sqrt(np.mean(w2_t ** 2)))

def trajectory_metric_summary(
    mean: Array,
    cov: Array,
    mean_ref: Array,
    cov_ref: Array,
) -> Dict[str, float]:
    """Time-aggregated trajectory metrics."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    mean_ref = np.asarray(mean_ref, dtype=float)
    cov_ref = np.asarray(cov_ref, dtype=float)

    w2 = w2_rms(mean, cov, mean_ref, cov_ref)
    mean_rmse = float(np.sqrt(np.mean(np.sum((mean - mean_ref) ** 2, axis=1))))
    return {"W2rms": float(w2), "mean_RMSE": mean_rmse}
