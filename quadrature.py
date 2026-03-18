
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from models import Array
from solver import chol_spd, solve_triangular, sym

def gh_tensor_rule(order: int, dim: int) -> Tuple[Array, Array]:
    """Tensor Gauss–Hermite for expectation under N(0, I_dim).
    """

    xs, ws = np.polynomial.hermite.hermgauss(int(order))  # integrates f(x) e^{-x^2}

    # For N(0,1): E[f(U)] = ∫ f(u) φ(u) du with φ(u)=exp(-u^2/2)/sqrt(2π)
    # Convert hermgauss: ∫ f(x) e^{-x^2} dx ≈ Σ ws_i f(xs_i)
    # Let u = sqrt(2) x => φ(u) du = e^{-x^2}/sqrt(π) dx.
    u1 = np.sqrt(2.0) * xs
    w1 = ws / np.sqrt(np.pi)

    grids = np.meshgrid(*([u1] * dim), indexing="ij")
    X = np.stack([g.reshape(-1) for g in grids], axis=1)

    w_grids = np.meshgrid(*([w1] * dim), indexing="ij")
    w = np.prod(np.stack([g.reshape(-1) for g in w_grids], axis=1), axis=1)

    # numerical guard
    w = np.maximum(w, 0.0)
    w = w / float(np.sum(w))
    return X.astype(float), w.astype(float)

def gh_tensor_nodes(order: int, dim: int) -> Array:
    """Return only the Gauss--Hermite nodes for ``N(0, I_dim)``."""
    X, _ = gh_tensor_rule(order=order, dim=dim)
    return X

def rbf_K(X: Array, ell: float, alpha2: float = 1.0, jitter: float = 1e-10) -> Array:
    """EQ kernel Gram matrix: k(x,x') = alpha2 * exp(-||x-x'||^2/(2 ell^2))."""
    X = np.asarray(X, dtype=float)
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    K = float(alpha2) * np.exp(-0.5 * d2 / (float(ell) ** 2))
    K = K + float(jitter) * np.eye(X.shape[0], dtype=float)
    return K

def bq_L_stdnormal(X: Array, ell: float, alpha2: float = 1.0) -> Array:
    """
    L_ij = E[ k(x_i, U) k(U, x_j) ]  for U ~ N(0, I)
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    ell = float(ell)
    alpha2 = float(alpha2)

    r = np.sum(X ** 2, axis=1)     # (N,)
    dot = X @ X.T                  # (N,N)

    c = (alpha2 ** 2) * ((ell ** 2) / (ell ** 2 + 2.0)) ** (0.5 * d)
    E = (2.0 * dot - (ell ** 2 + 1.0) * (r[:, None] + r[None, :])) / (2.0 * ell ** 2 * (ell ** 2 + 2.0))
    return c * np.exp(E)

def bq_bhkf_weights_stdnormal(
    X: Array,
    *,
    ell: float,
    alpha2: float = 1.0,
    jitter: float = 1e-10,
) -> Tuple[Array, Array, float]:
    """
    Returns (w, W, diag_add) for BHKF:
      w = K^{-1} l
      W = K^{-1} L K^{-1}
      diag_add = alpha2 - tr(K^{-1} L)
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    ell = float(ell)
    alpha2 = float(alpha2)
    jitter = float(jitter)

    # K and l
    K = rbf_K(X, ell=ell, alpha2=alpha2, jitter=jitter)
    c = (ell ** 2 / (ell ** 2 + 1.0)) ** (0.5 * d)
    quad = np.sum(X ** 2, axis=1)
    l = alpha2 * c * np.exp(-0.5 * quad / (ell ** 2 + 1.0))

    # L
    Lmat = bq_L_stdnormal(X, ell=ell, alpha2=alpha2)

    # Cholesky solves for stability
    Lk = chol_spd(K, jitter=0.0, max_tries=8)  # K already has jitter

    # w = K^{-1} l
    y = solve_triangular(Lk, l, lower=True)
    w = solve_triangular(Lk.T, y, lower=False)

    # tmp = K^{-1} L
    tmp = solve_triangular(Lk, Lmat, lower=True)
    tmp = solve_triangular(Lk.T, tmp, lower=False)
    tr_KinvL = float(np.trace(tmp))

    # W = K^{-1} L K^{-1} (compute as solve(K, tmp.T).T)
    tmp2 = solve_triangular(Lk, tmp.T, lower=True)
    tmp2 = solve_triangular(Lk.T, tmp2, lower=False)
    W = sym(tmp2.T)

    diag_add = alpha2 - tr_KinvL
    return w, W, float(diag_add)
