"""Centralized definitions for all functions in this repository.

This module collects and defines all reusable functions from the standalone
scripts so you can import them from one place. Functions duplicated across
modules are provided in two versions: the primary versions keep their original
names, while variants originating from ``Propagating_Model_probnum_package.py``
use a ``_probnum`` suffix to avoid name collisions.
"""

from __future__ import annotations

import math
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp

# ============================================================
# Linear_ODE.py functions
# ============================================================


def gaussian_w2_distance(mean1, var1, mean2, var2):
    """
    mean*, var* : arrays over time (same shape)

    For each time k:
      p_k = N(mean1[k], var1[k])
      q_k = N(mean2[k], var2[k])

    W2^2(p_k, q_k) = (m1 - m2)^2 + (sqrt(v1) - sqrt(v2))^2

    Aggregate over time:
      D = sqrt( mean_k W2^2(p_k, q_k) )
    """
    mean1 = np.asarray(mean1)
    mean2 = np.asarray(mean2)
    var1 = np.asarray(var1)
    var2 = np.asarray(var2)

    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)

    d2 = (mean1 - mean2) ** 2 + (std1 - std2) ** 2
    D = math.sqrt(np.mean(d2))
    return D


def analytic_joint_gaussian(t_grid, a, b, m_theta, P_theta):
    t = np.asarray(t_grid)
    if a != 0.0:
        L = np.exp(a * t)
        c = (b / a) * (np.exp(a * t) - 1.0)
    else:
        L = np.ones_like(t)
        c = b * t
    mean = L * m_theta + c
    var = (L ** 2) * P_theta
    return mean, var, L


def solve_single_theta_lsoda(theta, t_grid, a, b, rtol=1e-12, atol=1e-12):
    def f(t, y):
        return a * y + b

    sol = solve_ivp(
        f,
        (float(t_grid[0]), float(t_grid[-1])),
        y0=[float(theta)],
        t_eval=t_grid,
        method="LSODA",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"LSODA failed: {sol.message}")
    return sol.y[0]


def mc_lsoda(t_grid, a, b, m_theta, P_theta, n_samples, rtol=1e-12, atol=1e-12, seed=0):
    """
    Standard MC+LSODA with n_samples, used for reference and timing.
    """
    rng = np.random.default_rng(seed)
    thetas = rng.normal(loc=m_theta, scale=math.sqrt(P_theta), size=n_samples)
    N_t = len(t_grid)
    Y = np.empty((n_samples, N_t))
    for i, theta in enumerate(thetas):
        Y[i] = solve_single_theta_lsoda(theta, t_grid, a, b, rtol, atol)
    mean = Y.mean(axis=0)
    var = Y.var(axis=0, ddof=1)
    return mean, var


def pn_kalman_path(theta, t_grid, a, b, q_c=1e-2, r_var=1e-6, with_sensitivity=False):
    """
    Run PN Kalman filter for dy/dt = a y + b with IWP(1) prior.

    State: x_k = [y_k, v_k]^T, v ≈ dy/dt.
    Prior: IWP(1), discretized with step h.
    ODE pseudo-observation: b = -a y + v + noise.

    If with_sensitivity=True, also propagate
      J_k = d x_mean_k / d theta
    and return J_theta_y = d y_mean_k / d theta.
    """
    t_grid = np.asarray(t_grid)
    N = t_grid.size
    dim = 2

    # Measurement model: z = H x + eps, with z = b (scalar)
    # ODE: v = a y + b  ->  b = -a y + v
    H = np.array([[-a, 1.0]])
    R = np.array([[r_var]])

    x_mean = np.zeros((N, dim))
    P = np.zeros((N, dim, dim))

    # initial state: y(0)=theta, v(0)=a theta + b
    x_mean[0] = np.array([theta, a * theta + b])
    P[0] = np.zeros((dim, dim))

    if with_sensitivity:
        # J[k] = d x_mean[k] / d theta, shape (2,)
        # At t=0: y = theta, v = a theta + b
        # so dy/dtheta = 1, dv/dtheta = a
        J = np.zeros_like(x_mean)
        J[0] = np.array([1.0, a])

    I = np.eye(dim)

    for k in range(1, N):
        h = t_grid[k] - t_grid[k - 1]
        A = np.array([[1.0, h], [0.0, 1.0]])
        Q = q_c * np.array([[h**3 / 3.0, h**2 / 2.0], [h**2 / 2.0, h]])

        # predict
        x_pred = A @ x_mean[k - 1]
        P_pred = A @ P[k - 1] @ A.T + Q

        if with_sensitivity:
            # J_pred = A J_{k-1}  (no explicit theta in dynamics)
            J_pred = A @ J[k - 1]

        # update with ODE pseudo-observation: b = -a y + v
        z = np.array([b])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innov = z - H @ x_pred

        x_new = x_pred + (K @ innov).ravel()
        P_new = (I - K @ H) @ P_pred

        x_mean[k] = x_new
        P[k] = P_new

        if with_sensitivity:
            # J_new = (I - K H) J_pred
            J[k] = (I - K @ H) @ J_pred

    y_mean = x_mean[:, 0]
    y_var = P[:, 0, 0]

    if with_sensitivity:
        J_theta_y = J[:, 0]
        return y_mean, y_var, J_theta_y
    else:
        return y_mean, y_var


def pn_kalman_state_with_sensitivities(theta, t_grid, a, b, q_c=1e-2, r_var=1e-6):
    """
    Run PN Kalman filter and propagate:

      - x_mean[k]  : filtered mean of x_k = [y_k, v_k]^T
      - P[k]       : filtered covariance of x_k
      - Jx[k]      : d m_k / d x_0   (shape 2x2)
      - Jtheta[k]  : d m_k / d theta (shape 2x1)

    in a linear-Gaussian model with IWP(1) prior and
    ODE pseudo-observation b = -a y + v + noise.
    """
    t_grid = np.asarray(t_grid)
    N = t_grid.size
    d = 2
    p = 1

    # Measurement model: z = H x + eps, with z = b
    H = np.array([[-a, 1.0]])
    R = np.array([[r_var]])

    # Filtered mean and covariance at fixed theta
    x_mean = np.zeros((N, d))
    P = np.zeros((N, d, d))
    x_mean[0] = np.array([theta, a * theta + b])
    P[0] = np.zeros((d, d))

    # Sensitivities at k=0
    Jx = np.zeros((N, d, d))
    Jtheta = np.zeros((N, d, p))
    Jx[0] = np.eye(d)
    Jtheta[0] = np.zeros((d, p))

    I = np.eye(d)

    for k in range(1, N):
        h = t_grid[k] - t_grid[k - 1]

        # IWP(1) transition
        A = np.array([[1.0, h], [0.0, 1.0]])
        Q = q_c * np.array([[h**3 / 3.0, h**2 / 2.0], [h**2 / 2.0, h]])

        # Prediction
        x_pred = A @ x_mean[k - 1]
        P_pred = A @ P[k - 1] @ A.T + Q

        # Sensitivity prediction (no explicit theta in dynamics: B = 0)
        B = np.zeros((d, p))
        Jx_minus = A @ Jx[k - 1]
        Jtheta_minus = A @ Jtheta[k - 1] + B

        # Update with pseudo-observation b = -a y + v
        z = np.array([b])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innov = z - H @ x_pred
        x_new = x_pred + (K @ innov).ravel()
        P_new = (I - K @ H) @ P_pred

        x_mean[k] = x_new
        P[k] = P_new

        # Sensitivity update (no explicit theta in obs: D = 0)
        D = np.zeros((1, p))
        KH = K @ H
        Jx[k] = (I - KH) @ Jx_minus
        Jtheta[k] = (I - KH) @ Jtheta_minus - K @ D

    return x_mean, P, Jx, Jtheta


def pn_joint_gaussian(t_grid, a, b, m_theta, P_theta, q_c, r_var):
    """
    Goal distribution p_goal(y(t)) for the probabilistic ODE solver.

    This combines:
      - the conditional Kalman covariance P_k (uncertainty from the
        IWP(1) prior and pseudo-observations), and
      - the propagated uncertainty from the random pair (x_0, theta).

    The latter is described by the sensitivities Jx_k, Jtheta_k and the
    joint covariance Sigma_0 of (x_0, theta):

        P_goal_k = P_k + J_k Sigma_0 J_k^T,

    with J_k = [Jx_k  Jtheta_k].
    """
    # Run filter at theta = m_theta and get full state + sensitivities
    x_mean, P_bar, Jx, Jtheta = pn_kalman_state_with_sensitivities(
        m_theta, t_grid, a, b, q_c, r_var
    )

    # Build Sigma_0 for (x_0, theta).
    # Deterministic relationship x_0 = G theta, with
    #   G = [1, a]^T,
    # and theta ~ N(m_theta, P_theta).
    G = np.array([[1.0], [a]])

    # P0      = Cov(x_0)        = G P_theta G^T
    # P0theta = Cov(x_0, theta) = G P_theta
    P0 = P_theta * (G @ G.T)
    P0theta = P_theta * G

    Sigma0 = np.block([
        [P0, P0theta],
        [P0theta.T, np.array([[P_theta]])],
    ])

    N = len(t_grid)
    y_mean = x_mean[:, 0].copy()
    y_var = np.empty(N)

    for k in range(N):
        # Stack sensitivities into J_k = [Jx_k  Jtheta_k], shape (2,3)
        J_k = np.hstack([Jx[k], Jtheta[k]])

        # Goal covariance of the full state x_k
        P_goal_k = P_bar[k] + J_k @ Sigma0 @ J_k.T

        # Extract variance of y_k (first component)
        y_var[k] = P_goal_k[0, 0]

    return y_mean, y_var


def incremental_mc_lsoda_until_tol(
    t_grid,
    a,
    b,
    m_theta,
    P_theta,
    ref_mean,
    ref_var,
    dist_tol,
    max_samples=50_000,
    batch_size=100,
    rtol=1e-12,
    atol=1e-12,
    seed=42,
):
    """
    Incremental Monte Carlo with LSODA:
      - draw samples in batches of 'batch_size'
      - after each batch, update running mean/var using Welford
      - compute W2 distance to (ref_mean, ref_var)
      - stop as soon as distance <= dist_tol, or when max_samples reached

    Returns:
      n_used, mean_est, var_est
    """
    rng = np.random.default_rng(seed)
    N_t = len(t_grid)

    # Welford initialisation
    n = 0
    mean_est = np.zeros(N_t)
    M2 = np.zeros(N_t)  # sum of squared deviations

    while n < max_samples:
        # How many new samples in this batch?
        remaining = max_samples - n
        this_batch = min(batch_size, remaining)

        # Draw new thetas
        thetas = rng.normal(loc=m_theta, scale=math.sqrt(P_theta), size=this_batch)

        # Process each theta one by one for Welford updates
        for theta in thetas:
            y = solve_single_theta_lsoda(theta, t_grid, a, b, rtol, atol)
            n += 1
            # Welford update
            delta = y - mean_est
            mean_est = mean_est + delta / n
            delta2 = y - mean_est
            M2 = M2 + delta * delta2

        # After each batch, check distance (if we have at least 2 samples)
        if n >= 2:
            var_est = M2 / (n - 1)
            dist = gaussian_w2_distance(ref_mean, ref_var, mean_est, var_est)
            if dist <= dist_tol:
                return n, mean_est.copy(), var_est.copy()

    # If we get here, we hit max_samples without meeting tol
    var_est = M2 / max(1, (n - 1))
    return n, mean_est, var_est


def time_call(fn):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return t1 - t0


# ============================================================
# dual_try.py functions
# ============================================================


def OU_model(lamb, sigma_stat, dt):
    F = np.exp(-lamb * dt)
    Q = sigma_stat**2 * (1.0 - np.exp(-2.0 * lamb * dt))
    return F, Q


def Simulate_Data(lamb=1.2, sigma_x=1.0, sigma_y=0.2, T=10.0, dt=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(0.0, T, dt)
    n = len(t)
    F, Q = OU_model(lamb, sigma_x, dt)
    R = sigma_y**2
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = rng.normal(0.0, sigma_x)
    y[0] = x[0] + rng.normal(0.0, np.sqrt(R))
    for i in range(1, n):
        x[i] = F * x[i - 1] + rng.normal(0.0, np.sqrt(Q))
        y[i] = x[i] + rng.normal(0.0, np.sqrt(R))

    return t, x, y, F, Q, R


def OU_kernel(t, lamb, sigma_x):
    """Stationary OU covariance kernel on a grid t."""
    dt = np.abs(t[:, None] - t[None, :])
    return (sigma_x**2) * np.exp(-lamb * dt)


def chol_solve(L, b):
    """Solve (L L^T) x = b for x, given lower-triangular L (Cholesky)."""
    z = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, z)
    return x


def dual_smoother(y, K, R, jitter=1e-9):
    # smoother algorithm
    n = len(y)
    A = K + (R + jitter) * np.eye(n)
    # Cholesky factor of A
    L = np.linalg.cholesky(A)
    # alpha
    alpha = chol_solve(L, y)
    m = K @ alpha
    # For diagonal of posterior covariance: diag(K - K A^{-1} K)
    # Compute A^{-1} K
    AinvK = chol_solve(L, K)
    diag_term = np.einsum("ij,ji->i", K, AinvK)
    P_diag = np.diag(K) - diag_term
    return m, P_diag


def dual_filter(y, K, R, jitter=1e-9):
    # filtering algorithm
    n = len(y)
    m_flt = np.zeros(n)
    P_flt = np.zeros(n)

    # Step k=1
    A11 = K[0, 0] + R + jitter
    L = np.array([[np.sqrt(A11)]])
    # alpha_1
    alpha = chol_solve(L, y[:1])
    # mean at t_0
    m_flt[0] = K[0, 0] * alpha[0]
    # variance at t_0
    s = chol_solve(L, K[:1, 0])
    P_flt[0] = K[0, 0] - K[0, :1] @ s

    for k in range(1, n):
        # Build A_k
        a = K[:k, k].copy()
        a_nn = K[k, k] + R + jitter

        # Compute new column in L by solving L w = a
        w = np.linalg.solve(L, a)
        l_kk = np.sqrt(a_nn - np.dot(w, w))
        # Append to L
        L = np.block([
            [L, np.zeros((k, 1))],
            [w[None, :], np.array([[l_kk]])],
        ])

        # Update alpha by solving (L L^T) alpha = y[:k+1]
        alpha = chol_solve(L, y[:k+1])

        # Posterior mean at t_k using only y[:k+1]
        m_flt[k] = K[k, :k+1] @ alpha

        # Posterior variance at t_k: diag element
        s = chol_solve(L, K[:k+1, k])
        P_flt[k] = K[k, k] - K[k, :k+1] @ s

    return m_flt, P_flt


# ============================================================
# Propagating_Model.py functions (primary versions)
# ============================================================


def spherical_cubature(mu, Sigma):
    """
    Spherical cubature algorithm for N(mu, Sigma).
    """
    mu = np.atleast_1d(mu)
    d = mu.shape[0]
    jitter = 1e-12 * np.eye(d)
    L = cholesky(Sigma + jitter)
    nodes = []
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = 1.0
        shift = math.sqrt(d) * (L @ ei)
        nodes.append(mu + shift)
        nodes.append(mu - shift)
    nodes = np.stack(nodes, axis=0)
    w = np.full(2 * d, 1.0 / (2 * d))
    return nodes, w


def gauss_hermite_cubature(mu, Sigma, n_points_1d=5):
    """
    Multivariate Gauss–Hermite cubature for a Gaussian N(mu, Sigma).
    """
    mu = np.atleast_1d(mu)
    d = mu.shape[0]

    # Cholesky of covariance with jitter for stability
    jitter = 1e-12 * np.eye(d)
    L = cholesky(Sigma + jitter)

    # 1D Gauss–Hermite nodes/weights for ∫ e^{-x^2} f(x) dx
    x_1d, w_1d = hermgauss(n_points_1d)

    # Build d-dimensional tensor-product grid
    grids = np.meshgrid(*([x_1d] * d), indexing="ij")
    u_grid = np.stack(grids, axis=-1).reshape(-1, d)

    # Corresponding product weights
    w_grids = np.meshgrid(*([w_1d] * d), indexing="ij")
    w_prod = np.prod(np.stack(w_grids, axis=-1), axis=-1).reshape(-1)

    # Convert to standard normal Z ~ N(0, I):
    z_nodes = np.sqrt(2.0) * u_grid

    # Weights for expectation w.r.t. Z ~ N(0, I)
    w = (1.0 / (np.pi ** (d / 2.0))) * w_prod

    # Transform Z to theta = mu + L Z for N(mu, Sigma)
    nodes = mu + z_nodes @ L.T

    return nodes, w


def logistic_fun(t, y, a, b):
    # y' = a y (1 - y/b)
    return a * y * (1.0 - y / b)


def logistic_jacobian(t, y, a, b):
    """Analytic Jacobian of the logistic RHS."""
    return np.array([[a * (1.0 - 2.0 * y[0] / b)]])


def fhn_fun(t, y, a, b, c, d):
    # FitzHugh–Nagumo in (y1, y2)
    y1, y2 = y
    dy1 = y1 - (y1**3) / 3.0 - y2 + a
    dy2 = (y1 + b - c * y2) / d
    return np.array([dy1, dy2])


def lotkavolterra_fun(t, y, a, b, c, d):
    # [ y1' = a*y1 - b*y1*y2,  y2' = -c*y2 + d*y1*y2 ]
    y1, y2 = y
    return np.array([a * y1 - b * y1 * y2, -c * y2 + d * y1 * y2])


def vanderpol_fun(t, y, mu):
    # y1' = y2
    # y2' = mu * (1 - y1**2) * y2 - y1
    y1, y2 = y
    return np.array([y2, mu * (1 - y1**2) * y2 - y1])


def J_fhn(t, y, a=0.0, b=0.08, c=0.07, d=1.25):
    y1, y2 = y
    df1_dy1 = 1.0 - y1**2
    df1_dy2 = -1.0
    df2_dy1 = 1.0 / d
    df2_dy2 = -c / d
    return np.array([[df1_dy1, df1_dy2], [df2_dy1, df2_dy2]])


def J_lv(t, y, a=5.0, b=0.5, c=5.0, d=0.5):
    y1, y2 = y
    df1_dy1 = a - b * y2
    df1_dy2 = -b * y1
    df2_dy1 = d * y2
    df2_dy2 = -c + d * y1
    return np.array([[df1_dy1, df1_dy2], [df2_dy1, df2_dy2]])


def J_vdp(t, y, mu=0.05):
    y1, y2 = y
    df1_dy1 = 0.0
    df1_dy2 = 1.0
    df2_dy1 = -2.0 * mu * y1 * y2 - 1.0
    df2_dy2 = mu * (1.0 - y1**2)
    return np.array([[df1_dy1, df1_dy2], [df2_dy1, df2_dy2]])


def _numerical_jacobian(fun, t, y, eps=1e-6):
    """
    Finite-difference Jacobian used when an analytic Jacobian is not provided.
    """
    y = np.asarray(y, dtype=float)
    d = y.size
    J = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = eps
        f_plus = fun(t, y + e)
        f_minus = fun(t, y - e)
        J[:, i] = (f_plus - f_minus) / (2.0 * eps)
    return J


def integrate_deterministic(fun, t_span, y0, args=(), t_eval=None, rtol=1e-6, atol=1e-8):
    sol = solve_ivp(
        fun,
        t_span,
        y0,
        method="LSODA",
        args=args,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        warnings.warn(f"Integration failed: {sol.message}")
    return sol.t, sol.y


def integrate_probnum(
    fun,
    jac_fun,
    t_span,
    y0,
    args=(),
    t_eval=None,
    kappa2=1.0,
    R_scale=1e-6,
):
    """
    Probabilistic ODE solver following the EK1-style IWP(1) filter used in
    ProbNum (see https://arxiv.org/pdf/2503.04684). This mirrors the structure
    of :func:`ek1_iwp1_goal_cov` but solves a deterministic IVP while returning a
    distribution over solver states.

    Returns t_grid, mean_y (d, M), std_y (d, M).
    """
    if t_eval is None:
        raise ValueError("t_eval must be provided for probabilistic integration")

    t_eval = np.asarray(t_eval)
    if not np.allclose(np.diff(t_eval), np.diff(t_eval)[0]):
        warnings.warn(
            "t_eval is not uniform; EK1 step assumes fixed step size. "
            "Using first step size as approximation."
        )

    t0, t1 = t_span
    if not np.isclose(t_eval[0], t0):
        raise ValueError("t_eval[0] must match t_span[0] for EK1 integration")

    h = float(np.diff(t_eval)[0])
    d = len(y0)

    A, Q = iwp1_matrices(h, kappa2, d)
    E0, E1 = build_E0_E1(d)
    R = R_scale * np.eye(d)

    def f(t, y):
        return fun(t, y, *args)

    def J_f(t, y):
        if jac_fun is None:
            return _numerical_jacobian(lambda _t, _y: fun(_t, _y, *args), t, y)
        return jac_fun(t, y, *args)

    m = np.concatenate([y0, f(t0, y0)])
    P = 1e-12 * np.eye(2 * d)

    y_mean = np.zeros((d, t_eval.size))
    y_var = np.zeros_like(y_mean)

    y_mean[:, 0] = y0
    y_var[:, 0] = 0.0

    for k in range(1, t_eval.size):
        t = t_eval[k]

        m_pred = A @ m
        P_pred = A @ P @ A.T + Q

        y_pred = E0 @ m_pred
        f_val = f(t, y_pred)
        Jf = J_f(t, y_pred)

        h_pred = E1 @ m_pred - f_val
        H = np.hstack([-Jf, np.eye(d)])

        S = H @ P_pred @ H.T + R
        K = np.linalg.solve(S, (P_pred @ H.T).T).T

        m = m_pred + K @ (-h_pred)
        P = P_pred - K @ S @ K.T

        P_y = E0 @ P @ E0.T
        y_mean[:, k] = E0 @ m
        y_var[:, k] = np.diag(P_y)

    return t_eval, y_mean, np.sqrt(y_var)


def propagate_deterministic(
    system,
    t_span,
    t_eval,
    theta_mean,
    theta_cov,
    quad_method="spherical",
    n_gh_1d=5,
    solver="probnum",
):
    """
    Propagate uncertainty using a deterministic quadrature rule
    (spherical cubature or Gauss–Hermite) over theta.
    """
    name = system["name"]
    ode_fun = system["ode_fun"]
    jac_fun = system.get("jac_fun")
    theta_to_setup = system["theta_to_setup"]

    t0 = time.perf_counter()

    if quad_method == "spherical":
        nodes, w = spherical_cubature(theta_mean, theta_cov)
    elif quad_method == "gh":
        nodes, w = gauss_hermite_cubature(theta_mean, theta_cov, n_points_1d=n_gh_1d)
    else:
        raise ValueError(f"Unknown quad_method: {quad_method}")

    Y_nodes = []
    t_out = None
    for th in nodes:
        y0, params = theta_to_setup(th)
        if solver == "probnum":
            t, y_mean, y_std = integrate_probnum(ode_fun, jac_fun, t_span, y0, args=params, t_eval=t_eval)
            y = y_mean
            y_var = y_std**2
        elif solver == "deterministic":
            t, y_det = integrate_deterministic(ode_fun, t_span, y0, args=params, t_eval=t_eval)
            y = y_det
            y_var = np.zeros_like(y_det)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if t_out is None:
            t_out = t
        Y_nodes.append((y, y_var))

    # Weighted mean and variance (including solver covariance)
    Y_means = np.stack([p[0] for p in Y_nodes], axis=0)
    Y_vars = np.stack([p[1] for p in Y_nodes], axis=0)

    mean = np.tensordot(w, Y_means, axes=(0, 0))
    diffs = Y_means - mean[None, :, :]
    var = np.tensordot(w, Y_vars + diffs**2, axes=(0, 0))
    std = np.sqrt(var)

    t1 = time.perf_counter()
    return {
        "t": t_out,
        "mean": mean,
        "std": std,
        "time": t1 - t0,
        "method": quad_method,
        "name": name,
    }


def propagate_mc(system, t_span, t_eval, theta_mean, theta_cov, n_mc=400, solver="probnum"):
    """
    Monte Carlo reference propagation.
    """
    name = system["name"]
    ode_fun = system["ode_fun"]
    jac_fun = system.get("jac_fun")
    theta_to_setup = system["theta_to_setup"]

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    Y_mc = []
    Y_vars = []
    t_out = None
    for _ in range(n_mc):
        theta = rng.multivariate_normal(theta_mean, theta_cov)
        y0, params = theta_to_setup(theta)
        if solver == "probnum":
            t, y_mean, y_std = integrate_probnum(ode_fun, jac_fun, t_span, y0, args=params, t_eval=t_eval)
            y_var = y_std**2
            y = y_mean
        elif solver == "deterministic":
            t, y = integrate_deterministic(ode_fun, t_span, y0, args=params, t_eval=t_eval)
            y_var = np.zeros_like(y)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if t_out is None:
            t_out = t
        Y_mc.append(y)
        Y_vars.append(y_var)

    Y_mc = np.stack(Y_mc, axis=0)
    Y_vars = np.stack(Y_vars, axis=0)
    mc_mean = np.mean(Y_mc, axis=0)
    mc_var = np.mean(Y_vars, axis=0) + np.var(Y_mc, axis=0, ddof=1)
    mc_std = np.sqrt(mc_var)

    t1 = time.perf_counter()
    return {
        "t": t_out,
        "mean": mc_mean,
        "std": mc_std,
        "time": t1 - t0,
        "method": "mc",
        "name": name,
    }


def iwp1_matrices(h, kappa2, d):
    """
    Once-integrated Wiener process (IWP(1)) prior for d-dimensional y(t).

    State x = [y; v] \in R^{2d} with block transition:
      [ y_k ]   [1 h] [y_{k-1}] + w_k
      [ v_k ] = [0 1] [v_{k-1}]
    and w_k ~ N(0, Q(h)).
    """
    A_block = np.array([[1.0, h], [0.0, 1.0]])
    Q_block = kappa2 * np.array([[h**3 / 3.0, h**2 / 2.0], [h**2 / 2.0, h]])
    A = np.kron(np.eye(d), A_block)
    Q = np.kron(np.eye(d), Q_block)
    return A, Q


def build_E0_E1(d):
    """Projections: y = E0 x, v = E1 x for x = [y; v]."""
    E0 = np.hstack([np.eye(d), np.zeros((d, d))])
    E1 = np.hstack([np.zeros((d, d)), np.eye(d)])
    return E0, E1


def ek1_iwp1_goal_cov(f, J_f, mu0, Sigma0, T, h, kappa2=1.0, R_scale=1e-6):
    """
    EK1 ODE filter with IWP(1) prior + Jacobian-based goal covariance
    for uncertain initial y(0) ~ N(mu0, Sigma0).

    Returns:
      t_grid, m_list, P_list, P_goal_list
    """
    mu0 = np.asarray(mu0, dtype=float)
    d = mu0.shape[0]
    Sigma0 = np.asarray(Sigma0, dtype=float)

    A, Q = iwp1_matrices(h, kappa2, d)
    E0, E1 = build_E0_E1(d)
    R = R_scale * np.eye(d)

    # Initial derivative at mean
    f0 = f(0.0, mu0)
    Jf0 = J_f(0.0, mu0)

    # Linear mapping from theta=y(0) to x0=[y(0);v(0)] ≈ [I; Jf0]theta + const
    G = np.vstack([np.eye(d), Jf0])
    P0 = G @ Sigma0 @ G.T

    # EKF initial state: conditional on y(0)=mu0
    x0_mean = np.concatenate([mu0, f0])
    P_init = 1e-12 * np.eye(2 * d)

    N = int(round(T / h))
    t_grid = np.linspace(0.0, N * h, N + 1)

    m_list = np.zeros((N + 1, 2 * d))
    P_list = np.zeros((N + 1, 2 * d, 2 * d))
    P_goal_list = np.zeros_like(P_list)

    # Jacobian of m_k wrt x0
    Jx = np.eye(2 * d)

    # t=0
    m = x0_mean.copy()
    P = P_init.copy()
    m_list[0] = m
    P_list[0] = P
    P_goal_list[0] = P + Jx @ P0 @ Jx.T

    for k in range(1, N + 1):
        t = t_grid[k]

        # Prediction
        m_pred = A @ m
        P_pred = A @ P @ A.T + Q
        J_pred = A @ Jx

        # ODE "measurement": h(x)=v - f(y,t) = 0
        y_pred = E0 @ m_pred
        f_val = f(t, y_pred)
        Jf = J_f(t, y_pred)

        h_pred = E1 @ m_pred - f_val
        H = np.hstack([-Jf, np.eye(d)])

        S = H @ P_pred @ H.T + R
        K = np.linalg.solve(S, (P_pred @ H.T).T).T

        # Update
        m = m_pred + K @ (-h_pred)
        P = P_pred - K @ S @ K.T

        # Jacobian update
        I2d = np.eye(2 * d)
        Jx = (I2d - K @ H) @ J_pred

        # Store
        m_list[k] = m
        P_list[k] = P
        P_goal_list[k] = P + Jx @ P0 @ Jx.T

    return t_grid, m_list, P_list, P_goal_list


def extract_y_stats_from_P(m_list, P_list, d):
    """
    Project state statistics down to y(t) only.
    """
    E0, _ = build_E0_E1(d)
    Np1 = m_list.shape[0]
    y_mean = np.zeros((Np1, d))
    y_var = np.zeros((Np1, d))
    for k in range(Np1):
        m = m_list[k]
        P = P_list[k]
        y_mean[k] = (E0 @ m).reshape(-1)
        P_y = E0 @ P @ E0.T
        y_var[k] = np.diag(P_y)
    return y_mean, y_var


def propagate_pn_iwp1_goal(system, t_span, t_eval, theta_mean, theta_cov, kappa2=1.0, R_scale=1e-6):
    """
    Wrapper to run the IWP(1)+EK1+Jacobian goal-variance method
    for systems where theta is the uncertain initial state y(0).

    For logistic (parameter uncertainty), this method is not applicable:
    returns None.
    """
    name = system["name"]
    dim_y = system["dim_y"]

    # Only apply to problems where theta is y(0) (FHN, LV, VdP)
    if name == "FitzHugh–Nagumo":
        params = (0.0, 0.08, 0.07, 1.25)

        def f(t, y):
            return fhn_fun(t, y, *params)

        def J_f(t, y):
            return J_fhn(t, y, *params)

    elif name == "Lotka–Volterra":
        params = (5.0, 0.5, 5.0, 0.5)

        def f(t, y):
            return lotkavolterra_fun(t, y, *params)

        def J_f(t, y):
            return J_lv(t, y, *params)

    elif name == "Van der Pol":
        mu = 0.05

        def f(t, y):
            return vanderpol_fun(t, y, mu)

        def J_f(t, y):
            return J_vdp(t, y, mu)

    else:
        # Logistic: uncertainty in parameter, not initial state -> skip
        return None

    mu0 = theta_mean
    Sigma0 = theta_cov
    d = dim_y

    # Use same grid as t_eval (assumed uniform)
    T0, T1 = t_span
    assert T0 == 0.0, "EK1 assumes t0=0.0"
    N = len(t_eval) - 1
    T = T1 - T0
    h = T / N

    t0 = time.perf_counter()
    t_grid, m_list, P_list, P_goal_list = ek1_iwp1_goal_cov(
        f=f,
        J_f=J_f,
        mu0=mu0,
        Sigma0=Sigma0,
        T=T,
        h=h,
        kappa2=kappa2,
        R_scale=R_scale,
    )
    t1 = time.perf_counter()

    # Project
    y_mean_pn, y_var_pn = extract_y_stats_from_P(m_list, P_list, d=d)
    y_mean_goal, y_var_goal = extract_y_stats_from_P(m_list, P_goal_list, d=d)

    return {
        "t": t_grid,
        "mean_pn": y_mean_pn.T,
        "std_pn": np.sqrt(y_var_pn).T,
        "mean_goal": y_mean_goal.T,
        "std_goal": np.sqrt(y_var_goal).T,
        "time": t1 - t0,
        "method": "pn_iwp1_goal",
        "name": name,
    }


def make_logistic_problem():
    a = 3.0
    y0_fixed = np.array([0.05])

    def theta_to_setup(theta):
        # theta is scalar b
        b = float(theta[0])
        return y0_fixed.copy(), (a, b)

    return {
        "name": "Logistic",
        "ode_fun": lambda t, y, a, b: logistic_fun(t, y, a, b),
        "jac_fun": lambda t, y, a, b: logistic_jacobian(t, y, a, b),
        "theta_to_setup": theta_to_setup,
        "dim_y": 1,
        "dim_theta": 1,
    }


def make_fhn_problem():
    params = (0.0, 0.08, 0.07, 1.25)

    def theta_to_setup(theta):
        # theta is y0 (2,)
        y0 = theta.astype(float)
        return y0, params

    return {
        "name": "FitzHugh–Nagumo",
        "ode_fun": lambda t, y, a, b, c, d: fhn_fun(t, y, a, b, c, d),
        "jac_fun": lambda t, y, a, b, c, d: J_fhn(t, y, a, b, c, d),
        "theta_to_setup": theta_to_setup,
        "dim_y": 2,
        "dim_theta": 2,
    }


def make_lv_problem():
    params = (5.0, 0.5, 5.0, 0.5)

    def theta_to_setup(theta):
        y0 = theta.astype(float)
        return y0, params

    return {
        "name": "Lotka–Volterra",
        "ode_fun": lambda t, y, a, b, c, d: lotkavolterra_fun(t, y, a, b, c, d),
        "jac_fun": lambda t, y, a, b, c, d: J_lv(t, y, a, b, c, d),
        "theta_to_setup": theta_to_setup,
        "dim_y": 2,
        "dim_theta": 2,
    }


def make_vdp_problem():
    mu = 0.05

    def theta_to_setup(theta):
        y0 = theta.astype(float)
        return y0, (mu,)

    return {
        "name": "Van der Pol",
        "ode_fun": lambda t, y, mu: vanderpol_fun(t, y, mu),
        "jac_fun": lambda t, y, mu: J_vdp(t, y, mu),
        "theta_to_setup": theta_to_setup,
        "dim_y": 2,
        "dim_theta": 2,
    }


def plot_ci_compare(
    t,
    mean_sp,
    std_sp,
    mean_gh,
    std_gh,
    mean_mc,
    std_mc,
    title,
    ylabel="y",
    fname="plot.png",
    mean_pn=None,
    std_pn=None,
    mean_goal=None,
    std_goal=None,
):
    fig = plt.figure()

    # Spherical
    plt.plot(t, mean_sp, label="Spherical mean")
    plt.fill_between(t, mean_sp - 1.96 * std_sp, mean_sp + 1.96 * std_sp, alpha=0.25, label="Spherical 95% CI")

    # Gauss–Hermite
    plt.plot(t, mean_gh, linestyle="--", label="Gauss–Hermite mean")
    plt.fill_between(t, mean_gh - 1.96 * std_gh, mean_gh + 1.96 * std_gh, alpha=0.25, label="GH 95% CI")

    # Monte Carlo
    plt.plot(t, mean_mc, linestyle=":", label="MC mean")
    plt.fill_between(t, mean_mc - 1.96 * std_mc, mean_mc + 1.96 * std_mc, alpha=0.2, label="MC 95% CI")

    # PN-only (EK1 variance only)
    if mean_pn is not None and std_pn is not None:
        plt.plot(t, mean_pn, linestyle="-.", color="tab:red", label="PN mean (EK1)")
        plt.fill_between(t, mean_pn - 1.96 * std_pn, mean_pn + 1.96 * std_pn, alpha=0.2, color="tab:red", label="PN 95% CI")

    # PN+Jac goal variance
    if mean_goal is not None and std_goal is not None:
        plt.plot(t, mean_goal, linestyle="-", color="tab:green", label="Goal mean (PN+Jac)")
        plt.fill_between(
            t,
            mean_goal - 1.96 * std_goal,
            mean_goal + 1.96 * std_goal,
            alpha=0.2,
            color="tab:green",
            label="Goal 95% CI (PN+Jac)",
        )

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    path = os.path.join("data", fname)
    fig.savefig(path, dpi=150)
    plt.show()
    return path


# ============================================================
# Propagating_Model_probnum_package.py functions (suffix _probnum)
# ============================================================


def spherical_cubature_probnum(mu, Sigma):
    return spherical_cubature(mu, Sigma)


def gauss_hermite_cubature_probnum(mu, Sigma, n_points_1d=5):
    return gauss_hermite_cubature(mu, Sigma, n_points_1d=n_points_1d)


def logistic_fun_probnum(t, y, a, b):
    return logistic_fun(t, y, a, b)


def logistic_jacobian_probnum(t, y, a, b):
    return logistic_jacobian(t, y, a, b)


def fhn_fun_probnum(t, y, a, b, c, d):
    return fhn_fun(t, y, a, b, c, d)


def lotkavolterra_fun_probnum(t, y, a, b, c, d):
    return lotkavolterra_fun(t, y, a, b, c, d)


def vanderpol_fun_probnum(t, y, mu):
    return vanderpol_fun(t, y, mu)


def J_fhn_probnum(t, y, a=0.0, b=0.08, c=0.07, d=1.25):
    return J_fhn(t, y, a=a, b=b, c=c, d=d)


def J_lv_probnum(t, y, a=5.0, b=0.5, c=5.0, d=0.5):
    return J_lv(t, y, a=a, b=b, c=c, d=d)


def J_vdp_probnum(t, y, mu=0.05):
    return J_vdp(t, y, mu=mu)


def _numerical_jacobian_probnum(fun, t, y, eps=1e-6):
    return _numerical_jacobian(fun, t, y, eps=eps)


def integrate_deterministic_probnum(fun, t_span, y0, args=(), t_eval=None, rtol=1e-6, atol=1e-8):
    return integrate_deterministic(fun, t_span, y0, args=args, t_eval=t_eval, rtol=rtol, atol=atol)


def integrate_probnum_probnum(
    fun,
    jac_fun,
    t_span,
    y0,
    args=(),
    t_eval=None,
    method="EK1",
    algo_order=1,
    adaptive=False,
    step=None,
    diffusion_model="constant",
):
    """
    Wrapper around probnum.diffeq.probsolve_ivp using an EK1 ODE filter
    with a once-integrated Wiener process prior (IWP(1)).
    """
    from probnum.diffeq import probsolve_ivp

    if t_eval is None:
        raise ValueError("t_eval must be provided for probabilistic integration")

    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1:
        raise ValueError("t_eval must be a 1D array")

    t0, t1 = float(t_span[0]), float(t_span[1])

    # If we want a fixed-step solver but no explicit step is given,
    # use the spacing of t_eval as a default.
    if not adaptive and step is None:
        if t_eval.size > 1:
            step = float(t_eval[1] - t_eval[0])
        else:
            step = float(t1 - t0)

    def f(t, y):
        # ProbNum passes y as 1D ndarray; append args for our RHS
        return fun(t, np.asarray(y), *args)

    def df(t, y):
        if jac_fun is None:
            return _numerical_jacobian_probnum(lambda _t, _y: fun(_t, _y, *args), t, np.asarray(y))
        return jac_fun(t, np.asarray(y), *args)

    odesol = probsolve_ivp(
        f=f,
        t0=t0,
        tmax=t1,
        y0=y0,
        df=df,
        method=method,
        algo_order=algo_order,
        adaptive=adaptive,
        step=step,
        time_stops=t_eval,
        dense_output=True,
        diffusion_model=diffusion_model,
    )

    rv_eval = odesol(t_eval)
    y_mean = np.asarray(rv_eval.mean)
    y_std = np.asarray(rv_eval.std)

    # Transpose to match shape (d, M) used elsewhere in this script
    return t_eval, y_mean.T, y_std.T


def propagate_deterministic_probnum(
    system,
    t_span,
    t_eval,
    theta_mean,
    theta_cov,
    quad_method="spherical",
    n_gh_1d=5,
    solver="probnum",
    pn_kwargs=None,
):
    """
    Propagate uncertainty using a deterministic quadrature rule
    (spherical cubature or Gauss–Hermite) over theta.

    If solver == "probnum", each theta-node is solved with ProbNum's
    EK1 + IWP(1) ODE filter, as in Yao et al. (2025).
    If solver == "deterministic", we fall back to SciPy's LSODA.
    """
    name = system["name"]
    ode_fun = system["ode_fun"]
    jac_fun = system.get("jac_fun")
    theta_to_setup = system["theta_to_setup"]

    t0 = time.perf_counter()

    if quad_method == "spherical":
        nodes, w = spherical_cubature_probnum(theta_mean, theta_cov)
    elif quad_method == "gh":
        nodes, w = gauss_hermite_cubature_probnum(theta_mean, theta_cov, n_points_1d=n_gh_1d)
    else:
        raise ValueError(f"Unknown quad_method: {quad_method}")

    Y_nodes = []
    t_out = None

    for th in nodes:
        y0, params = theta_to_setup(th)

        if solver == "probnum":
            kwargs = pn_kwargs or {}
            t, y_mean, y_std = integrate_probnum_probnum(ode_fun, jac_fun, t_span, y0, args=params, t_eval=t_eval, **kwargs)
            y = y_mean
            y_var = y_std**2
        elif solver == "deterministic":
            t, y_det = integrate_deterministic_probnum(ode_fun, t_span, y0, args=params, t_eval=t_eval)
            y = y_det
            y_var = np.zeros_like(y_det)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if t_out is None:
            t_out = t
        Y_nodes.append((y, y_var))

    # Weighted mean and variance (including solver covariance)
    Y_means = np.stack([p[0] for p in Y_nodes], axis=0)
    Y_vars = np.stack([p[1] for p in Y_nodes], axis=0)

    mean = np.tensordot(w, Y_means, axes=(0, 0))
    diffs = Y_means - mean[None, :, :]
    var = np.tensordot(w, Y_vars + diffs**2, axes=(0, 0))
    std = np.sqrt(var)

    t1 = time.perf_counter()
    return {
        "t": t_out,
        "mean": mean,
        "std": std,
        "time": t1 - t0,
        "method": quad_method,
        "name": name,
    }


def propagate_mc_probnum(system, t_span, t_eval, theta_mean, theta_cov, n_mc=400, solver="deterministic"):
    """
    Monte Carlo reference propagation.

    As in Yao et al., this uses a classic ODE solver (LSODA) for all samples.
    """
    name = system["name"]
    ode_fun = system["ode_fun"]
    jac_fun = system.get("jac_fun")
    theta_to_setup = system["theta_to_setup"]

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    Y_mc = []
    Y_vars = []
    t_out = None
    for _ in range(n_mc):
        theta = rng.multivariate_normal(theta_mean, theta_cov)
        y0, params = theta_to_setup(theta)
        if solver == "probnum":
            t, y_mean, y_std = integrate_probnum_probnum(ode_fun, jac_fun, t_span, y0, args=params, t_eval=t_eval)
            y_var = y_std**2
            y = y_mean
        elif solver == "deterministic":
            t, y = integrate_deterministic_probnum(ode_fun, t_span, y0, args=params, t_eval=t_eval)
            y_var = np.zeros_like(y)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if t_out is None:
            t_out = t
        Y_mc.append(y)
        Y_vars.append(y_var)

    Y_mc = np.stack(Y_mc, axis=0)
    Y_vars = np.stack(Y_vars, axis=0)
    mc_mean = np.mean(Y_mc, axis=0)
    mc_var = np.mean(Y_vars, axis=0) + np.var(Y_mc, axis=0, ddof=1)
    mc_std = np.sqrt(mc_var)

    t1 = time.perf_counter()
    return {
        "t": t_out,
        "mean": mc_mean,
        "std": mc_std,
        "time": t1 - t0,
        "method": "mc",
        "name": name,
    }


def iwp1_matrices_probnum(h, kappa2, d):
    return iwp1_matrices(h, kappa2, d)


def build_E0_E1_probnum(d):
    return build_E0_E1(d)


def ek1_iwp1_goal_cov_probnum(f, J_f, mu0, Sigma0, T, h, kappa2=1.0, R_scale=1e-6):
    return ek1_iwp1_goal_cov(f, J_f, mu0, Sigma0, T, h, kappa2=kappa2, R_scale=R_scale)


def extract_y_stats_from_P_probnum(m_list, P_list, d):
    return extract_y_stats_from_P(m_list, P_list, d)


def propagate_pn_iwp1_goal_probnum(system, t_span, t_eval, theta_mean, theta_cov, kappa2=1.0, R_scale=1e-6):
    """
    Wrapper to run the IWP(1)+EK1+Jacobian goal-variance method
    for systems where theta is the uncertain initial state y(0).
    """
    name = system["name"]
    dim_y = system["dim_y"]

    # Only apply to problems where theta is y(0) (FHN, LV, VdP)
    if name == "FitzHugh–Nagumo":
        params = (0.0, 0.08, 0.07, 1.25)

        def f(t, y):
            return fhn_fun_probnum(t, y, *params)

        def J_f(t, y):
            return J_fhn_probnum(t, y, *params)

    elif name == "Lotka–Volterra":
        params = (5.0, 0.5, 5.0, 0.5)

        def f(t, y):
            return lotkavolterra_fun_probnum(t, y, *params)

        def J_f(t, y):
            return J_lv_probnum(t, y, *params)

    elif name == "Van der Pol":
        mu = 0.05

        def f(t, y):
            return vanderpol_fun_probnum(t, y, mu)

        def J_f(t, y):
            return J_vdp_probnum(t, y, mu)

    else:
        return None

    mu0 = theta_mean
    Sigma0 = theta_cov
    d = dim_y

    # Use same grid as t_eval (assumed uniform)
    T0, T1 = t_span
    assert T0 == 0.0
    N = len(t_eval) - 1
    T = T1 - T0
    h = T / N

    t0 = time.perf_counter()
    t_grid, m_list, P_list, P_goal_list = ek1_iwp1_goal_cov_probnum(
        f=f,
        J_f=J_f,
        mu0=mu0,
        Sigma0=Sigma0,
        T=T,
        h=h,
        kappa2=kappa2,
        R_scale=R_scale,
    )
    t1 = time.perf_counter()

    # Project
    y_mean_pn, y_var_pn = extract_y_stats_from_P_probnum(m_list, P_list, d=d)
    y_mean_goal, y_var_goal = extract_y_stats_from_P_probnum(m_list, P_goal_list, d=d)

    return {
        "t": t_grid,
        "mean_pn": y_mean_pn.T,
        "std_pn": np.sqrt(y_var_pn).T,
        "mean_goal": y_mean_goal.T,
        "std_goal": np.sqrt(y_var_goal).T,
        "time": t1 - t0,
        "method": "pn_iwp1_goal",
        "name": name,
    }


def make_logistic_problem_probnum():
    return make_logistic_problem()


def make_fhn_problem_probnum():
    return make_fhn_problem()


def make_lv_problem_probnum():
    return make_lv_problem()


def make_vdp_problem_probnum():
    return make_vdp_problem()


def plot_ci_compare_probnum(
    t,
    mean_sp,
    std_sp,
    mean_gh,
    std_gh,
    mean_mc,
    std_mc,
    title,
    ylabel="y",
    fname="plot.png",
    mean_pn=None,
    std_pn=None,
    mean_goal=None,
    std_goal=None,
):
    return plot_ci_compare(
        t,
        mean_sp,
        std_sp,
        mean_gh,
        std_gh,
        mean_mc,
        std_mc,
        title,
        ylabel=ylabel,
        fname=fname,
        mean_pn=mean_pn,
        std_pn=std_pn,
        mean_goal=mean_goal,
        std_goal=std_goal,
    )


__all__ = [
    "gaussian_w2_distance",
    "analytic_joint_gaussian",
    "solve_single_theta_lsoda",
    "mc_lsoda",
    "pn_kalman_path",
    "pn_kalman_state_with_sensitivities",
    "pn_joint_gaussian",
    "incremental_mc_lsoda_until_tol",
    "time_call",
    "OU_model",
    "Simulate_Data",
    "OU_kernel",
    "chol_solve",
    "dual_smoother",
    "dual_filter",
    "spherical_cubature",
    "gauss_hermite_cubature",
    "logistic_fun",
    "logistic_jacobian",
    "fhn_fun",
    "lotkavolterra_fun",
    "vanderpol_fun",
    "J_fhn",
    "J_lv",
    "J_vdp",
    "_numerical_jacobian",
    "integrate_deterministic",
    "integrate_probnum",
    "propagate_deterministic",
    "propagate_mc",
    "iwp1_matrices",
    "build_E0_E1",
    "ek1_iwp1_goal_cov",
    "extract_y_stats_from_P",
    "propagate_pn_iwp1_goal",
    "make_logistic_problem",
    "make_fhn_problem",
    "make_lv_problem",
    "make_vdp_problem",
    "plot_ci_compare",
    "spherical_cubature_probnum",
    "gauss_hermite_cubature_probnum",
    "logistic_fun_probnum",
    "logistic_jacobian_probnum",
    "fhn_fun_probnum",
    "lotkavolterra_fun_probnum",
    "vanderpol_fun_probnum",
    "J_fhn_probnum",
    "J_lv_probnum",
    "J_vdp_probnum",
    "_numerical_jacobian_probnum",
    "integrate_deterministic_probnum",
    "integrate_probnum_probnum",
    "propagate_deterministic_probnum",
    "propagate_mc_probnum",
    "iwp1_matrices_probnum",
    "build_E0_E1_probnum",
    "ek1_iwp1_goal_cov_probnum",
    "extract_y_stats_from_P_probnum",
    "propagate_pn_iwp1_goal_probnum",
    "make_logistic_problem_probnum",
    "make_fhn_problem_probnum",
    "make_lv_problem_probnum",
    "make_vdp_problem_probnum",
    "plot_ci_compare_probnum",
]
