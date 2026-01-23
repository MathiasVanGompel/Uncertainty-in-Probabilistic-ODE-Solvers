# probabilistic_uncertainty_propagation_pgoal.py
#
# Uncertainty propagation over ODE solutions with:
#   - Deterministic solvers + spherical / Gauss–Hermite quadrature
#   - Monte Carlo reference
#   - Filtering-based probabilistic ODE solver (IWP(1) ODE filter)
#   - p_goal: mixture of PN solutions over quadrature nodes (fixed weights)
#   - Approximate parameter filter/smoother: reweight nodes by data likelihood
#
# Dependencies: numpy, scipy, matplotlib

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.polynomial.hermite import hermgauss
import os
import math
import warnings
import time

# Reproducibility
rng = np.random.default_rng(0)

# ======================
# Plotting configuration
# ======================
# "all"      -> plot MC, deterministic priors, PN-goal, param-filter, param-smoother + observations
# "pn_vs_mc" -> plot only one PN-like method (PN_METHOD) against MC reference + observations
PLOT_MODE = "pn_vs_mc"   # "all" or "pn_vs_mc"

# Which PN curve to use when PLOT_MODE == "pn_vs_mc"
# Options:
#   "pn_goal"        -> p_goal from PN ODE filter (GH quadrature)
#   "param_filter"   -> parameter filter (PN nodes, posterior θ-weights)
#   "param_smoother" -> parameter smoother (PN nodes, posterior θ-weights)
#   "gh_quadrature"       -> deterministic GH quadrature predictive
#   "spherical_quadrature"-> deterministic spherical quadrature predictive
PN_METHOD = "spherical_quadrature"

# ===========================
# Quadrature / cubature rules
# ===========================

def spherical_cubature(mu, Sigma):
    """
    Spherical cubature algorithm for N(mu, Sigma).
    """
    mu = np.atleast_1d(mu)
    d = mu.shape[0]
    # numerical stability
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
    w = np.full(2*d, 1.0/(2*d))
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
    x_1d, w_1d = hermgauss(n_points_1d)  # shape (n_points_1d,)

    # Build d-dimensional tensor-product grid
    grids = np.meshgrid(*([x_1d] * d), indexing='ij')
    u_grid = np.stack(grids, axis=-1).reshape(-1, d)  # (K, d)

    # Corresponding product weights
    w_grids = np.meshgrid(*([w_1d] * d), indexing='ij')
    w_prod = np.prod(np.stack(w_grids, axis=-1), axis=-1).reshape(-1)  # (K,)

    # Convert to standard normal Z ~ N(0, I):
    z_nodes = np.sqrt(2.0) * u_grid  # (K, d)

    # Weights for expectation w.r.t. Z ~ N(0, I)
    w = (1.0 / (np.pi ** (d / 2.0))) * w_prod  # (K,)

    # Transform Z to theta = mu + L Z for N(mu, Sigma)
    nodes = mu + z_nodes @ L.T  # (K, d)

    return nodes, w


# =======================
# Deterministic ODE setup
# =======================

def logistic_fun(t, y, a, b):
    # y' = a y (1 - y/b)
    return a * y * (1.0 - y / b)

def fhn_fun(t, y, a, b, c, d):
    # FitzHugh–Nagumo in (y1, y2)
    y1, y2 = y
    dy1 = y1 - (y1**3)/3.0 - y2 + a
    dy2 = (y1 + b - c*y2)/d
    return np.array([dy1, dy2])

def lotkavolterra_fun(t, y, a, b, c, d):
    # [ y1' = a*y1 - b*y1*y2,  y2' = -c*y2 + d*y1*y2 ]
    y1, y2 = y
    return np.array([a*y1 - b*y1*y2, -c*y2 + d*y1*y2])

def vanderpol_fun(t, y, mu):
    # y1' = y2
    # y2' = mu * (1 - y1**2) * y2 - y1
    y1, y2 = y
    return np.array([y2, mu*(1 - y1**2)*y2 - y1])

# Wrapper to integrate with LSODA
def integrate(fun, t_span, y0, args=(), t_eval=None, rtol=1e-6, atol=1e-8):
    sol = solve_ivp(fun, t_span, y0, method='LSODA', args=args, t_eval=t_eval,
                    rtol=rtol, atol=atol)
    if not sol.success:
        warnings.warn(f"Integration failed: {sol.message}")
    return sol.t, sol.y  # t shape (M,), y shape (dim_y, M)


# ==========================================
# Deterministic + Monte Carlo uncertainty
# ==========================================

def propagate_deterministic(system, t_span, t_eval, theta_mean, theta_cov,
                            quad_method="spherical", n_gh_1d=5):
    """
    Propagate uncertainty using a deterministic quadrature rule and
    a deterministic ODE solver. This matches the *non-PN* baseline
    in Yao et al.: uncertainty only from θ / initial conditions.
    """
    name = system['name']
    ode_fun = system['ode_fun']
    theta_to_setup = system['theta_to_setup']

    t0 = time.perf_counter()

    if quad_method == "spherical":
        nodes, w = spherical_cubature(theta_mean, theta_cov)
    elif quad_method == "gh":
        nodes, w = gauss_hermite_cubature(theta_mean, theta_cov,
                                          n_points_1d=n_gh_1d)
    else:
        raise ValueError(f"Unknown quad_method: {quad_method}")

    # Integrate from each node
    Y_nodes = []  # list of arrays (dim_y, M)
    t_out = None
    for th in nodes:
        y0, params = theta_to_setup(th)
        t, y = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
        if t_out is None:
            t_out = t
        Y_nodes.append(y)
    Y_nodes = np.stack(Y_nodes, axis=0)  # (K, dim_y, M)

    # Weighted mean and variance over nodes (prior predictive)
    mean = np.tensordot(w, Y_nodes, axes=(0, 0))  # (dim_y, M)
    diffs = Y_nodes - mean[None, :, :]
    var = np.tensordot(w, diffs**2, axes=(0, 0))  # (dim_y, M)
    std = np.sqrt(np.maximum(var, 0.0))

    t1 = time.perf_counter()
    return {
        't': t_out,
        'mean': mean,
        'std': std,
        'time': t1 - t0,
        'method': quad_method,
        'name': name,
        'nodes': nodes,
        'w': w,
        'Y_nodes': Y_nodes
    }


def propagate_mc(system, t_span, t_eval, theta_mean, theta_cov, n_mc=400):
    """
    Monte Carlo reference propagation (non-PN).
    """
    name = system['name']
    ode_fun = system['ode_fun']
    theta_to_setup = system['theta_to_setup']

    t0 = time.perf_counter()

    Y_mc = []
    t_out = None
    for _ in range(n_mc):
        theta = rng.multivariate_normal(theta_mean, theta_cov)
        y0, params = theta_to_setup(theta)
        t, y = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
        if t_out is None:
            t_out = t
        Y_mc.append(y)
    Y_mc = np.stack(Y_mc, axis=0)  # (n_mc, dim_y, M)
    mc_mean = np.mean(Y_mc, axis=0)
    mc_std = np.std(Y_mc, axis=0, ddof=1)

    t1 = time.perf_counter()
    return {
        't': t_out,
        'mean': mc_mean,
        'std': mc_std,
        'time': t1 - t0,
        'method': 'mc',
        'name': name
    }


# ============================================
# NEW: Filtering-based PN ODE solver (IWP(1))
# ============================================

def jacobian_f_y(ode_fun, t, y, params, eps=1e-6):
    """
    Numerical Jacobian ∂f/∂y evaluated at (t, y, params).
    """
    y = np.asarray(y, dtype=float)
    d = y.size
    f0 = ode_fun(t, y, *params)
    f0 = np.asarray(f0, dtype=float)
    J = np.zeros((d, d), dtype=float)
    for j in range(d):
        y_pert = y.copy()
        y_pert[j] += eps
        f1 = ode_fun(t, y_pert, *params)
        f1 = np.asarray(f1, dtype=float)
        J[:, j] = (f1 - f0) / eps
    return J


def iwp1_matrices(dt, dim_y, diffusion=1.0):
    """
    State transition A(h) and process noise covariance Q(h) for an
    integrated Wiener process of order 1 (IWP(1)) over step dt.

    For each component:
      [y_{k+1}]   [1 h] [y_k]   + q_c * noise
      [v_{k+1}] = [0 1] [v_k]

    Q_block = q_c * [[h^3/3, h^2/2], [h^2/2, h]]
    Then A = A_block ⊗ I_d, Q = Q_block ⊗ I_d.
    """
    h = float(dt)
    A_block = np.array([[1.0, h],
                        [0.0, 1.0]])
    Q_block = diffusion * np.array([[h**3 / 3.0, h**2 / 2.0],
                                    [h**2 / 2.0, h]])

    I_d = np.eye(dim_y)
    A = np.kron(A_block, I_d)  # (2d, 2d)
    Q = np.kron(Q_block, I_d)  # (2d, 2d)
    return A, Q


def ode_filter_iwp1(ode_fun, t_eval, y0, params,
                    diffusion=1.0, meas_var=1e-6):
    """
    Basic IWP(1)-based ODE filter (extended Kalman filter):

    State: x = [y; v] ∈ R^{2d}, with d = dim_y.
    Prior: IWP(1) with diffusion "diffusion".
    Measurement at each time step (after prediction):

        r_k = v(t_k) - f(y(t_k), t_k, θ) ≈ 0

    implemented as a nonlinear measurement model r_k(x_k) with mean zero
    and covariance R = meas_var * I.

    This yields a probabilistic numerical solution p(x(t_k) | "ODE data"),
    from which we extract p(y(t_k) | ...).
    """
    t_eval = np.asarray(t_eval, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    dim_y = y0.size
    M = t_eval.size

    # Initial mean and covariance
    f0 = ode_fun(t_eval[0], y0, *params)
    f0 = np.asarray(f0, dtype=float)
    x_mean = np.concatenate([y0, f0])         # (2d,)
    x_cov = 1e-8 * np.eye(2 * dim_y)          # almost deterministic initially

    mean_y = np.zeros((dim_y, M))
    cov_y = np.zeros((dim_y, dim_y, M))
    mean_y[:, 0] = y0
    cov_y[:, :, 0] = x_cov[:dim_y, :dim_y]

    for k in range(M - 1):
        dt = t_eval[k + 1] - t_eval[k]
        A, Q = iwp1_matrices(dt, dim_y, diffusion=diffusion)

        # Predict
        x_mean = A @ x_mean
        x_cov = A @ x_cov @ A.T + Q

        # Nonlinear measurement at t_{k+1}: r = v - f(y, t_{k+1}) ≈ 0
        y_pred = x_mean[:dim_y]
        v_pred = x_mean[dim_y:]
        f_pred = ode_fun(t_eval[k + 1], y_pred, *params)
        f_pred = np.asarray(f_pred, dtype=float)

        r = v_pred - f_pred  # residual (dim_y,)

        # Linearise measurement model: r(x) = v - f(y)
        # => ∂r/∂y = -∂f/∂y,  ∂r/∂v = I
        J_f = jacobian_f_y(ode_fun, t_eval[k + 1], y_pred, params)
        H = np.zeros((dim_y, 2 * dim_y))
        H[:, :dim_y] = -J_f
        H[:, dim_y:] = np.eye(dim_y)

        R = meas_var * np.eye(dim_y)
        S = H @ x_cov @ H.T + R
        K = x_cov @ H.T @ np.linalg.inv(S)

        # We "observe" r_obs = 0, so innovation = 0 - r
        x_mean = x_mean - K @ r
        x_cov = x_cov - K @ S @ K.T
        x_cov = 0.5 * (x_cov + x_cov.T)  # symmetrise

        mean_y[:, k + 1] = x_mean[:dim_y]
        cov_y[:, :, k + 1] = x_cov[:dim_y, :dim_y]

    return mean_y, cov_y


def propagate_pn_goal(system, t_eval, theta_mean, theta_cov,
                      quad_method="gh", n_gh_1d=5,
                      diffusion=1.0, meas_var=1e-6):
    """
    Propagate parameter / initial-condition uncertainty via:
      - Filtering-based PN ODE solver (IWP(1) ODE filter) at each quadrature node.
      - p_goal: mixture of these PN solutions over θ, using *fixed* quadrature weights.

    This is the p_goal( x(t) | D_PN ) in your notation: pushforward of p(θ)
    through the PN ODE solver. No inference on θ here.
    """
    name = system['name']
    ode_fun = system['ode_fun']
    theta_to_setup = system['theta_to_setup']
    t_eval = np.asarray(t_eval, dtype=float)

    # Quadrature over θ
    if quad_method == "spherical":
        nodes, w = spherical_cubature(theta_mean, theta_cov)
    elif quad_method == "gh":
        nodes, w = gauss_hermite_cubature(theta_mean, theta_cov,
                                          n_points_1d=n_gh_1d)
    else:
        raise ValueError(f"Unknown quad_method: {quad_method}")

    K = nodes.shape[0]
    dim_y = system['dim_y']
    M = t_eval.size

    Y_mean_nodes = np.zeros((K, dim_y, M))
    Y_cov_nodes = np.zeros((K, dim_y, dim_y, M))

    t0 = time.perf_counter()
    for i, theta in enumerate(nodes):
        y0, params = theta_to_setup(theta)
        mean_y, cov_y = ode_filter_iwp1(ode_fun, t_eval, y0, params,
                                        diffusion=diffusion,
                                        meas_var=meas_var)
        Y_mean_nodes[i] = mean_y
        Y_cov_nodes[i] = cov_y
    t1 = time.perf_counter()

    # Mixture mean E[x | p_goal]
    mean_goal = np.tensordot(w, Y_mean_nodes, axes=(0, 0))  # (dim_y, M)

    # Law of total variance under the mixture:
    # Var[x] = E_theta[Cov[x | θ]] + Var_theta(E[x | θ])
    EY_outer = np.zeros((dim_y, dim_y, M))
    for k in range(K):
        mu_k = Y_mean_nodes[k]   # (dim_y, M)
        cov_k = Y_cov_nodes[k]   # (dim_y, dim_y, M)
        EY_outer += w[k] * (cov_k + np.einsum("im,jm->ijm", mu_k, mu_k))

    diag_EY = EY_outer.diagonal(axis1=0, axis2=1)  # (M, dim_y)
    var_goal = diag_EY.T - mean_goal**2            # (dim_y, M)
    std_goal = np.sqrt(np.maximum(var_goal, 0.0))

    return {
        't': t_eval,
        'mean': mean_goal,
        'std': std_goal,
        'time': t1 - t0,
        'method': f'pn_goal_{quad_method}',
        'name': name,
        'nodes': nodes,
        'w': w,
        'Y_mean_nodes': Y_mean_nodes,
        'Y_cov_nodes': Y_cov_nodes
    }


def compute_param_filter_smoother_from_pn(res_pn, y_obs, obs_std):
    """
    Approximate parameter filter / smoother on top of PN solutions.

    Model:
      θ ~ prior p(θ) (quadrature weights)
      x(t_k) | θ  ~ N( μ_{i,k}, Σ_{i,k} )   (from PN ODE filter)
      y_k | x_k  ~ N( x_k, R )

    Then:
      p(y_k | θ_i) = ∫ N(y_k | x, R) N(x | μ_i, Σ_i) dx
                   = N(y_k | μ_i, Σ_i + R)

    We use these p(y_k | θ_i) to update θ-weights:

      p(θ_i | y_{1:k}) ∝ w_i * ∏_{j≤k} p(y_j | θ_i)

    and approximate:

      p(x_k | y_{1:k}) ≈ Σ_i p(θ_i | y_{1:k}) * p(x_k | θ_i),

    where p(x_k | θ_i) remains the PN prior from the ODE filter.
    This is your p_filter-type quantity, but with PN uncertainty included.
    """
    Y_mean = np.asarray(res_pn['Y_mean_nodes'])   # (K, dim_y, M)
    Y_cov = np.asarray(res_pn['Y_cov_nodes'])     # (K, dim_y, dim_y, M)
    w = np.asarray(res_pn['w'])                   # (K,)
    t = res_pn['t']
    name = res_pn['name']
    method = res_pn['method']

    y_obs = np.asarray(y_obs)
    if y_obs.shape != Y_mean.shape[1:]:
        raise ValueError(
            f"y_obs must have shape (dim_y, M) = {Y_mean.shape[1:]}, "
            f"got {y_obs.shape}"
        )

    K, dim_y, M = Y_mean.shape

    obs_std = np.asarray(obs_std, dtype=float)
    if obs_std.ndim == 0:
        obs_std = np.full(dim_y, float(obs_std))
    elif obs_std.shape[0] != dim_y:
        raise ValueError("obs_std must be scalar or have length dim_y")
    obs_var = obs_std**2  # (dim_y,)

    # Prior log-weights
    if np.any(w <= 0):
        raise ValueError("Quadrature weights must be positive to form a posterior.")
    log_w_prior = np.log(w)  # (K,)

    # Log-likelihoods log p(y_k | θ_i)
    loglik = np.zeros((K, M))
    log_two_pi = np.log(2.0 * np.pi)

    for i in range(K):
        mu_i = Y_mean[i]                       # (dim_y, M)
        cov_i = Y_cov[i]                       # (dim_y, dim_y, M)
        var_i = cov_i.diagonal(axis1=0, axis2=1).T  # (dim_y, M)
        total_var = var_i + obs_var[:, None]        # (dim_y, M)
        diff = y_obs - mu_i                          # (dim_y, M)
        ll = -0.5 * (log_two_pi + np.log(total_var) + diff**2 / total_var)
        loglik[i] = ll.sum(axis=0)  # sum over components

    # Filtering: cumulative log-likelihood
    cum_loglik = np.cumsum(loglik, axis=1)         # (K, M)
    logw_post = log_w_prior[:, None] + cum_loglik  # (K, M)

    max_logw = np.max(logw_post, axis=0, keepdims=True)
    w_tilde = np.exp(logw_post - max_logw)
    w_norm = w_tilde / np.sum(w_tilde, axis=0, keepdims=True)  # (K, M)

    # Filter mean and variance as mixture over PN node distributions
    mean_filter = np.sum(w_norm[:, None, :] * Y_mean, axis=0)  # (dim_y, M)

    second_moment_filter = np.zeros((dim_y, M))
    for i in range(K):
        mu_i = Y_mean[i]  # (dim_y, M)
        var_i = Y_cov[i].diagonal(axis1=0, axis2=1).T  # (dim_y, M)
        second_moment_filter += w_norm[i][None, :] * (var_i + mu_i**2)
    var_filter = np.maximum(second_moment_filter - mean_filter**2, 0.0)
    std_filter = np.sqrt(var_filter)

    # Smoother: p(θ | y_{1:T}) using all times at once
    final_logw = log_w_prior + loglik.sum(axis=1)    # (K,)
    max_final = np.max(final_logw)
    w_smooth_unnorm = np.exp(final_logw - max_final)
    w_smooth = w_smooth_unnorm / np.sum(w_smooth_unnorm)  # (K,)

    mean_smooth = np.sum(w_smooth[:, None, None] * Y_mean, axis=0)  # (dim_y, M)

    second_moment_smooth = np.zeros((dim_y, M))
    for i in range(K):
        mu_i = Y_mean[i]
        var_i = Y_cov[i].diagonal(axis1=0, axis2=1).T
        second_moment_smooth += w_smooth[i] * (var_i + mu_i**2)
    var_smooth = np.maximum(second_moment_smooth - mean_smooth**2, 0.0)
    std_smooth = np.sqrt(var_smooth)

    res_filter = {
        't': t,
        'mean': mean_filter,
        'std': std_filter,
        'time': np.nan,
        'method': f'param_filter_{method}',
        'name': name
    }
    res_smoother = {
        't': t,
        'mean': mean_smooth,
        'std': std_smooth,
        'time': np.nan,
        'method': f'param_smoother_{method}',
        'name': name
    }
    return res_filter, res_smoother


# ===============
# ODE problems
# ===============

def make_logistic_problem():
    # 1) Logistic: y' = a*y*(1 - y/b), a fixed, b ~ N(3, 0.01), y0 = 0.05
    a = 3.0
    y0_fixed = np.array([0.05])
    def theta_to_setup(theta):
        # theta is scalar b
        b = float(theta[0])
        return y0_fixed.copy(), (a, b)
    return {
        'name': 'Logistic',
        'ode_fun': lambda t, y, a, b: logistic_fun(t, y, a, b),
        'theta_to_setup': theta_to_setup,
        'dim_y': 1,
        'dim_theta': 1
    }

def make_fhn_problem():
    # 2) FitzHugh–Nagumo: uncertain initial y(0) ~ N([0.5, 1], 0.1 I2);
    #    params fixed (a=0, b=0.08, c=0.07, d=1.25)
    params = (0.0, 0.08, 0.07, 1.25)
    def theta_to_setup(theta):
        # theta is y0 (2,)
        y0 = theta.astype(float)
        return y0, params
    return {
        'name': 'FitzHugh–Nagumo',
        'ode_fun': lambda t, y, a, b, c, d: fhn_fun(t, y, a, b, c, d),
        'theta_to_setup': theta_to_setup,
        'dim_y': 2,
        'dim_theta': 2
    }

def make_lv_problem():
    # 3) Lotka–Volterra: uncertain initial y(0) ~ N([5,5], 0.3 I2);
    #    params fixed (a=5, b=0.5, c=5, d=0.5)
    params = (5.0, 0.5, 5.0, 0.5)
    def theta_to_setup(theta):
        y0 = theta.astype(float)
        return y0, params
    return {
        'name': 'Lotka–Volterra',
        'ode_fun': lambda t, y, a, b, c, d: lotkavolterra_fun(t, y, a, b, c, d),
        'theta_to_setup': theta_to_setup,
        'dim_y': 2,
        'dim_theta': 2
    }

def make_vdp_problem():
    # 4) Van der Pol: uncertain initial y(0) ~ N([5,5], 2 I2); param mu fixed 0.05
    mu = 0.05
    def theta_to_setup(theta):
        y0 = theta.astype(float)
        return y0, (mu,)
    return {
        'name': 'Van der Pol',
        'ode_fun': lambda t, y, mu: vanderpol_fun(t, y, mu),
        'theta_to_setup': theta_to_setup,
        'dim_y': 2,
        'dim_theta': 2
    }


# =====================================
# Set up problems & generate observations
# =====================================

problems = []

# Logistic settings
problems.append({
    'problem': make_logistic_problem(),
    'theta_mean': np.array([3.0]),
    'theta_cov': np.array([[0.01]]),  # variance 0.01
    't_span': (0.0, 3.0),
    't_eval': np.linspace(0.0, 3.0, 400),
    'n_mc': 600,
    'obs_std': np.array([0.03])  # observation noise for y
})

# FHN settings
problems.append({
    'problem': make_fhn_problem(),
    'theta_mean': np.array([0.5, 1.0]),
    'theta_cov': 0.1 * np.eye(2),
    't_span': (0.0, 7.0),
    't_eval': np.linspace(0.0, 7.0, 700),
    'n_mc': 500,
    'obs_std': np.array([0.1, 0.1])
})

# Lotka–Volterra settings
problems.append({
    'problem': make_lv_problem(),
    'theta_mean': np.array([5.0, 5.0]),
    'theta_cov': 0.3 * np.eye(2),
    't_span': (0.0, 2.0),
    't_eval': np.linspace(0.0, 2.0, 400),
    'n_mc': 600,
    'obs_std': np.array([0.3, 0.3])
})

# Van der Pol settings
problems.append({
    'problem': make_vdp_problem(),
    'theta_mean': np.array([5.0, 5.0]),
    'theta_cov': 2.0 * np.eye(2),
    't_span': (0.0, 10.0),
    't_eval': np.linspace(0.0, 10.0, 1200),
    'n_mc': 500,
    'obs_std': np.array([0.5, 0.5])
})

# Generate synthetic observations y_k = x_k + eps_k for each problem,
# using the *prior mean* θ as "truth".
for model in problems:
    problem = model['problem']
    theta_mean = model['theta_mean']
    t_span = model['t_span']
    t_eval = model['t_eval']
    obs_std = model['obs_std']

    ode_fun = problem['ode_fun']
    theta_to_setup = problem['theta_to_setup']

    # "True" θ
    theta_true = theta_mean.copy()
    y0_true, params_true = theta_to_setup(theta_true)
    t_true, x_true = integrate(ode_fun, t_span, y0_true, args=params_true, t_eval=t_eval)

    dim_y = problem['dim_y']
    assert x_true.shape[0] == dim_y

    obs_std_arr = np.asarray(obs_std, dtype=float)
    if obs_std_arr.ndim == 0:
        obs_std_arr = np.full(dim_y, float(obs_std_arr))
    assert obs_std_arr.shape[0] == dim_y

    M = x_true.shape[1]
    noise = rng.normal(loc=0.0, scale=obs_std_arr[:, None], size=(dim_y, M))
    y_obs = x_true + noise

    model['theta_true'] = theta_true
    model['x_true'] = x_true
    model['y_obs'] = y_obs


# ===============
# Run algorithms
# ===============

results = []
timings = []

for model in problems:
    problem = model['problem']
    theta_mean = model['theta_mean']
    theta_cov = model['theta_cov']
    t_span = model['t_span']
    t_eval = model['t_eval']
    n_mc = model['n_mc']
    y_obs = model['y_obs']
    obs_std = model['obs_std']

    # Deterministic spherical/GH quadrature predictive
    res_sp = propagate_deterministic(
        problem, t_span, t_eval, theta_mean, theta_cov,
        quad_method="spherical"
    )
    res_gh = propagate_deterministic(
        problem, t_span, t_eval, theta_mean, theta_cov,
        quad_method="gh", n_gh_1d=5
    )

    # Monte Carlo reference
    res_mc = propagate_mc(
        problem, t_span, t_eval, theta_mean, theta_cov,
        n_mc=n_mc
    )

    # PN ODE filter + p_goal (Gauss–Hermite)  <-- THIS IS THE KEY NEW PIECE
    res_pn_goal = propagate_pn_goal(
        problem, t_eval, theta_mean, theta_cov,
        quad_method="gh", n_gh_1d=5,
        diffusion=1.0, meas_var=1e-6
    )

    # Parameter filter/smoother based on PN nodes + observations
    res_pf, res_ps = compute_param_filter_smoother_from_pn(
        res_pn_goal, y_obs, obs_std
    )

    results.append({
        'name': problem['name'],
        'spherical': res_sp,
        'gh': res_gh,
        'mc': res_mc,
        'pn_goal': res_pn_goal,
        'param_filter': res_pf,
        'param_smoother': res_ps,
        'y_obs': y_obs
    })

    timings.append({
        'name': problem['name'],
        'time_spherical': res_sp['time'],
        'time_gh': res_gh['time'],
        'time_mc': res_mc['time'],
        'time_pn_goal': res_pn_goal['time']
    })

# Print timing comparison
for tm in timings:
    name = tm['name']
    t_sp = tm['time_spherical']
    t_gh = tm['time_gh']
    t_mc = tm['time_mc']
    t_pn = tm['time_pn_goal']
    print(f"{name}:")
    print(f"  Spherical time       = {t_sp:.3f} s")
    print(f"  Gauss–Hermite time   = {t_gh:.3f} s")
    print(f"  MC time              = {t_mc:.3f} s")
    print(f"  PN (p_goal, GH) time = {t_pn:.3f} s")
    print(f"  MC / Spherical = {t_mc / t_sp:.2f}x,  MC / GH = {t_mc / t_gh:.2f}x, MC / PN = {t_mc / t_pn:.2f}x\n")


# =========
# Plotting
# =========

outdir = "data"
os.makedirs(outdir, exist_ok=True)
saved_files = []

y_obs = None
def plot_ci_compare(t,
                    mean_mc, std_mc,
                    mean_sp, std_sp,
                    mean_gh, std_gh,
                    mean_pn_goal, std_pn_goal,
                    mean_pf, std_pf,
                    mean_ps, std_ps,
                    y_obs=None,
                    title="", ylabel="y", fname="plot.png"):
    fig = plt.figure()

    # Observations
    if y_obs is not None and y_obs.ndim == 2:
        plt.scatter(t, y_obs[0], s=10, alpha=0.4, label="Observations")

    # MC
    plt.plot(t, mean_mc, linestyle=":", label="MC mean")
    plt.fill_between(t, mean_mc - 1.96*std_mc, mean_mc + 1.96*std_mc,
                     alpha=0.2, label="MC 95% CI")

    # Deterministic spherical quadrature
    plt.plot(t, mean_sp, label="Spherical quadrature mean")
    plt.fill_between(t, mean_sp - 1.96*std_sp, mean_sp + 1.96*std_sp,
                     alpha=0.2, label="Spherical 95% CI")

    # Deterministic GH quadrature
    plt.plot(t, mean_gh, linestyle="--", label="GH quadrature mean")
    plt.fill_between(t, mean_gh - 1.96*std_gh, mean_gh + 1.96*std_gh,
                     alpha=0.2, label="GH 95% CI")

    # PN p_goal
    plt.plot(t, mean_pn_goal, linestyle="-.", linewidth=2,
             label="PN p_goal mean")
    plt.fill_between(t, mean_pn_goal - 1.96*std_pn_goal,
                     mean_pn_goal + 1.96*std_pn_goal,
                     alpha=0.15, label="PN p_goal 95% CI")

    # Parameter filter
    plt.plot(t, mean_pf, linestyle="-.", label="Param filter mean")
    plt.fill_between(t, mean_pf - 1.96*std_pf, mean_pf + 1.96*std_pf,
                     alpha=0.15, label="Param filter 95% CI")

    # Parameter smoother
    plt.plot(t, mean_ps, linestyle="-.", label="Param smoother mean")
    plt.fill_between(t, mean_ps - 1.96*std_ps, mean_ps + 1.96*std_ps,
                     alpha=0.15, label="Param smoother 95% CI")

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    saved_files.append(path)


def plot_pn_vs_mc(t,
                  mean_mc, std_mc,
                  mean_pn, std_pn,
                  y_obs=None,
                  pn_label="PN",
                  title="", ylabel="y", fname="plot_pn_mc.png"):
    """
    Plot only one PN/quad method vs MC reference (plus observations).
    """
    fig, ax = plt.subplots()
    y_obs = None
    if y_obs is not None and y_obs.ndim == 2:
        ax.scatter(t, y_obs[0], s=10, alpha=0.4, color="0.5", label="Observations")

    # MC reference
    ax.plot(t, mean_mc, linestyle=":", label="MC mean")
    ax.fill_between(
        t,
        mean_mc - 1.96 * std_mc,
        mean_mc + 1.96 * std_mc,
        alpha=0.25,
        label="MC 95% CI",
    )

    # PN / quadrature method
    ax.plot(t, mean_pn, label=f"{pn_label} mean")
    ax.fill_between(
        t,
        mean_pn - 1.96 * std_pn,
        mean_pn + 1.96 * std_pn,
        alpha=0.3,
        label=f"{pn_label} 95% CI",
    )

    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    saved_files.append(path)


# Generate plots
counter = 1
for res in results:
    name = res['name']
    t = res['spherical']['t']  # common t

    sp_mean = res['spherical']['mean']
    sp_std = res['spherical']['std']

    gh_mean = res['gh']['mean']
    gh_std = res['gh']['std']

    mc_mean = res['mc']['mean']
    mc_std = res['mc']['std']

    pn_goal_mean = res['pn_goal']['mean']
    pn_goal_std = res['pn_goal']['std']

    pf_mean = res['param_filter']['mean']
    pf_std = res['param_filter']['std']

    ps_mean = res['param_smoother']['mean']
    ps_std = res['param_smoother']['std']

    y_obs = res['y_obs']

    # Decide which curve to use in PN-vs-MC mode
    if PN_METHOD == "pn_goal":
        pn_mean_full = pn_goal_mean
        pn_std_full = pn_goal_std
        pn_label_base = "PN p_goal (GH)"
    elif PN_METHOD == "param_filter":
        pn_mean_full = pf_mean
        pn_std_full = pf_std
        pn_label_base = "Param filter (PN+GH)"
    elif PN_METHOD == "param_smoother":
        pn_mean_full = ps_mean
        pn_std_full = ps_std
        pn_label_base = "Param smoother (PN+GH)"
    elif PN_METHOD == "gh_quadrature":
        pn_mean_full = gh_mean
        pn_std_full = gh_std
        pn_label_base = "GH quadrature"
    elif PN_METHOD == "spherical_quadrature":
        pn_mean_full = sp_mean
        pn_std_full = sp_std
        pn_label_base = "Spherical quadrature"
    else:
        raise ValueError(f"Unknown PN_METHOD: {PN_METHOD}")

    if sp_mean.shape[0] == 1:
        # 1D state
        if PLOT_MODE == "all":
            plot_ci_compare(
                t,
                mc_mean[0], mc_std[0],
                sp_mean[0], sp_std[0],
                gh_mean[0], gh_std[0],
                pn_goal_mean[0], pn_goal_std[0],
                pf_mean[0], pf_std[0],
                ps_mean[0], ps_std[0],
                y_obs=y_obs,
                title=f"{name}: component 1",
                ylabel="y",
                fname=f"{counter:02d}_{name.replace(' ', '_')}_all.png"
            )
        elif PLOT_MODE == "pn_vs_mc":
            plot_pn_vs_mc(
                t,
                mc_mean[0], mc_std[0],
                pn_mean_full[0], pn_std_full[0],
                y_obs=y_obs,
                pn_label=pn_label_base,
                title=f"{name}: component 1",
                ylabel="y",
                fname=f"{counter:02d}_{name.replace(' ', '_')}_PN_vs_MC_y.png"
            )
        else:
            raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE}")
        counter += 1
    else:
        # Multi-dimensional: plot each component separately
        for k in range(sp_mean.shape[0]):
            y_obs_k = y_obs if y_obs.shape[0] == 1 else y_obs[[k]]

            if PLOT_MODE == "all":
                plot_ci_compare(
                    t,
                    mc_mean[k], mc_std[k],
                    sp_mean[k], sp_std[k],
                    gh_mean[k], gh_std[k],
                    pn_goal_mean[k], pn_goal_std[k],
                    pf_mean[k], pf_std[k],
                    ps_mean[k], ps_std[k],
                    y_obs=y_obs_k,
                    title=f"{name}: component {k+1}",
                    ylabel=f"y{k+1}",
                    fname=f"{counter:02d}_{name.replace(' ', '_')}_all_y{k+1}.png"
                )
            elif PLOT_MODE == "pn_vs_mc":
                plot_pn_vs_mc(
                    t,
                    mc_mean[k], mc_std[k],
                    pn_mean_full[k], pn_std_full[k],
                    y_obs=y_obs_k,
                    pn_label=pn_label_base,
                    title=f"{name}: component {k+1}",
                    ylabel=f"y{k+1}",
                    fname=f"{counter:02d}_{name.replace(' ', '_')}_PN_vs_MC_y{k+1}.png"
                )
            else:
                raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE}")

            counter += 1

print("\n=== PN p_goal std diagnostics ===")
for res in results:
    name = res['name']
    pg_std = res['pn_goal']['std']
    print(f"{name}: p_goal std: min={pg_std.min():.3e}, max={pg_std.max():.3e}")

print("Saved plot files:")
for p in saved_files:
    print("  ", p)
