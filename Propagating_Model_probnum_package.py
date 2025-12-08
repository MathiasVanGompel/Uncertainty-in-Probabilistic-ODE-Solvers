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

# ============================================================
# Cubature / Quadrature methods
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

# ============================================================
# ODEs and integration
# ============================================================

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


# Analytic Jacobians of the vector fields (for EK1 / PN)
def J_fhn(t, y, a=0.0, b=0.08, c=0.07, d=1.25):
    y1, y2 = y
    df1_dy1 = 1.0 - y1**2
    df1_dy2 = -1.0
    df2_dy1 = 1.0 / d
    df2_dy2 = -c / d
    return np.array([[df1_dy1, df1_dy2],
                     [df2_dy1, df2_dy2]])


def J_lv(t, y, a=5.0, b=0.5, c=5.0, d=0.5):
    y1, y2 = y
    df1_dy1 = a - b * y2
    df1_dy2 = -b * y1
    df2_dy1 = d * y2
    df2_dy2 = -c + d * y1
    return np.array([[df1_dy1, df1_dy2],
                     [df2_dy1, df2_dy2]])


def J_vdp(t, y, mu=0.05):
    y1, y2 = y
    df1_dy1 = 0.0
    df1_dy2 = 1.0
    df2_dy1 = -2.0 * mu * y1 * y2 - 1.0
    df2_dy2 = mu * (1.0 - y1**2)
    return np.array([[df1_dy1, df1_dy2],
                     [df2_dy1, df2_dy2]])


def _numerical_jacobian(fun, t, y, eps=1e-6):
    """
    Finite-difference Jacobian used when an analytic Jacobian is not provided.
    """
    y = np.asarray(y, dtype=float)
    d = y.size
    J = np.zeros((d, d))
    f0 = fun(t, y)
    for i in range(d):
        e = np.zeros(d)
        e[i] = eps
        f_plus = fun(t, y + e)
        f_minus = fun(t, y - e)
        J[:, i] = (f_plus - f_minus) / (2.0 * eps)
    return J


# Wrapper to integrate with LSODA (classic deterministic reference)
def integrate_deterministic(fun, t_span, y0, args=(), t_eval=None,
                            rtol=1e-6, atol=1e-8):
    sol = solve_ivp(fun, t_span, y0, method='LSODA', args=args, t_eval=t_eval,
                    rtol=rtol, atol=atol)
    if not sol.success:
        warnings.warn(f"Integration failed: {sol.message}")
    return sol.t, sol.y  # t shape (M,), y shape (d, M)


# ProbNum-based IWP(1)+EK1 solver (used inside the quadrature propagations)
def integrate_probnum(
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
    with a once-integrated Wiener process prior (IWP(1))

    Returns
    -------
    t_eval : (M,)
    y_mean : (d, M)
    y_std  : (d, M)
    """
    try:
        from probnum.diffeq import probsolve_ivp
    except ImportError as e:
        raise ImportError(
            "ProbNum is required for the probabilistic ODE solver. "
            "Install via `pip install probnum`."
        ) from e

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
            return _numerical_jacobian(lambda _t, _y: fun(_t, _y, *args),
                                       t, np.asarray(y))
        return jac_fun(t, np.asarray(y), *args)

    odesol = probsolve_ivp(
        f=f,
        t0=t0,
        tmax=t1,
        y0=y0,
        df=df,
        method=method,
        algo_order=algo_order,      # IWP(1)
        adaptive=adaptive,          # fixed-step as in the paper
        step=step,
        time_stops=t_eval,          # ensure we step through all t_eval
        dense_output=True,
        diffusion_model=diffusion_model,  # automatic calibration of time-constant diffusion
    )

    rv_eval = odesol(t_eval)
    y_mean = np.asarray(rv_eval.mean)        # (M, d)
    y_std = np.asarray(rv_eval.std)         # (M, d)

    # Transpose to match shape (d, M) used elsewhere in this script
    return t_eval, y_mean.T, y_std.T

# ============================================================
# Propagation: deterministic quadrature + Monte Carlo
# ============================================================

def propagate_deterministic(system, t_span, t_eval, theta_mean, theta_cov,
                            quad_method="spherical", n_gh_1d=5,
                            solver="probnum", pn_kwargs=None):
    """
    Propagate uncertainty using a deterministic quadrature rule
    (spherical cubature or Gauss–Hermite) over theta.

    If solver == "probnum", each theta-node is solved with ProbNum's
    EK1 + IWP(1) ODE filter, as in Yao et al. (2025).
    If solver == "deterministic", we fall back to SciPy's LSODA.
    """
    name = system['name']
    ode_fun = system['ode_fun']
    jac_fun = system.get('jac_fun')
    theta_to_setup = system['theta_to_setup']

    t0 = time.perf_counter()

    if quad_method == "spherical":
        nodes, w = spherical_cubature(theta_mean, theta_cov)
    elif quad_method == "gh":
        nodes, w = gauss_hermite_cubature(theta_mean, theta_cov,
                                          n_points_1d=n_gh_1d)
    else:
        raise ValueError(f"Unknown quad_method: {quad_method}")

    Y_nodes = []  # list of tuples (mean_y, var_y) each of shape (dim_y, M)
    t_out = None

    for th in nodes:
        y0, params = theta_to_setup(th)

        if solver == "probnum":
            # Use ProbNum EK1 IWP(1) ODE filter
            kwargs = pn_kwargs or {}
            t, y_mean, y_std = integrate_probnum(
                ode_fun, jac_fun, t_span, y0, args=params, t_eval=t_eval,
                **kwargs
            )
            y = y_mean
            y_var = y_std**2
        elif solver == "deterministic":
            # Classic deterministic ODE solver (LSODA)
            t, y_det = integrate_deterministic(
                ode_fun, t_span, y0, args=params, t_eval=t_eval
            )
            y = y_det
            y_var = np.zeros_like(y_det)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if t_out is None:
            t_out = t
        Y_nodes.append((y, y_var))

    # Weighted mean and variance (including solver covariance)
    Y_means = np.stack([p[0] for p in Y_nodes], axis=0)  # (N_nodes, dim_y, M)
    Y_vars = np.stack([p[1] for p in Y_nodes], axis=0)   # (N_nodes, dim_y, M)

    mean = np.tensordot(w, Y_means, axes=(0, 0))  # (dim_y, M)
    diffs = Y_means - mean[None, :, :]            # (N_nodes, dim_y, M)
    var = np.tensordot(w, Y_vars + diffs**2, axes=(0, 0))  # (dim_y, M)
    std = np.sqrt(var)

    t1 = time.perf_counter()
    return {
        't': t_out,
        'mean': mean,
        'std': std,
        'time': t1 - t0,
        'method': quad_method,
        'name': name
    }


def propagate_mc(system, t_span, t_eval, theta_mean, theta_cov, n_mc=400,
                 solver="deterministic"):
    """
    Monte Carlo reference propagation.

    As in Yao et al., this uses a classic ODE solver (LSODA) for all samples.
    """
    name = system['name']
    ode_fun = system['ode_fun']
    jac_fun = system.get('jac_fun')
    theta_to_setup = system['theta_to_setup']

    t0 = time.perf_counter()

    Y_mc = []
    Y_vars = []
    t_out = None
    for _ in range(n_mc):
        theta = rng.multivariate_normal(theta_mean, theta_cov)
        y0, params = theta_to_setup(theta)
        if solver == "probnum":
            # Optionally, one could also use ProbNum here,
            # but the paper uses LSODA as MC reference.
            t, y_mean, y_std = integrate_probnum(
                ode_fun, jac_fun, t_span, y0, args=params, t_eval=t_eval
            )
            y_var = y_std**2
            y = y_mean
        elif solver == "deterministic":
            t, y = integrate_deterministic(
                ode_fun, t_span, y0, args=params, t_eval=t_eval
            )
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
        't': t_out,
        'mean': mc_mean,
        'std': mc_std,
        'time': t1 - t0,
        'method': 'mc',
        'name': name
    }

# ============================================================
# Gauss–Markov IWP(1) + EK1 goal-variance method (PN & PN+Jac)
# ============================================================

def iwp1_matrices(h, kappa2, d):
    """
    Once-integrated Wiener process (IWP(1)) prior for d-dimensional y(t).

    State x = [y; v] \in R^{2d} with block transition:
      [ y_k ]   [1 h] [y_{k-1}] + w_k
      [ v_k ] = [0 1] [v_{k-1}]
    and w_k ~ N(0, Q(h)).
    """
    A_block = np.array([[1.0, h],
                        [0.0, 1.0]])
    Q_block = kappa2 * np.array([[h**3 / 3.0, h**2 / 2.0],
                                 [h**2 / 2.0, h]])
    A = np.kron(np.eye(d), A_block)
    Q = np.kron(np.eye(d), Q_block)
    return A, Q


def build_E0_E1(d):
    """Projections: y = E0 x, v = E1 x for x = [y; v]."""
    E0 = np.hstack([np.eye(d), np.zeros((d, d))])  # d x 2d
    E1 = np.hstack([np.zeros((d, d)), np.eye(d)])  # d x 2d
    return E0, E1


def ek1_iwp1_goal_cov(
    f, J_f, mu0, Sigma0,
    T, h, kappa2=1.0, R_scale=1e-6
):
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
    G = np.vstack([np.eye(d), Jf0])  # (2d x d)
    P0 = G @ Sigma0 @ G.T            # Cov(x0) induced by initial y uncertainty

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
        H = np.hstack([-Jf, np.eye(d)])  # d x 2d

        S = H @ P_pred @ H.T + R
        K = np.linalg.solve(S, (P_pred @ H.T).T).T  # 2d x d

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


def propagate_pn_iwp1_goal(system, t_span, t_eval, theta_mean, theta_cov,
                           kappa2=1.0, R_scale=1e-6):
    """
    Wrapper to run the IWP(1)+EK1+Jacobian goal-variance method
    for systems where theta is the uncertain initial state y(0).

    For logistic (parameter uncertainty), this method is not applicable:
    returns None.
    """
    name = system['name']
    dim_y = system['dim_y']

    # Only apply to problems where theta is y(0) (FHN, LV, VdP)
    if name == 'FitzHugh–Nagumo':
        params = (0.0, 0.08, 0.07, 1.25)
        def f(t, y):   return fhn_fun(t, y, *params)
        def J_f(t, y): return J_fhn(t, y, *params)
    elif name == 'Lotka–Volterra':
        params = (5.0, 0.5, 5.0, 0.5)
        def f(t, y):   return lotkavolterra_fun(t, y, *params)
        def J_f(t, y): return J_lv(t, y, *params)
    elif name == 'Van der Pol':
        mu = 0.05
        def f(t, y):   return vanderpol_fun(t, y, mu)
        def J_f(t, y): return J_vdp(t, y, mu)
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
        f=f, J_f=J_f,
        mu0=mu0, Sigma0=Sigma0,
        T=T, h=h,
        kappa2=kappa2, R_scale=R_scale
    )
    t1 = time.perf_counter()

    # Project
    y_mean_pn,   y_var_pn   = extract_y_stats_from_P(m_list, P_list, d=d)
    y_mean_goal, y_var_goal = extract_y_stats_from_P(m_list, P_goal_list, d=d)

    return {
        't': t_grid,
        'mean_pn': y_mean_pn.T,       # shape (d, M) to match others
        'std_pn':  np.sqrt(y_var_pn).T,
        'mean_goal': y_mean_goal.T,
        'std_goal':  np.sqrt(y_var_goal).T,
        'time': t1 - t0,
        'method': 'pn_iwp1_goal',
        'name': name
    }

# ============================================================
# All ODE problems
# ============================================================

def make_logistic_problem():
    a = 3.0
    y0_fixed = np.array([0.05])

    def theta_to_setup(theta):
        # theta is scalar b
        b = float(theta[0])
        return y0_fixed.copy(), (a, b)

    return {
        'name': 'Logistic',
        'ode_fun': lambda t, y, a, b: logistic_fun(t, y, a, b),
        'jac_fun': lambda t, y, a, b: logistic_jacobian(t, y, a, b),
        'theta_to_setup': theta_to_setup,
        'dim_y': 1,
        'dim_theta': 1
    }


def make_fhn_problem():
    params = (0.0, 0.08, 0.07, 1.25)

    def theta_to_setup(theta):
        # theta is y0 (2,)
        y0 = theta.astype(float)
        return y0, params

    return {
        'name': 'FitzHugh–Nagumo',
        'ode_fun': lambda t, y, a, b, c, d: fhn_fun(t, y, a, b, c, d),
        'jac_fun': lambda t, y, a, b, c, d: J_fhn(t, y, a, b, c, d),
        'theta_to_setup': theta_to_setup,
        'dim_y': 2,
        'dim_theta': 2
    }


def make_lv_problem():
    params = (5.0, 0.5, 5.0, 0.5)

    def theta_to_setup(theta):
        y0 = theta.astype(float)
        return y0, params

    return {
        'name': 'Lotka–Volterra',
        'ode_fun': lambda t, y, a, b, c, d: lotkavolterra_fun(t, y, a, b, c, d),
        'jac_fun': lambda t, y, a, b, c, d: J_lv(t, y, a, b, c, d),
        'theta_to_setup': theta_to_setup,
        'dim_y': 2,
        'dim_theta': 2
    }


def make_vdp_problem():
    mu = 0.05

    def theta_to_setup(theta):
        y0 = theta.astype(float)
        return y0, (mu,)

    return {
        'name': 'Van der Pol',
        'ode_fun': lambda t, y, mu: vanderpol_fun(t, y, mu),
        'jac_fun': lambda t, y, mu: J_vdp(t, y, mu),
        'theta_to_setup': theta_to_setup,
        'dim_y': 2,
        'dim_theta': 2
    }

# ============================================================
# Run algorithms
# ============================================================

problems = []

# Logistic
problems.append({
    'problem': make_logistic_problem(),
    'theta_mean': np.array([3.0]),
    'theta_cov': np.array([[0.01]]),
    't_span': (0.0, 3.0),
    't_eval': np.linspace(0.0, 3.0, 400),
    'n_mc': 600,
    # Step size used by ProbNum EK1 + IWP(1) in Yao et al.
    'pn_step': 1e-2,
})

# FHN
problems.append({
    'problem': make_fhn_problem(),
    'theta_mean': np.array([0.5, 1.0]),
    'theta_cov': 0.1 * np.eye(2),
    't_span': (0.0, 7.0),
    't_eval': np.linspace(0.0, 7.0, 700),
    'n_mc': 500,
    'pn_step': 5e-3,
})

# Lotka–Volterra
problems.append({
    'problem': make_lv_problem(),
    'theta_mean': np.array([5.0, 5.0]),
    'theta_cov': 0.3 * np.eye(2),
    't_span': (0.0, 2.0),
    't_eval': np.linspace(0.0, 2.0, 1200),
    'n_mc': 1200,
    'pn_step': 5e-3,
})

# Van der Pol
problems.append({
    'problem': make_vdp_problem(),
    'theta_mean': np.array([5.0, 5.0]),
    'theta_cov': 2.0 * np.eye(2),
    't_span': (0.0, 10.0),
    't_eval': np.linspace(0.0, 10.0, 1200),
    'n_mc': 500,
    'pn_step': 1e-2,
})

results = []
timings = []

for model in problems:
    problem = model['problem']
    theta_mean = model['theta_mean']
    theta_cov = model['theta_cov']
    t_span = model['t_span']
    t_eval = model['t_eval']
    n_mc = model['n_mc']
    pn_step = model.get('pn_step', None)

    pn_kwargs = {}
    if pn_step is not None:
        pn_kwargs['step'] = pn_step

    # Spherical + ProbNum EK1/IWP(1)
    res_sp = propagate_deterministic(
        problem, t_span, t_eval, theta_mean, theta_cov,
        quad_method="spherical",
        solver="probnum",
        pn_kwargs=pn_kwargs
    )

    # Gauss–Hermite + ProbNum EK1/IWP(1)
    res_gh = propagate_deterministic(
        problem, t_span, t_eval, theta_mean, theta_cov,
        quad_method="gh", n_gh_1d=5,
        solver="probnum",
        pn_kwargs=pn_kwargs
    )

    # Monte Carlo reference (SciPy LSODA, not ProbNum)
    res_mc = propagate_mc(
        problem, t_span, t_eval, theta_mean, theta_cov,
        n_mc=n_mc,
        solver="deterministic"
    )

    # Probabilistic ODE solver + goal variance (for initial-state uncertainty)
    res_pn = propagate_pn_iwp1_goal(
        problem, t_span, t_eval, theta_mean, theta_cov,
        kappa2=1.0, R_scale=1e-6
    )

    results.append({
        'name': problem['name'],
        'spherical': res_sp,
        'gh': res_gh,
        'mc': res_mc,
        'pn': res_pn
    })

    timings.append({
        'name': problem['name'],
        'time_spherical': res_sp['time'],
        'time_gh': res_gh['time'],
        'time_mc': res_mc['time'],
        'time_pn': res_pn['time'] if res_pn is not None else None
    })

# Print timing comparison
for tm in timings:
    name = tm['name']
    t_sp = tm['time_spherical']
    t_gh = tm['time_gh']
    t_mc = tm['time_mc']
    t_pn = tm['time_pn']
    print(f"{name}:")
    print(f"  Spherical (ProbNum) time = {t_sp:.3f} s")
    print(f"  Gauss–Hermite (ProbNum) time = {t_gh:.3f} s")
    print(f"  MC (LSODA) time = {t_mc:.3f} s")
    if t_pn is not None:
        print(f"  PN+Jac (IWP1+EK1) time = {t_pn:.3f} s")
    print()

# ============================================================
# Plotting
# ============================================================

outdir = "data"
os.makedirs(outdir, exist_ok=True)
saved_files = []


def plot_ci_compare(t,
                    mean_sp, std_sp,
                    mean_gh, std_gh,
                    mean_mc, std_mc,
                    title, ylabel="y", fname="plot.png",
                    mean_pn=None, std_pn=None,
                    mean_goal=None, std_goal=None):
    fig = plt.figure()

    # Spherical
    plt.plot(t, mean_sp, label="Spherical mean (PN)")
    plt.fill_between(t, mean_sp - 1.96 * std_sp, mean_sp + 1.96 * std_sp,
                     alpha=0.25, label="Spherical 95% CI (PN)")

    # Gauss–Hermite
    plt.plot(t, mean_gh, linestyle="--", label="Gauss–Hermite mean (PN)")
    plt.fill_between(t, mean_gh - 1.96 * std_gh, mean_gh + 1.96 * std_gh,
                     alpha=0.25, label="GH 95% CI (PN)")

    # Monte Carlo
    plt.plot(t, mean_mc, linestyle=":", label="MC mean (LSODA)")
    plt.fill_between(t, mean_mc - 1.96 * std_mc, mean_mc + 1.96 * std_mc,
                     alpha=0.2, label="MC 95% CI")

    # PN-only (EK1 variance only)
    if mean_pn is not None and std_pn is not None:
        plt.plot(t, mean_pn, linestyle="-.", color="tab:red",
                 label="PN mean (EK1)")
        plt.fill_between(t, mean_pn - 1.96 * std_pn, mean_pn + 1.96 * std_pn,
                         alpha=0.2, color="tab:red", label="PN 95% CI")

    # PN+Jac goal variance
    if mean_goal is not None and std_goal is not None:
        plt.plot(t, mean_goal, linestyle="-", color="tab:green",
                 label="Goal mean (PN+Jac)")
        plt.fill_between(t, mean_goal - 1.96 * std_goal,
                         mean_goal + 1.96 * std_goal,
                         alpha=0.2, color="tab:green",
                         label="Goal 95% CI (PN+Jac)")

    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150)
    plt.show()
    saved_files.append(path)


# Generate plots
counter = 1
for res in results:
    name = res['name']
    t = res['spherical']['t']

    sp_mean = res['spherical']['mean']
    sp_std = res['spherical']['std']

    gh_mean = res['gh']['mean']
    gh_std = res['gh']['std']

    mc_mean = res['mc']['mean']
    mc_std = res['mc']['std']

    pn_res = res['pn']

    if sp_mean.shape[0] == 1:
        # Logistic: PN method is None, so no PN curves
        mean_pn = std_pn = mean_goal = std_goal = None
        if pn_res is not None:
            mean_pn = pn_res['mean_pn'][0]
            std_pn = pn_res['std_pn'][0]
            mean_goal = pn_res['mean_goal'][0]
            std_goal = pn_res['std_goal'][0]

        plot_ci_compare(
            t,
            sp_mean[0], sp_std[0],
            gh_mean[0], gh_std[0],
            mc_mean[0], mc_std[0],
            f"{name}: component 1",
            ylabel="y",
            fname=f"{counter:02d}_{name.replace(' ', '_')}_y.png",
            mean_pn=mean_pn, std_pn=std_pn,
            mean_goal=mean_goal, std_goal=std_goal
        )
        counter += 1
    else:
        for k in range(sp_mean.shape[0]):
            mean_pn = std_pn = mean_goal = std_goal = None
            if pn_res is not None:
                mean_pn = pn_res['mean_pn'][k]
                std_pn = pn_res['std_pn'][k]
                mean_goal = pn_res['mean_goal'][k]
                std_goal = pn_res['std_goal'][k]

            plot_ci_compare(
                t,
                sp_mean[k], sp_std[k],
                gh_mean[k], gh_std[k],
                mc_mean[k], mc_std[k],
                f"{name}: component {k + 1}",
                ylabel=f"y{k + 1}",
                fname=f"{counter:02d}_{name.replace(' ', '_')}_y{k + 1}.png",
                mean_pn=mean_pn, std_pn=std_pn,
                mean_goal=mean_goal, std_goal=std_goal
            )
            counter += 1

print("Saved plot files:")
for p in saved_files:
    print("  ", p)
