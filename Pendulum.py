"""
prob_double_pendulum.py

Probabilistic double pendulum ODE solver using:
- sigma-point (Gauss–Hermite tensor grid) quadrature
- Bayesian quadrature (with integral variance)
plus optional Monte Carlo for validation.

Designed to match the style of your Uncertainty-in-Probabilistic-ODE-Solvers repo:
- problem factory `make_double_pendulum_problem`
- generic `propagate_quadrature` and `propagate_bayesian_quadrature`
- main section that runs an example and plots results
"""

import math
import time
import warnings

import numpy as np
import numpy.linalg as npla
from scipy.integrate import solve_ivp
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 0. Design choice: Gauss–Hermite order per dimension
# ----------------------------------------------------------------------

GH_ORDER_PER_DIM = 30  # 3 -> 9 nodes in 2D, 5 -> 25 nodes, etc.


# ----------------------------------------------------------------------
# 1. Double pendulum ODE
# ----------------------------------------------------------------------

def double_pendulum_fun(t, y, L1, L2, m1, m2, g=9.81):
    """
    ODE function for the double pendulum.

    State:
        y = [theta1, theta2, omega1, omega2]
    where
        theta1, theta2 : angles (rad)
        omega1, omega2 : angular velocities (rad/s)

    Returns:
        y_dot = [d(theta1)/dt, d(theta2)/dt, d(omega1)/dt, d(omega2)/dt]
    """
    theta1, theta2, omega1, omega2 = y

    # Common terms
    delta = theta1 - theta2
    cos_delta = math.cos(2 * theta1 - 2 * theta2)
    sin_delta = math.sin(theta1 - theta2)

    # Denominators
    denom1 = L1 * (2 * m1 + m2 - m2 * cos_delta)
    denom2 = L2 * (2 * m1 + m2 - m2 * cos_delta)

    # Numerators (standard double pendulum equations)
    num1 = (
        -g * (2 * m1 + m2) * math.sin(theta1)
        - m2 * g * math.sin(theta1 - 2 * theta2)
        - 2 * m2 * sin_delta * (omega2**2 * L2 + omega1**2 * L1 * math.cos(delta))
    )
    omega1_dot = num1 / denom1

    num2 = (
        2
        * sin_delta
        * (
            omega1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * math.cos(theta1)
            + omega2**2 * L2 * m2 * math.cos(delta)
        )
    )
    omega2_dot = num2 / denom2

    # Time derivative of state
    return np.array([omega1, omega2, omega1_dot, omega2_dot], dtype=float)


# ----------------------------------------------------------------------
# RBF kernel + Bayesian quadrature weights + integral variance
# ----------------------------------------------------------------------

def rbf_kernel(x, y, lengthscale):
    """
    Isotropic RBF kernel with unit amplitude.

    k(x, y) = exp(-0.5 * ||x - y||^2 / l^2)
    """
    diff = x - y
    r2 = np.dot(diff, diff)
    return math.exp(-0.5 * r2 / (lengthscale**2))


def compute_bq_weights(nodes, mu, cov,
                       lengthscale=None,
                       n_samples_z=2000,
                       n_samples_pi=2000,
                       seed=0):
    """
    Compute Bayesian quadrature weights for integral over theta ~ N(mu, cov),
    using a GP with RBF kernel and zero mean.

    Also returns the posterior *integral variance* Var_BQ[I] = Π - z^T K^{-1} z.

    We approximate:
    - z_i = E_theta[k(x_i, theta)] via Monte Carlo,
    - Π   = E_{theta,theta'}[k(theta, theta')] via Monte Carlo over pairs.

    Args:
        nodes:       (M, d) design points (theta nodes)
        mu:          (d,) mean of Gaussian p(theta)
        cov:         (d,d) covariance of Gaussian p(theta)
        lengthscale: scalar RBF lengthscale (if None, set from cov)
        n_samples_z: MC samples to approximate z
        n_samples_pi:MC samples to approximate Π
        seed:        RNG seed

    Returns:
        w:        (M,) BQ weights (normalized so sum(w) ≈ 1)
        int_var:  scalar posterior BQ integral variance (unit-amplitude GP)
    """
    nodes = np.asarray(nodes)
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    M, d = nodes.shape

    # Default lengthscale ~ typical std of theta
    if lengthscale is None:
        stds = np.sqrt(np.diag(cov))
        lengthscale = float(np.mean(stds)) if np.all(stds > 0) else 1.0

    # Kernel matrix K
    K = np.empty((M, M))
    for i in range(M):
        for j in range(M):
            K[i, j] = rbf_kernel(nodes[i], nodes[j], lengthscale)

    # Approximate z_i = E_theta[k(x_i, theta)] via Monte Carlo
    rng = np.random.default_rng(seed)
    thetas_z = rng.multivariate_normal(mu, cov, size=n_samples_z)  # (n_samples_z, d)
    z = np.empty(M)
    for i in range(M):
        diffs = thetas_z - nodes[i]                 # (n_samples_z, d)
        r2 = np.sum(diffs**2, axis=1)              # (n_samples_z,)
        k_vals = np.exp(-0.5 * r2 / (lengthscale**2))
        z[i] = np.mean(k_vals)

    # Approximate Π = E_{theta,theta'}[k(theta, theta')] via MC over pairs
    thetas1 = rng.multivariate_normal(mu, cov, size=n_samples_pi)
    thetas2 = rng.multivariate_normal(mu, cov, size=n_samples_pi)
    k_pair = []
    for i in range(n_samples_pi):
        diff = thetas1[i] - thetas2[i]
        r2 = np.dot(diff, diff)
        k_pair.append(math.exp(-0.5 * r2 / (lengthscale**2)))
    Pi = float(np.mean(k_pair))

    # Solve for weights and integral variance
    jitter = 1e-8 * np.eye(M)
    K_reg = K + jitter

    # Solve K w = z
    w = np.linalg.solve(K_reg, z)

    # Normalize so that constant functions integrate correctly
    w_sum = np.sum(w)
    if abs(w_sum) > 1e-12:
        w /= w_sum

    # Posterior BQ integral variance: Var[I] = Π - z^T K^{-1} z
    # Use Cholesky for numerical stability.
    L = np.linalg.cholesky(K_reg)
    tmp = np.linalg.solve(L, z)
    K_inv_z = np.linalg.solve(L.T, tmp)
    int_var = Pi - float(z @ K_inv_z)

    # Numerical safety: clip tiny negatives
    int_var = max(int_var, 0.0)

    return w, int_var


# ----------------------------------------------------------------------
# 2. Gauss–Hermite tensor grid for N(mu, Sigma)
# ----------------------------------------------------------------------

def gauss_hermite_design(mu, Sigma, order_per_dim=3):
    """
    Multivariate Gauss–Hermite quadrature nodes and weights for a Gaussian N(mu, Sigma).

    We build a tensor grid of 1D Gauss–Hermite nodes and weights for Z ~ N(0, I_d),
    then transform via theta = mu + L z, where Sigma = L L^T.

    In d dimensions, with order_per_dim = n, we get n^d nodes.

    Args:
        mu:           (d,) mean
        Sigma:        (d,d) covariance
        order_per_dim: 1D Gauss–Hermite order

    Returns:
        nodes: (M, d) where M = order_per_dim**d
        w:     (M,) weights, summing to 1
    """
    mu = np.atleast_1d(mu)
    d = mu.shape[0]
    Sigma = np.asarray(Sigma)

    # 1D Gauss–Hermite nodes/weights for integral ∫ f(x) e^{-x^2} dx
    x1, w1 = hermgauss(order_per_dim)
    # Standard normal Z ~ N(0,1): z = sqrt(2)*x
    z1 = np.sqrt(2.0) * x1

    # Build tensor-product grid in Z-space
    grids = np.meshgrid(*([np.arange(order_per_dim)] * d), indexing="ij")
    grids = [g.flatten() for g in grids]  # each is (M,)
    M = order_per_dim**d

    z = np.zeros((M, d))
    w = np.ones(M)
    for dim in range(d):
        idx = grids[dim]
        z[:, dim] = z1[idx]
        w *= w1[idx]   # tensor product of 1D weights

    # Normalization for d-dimensional standard normal: divide by pi^(d/2)
    w /= (math.pi ** (d / 2))

    # Transform to N(mu, Sigma) using Cholesky
    jitter = 1e-12 * np.eye(d)
    L = npla.cholesky(Sigma + jitter)
    # nodes = mu + L @ z_i  (z is (M,d), L is (d,d))
    nodes = mu[None, :] + z @ L.T  # (M,d)

    return nodes, w


# ----------------------------------------------------------------------
# 3. Generic ODE integrator wrapper
# ----------------------------------------------------------------------

def integrate(fun, t_span, y0, args=(), t_eval=None, rtol=1e-6, atol=1e-8):
    """
    Integrate the ODE y' = fun(t, y, *args) over [t_span[0], t_span[1]]
    with initial state y0, using SciPy's solve_ivp (LSODA).

    Returns:
        t_arr: (M,) time points
        y_arr: (state_dim, M) solution values
    """
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
        warnings.warn(f"ODE integration failed: {sol.message}")
    return sol.t, sol.y


# ----------------------------------------------------------------------
# 4. Problem definition: double pendulum
# ----------------------------------------------------------------------

def make_double_pendulum_problem():
    """
    Build a problem dict in the same spirit as your other problems.

    We treat the initial angles as uncertain parameters:
        theta = [theta1_init, theta2_init]

    The initial angular velocities are set to zero (you can extend theta
    to include them if you want).
    """
    # Fixed physical parameters (you can expose these as uncertain too)
    L1 = 1.0
    L2 = 1.0
    m1 = 1.0
    m2 = 1.0

    def theta_to_setup(theta):
        """
        Map parameter vector theta -> (initial_state, params) for the ODE.

        theta: [theta1_init, theta2_init]
        """
        theta1_0 = float(theta[0])
        theta2_0 = float(theta[1])

        # Initial state: [theta1, theta2, omega1, omega2]
        y0 = np.array([theta1_0, theta2_0, 0.0, 0.0], dtype=float)

        # Parameters to pass to double_pendulum_fun via solve_ivp(*args)
        params = (L1, L2, m1, m2)
        return y0, params

    system = {
        "name": "Double Pendulum",
        "ode_fun": lambda t, y, L1, L2, m1, m2: double_pendulum_fun(
            t, y, L1, L2, m1, m2
        ),
        "theta_to_setup": theta_to_setup,
        "dim_y": 4,
        "dim_theta": 2,
    }
    return system


# ----------------------------------------------------------------------
# 5. Probabilistic propagation via quadrature (+ optional MC)
# ----------------------------------------------------------------------

def propagate_quadrature(system, t_span, t_eval, theta_mean, theta_cov, n_mc=0):
    """
    Propagate uncertainty through an ODE using Gauss–Hermite quadrature
    (tensor grid in parameter space), plus optional Monte Carlo for validation.

    Args:
        system:      dict with keys:
                        'ode_fun'       : callable f(t, y, *params)
                        'theta_to_setup': callable mapping theta -> (y0, params)
                        'dim_y'         : state dimension
                        'dim_theta'     : parameter dimension
        t_span:      (t0, tf)
        t_eval:      array of times to evaluate solution at
        theta_mean:  (dim_theta,) mean of Gaussian over theta
        theta_cov:   (dim_theta, dim_theta) covariance of Gaussian over theta
        n_mc:        number of Monte Carlo trajectories (0 to disable)

    Returns:
        results: dict with keys:
            't'       : time array
            'sp_mean' : quadrature mean (dim_y, M)
            'sp_std'  : quadrature std (dim_y, M)
            'sp_time' : elapsed time for quadrature
            'mc_mean' : Monte Carlo mean (dim_y, M), if n_mc > 0
            'mc_std'  : Monte Carlo std  (dim_y, M), if n_mc > 0
            'mc_time' : elapsed time for MC, if n_mc > 0
    """
    ode_fun = system["ode_fun"]
    theta_to_setup = system["theta_to_setup"]

    # 1) Gauss–Hermite quadrature nodes + weights in theta-space
    t0 = time.perf_counter()
    nodes, w = gauss_hermite_design(theta_mean, theta_cov, order_per_dim=GH_ORDER_PER_DIM)
    # nodes: (n_nodes, dim_theta), w: (n_nodes,)

    # 2) Integrate ODE for each node
    Y = []
    t_arr = None
    for theta in nodes:
        y0, params = theta_to_setup(theta)
        t_arr, y_arr = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
        Y.append(y_arr)
    Y = np.stack(Y, axis=0)  # (n_nodes, dim_y, M_t)

    # 3) Weighted mean and variance over nodes
    sp_mean = np.tensordot(w, Y, axes=(0, 0))  # (dim_y, M_t)
    diffs = Y - sp_mean[None, :, :]
    sp_var = np.tensordot(w, diffs**2, axes=(0, 0))  # (dim_y, M_t)
    sp_var = np.maximum(sp_var, 0.0)
    sp_std = np.sqrt(sp_var)
    sp_time = time.perf_counter() - t0

    results = {
        "t": t_arr,
        "sp_mean": sp_mean,
        "sp_std": sp_std,
        "sp_time": sp_time,
    }

    # 4) Monte Carlo validation (optional)
    if n_mc and n_mc > 0:
        mc_start = time.perf_counter()
        Y_mc = []
        rng = np.random.default_rng(seed=0)
        for _ in range(n_mc):
            theta_sample = rng.multivariate_normal(theta_mean, theta_cov)
            y0, params = theta_to_setup(theta_sample)
            _, y_arr = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
            Y_mc.append(y_arr)
        Y_mc = np.stack(Y_mc, axis=0)  # (n_mc, dim_y, M_t)
        mc_mean = np.mean(Y_mc, axis=0)
        mc_std = np.std(Y_mc, axis=0, ddof=0)
        mc_time = time.perf_counter() - mc_start

        results["mc_mean"] = mc_mean
        results["mc_std"] = mc_std
        results["mc_time"] = mc_time

    return results


# ----------------------------------------------------------------------
# Bayesian quadrature propagation with integral variance
# ----------------------------------------------------------------------

def propagate_bayesian_quadrature(system, t_span, t_eval, theta_mean, theta_cov,
                                  n_mc=0, n_bq_samples=2000, bq_lengthscale=None):
    """
    Propagate uncertainty through an ODE using Bayesian quadrature.

    - Uses Gauss–Hermite tensor grid nodes as design points in theta-space.
    - Computes BQ weights *and* the posterior integral variance Var_BQ[I].
    - Parametric variance (due to theta) is computed via BQ weights as before.
    - Total variance = parametric variance + integral-variance term.

    Args:
        system:      problem dict (ode_fun, theta_to_setup, ...)
        t_span:      (t0, tf)
        t_eval:      time grid
        theta_mean:  (d_theta,) mean of N(theta_mean, theta_cov)
        theta_cov:   (d_theta, d_theta) covariance
        n_mc:        optional number of MC trajectories for comparison
        n_bq_samples:MC samples used for z and Π in BQ
        bq_lengthscale: RBF lengthscale (None => infer from theta_cov)

    Returns:
        dict with:
            't'
            'bq_mean'
            'bq_std_param'   : std from param uncertainty only
            'bq_std_total'   : std including BQ integral variance
            'bq_int_var'     : scalar BQ integral variance
            'bq_time'
            (and 'mc_mean', 'mc_std', 'mc_time' if n_mc>0)
    """
    ode_fun = system["ode_fun"]
    theta_to_setup = system["theta_to_setup"]

    # 1) Use the same Gauss–Hermite design nodes as quadrature
    nodes, _ = gauss_hermite_design(theta_mean, theta_cov, order_per_dim=GH_ORDER_PER_DIM)

    t0 = time.perf_counter()

    # BQ weights + integral variance for scalar integrals
    w_bq, int_var = compute_bq_weights(
        nodes,
        theta_mean,
        theta_cov,
        lengthscale=bq_lengthscale,
        n_samples_z=n_bq_samples,
        n_samples_pi=n_bq_samples,
        seed=0,
    )

    # 2) Integrate ODE for each node
    Y = []
    t_arr = None
    for theta in nodes:
        y0, params = theta_to_setup(theta)
        t_arr, y_arr = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
        Y.append(y_arr)
    Y = np.stack(Y, axis=0)  # (M_nodes, dim_y, M_t)

    # 3) Parametric mean and variance via BQ weights
    bq_mean = np.tensordot(w_bq, Y, axes=(0, 0))  # (dim_y, M_t)

    Y_sq = Y**2
    bq_second_moment = np.tensordot(w_bq, Y_sq, axes=(0, 0))  # (dim_y, M_t)
    bq_var_param = bq_second_moment - bq_mean**2
    bq_var_param = np.maximum(bq_var_param, 0.0)  # clip numerical negatives

    # 4) Add integral variance as extra term (same scalar added everywhere)
    bq_var_total = bq_var_param + int_var
    bq_var_total = np.maximum(bq_var_total, 0.0)
    bq_std_param = np.sqrt(bq_var_param)
    bq_std_total = np.sqrt(bq_var_total)

    bq_time = time.perf_counter() - t0

    results = {
        "t": t_arr,
        "bq_mean": bq_mean,
        "bq_std_param": bq_std_param,
        "bq_std_total": bq_std_total,
        "bq_int_var": float(int_var),
        "bq_time": bq_time,
    }

    # 5) Optional Monte Carlo for comparison
    if n_mc and n_mc > 0:
        mc_start = time.perf_counter()
        Y_mc = []
        rng = np.random.default_rng(seed=1)
        for _ in range(n_mc):
            theta_sample = rng.multivariate_normal(theta_mean, theta_cov)
            y0, params = theta_to_setup(theta_sample)
            _, y_arr = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
            Y_mc.append(y_arr)
        Y_mc = np.stack(Y_mc, axis=0)
        mc_mean = np.mean(Y_mc, axis=0)
        mc_std = np.std(Y_mc, axis=0, ddof=0)
        mc_time = time.perf_counter() - mc_start

        results["mc_mean"] = mc_mean
        results["mc_std"] = mc_std
        results["mc_time"] = mc_time

    return results


# ----------------------------------------------------------------------
# 6. Example main: run and plot
# ----------------------------------------------------------------------

def main():
    problem = make_double_pendulum_problem()

    # Uncertainty in initial angles [theta1, theta2]
    theta_mean = np.array([2.0, 0.0])
    theta_cov = np.diag([0.1**2, 0.1**2])

    t_span = (0.0, 10.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1001)
    N_MC = 100

    print(f"Using Gauss–Hermite order {GH_ORDER_PER_DIM} per dimension => "
          f"{GH_ORDER_PER_DIM**theta_mean.size} design points")

    # 1) Gauss–Hermite quadrature (deterministic)
    print("Running quadrature propagation...")
    sp_result = propagate_quadrature(
        problem,
        t_span,
        t_eval,
        theta_mean,
        theta_cov,
        n_mc=N_MC,
    )

    # 2) Bayesian quadrature
    print("Running Bayesian quadrature propagation...")
    bq_result = propagate_bayesian_quadrature(
        problem,
        t_span,
        t_eval,
        theta_mean,
        theta_cov,
        n_mc=0,          # reuse MC for comparison
        n_bq_samples=2000,  # MC samples for kernel expectations
        bq_lengthscale=None # let it infer from theta_cov
    )

    t = sp_result["t"]

    # Extract quadrature ("SP")
    sp_mean = sp_result["sp_mean"]
    sp_std = sp_result["sp_std"]

    # Extract BQ
    bq_mean = bq_result["bq_mean"]
    bq_std_param = bq_result["bq_std_param"]
    bq_std_total = bq_result["bq_std_total"]

    # Monte Carlo (use from one of the results; they’re the same)
    mc_mean = sp_result.get("mc_mean", None)
    mc_std = sp_result.get("mc_std", None)

    print(f"Quadrature time: {sp_result['sp_time']:.3f} s")
    print(f"BQ time:         {bq_result['bq_time']:.3f} s")
    if mc_mean is not None:
        print(f"MC time:         {sp_result['mc_time']:.3f} s")

    # Plot θ1, θ2: quadrature vs BQ vs MC
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    labels = [r"$\theta_1$", r"$\theta_2$"]

    for i in range(2):
        ax = axes[i]

        # Quadrature
        ax.plot(t, sp_mean[i], label="GH mean")
        ax.fill_between(
            t,
            sp_mean[i] - 2 * sp_std[i],
            sp_mean[i] + 2 * sp_std[i],
            alpha=0.2,
            label="GH ±2σ",
        )

        # Bayesian quadrature
        ax.plot(t, bq_mean[i], label="BQ mean")
        ax.fill_between(
            t,
            bq_mean[i] - 2 * bq_std_param[i],
            bq_mean[i] + 2 * bq_std_param[i],
            alpha=0.2,
            label="BQ ±2σ (param)",
        )
        ax.fill_between(
            t,
            bq_mean[i] - 2 * bq_std_total[i],
            bq_mean[i] + 2 * bq_std_total[i],
            alpha=0.15,
            label="BQ ±2σ (param+BQ)",
        )

        # Monte Carlo
        if mc_mean is not None:
            ax.plot(t, mc_mean[i], "--", label="MC mean")
            ax.fill_between(
                t,
                mc_mean[i] - 2 * mc_std[i],
                mc_mean[i] + 2 * mc_std[i],
                alpha=0.15,
                label="MC ±2σ",
            )

        ax.set_ylabel(f"{labels[i]} (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Double Pendulum: GH quadrature vs Bayesian Quadrature vs Monte Carlo")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
