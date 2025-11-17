# Propagation_model_with_GaussHermite_and_Spherical_comparison
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

# Cubature / Quadrature methods

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
    # For each dimension, z = sqrt(2) * u
    z_nodes = np.sqrt(2.0) * u_grid  # (K, d)

    # Weights for expectation w.r.t. Z ~ N(0, I)
    # E[f(Z)] = (1 / π^{d/2}) Σ (prod w_1d) f(√2 * x_1d)
    w = (1.0 / (np.pi ** (d / 2.0))) * w_prod  # (K,)

    # Transform Z to theta = mu + L Z for N(mu, Sigma)
    nodes = mu + z_nodes @ L.T  # (K, d)

    return nodes, w


# ODEs and integration


# all functions from paper
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
    # if problem emerges
    if not sol.success:
        warnings.warn(f"Integration failed: {sol.message}")
    return sol.t, sol.y  # t shape (M,), y shape (d, M)



# Propagation: deterministic + Monte Carlo


def propagate_deterministic(system, t_span, t_eval, theta_mean, theta_cov,
                            quad_method="spherical", n_gh_1d=5):
    """
    Propagate uncertainty using a deterministic quadrature rule
    """
    name = system['name']
    ode_fun = system['ode_fun']
    theta_to_setup = system['theta_to_setup']

    # Choose nodes
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

    # Weighted mean and variance over nodes
    mean = np.tensordot(w, Y_nodes, axes=(0, 0))  # (dim_y, M)
    diffs = Y_nodes - mean[None, :, :]
    var = np.tensordot(w, diffs**2, axes=(0, 0))  # (dim_y, M)
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


def propagate_mc(system, t_span, t_eval, theta_mean, theta_cov, n_mc=400):
    """
    Monte Carlo reference propagation.
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



# All ODEs


# 1) Logistic: y' = a*y*(1 - y/b), a fixed, b ~ N(3, 0.01), y0 = 0.05
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
        'theta_to_setup': theta_to_setup,
        'dim_y': 1,
        'dim_theta': 1
    }

# 2) FitzHugh–Nagumo: uncertain initial y(0) ~ N([0.5, 1], 0.1 I2); params fixed
#    (a=0, b=0.08, c=0.07, d=1.25)
def make_fhn_problem():
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

# 3) Lotka–Volterra: uncertain initial y(0) ~ N([5,5], 0.3 I2);
#    params fixed (a=5, b=0.5, c=5, d=0.5)
def make_lv_problem():
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

# 4) Van der Pol: uncertain initial y(0) ~ N([5,5], 2 I2); param mu fixed 0.05
def make_vdp_problem():
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


# ===============
# Run algorithms
# ===============

problems = []

# Logistic settings
problems.append({
    'problem': make_logistic_problem(),
    'theta_mean': np.array([3.0]),
    'theta_cov': np.array([[0.01]]),  # variance 0.01
    't_span': (0.0, 3.0),
    't_eval': np.linspace(0.0, 3.0, 400),
    'n_mc': 600
})

# FHN settings
problems.append({
    'problem': make_fhn_problem(),
    'theta_mean': np.array([0.5, 1.0]),
    'theta_cov': 0.1 * np.eye(2),
    't_span': (0.0, 7.0),
    't_eval': np.linspace(0.0, 7.0, 700),
    'n_mc': 500
})

# Lotka–Volterra settings
problems.append({
    'problem': make_lv_problem(),
    'theta_mean': np.array([5.0, 5.0]),
    'theta_cov': 0.3 * np.eye(2),
    't_span': (0.0, 2.0),
    't_eval': np.linspace(0.0, 2.0, 400),
    'n_mc': 600
})

# Van der Pol settings
problems.append({
    'problem': make_vdp_problem(),
    'theta_mean': np.array([5.0, 5.0]),
    'theta_cov': 2.0 * np.eye(2),
    't_span': (0.0, 10.0),
    't_eval': np.linspace(0.0, 10.0, 1200),
    'n_mc': 500
})


# Storage for all results
results = []
timings = []

# For each problem, run spherical, Gauss–Hermite, and MC
for model in problems:
    problem = model['problem']
    theta_mean = model['theta_mean']
    theta_cov = model['theta_cov']
    t_span = model['t_span']
    t_eval = model['t_eval']
    n_mc = model['n_mc']

    # Spherical cubature
    res_sp = propagate_deterministic(
        problem, t_span, t_eval, theta_mean, theta_cov,
        quad_method="spherical"
    )
    # Gauss–Hermite
    res_gh = propagate_deterministic(
        problem, t_span, t_eval, theta_mean, theta_cov,
        quad_method="gh", n_gh_1d=5
    )
    # Monte Carlo reference
    res_mc = propagate_mc(
        problem, t_span, t_eval, theta_mean, theta_cov,
        n_mc=n_mc
    )

    results.append({
        'name': problem['name'],
        'spherical': res_sp,
        'gh': res_gh,
        'mc': res_mc
    })

    timings.append({
        'name': problem['name'],
        'time_spherical': res_sp['time'],
        'time_gh': res_gh['time'],
        'time_mc': res_mc['time']
    })

# Print timing comparison
for tm in timings:
    name = tm['name']
    t_sp = tm['time_spherical']
    t_gh = tm['time_gh']
    t_mc = tm['time_mc']
    print(f"{name}:")
    print(f"  Spherical time = {t_sp:.3f} s")
    print(f"  Gauss–Hermite time = {t_gh:.3f} s")
    print(f"  MC time = {t_mc:.3f} s")
    print(f"  MC / Spherical = {t_mc / t_sp:.2f}x,  MC / GH = {t_mc / t_gh:.2f}x\n")



# Plotting


outdir = "data"
os.makedirs(outdir, exist_ok=True)
saved_files = []

def plot_ci_compare(t,
                    mean_sp, std_sp,
                    mean_gh, std_gh,
                    mean_mc, std_mc,
                    title, ylabel="y", fname="plot.png"):
    fig = plt.figure()
    # Spherical
    plt.plot(t, mean_sp, label="Spherical mean")
    plt.fill_between(t, mean_sp - 1.96*std_sp, mean_sp + 1.96*std_sp,
                     alpha=0.25, label="Spherical 95% CI")

    # Gauss–Hermite
    plt.plot(t, mean_gh, linestyle="--", label="Gauss–Hermite mean")
    plt.fill_between(t, mean_gh - 1.96*std_gh, mean_gh + 1.96*std_gh,
                     alpha=0.25, label="GH 95% CI")

    # Monte Carlo
    plt.plot(t, mean_mc, linestyle=":", label="MC mean")
    plt.fill_between(t, mean_mc - 1.96*std_mc, mean_mc + 1.96*std_mc,
                     alpha=0.2, label="MC 95% CI")

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
    t = res['spherical']['t']  # common t

    sp_mean = res['spherical']['mean']
    sp_std = res['spherical']['std']

    gh_mean = res['gh']['mean']
    gh_std = res['gh']['std']

    mc_mean = res['mc']['mean']
    mc_std = res['mc']['std']

    if sp_mean.shape[0] == 1:
        plot_ci_compare(
            t,
            sp_mean[0], sp_std[0],
            gh_mean[0], gh_std[0],
            mc_mean[0], mc_std[0],
            f"{name}: component 1",
            ylabel="y",
            fname=f"{counter:02d}_{name.replace(' ', '_')}_y.png"
        )
        counter += 1
    else:
        for k in range(sp_mean.shape[0]):
            plot_ci_compare(
                t,
                sp_mean[k], sp_std[k],
                gh_mean[k], gh_std[k],
                mc_mean[k], mc_std[k],
                f"{name}: component {k+1}",
                ylabel=f"y{k+1}",
                fname=f"{counter:02d}_{name.replace(' ', '_')}_y{k+1}.png"
            )
            counter += 1

print("Saved plot files:")
for p in saved_files:
    print("  ", p)
