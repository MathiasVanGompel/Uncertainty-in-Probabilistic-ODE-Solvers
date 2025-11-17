# Propagation_model
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import math
import warnings

# Reproducibility
rng = np.random.default_rng(0)

def spherical_cubature(mu, Sigma):
    #spherical_cubature algorithm
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

#all functions from paper
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
    # y2' = mu * (1 - y1^2) * y2 - y1
    y1, y2 = y
    return np.array([y2, mu*(1 - y1**2)*y2 - y1])

#Wrapper to integrate with LSODA
def integrate(fun, t_span, y0, args=(), t_eval=None, rtol=1e-6, atol=1e-8):
    sol = solve_ivp(fun, t_span, y0, method='LSODA', args=args, t_eval=t_eval, rtol=rtol, atol=atol)
    #if problem emerges
    if not sol.success:
        warnings.warn(f"Integration failed: {sol.message}")
    return sol.t, sol.y  # t shape (M,), y shape (d, M)

#Quadrature-based propagation
def propagate_quadrature(system, t_span, t_eval, theta_mean, theta_cov, n_mc=400):
    """
    system: dictionary defining the ODE and mapping from theta to (y0, params).
      Keys:
        'name': str
        'ode_fun': callable(t, y, *params)
        'theta_to_setup': callable(theta) -> (y0 (d,), params (tuple))
        'dim_y': int
        'dim_theta': int
    Returns:
        results dict with times, mean and std (component-wise) for quadrature (SP) and MC reference.
    """
    name = system['name']
    ode_fun = system['ode_fun']
    theta_to_setup = system['theta_to_setup']
    dim_y = system['dim_y']
    dim_theta = system['dim_theta']

    #Sigma points
    nodes, w = spherical_cubature(theta_mean, theta_cov)
    # Storage for each node's trajectory
    Y_nodes = []  # list of arrays shape (dim_y, M)
    for th in nodes:
        y0, params = theta_to_setup(th)
        t, y = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
        Y_nodes.append(y)
    Y_nodes = np.stack(Y_nodes, axis=0)  # shape (2d, dim_y, M)

    # Weighted mean across nodes
    sp_mean = np.tensordot(w, Y_nodes, axes=(0,0))  # shape (dim_y, M)

    # Weighted variance across nodes (No PN part)
    # var = sum_i w_i (y_i - mean)^2
    diffs = Y_nodes - sp_mean[None, :, :]
    sp_var = np.tensordot(w, diffs**2, axes=(0,0))   # shape (dim_y, M)
    sp_std = np.sqrt(sp_var)

    # Monte Carlo reference
    Y_mc = []
    for _ in range(n_mc):
        theta = rng.multivariate_normal(theta_mean, theta_cov)
        y0, params = theta_to_setup(theta)
        t, y = integrate(ode_fun, t_span, y0, args=params, t_eval=t_eval)
        Y_mc.append(y)
    Y_mc = np.stack(Y_mc, axis=0)  # (n_mc, dim_y, M)
    mc_mean = np.mean(Y_mc, axis=0)
    mc_std = np.std(Y_mc, axis=0, ddof=1)

    return {
        't': t,
        'sp_mean': sp_mean,
        'sp_std': sp_std,
        'mc_mean': mc_mean,
        'mc_std': mc_std,
        'name': name
    }

#All ODEs

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

# 2) FitzHugh–Nagumo: uncertain initial y(0) ~ N([0.5, 1], 0.1 I2); params fixed (a=0, b=0.08, c=0.07, d=1.25)
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

# 3) Lotka–Volterra: uncertain initial y(0) ~ N([5,5], 0.3 I2); params fixed (a=5, b=0.5, c=5, d=0.5)
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

# 4) Van der Pol: uncertain initial y(0) ~ N([5,5], 2 I2); param mu (=a in paper) fixed 0.05
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

#Run algorithms

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

results = []
for model in problems:
    res = propagate_quadrature(
        model['problem'],
        model['t_span'],
        model['t_eval'],
        model['theta_mean'],
        model['theta_cov'],
        n_mc=model['n_mc']
    )
    results.append(res)

#Plotting

outdir = "data"
os.makedirs(outdir, exist_ok=True)
saved_files = []

def plot_ci(t, mean, std, mc_mean, mc_std, title, ylabel="y", fname="plot.png"):
    fig = plt.figure()
    plt.plot(t, mean, label="proposed (SP) mean")
    plt.fill_between(t, mean - 1.96*std, mean + 1.96*std, alpha=0.3, label="proposed 95% CI")
    plt.plot(t, mc_mean, linestyle="--", label="MC mean")
    plt.fill_between(t, mc_mean - 1.96*mc_std, mc_mean + 1.96*mc_std, alpha=0.2, label="MC 95% CI")
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
    t = res['t']
    sp_mean = res['sp_mean']
    sp_std = res['sp_std']
    mc_mean = res['mc_mean']
    mc_std = res['mc_std']
    if sp_mean.shape[0] == 1:
        plot_ci(t, sp_mean[0], sp_std[0], mc_mean[0], mc_std[0], f"{name}: component 1", ylabel="y",
                fname=f"{counter:02d}_{name.replace(' ', '_')}_y.png")
        counter += 1
    else:
        for k in range(sp_mean.shape[0]):
            plot_ci(t, sp_mean[k], sp_std[k], mc_mean[k], mc_std[k], f"{name}: component {k+1}", ylabel=f"y{k+1}",
                    fname=f"{counter:02d}_{name.replace(' ', '_')}_y{k+1}.png")
            counter += 1