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

from all_functions import (
    spherical_cubature_probnum as spherical_cubature,
    gauss_hermite_cubature_probnum as gauss_hermite_cubature,
    logistic_fun_probnum as logistic_fun,
    logistic_jacobian_probnum as logistic_jacobian,
    fhn_fun_probnum as fhn_fun,
    lotkavolterra_fun_probnum as lotkavolterra_fun,
    vanderpol_fun_probnum as vanderpol_fun,
    J_fhn_probnum as J_fhn,
    J_lv_probnum as J_lv,
    J_vdp_probnum as J_vdp,
    _numerical_jacobian_probnum as _numerical_jacobian,
    integrate_deterministic_probnum as integrate_deterministic,
    integrate_probnum_probnum as integrate_probnum,
    propagate_deterministic_probnum as propagate_deterministic,
    propagate_mc_probnum as propagate_mc,
    iwp1_matrices_probnum as iwp1_matrices,
    build_E0_E1_probnum as build_E0_E1,
    ek1_iwp1_goal_cov_probnum as ek1_iwp1_goal_cov,
    extract_y_stats_from_P_probnum as extract_y_stats_from_P,
    propagate_pn_iwp1_goal_probnum as propagate_pn_iwp1_goal,
    make_logistic_problem_probnum as make_logistic_problem,
    make_fhn_problem_probnum as make_fhn_problem,
    make_lv_problem_probnum as make_lv_problem,
    make_vdp_problem_probnum as make_vdp_problem,
    plot_ci_compare_probnum as plot_ci_compare,
)

problems = []

# Logistic
problems.append({
    'problem': make_logistic_problem(),
    'theta_mean': np.array([3.0]),
    'theta_cov': np.array([[0.01]]),
    't_span': (0.0, 3.0),
    't_eval': np.linspace(0.0, 3.0, 400),
    'n_mc': 600,
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

        saved_path = plot_ci_compare(
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
        saved_files.append(saved_path)
        counter += 1
    else:
        for k in range(sp_mean.shape[0]):
            mean_pn = std_pn = mean_goal = std_goal = None
            if pn_res is not None:
                mean_pn = pn_res['mean_pn'][k]
                std_pn = pn_res['std_pn'][k]
                mean_goal = pn_res['mean_goal'][k]
                std_goal = pn_res['std_goal'][k]

            saved_path = plot_ci_compare(
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
            saved_files.append(saved_path)
            counter += 1

print("Saved plot files:")
for p in saved_files:
    print("  ", p)
