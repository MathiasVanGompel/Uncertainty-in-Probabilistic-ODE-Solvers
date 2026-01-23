import numpy as np
import math
import time
import os
from scipy.integrate import solve_ivp  # LSODA
import matplotlib.pyplot as plt

from all_functions import (
    gaussian_w2_distance,
    analytic_joint_gaussian,
    solve_single_theta_lsoda,
    mc_lsoda,
    pn_kalman_path,
    pn_kalman_state_with_sensitivities,
    pn_joint_gaussian,
    incremental_mc_lsoda_until_tol,
    time_call,
)

# ============================================================
# Problem: dy/dt = a y + b, y(0) = theta ~ N(m_theta, P_theta)
# ============================================================

a = 1.0
b = 0.0
m_theta = 1.0
P_theta = 0.01

T = 3.0
dt = 0.01
t_grid = np.arange(0.0, T + 1e-12, dt)

# PN hyperparameters
q_c = 1e-2
r_var = 1e-6

# LSODA reference
N_REF = 500_000  # big reference run
REF_FILE = f"lsoda_reference_N{N_REF}.npz"  # cache file on disk

z_95 = 1.96  # for 95% CI

# ============================================================
# 5. Build or load LSODA reference (N_REF = 500,000)
# ============================================================

if os.path.exists(REF_FILE):
    print(f"Loading MC+LSODA reference from '{REF_FILE}'...")
    data = np.load(REF_FILE)
    ref_mean = data["ref_mean"]
    ref_var = data["ref_var"]
    print("Loaded reference from file.\n")
else:
    print(f"Computing MC+LSODA reference with N_REF = {N_REF} samples...")
    ref_mean, ref_var = mc_lsoda(
        t_grid, a, b, m_theta, P_theta,
        n_samples=N_REF, rtol=1e-12, atol=1e-12, seed=123
    )
    # Save for future runs
    np.savez(REF_FILE, ref_mean=ref_mean, ref_var=ref_var)
    print(f"Done computing LSODA reference. Saved to '{REF_FILE}'.\n")


# ============================================================
# 6. Distances of baseline methods to reference
# ============================================================

# Analytic joint Gaussian
analytic_mean, analytic_var, L_exact = analytic_joint_gaussian(
    t_grid, a, b, m_theta, P_theta
)
dist_analytic = gaussian_w2_distance(ref_mean, ref_var,
                                     analytic_mean, analytic_var)
print(f"Analytic joint Gaussian:  W2 distance to ref = {dist_analytic:.4e}")

# PN-only (theta fixed to m_theta, no parameter uncertainty)
pn_mean, pn_var = pn_kalman_path(m_theta, t_grid, a, b, q_c, r_var,
                                 with_sensitivity=False)
dist_pn = gaussian_w2_distance(ref_mean, ref_var, pn_mean, pn_var)
print(f"PN-only:                  W2 distance to ref = {dist_pn:.4e}")

# PN + joint Gaussian (using full J_k, Sigma_0)
pn_joint_mean, pn_joint_var = pn_joint_gaussian(
    t_grid, a, b, m_theta, P_theta, q_c, r_var
)
dist_pn_joint = gaussian_w2_distance(ref_mean, ref_var,
                                     pn_joint_mean, pn_joint_var)
print(f"PN + joint Gaussian:      W2 distance to ref = {dist_pn_joint:.4e}\n")

# Use the best (smallest) of these as DIST_TOL
DIST_TOL = min(dist_pn, dist_pn_joint)
print(f"Using DIST_TOL = min(...) = {DIST_TOL:.4e}\n")


# ============================================================
# 7. Incremental MC+LSODA calibration
#    Reuse all samples, check every 'batch_size' samples
# ============================================================

print("Incremental MC+LSODA calibration (reusing all samples):")
best_N_mc_lsoda, mc_lsoda_mean_bestN, mc_lsoda_var_bestN = \
    incremental_mc_lsoda_until_tol(
        t_grid, a, b, m_theta, P_theta,
        ref_mean, ref_var, DIST_TOL,
        max_samples=50_000,   # you can change this cap
        batch_size=100,       # check every 100 samples
        rtol=1e-12, atol=1e-12,
        seed=42
    )
print(f"\nChosen N_MC_LSODA (incremental) = {best_N_mc_lsoda}.\n")


time_analytic = time_call(lambda: analytic_joint_gaussian(t_grid, a, b, m_theta, P_theta))
time_pn = time_call(lambda: pn_kalman_path(m_theta, t_grid, a, b, q_c, r_var,
                                           with_sensitivity=False))
time_pn_joint = time_call(lambda: pn_joint_gaussian(t_grid, a, b, m_theta, P_theta, q_c, r_var))
time_mc_lsoda_bestN = time_call(lambda: mc_lsoda(t_grid, a, b, m_theta, P_theta,
                                                 n_samples=best_N_mc_lsoda,
                                                 rtol=1e-12, atol=1e-12, seed=42))

print("=== Timings (single run each) ===")
print(f"Analytic joint Gaussian:      {time_analytic:.4e} s")
print(f"PN-only:                      {time_pn:.4e} s")
print(f"PN + joint Gaussian:          {time_pn_joint:.4e} s")
print(f"MC + LSODA (N={best_N_mc_lsoda}): {time_mc_lsoda_bestN:.4e} s")
print("=================================")


# ============================================================
# 9. Plot: mean + 95% CI bands
# ============================================================

ref_std = np.sqrt(ref_var)
ref_lower = ref_mean - z_95 * ref_std
ref_upper = ref_mean + z_95 * ref_std

pn_joint_std = np.sqrt(pn_joint_var)
pn_joint_lower = pn_joint_mean - z_95 * pn_joint_std
pn_joint_upper = pn_joint_mean + z_95 * pn_joint_std

mc_lsoda_std_bestN = np.sqrt(mc_lsoda_var_bestN)
mc_lsoda_lower_bestN = mc_lsoda_mean_bestN - z_95 * mc_lsoda_std_bestN
mc_lsoda_upper_bestN = mc_lsoda_mean_bestN + z_95 * mc_lsoda_std_bestN

plt.figure(figsize=(8, 5))

# LSODA reference band
plt.fill_between(t_grid, ref_lower, ref_upper,
                 alpha=0.2, label=f"MC+LSODA (N={N_REF}) 95% CI")
plt.plot(t_grid, ref_mean,
         label="MC+LSODA mean (ref)", linewidth=2)

# PN + joint Gaussian band
plt.fill_between(t_grid, pn_joint_lower, pn_joint_upper,
                 alpha=0.2, label="PN + joint Gaussian 95% CI")
plt.plot(t_grid, pn_joint_mean,
         label="PN + joint Gaussian mean", linewidth=2, linestyle="--")

# MC + LSODA band (incremental calibrated N)
plt.fill_between(t_grid, mc_lsoda_lower_bestN, mc_lsoda_upper_bestN,
                 alpha=0.2, label=f"MC+LSODA (N={best_N_mc_lsoda}) 95% CI")
plt.plot(t_grid, mc_lsoda_mean_bestN,
         label=f"MC+LSODA mean (N={best_N_mc_lsoda})", linewidth=1.5, linestyle=":")

plt.xlabel("t")
plt.ylabel("$y(t)$")
plt.title("Linear IVP: mean and 95% CI bands\n"
          "MC+LSODA ref vs PN+joint vs MC+LSODA")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 10. Plot: difference in covariance over time
#      PN + joint Gaussian vs MC+LSODA reference
# ============================================================

cov_diff = pn_joint_var - ref_var
abs_cov_diff = np.abs(cov_diff)
big_diff = np.abs(pn_var - ref_var)

# If you also want a separate plot for the absolute difference, uncomment:
plt.figure(figsize=(8, 4))
plt.plot(t_grid, abs_cov_diff, label=r"$|\mathrm{Var}_{\text{PN+joint}} - \mathrm{Var}_{\text{ref}}|$")
plt.plot(t_grid, big_diff, label=r"$|\mathrm{Var}_{\text{PN}} - \mathrm{Var}_{\text{ref}}|$")
plt.xlabel("t")
plt.ylabel("Absolute variance difference")
plt.title("Absolute difference in variance over time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(t_grid, ref_var,
         label=f"MC+LSODA variance (ref, N={N_REF})",
         linewidth=2)
plt.plot(t_grid, pn_joint_var,
         label="PN + joint Gaussian variance",
         linestyle="--", linewidth=2)
plt.plot(t_grid, mc_lsoda_var_bestN,
         label=f"MC+LSODA variance (N={best_N_mc_lsoda})",
         linestyle=":", linewidth=1.5)

plt.xlabel("t")
plt.ylabel(r"$\mathrm{Var}[y(t)]$")
plt.title("Variance over time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
