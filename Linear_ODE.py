import numpy as np
import math
import time
import os
from scipy.integrate import solve_ivp  # LSODA
import matplotlib.pyplot as plt

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
# Helper: W2 distance between time-indexed 1D Gaussians
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

    d2 = (mean1 - mean2)**2 + (std1 - std2)**2
    D = math.sqrt(np.mean(d2))
    return D


# ============================================================
# 1. Analytic joint Gaussian (exact for this linear IVP)
# ============================================================

def analytic_joint_gaussian(t_grid, a, b, m_theta, P_theta):
    t = np.asarray(t_grid)
    if a != 0.0:
        L = np.exp(a * t)
        c = (b / a) * (np.exp(a * t) - 1.0)
    else:
        L = np.ones_like(t)
        c = b * t
    mean = L * m_theta + c
    var = (L**2) * P_theta
    return mean, var, L


# ============================================================
# 2. MC + LSODA
# ============================================================

def solve_single_theta_lsoda(theta, t_grid, a, b,
                             rtol=1e-12, atol=1e-12):
    def f(t, y):
        return a * y + b
    sol = solve_ivp(
        f, (float(t_grid[0]), float(t_grid[-1])),
        y0=[float(theta)],
        t_eval=t_grid,
        method="LSODA",
        rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"LSODA failed: {sol.message}")
    return sol.y[0]


def mc_lsoda(t_grid, a, b, m_theta, P_theta,
             n_samples, rtol=1e-12, atol=1e-12, seed=0):
    """
    Standard MC+LSODA with n_samples, used for reference and timing.
    """
    rng = np.random.default_rng(seed)
    thetas = rng.normal(loc=m_theta, scale=math.sqrt(P_theta),
                        size=n_samples)
    N_t = len(t_grid)
    Y = np.empty((n_samples, N_t))
    for i, theta in enumerate(thetas):
        Y[i] = solve_single_theta_lsoda(theta, t_grid, a, b, rtol, atol)
    mean = Y.mean(axis=0)
    var = Y.var(axis=0, ddof=1)
    return mean, var


# ============================================================
# 3. PN Kalman solver (IWP(1) + ODE pseudo-observations)
#    (a) Your original interface for y(t)
# ============================================================

def pn_kalman_path(theta, t_grid, a, b, q_c=1e-2, r_var=1e-6,
                   with_sensitivity=False):
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
    H = np.array([[-a, 1.0]])    # shape (1,2)
    R = np.array([[r_var]])      # shape (1,1)

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
        A = np.array([[1.0, h],
                      [0.0, 1.0]])
        Q = q_c * np.array([[h**3 / 3.0, h**2 / 2.0],
                            [h**2 / 2.0, h]])

        # predict
        x_pred = A @ x_mean[k - 1]
        P_pred = A @ P[k - 1] @ A.T + Q

        if with_sensitivity:
            # J_pred = A J_{k-1}  (no explicit theta in dynamics)
            J_pred = A @ J[k - 1]

        # update with ODE pseudo-observation: b = -a y + v
        z = np.array([b])   # scalar pseudo-measurement
        S = H @ P_pred @ H.T + R         # shape (1,1)
        K = P_pred @ H.T @ np.linalg.inv(S)  # shape (2,1)

        innov = z - H @ x_pred           # shape (1,)

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
        J_theta_y = J[:, 0]   # dy_mean / d theta
        return y_mean, y_var, J_theta_y
    else:
        return y_mean, y_var


# ============================================================
# 3b. Full state + sensitivities w.r.t x0 and theta
#      (used for the joint / goal covariance)
# ============================================================

def pn_kalman_state_with_sensitivities(theta, t_grid, a, b,
                                       q_c=1e-2, r_var=1e-6):
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
    d = 2   # state dimension
    p = 1   # parameter dimension (theta is scalar)

    # Measurement model: z = H x + eps, with z = b
    H = np.array([[-a, 1.0]])    # (1,2)
    R = np.array([[r_var]])      # (1,1)

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
        A = np.array([[1.0, h],
                      [0.0, 1.0]])
        Q = q_c * np.array([[h**3 / 3.0, h**2 / 2.0],
                            [h**2 / 2.0, h]])

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


# ============================================================
# 4. PN + joint Gaussian (goal covariance via J_k and Sigma_0)
# ============================================================

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
    G = np.array([[1.0],
                  [a]])

    # P0      = Cov(x_0)        = G P_theta G^T
    # P0theta = Cov(x_0, theta) = G P_theta
    P0 = P_theta * (G @ G.T)   # 2x2
    P0theta = P_theta * G      # 2x1

    Sigma0 = np.block([
        [P0,           P0theta],
        [P0theta.T,    np.array([[P_theta]])]
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

def incremental_mc_lsoda_until_tol(t_grid, a, b, m_theta, P_theta,
                                   ref_mean, ref_var, dist_tol,
                                   max_samples=50_000, batch_size=100,
                                   rtol=1e-12, atol=1e-12, seed=42):
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
        thetas = rng.normal(loc=m_theta, scale=math.sqrt(P_theta),
                            size=this_batch)

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
            dist = gaussian_w2_distance(ref_mean, ref_var,
                                        mean_est, var_est)
            print(f"  n = {n:6d}: W2(MC+LSODA_n, ref) = {dist:.4e}")
            if dist <= dist_tol:
                print(f"  -> stopping early at n = {n}, dist <= dist_tol")
                return n, mean_est.copy(), var_est.copy()

    # If we get here, we hit max_samples without meeting tol
    var_est = M2 / max(1, (n - 1))
    print(f"Reached max_samples = {max_samples} without meeting dist_tol.")
    return n, mean_est, var_est


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


# ============================================================
# 8. Timings: each method once
# ============================================================

def time_call(fn):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return t1 - t0

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

