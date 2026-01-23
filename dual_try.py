import numpy as np
import matplotlib.pyplot as plt

from all_functions import (
    OU_model,
    Simulate_Data,
    OU_kernel,
    chol_solve,
    dual_smoother,
    dual_filter,
)

if __name__ == "__main__":
    # simulate
    t, x_true, y_obs, F, Q, R = Simulate_Data(
        lamb=1.1, sigma_x=1.0, sigma_y=0.5, T=12.0, dt=0.05, rng=np.random.default_rng(0)
    )
    lamb = 1.1
    sigma_x = 1.0
    # Build OU kernel on this grid
    K = OU_kernel(t, lamb=lamb, sigma_x=sigma_x)

    # filtering and smoothing
    m_flt, P_flt = dual_filter(y_obs, K, R)
    m_smt, P_smt = dual_smoother(y_obs, K, R)

    #Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, y_obs, ".", alpha=0.6, label="observations")
    ax.plot(t, x_true, lw=1.0, label="x_true")
    ax.plot(t, m_flt, lw=2.0, label="filtered mean (dual GP)")
    ax.fill_between(t, m_flt - 2*np.sqrt(P_flt), m_flt + 2*np.sqrt(P_flt),
                    alpha=0.2, label="filtered ±2σ")
    ax.plot(t, m_smt, lw=2.0, linestyle="--", label="smoothed mean (dual GP)")
    ax.fill_between(t, m_smt - 2*np.sqrt(P_smt), m_smt + 2*np.sqrt(P_smt),
                    alpha=0.2, label="smoothed ±2σ")
    ax.set_title("OU + Bayesian Dual (Kernel) Filtering & Smoothing")
    ax.set_xlabel("time"); ax.set_ylabel("state")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
