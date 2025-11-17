import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# dx_t = -λ x_t dt + sqrt(2 λ σ_x^2) dW_t

def OU_model(lamb, sigma_stat, dt):
    F = np.exp(-lamb * dt)
    Q = sigma_stat**2 * (1.0 - np.exp(-2.0 * lamb * dt))
    return F, Q

def Simulate_Data(lamb=1.2, sigma_x=1.0, sigma_y=0.2, T=10.0, dt=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(0.0, T, dt)
    n = len(t)
    F, Q = OU_model(lamb, sigma_x, dt)  # stationary OU discretization
    R = sigma_y**2
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = rng.normal(0.0, sigma_x)
    y[0] = x[0] + rng.normal(0.0, np.sqrt(R))
    for i in range(1, n):
        x[i] = F * x[i - 1] + rng.normal(0.0, np.sqrt(Q))
        y[i] = x[i] + rng.normal(0.0, np.sqrt(R))

    return t, x, y, F, Q, R

# Dual formulation
# K_ij = sigma_x^2 * exp(-lamb * |t_i - t_j|)
# Smoother
# Filter

def OU_kernel(t, lamb, sigma_x):
    """Stationary OU covariance kernel on a grid t."""
    dt = np.abs(t[:, None] - t[None, :]) # Make it into a matrix
    return (sigma_x**2) * np.exp(-lamb * dt)

def chol_solve(L, b):
    """Solve (L L^T) x = b for x, given lower-triangular L (Cholesky)."""
    #Lz = b
    z = np.linalg.solve(L, b)
    #L^T x = z
    x = np.linalg.solve(L.T, z)
    return x

def dual_smoother(y, K, R, jitter=1e-9):
    #smoother algorithm
    n = len(y)
    A = K + (R + jitter) * np.eye(n)
    # Cholesky factor of A
    L = np.linalg.cholesky(A)
    # alpha
    alpha = chol_solve(L, y)
    m = K @ alpha  # posterior mean
    # For diagonal of posterior covariance: diag(K - K A^{-1} K)
    # Compute A^{-1} K
    AinvK = chol_solve(L, K)  # (n x n)
    diag_term = np.einsum("ij,ji->i", K, AinvK)  # diag(K @ A^{-1} @ K)
    P_diag = np.diag(K) - diag_term
    return m, P_diag

def dual_filter(y, K, R, jitter=1e-9):
    #filtering algorithm
    n = len(y)
    m_flt = np.zeros(n)
    P_flt = np.zeros(n)

    # Step k=1
    A11 = K[0, 0] + R + jitter
    L = np.array([[np.sqrt(A11)]])  # 1x1
    # alpha_1
    alpha = chol_solve(L, y[:1])  # shape (1,)
    # mean at t_0
    m_flt[0] = K[0, 0] * alpha[0]  # since K[0,:1] @ alpha
    # variance at t_0
    s = chol_solve(L, K[:1, 0])   # A_1^{-1} K[:,0] (first col)
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
            [L,                  np.zeros((k, 1))],
            [w[None, :],         np.array([[l_kk]])]
        ])

        # Update alpha by solving (L L^T) alpha = y[:k+1]
        alpha = chol_solve(L, y[:k+1])

        # Posterior mean at t_k using only y[:k+1]
        m_flt[k] = K[k, :k+1] @ alpha

        # Posterior variance at t_k: diag element
        s = chol_solve(L, K[:k+1, k])  # A_k^{-1} K[:k+1, k]
        P_flt[k] = K[k, k] - K[k, :k+1] @ s

    return m_flt, P_flt

if __name__ == "__main__":
    # simulate
    t, x_true, y_obs, F, Q, R = Simulate_Data(
        lamb=1.1, sigma_x=1.0, sigma_y=0, T=12.0, dt=0.05, rng=np.random.default_rng(0)
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
