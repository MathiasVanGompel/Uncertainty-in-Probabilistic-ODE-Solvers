
import math
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy import dtype, ndarray

from scipy.linalg import solve_triangular as _scipy_solve_triangular
from scipy.integrate import solve_ivp as _scipy_solve_ivp

from models import AdaptiveStats, Array, PropResult, ODEProblem

def sym(A: Array) -> Array:
    return 0.5 * (A + A.T)

def chol_spd(A: Array, jitter: float = 1e-12, max_tries: int = 8) -> Array:
    A = np.asarray(A, dtype=float)
    I = np.eye(A.shape[0], dtype=float)
    lam = 0.0
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(sym(A + (lam + jitter) * I))
        except np.linalg.LinAlgError:
            lam = 10.0 * (lam + jitter) if lam > 0 else jitter
    w, V = np.linalg.eigh(sym(A))
    w = np.maximum(w, jitter)
    return np.linalg.cholesky(V @ np.diag(w) @ V.T)

def solve_triangular(L: Array, B: Array, *, lower: bool) -> Array:
    return _scipy_solve_triangular(L, B, lower=lower, check_finite=False)

def kron_eye(d: int, M: Array) -> Array:
    return np.kron(np.eye(d, dtype=float), M)


def integrate_reference(
    f: Callable[[float, Array, Optional[Array]], Array],
    *,
    t_span: Tuple[float, float],
    y0: Array,
    theta: Optional[Array],
    t_eval: Array,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> ndarray[tuple[Any, ...], dtype[Any]] | None:
    """Deterministic reference integrator used for Monte Carlo.
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float).reshape(-1)

    def rhs(t: float, y: Array) -> Array:
        return np.asarray(f(float(t), np.asarray(y, dtype=float).reshape(-1), theta), dtype=float).reshape(-1)
    sol = _scipy_solve_ivp(
        rhs,
        (t0, t1),
        y0,
        t_eval=t_eval,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
    )
    Y = np.asarray(sol.y, dtype=float).T  # (d, T) -> (T, d)
    return Y

class IWPSSM:
    """Integrated Wiener Process prior of order q (state dim q+1 per ODE component)."""

    def __init__(self, q: int):
        self.q = int(q)
        self.d = self.q + 1

    def construct_A(self, h: float) -> Array:
        d = self.d
        A = np.zeros((d, d), dtype=float)
        for i in range(d):
            for j in range(d):
                if i <= j:
                    A[i, j] = (h ** (j - i)) / math.factorial(j - i)
        return A

    def construct_Q(self, h: float) -> Array:
        d = self.d
        Q = np.zeros((d, d), dtype=float)
        q = self.q
        for i in range(d):
            for j in range(d):
                exponent = 2 * q + 1 - i - j
                denom = exponent * math.factorial(q - i) * math.factorial(q - j)
                Q[i, j] = (h ** exponent) / denom
        return Q

    def construct_T(self, h: float) -> Array:
        """Diagonal preconditioner T(h)."""
        d = self.d
        T = np.zeros((d, d), dtype=float)
        q = self.q
        for i in range(d):
            T[i, i] = math.sqrt(h) * (h ** (q - i)) / math.factorial(q - i)
        return T

    def construct_T_inv(self, h: float) -> Array:
        d = self.d
        Tinv = np.zeros((d, d), dtype=float)
        q = self.q
        for i in range(d):
            Tinv[i, i] = 1.0 / (math.sqrt(h) * (h ** (q - i)) / math.factorial(q - i))
        return Tinv

class AdaptiveIWP_EKS1_Prop_Sqrt:
    """
    Adaptive EKF1/EKS1 ODE filter with:
      - step-size preconditioning for IWP
      - square-root (Cholesky) predict/update
      - Jacobian recursion for prop covariance

    Measurement: z(t) = y'(t) - f(t, y(t), theta) = 0
    """

    def __init__(
        self,
        f: Callable[[float, Array, Optional[Array]], Array],
        Jf: Callable[[float, Array, Optional[Array]], Array],
        *,
        dtheta: Optional[Callable[[float, Array, Optional[Array]], Array]] = None,
        q: int = 3,
        use_smoother: bool = True,
        atol: float = 1e-9,
        rtol: float = 1e-5,
        rho: float = 0.7,
        eta_min: float = 0.0001,
        eta_max: float = 1000.0,
        diffusion_init: float = 1.0,
        diffusion_floor: float = 1e-16,
        diffusion_ceiling: float = 1e16,
        R: float = 1e-12,
        max_reject: int = 2_000_000,
        max_steps: int = 20_000_000,
        init_deriv_var: float = 1e-2,
        precondition: bool = True,
    ):

        self.f = f
        self.Jf = Jf
        self.dtheta = dtheta

        self.q = int(q)
        self.d_temp = self.q + 1
        self.use_smoother = bool(use_smoother)
        self.precondition = bool(precondition)

        self.atol = float(atol)
        self.rtol = float(rtol)
        self.rho = float(rho)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)

        self.diffusion_init = float(diffusion_init)
        self.diffusion = float(diffusion_init)
        self.diff_floor = float(diffusion_floor)
        self.diff_ceil = float(diffusion_ceiling)

        self.R = float(R)
        self.max_reject = int(max_reject)
        self.max_steps = int(max_steps)
        self.init_deriv_var = float(init_deriv_var)

        self.iwp = IWPSSM(self.q)

        # constant preconditioned matrices for h=1:
        self.A_pre, self.Q_pre = self.preconditioned_constants()
        self.chol_Q_pre = chol_spd(self.Q_pre, jitter=1e-15, max_tries=8)

    def preconditioned_constants(self) -> Tuple[Array, Array]:
        """
        Preconditioned matrices A~, Q~ are step-size-independent for IWP:
          A~ = T^{-1}(h) A(h) T(h), Q~ = T^{-1}(h) Q(h) T^{-T}(h)
        so we can compute them at h=1.
        """
        h = 1.0
        A = self.iwp.construct_A(h)
        Q = self.iwp.construct_Q(h)
        T = self.iwp.construct_T(h)
        Tinv = self.iwp.construct_T_inv(h)
        Atil = Tinv @ A @ T
        Qtil = Tinv @ Q @ Tinv.T
        return Atil, sym(Qtil)

    @staticmethod
    def E0(d_ode: int, d_temp: int) -> Array:
        D = d_ode * d_temp
        E0 = np.zeros((d_ode, D), dtype=float)
        for i in range(d_ode):
            E0[i, i * d_temp + 0] = 1.0
        return E0

    @staticmethod
    def E1(d_ode: int, d_temp: int) -> Array:
        D = d_ode * d_temp
        E1 = np.zeros((d_ode, D), dtype=float)
        for i in range(d_ode):
            E1[i, i * d_temp + 1] = 1.0
        return E1

    def infer_d_ode(self, m: Array) -> int:
        m = np.asarray(m, dtype=float).reshape(-1)
        return int(m.size // self.d_temp)

    def T_full(self, d_ode: int, h: float) -> Array:
        return kron_eye(d_ode, self.iwp.construct_T(h))

    def Tinv_full(self, d_ode: int, h: float) -> Array:
        return kron_eye(d_ode, self.iwp.construct_T_inv(h))

    def innovation_H_D(
        self, t: float, m_can_pred: Array, d_ode: int, theta: Optional[Array]
    ) -> Tuple[Array, Array, Optional[Array]]:
        """
        Measurement:
          z = E1 x - f(t, E0 x, theta) = 0
        Linearization:
          H = E1 - Jf E0
          D = - df/dtheta
        """
        d_temp = self.d_temp
        m_can_pred = np.asarray(m_can_pred, dtype=float).reshape(-1)

        y = m_can_pred[0::d_temp]      # E0 x
        ydot = m_can_pred[1::d_temp]   # E1 x

        fval = np.asarray(self.f(t, y, theta), dtype=float).reshape(-1)
        zhat = (ydot - fval).reshape(d_ode, 1)

        J = np.asarray(self.Jf(t, y, theta), dtype=float)

        # H in canonical coordinates: E1 - Jf E0
        H = np.zeros((d_ode, d_ode * d_temp), dtype=float)
        # Fill via blocks: for each component i, columns i*d_temp:(i+1)*d_temp
        for i in range(d_ode):
            base = i * d_temp
            H[i, base + 1] = 1.0
            # -Jf row i multiplies y components (positions base of each ode dim)
            for j in range(d_ode):
                H[i, j * d_temp + 0] -= J[i, j]

        Dth = None
        if self.dtheta is not None and theta is not None:
            G = np.asarray(self.dtheta(t, y, theta), dtype=float)
            Dth = -G  # z = ydot - f => ∂z/∂theta = -∂f/∂theta

        return zhat, H, Dth

    def init_state(
        self,
        t0: float,
        y0: Array,
        theta: Optional[Array],
        d_ode: int,
    ) -> Tuple[Array, Array]:
        """
        Initialize mean and Cholesky factor.

        For mean:
          x0[0] = y0
          x0[1] = f(t0, y0, theta)
        For covariance:
          small variance init_deriv_var on derivative components;
          y0 component variance is set to 0 here (since conditional run).
        """
        d_temp = self.d_temp
        D = d_ode * d_temp

        m = np.zeros((D, 1), dtype=float)
        y0 = np.asarray(y0, dtype=float).reshape(-1)

        # Fill y
        for i in range(d_ode):
            m[i * d_temp + 0, 0] = y0[i]

        # Fill first derivative with f
        f0 = np.asarray(self.f(t0, y0, theta), dtype=float).reshape(-1)
        for i in range(d_ode):
            m[i * d_temp + 1, 0] = f0[i]

        # covariance (canonical)
        P = (self.init_deriv_var) * np.eye(D, dtype=float)
        # conditional on y0: set y component variance to zero
        for i in range(d_ode):
            P[i * d_temp + 0, i * d_temp + 0] = 0.0
        cholP = chol_spd(P, jitter=1e-18, max_tries=8)
        return m, cholP

    def sqrt_predict(
        self,
        m_til: Array,
        cholP_til: Array,
        *,
        d_ode: int,
        diffusion: float,
    ) -> Tuple[Array, Array]:
        """Square-root prediction in preconditioned coordinates."""
        Atil_full = kron_eye(d_ode, self.A_pre)

        # process noise chol
        cholQ_full = math.sqrt(float(diffusion)) * kron_eye(d_ode, self.chol_Q_pre)

        m_pred = Atil_full @ m_til

        # QR on stacked matrix (n + n, n)
        # stack = [A L, Lq]^T
        AL = Atil_full @ cholP_til
        stack = np.hstack([AL, cholQ_full]).T
        _, R = np.linalg.qr(stack, mode="reduced")
        cholP_pred = R.T
        return m_pred, cholP_pred

    def sqrt_update(
        self,
        m_til_pred: Array,
        cholP_til_pred: Array,
        *,
        zhat: Array,
        H_til: Array,
        Rm: Array,
    ) -> Tuple[Array, Array, Array]:
        """
        Square-root update in preconditioned coordinates.
        """
        n = cholP_til_pred.shape[0]
        m = zhat.shape[0]

        # chol_R
        cholR = chol_spd(Rm, jitter=0.0, max_tries=2)

        # Compute chol_S via QR of [H L, cholR]
        HL = H_til @ cholP_til_pred                 # (m, n)
        stackS = np.hstack([HL, cholR]).T           # (n+m, m)
        _, R = np.linalg.qr(stackS, mode="reduced")
        cholS = R.T                                 # (m, m), lower

        # Kalman gain: K = P H^T S^{-1} using chol factors
        # PHt = L L^T H^T = L (L^T H^T)
        PHt = cholP_til_pred @ (cholP_til_pred.T @ H_til.T)  # (n, m)

        tmp = solve_triangular(cholS, PHt.T, lower=True)
        tmp = solve_triangular(cholS.T, tmp, lower=False)
        K = tmp.T  # (n, m)

        # z=0 => innovation = -zhat
        m_new = m_til_pred - K @ zhat

        # Updated chol(P) via QR of [(I-KH)L, K cholR]
        I = np.eye(n, dtype=float)
        A = (I - K @ H_til) @ cholP_til_pred
        B = K @ cholR
        stackP = np.hstack([A, B]).T
        _, R2 = np.linalg.qr(stackP, mode="reduced")
        cholP_new = R2.T

        return m_new, cholP_new, K

    def trial_step_sqrt(
        self,
        t_new: float,
        h: float,
        m_can: Array,
        cholP_can: Array,
        theta: Optional[Array],
        *,
        d_ode: int,
    ) -> Tuple[Array, Array, Array, Array, Array, Array, Optional[Array], float]:
        """
        One trial step (predict+update), returning:
          m_can_new, cholP_can_new,
          m_can_pred, cholP_can_pred,
          K_can, H_can, Dth, kappa2_hat
        """
        d_temp = self.d_temp
        Dstate = d_ode * d_temp

        if self.precondition:
            T = self.T_full(d_ode, h)
            Tinv = self.Tinv_full(d_ode, h)
        else:
            T = np.eye(Dstate, dtype=float)
            Tinv = np.eye(Dstate, dtype=float)

        # transform to preconditioned coords
        m_til = Tinv @ m_can
        cholP_til = Tinv @ cholP_can

        # predict (preconditioned)
        m_til_pred, cholP_til_pred = self.sqrt_predict(m_til, cholP_til, d_ode=d_ode, diffusion=self.diffusion)

        # back to canonical for linearization
        m_can_pred = T @ m_til_pred
        cholP_can_pred = T @ cholP_til_pred

        # measurement linearization (canonical)
        zhat, H_can, Dth = self.innovation_H_D(t_new, m_can_pred, d_ode, theta)

        # update in preconditioned coords: H_til = H_can @ T
        H_til = H_can @ T
        Rm = self.R * np.eye(d_ode, dtype=float)

        m_til_new, cholP_til_new, K_til = self.sqrt_update(
            m_til_pred, cholP_til_pred, zhat=zhat, H_til=H_til, Rm=Rm
        )

        # back to canonical
        m_can_new = T @ m_til_new
        cholP_can_new = T @ cholP_til_new

        # canonical gain: x_can = T x_til -> K_can = T K_til
        K_can = T @ K_til

        # diffusion calibration
        # Use canonical base S_base = H_can Q(h;diff=1) H_can^T
        Q_loc = kron_eye(d_ode, self.iwp.construct_Q(h))
        S_base = sym(H_can @ Q_loc @ H_can.T)
        try:
            val = zhat.T @ np.linalg.solve(S_base + 1e-18 * np.eye(d_ode), zhat)

            # Extract scalar safely
            kappa2_hat = float(val.squeeze() / d_ode)

        except np.linalg.LinAlgError:
            kappa2_hat = float(self.diffusion)

        kappa2_hat = float(np.clip(kappa2_hat, self.diff_floor, self.diff_ceil))

        return (
            m_can_new, cholP_can_new,
            m_can_pred, cholP_can_pred,
            K_can, H_can, Dth,
            kappa2_hat,
        )

    def solve(
        self,
        t_span: Tuple[float, float],
        y0_mean: Array,
        *,
        theta_mean: Optional[Array] = None,
        Sigma_inputs: Optional[Array] = None,
        p_y0: Optional[Array] = None,
        p_theta: Optional[Array] = None,
        cross_y0_theta: Optional[Array] = None,
        h0: float = 1e-2,
        h_min: float = 1e-12,
        h_max: float = 1.0,
        t_eval: Optional[Array] = None,
    ) -> PropResult:
        """
        Solve IVP on [t0,t1] and return (mean, cond cov, prop cov) on t_eval.
        """

        self.diffusion = float(self.diffusion_init)

        t0, t1 = float(t_span[0]), float(t_span[1])

        y0_mean = np.asarray(y0_mean, dtype=float).reshape(-1)
        d_ode = int(y0_mean.size)
        if theta_mean is not None:
            theta_mean = np.asarray(theta_mean, dtype=float).reshape(-1)

        # Build Sigma_inputs if not provided.
        if Sigma_inputs is None:
            Py0 = np.zeros((d_ode, d_ode), dtype=float) if p_y0 is None else np.asarray(p_y0, dtype=float)

            p_dim = 0 if theta_mean is None else int(theta_mean.size)
            Pth = np.zeros((p_dim, p_dim), dtype=float) if p_theta is None else np.asarray(p_theta, dtype=float)

            C = np.zeros((d_ode, p_dim), dtype=float) if cross_y0_theta is None else np.asarray(cross_y0_theta, dtype=float)

            Sigma_inputs = np.block([[Py0, C], [C.T, Pth]]) if p_dim > 0 else Py0
        else:
            Sigma_inputs = np.asarray(Sigma_inputs, dtype=float)

        p_dim = 0 if theta_mean is None else int(theta_mean.size)
        in_dim = d_ode + p_dim

        # evaluation grid
        if t_eval is None:
            t_eval = np.array([t0, t1], dtype=float)
        else:
            t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

        # init state
        m, cholP = self.init_state(t0, y0_mean, theta_mean, d_ode)

        # Jacobian recursion init:
        Dstate = d_ode * self.d_temp
        J = np.zeros((Dstate, in_dim), dtype=float)
        # y0 influences y component directly at t0
        for i in range(d_ode):
            J[i * self.d_temp + 0, i] = 1.0

        # storage
        ts: List[float] = [t0]
        m_pred_list: List[Array] = [m.copy()]
        cholP_pred_list: List[Array] = [cholP.copy()]
        m_filt_list: List[Array] = [m.copy()]
        cholP_filt_list: List[Array] = [cholP.copy()]
        dt_list: List[float] = [0.0]
        K_list: List[Array] = [np.zeros((Dstate, d_ode), dtype=float)]
        H_list: List[Array] = [np.zeros((d_ode, Dstate), dtype=float)]
        Dth_list: List[Optional[Array]] = [None]
        J_filt_list: List[Array] = [J.copy()]

        accepted = 0
        rejected = 0
        diffusion_hist: List[float] = [self.diffusion]
        min_h_seen = float("inf")
        max_h_seen = 0.0

        t = t0
        h = float(h0)
        step_budget = self.max_steps
        reject_budget = self.max_reject

        eval_idx = 1  # next index in t_eval to hit

        E0 = self.E0(d_ode, self.d_temp)

        # output arrays on t_eval (subset of accepted grid)
        out_t = [t0]
        out_mean = [ (E0 @ m).reshape(-1) ]
        # conditional covariance of y(t) = E0 P E0^T
        P0 = cholP @ cholP.T
        out_cov_cond = [ E0 @ P0 @ E0.T ]
        out_cov_prop = [ E0 @ (P0 + J @ Sigma_inputs @ J.T) @ E0.T ]

        # adaptive loop
        while t < t1 - 1e-15:
            if step_budget <= 0:
                raise RuntimeError("Exceeded max_steps.")
            step_budget -= 1

            t_target = float(t_eval[eval_idx]) if eval_idx < len(t_eval) else t1
            if t + h > t_target:
                h = t_target - t
            if h < h_min:
                h = h_min
            t_new = t + h

            (m_new, cholP_new, m_pred, cholP_pred, K_can, H_can, Dth, kappa2_hat) = self.trial_step_sqrt(
                t_new, h, m, cholP, theta_mean, d_ode=d_ode
            )

            # local error metric based on calibrated defect std dev
            Q_loc = kron_eye(d_ode, self.iwp.construct_Q(h))  # diffusion=1
            S_base = sym(H_can @ Q_loc @ H_can.T)
            kappa2 = float(kappa2_hat)
            D_vec = np.sqrt(np.maximum(np.diag(kappa2 * S_base), 0.0))

            y_prev = (E0 @ m).reshape(-1)
            y_curr = (E0 @ m_new).reshape(-1)
            eps = self.atol + self.rtol * np.maximum(np.abs(y_prev), np.abs(y_curr))
            eps = np.maximum(eps, 1e-30)
            E = float(np.sqrt(np.mean((D_vec / eps) ** 2)))

            expo = 1.0 / (self.q + 1.0)
            if E <= 0.0 or not np.isfinite(E):
                h_suggest = min(h_max, max(h_min, h * self.eta_max))
            else:
                h_suggest = h * self.rho * (E ** (-expo))
                h_suggest = float(np.clip(h_suggest, self.eta_min * h, self.eta_max * h))
                h_suggest = float(np.clip(h_suggest, h_min, h_max))

            if E <= 1.0 or h <= h_min:
                # accept
                accepted += 1
                min_h_seen = min(min_h_seen, h)
                max_h_seen = max(max_h_seen, h)

                # update diffusion
                self.diffusion = float(kappa2_hat)
                diffusion_hist.append(self.diffusion)

                # Jacobian recursion
                # prediction Jacobian: Jp = A(h) J
                A_can = kron_eye(d_ode, self.iwp.construct_A(h))
                Jp = A_can @ J

                # filter Jacobian: Jn = (I - K H) Jp  - K D_theta
                Istate = np.eye(Dstate, dtype=float)
                Jn = (Istate - K_can @ H_can) @ Jp
                if p_dim > 0 and Dth is not None:
                    # theta columns start at d_ode
                    Jn[:, d_ode:] = Jn[:, d_ode:] - (K_can @ Dth)

                # store accepted step
                ts.append(t_new)
                dt_list.append(h)
                m_pred_list.append(m_pred)
                cholP_pred_list.append(cholP_pred)
                m_filt_list.append(m_new)
                cholP_filt_list.append(cholP_new)
                K_list.append(K_can)
                H_list.append(H_can)
                Dth_list.append(Dth)
                J_filt_list.append(Jn)

                # advance
                t = t_new
                m = m_new
                cholP = cholP_new
                J = Jn

                # record outputs
                if eval_idx < len(t_eval) and abs(t - t_target) <= 1e-12:
                    out_t.append(t)
                    out_mean.append((E0 @ m).reshape(-1))
                    Pcan = cholP @ cholP.T
                    out_cov_cond.append(E0 @ Pcan @ E0.T)
                    out_cov_prop.append(E0 @ (Pcan + J @ Sigma_inputs @ J.T) @ E0.T)
                    eval_idx += 1

                h = h_suggest
                reject_budget = self.max_reject

            else:
                # reject
                rejected += 1
                reject_budget -= 1
                if reject_budget <= 0:
                    raise RuntimeError("Exceeded max_reject.")
                h = h_suggest

        # optional smoothing
        if self.use_smoother and len(ts) > 2:
            # build maps from time to index
            t_to_idx = {float(tv): i for i, tv in enumerate(ts)}
            # reconstruct covariances
            m_f = [mf.copy() for mf in m_filt_list]
            P_f = [cf @ cf.T for cf in cholP_filt_list]
            m_p = [mp.copy() for mp in m_pred_list]
            P_p = [cp @ cp.T for cp in cholP_pred_list]
            # smoother arrays
            m_s = [None] * len(ts)
            P_s = [None] * len(ts)
            m_s[-1] = m_f[-1]
            P_s[-1] = P_f[-1]
            # backward RTS
            for k in range(len(ts) - 2, -1, -1):
                h_kp1 = dt_list[k + 1]
                A_can = kron_eye(d_ode, self.iwp.construct_A(h_kp1))
                # G = P_f[k] A^T P_p[k+1]^{-1}
                # solve P_p[k+1] X = A P_f[k]  => G = X^T
                Lp = cholP_pred_list[k + 1]
                B = A_can @ P_f[k]
                Y = solve_triangular(Lp, B, lower=True)
                X = solve_triangular(Lp.T, Y, lower=False)
                G = X.T

                m_s[k] = m_f[k] + G @ (m_s[k + 1] - m_p[k + 1])
                P_s[k] = sym(P_f[k] + G @ (P_s[k + 1] - P_p[k + 1]) @ G.T)

            # recompute outputs on out_t from smoothed states
            out_mean_s = []
            out_cov_cond_s = []
            for tv in out_t:
                idx = t_to_idx[float(tv)]
                out_mean_s.append((E0 @ m_s[idx]).reshape(-1))
                out_cov_cond_s.append(E0 @ P_s[idx] @ E0.T)
            out_mean = out_mean_s
            out_cov_cond = out_cov_cond_s

        stats = AdaptiveStats(
            accepted_steps=int(accepted),
            rejected_steps=int(rejected),
            min_step=float(min_h_seen if np.isfinite(min_h_seen) else 0.0),
            max_step=float(max_h_seen),
            diffusion_history=list(diffusion_hist),
        )

        return PropResult(
            t=np.asarray(out_t, dtype=float),
            mean=np.asarray(out_mean, dtype=float),
            cov_cond=np.asarray(out_cov_cond, dtype=float),
            cov_prop=np.asarray(out_cov_prop, dtype=float),
            stats=stats,
        )

def integrate_deterministic(
    problem: ODEProblem,
    *,
    y0: Array,
    theta: Optional[Array],
    t_eval: Array,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> Array:
    """Deterministic ODE solve for a single (y0, theta) at t_eval.
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    t0, t1 = float(problem.t_span[0]), float(problem.t_span[1])
    y0 = np.asarray(y0, dtype=float).reshape(-1)

    def rhs(t: float, y: Array) -> Array:
        return np.asarray(problem.f(float(t), np.asarray(y, dtype=float).reshape(-1), theta), dtype=float)

    sol = _scipy_solve_ivp(
        rhs,
        (t0, t1),
        y0,
        t_eval=t_eval,
        method=str(method),
        rtol=float(rtol),
        atol=float(atol),
        vectorized=False,
    )
    return np.asarray(sol.y.T, dtype=float)
