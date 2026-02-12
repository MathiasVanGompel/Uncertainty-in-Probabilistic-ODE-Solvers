"""
How to run (examples):
  python uncertainty_propagation.py --problem logistic --method both
  python uncertainty_propagation.py --problem lv --method BHKF --gh-order 3
  python uncertainty_propagation.py --problem vdp --method goal --no-smoother
  python uncertainty_propagation.py --problem logistic --method both --mc-samples 2000 --plot-together
  python uncertainty_propagation.py --problem logistic --method goal --mc-samples 5000 --plot-together --compare-comp 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import argparse
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype, ndarray

try:
    from scipy.linalg import solve_triangular as _scipy_solve_triangular
except Exception:  # pragma: no cover
    _scipy_solve_triangular = None

try:
    from scipy.integrate import solve_ivp as _scipy_solve_ivp
except Exception:  # pragma: no cover
    _scipy_solve_ivp = None

try:
    from scipy.integrate import solve_ivp as _scipy_solve_ivp
except Exception:  # pragma: no cover
    _scipy_solve_ivp = None

Array = np.ndarray


#help functions

def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)

def _chol_spd(A: Array, jitter: float = 1e-12, max_tries: int = 8) -> Array:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Need square matrix for Cholesky.")
    I = np.eye(A.shape[0], dtype=float)
    lam = 0.0
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(_sym(A + (lam + jitter) * I))
        except np.linalg.LinAlgError:
            lam = 10.0 * (lam + jitter) if lam > 0 else jitter
    # final attempt: eigen fix
    w, V = np.linalg.eigh(_sym(A))
    w = np.maximum(w, jitter)
    return np.linalg.cholesky(V @ np.diag(w) @ V.T)

def _solve_triangular(L: Array, B: Array, *, lower: bool) -> Array:
    if _scipy_solve_triangular is not None:
        return _scipy_solve_triangular(L, B, lower=lower, check_finite=False)
    return np.linalg.solve(L, B)

def _kron_eye(d: int, M: Array) -> Array:
    return np.kron(np.eye(d, dtype=float), M)


def _integrate_reference(
    f: Callable[[float, Array, Optional[Array]], Array],
    *,
    t_span: Tuple[float, float],
    y0: Array,
    theta: Optional[Array],
    t_eval: Array,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    rk4_substeps: int = 10,
) -> ndarray[tuple[Any, ...], dtype[Any]] | None:
    """Deterministic reference integrator used for Monte Carlo.

    Returns y(t_eval) with shape (T, d).
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float).reshape(-1)

    def rhs(t: float, y: Array) -> Array:
        return np.asarray(f(float(t), np.asarray(y, dtype=float).reshape(-1), theta), dtype=float).reshape(-1)

    if _scipy_solve_ivp is not None:
        sol = _scipy_solve_ivp(
            rhs,
            (t0, t1),
            y0,
            t_eval=t_eval,
            method=str(method),
            rtol=float(rtol),
            atol=float(atol),
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        Y = np.asarray(sol.y, dtype=float).T  # (d, T) -> (T, d)
        return Y



# IWP prior (order q, dimension q+1)

class IWPSSM:
    """Integrated Wiener Process prior of order q (state dim q+1 per ODE component)."""

    def __init__(self, q: int):
        if q < 1:
            raise ValueError("Need q>=1")
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


# ODE Problem definition

@dataclass
class ODEProblem:
    name: str
    f: Callable[[float, Array, Optional[Array]], Array]          # f(t, y, theta) -> (d,)
    Jf: Callable[[float, Array, Optional[Array]], Array]         # df/dy         -> (d,d)
    dtheta: Optional[Callable[[float, Array, Optional[Array]], Array]] = None  # df/dtheta -> (d,p)
    t_span: Tuple[float, float] = (0.0, 1.0)
    y0_mean: Optional[Array] = None
    y0_cov: Optional[Array] = None
    theta_mean: Optional[Array] = None
    theta_cov: Optional[Array] = None


# main algorithm

@dataclass
class AdaptiveStats:
    accepted_steps: int
    rejected_steps: int
    min_step: float
    max_step: float
    diffusion_history: List[float]

@dataclass
class GoalResult:
    t: Array                 # (N_out,)
    mean: Array              # (N_out, d_ode)
    cov_cond: Array          # (N_out, d_ode, d_ode)
    cov_goal: Array          # (N_out, d_ode, d_ode)
    stats: AdaptiveStats


class AdaptiveIWP_EKS1_Goal_Sqrt:
    """
    Adaptive EKF1/EKS1 ODE filter with:
      - step-size preconditioning for IWP
      - square-root (Cholesky) predict/update
      - Jacobian recursion for goal covariance

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
        if q < 1:
            raise ValueError("Need q>=1.")

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

        self._iwp = IWPSSM(self.q)

        # constant preconditioned matrices for h=1:
        self._A_pre, self._Q_pre = self._preconditioned_constants()
        self._chol_Q_pre = _chol_spd(self._Q_pre, jitter=1e-15, max_tries=8)

    def _preconditioned_constants(self) -> Tuple[Array, Array]:
        """
        Preconditioned matrices A~, Q~ are step-size-independent for IWP:
          A~ = T^{-1}(h) A(h) T(h), Q~ = T^{-1}(h) Q(h) T^{-T}(h)
        so we can compute them at h=1.
        """
        h = 1.0
        A = self._iwp.construct_A(h)
        Q = self._iwp.construct_Q(h)
        T = self._iwp.construct_T(h)
        Tinv = self._iwp.construct_T_inv(h)
        Atil = Tinv @ A @ T
        Qtil = Tinv @ Q @ Tinv.T
        return Atil, _sym(Qtil)

    @staticmethod
    def _E0(d_ode: int, d_temp: int) -> Array:
        D = d_ode * d_temp
        E0 = np.zeros((d_ode, D), dtype=float)
        for i in range(d_ode):
            E0[i, i * d_temp + 0] = 1.0
        return E0

    @staticmethod
    def _E1(d_ode: int, d_temp: int) -> Array:
        D = d_ode * d_temp
        E1 = np.zeros((d_ode, D), dtype=float)
        for i in range(d_ode):
            E1[i, i * d_temp + 1] = 1.0
        return E1

    def _infer_d_ode(self, m: Array) -> int:
        m = np.asarray(m, dtype=float).reshape(-1)
        if m.size % self.d_temp != 0:
            raise ValueError("State length not divisible by (q+1).")
        return int(m.size // self.d_temp)

    def _T_full(self, d_ode: int, h: float) -> Array:
        return _kron_eye(d_ode, self._iwp.construct_T(h))

    def _Tinv_full(self, d_ode: int, h: float) -> Array:
        return _kron_eye(d_ode, self._iwp.construct_T_inv(h))

    def _innovation_H_D(
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
        if J.shape != (d_ode, d_ode):
            raise ValueError(f"Jf must return ({d_ode},{d_ode}), got {J.shape}")

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
            if G.ndim != 2 or G.shape[0] != d_ode:
                raise ValueError("dtheta must return (d_ode, p).")
            Dth = -G  # z = ydot - f => ∂z/∂theta = -∂f/∂theta

        return zhat, H, Dth

    def _init_state(
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
        if y0.size != d_ode:
            raise ValueError("bad y0 shape")

        # Fill y
        for i in range(d_ode):
            m[i * d_temp + 0, 0] = y0[i]

        # Fill first derivative with f
        f0 = np.asarray(self.f(t0, y0, theta), dtype=float).reshape(-1)
        if f0.shape != (d_ode,):
            raise ValueError("f returned wrong shape")
        for i in range(d_ode):
            m[i * d_temp + 1, 0] = f0[i]

        # covariance (canonical)
        P = (self.init_deriv_var) * np.eye(D, dtype=float)
        # conditional on y0: set y component variance to zero
        for i in range(d_ode):
            P[i * d_temp + 0, i * d_temp + 0] = 0.0
        cholP = _chol_spd(P, jitter=1e-18, max_tries=8)
        return m, cholP

    def _sqrt_predict(
        self,
        m_til: Array,
        cholP_til: Array,
        *,
        d_ode: int,
        diffusion: float,
    ) -> Tuple[Array, Array]:
        """Square-root prediction in preconditioned coordinates."""
        Atil_full = _kron_eye(d_ode, self._A_pre)

        # process noise chol (scaled by sqrt(diffusion))
        cholQ_full = math.sqrt(float(diffusion)) * _kron_eye(d_ode, self._chol_Q_pre)

        m_pred = Atil_full @ m_til

        # QR on stacked matrix (n + n, n)
        # stack = [A L, Lq]^T
        AL = Atil_full @ cholP_til
        stack = np.hstack([AL, cholQ_full]).T
        _, R = np.linalg.qr(stack, mode="reduced")
        cholP_pred = R.T
        return m_pred, cholP_pred

    def _sqrt_update(
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

        Returns:
          m_new, cholP_new, K_til
        """
        n = cholP_til_pred.shape[0]
        m = zhat.shape[0]

        # chol_R
        cholR = _chol_spd(Rm, jitter=0.0, max_tries=2)

        # Compute chol_S via QR of [H L, cholR]
        HL = H_til @ cholP_til_pred                 # (m, n)
        stackS = np.hstack([HL, cholR]).T           # (n+m, m)
        _, R = np.linalg.qr(stackS, mode="reduced")
        cholS = R.T                                 # (m, m), lower

        # Kalman gain: K = P H^T S^{-1} using chol factors
        # PHt = L L^T H^T = L (L^T H^T)
        PHt = cholP_til_pred @ (cholP_til_pred.T @ H_til.T)  # (n, m)

        tmp = _solve_triangular(cholS, PHt.T, lower=True)
        tmp = _solve_triangular(cholS.T, tmp, lower=False)
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

    def _trial_step_sqrt(
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
            T = self._T_full(d_ode, h)
            Tinv = self._Tinv_full(d_ode, h)
        else:
            T = np.eye(Dstate, dtype=float)
            Tinv = np.eye(Dstate, dtype=float)

        # transform to preconditioned coords
        m_til = Tinv @ m_can
        cholP_til = Tinv @ cholP_can

        # predict (preconditioned)
        m_til_pred, cholP_til_pred = self._sqrt_predict(m_til, cholP_til, d_ode=d_ode, diffusion=self.diffusion)

        # back to canonical for linearization
        m_can_pred = T @ m_til_pred
        cholP_can_pred = T @ cholP_til_pred

        # measurement linearization (canonical)
        zhat, H_can, Dth = self._innovation_H_D(t_new, m_can_pred, d_ode, theta)

        # update in preconditioned coords: H_til = H_can @ T
        H_til = H_can @ T
        Rm = self.R * np.eye(d_ode, dtype=float)

        m_til_new, cholP_til_new, K_til = self._sqrt_update(
            m_til_pred, cholP_til_pred, zhat=zhat, H_til=H_til, Rm=Rm
        )

        # back to canonical
        m_can_new = T @ m_til_new
        cholP_can_new = T @ cholP_til_new

        # canonical gain: x_can = T x_til -> K_can = T K_til
        K_can = T @ K_til

        # diffusion calibration (scalar quasi-MLE, same idea as your previous file)
        # Use canonical base S_base = H_can Q(h;diff=1) H_can^T
        Q_loc = _kron_eye(d_ode, self._iwp.construct_Q(h))
        S_base = _sym(H_can @ Q_loc @ H_can.T)
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
    ) -> GoalResult:
        """
        Solve IVP on [t0,t1] and return (mean, cond cov, goal cov) on t_eval.
        """

        self.diffusion = float(self.diffusion_init)

        t0, t1 = float(t_span[0]), float(t_span[1])
        if t1 <= t0:
            raise ValueError("Need t_span[1] > t_span[0].")

        y0_mean = np.asarray(y0_mean, dtype=float).reshape(-1)
        d_ode = int(y0_mean.size)
        if theta_mean is not None:
            theta_mean = np.asarray(theta_mean, dtype=float).reshape(-1)

        # Build Sigma_inputs if not provided.
        if Sigma_inputs is None:
            Py0 = np.zeros((d_ode, d_ode), dtype=float) if p_y0 is None else np.asarray(p_y0, dtype=float)
            if Py0.shape != (d_ode, d_ode):
                raise ValueError(f"p_y0 must be ({d_ode},{d_ode})")

            p_dim = 0 if theta_mean is None else int(theta_mean.size)
            Pth = np.zeros((p_dim, p_dim), dtype=float) if p_theta is None else np.asarray(p_theta, dtype=float)
            if p_dim > 0 and Pth.shape != (p_dim, p_dim):
                raise ValueError(f"p_theta must be ({p_dim},{p_dim})")

            C = np.zeros((d_ode, p_dim), dtype=float) if cross_y0_theta is None else np.asarray(cross_y0_theta, dtype=float)
            if p_dim > 0 and C.shape != (d_ode, p_dim):
                raise ValueError(f"cross_y0_theta must be ({d_ode},{p_dim})")

            Sigma_inputs = np.block([[Py0, C], [C.T, Pth]]) if p_dim > 0 else Py0
        else:
            Sigma_inputs = np.asarray(Sigma_inputs, dtype=float)

        p_dim = 0 if theta_mean is None else int(theta_mean.size)
        in_dim = d_ode + p_dim
        if Sigma_inputs.shape != (in_dim, in_dim):
            raise ValueError(f"Sigma_inputs must be ({in_dim},{in_dim})")

        # evaluation grid
        if t_eval is None:
            t_eval = np.array([t0, t1], dtype=float)
        else:
            t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

        if t_eval[0] < t0 - 1e-15 or t_eval[-1] > t1 + 1e-15:
            raise ValueError("t_eval must lie within t_span.")

        # init state
        m, cholP = self._init_state(t0, y0_mean, theta_mean, d_ode)

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

        E0 = self._E0(d_ode, self.d_temp)

        # output arrays on t_eval (subset of accepted grid)
        out_t = [t0]
        out_mean = [ (E0 @ m).reshape(-1) ]
        # conditional covariance of y(t) = E0 P E0^T
        P0 = cholP @ cholP.T
        out_cov_cond = [ E0 @ P0 @ E0.T ]
        out_cov_goal = [ E0 @ (P0 + J @ Sigma_inputs @ J.T) @ E0.T ]

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

            (m_new, cholP_new, m_pred, cholP_pred, K_can, H_can, Dth, kappa2_hat) = self._trial_step_sqrt(
                t_new, h, m, cholP, theta_mean, d_ode=d_ode
            )

            # local error metric based on calibrated defect std dev
            Q_loc = _kron_eye(d_ode, self._iwp.construct_Q(h))  # diffusion=1
            S_base = _sym(H_can @ Q_loc @ H_can.T)
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
                A_can = _kron_eye(d_ode, self._iwp.construct_A(h))
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

                # record outputs if we just hit the next t_eval
                if eval_idx < len(t_eval) and abs(t - t_target) <= 1e-12:
                    out_t.append(t)
                    out_mean.append((E0 @ m).reshape(-1))
                    Pcan = cholP @ cholP.T
                    out_cov_cond.append(E0 @ Pcan @ E0.T)
                    out_cov_goal.append(E0 @ (Pcan + J @ Sigma_inputs @ J.T) @ E0.T)
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

        # optional smoothing: RTS on state, then recompute goal cov on smoothed trajectory
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
                A_can = _kron_eye(d_ode, self._iwp.construct_A(h_kp1))
                # G = P_f[k] A^T P_p[k+1]^{-1}
                # solve P_p[k+1] X = A P_f[k]  => G = X^T
                Lp = cholP_pred_list[k + 1]
                B = A_can @ P_f[k]
                Y = _solve_triangular(Lp, B, lower=True)
                X = _solve_triangular(Lp.T, Y, lower=False)
                G = X.T

                m_s[k] = m_f[k] + G @ (m_s[k + 1] - m_p[k + 1])
                P_s[k] = _sym(P_f[k] + G @ (P_s[k + 1] - P_p[k + 1]) @ G.T)

            # recompute outputs on out_t from smoothed states but keep same goal-J recursion
            out_mean_s = []
            out_cov_cond_s = []
            for tv in out_t:
                idx = t_to_idx[float(tv)]
                out_mean_s.append((E0 @ m_s[idx]).reshape(-1))
                out_cov_cond_s.append(E0 @ P_s[idx] @ E0.T)
            out_mean = out_mean_s
            out_cov_cond = out_cov_cond_s
            # out_cov_goal stays from filtering (goal-J recursion), unless you explicitly want smoothed goal.

        stats = AdaptiveStats(
            accepted_steps=int(accepted),
            rejected_steps=int(rejected),
            min_step=float(min_h_seen if np.isfinite(min_h_seen) else 0.0),
            max_step=float(max_h_seen),
            diffusion_history=list(diffusion_hist),
        )

        return GoalResult(
            t=np.asarray(out_t, dtype=float),
            mean=np.asarray(out_mean, dtype=float),
            cov_cond=np.asarray(out_cov_cond, dtype=float),
            cov_goal=np.asarray(out_cov_goal, dtype=float),
            stats=stats,
        )


# GH nodes  + BQ weights

def gh_tensor_nodes(order: int, dim: int) -> Array:
    """
    Tensor Gauss-Hermite nodes for N(0,I).
    Returns X with shape (N, dim).
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    xs, ws = np.polynomial.hermite.hermgauss(order)  # for integral exp(-x^2)
    # Convert to standard normal:
    # ∫ f(u) N(0,1) du  ≈ Σ w_i f( sqrt(2)*x_i ) / sqrt(pi)
    u1 = math.sqrt(2.0) * xs
    grids = np.meshgrid(*([u1] * dim), indexing="ij")
    X = np.stack([g.reshape(-1) for g in grids], axis=1)
    return X

def rbf_K(X: Array, ell: float, alpha2: float = 1.0, jitter: float = 1e-10) -> Array:
    """EQ kernel Gram matrix: k(x,x') = alpha2 * exp(-||x-x'||^2/(2 ell^2))."""
    X = np.asarray(X, dtype=float)
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    K = float(alpha2) * np.exp(-0.5 * d2 / (float(ell) ** 2))
    K = K + float(jitter) * np.eye(X.shape[0], dtype=float)
    return K


def bq_L_stdnormal(X: Array, ell: float, alpha2: float = 1.0) -> Array:
    """
    L_ij = E[ k(x_i, U) k(U, x_j) ]  for U ~ N(0, I)
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    ell = float(ell)
    alpha2 = float(alpha2)

    r = np.sum(X ** 2, axis=1)     # (N,)
    dot = X @ X.T                  # (N,N)

    c = (alpha2 ** 2) * ((ell ** 2) / (ell ** 2 + 2.0)) ** (0.5 * d)
    E = (2.0 * dot - (ell ** 2 + 1.0) * (r[:, None] + r[None, :])) / (2.0 * ell ** 2 * (ell ** 2 + 2.0))
    return c * np.exp(E)


def bq_bhkf_weights_stdnormal(
    X: Array,
    *,
    ell: float,
    alpha2: float = 1.0,
    jitter: float = 1e-10,
) -> Tuple[Array, Array, float]:
    """
    Returns (w, W, diag_add) for BHKF:
      w = K^{-1} l
      W = K^{-1} L K^{-1}
      diag_add = alpha2 - tr(K^{-1} L)
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    ell = float(ell)
    alpha2 = float(alpha2)
    jitter = float(jitter)

    # K and l
    K = rbf_K(X, ell=ell, alpha2=alpha2, jitter=jitter)
    c = (ell ** 2 / (ell ** 2 + 1.0)) ** (0.5 * d)
    quad = np.sum(X ** 2, axis=1)
    l = alpha2 * c * np.exp(-0.5 * quad / (ell ** 2 + 1.0))

    # L
    Lmat = bq_L_stdnormal(X, ell=ell, alpha2=alpha2)

    # Cholesky solves for stability
    Lk = _chol_spd(K, jitter=0.0, max_tries=8)  # K already has jitter

    # w = K^{-1} l
    y = _solve_triangular(Lk, l, lower=True)
    w = _solve_triangular(Lk.T, y, lower=False)

    # tmp = K^{-1} L
    tmp = _solve_triangular(Lk, Lmat, lower=True)
    tmp = _solve_triangular(Lk.T, tmp, lower=False)
    tr_KinvL = float(np.trace(tmp))

    # W = K^{-1} L K^{-1} (compute as solve(K, tmp.T).T)
    tmp2 = _solve_triangular(Lk, tmp.T, lower=True)
    tmp2 = _solve_triangular(Lk.T, tmp2, lower=False)
    W = _sym(tmp2.T)

    diag_add = alpha2 - tr_KinvL
    return w, W, float(diag_add)

# ODEs
def problems_paper() -> Dict[str, ODEProblem]:
    """
    Different ODEs (from Yao paper)
    """
    probs: Dict[str, ODEProblem] = {}

    # 1) Linear (scalar): y' = a y + b, with a=1, b=0; y(0) ~ N(1, 0.01)
    a_lin, b_lin = 1.0, 0.0

    def f_lin(t: float, y: Array, th: Optional[Array]) -> Array:
        return np.array([a_lin * float(y[0]) + b_lin], dtype=float)

    def J_lin(t: float, y: Array, th: Optional[Array]) -> Array:
        return np.array([[a_lin]], dtype=float)

    probs["linear"] = ODEProblem(
        name="linear",
        f=f_lin,
        Jf=J_lin,
        dtheta=None,
        t_span=(0.0, 3.0),
        y0_mean=np.array([1.0]),
        y0_cov=np.array([[1e-2]]),
        theta_mean=None,
        theta_cov=None,
    )

    # 2) Logistic (scalar): y' = a y (1 - y/b), with a=3 fixed, b ~ N(3, 0.01), y(0)=0.05
    a_log = 3.0

    def f_log(t: float, y: Array, th: Optional[Array]) -> Array:
        b = float(th[0]) if th is not None else 3.0
        x = float(y[0])
        return np.array([a_log * x * (1.0 - x / b)], dtype=float)

    def J_log(t: float, y: Array, th: Optional[Array]) -> Array:
        b = float(th[0]) if th is not None else 3.0
        x = float(y[0])
        return np.array([[a_log * (1.0 - 2.0 * x / b)]], dtype=float)

    def dth_log(t: float, y: Array, th: Optional[Array]) -> Array:
        # df/db = a * x^2 / b^2
        b = float(th[0])
        x = float(y[0])
        return np.array([[a_log * x * x / (b * b)]], dtype=float)

    probs["logistic"] = ODEProblem(
        name="logistic",
        f=f_log,
        Jf=J_log,
        dtheta=dth_log,
        t_span=(0.0, 3.0),
        y0_mean=np.array([0.05]),
        y0_cov=np.array([[0.0]]),
        theta_mean=np.array([3.0]),          # theta = [b]
        theta_cov=np.array([[1e-2]]),
    )

    # 3) FitzHugh–Nagumo (2D):
    #   y1' = y1 - (1/3) y1^3 - y2 + a
    #   y2' = (1/d) (y1 + b - c y2)
    # with a=0, b=0.08, c=0.07, d=1.25; y(0) ~ N([0.5,1], 0.1 I2)
    a_fhn, b_fhn, c_fhn, d_fhn = 0.0, 0.08, 0.07, 1.25

    def f_fhn(t: float, y: Array, th: Optional[Array]) -> Array:
        y1, y2 = float(y[0]), float(y[1])
        dy1 = y1 - (y1 ** 3) / 3.0 - y2 + a_fhn
        dy2 = (y1 + b_fhn - c_fhn * y2) / d_fhn
        return np.array([dy1, dy2], dtype=float)

    def J_fhn(t: float, y: Array, th: Optional[Array]) -> Array:
        y1, y2 = float(y[0]), float(y[1])
        return np.array([[1.0 - y1 * y1, -1.0],
                         [1.0 / d_fhn, -c_fhn / d_fhn]], dtype=float)

    probs["fhn"] = ODEProblem(
        name="fhn",
        f=f_fhn,
        Jf=J_fhn,
        dtheta=None,
        t_span=(0.0, 7.0),
        y0_mean=np.array([0.5, 1.0]),
        y0_cov=0.1 * np.eye(2),
        theta_mean=None,
        theta_cov=None,
    )

    # 4) Lotka–Volterra (2D):
    #   y1' = a y1 - b y1 y2
    #   y2' = -c y2 + d y1 y2
    # with a=5, b=0.5, c=5, d=0.5; y(0) ~ N([5,5], 0.3 I2)
    a_lv, b_lv, c_lv, d_lv = 5.0, 0.5, 5.0, 0.5

    def f_lv(t: float, y: Array, th: Optional[Array]) -> Array:
        y1, y2 = float(y[0]), float(y[1])
        dy1 = a_lv * y1 - b_lv * y1 * y2
        dy2 = -c_lv * y2 + d_lv * y1 * y2
        return np.array([dy1, dy2], dtype=float)

    def J_lv(t: float, y: Array, th: Optional[Array]) -> Array:
        y1, y2 = float(y[0]), float(y[1])
        return np.array([[a_lv - b_lv * y2, -b_lv * y1],
                         [d_lv * y2, -c_lv + d_lv * y1]], dtype=float)

    probs["lv"] = ODEProblem(
        name="lv",
        f=f_lv,
        Jf=J_lv,
        dtheta=None,
        t_span=(0.0, 2.0),
        y0_mean=np.array([5.0, 5.0]),
        y0_cov=0.3 * np.eye(2),
        theta_mean=None,
        theta_cov=None,
    )

    # 5) Van der Pol (2D):
    #   y1' = y2
    #   y2' = a (1 - y1^2) y2 - y1
    # with a=0.05; y(0) ~ N([5,5], 2 I2)
    a_vdp = 0.05

    def f_vdp(t: float, y: Array, th: Optional[Array]) -> Array:
        y1, y2 = float(y[0]), float(y[1])
        return np.array([y2, a_vdp * (1.0 - y1 * y1) * y2 - y1], dtype=float)

    def J_vdp(t: float, y: Array, th: Optional[Array]) -> Array:
        y1, y2 = float(y[0]), float(y[1])
        return np.array([[0.0, 1.0],
                         [-2.0 * a_vdp * y1 * y2 - 1.0, a_vdp * (1.0 - y1 * y1)]], dtype=float)

    probs["vdp"] = ODEProblem(
        name="vdp",
        f=f_vdp,
        Jf=J_vdp,
        dtheta=None,
        t_span=(0.0, 10.0),
        y0_mean=np.array([5.0, 5.0]),
        y0_cov=2.0 * np.eye(2),
        theta_mean=None,
        theta_cov=None,
    )

    return probs



# propagation methods

def propagate_goal(
    problem: ODEProblem,
    *,
    q: int,
    use_smoother: bool,
    precondition: bool,
    atol: float,
    rtol: float,
    R: float,
    diffusion_init: float,
    h0: float,
    h_max: float,
    t_eval: Array,
) -> GoalResult:
    solver = AdaptiveIWP_EKS1_Goal_Sqrt(
        problem.f, problem.Jf, dtheta=problem.dtheta,
        q=q,
        use_smoother=use_smoother,
        precondition=precondition,
        atol=atol,
        rtol=rtol,
        R=R,
        diffusion_init=diffusion_init,
    )
    return solver.solve(
        t_span=problem.t_span,
        y0_mean=problem.y0_mean,
        theta_mean=problem.theta_mean,
        p_y0=problem.y0_cov,
        p_theta=problem.theta_cov,
        t_eval=t_eval,
        h0=h0,
        h_max=h_max,
    )

def propagate_gh_bq(
    problem: ODEProblem,
    *,
    q: int,
    use_smoother: bool,
    precondition: bool,
    atol: float,
    rtol: float,
    R: float,
    diffusion_init: float,
    h0: float,
    h_max: float,
    t_eval: Array,
    gh_order: int,
    bq_ell: float,
) -> Tuple[Array, Array, Array, Array]:
    """
    GH nodes + BQ weights over joint Gaussian in (y0, theta).
    Returns:
      t, mean, cov_total, cov_pn
    """
    y0m = np.asarray(problem.y0_mean, dtype=float).reshape(-1)
    d_ode = y0m.size
    thm = None if problem.theta_mean is None else np.asarray(problem.theta_mean, dtype=float).reshape(-1)
    p_dim = 0 if thm is None else thm.size

    # joint mean and cov
    m_in = np.concatenate([y0m, thm]) if p_dim > 0 else y0m
    Py0 = np.zeros((d_ode, d_ode), dtype=float) if problem.y0_cov is None else np.asarray(problem.y0_cov, dtype=float)
    if p_dim > 0:
        Pth = (np.zeros((p_dim, p_dim), dtype=float) if problem.theta_cov is None
               else np.asarray(problem.theta_cov, dtype=float)) if p_dim > 0 else None

    else:
        Pth = None

    if p_dim > 0:
        P = np.block([[Py0, np.zeros((d_ode, p_dim))],
                      [np.zeros((p_dim, d_ode)), Pth]])
    else:
        P = Py0

    # reduce to uncertain subspace if some variances are ~0
    diag = np.diag(P)
    active = np.where(diag > 0.0)[0]
    if active.size == 0:
        # deterministic inputs: just one solve
        res = propagate_goal(problem, q=q, use_smoother=use_smoother, precondition=precondition,
                             atol=atol, rtol=rtol, R=R, diffusion_init=diffusion_init,
                             h0=h0, h_max=h_max, t_eval=t_eval)
        return res.t, res.mean, res.cov_cond, res.cov_cond

    m_red = m_in[active]
    P_red = P[np.ix_(active, active)]
    L = _chol_spd(P_red, jitter=1e-18, max_tries=8)

    # GH nodes in whitened coordinates u ~ N(0,I)
    X = gh_tensor_nodes(gh_order, dim=active.size)  # (N, d_active)
    w, W, diag_add = bq_bhkf_weights_stdnormal(X, ell=bq_ell, alpha2=1.0, jitter=1e-10)


    N = X.shape[0]
    # storage of each node solution
    mus = []
    cov_pns = []

    # reuse one solver instance for all nodes (diffusion is reset per solve)
    solver = AdaptiveIWP_EKS1_Goal_Sqrt(
        problem.f, problem.Jf, dtheta=problem.dtheta,
        q=q,
        use_smoother=use_smoother,
        precondition=precondition,
        atol=atol,
        rtol=rtol,
        R=R,
        diffusion_init=diffusion_init,
    )

    for i in range(N):
        u = X[i]
        x_red = m_red + L @ u
        x_full = m_in.copy()
        x_full[active] = x_red

        y0_i = x_full[:d_ode]
        th_i = None if p_dim == 0 else x_full[d_ode:]

        # conditional solve: set p_y0=p_theta=0 to get pure PN
        res = solver.solve(
            t_span=problem.t_span,
            y0_mean=y0_i,
            theta_mean=th_i,
            p_y0=np.zeros((d_ode, d_ode)),
            p_theta=np.zeros((p_dim, p_dim)) if p_dim > 0 else None,
            t_eval=t_eval,
            h0=h0,
            h_max=h_max,
        )
        mus.append(res.mean)         # (T, d_ode)
        cov_pns.append(res.cov_cond) # (T, d_ode, d_ode)

    mus = np.asarray(mus, dtype=float)         # (N, T, d)
    cov_pns = np.asarray(cov_pns, dtype=float) # (N, T, d, d)
    t = np.asarray(t_eval, dtype=float).reshape(-1)
    Tn = t.size

    mean = np.tensordot(w, mus, axes=(0, 0))  # (T, d)
    cov_pn = np.tensordot(w, cov_pns, axes=(0, 0))  # (T, d, d)

    # BHKF covariance of conditional means (paper Eq. (32))
    Tn = mean.shape[0]
    d_ode = mean.shape[1]
    I = np.eye(d_ode, dtype=float)

    cov_bq = np.zeros((Tn, d_ode, d_ode), dtype=float)
    for k in range(Tn):
        G = mus[:, k, :]  # (N, d)
        mu = mean[k]  # (d,)
        cov_bq[k] = _sym(G.T @ W @ G - np.outer(mu, mu) + diag_add * I)

    cov_total = cov_pn + cov_bq
    return t, mean, cov_total, cov_pn


# Monte Carlo reference

def _integrate_deterministic(
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

    if _scipy_solve_ivp is not None:
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
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return np.asarray(sol.y.T, dtype=float)


def propagate_mc_reference(
    problem: ODEProblem,
    *,
    n_samples: int,
    t_eval: Array,
    seed: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> Tuple[Array, Array, Array]:
    """Monte Carlo reference over the (y0, theta) input distribution.

    Returns:
      t, mean_mc, cov_mc
    where cov_mc is the empirical covariance across samples at each time.

    Notes:
      - This is a *reference* for input uncertainty only (no PN).
      - If some input dimensions have 0 variance, they are treated as deterministic.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    y0m = np.asarray(problem.y0_mean, dtype=float).reshape(-1)
    d_ode = int(y0m.size)
    thm = None if problem.theta_mean is None else np.asarray(problem.theta_mean, dtype=float).reshape(-1)
    p_dim = 0 if thm is None else int(thm.size)

    m_in = np.concatenate([y0m, thm]) if p_dim > 0 else y0m

    Py0 = np.zeros((d_ode, d_ode), dtype=float) if problem.y0_cov is None else np.asarray(problem.y0_cov, dtype=float)
    if Py0.shape != (d_ode, d_ode):
        raise ValueError(f"problem.y0_cov must be ({d_ode},{d_ode})")

    if p_dim > 0:
        Pth = (np.zeros((p_dim, p_dim), dtype=float) if problem.theta_cov is None
               else np.asarray(problem.theta_cov, dtype=float))
        if Pth.shape != (p_dim, p_dim):
            raise ValueError(f"problem.theta_cov must be ({p_dim},{p_dim})")
        P = np.block([[Py0, np.zeros((d_ode, p_dim))],
                      [np.zeros((p_dim, d_ode)), Pth]])
    else:
        P = Py0

    diag = np.diag(P)
    active = np.where(diag > 0.0)[0]

    rng = np.random.default_rng(int(seed))

    if active.size > 0:
        P_red = P[np.ix_(active, active)]
        L = _chol_spd(P_red, jitter=1e-18, max_tries=8)
        Z = rng.standard_normal(size=(n_samples, active.size))
        X_red = m_in[active][None, :] + Z @ L.T  # (N, d_active)
    else:
        X_red = np.zeros((n_samples, 0), dtype=float)

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    Tn = int(t_eval.size)
    samples = np.zeros((n_samples, Tn, d_ode), dtype=float)

    for i in range(n_samples):
        x_full = m_in.copy()
        if active.size > 0:
            x_full[active] = X_red[i]

        y0_i = x_full[:d_ode]
        th_i = None if p_dim == 0 else x_full[d_ode:]
        ys = _integrate_deterministic(
            problem,
            y0=y0_i,
            theta=th_i,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        samples[i] = ys

    mean = np.mean(samples, axis=0)
    centered = samples - mean[None, :, :]
    cov = np.einsum("ntd,nte->tde", centered, centered) / float(max(n_samples - 1, 1))
    return t_eval, mean, cov


# Plotting
def _ensure_dir(p: str) -> None:
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def plot_bands(
    t: Array,
    mean: Array,
    cov: Array,
    *,
    title: str,
    outpath: str,
    labels: Optional[List[str]] = None,
):
    t = np.asarray(t, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    d = mean.shape[1]

    fig, ax = plt.subplots(figsize=(9, 4))
    for j in range(d):
        mu = mean[:, j]
        sd = np.sqrt(np.maximum(cov[:, j, j], 0.0))
        lab = labels[j] if labels is not None else f"comp {j}"
        ax.plot(t, mu, label=lab)
        ax.fill_between(t, mu - 1.96 * sd, mu + 1.96 * sd, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_var_decomp(
    t: Array,
    cov_total: Array,
    cov_pn: Array,
    *,
    title: str,
    outpath: str,
    comp: int = 0,
):
    t = np.asarray(t, dtype=float).reshape(-1)
    cov_total = np.asarray(cov_total, dtype=float)
    cov_pn = np.asarray(cov_pn, dtype=float)

    total = cov_total[:, comp, comp]
    pn = cov_pn[:, comp, comp]
    nonpn = np.maximum(total - pn, 0.0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, total, label="total var")
    ax.plot(t, pn, label="PN var")
    ax.plot(t, nonpn, label="non-PN var")
    ax.set_title(title + f" (component {comp})")
    ax.set_xlabel("t")
    ax.set_ylabel("variance")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_compare_component(
    t: Array,
    series: List[Tuple[str, Array, Array]],
    *,
    title: str,
    outpath: str,
    comp: int = 0,
):
    """Overlay multiple (mean, cov) trajectories for a single component."""
    t = np.asarray(t, dtype=float).reshape(-1)
    if len(series) == 0:
        raise ValueError("Need at least one series")

    fig, ax = plt.subplots(figsize=(9, 4))
    for label, mean, cov in series:
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if mean.ndim != 2 or cov.ndim != 3:
            raise ValueError("mean must be (T,d) and cov must be (T,d,d)")
        if comp < 0 or comp >= mean.shape[1]:
            raise ValueError(f"comp index {comp} out of range for d={mean.shape[1]}")

        mu = mean[:, comp]
        sd = np.sqrt(np.maximum(cov[:, comp, comp], 0.0))
        (ln,) = ax.plot(t, mu, label=label)
        ax.fill_between(t, mu - 1.96 * sd, mu + 1.96 * sd, color=ln.get_color(), alpha=0.15)

    ax.set_title(title + f" (component {comp})")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main() -> None:
    probs = problems_paper()

    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=str, default="logistic", choices=sorted(probs.keys()))
    ap.add_argument("--method", type=str, default="both", choices=["goal", "BHKF", "both", "MC"])
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--no-smoother", action="store_true")
    ap.add_argument("--no-precondition", action="store_true")
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--R", type=float, default=1e-12)
    ap.add_argument("--diffusion-init", type=float, default=1.0)
    ap.add_argument("--h0", type=float, default=1e-2)
    ap.add_argument("--h-max", type=float, default=1.0)

    ap.add_argument("--t-steps", type=int, default=200)
    ap.add_argument("--gh-order", type=int, default=6)
    ap.add_argument("--bq-ell", type=float, default=1.0)

    # Monte Carlo reference (input uncertainty only)
    ap.add_argument("--mc-samples", type=int, default=0, help="If >0, compute an MC reference with this many samples.")
    ap.add_argument("--mc-seed", type=int, default=0)
    ap.add_argument("--mc-rtol", type=float, default=1e-10)
    ap.add_argument("--mc-atol", type=float, default=1e-12)
    ap.add_argument("--mc-method", type=str, default="DOP853", help="solve_ivp method name (ignored for RK4 fallback)")

    # Plotting
    ap.add_argument("--plot-together", action="store_true", help="Overlay available methods in one comparison plot.")
    ap.add_argument("--compare-comp", type=int, default=0, help="Component index to compare when --plot-together is set.")

    ap.add_argument("--figdir", type=str, default="figures")

    args = ap.parse_args()

    prob = probs[args.problem]
    t0, t1 = prob.t_span
    t_eval = np.linspace(t0, t1, int(args.t_steps), dtype=float)

    use_smoother = not args.no_smoother
    precondition = not args.no_precondition

    _ensure_dir(args.figdir)

    if args.method == "MC" and args.mc_samples <= 0:
        raise ValueError("--method MC requires --mc-samples > 0")

    compare_series: List[Tuple[str, Array, Array]] = []

    def _align_to_eval(t_src: Array, mean_src: Array, cov_src: Array) -> Tuple[Array, Array]:
        """Align (mean,cov) defined on t_src onto the global t_eval via linear interpolation."""
        t_src = np.asarray(t_src, dtype=float).reshape(-1)
        mean_src = np.asarray(mean_src, dtype=float)
        cov_src = np.asarray(cov_src, dtype=float)
        if t_src.shape[0] == t_eval.shape[0] and np.allclose(t_src, t_eval, rtol=0.0, atol=1e-10):
            return mean_src, cov_src
        d = mean_src.shape[1]
        mean_al = np.stack([np.interp(t_eval, t_src, mean_src[:, j]) for j in range(d)], axis=1)
        cov_al = np.zeros((t_eval.size, d, d), dtype=float)
        for i in range(d):
            for j in range(d):
                cov_al[:, i, j] = np.interp(t_eval, t_src, cov_src[:, i, j])
        return mean_al, cov_al

    if args.method in ("goal", "both"):
        t_start = time.perf_counter()
        res = propagate_goal(
            prob,
            q=args.q,
            use_smoother=use_smoother,
            precondition=precondition,
            atol=args.atol,
            rtol=args.rtol,
            R=args.R,
            diffusion_init=args.diffusion_init,
            h0=args.h0,
            h_max=args.h_max,
            t_eval=t_eval,
        )
        t_end = time.perf_counter()
        print(
            f"[goal] accepted={res.stats.accepted_steps}, rejected={res.stats.rejected_steps}, "
            f"h_min={res.stats.min_step:.3e}, h_max={res.stats.max_step:.3e}, "
            f"runtime={t_end - t_start:.2f}s"
        )

        plot_bands(
            res.t,
            res.mean,
            res.cov_goal,
            title=f"{prob.name}: goal propagation (total)",
            outpath=os.path.join(args.figdir, f"{prob.name}_goal_total.png"),
        )
        plot_bands(
            res.t,
            res.mean,
            res.cov_cond,
            title=f"{prob.name}: goal propagation (PN only)",
            outpath=os.path.join(args.figdir, f"{prob.name}_goal_pn.png"),
        )

        m_al, c_al = _align_to_eval(res.t, res.mean, res.cov_goal)
        compare_series.append(("goal", m_al, c_al))

    if args.method in ("BHKF", "both"):
        t_start = time.perf_counter()
        t, mean, cov_total, cov_pn = propagate_gh_bq(
            prob,
            q=args.q,
            use_smoother=use_smoother,
            precondition=precondition,
            atol=args.atol,
            rtol=args.rtol,
            R=args.R,
            diffusion_init=args.diffusion_init,
            h0=args.h0,
            h_max=args.h_max,
            t_eval=t_eval,
            gh_order=args.gh_order,
            bq_ell=args.bq_ell,
        )
        t_end = time.perf_counter()
        n_unc = (prob.y0_mean.size) + (0 if prob.theta_mean is None else prob.theta_mean.size)
        print(
            f"[BHKF] gh_order={args.gh_order}, approx_nodes={args.gh_order**n_unc} "
            f"(before dropping zero-variance dims), runtime={t_end - t_start:.2f}s"
        )

        plot_bands(
            t,
            mean,
            cov_total,
            title=f"{prob.name}: BHKF propagation (total)",
            outpath=os.path.join(args.figdir, f"{prob.name}_BHKF_total.png"),
        )
        plot_var_decomp(
            t,
            cov_total,
            cov_pn,
            title=f"{prob.name}: BHKF variance decomposition",
            outpath=os.path.join(args.figdir, f"{prob.name}_BHKF_decomp.png"),
            comp=0,
        )

        m_al, c_al = _align_to_eval(t, mean, cov_total)
        compare_series.append(("BHKF", m_al, c_al))

    if args.mc_samples > 0 or args.method == "MC":
        t_start = time.perf_counter()
        t_mc, mean_mc, cov_mc = propagate_mc_reference(
            prob,
            n_samples=int(args.mc_samples),
            t_eval=t_eval,
            seed=int(args.mc_seed),
            rtol=float(args.mc_rtol),
            atol=float(args.mc_atol),
            method=str(args.mc_method),
        )
        t_end = time.perf_counter()
        print(f"[MC] samples={int(args.mc_samples)}, runtime={t_end - t_start:.2f}s")

        plot_bands(
            t_mc,
            mean_mc,
            cov_mc,
            title=f"{prob.name}: Monte Carlo reference (input uncertainty)",
            outpath=os.path.join(args.figdir, f"{prob.name}_MC_total.png"),
        )

        m_al, c_al = _align_to_eval(t_mc, mean_mc, cov_mc)
        compare_series.append(("MC", m_al, c_al))

    if args.plot_together:
        if len(compare_series) == 0:
            raise RuntimeError("Nothing to plot: select --method goal/BHKF/both and/or set --mc-samples")
        methods_str = " + ".join([lbl for (lbl, _, _) in compare_series])
        plot_compare_component(
            t_eval,
            compare_series,
            title=f"{prob.name}: comparison ({methods_str})",
            outpath=os.path.join(args.figdir, f"{prob.name}_compare_comp{int(args.compare_comp)}.png"),
            comp=int(args.compare_comp),
        )

    print(f"Saved figures to: {os.path.abspath(args.figdir)}")


if __name__ == "__main__":
    main()
