
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from models import Array, ODEProblem

def problems_paper() -> Dict[str, ODEProblem]:
    """
    Different ODEs
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

    # 3) Lotka–Volterra (2D):
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

    return probs
