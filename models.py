
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

Array = np.ndarray

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

@dataclass
class AdaptiveStats:
    accepted_steps: int
    rejected_steps: int
    min_step: float
    max_step: float
    diffusion_history: List[float]

@dataclass
class PropResult:
    t: Array                 # (N_out,)
    mean: Array              # (N_out, d_ode)
    cov_cond: Array          # (N_out, d_ode, d_ode)
    cov_prop: Array          # (N_out, d_ode, d_ode)
    stats: AdaptiveStats

@dataclass
class DecomposedMoments:
    """
    All covariance arrays are shaped (T, d, d) and satisfy
      cov_total = cov_num + cov_input + cov_quad
    """

    t: Array
    mean: Array
    cov_total: Array
    cov_num: Array
    cov_input: Array
    cov_quad: Array

@dataclass
class MCStats:
    n: int
    runtime_s: float
    mean: Array
    cov: Array
