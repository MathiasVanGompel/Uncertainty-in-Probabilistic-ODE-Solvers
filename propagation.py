

from typing import Tuple

import numpy as np

from models import Array, DecomposedMoments, ODEProblem
from quadrature import bq_bhkf_weights_stdnormal, gh_tensor_rule
from solver import AdaptiveIWP_EKS1_Prop_Sqrt, chol_spd, sym

def propagate_prop(
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
):
    solver = AdaptiveIWP_EKS1_Prop_Sqrt(
        problem.f,
        problem.Jf,
        dtheta=problem.dtheta,
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

def sym(A: Array) -> Array:
    """Symmetrise 2D or 3D arrays.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        return 0.5 * (A + A.T)
    return 0.5 * (A + np.swapaxes(A, 1, 2))

def propagate_prop_decomposed(
    problem: ODEProblem,
    *,
    t_eval: Array,
    q: int = 3,
    use_smoother: bool = True,
    precondition: bool = True,
    atol: float = 1e-9,
    rtol: float = 1e-5,
    R: float = 1e-12,
    diffusion_init: float = 1.0,
    h0: float = 1e-2,
    h_max: float = 1.0,
) -> DecomposedMoments:
    """First order (Jacobian recursion) propagation with uncertainty separation.

    Numerical/PN-only:   cov_num  := P̄_k (conditional solver covariance)
    Input/physical:     cov_input:= J Σ_u J^T = cov_prop - cov_cond
    """
    res = propagate_prop(
        problem,
        q=int(q),
        use_smoother=bool(use_smoother),
        precondition=bool(precondition),
        atol=float(atol),
        rtol=float(rtol),
        R=float(R),
        diffusion_init=float(diffusion_init),
        h0=float(h0),
        h_max=float(h_max),
        t_eval=np.asarray(t_eval, dtype=float),
    )

    cov_num = np.asarray(res.cov_cond, dtype=float)
    cov_total = np.asarray(res.cov_prop, dtype=float)
    cov_input = sym(cov_total - cov_num)
    cov_quad = np.zeros_like(cov_num)

    return DecomposedMoments(
        t=np.asarray(res.t, dtype=float),
        mean=np.asarray(res.mean, dtype=float),
        cov_total=cov_total,
        cov_num=cov_num,
        cov_input=cov_input,
        cov_quad=cov_quad,
    )

def joint_input_gaussian(problem: ODEProblem) -> Tuple[Array, Array, Array, Array, int, int, Array]:
    """Return (m_in, P, active_idx, chol(P_red), d_ode, p_dim, active_map).
    """
    y0m = np.asarray(problem.y0_mean, dtype=float).reshape(-1)
    d_ode = int(y0m.size)
    thm = None if problem.theta_mean is None else np.asarray(problem.theta_mean, dtype=float).reshape(-1)
    p_dim = 0 if thm is None else int(thm.size)

    m_in = np.concatenate([y0m, thm]) if p_dim > 0 else y0m

    Py0 = np.zeros((d_ode, d_ode), dtype=float) if problem.y0_cov is None else np.asarray(problem.y0_cov, dtype=float)

    if p_dim > 0:
        Pth = np.zeros((p_dim, p_dim), dtype=float) if problem.theta_cov is None else np.asarray(problem.theta_cov, dtype=float)
        P = np.block([[Py0, np.zeros((d_ode, p_dim))], [np.zeros((p_dim, d_ode)), Pth]])
    else:
        P = Py0

    diag = np.diag(P)
    active = np.where(diag > 0.0)[0]

    if active.size > 0:
        P_red = P[np.ix_(active, active)]
        L = chol_spd(P_red, jitter=1e-18, max_tries=8)
    else:
        L = np.zeros((0, 0), dtype=float)

    return m_in, P, active, L, d_ode, p_dim, diag

def propagate_bhkf_decomposed(
    problem: ODEProblem,
    *,
    t_eval: Array,
    q: int = 3,
    use_smoother: bool = True,
    precondition: bool = True,
    atol: float = 1e-9,
    rtol: float = 1e-5,
    R: float = 1e-12,
    diffusion_init: float = 1.0,
    h0: float = 1e-2,
    h_max: float = 1.0,
    gh_order: int = 3,
    bq_ell: float = 1.0,
    bq_alpha2: float = 1.0,
    include_bq_inflation: bool = True,
    use_bq_covariance: bool = True,
) -> DecomposedMoments:
    """BHKF moment propagation with explicit covariance decomposition.
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

    m_in, P, active, L, d_ode, p_dim, _diag = joint_input_gaussian(problem)

    if active.size == 0:
        solver = AdaptiveIWP_EKS1_Prop_Sqrt(
            problem.f,
            problem.Jf,
            dtheta=problem.dtheta,
            q=int(q),
            use_smoother=bool(use_smoother),
            precondition=bool(precondition),
            atol=float(atol),
            rtol=float(rtol),
            R=float(R),
            diffusion_init=float(diffusion_init),
        )
        res = solver.solve(
            t_span=problem.t_span,
            y0_mean=problem.y0_mean,
            theta_mean=problem.theta_mean,
            p_y0=np.zeros((d_ode, d_ode)),
            p_theta=np.zeros((p_dim, p_dim)) if p_dim > 0 else None,
            t_eval=t_eval,
            h0=float(h0),
            h_max=float(h_max),
        )
        cov_num = np.asarray(res.cov_cond, dtype=float)
        cov_input = np.zeros_like(cov_num)
        cov_quad = np.zeros_like(cov_num)
        cov_total = cov_num.copy()
        return DecomposedMoments(
            t=np.asarray(res.t, dtype=float),
            mean=np.asarray(res.mean, dtype=float),
            cov_total=cov_total,
            cov_num=cov_num,
            cov_input=cov_input,
            cov_quad=cov_quad,
        )

    X, wbar = gh_tensor_rule(int(gh_order), int(active.size))
    w_bq, W_bq, diag_add = bq_bhkf_weights_stdnormal(
        X,
        ell=float(bq_ell),
        alpha2=float(bq_alpha2),
        jitter=1e-10,
    )

    n_nodes = X.shape[0]
    solver = AdaptiveIWP_EKS1_Prop_Sqrt(
        problem.f,
        problem.Jf,
        dtheta=problem.dtheta,
        q=int(q),
        use_smoother=bool(use_smoother),
        precondition=bool(precondition),
        atol=float(atol),
        rtol=float(rtol),
        R=float(R),
        diffusion_init=float(diffusion_init),
    )

    mus = np.zeros((n_nodes, t_eval.size, d_ode), dtype=float)
    cov_pns = np.zeros((n_nodes, t_eval.size, d_ode, d_ode), dtype=float)

    for i in range(n_nodes):
        x_red = m_in[active] + (L @ X[i])
        x_full = m_in.copy()
        x_full[active] = x_red

        y0_i = x_full[:d_ode]
        th_i = None if p_dim == 0 else x_full[d_ode:]

        res = solver.solve(
            t_span=problem.t_span,
            y0_mean=y0_i,
            theta_mean=th_i,
            p_y0=np.zeros((d_ode, d_ode)),
            p_theta=np.zeros((p_dim, p_dim)) if p_dim > 0 else None,
            t_eval=t_eval,
            h0=float(h0),
            h_max=float(h_max),
        )
        mus[i] = np.asarray(res.mean, dtype=float)
        cov_pns[i] = np.asarray(res.cov_cond, dtype=float)

    cov_num = np.tensordot(wbar, cov_pns, axes=(0, 0))

    if use_bq_covariance:
        mean = np.tensordot(w_bq, mus, axes=(0, 0))
        cov_input = np.zeros((t_eval.size, d_ode, d_ode), dtype=float)
        for k in range(t_eval.size):
            G = mus[:, k, :]
            mu = mean[k]
            cov_input[k] = sym(G.T @ W_bq @ G - np.outer(mu, mu))
    else:
        mean = np.tensordot(wbar, mus, axes=(0, 0))
        cov_input = np.zeros((t_eval.size, d_ode, d_ode), dtype=float)
        for k in range(t_eval.size):
            G = mus[:, k, :]
            mu = mean[k]
            dG = G - mu[None, :]
            cov_input[k] = sym(dG.T @ (wbar[:, None] * dG))

    cov_quad = np.zeros_like(cov_input)
    if include_bq_inflation:
        eye = np.eye(d_ode, dtype=float)
        for k in range(t_eval.size):
            cov_quad[k] = float(diag_add) * eye

    cov_total = sym(cov_num + cov_input + cov_quad)
    return DecomposedMoments(
        t=t_eval,
        mean=mean,
        cov_total=cov_total,
        cov_num=cov_num,
        cov_input=cov_input,
        cov_quad=cov_quad,
    )

def propagate_sigma_point_decomposed(
    problem: ODEProblem,
    *,
    t_eval: Array,
    q: int = 3,
    use_smoother: bool = True,
    precondition: bool = True,
    atol: float = 1e-9,
    rtol: float = 1e-5,
    R: float = 1e-12,
    diffusion_init: float = 1.0,
    h0: float = 1e-2,
    h_max: float = 1.0,
    gh_order: int = 3,
) -> DecomposedMoments:
    """Classical Gauss--Hermite sigma-point quadrature without BQ inflation."""
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    m_in, P, active, L, d_ode, p_dim, _diag = joint_input_gaussian(problem)

    if active.size == 0:
        solver = AdaptiveIWP_EKS1_Prop_Sqrt(
            problem.f,
            problem.Jf,
            dtheta=problem.dtheta,
            q=int(q),
            use_smoother=bool(use_smoother),
            precondition=bool(precondition),
            atol=float(atol),
            rtol=float(rtol),
            R=float(R),
            diffusion_init=float(diffusion_init),
        )
        res = solver.solve(
            t_span=problem.t_span,
            y0_mean=problem.y0_mean,
            theta_mean=problem.theta_mean,
            p_y0=np.zeros((d_ode, d_ode)),
            p_theta=np.zeros((p_dim, p_dim)) if p_dim > 0 else None,
            t_eval=t_eval,
            h0=float(h0),
            h_max=float(h_max),
        )
        cov_num = np.asarray(res.cov_cond, dtype=float)
        cov_input = np.zeros_like(cov_num)
        cov_quad = np.zeros_like(cov_num)
        cov_total = cov_num.copy()
        return DecomposedMoments(res.t, res.mean, cov_total, cov_num, cov_input, cov_quad)

    X, w = gh_tensor_rule(int(gh_order), int(active.size))
    n_nodes = X.shape[0]
    solver = AdaptiveIWP_EKS1_Prop_Sqrt(
        problem.f,
        problem.Jf,
        dtheta=problem.dtheta,
        q=int(q),
        use_smoother=bool(use_smoother),
        precondition=bool(precondition),
        atol=float(atol),
        rtol=float(rtol),
        R=float(R),
        diffusion_init=float(diffusion_init),
    )

    mus = np.zeros((n_nodes, t_eval.size, d_ode), dtype=float)
    cov_pns = np.zeros((n_nodes, t_eval.size, d_ode, d_ode), dtype=float)

    for i in range(n_nodes):
        x_red = m_in[active] + (L @ X[i])
        x_full = m_in.copy()
        x_full[active] = x_red
        y0_i = x_full[:d_ode]
        th_i = None if p_dim == 0 else x_full[d_ode:]

        res = solver.solve(
            t_span=problem.t_span,
            y0_mean=y0_i,
            theta_mean=th_i,
            p_y0=np.zeros((d_ode, d_ode)),
            p_theta=np.zeros((p_dim, p_dim)) if p_dim > 0 else None,
            t_eval=t_eval,
            h0=float(h0),
            h_max=float(h_max),
        )
        mus[i] = np.asarray(res.mean, dtype=float)
        cov_pns[i] = np.asarray(res.cov_cond, dtype=float)

    mean = np.tensordot(w, mus, axes=(0, 0))
    cov_num = np.tensordot(w, cov_pns, axes=(0, 0))
    cov_input = np.zeros((t_eval.size, d_ode, d_ode), dtype=float)
    for k in range(t_eval.size):
        G = mus[:, k, :]
        mu = mean[k]
        dG = G - mu[None, :]
        cov_input[k] = sym(dG.T @ (w[:, None] * dG))
    cov_quad = np.zeros_like(cov_input)
    cov_total = sym(cov_num + cov_input)
    return DecomposedMoments(t_eval, mean, cov_total, cov_num, cov_input, cov_quad)
