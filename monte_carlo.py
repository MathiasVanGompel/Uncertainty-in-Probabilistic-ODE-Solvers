


import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from metrics import w2_rms
from models import Array, MCStats, ODEProblem
from solver import chol_spd, integrate_deterministic

def sym(A: Array) -> Array:
    """Symmetrise 2D or 3D arrays."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        return 0.5 * (A + A.T)
    return 0.5 * (A + np.swapaxes(A, 1, 2))

def cache_key(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def mc_reference_push(
    problem: ODEProblem,
    *,
    n_samples: int,
    t_eval: Array,
    seed: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
    chunk_size: int = 512,
) -> Tuple[Array, Array, Array]:
    """MC estimate of the pushforward (physical uncertainty only).
    """
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)

    # input Gaussian
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

    rng = np.random.default_rng(int(seed))
    if active.size > 0:
        P_red = P[np.ix_(active, active)]
        L = chol_spd(P_red, jitter=1e-18, max_tries=8)
    else:
        L = np.zeros((0, 0), dtype=float)

    sum_x = np.zeros((t_eval.size, d_ode), dtype=float)
    sum_xx = np.zeros((t_eval.size, d_ode, d_ode), dtype=float)

    done = 0
    while done < n_samples:
        m = int(min(int(chunk_size), int(n_samples - done)))
        done += m

        if active.size > 0:
            Z = rng.standard_normal(size=(m, active.size))
            X_red = m_in[active][None, :] + Z @ L.T
        else:
            X_red = np.zeros((m, 0), dtype=float)

        Y = np.zeros((m, t_eval.size, d_ode), dtype=float)
        for i in range(m):
            x_full = m_in.copy()
            if active.size > 0:
                x_full[active] = X_red[i]
            y0_i = x_full[:d_ode]
            th_i = None if p_dim == 0 else x_full[d_ode:]
            Y[i] = integrate_deterministic(
                problem,
                y0=y0_i,
                theta=th_i,
                t_eval=t_eval,
                rtol=float(rtol),
                atol=float(atol),
                method=str(method),
            )

        sum_x += np.sum(Y, axis=0)
        sum_xx += np.einsum("ntd,nte->tde", Y, Y)

    mean = sum_x / float(n_samples)
    # unbiased covariance across samples
    cov = (sum_xx - float(n_samples) * np.einsum("td,te->tde", mean, mean)) / float(max(n_samples - 1, 1))
    cov = sym(cov)
    return t_eval, mean, cov

def mc_reference_cached(
    problem: ODEProblem,
    *,
    n_ref: int,
    t_eval: Array,
    cache_dir: str | Path = "mc_cache",
    seed: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
    chunk_size: int = 512,
    overwrite: bool = False,
) -> Tuple[Path, Array, Array, Array]:
    """Compute (or load) a cached large-sample MC reference."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    key = cache_key(
        {
            "problem": problem.name,
            "t_span": tuple(map(float, problem.t_span)),
            "t_eval": t_eval.tolist(),
            "n_ref": int(n_ref),
            "seed": int(seed),
            "rtol": float(rtol),
            "atol": float(atol),
            "method": str(method),
            "y0_mean": np.asarray(problem.y0_mean, dtype=float).tolist(),
            "y0_cov": None if problem.y0_cov is None else np.asarray(problem.y0_cov, dtype=float).tolist(),
            "theta_mean": None if problem.theta_mean is None else np.asarray(problem.theta_mean, dtype=float).tolist(),
            "theta_cov": None if problem.theta_cov is None else np.asarray(problem.theta_cov, dtype=float).tolist(),
        }
    )
    path = cache_dir / f"{problem.name}_mc_ref_{key}.npz"
    if path.exists() and not overwrite:
        dat = np.load(path, allow_pickle=False)
        return path, dat["t"], dat["mean"], dat["cov"]

    t0 = time.perf_counter()
    t, mean, cov = mc_reference_push(
        problem,
        n_samples=int(n_ref),
        t_eval=t_eval,
        seed=int(seed),
        rtol=float(rtol),
        atol=float(atol),
        method=str(method),
        chunk_size=int(chunk_size),
    )
    t1 = time.perf_counter()
    np.savez_compressed(path, t=t, mean=mean, cov=cov, runtime_s=float(t1 - t0), n_ref=int(n_ref))
    return path, t, mean, cov

def mc_run_cached(
    problem: ODEProblem,
    *,
    n_samples: int,
    t_eval: Array,
    cache_dir: str | Path = "mc_cache",
    seed: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
    chunk_size: int = 512,
    overwrite: bool = False,
) -> Tuple[Path, Array, Array, Array, float]:
    """Compute (or load) a cached MC estimate for a given sample size.

    This is the same deterministic Monte Carlo as `mc_reference_cached`, but meant
    for repeated *comparison* runs at smaller N (so we can avoid rerunning them).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    key = cache_key(
        {
            "problem": problem.name,
            "t_span": tuple(map(float, problem.t_span)),
            "t_eval": t_eval.tolist(),
            "n_samples": int(n_samples),
            "seed": int(seed),
            "rtol": float(rtol),
            "atol": float(atol),
            "method": str(method),
            "chunk_size": int(chunk_size),
            "y0_mean": np.asarray(problem.y0_mean, dtype=float).tolist(),
            "y0_cov": None if problem.y0_cov is None else np.asarray(problem.y0_cov, dtype=float).tolist(),
            "theta_mean": None if problem.theta_mean is None else np.asarray(problem.theta_mean, dtype=float).tolist(),
            "theta_cov": None if problem.theta_cov is None else np.asarray(problem.theta_cov, dtype=float).tolist(),
        }
    )
    path = cache_dir / f"{problem.name}_mc_run_{key}.npz"
    if path.exists() and not overwrite:
        dat = np.load(path, allow_pickle=False)
        return path, dat["t"], dat["mean"], dat["cov"], float(dat.get("runtime_s", np.nan))

    t0 = time.perf_counter()
    t, mean, cov = mc_reference_push(
        problem,
        n_samples=int(n_samples),
        t_eval=t_eval,
        seed=int(seed),
        rtol=float(rtol),
        atol=float(atol),
        method=str(method),
        chunk_size=int(chunk_size),
    )
    t1 = time.perf_counter()
    runtime_s = float(t1 - t0)
    np.savez_compressed(path, t=t, mean=mean, cov=cov, runtime_s=runtime_s, n_samples=int(n_samples))
    return path, t, mean, cov, runtime_s

def mc_accuracy_distribution(
    problem: ODEProblem,
    *,
    t_eval: Array,
    mean_ref: Array,
    cov_ref: Array,
    sample_sizes: Sequence[int],
    reps: int = 10,
    single_run_n: int = 100_000,
    seed: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
    chunk_size: int = 512,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Dict[int, List[Tuple[float, float]]]:
    """Return {N: [(runtime_s, W2rms), ...]} for MC estimates vs a fixed reference.
    """
    out: Dict[int, List[Tuple[float, float]]] = {}
    t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
    for N in sample_sizes:
        N = int(N)
        n_rep = 1 if N == int(single_run_n) else int(reps)
        vals: List[Tuple[float, float]] = []
        for r in range(n_rep):
            if cache_dir is None:
                t0 = time.perf_counter()
                _, mean, cov = mc_reference_push(
                    problem,
                    n_samples=N,
                    t_eval=t_eval,
                    seed=int(seed + 100_000 * N + r),
                    rtol=float(rtol),
                    atol=float(atol),
                    method=str(method),
                    chunk_size=int(chunk_size),
                )
                t1 = time.perf_counter()
                runtime_s = float(t1 - t0)
            else:
                _p, _t, mean, cov, runtime_s = mc_run_cached(
                    problem,
                    n_samples=N,
                    t_eval=t_eval,
                    cache_dir=cache_dir,
                    seed=int(seed + 100_000 * N + r),
                    rtol=float(rtol),
                    atol=float(atol),
                    method=str(method),
                    chunk_size=int(chunk_size),
                    overwrite=bool(overwrite),
                )
            w2 = w2_rms(mean, cov, mean_ref, cov_ref)
            vals.append((float(runtime_s), float(w2)))
        out[N] = vals
    return out
