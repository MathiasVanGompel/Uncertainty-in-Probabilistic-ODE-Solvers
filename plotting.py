
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from models import DecomposedMoments
from plotstyle import full_width_figure, savefig


def diag(cov: np.ndarray) -> np.ndarray:
    """Return diagonal across time: (T,d,d) -> (T,d)."""
    return np.diagonal(cov, axis1=1, axis2=2)

def clip_pos(x: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), float(eps))

def band_std(cov: np.ndarray, comp: int) -> np.ndarray:
    return np.sqrt(np.maximum(diag(cov)[:, comp], 0.0))

def fill_band(
    ax: plt.Axes,
    t: np.ndarray,
    m: np.ndarray,
    s: np.ndarray,
    *,
    label: str,
    alpha: float = 0.15,
    linestyle: str = "-",
    linewidth: float = 1.5,
    zorder: int = 2,
):
    ax.plot(t, m, linestyle=linestyle, linewidth=linewidth, label=label, zorder=zorder)
    ax.fill_between(t, m - 1.96 * s, m + 1.96 * s, alpha=alpha, zorder=zorder)

def plot_bands_prop_total_vs_only(
    t: np.ndarray,
    mean_ref: np.ndarray,
    cov_ref: np.ndarray,
    prop: DecomposedMoments,
    *,
    comp: int,
    title: str,
    outpath: Path,
):
    """Reference + First-order total + First-order PN-only bands."""
    fig, ax = plt.subplots()

    mref = mean_ref[:, comp]
    sref = band_std(cov_ref, comp)
    fill_band(ax, t, mref, sref, label="MC reference", alpha=0.10, linewidth=1.7, zorder=1)

    mg = prop.mean[:, comp]
    fill_band(
        ax,
        t,
        mg,
        band_std(prop.cov_total, comp),
        label="First-order total",
        alpha=0.18,
        linestyle="-",
        linewidth=1.6,
        zorder=3,
    )
    fill_band(
        ax,
        t,
        mg,
        band_std(prop.cov_num, comp),
        label="First-order PN-only (numerical)",
        alpha=0.10,
        linestyle="--",
        linewidth=1.3,
        zorder=4,
    )

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_bands_all_methods(
    t: np.ndarray,
    mean_ref: np.ndarray,
    cov_ref: np.ndarray,
    prop: DecomposedMoments,
    bhkf: DecomposedMoments,
    *,
    comp: int,
    title: str,
    outpath: Path,
):
    """Reference + First-order(total+PN-only) + BHKF"""
    fig, ax = plt.subplots()

    mref = mean_ref[:, comp]
    sref = band_std(cov_ref, comp)
    fill_band(ax, t, mref, sref, label="MC reference", alpha=0.08, linewidth=1.6, zorder=1)

    mg = prop.mean[:, comp]
    fill_band(ax, t, mg, band_std(prop.cov_total, comp), label="First-order total", alpha=0.14, zorder=3)
    fill_band(ax, t, mg, band_std(prop.cov_num, comp), label="First-order PN-only", alpha=0.08, linestyle="--", zorder=4)

    mb = bhkf.mean[:, comp]
    fill_band(ax, t, mb, band_std(bhkf.cov_total, comp), label="BHKF total", alpha=0.12, zorder=3)


    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_variance_decomposition_line(
    t: np.ndarray,
    dec: DecomposedMoments,
    *,
    comp: int,
    title: str,
    outpath: Path,
):
    var_num = np.maximum(dec.cov_num[:, comp, comp], 0.0)
    var_inp = np.maximum(dec.cov_input[:, comp, comp], 0.0)
    var_quad = np.maximum(dec.cov_quad[:, comp, comp], 0.0)
    var_tot = np.maximum(dec.cov_total[:, comp, comp], 0.0)

    fig, ax = plt.subplots()
    ax.plot(t, var_tot, label="total")
    ax.plot(t, var_num, label="numerical (PN-only)")
    ax.plot(t, var_inp, label="input (physical)")
    ax.plot(t, var_quad, label="quadrature (BQ inflation)")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(f"variance (component {comp})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_variance_decomposition_stack(
    t: np.ndarray,
    dec: DecomposedMoments,
    *,
    comp: int,
    title: str,
    outpath: Path,
):
    var_num = np.maximum(dec.cov_num[:, comp, comp], 0.0)
    var_inp = np.maximum(dec.cov_input[:, comp, comp], 0.0)
    var_quad = np.maximum(dec.cov_quad[:, comp, comp], 0.0)

    fig, ax = plt.subplots()
    ax.stackplot(
        t,
        var_num,
        var_inp,
        var_quad,
        labels=["numerical", "input", "quadrature"],
        alpha=0.6,
    )
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(f"variance (component {comp})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_variance_decomposition_ratio(
    t: np.ndarray,
    dec: DecomposedMoments,
    *,
    comp: int,
    title: str,
    outpath: Path,
):
    var_num = np.maximum(dec.cov_num[:, comp, comp], 0.0)
    var_inp = np.maximum(dec.cov_input[:, comp, comp], 0.0)
    var_quad = np.maximum(dec.cov_quad[:, comp, comp], 0.0)
    var_tot = np.maximum(dec.cov_total[:, comp, comp], 0.0)

    denom = clip_pos(var_tot, 1e-16)
    r_num = var_num / denom
    r_inp = var_inp / denom
    r_quad = var_quad / denom

    fig, ax = plt.subplots()
    ax.plot(t, r_num, label="numerical/total")
    ax.plot(t, r_inp, label="input/total")
    ax.plot(t, r_quad, label="quadrature/total")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(f"variance fraction (component {comp})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_variance_component_comparison(
    t: np.ndarray,
    prop: DecomposedMoments,
    bhkf: DecomposedMoments,
    *,
    comp: int,
    outdir: Path,
    prob_name: str,
):
    """Compare the individual variance components across methods."""
    components = [
        ("numerical", "cov_num"),
        ("input", "cov_input"),
        ("quadrature", "cov_quad"),
        ("total", "cov_total"),
    ]

    for cname, field in components:
        fig, ax = plt.subplots()
        vg = np.maximum(getattr(prop, field)[:, comp, comp], 0.0)
        vb = np.maximum(getattr(bhkf, field)[:, comp, comp], 0.0)
        ax.plot(t, vg, label=f"First-order {cname}")
        ax.plot(t, vb, label=f"BHKF {cname}")
        ax.set_yscale("log")
        ax.set_title(f"{prob_name}: variance component comparison ({cname}, comp {comp})")
        ax.set_xlabel("t")
        ax.set_ylabel("variance")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        savefig(fig, outdir / f"{prob_name}_varcomp_{cname}_comp{comp}.png")

def mean_error_t(mean: np.ndarray, mean_ref: np.ndarray) -> np.ndarray:
    e = mean - mean_ref
    return np.linalg.norm(e, axis=1)

def mean_rel_error_t(mean: np.ndarray, mean_ref: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    num = mean_error_t(mean, mean_ref)
    den = np.linalg.norm(mean_ref, axis=1)
    return num / clip_pos(den, eps)

def var_error_t(cov: np.ndarray, cov_ref: np.ndarray, comp: int) -> np.ndarray:
    return np.abs(diag(cov)[:, comp] - diag(cov_ref)[:, comp])

def plot_error_time_series(
    t: np.ndarray,
    series: Dict[str, np.ndarray],
    *,
    yscale_log: bool = True,
    title: str,
    ylabel: str,
    outpath: Path,
):
    fig, ax = plt.subplots()
    for name, y in series.items():
        y = np.asarray(y, dtype=float)
        if yscale_log:
            y = clip_pos(y, 1e-16)
        ax.plot(t, y, label=name)
    if yscale_log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_final_time_summary(
    t: np.ndarray,
    metrics: Dict[str, float],
    *,
    title: str,
    outpath: Path,
):
    labels = list(metrics.keys())
    vals = np.array([metrics[k] for k in labels], dtype=float)
    vals = clip_pos(vals, 1e-16)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("value (log)")
    ax.grid(True, which="both", alpha=0.3, axis="y")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_accuracy_vs_runtime(
    mc_dist: Dict[int, List[Tuple[float, float]]],
    *,
    prop_pt: Tuple[float, float] | None,
    bhkf_pt: Tuple[float, float] | None,
    title: str,
    outpath: Path,
):
    with full_width_figure():
        fig, ax = plt.subplots()
    for N, vals in sorted(mc_dist.items()):
        rts = [v[0] for v in vals]
        w2s = [v[1] for v in vals]
        ax.scatter(rts, w2s, s=18, alpha=0.45)
        ax.scatter([float(np.median(rts))], [float(np.median(w2s))], marker="o", s=55)

    if prop_pt is not None:
        ax.scatter([prop_pt[0]], [prop_pt[1]], marker="*", s=150, label="First-order", zorder=6)
    if bhkf_pt is not None:
        ax.scatter([bhkf_pt[0]], [bhkf_pt[1]], marker="x", s=95, label="BHKF", zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("runtime (s, log)")
    ax.set_ylabel("W2rms (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    savefig(fig, outpath)

def plot_mc_w2_distribution_vs_runtime(
    mc_dist: Dict[int, List[Tuple[float, float]]],
    *,
    prop_pt: Tuple[float, float] | None,
    bhkf_pt: Tuple[float, float] | None,
    title: str,
    outpath: Path,
    use_median_runtime: bool = True,
):
    """MC W2rms distribution (y) vs runtime (x).
    """
    Ns = sorted(mc_dist.keys())

    positions: list[float] = []
    data: list[np.ndarray] = []
    kept_Ns: list[int] = []

    for N in Ns:
        vals = mc_dist[N]
        if len(vals) == 0:
            continue

        rts = np.asarray([v[0] for v in vals], dtype=float)
        w2s = np.asarray([v[1] for v in vals], dtype=float)

        x = float(np.median(rts) if use_median_runtime else np.mean(rts))
        positions.append(x)
        data.append(w2s)
        kept_Ns.append(int(N))

    positions = np.asarray(positions, dtype=float)

    if positions.size == 0:
        raise ValueError("mc_dist is empty; cannot make runtime comparison plot.")

    # Widths in data units; on log-x choose widths based on local spacing in decades.
    if positions.size == 1:
        widths = np.asarray([0.12 * positions[0]], dtype=float)
    else:
        logpos = np.log10(positions)
        dlog = np.diff(logpos)

        left_gap = np.r_[np.inf, dlog]
        right_gap = np.r_[dlog, np.inf]
        min_gap = np.minimum(left_gap, right_gap)

        w_dec = np.clip(0.35 * min_gap, 0.015, 0.06)
        widths = positions * (10.0 ** w_dec - 10.0 ** (-w_dec))

    fig, ax = plt.subplots()

    lw = float(plt.rcParams.get("lines.linewidth", 1.2))
    edge_col = plt.rcParams.get("axes.edgecolor", "0.15")

    cycle = plt.rcParams.get("axes.prop_cycle", None)
    cols = cycle.by_key().get("color", []) if cycle is not None else []
    median_col = cols[1] if len(cols) >= 2 else "C1"

    ax.boxplot(
        [d.tolist() for d in data],
        positions=positions,
        widths=widths,
        showfliers=False,
        manage_ticks=False,
        boxprops={"linewidth": lw, "color": edge_col},
        whiskerprops={"linewidth": lw, "color": edge_col},
        capprops={"linewidth": lw, "color": edge_col},
        medianprops={"linewidth": lw, "color": median_col},
    )

    if prop_pt is not None:
        ax.scatter(
            [prop_pt[0]],
            [prop_pt[1]],
            marker="x",
            s=70,
            label="First-order",
            zorder=6,
        )
    if bhkf_pt is not None:
        ax.scatter(
            [bhkf_pt[0]],
            [bhkf_pt[1]],
            marker="x",
            s=70,
            label="BHKF",
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("runtime (s, log)")
    ax.set_ylabel("W2rms (log)")
    ax.set_title("\n".join(textwrap.wrap(title, width=46)), pad=7)

    ax.grid(True, which="major", axis="y", alpha=0.30)
    ax.grid(False, which="minor", axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.06)

    def _fmt_N(N: int) -> str:
        if N >= 1000 and (N % 1000 == 0):
            return f"{N // 1000}k"
        return str(N)

    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    top.set_xticks(positions)
    top.set_xticklabels([_fmt_N(N) for N in kept_Ns])
    top.set_xlabel("Monte Carlo samples", labelpad=6)

    top.xaxis.set_ticks_position("top")
    top.xaxis.set_label_position("top")
    top.tick_params(axis="x", which="major", direction="out", pad=3)
    top.tick_params(axis="x", which="minor", length=0)
    top.minorticks_off()
    top.grid(False)

    top.spines["top"].set_visible(True)
    top.spines["bottom"].set_visible(False)
    top.spines["left"].set_visible(False)
    top.spines["right"].set_visible(False)

    if any(p is not None for p in (prop_pt, bhkf_pt)):
        ax.legend(
            loc="best",
            frameon=True,
            handlelength=1.0,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.25,
        )

    fig.tight_layout(pad=0.25, rect=(0.0, 0.0, 1.0, 0.95))
    savefig(fig, outpath)
