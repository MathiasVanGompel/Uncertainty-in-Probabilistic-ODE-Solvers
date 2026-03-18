import argparse
import time
from pathlib import Path
import numpy as np

from metrics import trajectory_gaussian_w2, trajectory_metric_summary, w2_rms
from monte_carlo import mc_accuracy_distribution, mc_reference_cached
from plotstyle import set_figure_format, setup_matplotlib
from plotting import (
    clip_pos,
    mean_error_t,
    mean_rel_error_t,
    var_error_t,
    plot_accuracy_vs_runtime,
    plot_bands_all_methods,
    plot_bands_prop_total_vs_only,
    plot_error_time_series,
    plot_final_time_summary,
    plot_mc_w2_distribution_vs_runtime,
    plot_variance_component_comparison,
    plot_variance_decomposition_line,
    plot_variance_decomposition_ratio,
    plot_variance_decomposition_stack,
)
from problems import problems_paper
from propagation import (
    propagate_bhkf_decomposed,
    propagate_prop_decomposed,
)

def build_arg_parser() -> argparse.ArgumentParser:
    probs = problems_paper()

    ap = argparse.ArgumentParser(
        description=(
            "Benchmark probabilistic ODE uncertainty propagation"
        )
    )
    ap.add_argument("--problem", type=str, default="logistic", choices=sorted(probs.keys()))
    ap.add_argument("--t-eval", type=int, default=200, help="Number of evaluation points in [t0, T].")

    ap.add_argument("--n-ref", type=int, default=1_000_000)
    ap.add_argument("--ref-seed", type=int, default=0)
    ap.add_argument("--mc-cache", type=str, default="mc_cache")
    ap.add_argument("--mc-overwrite", action="store_true")

    ap.add_argument(
        "--mc-sizes",
        type=str,
        default="200,500,1000,2000,5000,10000,20000,50000,100000",
        help="Comma-separated MC sample sizes.",
    )
    ap.add_argument("--mc-reps", type=int, default=10)
    ap.add_argument("--mc-single", type=int, default=100_000, help="This MC size is run only once.")
    ap.add_argument("--mc-seed", type=int, default=1)
    ap.add_argument("--mc-chunk", type=int, default=512)
    ap.add_argument("--mc-rtol", type=float, default=1e-10)
    ap.add_argument("--mc-atol", type=float, default=1e-12)
    ap.add_argument("--mc-method", type=str, default="DOP853")

    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--no-smoother", action="store_true")
    ap.add_argument("--no-precondition", action="store_true")
    ap.add_argument("--pn-rtol", type=float, default=1e-5)
    ap.add_argument("--pn-atol", type=float, default=1e-9)
    ap.add_argument("--R", type=float, default=1e-12)
    ap.add_argument("--diffusion-init", type=float, default=1.0)
    ap.add_argument("--h0", type=float, default=1e-2)
    ap.add_argument("--h-max", type=float, default=1.0)

    ap.add_argument("--gh-order", type=int, default=12)
    ap.add_argument("--bq-ell", type=float, default=1.0)
    ap.add_argument("--bq-alpha2", type=float, default=1.0)
    ap.add_argument("--no-bq-inflation", action="store_true")
    ap.add_argument("--no-bq-cov", action="store_true")
    ap.add_argument("--run-sigma-point", action="store_true", help="Also run classical Gauss--Hermite sigma-point propagation.")

    ap.add_argument("--compare-comp", type=int, default=0)
    ap.add_argument("--figdir", type=str, default="figs_uncertainty")
    ap.add_argument("--fig-format", type=str, default="pdf", choices=["pdf", "png"])
    return ap

def parse_int_csv(raw: str) -> list[int]:
    return [int(tok) for tok in str(raw).split(",") if tok.strip()]

def main() -> None:
    probs = problems_paper()
    args = build_arg_parser().parse_args()

    set_figure_format(args.fig_format)
    setup_matplotlib()

    prob = probs[str(args.problem)]
    t0, t1 = float(prob.t_span[0]), float(prob.t_span[1])
    t_eval = np.linspace(t0, t1, int(args.t_eval), dtype=float)
    figdir = args.figdir

    ref_path, _t_ref, mean_ref, cov_ref = mc_reference_cached(
        prob,
        n_ref=int(args.n_ref),
        t_eval=t_eval,
        cache_dir=args.mc_cache,
        seed=int(args.ref_seed),
        rtol=float(args.mc_rtol),
        atol=float(args.mc_atol),
        method=str(args.mc_method),
        chunk_size=int(args.mc_chunk),
        overwrite=bool(args.mc_overwrite),
    )
    print(f"[ref] loaded/computed MC reference at {ref_path}")

    prop_t0 = time.perf_counter()
    prop = propagate_prop_decomposed(
        prob,
        t_eval=t_eval,
        q=int(args.q),
        use_smoother=not bool(args.no_smoother),
        precondition=not bool(args.no_precondition),
        atol=float(args.pn_atol),
        rtol=float(args.pn_rtol),
        R=float(args.R),
        diffusion_init=float(args.diffusion_init),
        h0=float(args.h0),
        h_max=float(args.h_max),
    )
    prop_rt = float(time.perf_counter() - prop_t0)
    prop_w2_total = w2_rms(prop.mean, prop.cov_total, mean_ref, cov_ref)
    prop_w2_input = w2_rms(prop.mean, prop.cov_input, mean_ref, cov_ref)
    prop_metrics = trajectory_metric_summary(prop.mean, prop.cov_total, mean_ref, cov_ref)
    print(
        f"[prop] runtime={prop_rt:.3f}s  W2rms_total={prop_w2_total:.4e}  "
        f"W2rms_input={prop_w2_input:.4e}  metrics={prop_metrics}"
    )

    bhkf_t0 = time.perf_counter()
    bhkf = propagate_bhkf_decomposed(
        prob,
        t_eval=t_eval,
        q=int(args.q),
        use_smoother=not bool(args.no_smoother),
        precondition=not bool(args.no_precondition),
        atol=float(args.pn_atol),
        rtol=float(args.pn_rtol),
        R=float(args.R),
        diffusion_init=float(args.diffusion_init),
        h0=float(args.h0),
        h_max=float(args.h_max),
        gh_order=int(args.gh_order),
        bq_ell=float(args.bq_ell),
        bq_alpha2=float(args.bq_alpha2),
        include_bq_inflation=not bool(args.no_bq_inflation),
        use_bq_covariance=not bool(args.no_bq_cov),
    )
    bhkf_rt = float(time.perf_counter() - bhkf_t0)
    bhkf_w2_total = w2_rms(bhkf.mean, bhkf.cov_total, mean_ref, cov_ref)
    bhkf_w2_input = w2_rms(bhkf.mean, bhkf.cov_input, mean_ref, cov_ref)
    bhkf_metrics = trajectory_metric_summary(bhkf.mean, bhkf.cov_total, mean_ref, cov_ref)
    print(
        f"[BHKF] runtime={bhkf_rt:.3f}s  W2rms_total={bhkf_w2_total:.4e}  "
        f"W2rms_input={bhkf_w2_input:.4e}  metrics={bhkf_metrics}"
    )

    sizes = parse_int_csv(args.mc_sizes)
    mc_dist = mc_accuracy_distribution(
        prob,
        t_eval=t_eval,
        mean_ref=mean_ref,
        cov_ref=cov_ref,
        sample_sizes=sizes,
        reps=int(args.mc_reps),
        single_run_n=int(args.mc_single),
        seed=int(args.mc_seed),
        rtol=float(args.mc_rtol),
        atol=float(args.mc_atol),
        method=str(args.mc_method),
        chunk_size=int(args.mc_chunk),
        cache_dir=(Path(args.mc_cache) / "mc_compare"),
        overwrite=bool(args.mc_overwrite),
    )

    comp = int(args.compare_comp)
    title_map = {
        "linear": "Linear",
        "logistic": "Logistic",
        "lv": "Lotka–Volterra",
    }
    title = title_map.get(prob.name, prob.name)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    plot_bands_prop_total_vs_only(
        t_eval,
        mean_ref,
        cov_ref,
        prop,
        comp=comp,
        title=title,
        outpath=figdir / f"{prob.name}_prop_total_vs_only_comp{comp}.png",
    )
    plot_bands_all_methods(
        t_eval,
        mean_ref,
        cov_ref,
        prop,
        bhkf,
        comp=comp,
        title=title,
        outpath=figdir / f"{prob.name}_bands_all_comp{comp}.png",
    )

    for name, dec in [("prop", prop), ("bhkf", bhkf)]:
        plot_variance_decomposition_line(
            t_eval,
            dec,
            comp=comp,
            title=title,
            outpath=figdir / f"{prob.name}_{name}_var_decomp_line_comp{comp}.png",
        )
        plot_variance_decomposition_stack(
            t_eval,
            dec,
            comp=comp,
            title=title,
            outpath=figdir / f"{prob.name}_{name}_var_decomp_stack_comp{comp}.png",
        )
        plot_variance_decomposition_ratio(
            t_eval,
            dec,
            comp=comp,
            title=title,
            outpath=figdir / f"{prob.name}_{name}_var_decomp_ratio_comp{comp}.png",
        )

    plot_variance_component_comparison(
        t_eval,
        prop,
        bhkf,
        comp=comp,
        outdir=figdir,
        prob_name=prob.name,
    )

    w2_prop_t = trajectory_gaussian_w2(prop.mean, prop.cov_total, mean_ref, cov_ref)
    w2_bhkf_t = trajectory_gaussian_w2(bhkf.mean, bhkf.cov_total, mean_ref, cov_ref)
    series_w2 = {"First-order total": w2_prop_t, "BHKF total": w2_bhkf_t}
    plot_error_time_series(
        t_eval,
        series_w2,
        title=title,
        ylabel="W2",
        outpath=figdir / f"{prob.name}_w2_over_time.png",
    )

    series_mean_abs = {
        "First-order": mean_error_t(prop.mean, mean_ref),
        "BHKF": mean_error_t(bhkf.mean, mean_ref),
    }
    series_mean_rel = {
        "First-order": mean_rel_error_t(prop.mean, mean_ref),
        "BHKF": mean_rel_error_t(bhkf.mean, mean_ref),
    }

    plot_error_time_series(
        t_eval,
        series_mean_abs,
        title=title,
        ylabel="mean error (abs)",
        outpath=figdir / f"{prob.name}_mean_error_abs.png",
    )
    plot_error_time_series(
        t_eval,
        series_mean_rel,
        title=title,
        ylabel="mean error (rel)",
        outpath=figdir / f"{prob.name}_mean_error_rel.png",
    )

    series_var_abs = {
        "First-order total var": var_error_t(prop.cov_total, cov_ref, comp),
        "First-order input var": var_error_t(prop.cov_input, cov_ref, comp),
        "BHKF total var": var_error_t(bhkf.cov_total, cov_ref, comp),
        "BHKF input var": var_error_t(bhkf.cov_input, cov_ref, comp),
    }

    plot_error_time_series(
        t_eval,
        series_var_abs,
        title=title,
        ylabel="variance error (abs)",
        outpath=figdir / f"{prob.name}_var_error_abs_comp{comp}.png",
    )

    plot_accuracy_vs_runtime(
        mc_dist,
        prop_pt=(prop_rt, prop_w2_input),
        bhkf_pt=(bhkf_rt, bhkf_w2_input),
        title=title,
        outpath=figdir / f"{prob.name}_accuracy_vs_runtime.png",
    )
    plot_mc_w2_distribution_vs_runtime(
        mc_dist,
        prop_pt=(prop_rt, prop_w2_input),
        bhkf_pt=(bhkf_rt, bhkf_w2_input),
        title=title,
        outpath=figdir / f"{prob.name}_mc_w2_distribution_runtime.png",
    )

    k_last = -1
    summary = {
        "First-order W2(t_end)": float(clip_pos(w2_prop_t[k_last])),
        "BHKF W2(t_end)": float(clip_pos(w2_bhkf_t[k_last])),
        "First-order mean err(t_end)": float(clip_pos(series_mean_abs["First-order"][k_last])),
        "BHKF mean err(t_end)": float(clip_pos(series_mean_abs["BHKF"][k_last])),
        "First-order total var err(t_end)": float(clip_pos(series_var_abs["First-order total var"][k_last])),
        "First-order input var err(t_end)": float(clip_pos(series_var_abs["First-order input var"][k_last])),
        "BHKF total var err(t_end)": float(clip_pos(series_var_abs["BHKF total var"][k_last])),
        "BHKF input var err(t_end)": float(clip_pos(series_var_abs["BHKF input var"][k_last])),
    }

    plot_final_time_summary(
        t_eval,
        summary,
        title=f"{prob.name}: final-time error snapshot (log scale)",
        outpath=figdir / f"{prob.name}_final_time_summary.png",
    )

    print(f"[done] figures written to: {figdir}")

if __name__ == "__main__":
    main()
