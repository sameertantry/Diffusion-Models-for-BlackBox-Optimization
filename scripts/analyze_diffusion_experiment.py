#!/usr/bin/env python3
"""
Comprehensive analysis of a DiffusionOptimizer experiment.

Reads verbose_logs (.npz), experiment_results.csv, and experiment_config.json
from a COCO/BBOB experiment directory and produces:
  1. A PDF report with convergence, diversity, tau-evolution, and anomaly plots.
  2. A textual summary printed to stdout (and saved as analysis_report.txt).

Usage
-----
    python scripts/analyze_diffusion_experiment.py <experiment_dir> [--output-dir DIR]

Example
-------
    python scripts/analyze_diffusion_experiment.py \
        exdata/DiffusionV2/model-scale-tuning-0011
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

BBOB_GROUPS: Dict[str, List[int]] = {
    "Separable (f1-f5)": [1, 2, 3, 4, 5],
    "Moderate (f6-f9)": [6, 7, 8, 9],
    "Ill-conditioned (f10-f14)": [10, 11, 12, 13, 14],
    "Multi-modal (f15-f19)": [15, 16, 17, 18, 19],
    "Weakly-structured (f20-f24)": [20, 21, 22, 23, 24],
}
BBOB_GROUP_FOR_FUNC = {}
for grp, fids in BBOB_GROUPS.items():
    for fid in fids:
        BBOB_GROUP_FOR_FUNC[fid] = grp


@dataclass
class RunData:
    func_id: int
    dim: int
    instance: int
    npz_path: str
    samples: np.ndarray
    values: np.ndarray      # original f(x)
    indexes: np.ndarray      # 1-based iteration index
    taus: np.ndarray         # min(y_data) = -max(f(x)) in elite buffer
    fopt: Optional[float] = None
    best_csv: Optional[float] = None
    converged: Optional[bool] = None


def parse_npz_filename(fname: str) -> Tuple[int, int, int]:
    parts = fname.split("_")
    func_id = int(parts[1][1:])
    dim = int(parts[2][1:])
    inst = int(parts[3][1:])
    return func_id, dim, inst


def load_experiment(exp_dir: str) -> Tuple[List[RunData], Dict[str, Any], pd.DataFrame]:
    exp_dir = Path(exp_dir)

    with open(exp_dir / "experiment_config.json") as f:
        config = json.load(f)

    results_df = pd.read_csv(exp_dir / "experiment_results.csv")

    vlog_dir = exp_dir / "verbose_logs"
    npz_files = sorted(vlog_dir.glob("*.npz"))

    runs: List[RunData] = []
    for npz_path in npz_files:
        fid, dim, inst = parse_npz_filename(npz_path.name)
        data = np.load(str(npz_path))

        row = results_df[
            (results_df["function_id"] == fid)
            & (results_df["dimension"] == dim)
            & (results_df["instance_id"] == inst)
        ]
        fopt = float(row["fopt"].iloc[0]) if len(row) else None
        best_csv = float(row["best_value"].iloc[0]) if len(row) else None
        converged = bool(row["converged"].iloc[0]) if len(row) else None

        runs.append(RunData(
            func_id=fid, dim=dim, instance=inst,
            npz_path=str(npz_path),
            samples=data["samples"],
            values=data["values"].flatten(),
            indexes=data["indexes"].flatten(),
            taus=data["taus"].flatten(),
            fopt=fopt, best_csv=best_csv, converged=converged,
        ))

    return runs, config, results_df


# ======================================================================= #
#  Per-run derived metrics                                                 #
# ======================================================================= #

def per_iteration_stats(run: RunData) -> pd.DataFrame:
    """Aggregate statistics per ask/tell iteration."""
    iters = np.unique(run.indexes)
    rows = []
    best_so_far = np.inf
    cumulative_evals = 0
    for it in iters:
        mask = run.indexes == it
        vals = run.values[mask]
        samps = run.samples[mask]
        tau = run.taus[mask][0]
        n = mask.sum()
        cumulative_evals += n
        iter_best = vals.min()
        best_so_far = min(best_so_far, iter_best)

        per_dim_std = samps.std(axis=0)

        rows.append({
            "iteration": int(it),
            "n_samples": n,
            "cumulative_evals": cumulative_evals,
            "val_min": vals.min(),
            "val_median": np.median(vals),
            "val_mean": vals.mean(),
            "val_max": vals.max(),
            "val_std": vals.std(),
            "val_p25": np.percentile(vals, 25),
            "val_p75": np.percentile(vals, 75),
            "best_so_far": best_so_far,
            "gap": best_so_far - run.fopt if run.fopt is not None else np.nan,
            "tau": tau,
            "tau_as_fmax": -tau,  # max f(x) in elite buffer
            "sample_std_mean": per_dim_std.mean(),
            "sample_std_min": per_dim_std.min(),
            "sample_std_max": per_dim_std.max(),
            "sample_range_mean": (samps.max(axis=0) - samps.min(axis=0)).mean(),
            "sample_centroid_shift": np.nan,  # filled below
        })

    df = pd.DataFrame(rows)
    if len(df) > 1:
        for i in range(1, len(df)):
            prev_mask = run.indexes == df.iloc[i - 1]["iteration"]
            curr_mask = run.indexes == df.iloc[i]["iteration"]
            c_prev = run.samples[prev_mask].mean(axis=0)
            c_curr = run.samples[curr_mask].mean(axis=0)
            df.loc[df.index[i], "sample_centroid_shift"] = float(np.linalg.norm(c_curr - c_prev))
    return df


# ======================================================================= #
#  Anomaly detection                                                       #
# ======================================================================= #

@dataclass
class Anomaly:
    run_label: str
    category: str
    iteration: Optional[int]
    description: str
    severity: str  # "low", "medium", "high", "critical"


def detect_anomalies(run: RunData, it_df: pd.DataFrame) -> List[Anomaly]:
    label = f"f{run.func_id}_d{run.dim}_i{run.instance}"
    anomalies: List[Anomaly] = []

    # 1. Tau regression: tau should be monotonically non-decreasing
    #    (since tau = min(y_data) = -max(f(x)), improving means tau increases)
    taus_per_iter = it_df["tau"].values
    for i in range(1, len(taus_per_iter)):
        if taus_per_iter[i] < taus_per_iter[i - 1] - 1e-8:
            anomalies.append(Anomaly(
                label, "tau_regression", int(it_df.iloc[i]["iteration"]),
                f"tau decreased from {taus_per_iter[i-1]:.4g} to {taus_per_iter[i]:.4g} "
                f"(delta={taus_per_iter[i]-taus_per_iter[i-1]:.4g})",
                "medium",
            ))

    # 2. Value explosion: max(f(x)) in iteration >> best_so_far
    for _, row in it_df.iterrows():
        ratio = row["val_max"] / max(abs(row["best_so_far"]), 1e-10)
        if ratio > 1e4:
            anomalies.append(Anomaly(
                label, "value_explosion", int(row["iteration"]),
                f"max(f(x))={row['val_max']:.4g} vs best_so_far={row['best_so_far']:.4g} "
                f"(ratio={ratio:.1f}x)",
                "high" if ratio > 1e6 else "medium",
            ))

    # 3. Sample collapse: std drops below threshold
    for _, row in it_df.iterrows():
        if row["sample_std_mean"] < 0.01:
            anomalies.append(Anomaly(
                label, "sample_collapse", int(row["iteration"]),
                f"avg per-dim std={row['sample_std_mean']:.6f}",
                "high",
            ))

    # 4. Stagnation: no improvement in best_so_far for >25% of total iterations
    n_iters = len(it_df)
    stag_window = max(int(n_iters * 0.25), 3)
    gaps = it_df["gap"].values
    for i in range(stag_window, n_iters):
        if abs(gaps[i] - gaps[i - stag_window]) < 1e-10:
            anomalies.append(Anomaly(
                label, "stagnation", int(it_df.iloc[i]["iteration"]),
                f"no improvement for {stag_window} iterations "
                f"(gap={gaps[i]:.4g})",
                "medium" if gaps[i] > 1.0 else "low",
            ))
            break

    # 5. Final gap still large
    if run.fopt is not None:
        final_gap = it_df.iloc[-1]["gap"]
        if final_gap > 10.0:
            anomalies.append(Anomaly(
                label, "large_final_gap", None,
                f"final gap={final_gap:.4g} (fopt={run.fopt:.4g}, best={it_df.iloc[-1]['best_so_far']:.4g})",
                "critical" if final_gap > 100 else "high",
            ))

    # 6. Exploration producing mostly terrible points (median >> best)
    late_start = max(1, int(n_iters * 0.5))
    late_df = it_df.iloc[late_start:]
    if len(late_df) > 0:
        median_ratio = late_df["val_median"].median() / max(abs(late_df["best_so_far"].iloc[-1]), 1e-10)
        if median_ratio > 100:
            anomalies.append(Anomaly(
                label, "poor_late_sampling", None,
                f"median f(x) in late iterations is {median_ratio:.0f}x worse than best",
                "high",
            ))

    # 7. Non-monotonic best_so_far (should never happen)
    bsf = it_df["best_so_far"].values
    for i in range(1, len(bsf)):
        if bsf[i] > bsf[i - 1] + 1e-10:
            anomalies.append(Anomaly(
                label, "best_regression", int(it_df.iloc[i]["iteration"]),
                f"best_so_far increased from {bsf[i-1]:.4g} to {bsf[i]:.4g}",
                "critical",
            ))
            break

    return anomalies


# ======================================================================= #
#  Plotting helpers                                                         #
# ======================================================================= #

def plot_convergence_grid(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    pdf: PdfPages,
):
    """One page per dimension: convergence curves for all functions."""
    dims = sorted(set(r.dim for r in runs))
    for dim in dims:
        dim_runs = [r for r in runs if r.dim == dim]
        funcs = sorted(set(r.func_id for r in dim_runs))
        n_funcs = len(funcs)
        ncols = 6
        nrows = (n_funcs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Convergence — Dimension {dim}", fontsize=16, y=1.02)

        for idx, fid in enumerate(funcs):
            ax = axes[idx // ncols, idx % ncols]
            for r in dim_runs:
                if r.func_id != fid:
                    continue
                key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
                df = it_dfs[key]
                ax.semilogy(
                    df["cumulative_evals"], df["gap"].clip(lower=1e-12),
                    label=f"inst {r.instance}", linewidth=1.5,
                )
            if dim_runs[0].fopt is not None:
                ax.axhline(0.1, color="green", linestyle="--", alpha=0.5, label="precision=0.1")
            ax.set_title(f"f{fid} ({BBOB_GROUP_FOR_FUNC.get(fid, '?')})", fontsize=9)
            ax.set_xlabel("evaluations")
            ax.set_ylabel("gap to fopt")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(n_funcs, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_tau_evolution(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    pdf: PdfPages,
):
    """Tau (elite buffer floor) evolution over iterations."""
    dims = sorted(set(r.dim for r in runs))
    for dim in dims:
        dim_runs = [r for r in runs if r.dim == dim]
        funcs = sorted(set(r.func_id for r in dim_runs))
        n_funcs = len(funcs)
        ncols = 6
        nrows = (n_funcs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Tau Evolution (= −max(f(x)) in buffer) — Dimension {dim}", fontsize=16, y=1.02)

        for idx, fid in enumerate(funcs):
            ax = axes[idx // ncols, idx % ncols]
            for r in dim_runs:
                if r.func_id != fid:
                    continue
                key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
                df = it_dfs[key]
                tau_vals = df["tau"].values
                ax.plot(
                    df["iteration"], tau_vals,
                    label=f"inst {r.instance}", linewidth=1.5,
                )
            ax.set_title(f"f{fid}", fontsize=9)
            ax.set_xlabel("iteration")
            ax.set_ylabel("tau (−max f(x) in buffer)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(n_funcs, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_value_distribution(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    pdf: PdfPages,
):
    """Per-iteration value distribution (percentile bands) for selected runs."""
    dims = sorted(set(r.dim for r in runs))
    for dim in dims:
        dim_runs = [r for r in runs if r.dim == dim and r.instance == 1]
        funcs = sorted(set(r.func_id for r in dim_runs))
        n_funcs = len(funcs)
        ncols = 6
        nrows = (n_funcs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Value Distribution (instance 1) — Dimension {dim}", fontsize=16, y=1.02)

        for idx, fid in enumerate(funcs):
            ax = axes[idx // ncols, idx % ncols]
            r_list = [r for r in dim_runs if r.func_id == fid]
            if not r_list:
                continue
            r = r_list[0]
            key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
            df = it_dfs[key]

            ax.fill_between(df["iteration"], df["val_p25"], df["val_p75"],
                            alpha=0.3, label="p25-p75")
            ax.plot(df["iteration"], df["val_median"], label="median", linewidth=1.5)
            ax.plot(df["iteration"], df["best_so_far"], label="best", linewidth=1.5, color="red")
            if r.fopt is not None:
                ax.axhline(r.fopt, color="green", linestyle="--", alpha=0.5, label="fopt")
            ax.set_title(f"f{fid}", fontsize=9)
            ax.set_xlabel("iteration")
            ax.set_ylabel("f(x)")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        for idx in range(n_funcs, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_sample_diversity(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    pdf: PdfPages,
):
    """Sample diversity (per-dim std) over iterations."""
    dims = sorted(set(r.dim for r in runs))
    for dim in dims:
        dim_runs = [r for r in runs if r.dim == dim and r.instance == 1]
        funcs = sorted(set(r.func_id for r in dim_runs))
        n_funcs = len(funcs)
        ncols = 6
        nrows = (n_funcs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Sample Diversity (mean per-dim std, inst 1) — Dimension {dim}", fontsize=16, y=1.02)

        for idx, fid in enumerate(funcs):
            ax = axes[idx // ncols, idx % ncols]
            r_list = [r for r in dim_runs if r.func_id == fid]
            if not r_list:
                continue
            r = r_list[0]
            key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
            df = it_dfs[key]

            ax.plot(df["iteration"], df["sample_std_mean"], label="mean std", linewidth=1.5)
            ax.fill_between(df["iteration"], df["sample_std_min"], df["sample_std_max"],
                            alpha=0.2, label="min-max std")
            ax.set_title(f"f{fid}", fontsize=9)
            ax.set_xlabel("iteration")
            ax.set_ylabel("per-dim std of samples")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(n_funcs, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_group_summary(results_df: pd.DataFrame, pdf: PdfPages):
    """Success rate and median gap per BBOB function group and dimension."""
    results_df = results_df.copy()
    results_df["group"] = results_df["function_id"].map(BBOB_GROUP_FOR_FUNC)
    dims = sorted(results_df["dimension"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Performance by BBOB Function Group", fontsize=14)

    groups = list(BBOB_GROUPS.keys())
    x = np.arange(len(groups))
    width = 0.25

    for i, dim in enumerate(dims):
        sub = results_df[results_df["dimension"] == dim]
        success_rates = []
        median_gaps = []
        for grp in groups:
            grp_df = sub[sub["group"] == grp]
            if len(grp_df) == 0:
                success_rates.append(0)
                median_gaps.append(np.nan)
            else:
                success_rates.append(grp_df["converged"].mean() * 100)
                median_gaps.append(grp_df["gap"].median())
        axes[0].bar(x + i * width, success_rates, width, label=f"dim={dim}")
        axes[1].bar(x + i * width, median_gaps, width, label=f"dim={dim}")

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Convergence Rate")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Median gap to fopt")
    axes[1].set_yscale("log")
    axes[1].set_title("Median Gap (log scale)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_scaling(results_df: pd.DataFrame, pdf: PdfPages):
    """How gap and success rate scale with dimension."""
    dims = sorted(results_df["dimension"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Dimension Scaling", fontsize=14)

    mean_gaps = [results_df[results_df["dimension"] == d]["gap"].mean() for d in dims]
    median_gaps = [results_df[results_df["dimension"] == d]["gap"].median() for d in dims]
    success_rates = [results_df[results_df["dimension"] == d]["converged"].mean() * 100 for d in dims]

    axes[0].bar(range(len(dims)), success_rates, tick_label=[str(d) for d in dims])
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Convergence rate (%)")
    axes[0].set_title("Success Rate vs Dimension")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(dims)), median_gaps, tick_label=[str(d) for d in dims])
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Median gap")
    axes[1].set_yscale("log")
    axes[1].set_title("Median Gap vs Dimension")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(range(len(dims)), mean_gaps, tick_label=[str(d) for d in dims])
    axes[2].set_xlabel("Dimension")
    axes[2].set_ylabel("Mean gap")
    axes[2].set_yscale("log")
    axes[2].set_title("Mean Gap vs Dimension")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_per_function_heatmap(results_df: pd.DataFrame, pdf: PdfPages):
    """Heatmap: log10(gap) for each (function, dimension) averaged over instances."""
    pivot = results_df.groupby(["function_id", "dimension"])["gap"].mean().unstack()
    log_pivot = np.log10(pivot.clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(log_pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(log_pivot.columns)))
    ax.set_xticklabels([str(c) for c in log_pivot.columns])
    ax.set_yticks(range(len(log_pivot.index)))
    ax.set_yticklabels([f"f{int(f)}" for f in log_pivot.index])
    ax.set_xlabel("Dimension")
    ax.set_ylabel("BBOB Function")
    ax.set_title("log₁₀(mean gap) — Red=bad, Green=good")
    fig.colorbar(im, ax=ax, shrink=0.6)

    for i in range(log_pivot.shape[0]):
        for j in range(log_pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=7,
                    color="white" if log_pivot.values[i, j] > 2 else "black")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_improvement_rate(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    pdf: PdfPages,
):
    """Where does improvement happen: early vs late budget fraction."""
    rows = []
    for r in runs:
        key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
        df = it_dfs[key]
        if len(df) < 4 or r.fopt is None:
            continue
        total_iters = len(df)
        mid = total_iters // 2
        gap_start = df.iloc[0]["gap"]
        gap_mid = df.iloc[mid]["gap"]
        gap_end = df.iloc[-1]["gap"]
        improvement_first_half = max(gap_start - gap_mid, 0)
        improvement_second_half = max(gap_mid - gap_end, 0)
        total_improvement = max(gap_start - gap_end, 1e-15)
        rows.append({
            "func_id": r.func_id, "dim": r.dim, "instance": r.instance,
            "group": BBOB_GROUP_FOR_FUNC.get(r.func_id, "?"),
            "first_half_frac": improvement_first_half / total_improvement,
            "second_half_frac": improvement_second_half / total_improvement,
            "gap_start": gap_start, "gap_mid": gap_mid, "gap_end": gap_end,
        })

    if not rows:
        return
    imp_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("When Does Improvement Happen?", fontsize=14)

    for i, dim in enumerate(sorted(imp_df["dim"].unique())):
        sub = imp_df[imp_df["dim"] == dim]
        axes[0].scatter(
            sub["func_id"], sub["first_half_frac"],
            label=f"dim={dim}", alpha=0.7, s=40,
        )
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Function ID")
    axes[0].set_ylabel("Fraction of improvement in first half")
    axes[0].set_title("First-half improvement fraction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    grouped = imp_df.groupby("group")["first_half_frac"].mean()
    axes[1].barh(range(len(grouped)), grouped.values)
    axes[1].set_yticks(range(len(grouped)))
    axes[1].set_yticklabels(grouped.index, fontsize=9)
    axes[1].set_xlabel("Mean first-half improvement fraction")
    axes[1].set_title("By function group")
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_drift(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    pdf: PdfPages,
):
    """Centroid shift between consecutive iterations — measures search movement."""
    dims = sorted(set(r.dim for r in runs))
    for dim in dims:
        dim_runs = [r for r in runs if r.dim == dim and r.instance == 1]
        funcs = sorted(set(r.func_id for r in dim_runs))
        n_funcs = len(funcs)
        ncols = 6
        nrows = (n_funcs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Centroid Drift (inst 1) — Dimension {dim}", fontsize=16, y=1.02)

        for idx, fid in enumerate(funcs):
            ax = axes[idx // ncols, idx % ncols]
            r_list = [r for r in dim_runs if r.func_id == fid]
            if not r_list:
                continue
            r = r_list[0]
            key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
            df = it_dfs[key]
            ax.plot(df["iteration"], df["sample_centroid_shift"], linewidth=1.5)
            ax.set_title(f"f{fid}", fontsize=9)
            ax.set_xlabel("iteration")
            ax.set_ylabel("‖Δcentroid‖")
            ax.grid(True, alpha=0.3)

        for idx in range(n_funcs, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def plot_exploitation_quality(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    pdf: PdfPages,
):
    """Ratio of median(f(x)) to best_so_far — measures how focused sampling is."""
    dims = sorted(set(r.dim for r in runs))
    for dim in dims:
        dim_runs = [r for r in runs if r.dim == dim and r.instance == 1]
        funcs = sorted(set(r.func_id for r in dim_runs))
        n_funcs = len(funcs)
        ncols = 6
        nrows = (n_funcs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Exploitation Quality: median(f)/best (inst 1) — Dim {dim}", fontsize=14, y=1.02)

        for idx, fid in enumerate(funcs):
            ax = axes[idx // ncols, idx % ncols]
            r_list = [r for r in dim_runs if r.func_id == fid]
            if not r_list:
                continue
            r = r_list[0]
            key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
            df = it_dfs[key]
            ratio = df["val_median"] / df["best_so_far"].clip(lower=1e-10).abs()
            ax.semilogy(df["iteration"], ratio.clip(lower=1e-3), linewidth=1.5)
            ax.axhline(1.0, color="green", linestyle="--", alpha=0.5)
            ax.set_title(f"f{fid}", fontsize=9)
            ax.set_xlabel("iteration")
            ax.set_ylabel("median / |best|")
            ax.grid(True, alpha=0.3)

        for idx in range(n_funcs, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ======================================================================= #
#  Text report                                                              #
# ======================================================================= #

def build_text_report(
    runs: List[RunData],
    it_dfs: Dict[str, pd.DataFrame],
    all_anomalies: List[Anomaly],
    config: Dict[str, Any],
    results_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("  DIFFUSION OPTIMIZER EXPERIMENT ANALYSIS REPORT")
    lines.append(sep)
    lines.append("")

    # Config summary
    lines.append("EXPERIMENT CONFIGURATION")
    lines.append("-" * 40)
    opt = config.get("optimizer", config)
    for k in ["num_timesteps", "beta_schedule", "prediction_type", "clip_sample",
              "hidden_dim", "depth", "batch_size", "lr_diffusion", "train_steps",
              "elite_per_dim", "elite_min", "explore_frac", "rank_temperature",
              "x_noise_std", "p_uncond", "cfg_scale", "ema_decay", "min_data",
              "conditioning", "noise_pred_arch"]:
        if k in opt:
            lines.append(f"  {k}: {opt[k]}")
    lines.append(f"  budget_multiplier: {config.get('budget_multiplier', '?')}")
    lines.append(f"  eval_batch_size: {config.get('eval_batch_size', '?')}")
    dims = sorted(results_df["dimension"].unique())
    lines.append(f"  dimensions: {dims}")
    lines.append(f"  functions: {sorted(results_df['function_id'].unique().tolist())}")
    lines.append("")

    # Overall performance
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 40)
    for dim in dims:
        sub = results_df[results_df["dimension"] == dim]
        lines.append(f"  Dimension {dim}:")
        lines.append(f"    Convergence rate: {sub['converged'].mean()*100:.1f}% ({sub['converged'].sum()}/{len(sub)})")
        lines.append(f"    Median gap: {sub['gap'].median():.4g}")
        lines.append(f"    Mean gap:   {sub['gap'].mean():.4g}")
        lines.append(f"    Max gap:    {sub['gap'].max():.4g}")
        budget = config.get("budget_multiplier", 1024) * dim
        lines.append(f"    Budget: {budget}")
    lines.append("")

    # Per function group
    lines.append("PERFORMANCE BY FUNCTION GROUP")
    lines.append("-" * 40)
    results_df_c = results_df.copy()
    results_df_c["group"] = results_df_c["function_id"].map(BBOB_GROUP_FOR_FUNC)
    for grp in BBOB_GROUPS:
        lines.append(f"\n  {grp}:")
        for dim in dims:
            sub = results_df_c[(results_df_c["group"] == grp) & (results_df_c["dimension"] == dim)]
            if len(sub) == 0:
                continue
            lines.append(f"    dim={dim}: conv={sub['converged'].mean()*100:.0f}%, "
                         f"med_gap={sub['gap'].median():.4g}, "
                         f"mean_gap={sub['gap'].mean():.4g}")
    lines.append("")

    # Worst performing runs
    lines.append("WORST PERFORMING RUNS (top 15 by gap)")
    lines.append("-" * 40)
    worst = results_df.nlargest(15, "gap")
    for _, row in worst.iterrows():
        lines.append(f"  f{int(row['function_id'])}_d{int(row['dimension'])}_i{int(row['instance_id'])}: "
                     f"gap={row['gap']:.4g}, best={row['best_value']:.4g}, fopt={row['fopt']:.4g}")
    lines.append("")

    # Best performing runs
    lines.append("BEST PERFORMING RUNS (top 10 converged)")
    lines.append("-" * 40)
    best = results_df[results_df["converged"]].nsmallest(10, "evaluations")
    for _, row in best.iterrows():
        lines.append(f"  f{int(row['function_id'])}_d{int(row['dimension'])}_i{int(row['instance_id'])}: "
                     f"evals={int(row['evaluations'])}, gap={row['gap']:.4g}")
    lines.append("")

    # Anomaly summary
    lines.append("ANOMALY SUMMARY")
    lines.append("-" * 40)
    by_cat = defaultdict(list)
    for a in all_anomalies:
        by_cat[a.category].append(a)

    for cat in sorted(by_cat):
        items = by_cat[cat]
        sev_counts = defaultdict(int)
        for a in items:
            sev_counts[a.severity] += 1
        lines.append(f"\n  {cat} ({len(items)} occurrences):")
        lines.append(f"    severity: {dict(sev_counts)}")
        for a in items[:5]:
            lines.append(f"    - [{a.severity}] {a.run_label}"
                         + (f" iter {a.iteration}" if a.iteration else "")
                         + f": {a.description}")
        if len(items) > 5:
            lines.append(f"    ... and {len(items) - 5} more")
    lines.append("")

    # Diversity analysis summary
    lines.append("SAMPLE DIVERSITY SUMMARY")
    lines.append("-" * 40)
    for dim in dims:
        dim_runs = [r for r in runs if r.dim == dim]
        std_first = []
        std_last = []
        for r in dim_runs:
            key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
            df = it_dfs[key]
            if len(df) >= 2:
                std_first.append(df.iloc[0]["sample_std_mean"])
                std_last.append(df.iloc[-1]["sample_std_mean"])
        if std_first:
            lines.append(f"  Dimension {dim}:")
            lines.append(f"    First iteration avg std: {np.mean(std_first):.4f} ± {np.std(std_first):.4f}")
            lines.append(f"    Last iteration avg std:  {np.mean(std_last):.4f} ± {np.std(std_last):.4f}")
            lines.append(f"    Std reduction ratio:     {np.mean(std_last)/np.mean(std_first):.3f}")
    lines.append("")

    # Tau dynamics summary
    lines.append("TAU DYNAMICS SUMMARY")
    lines.append("-" * 40)
    tau_regressions = [a for a in all_anomalies if a.category == "tau_regression"]
    lines.append(f"  Total tau regression events: {len(tau_regressions)}")
    affected_runs = set(a.run_label for a in tau_regressions)
    lines.append(f"  Runs with tau regression: {len(affected_runs)}/{len(runs)}")
    lines.append("")

    # Elite buffer analysis
    lines.append("ELITE BUFFER ANALYSIS")
    lines.append("-" * 40)
    opt_cfg = config.get("optimizer", config)
    elite_per_dim = opt_cfg.get("elite_per_dim", "?")
    elite_min = opt_cfg.get("elite_min", "?")
    budget_mult = config.get("budget_multiplier", 1024)
    batch = config.get("eval_batch_size", 64)
    for dim in dims:
        if isinstance(elite_per_dim, (int, float)):
            elite_size = max(int(elite_per_dim) * dim, int(elite_min) if isinstance(elite_min, (int, float)) else 64)
        else:
            elite_size = "?"
        budget = budget_mult * dim
        n_iters = budget // batch
        lines.append(f"  Dimension {dim}: elite_size={elite_size}, budget={budget}, "
                     f"iterations={n_iters}, fill_at_iter≈{max(1, elite_size // batch)}")
    lines.append("")

    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    return "\n".join(lines)


# ======================================================================= #
#  Main                                                                     #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser(description="Analyze a Diffusion Optimizer experiment.")
    parser.add_argument("experiment_dir", help="Path to the experiment directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots/report (default: <experiment_dir>/analysis)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading experiment from {exp_dir} ...")
    runs, config, results_df = load_experiment(str(exp_dir))
    print(f"  Loaded {len(runs)} runs")

    # Compute per-iteration stats
    print("Computing per-iteration statistics ...")
    it_dfs: Dict[str, pd.DataFrame] = {}
    for r in runs:
        key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
        it_dfs[key] = per_iteration_stats(r)

    # Detect anomalies
    print("Running anomaly detection ...")
    all_anomalies: List[Anomaly] = []
    for r in runs:
        key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
        anomalies = detect_anomalies(r, it_dfs[key])
        all_anomalies.extend(anomalies)

    n_crit = sum(1 for a in all_anomalies if a.severity == "critical")
    n_high = sum(1 for a in all_anomalies if a.severity == "high")
    n_med = sum(1 for a in all_anomalies if a.severity == "medium")
    n_low = sum(1 for a in all_anomalies if a.severity == "low")
    print(f"  Found {len(all_anomalies)} anomalies: "
          f"{n_crit} critical, {n_high} high, {n_med} medium, {n_low} low")

    # Generate PDF report
    pdf_path = output_dir / "analysis_report.pdf"
    print(f"Generating PDF report at {pdf_path} ...")
    with PdfPages(str(pdf_path)) as pdf:
        plot_per_function_heatmap(results_df, pdf)
        plot_group_summary(results_df, pdf)
        plot_dimension_scaling(results_df, pdf)
        plot_convergence_grid(runs, it_dfs, pdf)
        plot_tau_evolution(runs, it_dfs, pdf)
        plot_value_distribution(runs, it_dfs, pdf)
        plot_sample_diversity(runs, it_dfs, pdf)
        plot_centroid_drift(runs, it_dfs, pdf)
        plot_exploitation_quality(runs, it_dfs, pdf)
        plot_improvement_rate(runs, it_dfs, pdf)

    # Build and save text report
    report = build_text_report(runs, it_dfs, all_anomalies, config, results_df)
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to {report_path}")
    print(f"PDF saved to {pdf_path}")

    # Save raw per-iteration data
    all_rows = []
    for r in runs:
        key = f"f{r.func_id}_d{r.dim}_i{r.instance}"
        df = it_dfs[key].copy()
        df["func_id"] = r.func_id
        df["dim"] = r.dim
        df["instance"] = r.instance
        df["fopt"] = r.fopt
        df["group"] = BBOB_GROUP_FOR_FUNC.get(r.func_id, "?")
        all_rows.append(df)
    combined_df = pd.concat(all_rows, ignore_index=True)
    csv_path = output_dir / "per_iteration_stats.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"Per-iteration stats saved to {csv_path}")

    # Save anomalies
    anom_rows = [{"run": a.run_label, "category": a.category,
                  "iteration": a.iteration, "description": a.description,
                  "severity": a.severity} for a in all_anomalies]
    anom_df = pd.DataFrame(anom_rows)
    anom_path = output_dir / "anomalies.csv"
    anom_df.to_csv(anom_path, index=False)
    print(f"Anomalies saved to {anom_path}")


if __name__ == "__main__":
    main()
