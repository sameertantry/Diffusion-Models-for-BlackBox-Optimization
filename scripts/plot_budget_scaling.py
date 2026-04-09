#!/usr/bin/env python3
"""Plot budget scaling: convergence rate vs budget for DiffusionV2 and CMA-ES.

Usage::

    python scripts/plot_budget_scaling.py
    python scripts/plot_budget_scaling.py --output reports/budget_scaling.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/budget_scaling.png")
    parser.add_argument("--precision", type=float, default=0.1)
    args = parser.parse_args()

    budget_exps = {
        512:   ("exdata/rm/ablation/budget/C2_budget_512",   "exdata/rm/ablation/budget/C2_cmaes_budget_512"),
        1024:  ("exdata/rm/optimal/stage-3b/ls_rank/frac0.15", "exdata/predefence/CMA-ES"),
        2048:  ("exdata/rm/ablation/budget/C2_budget_2048",  "exdata/rm/ablation/budget/C2_cmaes_budget_2048"),
        4096:  ("exdata/rm/ablation/budget/C2_budget_4096",  "exdata/rm/ablation/budget/C2_cmaes_budget_4096"),
        10000: ("exdata/rm/ablation/budget/C2_budget_10k",   "exdata/rm/ablation/budget/C2_cmaes_budget_10k"),
    }

    budgets = sorted(budget_exps.keys())
    dims = [2, 5, 10]

    # Collect data
    data = {
        "ours": {d: [] for d in dims + ["all"]},
        "cma":  {d: [] for d in dims + ["all"]},
    }

    for budget in budgets:
        ours_path, cma_path = budget_exps[budget]
        ours = pd.read_csv(Path(ours_path) / "experiment_results.csv")
        cma  = pd.read_csv(Path(cma_path)  / "experiment_results.csv")

        for method, df in [("ours", ours), ("cma", cma)]:
            data[method]["all"].append(
                (df["gap"] < args.precision).mean() * 100
            )
            for d in dims:
                dd = df[df["dimension"] == d]
                data[method][d].append(
                    (dd["gap"] < args.precision).mean() * 100
                )

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    titles = ["Overall", "dim = 2", "dim = 5", "dim = 10"]
    dim_keys = ["all", 2, 5, 10]

    for ax, title, dk in zip(axes, titles, dim_keys):
        ax.plot(budgets, data["ours"][dk], "o-", color="#2196F3",
                linewidth=2, markersize=7, label="DiffusionV2 (ours)")
        ax.plot(budgets, data["cma"][dk],  "s--", color="#F44336",
                linewidth=2, markersize=7, label="CMA-ES")

        ax.set_xscale("log")
        ax.set_xlabel("Budget (× dimension)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"Convergence rate (%) [gap < {args.precision}]", fontsize=11)
    axes[0].legend(fontsize=10, loc="lower right")
    axes[0].set_ylim(0, 100)

    fig.suptitle("Budget Scaling: DiffusionV2 vs CMA-ES on BBOB",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
