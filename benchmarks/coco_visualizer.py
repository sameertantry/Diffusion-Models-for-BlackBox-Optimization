"""COCO/BBOB visualization runner for diffusion-based optimizers.

For each BBOB function f1–f24 in 2D, this module:

1. Evaluates the function on a dense mesh grid.
2. Runs the optimizer (ask/tell loop) to collect sampled points.
3. Produces a 3D surface plot (x1, x2, f(x1,x2)) with diffusion-
   sampled points overlaid as a scatter layer.

Usage
-----
From the command line::

    python -m benchmarks.coco_visualizer --config configs/coco_diffusion.json
    python -m benchmarks.coco_visualizer --config configs/coco_diffusion.json --output-dir plots

Or programmatically::

    from benchmarks.coco_visualizer import run_coco_visualization
    run_coco_visualization(config_path="configs/coco_diffusion.json")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Lazy import guard                                                            #
# --------------------------------------------------------------------------- #
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cocoex  # noqa: F401

from methods.base import BaseOptimizer
from methods.cma_es import CMAES
from methods.diffusion import DiffusionOptimizer
from methods.diffusion_bbo import DiffusionBBO
from methods.tpe import TPE

OptimizerFactory = Callable[[int, np.ndarray, np.ndarray], BaseOptimizer]


# =========================================================================== #
#  Factories (mirrors coco_wrapper)                                            #
# =========================================================================== #


def _make_diffusion_factory(**kw: Any) -> OptimizerFactory:
    def _f(d: int, lo: np.ndarray, hi: np.ndarray) -> DiffusionOptimizer:
        return DiffusionOptimizer(input_dim=d, bounds=(lo, hi), **kw)
    return _f


def _make_cmaes_factory(**kw: Any) -> OptimizerFactory:
    def _f(d: int, lo: np.ndarray, hi: np.ndarray) -> CMAES:
        return CMAES(input_dim=d, bounds=(lo, hi), **kw)
    return _f


def _make_tpe_factory(**kw: Any) -> OptimizerFactory:
    def _f(d: int, lo: np.ndarray, hi: np.ndarray) -> TPE:
        return TPE(input_dim=d, bounds=(lo, hi), **kw)
    return _f


def _make_diffusion_bbo_factory(**kw: Any) -> OptimizerFactory:
    def _f(d: int, lo: np.ndarray, hi: np.ndarray) -> DiffusionBBO:
        return DiffusionBBO(input_dim=d, bounds=(lo, hi), **kw)
    return _f


_FACTORY_BUILDERS: Dict[str, Callable[..., OptimizerFactory]] = {
    "diffusion": _make_diffusion_factory,
    "diffusion_bbo": _make_diffusion_bbo_factory,
    "cma_es": _make_cmaes_factory,
    "tpe": _make_tpe_factory,
}


def _make_factory(method: str, **kw: Any) -> OptimizerFactory:
    key = method.lower().strip()
    if key not in _FACTORY_BUILDERS:
        raise ValueError(f"Unknown method '{method}'. Supported: {sorted(_FACTORY_BUILDERS)}")
    return _FACTORY_BUILDERS[key](**kw)


# =========================================================================== #
#  Surface + scatter plotting                                                  #
# =========================================================================== #


def _evaluate_on_grid(
    problem: "cocoex.Problem",
    grid_res: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a 2-D COCO problem on a regular mesh grid.

    Returns (X1, X2, Z) suitable for ``plot_surface``.
    """
    lo = np.array(problem.lower_bounds)
    hi = np.array(problem.upper_bounds)
    x1 = np.linspace(lo[0], hi[0], grid_res)
    x2 = np.linspace(lo[1], hi[1], grid_res)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.empty_like(X1)
    for i in range(grid_res):
        for j in range(grid_res):
            Z[i, j] = problem(np.array([X1[i, j], X2[i, j]]))
    return X1, X2, Z


def _run_optimizer_collect_points(
    optimizer: BaseOptimizer,
    problem: "cocoex.Problem",
    budget_multiplier: float,
    eval_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the ask/tell loop and return all evaluated (x, y, iteration) tuples."""
    budget = int(budget_multiplier * problem.dimension)
    if hasattr(optimizer, "budget"):
        optimizer.budget = budget

    xs: List[np.ndarray] = []
    ys: List[float] = []
    iters: List[int] = []

    x0 = np.array(problem.initial_solution, dtype=np.float64)
    y0 = float(problem(x0))
    optimizer.tell(x0.reshape(1, -1), np.array([y0]))
    xs.append(x0)
    ys.append(y0)
    iters.append(0)

    iteration = 1
    while problem.evaluations < budget:
        remaining = budget - problem.evaluations
        n_ask = min(eval_batch_size, remaining)
        if n_ask <= 0:
            break
        candidates = optimizer.ask(n=n_ask)
        values = np.array(
            [problem(candidates[i]) for i in range(len(candidates))],
            dtype=np.float64,
        )
        optimizer.tell(candidates, values)
        for ci, vi in zip(candidates, values):
            xs.append(ci)
            ys.append(float(vi))
            iters.append(iteration)
        iteration += 1

    return np.array(xs), np.array(ys), np.array(iters)


def _plot_surface_with_points(
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    pts_iter: np.ndarray,
    title: str,
    save_path: Path,
    elev: float = 30,
    azim: float = -60,
) -> None:
    """Render a 3-D surface with iteration-colored optimizer samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X1, X2, Z,
        cmap=cm.viridis,
        alpha=0.55,
        edgecolor="none",
        rcount=100,
        ccount=100,
    )

    n_iters = int(pts_iter.max()) + 1
    iter_norm = pts_iter.astype(float) / max(n_iters - 1, 1)
    scatter_cmap = cm.coolwarm

    sizes = 8 + 22 * iter_norm

    ax.scatter(
        pts_x[:, 0], pts_x[:, 1], pts_y,
        c=iter_norm, cmap=scatter_cmap,
        s=sizes, alpha=0.85, zorder=5,
        edgecolors="k", linewidths=0.3,
        depthshade=True,
    )

    sm = plt.cm.ScalarMappable(
        cmap=scatter_cmap,
        norm=mcolors.Normalize(vmin=0, vmax=n_iters - 1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10, aspect=25)
    cbar.set_label("Iteration", fontsize=11)

    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_zlabel("f(x₁, x₂)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.view_init(elev=elev, azim=azim)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =========================================================================== #
#  Main visualization runner                                                   #
# =========================================================================== #


def run_coco_visualization(
    config_path: str | Path,
    output_dir: str | Path = "plots/bbob_surfaces",
    grid_res: int = 200,
    func_ids: Optional[List[int]] = None,
    instance_id: int = 1,
    elev: float = 30,
    azim: float = -60,
) -> Path:
    """Run the visualization pipeline.

    Parameters
    ----------
    config_path : str or Path
        Path to the JSON config (same format as ``coco_wrapper``).
    output_dir : str or Path
        Directory where plots are saved.
    grid_res : int
        Number of grid points per axis for the surface.
    func_ids : list of int, optional
        Which function ids to plot (default: 1–24).
    instance_id : int
        BBOB instance id to use.
    elev, azim : float
        Camera elevation and azimuth for the 3-D projection.

    Returns
    -------
    Path
        The output directory containing the saved plots.
    """
    import cocoex

    config_path = Path(config_path)
    output_dir = Path(output_dir)

    with config_path.open("r", encoding="utf-8") as fh:
        config: Dict[str, Any] = json.load(fh)

    method = config.get("method", "diffusion")
    optimizer_kwargs = config.get("optimizer", {})
    budget_multiplier = config.get("budget_multiplier", 1024)
    eval_batch_size = config.get("eval_batch_size", 64)
    algorithm_name = config.get("algorithm_name", method.upper())

    factory = _make_factory(method, **optimizer_kwargs)

    if func_ids is None:
        func_ids = list(range(1, 25))

    suite_options = f"dimensions: 2 instance_indices: {instance_id}"
    suite = cocoex.Suite("bbob", "", suite_options)

    for problem in suite:
        fid = int(problem.id_function)
        if fid not in func_ids:
            continue

        print(f"\n{'='*60}")
        print(f"  Function f{fid} | dim={problem.dimension} | instance={problem.id_instance}")
        print(f"{'='*60}")

        # --- Evaluate function on grid (separate problem copy to avoid
        #     polluting the evaluation counter for the optimizer run). ---
        grid_suite = cocoex.Suite("bbob", "", suite_options)
        grid_problem = None
        for p in grid_suite:
            if int(p.id_function) == fid and int(p.id_instance) == instance_id:
                grid_problem = p
                break
        if grid_problem is None:
            print(f"  WARNING: could not find grid problem f{fid}, skipping.")
            continue

        print("  Evaluating surface grid...")
        X1, X2, Z = _evaluate_on_grid(grid_problem, grid_res=grid_res)

        # --- Run optimizer and collect sampled points ---
        print("  Running optimizer...")
        lo = np.array(problem.lower_bounds, dtype=np.float64)
        hi = np.array(problem.upper_bounds, dtype=np.float64)
        optimizer = factory(problem.dimension, lo, hi)
        pts_x, pts_y, pts_iter = _run_optimizer_collect_points(
            optimizer, problem, budget_multiplier, eval_batch_size,
        )
        n_iters = int(pts_iter.max()) + 1
        print(f"  Collected {len(pts_x)} evaluated points over {n_iters} iterations.")

        # --- Plot ---
        title = (
            f"BBOB f{fid} (2-D, instance {instance_id})\n"
            f"{algorithm_name} — {len(pts_x)} evals, {n_iters} iterations"
        )
        save_path = output_dir / f"bbob_f{fid:02d}_2d_{algorithm_name}.png"
        _plot_surface_with_points(
            X1, X2, Z, pts_x, pts_y, pts_iter,
            title=title,
            save_path=save_path,
            elev=elev,
            azim=azim,
        )

    print(f"\nAll plots saved to: {output_dir}")
    return output_dir


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3-D surface plots for BBOB f1–f24 (2-D) with "
            "diffusion-sampled points overlaid."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a JSON config file (same format as coco_wrapper).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots/bbob_surfaces",
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--grid-res", type=int, default=200,
        help="Grid resolution per axis for the surface mesh.",
    )
    parser.add_argument(
        "--func-ids", type=str, default=None,
        help="Comma-separated function ids to plot (default: 1-24).",
    )
    parser.add_argument(
        "--instance-id", type=int, default=1,
        help="BBOB instance id to use.",
    )
    parser.add_argument(
        "--elev", type=float, default=30,
        help="Camera elevation angle for the 3-D view.",
    )
    parser.add_argument(
        "--azim", type=float, default=-60,
        help="Camera azimuth angle for the 3-D view.",
    )
    return parser


def cli_main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    func_ids = None
    if args.func_ids is not None:
        func_ids = [int(x.strip()) for x in args.func_ids.split(",")]

    run_coco_visualization(
        config_path=args.config,
        output_dir=args.output_dir,
        grid_res=args.grid_res,
        func_ids=func_ids,
        instance_id=args.instance_id,
        elev=args.elev,
        azim=args.azim,
    )


if __name__ == "__main__":
    cli_main()
