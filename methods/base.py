"""Base abstract class for all optimization methods."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union
import time

import numpy as np


class BaseOptimizer(ABC):
    """Abstract base class for black-box optimization methods.

    All optimizers must implement this interface to ensure compatibility
    across different optimization algorithms and experiments.

    Attributes:
        name: Human-readable name of the optimizer.
        input_dim: Dimensionality of the input space.
        bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays.
        num_evals: Current number of function evaluations performed.
    """

    def __init__(
        self,
        name: str,
        input_dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        verbose: bool = False,
    ):
        """Initialize the optimizer.

        Args:
            name: Name of the optimizer.
            input_dim: Dimensionality of the input space.
            bounds: Tuple of (lower_bounds, upper_bounds) arrays.
        """
        self.name = name
        self.input_dim = input_dim
        self.bounds = bounds
        self.num_evals = 0
        self.verbose = verbose

        # Verbose sampling/evaluation trace (written as .npz).
        self._iteration_counter = 0
        self._pending_iteration_indices: list[int] = []
        self._verbose_samples: list[np.ndarray] = []
        self._verbose_values: list[float] = []
        self._verbose_indexes: list[int] = []
        self._verbose_taus: list[float] = []
        self._verbose_log_dir: Optional[Path] = None
        self.verbose_log_path: Optional[Path] = None
        self._verbose_problem_tag: Optional[str] = None

    @abstractmethod
    def ask(self, n: int = 1) -> np.ndarray:
        """Propose candidate solution(s) for evaluation.

        Args:
            n: Number of candidates to propose (default: 1).

        Returns:
            Array of shape (n, input_dim) with candidate inputs.
        """
        pass

    @abstractmethod
    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        """Report evaluation results to the optimizer.

        Args:
            x: Array of shape (n, input_dim) with evaluated inputs.
            y: Array of shape (n,) or (n, 1) with function values.
        """
        pass

    def warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Warm-start the optimizer with previously evaluated data.

        This is called *before* the main ask/tell loop to seed the
        optimizer with initial observations (e.g. an offline dataset).
        The default implementation is a no-op; subclasses should override
        if they can benefit from initial data.

        Args:
            x: Array of shape ``(n, input_dim)`` with evaluated inputs.
            y: Array of shape ``(n,)`` with objective values
               (minimisation convention: lower is better).
        """
        # Default: do nothing.  Subclasses override.
        pass

    def set_verbose_log_dir(self, directory: Union[str, Path]) -> None:
        """Set the directory where verbose .npz logs will be written.

        Must be called *before* the first ask/tell cycle for the path to
        take effect.  If not called, a default ``outputs/verbose_logs``
        directory is used.
        """
        self._verbose_log_dir = Path(directory)

    def _register_asked(self, candidates: np.ndarray) -> None:
        """Register candidates produced by one ask() iteration."""
        if not self.verbose:
            return
        self._iteration_counter += 1
        n = int(np.asarray(candidates).shape[0]) if candidates is not None else 0
        if n > 0:
            self._pending_iteration_indices.extend([self._iteration_counter] * n)

    def _register_told(
        self,
        x: np.ndarray,
        y: np.ndarray,
        tau: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """Register evaluated samples/values in tell() order.

        ``tau`` is optional and mainly used by diffusion-based methods.
        """
        if not self.verbose:
            return

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64)).flatten()

        n = min(len(x_arr), len(y_arr), len(self._pending_iteration_indices))
        if n <= 0:
            return

        if tau is None:
            tau_arr = np.full(n, np.nan, dtype=np.float64)
        else:
            tau_np = np.asarray(tau, dtype=np.float64).reshape(-1)
            if tau_np.size == 1:
                tau_arr = np.full(n, float(tau_np[0]), dtype=np.float64)
            else:
                tau_arr = np.full(n, np.nan, dtype=np.float64)
                m = min(n, tau_np.size)
                tau_arr[:m] = tau_np[:m]

        for i in range(n):
            self._verbose_samples.append(np.asarray(x_arr[i], dtype=np.float64).copy())
            self._verbose_values.append(float(y_arr[i]))
            self._verbose_indexes.append(int(self._pending_iteration_indices[i]))
            self._verbose_taus.append(float(tau_arr[i]))

        self._pending_iteration_indices = self._pending_iteration_indices[n:]
        self._flush_verbose_log()

    def set_problem_tag(self, tag: str) -> None:
        """Set a human-readable tag (e.g. 'f1_d2_i3') embedded in the npz filename."""
        self._verbose_problem_tag = tag

    def _flush_verbose_log(self) -> None:
        """Persist verbose arrays to disk as a single .npz file."""
        if not self.verbose:
            return
        if self.verbose_log_path is None:
            safe_name = self.name.lower().replace(" ", "_")
            ts = time.strftime("%Y%m%d_%H%M%S")
            unique = time.time_ns()
            tag = f"_{self._verbose_problem_tag}" if self._verbose_problem_tag else ""
            out_dir = self._verbose_log_dir or (Path("outputs") / "verbose_logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            self.verbose_log_path = out_dir / f"{safe_name}{tag}_{ts}_{unique}.npz"

        if self._verbose_samples:
            samples = np.vstack(self._verbose_samples).astype(np.float64, copy=False)
        else:
            samples = np.empty((0, self.input_dim), dtype=np.float64)
        values = np.asarray(self._verbose_values, dtype=np.float64).reshape(-1, 1)
        indexes = np.asarray(self._verbose_indexes, dtype=np.int64).reshape(-1, 1)
        taus = np.asarray(self._verbose_taus, dtype=np.float64).reshape(-1, 1)
        np.savez(
            self.verbose_log_path,
            samples=samples,
            values=values,
            indexes=indexes,
            taus=taus,
        )

    def plot_convergence(self, title: str = "") -> Optional[Path]:
        """Build a convergence plot from the verbose trace and save as PNG.

        The plot contains two subplots:
        1. **Top** — all evaluated f(x) values as a scatter over cumulative
           evaluation index, with the running-best curve overlaid.
        2. **Bottom** — per-iteration statistics: best, mean, and median
           f(x) for each iteration index. If available, tau is shown on a
           secondary y-axis.

        Returns the path to the saved PNG, or ``None`` if verbose logging
        is disabled or no data has been collected.
        """
        if not self.verbose or not self._verbose_values:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        values = np.asarray(self._verbose_values, dtype=np.float64)
        indexes = np.asarray(self._verbose_indexes, dtype=np.int64)
        taus = np.asarray(self._verbose_taus, dtype=np.float64)

        # --- running best over evaluation order ---
        running_best = np.minimum.accumulate(values)

        # --- per-iteration aggregates ---
        unique_iters = np.unique(indexes)
        iter_best = np.empty(len(unique_iters))
        iter_mean = np.empty(len(unique_iters))
        iter_median = np.empty(len(unique_iters))
        iter_tau = np.full(len(unique_iters), np.nan, dtype=np.float64)
        for i, it in enumerate(unique_iters):
            mask = indexes == it
            vals = values[mask]
            iter_best[i] = np.min(vals)
            iter_mean[i] = np.mean(vals)
            iter_median[i] = np.median(vals)
            tau_vals = taus[mask]
            finite_tau = tau_vals[np.isfinite(tau_vals)]
            if finite_tau.size > 0:
                iter_tau[i] = float(finite_tau[-1])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

        # -- top: scatter + running best --
        eval_idx = np.arange(1, len(values) + 1)
        ax1.scatter(eval_idx, values, s=4, alpha=0.35, label="f(x)")
        ax1.plot(eval_idx, running_best, color="red", linewidth=1.5,
                 label="running best")
        ax1.set_xlabel("Cumulative evaluation")
        ax1.set_ylabel("f(x)")
        ax1.set_title(title or f"{self.name} — convergence")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # -- bottom: per-iteration stats --
        ax2.plot(unique_iters, iter_best, marker=".", markersize=3,
                 linewidth=1.2, label="iter best")
        ax2.plot(unique_iters, iter_mean, marker=".", markersize=3,
                 linewidth=1.0, alpha=0.7, label="iter mean")
        ax2.plot(unique_iters, iter_median, marker=".", markersize=3,
                 linewidth=1.0, alpha=0.7, linestyle="--", label="iter median")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("f(x)")
        ax2.set_title("Per-iteration statistics")
        if np.any(np.isfinite(iter_tau)):
            ax2_tau = ax2.twinx()
            ax2_tau.plot(
                unique_iters, iter_tau, color="purple", linewidth=1.2, label="tau"
            )
            ax2_tau.set_ylabel("tau", color="purple")
            ax2_tau.tick_params(axis="y", labelcolor="purple")
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_tau.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
        else:
            ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        # Save next to the .npz file (same dir, same stem).
        if self.verbose_log_path is not None:
            png_path = self.verbose_log_path.with_suffix(".png")
        else:
            out_dir = self._verbose_log_dir or (Path("outputs") / "verbose_logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = self.name.lower().replace(" ", "_")
            png_path = out_dir / f"{safe_name}_convergence.png"

        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return png_path

    def _reset_verbose_trace(self) -> None:
        """Reset in-memory verbose trace buffers for a new run.

        The configured log directory (``_verbose_log_dir``) is preserved so
        that subsequent runs still write to the same result folder.
        """
        self._iteration_counter = 0
        self._pending_iteration_indices = []
        self._verbose_samples = []
        self._verbose_values = []
        self._verbose_indexes = []
        self._verbose_taus = []
        self.verbose_log_path = None
        self._verbose_problem_tag = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the optimizer's internal state.

        This method should reset all internal state to allow
        the optimizer to be reused for a new optimization run.
        """
        pass
