"""COCO/BBOB experiment runner for black-box optimizers.

This module provides a complete experiment harness that benchmarks any
:class:`~methods.base.BaseOptimizer` on COCO test suites using **only** the
public ask/tell/reset interface.

Supported methods:

- **diffusion** — :class:`~methods.diffusion.DiffusionOptimizer`
- **cma_es** — :class:`~methods.cma_es.CMAES`
- **tpe** — :class:`~methods.tpe.TPE`

The wrapper does NOT modify any optimizer implementation.  It instantiates a
fresh optimizer for each COCO problem (via a user-supplied or auto-selected
factory), feeds evaluations through the standard ask/tell loop, and relies on
COCO's :class:`~cocoex.Observer` and :class:`~cocoex.ExperimentRepeater`
infrastructure for data logging, budget management, and restart scheduling.

Usage
-----
From the command line::

    python -m benchmarks.coco_wrapper --config configs/coco_diffusion.json
    python -m benchmarks.coco_wrapper --config configs/coco_cmaes.json
    python -m benchmarks.coco_wrapper --config configs/coco_tpe.json

Or programmatically::

    from benchmarks.coco_wrapper import run_coco_experiment, make_diffusion_factory

    factory = make_diffusion_factory(guidance_strength=5.0, hidden_dim=256)
    result_folder = run_coco_experiment(
        optimizer_factory=factory,
        algorithm_name="DiffusionBBO",
        budget_multiplier=1e4,
        output_folder="exdata/DiffusionBBO",
    )

Notes
-----
- ``cocoex.Problem`` is a **minimisation** problem.
- :meth:`DiffusionOptimizer.tell` internally negates *y* to convert
  minimisation into the internal maximisation score, so we pass raw COCO
  objective values directly.
- COCO tracks evaluations internally via ``problem.evaluations``; the
  observer records every ``problem(x)`` call automatically.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Import path robustness                                                      #
#                                                                             #
# Running `python benchmarks/coco_wrapper.py` sets sys.path[0] to `benchmarks/`#
# which makes top-level imports like `methods.*` fail. We add the repo root to #
# sys.path so the script works from any working directory.                    #
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --------------------------------------------------------------------------- #
# Lazy import guard: cocoex is only required at runtime.                       #
# --------------------------------------------------------------------------- #
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cocoex  # noqa: F401 – used for type hints only

from methods.base import BaseOptimizer
from methods.bipop_cma_es import BIPOPCMAES
from methods.cma_es import CMAES
from methods.diffusion import DiffusionOptimizer
from methods.diffusion_bbo import DiffusionBBO
from methods.diffusion_v2 import DiffusionOptimizerV2
from methods.gp_qei import GPqEI
from methods.sep_cma_es import SepCMAES
from methods.tpe import TPE

# --------------------------------------------------------------------------- #
# Type alias for an optimizer factory.                                         #
# The factory receives (dimension, lower_bounds, upper_bounds) and must        #
# return a fresh BaseOptimizer-compatible instance.                            #
# --------------------------------------------------------------------------- #
OptimizerFactory = Callable[[int, np.ndarray, np.ndarray], BaseOptimizer]


# =========================================================================== #
#  CSV Logger                                                                  #
# =========================================================================== #


class COCOCSVLogger:
    """Logger that writes experiment results to a CSV file.

    Each row represents one problem instance (function + instance combination)
    with convergence metrics and statistics.
    """

    def __init__(self, output_path: Path):
        """Initialize the CSV logger.

        Parameters
        ----------
        output_path : Path
            Path where the CSV file will be written.
        """
        self.output_path = Path(output_path)
        self.records: List[Dict[str, Any]] = []

    def log_problem(
        self,
        method_name: str,
        problem: "cocoex.Problem",
        fopt: Optional[float],
        precision: float,
        converged: bool,
    ) -> None:
        """Log results for a single problem instance.

        Parameters
        ----------
        method_name : str
            Name of the optimization method.
        problem : cocoex.Problem
            The COCO problem instance.
        fopt : float, optional
            Known optimal function value.
        precision : float
            Precision threshold used for convergence.
        converged : bool
            Whether the custom precision criterion was met.
        """
        best_value = problem.best_observed_fvalue1
        evaluations = problem.evaluations
        final_target_hit = problem.final_target_hit

        # Calculate gap if fopt is available
        gap = None
        if fopt is not None:
            gap = best_value - fopt

        record = {
            "method_name": method_name,
            "function_id": problem.id_function,
            "instance_id": problem.id_instance,
            "dimension": problem.dimension,
            "best_value": best_value,
            "gap": gap,
            "evaluations": evaluations,
            "converged": converged,
            "final_target_hit": final_target_hit,
            "fopt": fopt,
            "precision": precision,
        }
        self.records.append(record)

    def write(self) -> None:
        """Write all logged records to the CSV file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define column order
        fieldnames = [
            "method_name",
            "function_id",
            "instance_id",
            "dimension",
            "best_value",
            "gap",
            "evaluations",
            "converged",
            "final_target_hit",
            "fopt",
            "precision",
        ]

        with self.output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            if self.records:
                writer.writerows(self.records)


# =========================================================================== #
#  Factories                                                                   #
# =========================================================================== #


def make_diffusion_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`DiffusionOptimizer` instances.

    All keyword arguments are forwarded verbatim to the
    :class:`DiffusionOptimizer` constructor.  ``input_dim`` and ``bounds``
    are filled automatically by the experiment runner from the COCO problem
    metadata.

    Parameters
    ----------
    **optimizer_kwargs
        Any keyword argument accepted by :class:`DiffusionOptimizer`
        (e.g. ``guidance_strength``, ``hidden_dim``, ``num_timesteps``).

    Returns
    -------
    OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> DiffusionOptimizer``.
    """

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> DiffusionOptimizer:
        return DiffusionOptimizer(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_cmaes_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`CMAES` instances.

    Parameters
    ----------
    **optimizer_kwargs
        Keyword arguments accepted by :class:`CMAES`
        (``seed``, ``sigma0``).

    Returns
    -------
    OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> CMAES``.
    """

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> CMAES:
        return CMAES(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_tpe_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`TPE` instances.

    Parameters
    ----------
    **optimizer_kwargs
        Keyword arguments accepted by :class:`TPE`
        (``seed``).

    Returns
    -------
    OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> TPE``.
    """

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> TPE:
        return TPE(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_gp_qei_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`GPqEI` instances.

    Parameters
    ----------
    **optimizer_kwargs
        Keyword arguments accepted by :class:`GPqEI`
        (``seed``, ``n_initial``, ``mc_samples``, ``num_restarts``,
        ``raw_samples``, ``max_train_size``, ``device``, ``dtype``).

    Returns
    -------
    OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> GPqEI``.
    """

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> GPqEI:
        return GPqEI(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_turbo_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`~methods.turbo.TuRBO` instances.

    Parameters
    ----------
    **optimizer_kwargs
        Keyword arguments accepted by :class:`~methods.turbo.TuRBO`
        (``seed``, ``n_initial``, ``n_trust_regions``, ``batch_size``,
        ``max_cholesky_size``, ``device``, ``dtype``).

    Returns
    -------
    OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> TuRBO``.
    """

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):  # type: ignore[return]
        from methods.turbo import TuRBO
        return TuRBO(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_diffusion_bbo_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`DiffusionBBO` instances.

    Parameters
    ----------
    **optimizer_kwargs
        Keyword arguments accepted by :class:`DiffusionBBO`
        (e.g. ``n_ensemble``, ``guidance_scale``, ``p_uncond``).

    Returns
    -------
    OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> DiffusionBBO``.
    """

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> DiffusionBBO:
        return DiffusionBBO(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_diffusion_v2_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`DiffusionOptimizerV2` instances.

    Parameters
    ----------
    **optimizer_kwargs
        Keyword arguments accepted by :class:`DiffusionOptimizerV2`
        (e.g. ``elite_min_per_dim``, ``elite_max_per_dim``, ``ema_decay``).

    Returns
    -------
    OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> DiffusionOptimizerV2``.
    """

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> DiffusionOptimizerV2:
        return DiffusionOptimizerV2(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_sep_cmaes_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`SepCMAES` instances."""

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> SepCMAES:
        return SepCMAES(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


def make_bipop_cmaes_factory(**optimizer_kwargs: Any) -> OptimizerFactory:
    """Create a factory that produces :class:`BIPOPCMAES` instances."""

    def _factory(
        dimension: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> BIPOPCMAES:
        return BIPOPCMAES(
            input_dim=dimension,
            bounds=(lower_bounds, upper_bounds),
            **optimizer_kwargs,
        )

    return _factory


# Registry: method name -> factory builder
_FACTORY_BUILDERS: Dict[str, Callable[..., OptimizerFactory]] = {
    "diffusion": make_diffusion_factory,
    "diffusion_v2": make_diffusion_v2_factory,
    "diffusion_bbo": make_diffusion_bbo_factory,
    "cma_es": make_cmaes_factory,
    "sep_cma_es": make_sep_cmaes_factory,
    "bipop_cma_es": make_bipop_cmaes_factory,
    "tpe": make_tpe_factory,
    "gp_qei": make_gp_qei_factory,
    "turbo": make_turbo_factory,
}


def make_factory(method: str, **optimizer_kwargs: Any) -> OptimizerFactory:
    """Create an optimizer factory by method name.

    Parameters
    ----------
    method : str
        One of ``"diffusion"``, ``"cma_es"``, ``"tpe"``.
    **optimizer_kwargs
        Method-specific keyword arguments forwarded to the optimizer
        constructor.

    Returns
    -------
    OptimizerFactory

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    key = method.lower().strip()
    if key not in _FACTORY_BUILDERS:
        raise ValueError(
            "Unknown method '{}'. Supported: {}".format(
                method, sorted(_FACTORY_BUILDERS),
            )
        )
    return _FACTORY_BUILDERS[key](**optimizer_kwargs)


# =========================================================================== #
#  Experiment runner                                                           #
# =========================================================================== #


class COCOExperimentRunner:
    """Orchestrates a full COCO/BBOB benchmarking experiment.

    This runner manages the complete COCO experiment lifecycle:

    1. Suite creation (problems, dimensions, instances).
    2. Observer creation (data logging).
    3. :class:`~cocoex.ExperimentRepeater` for budget control and restarts.
    4. For each problem attempt: instantiate a fresh optimizer via the
       supplied factory, execute the ask/tell loop, and track the outcome.

    The runner interacts with optimizers **exclusively** through the
    :meth:`~BaseOptimizer.ask` / :meth:`~BaseOptimizer.tell` /
    :meth:`~BaseOptimizer.reset` interface.

    Parameters
    ----------
    optimizer_factory : OptimizerFactory
        A callable ``(dimension, lower_bounds, upper_bounds) -> BaseOptimizer``
        that creates a fresh optimizer for a given problem.  Use
        :func:`make_factory` or one of the specific ``make_*_factory``
        helpers for convenience.
    algorithm_name : str
        Human-readable name written into COCO data files and used by
        ``cocopp`` for legends/labels.
    budget_multiplier : float
        Total evaluation budget expressed as a multiple of the problem
        dimension: ``budget = int(budget_multiplier * dimension)``.
        The COCO convention is typically 1e3 -- 1e5.
    eval_batch_size : int
        Number of candidate solutions requested per :meth:`ask` call.
    precision : float
        Convergence precision.  A problem is considered solved when
        ``f(x_best) - f_opt < precision``.  COCO's built-in precision
        is ``1e-8``; setting a larger value (e.g. ``1e-2``) makes the
        convergence criterion more lenient and can stop runs earlier.
    suite_name : str
        COCO suite identifier (e.g. ``"bbob"``, ``"bbob-largescale"``).
    suite_instance : str
        Suite instance filter (e.g. ``"instances: 1-5"``).
    suite_options : str
        Additional suite options (e.g. ``"dimensions: 2,3,5,10,20"``).
    """

    def __init__(
        self,
        optimizer_factory: OptimizerFactory,
        algorithm_name: str = "Optimizer",
        budget_multiplier: float = 1e4,
        eval_batch_size: int = 1,
        precision: float = 1e-8,
        suite_name: str = "bbob",
        suite_instance: str = "",
        suite_options: str = "",
    ) -> None:
        self.optimizer_factory = optimizer_factory
        self.algorithm_name = algorithm_name
        self.budget_multiplier = budget_multiplier
        self.eval_batch_size = eval_batch_size
        self.precision = precision
        self.suite_name = suite_name
        self.suite_instance = suite_instance
        self.suite_options = suite_options

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        output_folder: Optional[str] = None,
        post_process: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run the full COCO experiment.

        Parameters
        ----------
        output_folder : str, optional
            Name of the COCO result folder.  If ``None`` a default name is
            constructed from the algorithm name, suite, and budget.
        post_process : bool
            If ``True``, invokes ``cocopp.main`` on the result folder after
            the experiment finishes (requires ``cocopp`` to be installed).
        config : dict, optional
            The experiment configuration dict.  When provided it is saved
            as ``experiment_config.json`` inside the COCO result folder
            so that every run is fully reproducible.

        Returns
        -------
        str
            Absolute path to the result folder.
        """
        import cocoex  # runtime import

        if output_folder is None:
            output_folder = "{}_on_{}_{}D".format(
                self.algorithm_name,
                self.suite_name,
                int(self.budget_multiplier + 0.499),
            )

        # ---- COCO infrastructure ----------------------------------------
        suite = cocoex.Suite(
            self.suite_name, self.suite_instance, self.suite_options,
        )
        observer = cocoex.Observer(
            self.suite_name,
            "result_folder: {} algorithm_name: {}".format(
                output_folder, self.algorithm_name,
            ),
        )
        # Run exactly one sweep over the suite and do not require COCO's
        # `final_target_hit` successes (we only want one run per problem instance).
        repeater = cocoex.ExperimentRepeater(
            self.budget_multiplier,
            min_successes=0,
            max_sweeps=1,
        )
        minimal_print = cocoex.utilities.MiniPrint()
        timings: Dict[int, List[float]] = collections.defaultdict(list)

        # Initialize CSV logger for custom experiment logging
        result_folder = observer.result_folder
        csv_logger = COCOCSVLogger(
            Path(result_folder) / "experiment_results.csv",
        )

        # Save the experiment config into the result folder for
        # reproducibility.  The observer creates the directory on
        # construction, so `result_folder` is available immediately.
        if config is not None:
            config_dest = Path(result_folder) / "experiment_config.json"
            config_dest.parent.mkdir(parents=True, exist_ok=True)
            with config_dest.open("w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=4, default=str)

        time0 = time.time()
        problems_attempted = 0
        problems_solved = 0
        result_path = observer.result_folder

        # Track problems that reached our custom precision so we skip
        # them in subsequent sweeps (the COCO repeater only knows about
        # its hard-coded 1e-8 target and would otherwise keep retrying).
        custom_converged: set = set()
        # Track problems that have been logged to CSV (one row per problem)
        logged_problems: set = set()

        try:
            # ---- Main experiment loop (with restarts) -------------------
            while not repeater.done():
                for problem in suite:
                    if repeater.done(problem):
                        continue

                    # Skip problems already converged by our custom criterion.
                    if problem.id in custom_converged:
                        continue

                    problem.observe_with(observer)
                    time1 = time.time()
                    problems_attempted += 1

                    # Standard COCO practice: evaluate the zero vector for
                    # comparability across algorithms.  This also causes the
                    # observer to create the .dat file whose header contains
                    # Fopt, which we parse below.
                    problem(problem.dimension * [0])

                    # Extract the known optimal value from the observer data
                    # so we can check convergence against `self.precision`.
                    fopt = self._extract_fopt(problem, result_folder)

                    # Create a *fresh* optimizer for this problem (attempt).
                    optimizer = self._create_optimizer(problem)
                    if optimizer.verbose:
                        optimizer.set_verbose_log_dir(
                            Path(result_folder) / "verbose_logs"
                        )
                        optimizer.set_problem_tag(
                            f"f{problem.id_function}_d{problem.dimension}_i{problem.id_instance}"
                        )

                    # Run the ask / evaluate / tell loop.
                    self._run_on_problem(optimizer, problem, fopt=fopt)

                    if optimizer.verbose:
                        optimizer.plot_convergence(
                            title="{} — f{} d{} i{}".format(
                                self.algorithm_name,
                                problem.id_function,
                                problem.dimension,
                                problem.id_instance,
                            )
                        )

                    converged = self._is_converged(problem, fopt)
                    if converged:
                        problems_solved += 1
                        custom_converged.add(problem.id)

                    # Log this problem instance to CSV (only once per problem)
                    if problem.id not in logged_problems:
                        csv_logger.log_problem(
                            method_name=self.algorithm_name,
                            problem=problem,
                            fopt=fopt,
                            precision=self.precision,
                            converged=converged,
                        )
                        logged_problems.add(problem.id)

                    # Record timing for the first sweep only.
                    evals = max(1, problem.evaluations)
                    if repeater._sweeps == 1:
                        timings[problem.dimension].append(
                            (time.time() - time1) / evals,
                        )

                    repeater.track(problem)
                    minimal_print(problem)

                # If every problem in the suite has custom-converged, stop.
                if len(custom_converged) >= len(suite):
                    break
        except Exception:
            raise
        finally:
            # Always materialize the CSV table, even on failures.
            csv_logger.write()
            print("\n  CSV results:  {}".format(csv_logger.output_path))

        # ---- Summary ----------------------------------------------------
        elapsed = time.time() - time0

        print("\n" + "=" * 60)
        print("COCO experiment completed")
        print("=" * 60)
        print("  Algorithm:   {}".format(self.algorithm_name))
        print("  Suite:       {}".format(self.suite_name))
        print("  Budget:      {} x dimension".format(self.budget_multiplier))
        print("  Precision:   {:.0e}".format(self.precision))
        print("  Attempted:   {} problem instances".format(problems_attempted))
        print("  Solved:      {} (gap < {:.0e})".format(
            problems_solved, self.precision,
        ))
        print("  Wall time:   {:.1f}s".format(elapsed))
        print("  Results in:  {}".format(result_path))

        if timings:
            print("\n  Timing (seconds / evaluation):")
            print("    dim   median")
            print("    ----  ------")
            for dim in sorted(timings):
                ts = sorted(timings[dim])
                median = (ts[len(ts) // 2] + ts[-1 - len(ts) // 2]) / 2
                print("    {:4d}  {:.2e}".format(dim, median))

        print("=" * 60)

        # ---- Optional post-processing -----------------------------------
        if post_process:
            try:
                import cocopp

                print("\nRunning cocopp post-processing...")
                cocopp.main(result_path)
                print("Post-processing complete.")
            except ImportError:
                print(
                    "\nWARNING: cocopp is not installed. "
                    "Install with: pip install cocopp"
                )
            except Exception as exc:
                print("\nWARNING: cocopp failed: {}".format(exc))

        return result_path

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _create_optimizer(self, problem: "cocoex.Problem") -> BaseOptimizer:
        """Instantiate a fresh optimizer for *problem*."""
        dimension: int = problem.dimension
        lower: np.ndarray = np.array(problem.lower_bounds, dtype=np.float64)
        upper: np.ndarray = np.array(problem.upper_bounds, dtype=np.float64)
        return self.optimizer_factory(dimension, lower, upper)

    @staticmethod
    def _extract_fopt(
        problem: "cocoex.Problem",
        result_folder: str,
    ) -> Optional[float]:
        """Read ``f_opt`` for the current instance from the COCO ``.dat`` file.

        The observer writes a header line before each instance's data::

            % ... best noise-free fitness - Fopt (7.948000000000e+01) ...

        Because different BBOB instances of the same function have
        different optimal values, this method reads the **last** such
        header line in the file, which corresponds to the most recently
        observed (i.e. current) instance.
        """
        import glob as _glob
        import re as _re

        func_id = int(problem.id_function)
        dim = problem.dimension
        dat_pattern = str(
            Path(result_folder)
            / "data_f{}".format(func_id)
            / "bbobexp_f{}_DIM{}*.dat".format(func_id, dim)
        )
        dat_files = sorted(_glob.glob(dat_pattern))
        if not dat_files:
            return None

        fopt_re = _re.compile(r"Fopt\s*\(([^)]+)\)")
        last_fopt: Optional[float] = None
        with open(dat_files[-1], encoding="utf-8") as fh:
            for line in fh:
                m = fopt_re.search(line)
                if m is not None:
                    try:
                        last_fopt = float(m.group(1))
                    except ValueError:
                        pass
        return last_fopt

    def _is_converged(
        self,
        problem: "cocoex.Problem",
        fopt: Optional[float],
    ) -> bool:
        """Return ``True`` when the problem should be considered solved.

        Uses the custom :attr:`precision` when ``fopt`` is available;
        otherwise falls back to COCO's built-in ``final_target_hit``
        (precision ``1e-8``).
        """
        if problem.final_target_hit:
            return True
        if fopt is not None:
            gap = problem.best_observed_fvalue1 - fopt
            if gap < self.precision:
                return True
        return False

    def _run_on_problem(
        self,
        optimizer: BaseOptimizer,
        problem: "cocoex.Problem",
        fopt: Optional[float] = None,
    ) -> None:
        """Execute the ask/tell loop on a single COCO problem instance.

        The loop terminates when either:
        - the evaluation budget is exhausted, or
        - the convergence criterion is met (see :meth:`_is_converged`).

        Parameters
        ----------
        optimizer : BaseOptimizer
            A freshly created optimizer (from the factory).
        problem : cocoex.Problem
            The current COCO problem instance.
        fopt : float, optional
            Known optimal function value.  When provided, convergence is
            checked against :attr:`precision` instead of COCO's default
            ``1e-8``.
        """
        budget = int(self.budget_multiplier * problem.dimension)

        if hasattr(optimizer, "budget"):
            optimizer.budget = budget

        # Seed the optimizer with the COCO initial solution so it has at
        # least one data point before the first ask/tell cycle.
        x0 = np.array(problem.initial_solution, dtype=np.float64)
        y0: float = problem(x0)
        optimizer.tell(
            x0.reshape(1, -1),
            np.array([y0], dtype=np.float64),
        )

        while (
            problem.evaluations < budget
            and not self._is_converged(problem, fopt)
        ):
            remaining = budget - problem.evaluations
            n_ask = min(self.eval_batch_size, remaining)
            if n_ask <= 0:
                break

            # --- ask: optimizer proposes candidates ----------------------
            candidates = optimizer.ask(n=n_ask)

            # --- evaluate: COCO records every problem(x) call -----------
            values = np.array(
                [problem(candidates[i]) for i in range(len(candidates))],
                dtype=np.float64,
            )

            # --- tell: feed evaluations back to the optimizer -----------
            optimizer.tell(candidates, values)


# =========================================================================== #
#  Convenience function                                                        #
# =========================================================================== #


def run_coco_experiment(
    method: Optional[str] = None,
    optimizer_factory: Optional[OptimizerFactory] = None,
    algorithm_name: Optional[str] = None,
    budget_multiplier: float = 1e4,
    eval_batch_size: int = 1,
    precision: float = 1e-8,
    suite_name: str = "bbob",
    suite_instance: str = "",
    suite_options: str = "",
    output_folder: Optional[str] = None,
    post_process: bool = False,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """One-call entry point to run a COCO experiment.

    Either supply a pre-built *optimizer_factory*, or provide *method*
    (one of ``"diffusion"``, ``"diffusion_bbo"``, ``"cma_es"``, ``"tpe"``) together with
    *optimizer_kwargs* to have one created automatically.

    Parameters
    ----------
    method : str, optional
        Optimizer method name.  Used to auto-build the factory when
        *optimizer_factory* is ``None``.  Defaults to ``"diffusion"``
        when neither *method* nor *optimizer_factory* is supplied.
    optimizer_factory : callable, optional
        A ``(dimension, lower, upper) -> BaseOptimizer`` factory.
        Takes precedence over *method* when both are given.
    algorithm_name : str, optional
        Name for COCO data files.  Defaults to *method* in uppercase
        if not given.
    budget_multiplier : float
        Budget as a multiple of problem dimension.
    eval_batch_size : int
        Number of candidates per ask() call.
    precision : float
        Convergence precision (``f_best - f_opt < precision`` to solve).
    suite_name, suite_instance, suite_options : str
        COCO suite configuration strings.
    output_folder : str, optional
        Result folder name (auto-generated if ``None``).
    post_process : bool
        Run cocopp after the experiment.
    optimizer_kwargs : dict, optional
        Keyword arguments forwarded to the optimizer constructor when
        building the factory from *method*.
    config : dict, optional
        The full experiment configuration dict.  When provided it is
        persisted as ``experiment_config.json`` inside the result folder.

    Returns
    -------
    str
        Path to the COCO result folder.
    """
    if optimizer_factory is None:
        resolved_method = method or "diffusion"
        optimizer_factory = make_factory(
            resolved_method, **(optimizer_kwargs or {}),
        )
    else:
        resolved_method = method or "custom"

    if algorithm_name is None:
        algorithm_name = resolved_method.upper()

    runner = COCOExperimentRunner(
        optimizer_factory=optimizer_factory,
        algorithm_name=algorithm_name,
        budget_multiplier=budget_multiplier,
        eval_batch_size=eval_batch_size,
        precision=precision,
        suite_name=suite_name,
        suite_instance=suite_instance,
        suite_options=suite_options,
    )
    return runner.run(
        output_folder=output_folder,
        post_process=post_process,
        config=config,
    )


# =========================================================================== #
#  Configuration from JSON                                                     #
# =========================================================================== #


def load_coco_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate a COCO experiment JSON config.

    Expected schema::

        {
            "method": "diffusion",          // or "cma_es" or "tpe"
            "algorithm_name": "DiffusionBBO",
            "suite_name": "bbob",
            "suite_instance": "",
            "suite_options": "dimensions: 2,3,5,10,20",
            "budget_multiplier": 10000,
            "eval_batch_size": 1,
            "precision": 1e-8,              // convergence gap threshold
            "output_folder": "exdata/DiffusionBBO",
            "post_process": false,
            "optimizer": { ... method-specific kwargs ... }
        }

    Parameters
    ----------
    config_path : str or Path
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the config is not a valid JSON object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError("COCO config not found: {}".format(path))
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError("COCO config must be a JSON object (dict).")
    return config


_RUNNER_KEYS = {
    "algorithm_name",
    "suite_name",
    "suite_instance",
    "suite_options",
    "budget_multiplier",
    "eval_batch_size",
    "precision",
}


def run_from_config(config: Dict[str, Any]) -> str:
    """Build and run a COCO experiment from a parsed config dict.

    The ``"method"`` field selects which optimizer factory to use:

    - ``"diffusion"`` (default) — :func:`make_diffusion_factory`
    - ``"cma_es"`` — :func:`make_cmaes_factory`
    - ``"tpe"`` — :func:`make_tpe_factory`

    Parameters
    ----------
    config : dict
        Configuration as returned by :func:`load_coco_config`.

    Returns
    -------
    str
        Path to the COCO result folder.
    """
    method = config.get("method", "diffusion")

    # Separate runner-level keys from optimizer kwargs.
    runner_kwargs: Dict[str, Any] = {}
    for key in _RUNNER_KEYS:
        if key in config:
            runner_kwargs[key] = config[key]

    optimizer_kwargs = config.get("optimizer", {})
    output_folder = config.get("output_folder", None)
    post_process = config.get("post_process", False)

    return run_coco_experiment(
        method=method,
        optimizer_kwargs=optimizer_kwargs,
        output_folder=output_folder,
        post_process=post_process,
        config=config,
        **runner_kwargs,
    )


# =========================================================================== #
#  CLI entry point                                                             #
# =========================================================================== #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a COCO/BBOB experiment with a black-box optimizer "
            "(diffusion, diffusion_bbo, cma_es, tpe, or gp_qei)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a JSON configuration file.  If provided, all other "
            "CLI flags are ignored (except --post-process which can "
            "override the config value)."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="diffusion",
        choices=["diffusion", "diffusion_bbo", "cma_es", "tpe", "gp_qei"],
        help="Optimizer method to benchmark.",
    )
    parser.add_argument(
        "--algorithm-name",
        type=str,
        default=None,
        help="Algorithm name for COCO data files (default: method name).",
    )
    parser.add_argument(
        "--suite-name",
        type=str,
        default="bbob",
        help="COCO suite identifier.",
    )
    parser.add_argument(
        "--suite-instance",
        type=str,
        default="",
        help="Suite instance filter string.",
    )
    parser.add_argument(
        "--suite-options",
        type=str,
        default="",
        help="Additional suite options (e.g. 'dimensions: 2,3,5').",
    )
    parser.add_argument(
        "--budget-multiplier",
        type=float,
        default=1e4,
        help="Budget as a multiple of problem dimension.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Number of candidates per ask() call.",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1e-8,
        help="Convergence precision: stop when f_best - f_opt < precision.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="COCO result folder name.",
    )
    parser.add_argument(
        "--post-process",
        action="store_true",
        default=False,
        help="Run cocopp post-processing after the experiment.",
    )
    # --- Shared optimizer parameters ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sample/value/index verbose logging to .npz.",
    )
    # --- CMA-ES specific ---
    parser.add_argument("--sigma0", type=float, default=0.5)
    # --- Diffusion (original) specific ---
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--beta-schedule", type=str, default="linear")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-diffusion", type=float, default=1e-3)
    parser.add_argument("--lr-regressor", type=float, default=1e-3)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--guidance-strength", type=float, default=5.0)
    parser.add_argument(
        "--use-regressor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--min-data", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    # --- Diffusion-BBO specific ---
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--p-uncond", type=float, default=0.15)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--n-uae-samples", type=int, default=20)
    # --- GP-qEI specific ---
    parser.add_argument("--n-initial", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=256)
    parser.add_argument("--gp-num-restarts", type=int, default=10)
    parser.add_argument("--gp-raw-samples", type=int, default=512)
    parser.add_argument("--max-train-size", type=int, default=None)
    parser.add_argument("--gp-dtype", type=str, default="float64")
    return parser


# Maps CLI attribute names to optimizer kwarg names, grouped by method.
_CLI_KWARGS_SHARED = {
    "seed": "seed",
    "verbose": "verbose",
}

_CLI_KWARGS_CMAES = {
    "sigma0": "sigma0",
}

_CLI_KWARGS_DIFFUSION = {
    "num_timesteps": "num_timesteps",
    "beta_schedule": "beta_schedule",
    "hidden_dim": "hidden_dim",
    "time_embed_dim": "time_embed_dim",
    "depth": "depth",
    "batch_size": "batch_size",
    "lr_diffusion": "lr_diffusion",
    "lr_regressor": "lr_regressor",
    "train_steps": "train_steps",
    "quantile": "quantile",
    "guidance_strength": "guidance_strength",
    "use_regressor": "use_regressor",
    "min_data": "min_data",
    "device": "device",
}

_CLI_KWARGS_DIFFUSION_BBO = {
    "num_timesteps": "num_timesteps",
    "beta_schedule": "beta_schedule",
    "hidden_dim": "hidden_dim",
    "time_embed_dim": "time_embed_dim",
    "batch_size": "batch_size",
    "lr": "lr",
    "train_steps": "train_steps",
    "n_ensemble": "n_ensemble",
    "p_uncond": "p_uncond",
    "guidance_scale": "guidance_scale",
    "n_uae_samples": "n_uae_samples",
    "min_data": "min_data",
    "device": "device",
}

_CLI_KWARGS_GP_QEI = {
    "n_initial": "n_initial",
    "mc_samples": "mc_samples",
    "gp_num_restarts": "num_restarts",
    "gp_raw_samples": "raw_samples",
    "max_train_size": "max_train_size",
    "device": "device",
    "gp_dtype": "dtype",
}


def _cli_to_optimizer_kwargs(
    args: argparse.Namespace,
    method: str,
) -> Dict[str, Any]:
    """Extract optimizer kwargs from parsed CLI arguments for *method*."""
    mapping: Dict[str, str] = dict(_CLI_KWARGS_SHARED)
    if method == "diffusion":
        mapping.update(_CLI_KWARGS_DIFFUSION)
    elif method == "diffusion_bbo":
        mapping.update(_CLI_KWARGS_DIFFUSION_BBO)
    elif method == "cma_es":
        mapping.update(_CLI_KWARGS_CMAES)
    elif method == "gp_qei":
        mapping.update(_CLI_KWARGS_GP_QEI)
    # TPE only uses the shared keys (seed).

    kwargs: Dict[str, Any] = {}
    for cli_attr, kwarg_name in mapping.items():
        value = getattr(args, cli_attr, None)
        if value is not None:
            kwargs[kwarg_name] = value
    return kwargs


def cli_main() -> None:
    """CLI entry point for ``python -m benchmarks.coco_wrapper``."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.config is not None:
        config = load_coco_config(args.config)
        # Allow --post-process to override config
        if args.post_process:
            config["post_process"] = True
        result_folder = run_from_config(config)
    else:
        method = args.method
        optimizer_kwargs = _cli_to_optimizer_kwargs(args, method)
        # Build an effective config dict so it gets saved alongside
        # the COCO results for reproducibility.
        config: Dict[str, Any] = {
            "method": method,
            "algorithm_name": args.algorithm_name,
            "suite_name": args.suite_name,
            "suite_instance": args.suite_instance,
            "suite_options": args.suite_options,
            "budget_multiplier": args.budget_multiplier,
            "eval_batch_size": args.eval_batch_size,
            "precision": args.precision,
            "output_folder": args.output_folder,
            "post_process": args.post_process,
            "optimizer": optimizer_kwargs,
        }
        result_folder = run_coco_experiment(
            method=method,
            algorithm_name=args.algorithm_name,
            budget_multiplier=args.budget_multiplier,
            eval_batch_size=args.eval_batch_size,
            precision=args.precision,
            suite_name=args.suite_name,
            suite_instance=args.suite_instance,
            suite_options=args.suite_options,
            output_folder=args.output_folder,
            post_process=args.post_process,
            optimizer_kwargs=optimizer_kwargs,
            config=config,
        )

    print("\nResults written to: {}".format(result_folder))


if __name__ == "__main__":
    cli_main()
