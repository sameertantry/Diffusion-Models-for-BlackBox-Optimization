"""Tree-structured Parzen Estimator (TPE) optimizer wrapper."""

from typing import Tuple

import numpy as np
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from methods.base import BaseOptimizer


class TPE(BaseOptimizer):
    """Tree-structured Parzen Estimator optimizer using Optuna.
    
    This wrapper adapts Optuna's TPE sampler to the BaseOptimizer interface.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        seed: int = 42,
        verbose: bool = False,
    ):
        """Initialize the TPE optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tuple of (lower_bounds, upper_bounds) arrays.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            name="TPE",
            input_dim=input_dim,
            bounds=bounds,
            verbose=verbose,
        )
        
        self.seed = seed
        self.lower = np.atleast_1d(bounds[0])
        self.upper = np.atleast_1d(bounds[1])
        
        # Ensure bounds are broadcastable to input_dim
        if len(self.lower) == 1:
            self.lower = np.full(input_dim, self.lower[0])
        if len(self.upper) == 1:
            self.upper = np.full(input_dim, self.upper[0])
        
        # Initialize Optuna study
        import optuna
        self.study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=seed),
        )
        
        # Track pending trials
        self.pending_trials = []
        self.trial_to_candidate = {}
    
    def ask(self, n: int = 1) -> np.ndarray:
        """Propose candidate solution(s) for evaluation.
        
        Args:
            n: Number of candidates to propose (default: 1).
        
        Returns:
            Array of shape (n, input_dim) with candidate inputs.
        """
        candidates = []
        
        for _ in range(n):
            # Create a new trial
            trial = self.study.ask()
            self.pending_trials.append(trial)
            
            # Extract suggested values
            candidate = np.array([
                trial.suggest_float(f"x_{i}", float(self.lower[i]), float(self.upper[i]))
                for i in range(self.input_dim)
            ])
            
            candidates.append(candidate)
        
        candidates_arr = np.array(candidates)
        self._register_asked(candidates_arr)
        return candidates_arr
    
    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        """Report evaluation results to the optimizer.
        
        Args:
            x: Array of shape (n, input_dim) with evaluated inputs.
            y: Array of shape (n,) or (n, 1) with function values.
        """
        y = np.atleast_1d(y).flatten()
        self._register_told(x, y)
        
        # Report results for pending trials (in order)
        num_to_report = min(len(self.pending_trials), len(y))
        for i in range(num_to_report):
            trial = self.pending_trials[i]
            value = float(y[i])
            self.study.tell(trial, value, state=TrialState.COMPLETE)
            self.num_evals += 1
        
        # Clear reported trials
        self.pending_trials = self.pending_trials[num_to_report:]
    
    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    def warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Warm-start TPE by injecting completed trials from initial data.

        Each ``(x_i, y_i)`` pair is added as a completed trial so the
        internal Tree-structured Parzen Estimator has informative priors
        from the very first ``ask()`` call.

        Args:
            x: Array of shape ``(n, input_dim)`` – evaluated inputs.
            y: Array of shape ``(n,)`` – objective values (lower is better).
        """
        import optuna
        from optuna.distributions import FloatDistribution

        x = np.asarray(x, dtype=np.float64)
        y = np.atleast_1d(np.asarray(y, dtype=np.float64)).flatten()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if len(x) == 0 or len(y) == 0:
            return

        for xi, yi in zip(x, y):
            # Build the parameter dict matching ask() format
            params = {f"x_{d}": float(xi[d]) for d in range(self.input_dim)}
            distributions = {
                f"x_{d}": FloatDistribution(
                    float(self.lower[d]), float(self.upper[d])
                )
                for d in range(self.input_dim)
            }
            self.study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    values=[float(yi)],
                    state=TrialState.COMPLETE,
                )
            )

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the optimizer's internal state."""
        import optuna
        
        # Create a new study
        self.study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.seed),
        )
        
        # Reset tracking variables
        self.pending_trials = []
        self.num_evals = 0
        self._reset_verbose_trace()
