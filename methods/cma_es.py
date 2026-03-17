"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer wrapper."""

from typing import Tuple

import cma
import numpy as np

from methods.base import BaseOptimizer


class CMAES(BaseOptimizer):
    """CMA-ES optimizer using the cma library.
    
    This wrapper adapts the cma library to the BaseOptimizer interface.
    Supports continuous domains only.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        seed: int = 42,
        sigma0: float = 0.5,
        verbose: bool = False,
    ):
        """Initialize the CMA-ES optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tuple of (lower_bounds, upper_bounds) arrays.
            seed: Random seed for reproducibility.
            sigma0: Initial standard deviation (default: 0.5).
        """
        super().__init__(
            name="CMA-ES",
            input_dim=input_dim,
            bounds=bounds,
            verbose=verbose,
        )
        
        self.seed = seed
        self.sigma0 = sigma0
        self.lower = np.atleast_1d(bounds[0])
        self.upper = np.atleast_1d(bounds[1])
        
        # Ensure bounds are broadcastable to input_dim
        if len(self.lower) == 1:
            self.lower = np.full(input_dim, self.lower[0])
        if len(self.upper) == 1:
            self.upper = np.full(input_dim, self.upper[0])
        
        # Compute center and scale for CMA-ES
        self.center = (self.lower + self.upper) / 2.0
        self.scale = (self.upper - self.lower) / 2.0
        
        # Initialize CMA-ES
        self.es = None
        self._rng = None
        self._initialize_es()
        
        # Track pending candidates and population buffer
        self.pending_candidates = []
        self.population_buffer = []  # Store full population from CMA-ES
        self.population_buffer_normalized = []  # Store normalized versions
    
    def _initialize_es(self) -> None:
        """Initialize the CMA-ES optimizer."""
        self._rng = np.random.RandomState(self.seed)
        # Normalize initial center to [0, 0, ...] (CMA-ES works in normalized space)
        x0 = np.zeros(self.input_dim)
        
        # Initialize CMA-ES with normalized sigma
        # Scale sigma0 by the domain size
        normalized_sigma0 = self.sigma0 * np.min(self.scale)
        
        self.es = cma.CMAEvolutionStrategy(
            x0,
            normalized_sigma0,
            {
                "bounds": [np.full(self.input_dim, -1.0), np.full(self.input_dim, 1.0)],
                "seed": self.seed,
                "randn": self._rng.randn,
                "verb_disp": 0,  # Suppress output
                "verb_log": 0,
            },
        )
    
    def ask(self, n: int = 1) -> np.ndarray:
        """Propose candidate solution(s) for evaluation.
        
        Args:
            n: Number of candidates to propose (default: 1).
        
        Returns:
            Array of shape (n, input_dim) with candidate inputs.
        """
        candidates = []
        
        # If buffer is empty, ask CMA-ES for a full population
        if not self.population_buffer:
            # Ask CMA-ES for its full population (it determines the size automatically)
            population_normalized = self.es.ask()
            self.population_buffer_normalized = list(population_normalized)
            
            # Denormalize all candidates
            for candidate_normalized in population_normalized:
                candidate = self.center + np.array(candidate_normalized) * self.scale
                candidate = np.clip(candidate, self.lower, self.upper)
                self.population_buffer.append(candidate)
        
        # Serve candidates from buffer
        for _ in range(n):
            if not self.population_buffer:
                # Should not happen, but handle gracefully
                break
            candidate = self.population_buffer.pop(0)
            candidate_normalized = self.population_buffer_normalized.pop(0)
            candidates.append(candidate)
            self.pending_candidates.append(candidate_normalized)

        candidates_arr = np.array(candidates) if candidates else np.empty((0, self.input_dim))
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
        
        # Use the normalized versions we stored during ask()
        num_to_report = min(len(self.pending_candidates), len(y))
        solutions = self.pending_candidates[:num_to_report]
        values = y[:num_to_report].tolist()
        
        # Store results until we have a full population
        # CMA-ES expects all results from a generation at once
        if not hasattr(self, '_pending_solutions'):
            self._pending_solutions = []
            self._pending_values = []
        
        self._pending_solutions.extend(solutions)
        self._pending_values.extend(values)
        self.num_evals += len(solutions)
        
        # Remove reported candidates from pending list
        self.pending_candidates = self.pending_candidates[num_to_report:]
        
        # If we've collected a full population (or buffer is empty and we have results), tell CMA-ES
        # Check if we have enough for a full population by comparing with buffer state
        if not self.population_buffer and self._pending_solutions:
            # We've evaluated a full population, tell CMA-ES
            self.es.tell(self._pending_solutions, self._pending_values)
            self._pending_solutions = []
            self._pending_values = []
    
    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    def warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Warm-start CMA-ES by re-centering on the best observed point.

        The CMA evolution strategy is re-initialized with ``x0`` set to the
        best (lowest objective value) point in the provided data.  This
        replaces the default initialization at the center of the domain
        bounds and lets CMA-ES explore from a region of known good quality.

        Args:
            x: Array of shape ``(n, input_dim)`` – evaluated inputs.
            y: Array of shape ``(n,)`` – objective values (lower is better).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.atleast_1d(np.asarray(y, dtype=np.float64)).flatten()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if len(x) == 0 or len(y) == 0:
            return

        best_idx = int(np.argmin(y))
        best_x = x[best_idx]

        # Map to normalised space  [-1, 1]
        x0_normalized = (best_x - self.center) / np.where(
            self.scale > 0, self.scale, 1.0
        )
        x0_normalized = np.clip(x0_normalized, -1.0, 1.0)

        # Re-initialize the evolution strategy from the new starting point
        self._rng = np.random.RandomState(self.seed)
        normalized_sigma0 = self.sigma0 * np.min(self.scale)
        self.es = cma.CMAEvolutionStrategy(
            x0_normalized,
            normalized_sigma0,
            {
                "bounds": [
                    np.full(self.input_dim, -1.0),
                    np.full(self.input_dim, 1.0),
                ],
                "seed": self.seed,
                "randn": self._rng.randn,
                "verb_disp": 0,
                "verb_log": 0,
            },
        )

        # Clear any stale buffers
        self.pending_candidates = []
        self.population_buffer = []
        self.population_buffer_normalized = []
        self._pending_solutions = []
        self._pending_values = []

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the optimizer's internal state."""
        # Reinitialize CMA-ES
        self._initialize_es()
        
        # Reset tracking variables
        self.pending_candidates = []
        self.population_buffer = []
        self.population_buffer_normalized = []
        if hasattr(self, '_pending_solutions'):
            self._pending_solutions = []
            self._pending_values = []
        self.num_evals = 0
        self._reset_verbose_trace()
