"""REINFORCE (policy gradient) optimizer for black-box optimization.

Implements the REINFORCE algorithm (Williams, 1992) adapted for black-box
minimisation.  A diagonal Gaussian policy N(mu, diag(sigma^2)) is maintained
over the search space.  Candidates are sampled from this policy, evaluated on
the objective, and the policy parameters are updated via the REINFORCE gradient
estimator so that the distribution shifts towards regions of lower cost.

Features:
    - Rank-based fitness shaping (NES-style) for robustness to objective scale.
    - Mean baseline for variance reduction (when rank transform is off).
    - Entropy regularisation to encourage exploration.
    - Separate learning rates for mean and log-standard-deviation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from methods.base import BaseOptimizer


class REINFORCE(BaseOptimizer):
    """REINFORCE policy-gradient optimizer for black-box optimisation.

    Maintains a diagonal Gaussian policy ``N(mu, diag(sigma^2))`` and updates
    it using the REINFORCE gradient estimator.  Samples are accumulated in an
    internal buffer; when the buffer reaches ``pop_size`` a gradient step is
    performed.

    Attributes:
        mu: Current mean of the Gaussian policy, shape ``(input_dim,)``.
        log_std: Current log-standard-deviation, shape ``(input_dim,)``.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        seed: int = 42,
        lr: float = 0.01,
        std_lr: Optional[float] = None,
        pop_size: int = 32,
        init_std: float = 0.3,
        entropy_coeff: float = 0.0,
        use_rank_transform: bool = True,
    ):
        """Initialise the REINFORCE optimizer.

        Args:
            input_dim: Dimensionality of the search space.
            bounds: ``(lower, upper)`` arrays defining the search domain.
            seed: Random seed for reproducibility.
            lr: Learning rate for the mean parameters.
            std_lr: Learning rate for ``log_std``.  Defaults to ``lr``.
            pop_size: Number of samples to accumulate before each gradient
                update.  Setting ``eval_batch_size`` in the config to this
                value is the most efficient usage.
            init_std: Initial standard deviation for each dimension.
            entropy_coeff: Coefficient for entropy regularisation (>0
                encourages exploration).
            use_rank_transform: If ``True`` (default), apply NES-style
                rank-based fitness shaping instead of raw rewards with a
                mean baseline.
        """
        super().__init__(
            name="REINFORCE",
            input_dim=input_dim,
            bounds=bounds,
        )

        self.seed = seed
        self.lr = lr
        self.std_lr = std_lr if std_lr is not None else lr
        self.pop_size = pop_size
        self.init_std = init_std
        self.entropy_coeff = entropy_coeff
        self.use_rank_transform = use_rank_transform

        self.lower = np.atleast_1d(bounds[0]).astype(np.float64)
        self.upper = np.atleast_1d(bounds[1]).astype(np.float64)
        if len(self.lower) == 1:
            self.lower = np.full(input_dim, self.lower[0])
        if len(self.upper) == 1:
            self.upper = np.full(input_dim, self.upper[0])

        self._rng = np.random.RandomState(seed)

        # Policy parameters
        self.mu = (self.lower + self.upper) / 2.0
        self.log_std = np.full(input_dim, np.log(init_std), dtype=np.float64)

        # Accumulation buffers
        self._x_buffer: list = []
        self._y_buffer: list = []

        # Running baseline (used when rank transform is off)
        self._baseline: float = 0.0
        self._baseline_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sigma(self) -> np.ndarray:
        """Current standard deviation vector."""
        return np.exp(self.log_std)

    # ------------------------------------------------------------------
    # Fitness shaping
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_transform(y: np.ndarray) -> np.ndarray:
        """NES-style rank-based fitness shaping.

        Lower objective values receive higher (more positive) utilities so
        that the gradient update moves the policy towards better regions.

        Args:
            y: Objective values, shape ``(n,)``.

        Returns:
            Centred utility values, shape ``(n,)``.
        """
        n = len(y)
        ranks = np.empty(n, dtype=np.float64)
        order = np.argsort(y)  # ascending; best (lowest) first
        ranks[order] = np.arange(n, dtype=np.float64)

        utilities = np.maximum(0.0, np.log(n / 2.0 + 1.0) - np.log(ranks + 1.0))
        total = utilities.sum()
        if total > 0:
            utilities = utilities / total - 1.0 / n
        return utilities

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def ask(self, n: int = 1) -> np.ndarray:
        """Sample *n* candidate solutions from the current Gaussian policy.

        Args:
            n: Number of candidates to generate.

        Returns:
            Array of shape ``(n, input_dim)`` clipped to the search bounds.
        """
        sigma = self.sigma
        candidates = self._rng.randn(n, self.input_dim) * sigma + self.mu
        candidates = np.clip(candidates, self.lower, self.upper)
        return candidates

    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        """Record evaluations and update the policy when buffer is full.

        Args:
            x: Evaluated inputs, shape ``(n, input_dim)``.
            y: Objective values, shape ``(n,)``.
        """
        x = np.atleast_2d(x)
        y = np.atleast_1d(y).flatten()

        for xi, yi in zip(x, y):
            self._x_buffer.append(xi.copy())
            self._y_buffer.append(float(yi))
            self.num_evals += 1

        # Perform an update for every complete batch in the buffer
        while len(self._x_buffer) >= self.pop_size:
            self._update()

    # ------------------------------------------------------------------
    # Gradient update
    # ------------------------------------------------------------------

    def _update(self) -> None:
        """Single REINFORCE gradient-ascent step on the policy parameters."""
        xs = np.array(self._x_buffer[: self.pop_size])
        ys = np.array(self._y_buffer[: self.pop_size])

        # Consume the used samples; keep any overflow for the next update
        self._x_buffer = self._x_buffer[self.pop_size :]
        self._y_buffer = self._y_buffer[self.pop_size :]

        sigma = self.sigma
        sigma_sq = sigma ** 2 + 1e-8  # (d,)

        # --- Advantages ---
        if self.use_rank_transform:
            advantages = self._rank_transform(ys)
        else:
            rewards = -ys  # negate for minimisation
            n = len(rewards)
            self._baseline = (
                self._baseline * self._baseline_count + rewards.sum()
            ) / (self._baseline_count + n)
            self._baseline_count += n
            advantages = rewards - self._baseline

        # --- Score functions ---
        diff = xs - self.mu  # (n, d)
        score_mu = diff / sigma_sq  # ∇_mu log π
        score_log_std = diff ** 2 / sigma_sq - 1.0  # ∇_log_sigma log π

        # Weighted mean of score functions
        adv = advantages[:, None]  # (n, 1)
        grad_mu = (adv * score_mu).mean(axis=0)
        grad_log_std = (adv * score_log_std).mean(axis=0)

        # Entropy bonus: ∇_log_std H = 1 (entropy grows with sigma)
        if self.entropy_coeff > 0:
            grad_log_std += self.entropy_coeff

        # Gradient *ascent* (maximise expected reward ⟹ minimise objective)
        self.mu += self.lr * grad_mu
        self.log_std += self.std_lr * grad_log_std

        # Keep parameters in sensible ranges
        self.mu = np.clip(self.mu, self.lower, self.upper)
        self.log_std = np.clip(self.log_std, np.log(1e-4), np.log(10.0))

    # ------------------------------------------------------------------
    # Warm-start & reset
    # ------------------------------------------------------------------

    def warm_start(self, x: np.ndarray, y: np.ndarray) -> None:
        """Centre the policy on the best observed point.

        Args:
            x: Previously evaluated inputs, shape ``(n, input_dim)``.
            y: Corresponding objective values, shape ``(n,)``.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.atleast_1d(np.asarray(y, dtype=np.float64)).flatten()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if len(x) == 0 or len(y) == 0:
            return

        best_idx = int(np.argmin(y))
        self.mu = np.clip(x[best_idx].copy(), self.lower, self.upper)

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        self._rng = np.random.RandomState(self.seed)
        self.mu = (self.lower + self.upper) / 2.0
        self.log_std = np.full(self.input_dim, np.log(self.init_std), dtype=np.float64)
        self._x_buffer = []
        self._y_buffer = []
        self._baseline = 0.0
        self._baseline_count = 0
        self.num_evals = 0
