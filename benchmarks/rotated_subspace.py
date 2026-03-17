"""Rotated subspace benchmark wrapper for analytical functions."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from benchmarks.analytical import Ackley, Rastrigin, Rosenbrock, Sphere
from benchmarks.base import BaseBenchmark


class RotatedSubspaceBenchmark(BaseBenchmark):
    """Rotate inputs in R^D and evaluate on first d coordinates."""

    def __init__(
        self,
        base_function_name: str,
        ambient_dim: int,
        intrinsic_dim: int,
        seed: int = 0,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        base_name = base_function_name.lower()
        base_map = {
            "sphere": Sphere,
            "rosenbrock": Rosenbrock,
            "rastrigin": Rastrigin,
            "ackley": Ackley,
        }
        if base_name not in base_map:
            raise ValueError(
                f"Unknown base function: {base_function_name}. "
                f"Available: {list(base_map.keys())}"
            )

        if intrinsic_dim > ambient_dim:
            raise ValueError(
                f"intrinsic_dim ({intrinsic_dim}) must be <= ambient_dim ({ambient_dim})"
            )

        self.base_function_name = base_name
        self.ambient_dim = int(ambient_dim)
        self.intrinsic_dim = int(intrinsic_dim)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.rotation = self._sample_rotation()
        self._base_benchmark = base_map[base_name](input_dim=self.intrinsic_dim)

        if bounds is None:
            base_bounds = self._base_benchmark.bounds
            if base_bounds is not None:
                lower, upper = base_bounds
                bounds = (
                    np.full(self.ambient_dim, float(lower[0])),
                    np.full(self.ambient_dim, float(upper[0])),
                )

        super().__init__(
            name=f"rotated_subspace_{base_name}",
            input_dim=self.ambient_dim,
            output_dim=1,
            bounds=bounds,
        )

        self.metadata = {
            "base_function": self.base_function_name,
            "ambient_dim": self.ambient_dim,
            "intrinsic_dim": self.intrinsic_dim,
            "seed": self.seed,
        }

        if __debug__:
            self._assert_flat_directions()

    def _sample_rotation(self) -> np.ndarray:
        matrix = self._rng.normal(size=(self.ambient_dim, self.ambient_dim))
        q, _ = np.linalg.qr(matrix)
        return q

    def _assert_flat_directions(self) -> None:
        rng = np.random.default_rng(self.seed + 1)
        x = rng.normal(size=self.ambient_dim)
        tail = rng.normal(size=self.ambient_dim)
        tail[: self.intrinsic_dim] = 0.0
        flat_direction = self.rotation.T @ tail

        x_shifted = x + flat_direction
        z1 = (self.rotation @ x)[: self.intrinsic_dim]
        z2 = (self.rotation @ x_shifted)[: self.intrinsic_dim]
        np.testing.assert_allclose(z1, z2, rtol=1e-10, atol=1e-10)

        val1 = self._base_benchmark.evaluate(z1)
        val2 = self._base_benchmark.evaluate(z2)
        np.testing.assert_allclose(val1, val2, rtol=1e-10, atol=1e-10)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.ambient_dim:
            raise ValueError(
                f"Expected input dim {self.ambient_dim}, got {x.shape[1]}"
            )

        x_rot = x @ self.rotation.T
        z = x_rot[:, : self.intrinsic_dim]
        return self._base_benchmark.evaluate(z)

    def sample_random(self, n: int) -> np.ndarray:
        if self.bounds is None:
            return self._rng.normal(size=(n, self.ambient_dim))
        lower, upper = self.bounds
        return self._rng.uniform(lower, upper, size=(n, self.ambient_dim))

    def sample_initial_data(self, n: int) -> np.ndarray:
        return self.sample_random(n)

    def is_valid(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        finite = np.all(np.isfinite(x), axis=1)
        if self.bounds is None:
            return finite if finite.size > 1 else bool(finite[0])
        lower, upper = self.bounds
        within = np.all((x >= lower) & (x <= upper), axis=1)
        valid = finite & within
        return valid if valid.size > 1 else bool(valid[0])
