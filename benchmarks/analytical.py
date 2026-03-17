"""Analytical black-box optimization benchmarks.

This module contains classic analytical optimization functions commonly used
for benchmarking optimization algorithms.
"""

from typing import Optional, Tuple

import numpy as np

from benchmarks.base import BaseBenchmark


class Sphere(BaseBenchmark):
    """Sphere function: f(x) = sum(x^2).
    
    Global minimum: f(0) = 0
    Domain: typically [-5.12, 5.12]^d
    """
    
    def __init__(self, input_dim: int = 2, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Initialize the Sphere benchmark.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Optional custom bounds. Defaults to [-5.12, 5.12]^d.
        """
        if bounds is None:
            lower = np.full(input_dim, -5.12)
            upper = np.full(input_dim, 5.12)
            bounds = (lower, upper)
        
        super().__init__(
            name="Sphere",
            input_dim=input_dim,
            output_dim=1,
            bounds=bounds,
        )
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Sphere function.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Function values of shape (n_samples, 1) or (1,).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        values = np.sum(x ** 2, axis=1, keepdims=True)
        
        if values.shape[0] == 1:
            return values[0]
        return values
    
    def sample_random(self, n: int) -> np.ndarray:
        """Sample random inputs uniformly from the domain.
        
        Args:
            n: Number of samples to generate.
        
        Returns:
            Array of shape (n, input_dim) with random inputs.
        """
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, size=(n, self.input_dim))

    def sample_initial_data(self, n: int) -> np.ndarray:
        """Sample initial data (same as random for analytical benchmarks)."""
        return self.sample_random(n)
    
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if inputs are within bounds.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Boolean or boolean array indicating validity.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        lower, upper = self.bounds
        valid = np.all((x >= lower) & (x <= upper), axis=1)
        
        return valid[0] if len(valid) == 1 else valid


class Rosenbrock(BaseBenchmark):
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2).
    
    Global minimum: f(1, 1, ..., 1) = 0
    Domain: typically [-5, 10]^d
    """
    
    def __init__(self, input_dim: int = 2, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Initialize the Rosenbrock benchmark.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Optional custom bounds. Defaults to [-5, 10]^d.
        """
        if bounds is None:
            lower = np.full(input_dim, -5.0)
            upper = np.full(input_dim, 10.0)
            bounds = (lower, upper)
        
        super().__init__(
            name="Rosenbrock",
            input_dim=input_dim,
            output_dim=1,
            bounds=bounds,
        )
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Rosenbrock function.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Function values of shape (n_samples, 1) or (1,).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Rosenbrock: sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
        x_curr = x[:, :-1]
        x_next = x[:, 1:]
        values = np.sum(
            100 * (x_next - x_curr ** 2) ** 2 + (1 - x_curr) ** 2,
            axis=1,
            keepdims=True,
        )
        
        if values.shape[0] == 1:
            return values[0]
        return values
    
    def sample_random(self, n: int) -> np.ndarray:
        """Sample random inputs uniformly from the domain.
        
        Args:
            n: Number of samples to generate.
        
        Returns:
            Array of shape (n, input_dim) with random inputs.
        """
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, size=(n, self.input_dim))

    def sample_initial_data(self, n: int) -> np.ndarray:
        """Sample initial data (same as random for analytical benchmarks)."""
        return self.sample_random(n)
    
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if inputs are within bounds.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Boolean or boolean array indicating validity.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        lower, upper = self.bounds
        valid = np.all((x >= lower) & (x <= upper), axis=1)
        
        return valid[0] if len(valid) == 1 else valid


class Rastrigin(BaseBenchmark):
    """Rastrigin function: f(x) = 10*d + sum(x^2 - 10*cos(2*pi*x)).
    
    Global minimum: f(0) = 0
    Domain: typically [-5.12, 5.12]^d
    """
    
    def __init__(self, input_dim: int = 2, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Initialize the Rastrigin benchmark.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Optional custom bounds. Defaults to [-5.12, 5.12]^d.
        """
        if bounds is None:
            lower = np.full(input_dim, -5.12)
            upper = np.full(input_dim, 5.12)
            bounds = (lower, upper)
        
        super().__init__(
            name="Rastrigin",
            input_dim=input_dim,
            output_dim=1,
            bounds=bounds,
        )
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Rastrigin function.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Function values of shape (n_samples, 1) or (1,).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Rastrigin: 10*d + sum(x^2 - 10*cos(2*pi*x))
        values = (
            10 * self.input_dim
            + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1, keepdims=True)
        )
        
        if values.shape[0] == 1:
            return values[0]
        return values
    
    def sample_random(self, n: int) -> np.ndarray:
        """Sample random inputs uniformly from the domain.
        
        Args:
            n: Number of samples to generate.
        
        Returns:
            Array of shape (n, input_dim) with random inputs.
        """
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, size=(n, self.input_dim))

    def sample_initial_data(self, n: int) -> np.ndarray:
        """Sample initial data (same as random for analytical benchmarks)."""
        return self.sample_random(n)
    
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if inputs are within bounds.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Boolean or boolean array indicating validity.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        lower, upper = self.bounds
        valid = np.all((x >= lower) & (x <= upper), axis=1)
        
        return valid[0] if len(valid) == 1 else valid


class Ackley(BaseBenchmark):
    """Ackley function: f(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e.
    
    Global minimum: f(0) = 0
    Domain: typically [-32.768, 32.768]^d
    """
    
    def __init__(self, input_dim: int = 2, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Initialize the Ackley benchmark.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Optional custom bounds. Defaults to [-32.768, 32.768]^d.
        """
        if bounds is None:
            lower = np.full(input_dim, -32.768)
            upper = np.full(input_dim, 32.768)
            bounds = (lower, upper)
        
        super().__init__(
            name="Ackley",
            input_dim=input_dim,
            output_dim=1,
            bounds=bounds,
        )
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Ackley function.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Function values of shape (n_samples, 1) or (1,).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Ackley: -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e
        mean_sq = np.mean(x ** 2, axis=1, keepdims=True)
        mean_cos = np.mean(np.cos(2 * np.pi * x), axis=1, keepdims=True)
        values = (
            -20 * np.exp(-0.2 * np.sqrt(mean_sq))
            - np.exp(mean_cos)
            + 20
            + np.e
        )
        
        if values.shape[0] == 1:
            return values[0]
        return values
    
    def sample_random(self, n: int) -> np.ndarray:
        """Sample random inputs uniformly from the domain.
        
        Args:
            n: Number of samples to generate.
        
        Returns:
            Array of shape (n, input_dim) with random inputs.
        """
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, size=(n, self.input_dim))

    def sample_initial_data(self, n: int) -> np.ndarray:
        """Sample initial data (same as random for analytical benchmarks)."""
        return self.sample_random(n)
    
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if inputs are within bounds.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Boolean or boolean array indicating validity.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        lower, upper = self.bounds
        valid = np.all((x >= lower) & (x <= upper), axis=1)
        
        return valid[0] if len(valid) == 1 else valid
