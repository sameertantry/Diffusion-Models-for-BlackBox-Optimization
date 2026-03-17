"""Base abstract class for all optimization benchmarks."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseBenchmark(ABC):
    """Abstract base class for black-box optimization benchmarks.
    
    All benchmarks must implement this interface to ensure compatibility
    across different optimization algorithms and experiments.
    
    Attributes:
        name: Human-readable name of the benchmark.
        input_dim: Dimensionality of the input space.
        output_dim: Dimensionality of the output space (usually 1 for single-objective).
        bounds: Optional tuple of (lower_bounds, upper_bounds) as numpy arrays.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int = 1,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Initialize the benchmark.
        
        Args:
            name: Name of the benchmark.
            input_dim: Dimensionality of the input space.
            output_dim: Dimensionality of the output space (default: 1).
            bounds: Optional tuple of (lower_bounds, upper_bounds) arrays.
        """
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bounds = bounds
    
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the black-box function at given input(s).
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Output array of shape (n_samples, output_dim) or (output_dim,).
            For single-objective problems, output_dim is typically 1.
        """
        pass
    
    @abstractmethod
    def sample_random(self, n: int) -> np.ndarray:
        """Sample n random valid inputs from the domain.
        
        Args:
            n: Number of samples to generate.
        
        Returns:
            Array of shape (n, input_dim) with random valid inputs.
        """
        pass

    @abstractmethod
    def sample_initial_data(self, n: int) -> np.ndarray:
        """Sample n initial inputs for offline benchmarks.

        Args:
            n: Number of samples to generate.

        Returns:
            Array of shape (n, input_dim) with valid inputs.
        """
        pass
    
    @abstractmethod
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if input(s) satisfy domain constraints.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Boolean or boolean array indicating validity of each input.
        """
        pass
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for evaluate, allowing benchmark(x) syntax.
        
        Args:
            x: Input array of shape (n_samples, input_dim) or (input_dim,).
        
        Returns:
            Output array of shape (n_samples, output_dim) or (output_dim,).
        """
        return self.evaluate(x)
