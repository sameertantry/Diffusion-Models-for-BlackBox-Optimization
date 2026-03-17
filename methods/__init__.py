"""Optimization methods module.

This module provides a unified interface for different black-box optimization
algorithms, allowing them to be used interchangeably in experiments.
"""

from methods.base import BaseOptimizer
from methods.cma_es import CMAES
from methods.diffusion import DiffusionOptimizer
from methods.diffusion_bbo import DiffusionBBO
from methods.reinforce import REINFORCE
from methods.tpe import TPE

__all__ = [
    "BaseOptimizer",
    "TPE",
    "CMAES",
    "DiffusionOptimizer",
    "DiffusionBBO",
    "REINFORCE",
]
