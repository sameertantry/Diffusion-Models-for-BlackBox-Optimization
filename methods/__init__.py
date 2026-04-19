"""Optimization methods module.

This module provides a unified interface for different black-box optimization
algorithms, allowing them to be used interchangeably in experiments.
"""

from methods.base import BaseOptimizer
from methods.bipop_cma_es import BIPOPCMAES
from methods.cma_es import CMAES
from methods.diffusion import DiffusionOptimizer
from methods.diffusion_bbo import DiffusionBBO
from methods.diffusion_v2 import DiffusionOptimizerV2
from methods.gp_qei import GPqEI
from methods.reinforce import REINFORCE
from methods.sep_cma_es import SepCMAES
from methods.tpe import TPE
from methods.turbo import TuRBO

__all__ = [
    "BaseOptimizer",
    "BIPOPCMAES",
    "CMAES",
    "DiffusionBBO",
    "DiffusionOptimizer",
    "DiffusionOptimizerV2",
    "GPqEI",
    "REINFORCE",
    "SepCMAES",
    "TPE",
    "TuRBO",
]
