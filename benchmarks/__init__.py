"""Black-box optimization benchmarks.

This package provides a unified interface for various optimization benchmarks,
including analytical functions and real-world design problems.
"""

from benchmarks.analytical import Ackley, Rastrigin, Rosenbrock, Sphere
from benchmarks.base import BaseBenchmark
from benchmarks.embedded_subspace import EmbeddedSubspaceBenchmark
from benchmarks.rotated_subspace import RotatedSubspaceBenchmark

__all__ = [
    "BaseBenchmark",
    # Analytical benchmarks
    "Sphere",
    "Rosenbrock",
    "Rastrigin",
    "Ackley",
    "EmbeddedSubspaceBenchmark",
    "RotatedSubspaceBenchmark",
]

# Optional: Design-Bench benchmarks (require design-bench or downloaded data).
try:
    from benchmarks.design_benchmarks import (  # noqa: F401
        ANT,
        CHEMBL,
        DKITTY,
        SUPERCON,
        TFBIND,
        TFBIND8,
        TFBIND10,
    )
    __all__ += [
        "TFBIND", "TFBIND8", "TFBIND10",
        "SUPERCON", "ANT", "DKITTY", "CHEMBL",
    ]
except (ModuleNotFoundError, ImportError):
    pass
