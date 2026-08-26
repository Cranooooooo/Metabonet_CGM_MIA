"""tsgen_metrics: reusable generation-quality metrics for time series.

10 metrics (all lower = better). See core.evaluate and PROVENANCE for which are
validated against published references vs. own/proxy implementations.
"""
from .core import evaluate, PROVENANCE
from . import statistical, neural

__all__ = ["evaluate", "PROVENANCE", "statistical", "neural"]
__version__ = "0.1.0"
