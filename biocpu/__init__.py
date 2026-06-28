"""BioCPU — Biologically-inspired CPU-native neural computation framework."""

__version__ = "0.1.0"

from . import nn
from . import optim
from .context import LayerTrace, PhaseContext
from .parameter import Parameter

__all__ = [
    "__version__",
    "nn",
    "optim",
    "LayerTrace",
    "PhaseContext",
    "Parameter",
]
