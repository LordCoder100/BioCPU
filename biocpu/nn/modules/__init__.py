"""biocpu.nn.modules — core module classes."""

from .linear import SettleLinear
from .module import Module
from .sequential import Sequential

__all__ = ["Module", "Sequential", "SettleLinear"]
