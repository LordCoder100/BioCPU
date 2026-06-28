"""biocpu.nn.functional — low-level operations (import as F)."""


def __getattr__(name):
    """Lazy re-export: avoids circular import with nn.modules."""
    from . import _functions

    return getattr(_functions, name)


def __dir__():
    from . import _functions

    return dir(_functions)
