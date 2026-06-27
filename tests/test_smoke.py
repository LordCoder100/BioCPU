"""Smoke tests to verify BioCPU package installs and imports correctly."""

import biocpu


def test_version_exists() -> None:
    """Package exposes a version string."""
    assert hasattr(biocpu, "__version__")
    assert isinstance(biocpu.__version__, str)


def test_version_format() -> None:
    """Version follows semver (major.minor.patch)."""
    parts = biocpu.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
