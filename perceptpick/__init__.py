"""perceptpick — BOP-style benchmark for Object Grasping."""
__version__ = "0.1.0"

# Eager import of core types so the load order resolves the latent
# core ↔ utils.io / mesh_helpers / visualization circular imports correctly,
# regardless of which subpackage a user imports first.
from .core import _types as _types  # noqa: E402,F401
