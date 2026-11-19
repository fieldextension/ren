"""CUDA kernel source files.

Provides access to predefined .cu kernel sources and .cuh headers.
Kernels are compiled on-demand by the backend through the renderer.
"""

from functools import cache
from importlib.resources import files


@cache
def _load_headers() -> dict[str, str]:
    """Load all .cuh header files from this package."""
    headers: dict[str, str] = {}
    for item in files(__package__).iterdir():
        if item.name.endswith(".cuh"):
            headers[item.name] = item.read_text(encoding="utf-8")
    return headers


def get_headers() -> dict[str, str]:
    """Get headers dict for NVRTC compilation."""
    return _load_headers()


@cache
def load_kernel(filename: str) -> str:
    """Load a .cu kernel source file by name.

    Args:
        filename: Name of the .cu file (e.g., "ntt.cu")

    Returns:
        Source code as a string
    """
    return files(__package__).joinpath(filename).read_text(encoding="utf-8")


__all__ = ["get_headers", "load_kernel"]
