import os
import functools
import contextlib
from pathlib import Path
from typing import ClassVar, overload, Any

# Ren's context helpers are adapted from tinygrad's helpers.py.


@overload
def getenv(key: str) -> int: ...
@overload
def getenv[T](key: str, default: T) -> T: ...
@functools.cache
def getenv(key: str, default: Any = 0):
    return type(default)(os.getenv(key, default))


class Context(contextlib.ContextDecorator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.old_context: dict[str, int] = {}
        for k, v in self.kwargs.items():
            self.old_context[k] = ContextVar._cache[k].value
            ContextVar._cache[k].value = v

    def __exit__(self, *_):
        for k, v in self.old_context.items():
            ContextVar._cache[k].value = v


class ContextVar[T]:
    _cache: ClassVar[dict[str, ContextVar]] = {}
    value: T
    key: str

    def __init__(self, key: str, default_value: T):
        if key in ContextVar._cache:
            raise RuntimeError(f"attempt to recreate ContextVar {key}")
        ContextVar._cache[key] = self
        self.value = getenv(key, default_value)
        self.key = key

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x

    def __le__(self, x):
        return self.value <= x


DEBUG = ContextVar("DEBUG", 0)
REWRITE_STACK_LIMIT = ContextVar("REWRITE_STACK_LIMIT", 250000)
MAX_BUFFER_SIZE = ContextVar("MAX_BUFFER_SIZE", 0)
DEV = ContextVar("DEV", "")
# Profiling flag used by ren.prof to toggle tracing.
PROFILE = ContextVar("PROFILE", 0)
# NVTX markers for nsys/Nsight integration (requires nvtx package).
NVTX = ContextVar("NVTX", 0)
# Emit source line-number info in compiled kernels (NVRTC --generate-line-info).
# 1 = enabled, 0 = disabled (default).
LINEINFO = ContextVar("LINEINFO", 0)
# CUDA graph runner: 1 = enabled (default), 0 = disabled (queue fallback).
# Set GRAPH=0 to debug graph-related issues by forcing queue-based execution.
GRAPH = ContextVar("GRAPH", 1)
# Maximum pointwise dot outputs per CUDA register tile; 0 disables fusion.
DOT_TILE = ContextVar("DOT_TILE", 16)
# Compiler cache: 1 = enabled (default), 0 = disabled.
CCACHE = ContextVar("CCACHE", 1)
# Cache directory for compiled CUDA kernels and other artifacts.
CACHE_DIR = ContextVar("CACHE_DIR", str(Path.home() / ".cache" / "ren"))
