from typing import Any, Callable
from contextlib import contextmanager
import gc
import weakref


def unwrap[T](x: T | None) -> T:
    """Assert and unwrap an optional value. Narrows T|None to T for type checkers."""
    assert x is not None
    return x


@contextmanager
def disable_gc():
    """
    Context manager and decorator that disables garbage collection.
    Used to avoid GC pauses during performance-critical sections.
    """

    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class _WeakValueCache[V](weakref.WeakValueDictionary[tuple, V]):
    data: dict[tuple, weakref.ReferenceType[V]]
    _remove: Callable[[weakref.ReferenceType[V]], None]


class CacheMeta(type):
    _cache: _WeakValueCache[object]
    __cache_key__: Callable[..., tuple]

    def __init__(cls, name, bases, ns):
        super().__init__(name, bases, ns)
        cls._cache = _WeakValueCache()

        if not hasattr(cls, "__cache_key__"):
            raise TypeError(f"{cls.__name__} must define __cache_key__")

    def __call__(cls, *args: Any, **kwargs: Any):
        key = cls.__cache_key__(*args, **kwargs)
        cache = cls._cache
        # Use undocumented WeakValueDictionary internals to bypass its Python get/set wrappers while preserving cleanup.
        if (ref := cache.data.get(key)) is not None and (inst := ref()) is not None:
            return inst

        # Pass canonical key as args - ensures stored values match cache key
        inst = super().__call__(*key)
        cache.data[key] = weakref.KeyedRef(inst, cache._remove, key)
        return inst

    def _uncached[T](cls: type[T], *args: Any) -> T:
        """Construct without caching; args must be a guaranteed-unique canonical `__cache_key__`."""
        return type.__call__(cls, *args)


def find_contiguous_subset(src: tuple, dst: tuple) -> int | None:
    """Find start index where dst appears as a contiguous subslice in src.

    Returns 0 for identity (src == dst), None if dst is not a contiguous subset.
    """
    n = len(dst)
    if n > len(src):
        return None
    if n == len(src):
        return 0 if src == dst else None
    for i in range(len(src) - n + 1):
        if src[i : i + n] == dst:
            return i
    return None
