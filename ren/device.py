import atexit
import functools
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, NewType, TYPE_CHECKING, TypeGuard, overload

from ren.context import DEV, DEBUG

if TYPE_CHECKING:
    from ren.backend import Backend
    from ren.backends.cuda.backend import CUDABackend


CUDADevice = NewType("CUDADevice", str)


class _DeviceRegistry:
    """
    Global registry mapping device strings (\"python\", \"cuda:0\", ...) to Backend instances.
    Backends are created lazily on first access.
    """

    def __init__(self) -> None:
        self._backends: Dict[str, Backend] = {}
        self._scoped_default: ContextVar[str | None] = ContextVar("ren_scoped_device", default=None)

    @property
    def DEFAULT(self) -> str:
        """Canonical name of the default device."""
        return self._override or self._select_device

    @property
    def _override(self) -> str | None:
        """Explicit device from the current scope or DEV, without detection."""
        scoped = self._scoped_default.get()
        if scoped is not None:
            return scoped
        return self._canonicalize(DEV.value) if DEV.value else None

    @functools.cached_property
    def _select_device(self) -> str:
        try:
            from ren.backends.cuda import api as cuda
        except ModuleNotFoundError as e:
            if e.name not in ("cuda", "cuda.bindings"):
                raise
        else:
            if cuda.is_available():
                return "cuda:0"
        if DEBUG >= 1:
            print("CUDA unavailable; using python")
        return "python"

    @contextmanager
    def scope(self, device: str) -> Generator[None]:
        """Temporarily use ``device`` as the default in the current execution context."""
        with self._scoped_default.set(self._canonicalize(device)):
            yield

    def canonicalize(self, device: str | None) -> str:
        """Canonicalize a device string, using DEFAULT if None."""
        return self._canonicalize(device) if device is not None else self.DEFAULT

    @property
    def default(self) -> Backend:
        """Backend instance for the default device."""
        return self[self.DEFAULT]

    @functools.cache  # this class is a singleton
    def _canonicalize(self, name: str) -> str:
        lower = name.strip().lower()
        if lower in ("cpu", "python"):
            return "python"
        backend, _, index = lower.partition(":")
        if backend == "cuda" and (not index or (index.isascii() and index.isdecimal())):
            return f"cuda:{int(index or '0')}"
        raise ValueError(f"Invalid device {name!r}: expected 'python', 'cpu', or 'cuda[:index]' with a non-negative integer index")

    def is_cuda(self, name: str) -> TypeGuard[CUDADevice]:
        """Whether a device name selects CUDA, without opening the device."""
        try:
            return self._canonicalize(name).startswith("cuda:")
        except ValueError:
            return False

    @overload
    def __getitem__(self, name: CUDADevice) -> CUDABackend: ...

    @overload
    def __getitem__(self, name: str) -> Backend: ...

    def __getitem__(self, name: str) -> Backend:
        key = self._canonicalize(name)
        if key not in self._backends:
            self._backends[key] = self._create_backend(key)
            if DEBUG >= 1:
                print(f"backend created: device={key}")
        return self._backends[key]

    def _create_backend(self, key: str) -> Backend:
        if key == "python":
            from ren.backends.python import PythonBackend

            return PythonBackend()
        try:
            # Lazy import to avoid importing CUDA bindings unless needed.
            from ren.backends.cuda.backend import CUDABackend
        except ModuleNotFoundError as e:
            raise RuntimeError(f"CUDA device {key!r} is not available") from e

        return CUDABackend(key)


Device = _DeviceRegistry()

# Finalize all opened devices before Python starts tearing down modules
atexit.register(lambda: [Device._backends[k].finalize() for k in list(Device._backends.keys())])
