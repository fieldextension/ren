"""
CUDA queue primitives for multi-stream execution.

Provides compute and copy queues that map to separate GPU engines,
enabling overlap of compute and memory operations.
"""

from dataclasses import dataclass, field

from . import api as cuda

_CU_EVENT_DISABLE_TIMING = 0x02


def _create_event() -> int:
    return cuda.event_create(_CU_EVENT_DISABLE_TIMING)


def _create_stream() -> int:
    return cuda.stream_create()


@dataclass(slots=True)
class Signal:
    """CUDA event wrapper for cross-queue synchronization."""

    _event: int = field(default_factory=_create_event)

    def close(self) -> None:
        if self._event:
            cuda.event_destroy(self._event)
            self._event = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def record(self, stream: int) -> None:
        cuda.event_record(self._event, stream)

    def wait(self, stream: int) -> None:
        cuda.stream_wait_event(stream, self._event)

    def synchronize(self) -> None:
        cuda.event_synchronize(self._event)


@dataclass(slots=True)
class _QueueBase:
    """Shared base for compute and copy queues."""

    stream: int = field(default_factory=_create_stream)

    def close(self) -> None:
        if self.stream:
            cuda.stream_destroy(self.stream)
            self.stream = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def signal(self, sig: Signal):
        sig.record(self.stream)

    def wait(self, sig: Signal):
        sig.wait(self.stream)

    def synchronize(self):
        cuda.stream_synchronize(self.stream)


@dataclass(slots=True)
class ComputeQueue(_QueueBase):
    """Queue for kernel launches (GPU compute engine)."""

    def launch(self, func: int, grid: tuple, block: tuple, args: list, shared_mem: int = 0):
        cuda.launch_kernel(func, grid, block, args, self.stream, shared_mem)


@dataclass(slots=True)
class CopyQueue(_QueueBase):
    """Queue for memory transfers (GPU DMA engine)."""

    def copy_htod(self, dst: int, src: int, nbytes: int):
        cuda.memcpy_htod_async(dst, src, nbytes, self.stream)

    def copy_dtod(self, dst: int, src: int, nbytes: int):
        cuda.memcpy_dtod_async(dst, src, nbytes, self.stream)
