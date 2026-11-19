import ctypes
from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

from ren.allocator import Allocator, BufferSpec

from . import api as cuda
from .queue import Signal

if TYPE_CHECKING:
    from .backend import CUDABackend


class CUDAAllocator(Allocator):
    supports_transfer: bool = True

    # Host-to-device copies use a staging buffer ring adapted from tinygrad's HCQAllocator.
    # Sized to fit a full polynomial (n=2^16, ~30 RNS limbs, u64) in one DMA.
    STAGING_SIZE: int = 16 << 20  # 16 MB per staging buffer
    STAGING_COUNT: int = 16  # 256 MB total pinned host memory per device

    def __init__(self, backend: CUDABackend):
        self.backend = backend
        # Address order keeps replay bindings stable across different release orders.
        self._cache: dict[tuple[int, bool, bool], list[tuple[int, Any]]] = {}
        # Pre-allocate rotating staging buffers for async copyin
        cuda.ctx_set_current(backend.context)
        self._staging_bufs: list[Any] = [cuda.mem_host_alloc(self.STAGING_SIZE) for _ in range(self.STAGING_COUNT)]
        self._staging_events: list[Signal] = [Signal() for _ in range(self.STAGING_COUNT)]
        self._staging_next: int = 0
        # Record initial events so event_synchronize works on first use
        for ev in self._staging_events:
            ev.record(backend.copy_queue.stream)

    @staticmethod
    def _cache_key(spec: BufferSpec) -> tuple[int, bool, bool]:
        return spec.size * spec.dtype.itemsize, spec.host, spec.nolru

    def alloc(self, spec: BufferSpec):
        pool = self._cache.get(self._cache_key(spec))
        if pool:
            return heappop(pool)[1]
        try:
            return super().alloc(spec)
        except RuntimeError, MemoryError:
            self.free_cache()
            return super().alloc(spec)

    def free_cache(self) -> None:
        for (_, is_host, _), bufs in self._cache.items():
            free_fn = cuda.mem_host_free if is_host else cuda.mem_free
            for _, buf in bufs:
                free_fn(buf)
            bufs.clear()

    def free_staging(self) -> None:
        """Free all pre-allocated staging buffers. Called during backend finalize."""
        for buf in self._staging_bufs:
            cuda.mem_host_free(buf)
        for ev in self._staging_events:
            ev.close()
        self._staging_bufs.clear()
        self._staging_events.clear()
        self._staging_next = 0

    def _alloc(self, spec: BufferSpec):
        nbytes = spec.size * spec.dtype.itemsize
        cuda.ctx_set_current(self.backend.context)
        alloc_fn = cuda.mem_host_alloc if spec.host else cuda.mem_alloc
        return alloc_fn(nbytes)

    def _free(self, buf, spec: BufferSpec):
        if buf is None:
            return
        if spec.nolru:
            free_fn = cuda.mem_host_free if spec.host else cuda.mem_free
            free_fn(buf)
            return
        pool = self._cache.setdefault(self._cache_key(spec), [])
        heappush(pool, (int(buf), buf))

    def _copyin(self, buf, src: memoryview):
        nbytes = src.nbytes
        if nbytes == 0:
            return
        cuda.ctx_set_current(self.backend.context)
        # Blanket cross-queue sync (for use outside graph runner).
        self.backend.compute_queue.signal(self.backend._compute_signal)
        self.backend.copy_queue.wait(self.backend._compute_signal)
        self._enqueue_copyin(buf, src)
        self.backend.copy_queue.signal(self.backend._copy_signal)
        self.backend.compute_queue.wait(self.backend._copy_signal)

    def _enqueue_copyin(self, buf, src: memoryview):
        """Enqueue H2D DMA via rotating staging buffers on the copy queue.

        No cross-queue sync — caller handles ordering (either blanket sync
        in ``_copyin`` or event-based deps in the graph runner).
        """
        nbytes = src.nbytes
        if nbytes == 0:
            return
        if not isinstance(src, memoryview):
            src = memoryview(src)
        # For writable memoryviews, extract the base address once and use
        # pointer arithmetic per chunk (fast path, zero-copy).
        # For readonly memoryviews (e.g. from bytes), from_buffer() is
        # unavailable so we use per-chunk bytes() copies bounded by
        # STAGING_SIZE instead of copying the entire buffer upfront.
        readonly = src.readonly
        src_base = 0 if readonly else ctypes.addressof(ctypes.c_char.from_buffer(src))
        for offset in range(0, nbytes, self.STAGING_SIZE):
            chunk = min(self.STAGING_SIZE, nbytes - offset)
            idx = self._staging_next
            self._staging_next = (self._staging_next + 1) % self.STAGING_COUNT
            self._staging_events[idx].synchronize()
            if readonly:
                ctypes.memmove(int(self._staging_bufs[idx]), bytes(src[offset : offset + chunk]), chunk)
            else:
                ctypes.memmove(int(self._staging_bufs[idx]), src_base + offset, chunk)
            self.backend.copy_queue.copy_htod(int(buf) + offset, int(self._staging_bufs[idx]), chunk)
            self.backend.copy_queue.signal(self._staging_events[idx])

    def _copyout(self, dest: memoryview, buf):
        nbytes = dest.nbytes
        cuda.ctx_set_current(self.backend.context)
        # Ensure all compute kernels have finished before copying
        self.backend.compute_queue.synchronize()
        cuda.memcpy_dtoh_async(dest, buf, nbytes, self.backend.copy_queue.stream)
        self.backend.copy_queue.synchronize()

    def _offset(self, buf: Any, size: int, offset: int) -> Any:
        assert cuda.is_deviceptr(buf), "CUDAAllocator._offset expects a CUdeviceptr"
        return cuda.deviceptr_offset(buf, offset)

    def transfer(self, dest, src, size: int, src_dev, dest_dev) -> None:
        # Wait for both devices' compute before copy (HCQ _transfer pattern)
        cuda.ctx_set_current(src_dev.context)
        src_dev.compute_queue.signal(src_dev._compute_signal)
        cuda.ctx_set_current(dest_dev.context)
        dest_dev.compute_queue.signal(dest_dev._compute_signal)

        # D2D copy after both fences
        cuda.ctx_set_current(src_dev.context)
        src_dev.copy_queue.wait(src_dev._compute_signal)
        src_dev.copy_queue.wait(dest_dev._compute_signal)
        src_dev.copy_queue.copy_dtod(dest, src, size)
        src_dev.copy_queue.signal(src_dev._copy_signal)

        # Both devices wait for the copy before reusing its source or destination.
        src_dev.compute_queue.wait(src_dev._copy_signal)
        if src_dev is not dest_dev:
            cuda.ctx_set_current(dest_dev.context)
            dest_dev.compute_queue.wait(src_dev._copy_signal)


__all__ = ["CUDAAllocator"]
