from typing import Any
import sys

from ren.allocator import BufferSpec
from ren.device import Device
from ren.context import MAX_BUFFER_SIZE
from ren.dtype import DType


# Parts of Ren's buffer implementation are adapted from tinygrad's Buffer.
class Buffer:
    __slots__ = ("spec", "_buf", "_base", "_offset", "__weakref__")

    spec: BufferSpec

    _buf: Any
    _base: Buffer | None
    _offset: int  # byte offset within base

    def __init__(self, spec: BufferSpec, base: Buffer | None = None, offset: int = 0):
        self.spec = spec
        self._buf = None
        if base is not None:
            assert base._base is None, "base can't have a base"
            assert spec.device == base.device, "base must have the same device"
        self._base = base
        self._offset = offset

    @property
    def _allocator(self):
        return Device[self.device].allocator

    @property
    def dtype(self):
        return self.spec.dtype

    @property
    def size(self) -> int:
        return self.spec.size

    @property
    def device(self) -> str:
        return self.spec.device

    @property
    def base(self) -> Buffer:
        return self._base if self._base is not None else self

    @property
    def offset(self) -> int:
        # public offset is expressed in elements
        return self._offset // self.dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    def is_initialized(self) -> bool:
        if self._base is not None:
            return self._base.is_initialized() and self._buf is not None
        return self._buf is not None

    def is_allocated(self) -> bool:
        if self._base is not None:
            return self._base.is_allocated()
        return self._buf is not None

    def ensure_allocated(self) -> Buffer:
        if self._base is not None:
            self._base.ensure_allocated()
            if self._buf is None:
                self._buf = self._allocator.offset(self._base._buf, self.nbytes, self._offset)
            return self
        return self.allocate() if not self.is_initialized() else self

    def allocate(self, opaque=None) -> Buffer:
        assert self._base is None, "can't allocate a view buffer"
        assert self._buf is None, "can't allocate already allocated buffer"
        if MAX_BUFFER_SIZE > 0 and self.size > MAX_BUFFER_SIZE:
            raise RuntimeError(f"buffer of size {self.size / 1e6:.2f}M is too large")

        self._buf = opaque if opaque is not None else self._allocator.alloc(self.spec)
        return self

    def deallocate(self):
        assert self._buf is not None, "buffer must be allocated to deallocate"
        if self._base is None:
            self._allocator.free(self._buf, self.spec)
        del self._buf

    # Adapted from tinygrad's suppress_finalizing.
    def __del__(self):
        """Release storage, suppressing common errors during Python shutdown."""
        try:
            if self._buf is not None:
                self.deallocate()
        except RuntimeError, AttributeError, TypeError, ImportError:
            if not sys.is_finalizing():
                raise

    def copyin(self, mv: memoryview) -> Buffer:
        """
        Copy data from a host memoryview into this buffer.
        """
        assert len(mv) * mv.itemsize == self.nbytes, f"size mismatch, {len(mv)=}, {mv.itemsize=} != {self.nbytes=}"
        if self._base is not None and self.device == "python":
            self._base.ensure_allocated()
            base_mv = self._base.as_buffer()
            offset_bytes = self._offset
            base_mv[offset_bytes : offset_bytes + self.nbytes] = mv
        else:
            self.ensure_allocated()
            self._allocator.copyin(self._buf, mv)
        return self

    def copyout(self, mv: memoryview) -> memoryview:
        """
        Copy data from this buffer into a host memoryview.
        """
        assert len(mv) * mv.itemsize == self.nbytes, f"size mismatch, {len(mv)=}, {mv.itemsize=} != {self.nbytes=}"
        if self._base is not None and self.device == "python":
            self._base.ensure_allocated()
            base_mv = self._base.as_buffer()
            offset_bytes = self._offset
            mv[:] = base_mv[offset_bytes : offset_bytes + self.nbytes]
        else:
            self.ensure_allocated()
            self._allocator.copyout(mv, self._buf)
        return mv

    def as_buffer(self) -> memoryview:
        """
        Return a raw memoryview over the underlying storage.
        """
        # Ensure backing storage is materialized before taking a view.
        self.ensure_allocated()
        assert self._buf is not None, "buffer must be allocated to create a view"
        return self._allocator.as_buffer(self._buf, self.nbytes)

    def as_typed_buffer(self) -> memoryview:
        """
        Return a memoryview cast to the scalar dtype format.
        """
        fmt = self.dtype.fmt
        assert fmt is not None, f"{self.dtype} has no struct format for typed buffer view"
        # Only integer formats are supported through memoryview cast in this path.
        if fmt == "?" or DType.is_float_fmt(fmt):
            raise TypeError(f"as_typed_buffer only supports integer dtypes, got fmt={fmt!r}")
        assert DType.is_int_fmt(fmt)
        mv = self.as_buffer()
        # If the buffer view is already typed, return it directly.
        if mv.format == fmt:
            return mv
        # memoryview.cast only permits casting from bytes; fall back to raw view when already typed.
        if mv.format != "B":
            return mv
        return mv.cast(fmt)

    def view(self, size: int, offset: int) -> Buffer:
        assert offset >= 0 and size >= 0 and offset + size <= self.size, "invalid buffer view range"
        base = self._base or self
        parent_off = self._offset if self._base is not None else 0
        base_offset = parent_off + offset * self.dtype.itemsize
        spec = BufferSpec(dtype=self.dtype, size=size, device=self.device)
        return Buffer(spec, base=base, offset=base_offset)
