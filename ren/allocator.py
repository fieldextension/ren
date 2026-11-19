from typing import Any
from dataclasses import dataclass

from ren.dtype import DType


@dataclass(frozen=True, eq=True)
class BufferSpec:
    """
    Per-allocation options for buffers.
    """

    dtype: DType
    size: int
    device: str

    host: bool = False
    nolru: bool = False  # bypass LRU cache — free immediately


class Allocator:
    supports_transfer: bool = False

    def alloc(self, spec: BufferSpec):
        assert spec.size > 0, f"alloc size must be positive, getting {spec.size}"
        return self._alloc(spec)

    def free(self, buf: Any, spec: BufferSpec):
        self._free(buf, spec)

    def copyin(self, buf: Any, src: memoryview):
        self._copyin(buf, src)

    def copyout(self, dest: memoryview, buf: Any):
        self._copyout(dest, buf)

    def as_buffer(self, buf: Any, nbytes: int) -> memoryview:
        """
        Return a host-accessible bytes view of `buf`.

        Backends that support zero-copy host views may override this.
        """
        host = memoryview(bytearray(nbytes))
        self.copyout(host, buf)
        return host

    def transfer(self, dest, src, size: int, src_dev, dest_dev) -> None:
        raise NotImplementedError()

    def offset(self, buf: Any, size: int, offset: int) -> Any:
        return self._offset(buf, size, offset)

    def _alloc(self, spec: BufferSpec):
        raise NotImplementedError()

    def _free(self, buf, spec: BufferSpec):
        pass

    def _copyin(self, buf: Any, src: memoryview):
        raise NotImplementedError()

    def _copyout(self, dest: memoryview, buf: Any):
        raise NotImplementedError()

    def _offset(self, buf: Any, size: int, offset: int) -> Any:
        return buf
