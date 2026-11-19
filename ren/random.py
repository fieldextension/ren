from collections.abc import Sequence
import hashlib
import hmac

from ren.allocator import BufferSpec
from ren.buffer import Buffer
from ren.device import Device
from ren.dtype import dtypes


_DERIVE_DOMAIN = b"ren/chacha20/derive/v1"


class ChaChaRng:
    """Allocate an independent ChaCha20 stream to each random operation."""

    seed_size = 32

    def __init__(self, device: str, seed: bytes):
        if len(seed) != self.seed_size:
            raise ValueError("Expected a 32-byte seed")
        from ren.ir.node import Node

        self._seed = bytes(seed)
        spec = BufferSpec(dtype=dtypes.uint32, size=8, device=Device.canonicalize(device))
        self._key_node = Node.from_buffer(Buffer(spec).allocate().copyin(memoryview(self._seed)))
        self._next_stream_id = 0

    @property
    def device(self) -> str:
        return self._key_node.device

    def reserve(self) -> int:
        if self._next_stream_id >= 1 << 64:
            raise OverflowError("ChaCha20 stream identifiers exhausted")
        stream_id = self._next_stream_id
        self._next_stream_id += 1
        return stream_id

    def derive(self, path: str | Sequence[str]) -> ChaChaRng:
        parts = (path,) if isinstance(path, str) else tuple(path)
        if not parts:
            raise ValueError("Derivation path must not be empty")

        seed = self._seed
        for part in parts:
            if not isinstance(part, str):
                raise TypeError("Derivation path components must be strings")
            if not part:
                raise ValueError("Derivation path components must not be empty")
            try:
                encoded = part.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError("Derivation path components must be ASCII") from exc
            if len(encoded) >= 1 << 32:
                raise ValueError("Derivation path component is too long")
            seed = hmac.digest(seed, _DERIVE_DOMAIN + len(encoded).to_bytes(4, "little") + encoded, "sha256")

        return ChaChaRng(self.device, seed)


def global_rng():
    return Device.default.default_rng()


def set_global_rng_seed(seed: int):
    Device.default.set_default_rng(rng_from_int_seed(seed))


def rng_from_int_seed(seed: int, device: str | None = None) -> ChaChaRng:
    device = Device.canonicalize(device)
    seed_bytes = hashlib.sha256(str(seed).encode()).digest()
    return ChaChaRng(device, seed_bytes)
