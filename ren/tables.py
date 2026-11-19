from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import prod
from functools import cache, cached_property

import bisect
import ctypes
import itertools
from typing import ClassVar
import mpmath
import numpy as np

from ren.buffer import Buffer, BufferSpec
from ren.dtype import dtypes, DType, to_np_dtype
from ren.ir.node import from_rns_limb_count
import ren.montgomery as mont
from ren.prof import trace
from ren.ring import RingSpec
from ren.utils import find_contiguous_subset


def _readonly_array(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


@cache
def _get_bitrev_indices(n: int) -> np.ndarray:
    """Get bit-reversal permutation indices for size n (cached)."""
    bits = n.bit_length() - 1
    indices = np.arange(n, dtype=np.uint32)
    rev = np.zeros(n, dtype=np.uint32)
    for b in range(bits):
        rev |= ((indices >> b) & 1) << (bits - 1 - b)
    return _readonly_array(rev)


def _to_device(device: str, data: np.ndarray, dtype: DType) -> Buffer:
    """Copy numpy array to a device buffer (zero-copy memoryview)."""
    size = len(data)
    if size == 0:
        spec = BufferSpec(dtype=dtype, size=1, device=device)
        return Buffer(spec).allocate()
    spec = BufferSpec(dtype=dtype, size=size, device=device)
    buf = Buffer(spec).allocate()
    buf.copyin(memoryview(data).cast("B"))
    return buf


def _bytes_to_device(device: str, data: bytearray, dtype: DType) -> Buffer:
    size = len(data) // dtype.itemsize
    if size == 0:
        spec = BufferSpec(dtype=dtype, size=1, device=device)
        return Buffer(spec).allocate()
    spec = BufferSpec(dtype=dtype, size=size, device=device)
    buf = Buffer(spec).allocate()
    buf.copyin(memoryview(data).cast("B"))
    return buf


class NTTParams:
    """NTT parameters for a single (n, p) where p is a prime modulus. Lazy computation with shared intermediate."""

    __slots__ = ("n", "p", "root", "n_inv", "_psis_raw", "_psis_bitrev", "_psis_inv_bitrev", "_omega_powers", "_rotation_factors")

    def __init__(self, n: int, p: int, root: int, n_inv: int):
        self.n = n
        self.p = p
        self.root = root
        self.n_inv = n_inv
        self._psis_raw: np.ndarray | None = None
        self._psis_bitrev: np.ndarray | None = None
        self._psis_inv_bitrev: np.ndarray | None = None
        self._omega_powers: np.ndarray | None = None
        self._rotation_factors: dict[int, np.ndarray] = {}

    def _get_psis_raw(self) -> np.ndarray:
        if self._psis_raw is None:
            psis = np.empty(self.n, dtype=np.uint64)
            psi_power = 1
            for i in range(self.n):
                psis[i] = psi_power
                psi_power = (psi_power * self.root) % self.p
            self._psis_raw = psis
        return self._psis_raw

    @property
    def psis_bitrev(self) -> np.ndarray:
        if self._psis_bitrev is None:
            self._psis_bitrev = self._get_psis_raw()[_get_bitrev_indices(self.n)]
        return self._psis_bitrev

    @property
    def psis_inv_bitrev(self) -> np.ndarray:
        if self._psis_inv_bitrev is None:
            psis_raw = self._get_psis_raw()
            psis_inv = np.empty(self.n, dtype=np.uint64)
            psis_inv[0] = 1
            psis_inv[1:] = (self.p - psis_raw[-1:0:-1]) % self.p
            self._psis_inv_bitrev = psis_inv[_get_bitrev_indices(self.n)]
        return self._psis_inv_bitrev

    @property
    def omega_powers(self) -> np.ndarray:
        if self._omega_powers is None:
            psis_raw = self._get_psis_raw()
            half = self.n // 2
            omega = np.empty(self.n, dtype=np.uint64)
            omega[:half] = psis_raw[::2]
            omega[half:] = (self.p - psis_raw[::2]) % self.p
            self._omega_powers = omega
        return self._omega_powers

    def get_rotation_factors(self, k: int) -> np.ndarray:
        """Get complete rotation factors for NTT-domain rotation by k."""
        k = k % (2 * self.n)
        if k not in self._rotation_factors:
            k_norm = k % self.n
            negate = k >= self.n

            bitrev = _get_bitrev_indices(self.n)
            indices = (bitrev.astype(np.uint64) * k_norm) % self.n
            omega_ik = self.omega_powers[indices]
            psi_k = self._get_psis_raw()[k_norm]

            result = (omega_ik.astype(object) * int(psi_k)) % self.p
            if negate:
                result = (self.p - result) % self.p
            self._rotation_factors[k] = np.array(result, dtype=np.uint64)
        return self._rotation_factors[k]


@dataclass(slots=True, frozen=True)
class BaseChange:
    """Precomputed tables for RNS base conversion."""

    inv_src: tuple[int, ...]
    mi_mod: tuple[tuple[int, ...], ...]
    prod_src_mod_dst: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class _MontgomeryChangeHostTables:
    inv_src: np.ndarray
    mi_r2_scaled: np.ndarray
    prod_r2_scaled: np.ndarray


def _build_base_change(src: tuple[int, ...], dst: tuple[int, ...]) -> BaseChange:
    prod_src = 1
    for p in src:
        prod_src *= p
    inv_src: list[int] = []
    mi_mod_rows: list[tuple[int, ...]] = []
    prod_mod_dst: list[int] = []
    for p in src:
        Mi = prod_src // p
        inv_src.append(pow(Mi % p, -1, p))
    for q in dst:
        mi_mod_rows.append(tuple((prod_src // p) % q for p in src))
        prod_mod_dst.append(prod_src % q)

    return BaseChange(
        inv_src=tuple(inv_src),
        mi_mod=tuple(mi_mod_rows),
        prod_src_mod_dst=tuple(prod_mod_dst),
    )


@cache
def _ntt_params(n: int, q: int) -> NTTParams:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"NTT parameters require n to be a power of two, got n={n}")
    if q <= 1 or (q - 1) % (2 * n) != 0:
        raise ValueError(f"NTT parameters require q > 1 and (q-1) divisible by 2n, got n={n}, q={q}")

    n_inv = pow(n, -1, q)
    group_order = q - 1
    exponent = group_order // (2 * n)
    psi = 0
    for t in range(2, q):
        candidate = pow(t, exponent, q)
        if pow(candidate, n, q) != 1:
            psi = candidate
            break
    return NTTParams(n=n, p=q, root=psi, n_inv=n_inv)


@cache
def _get_base_change(src: tuple[int, ...], dst: tuple[int, ...]) -> BaseChange:
    return _build_base_change(src, dst)


def _r2_scaled_values(values: tuple[int, ...], moduli: tuple[int, ...], *, dtype: DType) -> np.ndarray:
    ctxs = tuple(mont.context(dtype, modulus) for modulus in moduli)
    return _readonly_array(
        np.array(
            [value * ctx.r2 % ctx.modulus for value, ctx in zip(values, ctxs, strict=True)],
            dtype=to_np_dtype(dtype),
        )
    )


def _r2_scaled_rows(rows: tuple[tuple[int, ...], ...], moduli: tuple[int, ...], *, dtype: DType) -> np.ndarray:
    ctxs = tuple(mont.context(dtype, modulus) for modulus in moduli)
    scaled = [value * ctx.r2 % ctx.modulus for row, ctx in zip(rows, ctxs, strict=True) for value in row]
    return _readonly_array(np.array(scaled, dtype=to_np_dtype(dtype)))


@cache
def _host_change_tables(src: tuple[int, ...], dst: tuple[int, ...], dtype: DType) -> _MontgomeryChangeHostTables:
    bc = _get_base_change(src, dst)
    return _MontgomeryChangeHostTables(
        inv_src=_readonly_array(np.array(bc.inv_src, dtype=to_np_dtype(dtype))),
        mi_r2_scaled=_r2_scaled_rows(bc.mi_mod, dst, dtype=dtype),
        prod_r2_scaled=_r2_scaled_values(bc.prod_src_mod_dst, dst, dtype=dtype),
    )


def _pack_u32_bignums(values: tuple[int, ...], limb_count: int) -> np.ndarray:
    """Pack ints as [len(values), limb_count] little-endian u32 bignums."""
    out = np.zeros((len(values), limb_count), dtype=np.uint32)
    for i, v in enumerate(values):
        for w in range(limb_count):
            out[i, w] = (v >> (32 * w)) & 0xFFFFFFFF
    return _readonly_array(out)


@cache
def crt_precompute(moduli: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Precompute partial products and modular inverses for CRT.

    Returns (partial_prods, inverses) where:
    - partial_prods[i] = product of moduli[0:i]
    - inverses[i] = partial_prods[i]^(-1) mod moduli[i]
    """
    k = len(moduli)
    partial_prods = [1] * k
    inverses = [1] * k

    prod = 1
    for i in range(1, k):
        prod *= moduli[i - 1]
        partial_prods[i] = prod
        inverses[i] = pow(prod % moduli[i], -1, moduli[i])

    return tuple(partial_prods), tuple(inverses)


@cache
def _host_inv_prod_mont(src: tuple[int, ...], dst: tuple[int, ...], dtype: DType) -> np.ndarray:
    vals: list[int] = []
    for q in dst:
        acc = 1
        for p in src:
            acc = (acc * pow(p, -1, q)) % q
        vals.append(mont.context(dtype, q).encode(acc))
    return _readonly_array(np.array(vals, dtype=to_np_dtype(dtype)))


def _require_compatible_ring_tables(src: RingSpec, dst: RingSpec) -> None:
    if src.n != dst.n:
        raise ValueError(f"ring degree mismatch: {src.n} != {dst.n}")
    if src.dtype is not dst.dtype:
        raise ValueError(f"ring dtype mismatch: {src.dtype} != {dst.dtype}")


class Tables:
    """
    Per-device table cache (singleton per device).

    Usage:
        tables = Tables("cuda:0")
        ring_tables = tables.ring(my_ring)
        primes_buf = ring_tables.primes
    """

    __slots__ = ("_device", "_rings", "_base_changes", "_rescales", "_dists")
    _instances: ClassVar[dict[str, Tables]] = {}

    _device: str
    _rings: dict[tuple, RingTables]
    _base_changes: dict[tuple, BaseChangeTables]
    _rescales: dict[tuple, RescaleTables]
    _dists: DistTables

    def __new__(cls, device: str) -> Tables:
        if device not in cls._instances:
            inst = super().__new__(cls)
            inst._device = device
            inst._rings = {}
            inst._base_changes = {}
            inst._rescales = {}
            inst._dists = DistTables(device)
            cls._instances[device] = inst
        return cls._instances[device]

    @property
    def device(self) -> str:
        return self._device

    def ring(self, ring: RingSpec) -> RingTables:
        key = ring.key
        if key not in self._rings:
            match = self._find_contiguous_parent(ring)
            if match is not None:
                parent, prime_offset = match
                self._rings[key] = RingTables(self.device, ring, parent=parent, prime_offset=prime_offset)
            else:
                self._rings[key] = RingTables(self.device, ring)
        return self._rings[key]

    def _find_contiguous_parent(self, ring: RingSpec) -> tuple[RingTables, int] | None:
        for rt in self._rings.values():
            if rt._ring.n != ring.n or rt._ring.dtype is not ring.dtype:
                continue
            start = find_contiguous_subset(rt._ring.moduli, ring.moduli)
            if start is not None:
                return rt, start
        return None

    def base_change(self, src: RingSpec, dst: RingSpec) -> BaseChangeTables:
        _require_compatible_ring_tables(src, dst)
        key = (src.key, dst.key)
        if key not in self._base_changes:
            self._base_changes[key] = BaseChangeTables(self.device, src, dst)
        return self._base_changes[key]

    def rescale(self, src: RingSpec, q_ring: RingSpec) -> RescaleTables:
        _require_compatible_ring_tables(src, q_ring)
        key = (src.key, q_ring.key)
        if key not in self._rescales:
            self._rescales[key] = RescaleTables(self.device, src, q_ring)
        return self._rescales[key]

    @property
    def dists(self) -> DistTables:
        return self._dists

    @classmethod
    def clear(cls, device: str | None = None) -> None:
        if device is None:
            cls._instances.clear()
        elif device in cls._instances:
            del cls._instances[device]


class RingTables:
    """
    Precomputed tables for a specific ring on a specific device.
    Uses @cached_property for lazy computation.
    When a parent RingTables exists, properties return zero-copy views
    into the parent's buffers instead of allocating separate device memory.
    """

    __slots__ = ("_device", "_ring", "_ntt_params", "_parent", "_prime_offset", "_rot_cache", "__dict__")

    def __init__(self, device: str, ring: RingSpec, parent: RingTables | None = None, prime_offset: int = 0):
        self._device = device
        self._ring = ring
        self._ntt_params: tuple[NTTParams, ...] = () if parent is not None else tuple(_ntt_params(ring.n, p) for p in ring.moduli)
        self._parent = parent
        self._prime_offset = prime_offset
        self._rot_cache: dict[int, Buffer] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def n(self) -> int:
        return self._ring.n

    @property
    def num_primes(self) -> int:
        return len(self._ring.moduli)

    @property
    def dtype(self) -> DType:
        return self._ring.dtype

    def _view(self, parent_buf: Buffer, size: int, offset: int) -> Buffer:
        return parent_buf.view(size, offset).ensure_allocated()

    @cached_property
    def primes(self) -> Buffer:
        if self._parent is not None:
            return self._view(self._parent.primes, self.num_primes, self._prime_offset)
        with trace("tables.primes", device=self.device):
            primes = np.array(self._ring.moduli, dtype=to_np_dtype(self.dtype))
            return _to_device(self.device, primes, self.dtype)

    @cached_property
    def mont_radix32(self) -> Buffer:
        # Base 2^32 in Montgomery form, for signed integer limb conversion.
        if self._parent is not None:
            return self._view(self._parent.mont_radix32, self.num_primes, self._prime_offset)
        vals = np.array([mont.context(self.dtype, q).encode(1 << 32) for q in self._ring.moduli], dtype=to_np_dtype(self.dtype))
        return _to_device(self.device, vals, self.dtype)

    @cached_property
    def ntt_psis(self) -> Buffer:
        if self._parent is not None:
            return self._view(self._parent.ntt_psis, self.n * self.num_primes, self.n * self._prime_offset)
        flat = np.concatenate([mont.context(self.dtype, params.p).encode(params.psis_bitrev) for params in self._ntt_params])
        return _to_device(self.device, flat, self.dtype)

    @cached_property
    def ntt_psis_inv(self) -> Buffer:
        if self._parent is not None:
            return self._view(self._parent.ntt_psis_inv, self.n * self.num_primes, self.n * self._prime_offset)
        flat = np.concatenate([mont.context(self.dtype, params.p).encode(params.psis_inv_bitrev) for params in self._ntt_params])
        return _to_device(self.device, flat, self.dtype)

    @cached_property
    def n_inv(self) -> Buffer:
        if self._parent is not None:
            return self._view(self._parent.n_inv, self.num_primes, self._prime_offset)
        n_invs = np.array(
            [mont.context(self.dtype, params.p).encode(params.n_inv) for params in self._ntt_params],
            dtype=to_np_dtype(self.dtype),
        )
        return _to_device(self.device, n_invs, self.dtype)

    @cached_property
    def mont_p_inv(self) -> Buffer:
        if self._parent is not None:
            return self._view(self._parent.mont_p_inv, self.num_primes, self._prime_offset)
        vals = np.array([mont.context(self.dtype, q).p_inv for q in self._ring.moduli], dtype=to_np_dtype(self.dtype))
        return _to_device(self.device, vals, self.dtype)

    @cached_property
    def mont_r2(self) -> Buffer:
        # R^2 mod p, for using Montgomery multiplication to convert to Montgomery form
        # It may be faster to do the conversion using Shoup multiplication, but that
        # requires storing twice as much data.
        if self._parent is not None:
            return self._view(self._parent.mont_r2, self.num_primes, self._prime_offset)
        vals = np.array([mont.context(self.dtype, q).r2 for q in self._ring.moduli], dtype=to_np_dtype(self.dtype))
        return _to_device(self.device, vals, self.dtype)

    def _from_rns_limb_count(self) -> int:
        return from_rns_limb_count(self._ring)

    @property
    def crt_precompute(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """CRT partial products and modular inverses for this ring's moduli."""
        return crt_precompute(self._ring.moduli)

    @cached_property
    def crt_inverses(self) -> Buffer:
        # partial_prods[i]^{-1} mod moduli[i], normal form, one scalar per prime.
        _, inverses = self.crt_precompute
        vals = np.array(inverses, dtype=to_np_dtype(self.dtype))
        return _to_device(self.device, vals, self.dtype)

    @cached_property
    def crt_partial_prods(self) -> Buffer:
        # partial_prods[i] = prod(moduli[:i]) as [k, limb_count] little-endian u32.
        partial_prods, _ = self.crt_precompute
        out = _pack_u32_bignums(partial_prods, self._from_rns_limb_count())
        return _to_device(self.device, out.reshape(-1), dtypes.uint32)

    @cached_property
    def crt_q(self) -> Buffer:
        out = _pack_u32_bignums((self._ring.q,), self._from_rns_limb_count())
        return _to_device(self.device, out.reshape(-1), dtypes.uint32)

    @cached_property
    def crt_q_half(self) -> Buffer:
        out = _pack_u32_bignums((self._ring.q // 2,), self._from_rns_limb_count())
        return _to_device(self.device, out.reshape(-1), dtypes.uint32)

    def get_rotation_factors(self, k: int) -> Buffer:
        k = k % (2 * self.n)
        if k not in self._rot_cache:
            if self._parent is not None:
                self._rot_cache[k] = self._view(self._parent.get_rotation_factors(k), self.n * self.num_primes, self.n * self._prime_offset)
            else:
                flat = np.concatenate([mont.context(self.dtype, params.p).encode(params.get_rotation_factors(k)) for params in self._ntt_params])
                self._rot_cache[k] = _to_device(self.device, flat, self.dtype)
        return self._rot_cache[k]


class BaseChangeTables:
    """Precomputed tables for RNS base conversion."""

    __slots__ = ("_device", "_src", "_dst", "_dst_src_indices_host", "__dict__")

    def __init__(self, device: str, src: RingSpec, dst: RingSpec):
        self._device = device
        self._src = src
        self._dst = dst
        src_indices = {modulus: idx for idx, modulus in enumerate(src.moduli)}
        missing = np.iinfo(np.uint32).max
        self._dst_src_indices_host = _readonly_array(np.array([src_indices.get(modulus, missing) for modulus in dst.moduli], dtype=np.uint32))

    @property
    def device(self) -> str:
        return self._device

    @cached_property
    def dst_src_indices(self) -> Buffer:
        return _to_device(self.device, self._dst_src_indices_host, dtypes.uint32)

    @cached_property
    def inv_src(self) -> Buffer:
        host = _host_change_tables(self._src.moduli, self._dst.moduli, self._dst.dtype)
        return _to_device(self.device, host.inv_src, self._dst.dtype)

    @cached_property
    def mi_r2_scaled(self) -> Buffer:
        host = _host_change_tables(self._src.moduli, self._dst.moduli, self._dst.dtype)
        return _to_device(self.device, host.mi_r2_scaled, self._dst.dtype)

    @cached_property
    def prod_r2_scaled(self) -> Buffer:
        host = _host_change_tables(self._src.moduli, self._dst.moduli, self._dst.dtype)
        return _to_device(self.device, host.prod_r2_scaled, self._dst.dtype)


@cache
def _host_encoded_scalar(value: int, moduli: tuple[int, ...], dtype: DType) -> np.ndarray:
    return _readonly_array(np.array([mont.context(dtype, modulus).encode(value % modulus) for modulus in moduli], dtype=to_np_dtype(dtype)))


class RescaleTables:
    """Precomputed tables for rescale."""

    __slots__ = (
        "_device",
        "_src",
        "_dst",
        "_removed_ring",
        "_added_prod",
        "_removed_src_indices_host",
        "_dst_src_indices_host",
        "__dict__",
    )

    def __init__(self, device: str, src: RingSpec, q_ring: RingSpec):
        self._device = device
        self._src = src
        self._dst = q_ring

        dst_moduli = set(q_ring.moduli)
        src_indices = {modulus: idx for idx, modulus in enumerate(src.moduli)}
        missing = np.iinfo(np.uint32).max
        removed_pairs = [(idx, modulus) for idx, modulus in enumerate(src.moduli) if modulus not in dst_moduli]
        removed_moduli = tuple(modulus for _, modulus in removed_pairs)
        self._removed_ring = RingSpec(n=src.n, moduli=removed_moduli, dtype=src.dtype)
        self._removed_src_indices_host = _readonly_array(np.array([idx for idx, _ in removed_pairs], dtype=np.uint32))
        self._dst_src_indices_host = _readonly_array(np.array([src_indices.get(modulus, missing) for modulus in q_ring.moduli], dtype=np.uint32))
        src_moduli = set(src.moduli)
        added_prod = 1
        for modulus in q_ring.moduli:
            if modulus not in src_moduli:
                added_prod *= modulus
        self._added_prod = added_prod

    @property
    def device(self) -> str:
        return self._device

    @property
    def added_prod(self) -> int:
        """Product of destination primes that were not in the source basis. 1 when dst is a subset of src."""
        return self._added_prod

    @cached_property
    def removed_src_indices(self) -> Buffer:
        return _to_device(self.device, self._removed_src_indices_host, dtypes.uint32)

    @cached_property
    def dst_src_indices(self) -> Buffer:
        return _to_device(self.device, self._dst_src_indices_host, dtypes.uint32)

    @cached_property
    def added_prod_removed_mont(self) -> Buffer:
        host = _host_encoded_scalar(self._added_prod, self._removed_ring.moduli, self._src.dtype)
        return _to_device(self.device, host, self._src.dtype)

    @cached_property
    def added_prod_dst_mont(self) -> Buffer:
        host = _host_encoded_scalar(self._added_prod, self._dst.moduli, self._dst.dtype)
        return _to_device(self.device, host, self._dst.dtype)

    @cached_property
    def inv_src(self) -> Buffer:
        return Tables(self.device).base_change(self._removed_ring, self._dst).inv_src

    @property
    def mi_r2_scaled(self) -> Buffer:
        return Tables(self.device).base_change(self._removed_ring, self._dst).mi_r2_scaled

    @property
    def prod_r2_scaled(self) -> Buffer:
        return Tables(self.device).base_change(self._removed_ring, self._dst).prod_r2_scaled

    @cached_property
    def inv_prod_mont(self) -> Buffer:
        host = _host_inv_prod_mont(self._removed_ring.moduli, self._dst.moduli, self._dst.dtype)
        return _to_device(self.device, host, self._dst.dtype)


def _tree_node_type(dtype: DType) -> type[ctypes.Structure]:
    match dtype:
        case dtypes.int8:
            ctype = ctypes.c_int8
        case dtypes.int16:
            ctype = ctypes.c_int16
        case dtypes.int32:
            ctype = ctypes.c_int32
        case _:
            raise ValueError(f"unsupported dtype {dtype}")

    class TreeNode(ctypes.Structure):
        _fields_ = [("split_point", ctypes.c_uint32), ("children", ctype * 3)]

    return TreeNode


class TreeNodeEncoder:
    def __init__(self, sample_dtype: DType):
        self.struct = _tree_node_type(sample_dtype)
        self.node_size = ctypes.sizeof(self.struct)
        self.sample_dtype = sample_dtype

    def pack_node(self, split_point: int, children: tuple[int, int, int]):
        if __debug__:
            cutoff = 1 << (8 * self.sample_dtype.itemsize - 1)
            for child in children:
                assert child < cutoff and ~child < cutoff
        return bytearray(self.struct(split_point=split_point, children=children))

    def unpack_node(self, bytes: memoryview, index: int) -> ctypes.Structure:
        offset = index * self.node_size
        return self.struct.from_buffer_copy(bytes[offset : offset + self.node_size])


@dataclass(frozen=True, slots=True)
class DiscreteGaussianPlan:
    base_sampler_table: Buffer
    sample_combination_table: Buffer | None
    base_sample_dtype: DType


class DistTables:
    """Precomputed ternary tree for random number generation using Knuth-Yao sampling.

    Each node of the tree contains a 32-bit value.  The generator will
    consume 32 uniform random bits and check whether these bits are less than,
    equal to, or greater than this value.  It will then either produce an
    output or move on to another node.  If the two values were equal,
    then the generator will use 32 new bits of randomness in the next
    comparison.  If the two values were not equal, it will use the same 32 bits
    of randomness again.

    It is assumed that the distribution produces values in some interval [0,n].
    """

    SAMPLING_SECURITY = 140
    # add some extra precision to account for rounding errors and the fact
    # that we may be combining multiple samples
    _SAMPLING_SECURITY_EXTRA_PRECISION = SAMPLING_SECURITY + 7
    MAX_DEPTH = (_SAMPLING_SECURITY_EXTRA_PRECISION + 31) // 32
    MPMATH_PRECISION = MAX_DEPTH * 32
    _mp = mpmath.mp.clone()
    _mp.prec = MPMATH_PRECISION
    # Discrete gaussian large with large standard deviation is implemented
    # using the algorithm described in https://eprint.iacr.org/2017/259.pdf.
    # Note that we write formulas in terms of standard deviations, while
    # the most instances of "s" in the paper are sqrt(2*pi) times the standard
    # deviation.  The reciprocal of "\eta" is also sqrt(2*pi) times a certain
    # standard deviation.
    _DISCRETE_GAUSSIAN_MAX_TABLE_SIZE = 1 << 13
    # By Lemma 2.2, the probability of choosing a value at least x stdevs from
    # 0 is at most 2 e^(-x^2 / 2) * (1 + epsilon) / (1 - epsilon),
    # where epsilon is very small.  (Here, we have accounted for the factor of
    # sqrt(2 * pi) mentioned above.)
    # We cut off when this quantity becomes smaller than
    # 2^(-_SAMPLING_SECURITY_EXTRA_PRECISION).
    _DISCRETE_GAUSSIAN_CUTOFF_NUM_STDEVS = _mp.sqrt(_mp.mpf(_SAMPLING_SECURITY_EXTRA_PRECISION + 1) * 2 * _mp.log(2))
    # _DISCRETE_GAUSSIAN_SMALL_STDEV_INV is chosen so that a discrete gaussian
    # with standard deviation 1 / _DISCRETE_GAUSSIAN_SMALL_STDEV_INV has
    # probability at least 1 / ( 1 + epsilon ) of being 1,
    # where epsilon = 2**(-_SAMPLING_SECURITY_EXTRA_PRECISION).
    # Equivalently, it is at least sqrt(2 * pi) times the quantity denoted
    # \eta_{\epsilon}(Z) in the paper.  See the bottom of p.5.
    _DISCRETE_GAUSSIAN_SMALL_STDEV_INV = _mp.sqrt(2 * _mp.log(2 + (1 << (_SAMPLING_SECURITY_EXTRA_PRECISION + 1))))
    # A factor used to estimate the value z_i appearing in equations (4)
    # and (5) in the paper in terms of s_i.  From these equations, we find
    # that z_i^2 is approximately s_i / (2 * eta).
    # We substitute stdev * sqrt(2 * pi) for s_i and
    # _DISCRETE_GAUSSIAN_SMALL_STDEV_INV / sqrt(2 * pi) for eta.
    # We find that z_i^2 is about stdev * pi / (_DISCRETE_GAUSSIAN_SMALL_STDEV_INV).
    _MULTI_DISCRETE_GAUSSIAN_Z_ESTIMATE_FACTOR = _mp.pi / (_DISCRETE_GAUSSIAN_SMALL_STDEV_INV)
    # The factor appearing in equation (4) in the paper.
    # We need to choose z_i at most
    # s_{i-1} / (sqrt(2) * eta)
    # = stdev_{i-1} * pi * sqrt(2) / _DISCRETE_GAUSSIAN_SMALL_STDEV_INV
    # = stdev_{i-1} * sqrt(2) * _MULTI_DISCRETE_GAUSSIAN_ESTIMATE_FACTOR
    _MULTI_DISCRETE_GAUSSIAN_Z_BOUND_FACTOR = _MULTI_DISCRETE_GAUSSIAN_Z_ESTIMATE_FACTOR * _mp.sqrt(2)
    del _DISCRETE_GAUSSIAN_SMALL_STDEV_INV

    __slots__ = ("_device", "__dict__")

    @classmethod
    def _large_discrete_gaussian_z(cls, stdev: mpmath.MpfType) -> tuple[int, mpmath.MpfType]:
        # We implement the algorithm SampleI from the paper.  We work backward
        # to find the value of s_0 that will make s_i = stdev * sqrt(2 * pi).
        # Find z, new_stdev so that a discrete gaussian with standard deviation
        # stdev can be approximated by z times a discrete gaussian with
        # standard deviation new_stdev plus z - 1 times another discrete
        # gaussian with standard deviation new_stdev.
        # See Lemma 5.1.
        z_estimate = int(cls._mp.sqrt(stdev * cls._MULTI_DISCRETE_GAUSSIAN_Z_ESTIMATE_FACTOR))
        # z_estimate is guaranteed to work; z_estimate + 1 may or may not work
        for z in range(z_estimate + 1, max(z_estimate, 1) - 1, -1):
            new_stdev = stdev / cls._mp.sqrt(z * z + (z - 1) * (z - 1))
            if z <= cls._MULTI_DISCRETE_GAUSSIAN_Z_BOUND_FACTOR * new_stdev:
                return (z, new_stdev)
        raise AssertionError("Failed to find a valid z")

    @classmethod
    def _discrete_gaussian_base_stdev_and_z_values(cls, stdev: Fraction) -> tuple[mpmath.MpfType, list[int]]:
        # If the standard deviation is large, we use Lemma 5.1 of
        # https://eprint.iacr.org/2017/259.pdf to repeatedly approximate a
        # discrete gaussian distribution by a linear combination of two
        # discrete gaussian distributions with smaller standard deviation,
        # until the standard deviation is small enough to use Knuth-Yao sampling.
        #
        # Unlike Algorithm 2 in the paper, we do not have a fixed small
        # standard deviation.  Instead, we choose the small standard deviation
        # by working backwards from `stdev`, choosing the largest allowed value
        # of the linear combination parameter z at each step.
        base_stdev = cls._mp.mpmathify(stdev)
        z_values = []
        while cls._discrete_gaussian_cdf_len(base_stdev) > cls._DISCRETE_GAUSSIAN_MAX_TABLE_SIZE:
            z, base_stdev = cls._large_discrete_gaussian_z(base_stdev)
            z_values.append(z)
        return (base_stdev, z_values)

    @classmethod
    @cache
    def gaussian_max_coefficient(cls, stdev: Fraction) -> int:
        """Upper bound on the magnitude of a sampled coefficient."""
        base, weights = cls._discrete_gaussian_base_stdev_and_z_values(stdev)
        # Each combination z*x + (z-1)*y multiplies the magnitude bound by 2*z-1.
        return (cls._discrete_gaussian_cdf_len(base) - 1) * prod(2 * z - 1 for z in weights)

    @classmethod
    def _discrete_gaussian_dtype(cls, cdf_len: int) -> DType:
        if cdf_len <= 1 << 7:
            return dtypes.int8
        elif cdf_len <= 1 << 15:
            return dtypes.int16
        elif cdf_len <= 1 << 31:
            return dtypes.int32
        else:
            raise ValueError("Table is too large")

    @classmethod
    @cache
    def _discrete_gaussian_plan_shape(cls, stdev: Fraction) -> tuple[DType, bool]:
        base_stdev, z_values = cls._discrete_gaussian_base_stdev_and_z_values(stdev)
        cdf_len = cls._discrete_gaussian_cdf_len(base_stdev)
        dtype = cls._discrete_gaussian_dtype(cdf_len)
        return (dtype, bool(z_values))

    @cache
    def discrete_gaussian(self, stdev: Fraction) -> DiscreteGaussianPlan:
        base_stdev, z_values = self._discrete_gaussian_base_stdev_and_z_values(stdev)
        tree_table, base_sample_dtype = self._discrete_gaussian_base(base_stdev)
        z_table = _to_device(self.device, np.array(z_values, dtype=np.uint64), dtypes.uint64) if z_values else None
        return DiscreteGaussianPlan(base_sampler_table=tree_table, sample_combination_table=z_table, base_sample_dtype=base_sample_dtype)

    @classmethod
    def _discrete_gaussian_cdf_len(cls, stdev: mpmath.MpfType):
        return int(cls._mp.ceil(cls._DISCRETE_GAUSSIAN_CUTOFF_NUM_STDEVS * stdev))

    @classmethod
    def _discrete_gaussian_cdf(cls, stdev: mpmath.MpfType) -> list[mpmath.MpfType]:
        upper_bound = cls._discrete_gaussian_cdf_len(stdev)
        factor = 1 / (2 * stdev * stdev)
        # Zero gets half the weight because the sampler chooses the sign separately.
        weights = (cls._mp.mpf("0.5") if n == 0 else cls._mp.exp(-n * n * factor) for n in range(upper_bound))
        cumulative = list(itertools.accumulate(weights))
        total = cumulative[-1]
        return [value / total for value in cumulative]

    def _discrete_gaussian_base(self, stdev: mpmath.MpfType) -> tuple[Buffer, DType]:
        if stdev <= 0:
            raise ValueError(f"expected positive stdev, got {stdev}")
        cdf = self._discrete_gaussian_cdf(stdev)
        dtype = self._discrete_gaussian_dtype(len(cdf))
        tree = self._generate_tree(cdf, self.MAX_DEPTH, TreeNodeEncoder(dtype))
        return _bytes_to_device(self.device, tree, dtype=dtypes.uint32), dtype

    def __init__(self, device: str):
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def _generate_tree(cls, cdf: list[mpmath.MpfType], max_depth: int, encoder: TreeNodeEncoder) -> bytearray:
        precision = 32 * max_depth
        total = 1 << precision
        cuts = [int(p * total) for p in cdf]
        end = bisect.bisect_left(cuts, total)
        if not end:
            return encoder.pack_node(0, (0, 0, 0))

        queue = deque([(0, end, precision - 32)])
        tree = bytearray()

        def child(start: int, end: int, shift: int) -> int:
            if start == end:
                return start
            queue.append((start, end, shift))
            return ~len(queue)

        while queue:
            start, end, shift = queue.popleft()
            low = cuts[start - 1] if start else 0
            midpoint = (low + cuts[end]) // 2
            pivot = bisect.bisect_left(cuts, midpoint, start, end)
            if pivot == end or (pivot > start and midpoint - cuts[pivot - 1] < cuts[pivot] - midpoint):
                pivot -= 1
            prefix = cuts[pivot] >> shift

            # All cuts with this prefix need the next word to distinguish their outcomes.
            first = bisect.bisect_left(cuts, prefix << shift, start, end)
            after = bisect.bisect_left(cuts, (prefix + 1) << shift, first, end)
            left = child(start, first, shift)
            equal = child(first, after, shift - 32) if shift else after
            right = child(after, end, shift)
            tree += encoder.pack_node(prefix & 0xFFFFFFFF, (left, equal, right))
        return tree
