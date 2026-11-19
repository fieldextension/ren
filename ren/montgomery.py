from functools import cache
from collections.abc import Sequence
from typing import overload

import numpy as np

from ren.dtype import DType, dtype_word_bits, to_np_dtype

type Scalar = int | np.integer
type ArrayLike = np.ndarray | Sequence[Scalar]

_SHIFT32 = np.uint64(32)
_LOW32 = np.uint64(0xFFFFFFFF)


def _mul_wide(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full 128-bit product of uint64 arrays, returned as ``(hi, lo)`` uint64 words.

    Each operand is split into 32-bit halves so every partial product fits in a
    uint64, then the four products are reassembled into the 128-bit result. This
    lets 64-bit Montgomery multiplication run as vectorised uint64 NumPy ops
    rather than falling back to Python-bigint ``object`` arrays.
    """
    a_lo = a & _LOW32
    a_hi = a >> _SHIFT32
    b_lo = b & _LOW32
    b_hi = b >> _SHIFT32
    lo_lo = a_lo * b_lo
    hi_lo = a_hi * b_lo
    lo_hi = a_lo * b_hi
    hi_hi = a_hi * b_hi
    # The bit-32 column: carry out of lo_lo plus the low halves of both cross terms.
    cross = (lo_lo >> _SHIFT32) + (hi_lo & _LOW32) + (lo_hi & _LOW32)
    lo = a * b
    hi = hi_hi + (hi_lo >> _SHIFT32) + (lo_hi >> _SHIFT32) + (cross >> _SHIFT32)
    return hi, lo


def add(a: np.ndarray, b: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Add reduced uint64 arrays modulo a broadcastable q < 2**63."""
    s = a + b
    # Subtract q only where it applies; avoids the underflow wrap (and its warning)
    # that a bare ``s - q`` would produce in the unselected branch.
    return s - np.where(s >= q, q, np.uint64(0))


def sub(a: np.ndarray, b: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Subtract reduced uint64 arrays modulo a broadcastable q < 2**63."""
    return np.where(a >= b, a - b, a + q - b)


def mul(aw: np.ndarray, bw: np.ndarray, q: np.ndarray, p_inv: np.ndarray, word_bits: int) -> np.ndarray:
    """Compute ``(a * b * R⁻¹) mod q`` on reduced uint64 arrays.

    Constants may broadcast over the inputs; q < 2**(word_bits - 1) and the result is reduced.
    For the 64-bit path, the 128-bit product and the ``t * q`` reduction term
    are formed with :func:`_mul_wide`, so no Python-bigint arithmetic is involved.
    """
    # Forming t relies on intentional unsigned wraparound.
    with np.errstate(over="ignore"):
        if word_bits == 64:
            z_hi, z_lo = _mul_wide(aw, bw)  # z = a * b (128-bit)
            t = z_lo * p_inv  # (z * p_inv) mod 2⁶⁴ — only the low word matters
            m_hi, _ = _mul_wide(t, q)  # m = t * q; its low word cancels z_lo, so only m_hi is used
            # t is chosen so the low 64 bits of z + m vanish; (z + m) >> 64 is then
            # exact: z_hi + m_hi plus the carry out of that cancelling low limb. That
            # carry is 1 exactly when z_lo ≠ 0 (then z_lo + m_lo = 2⁶⁴), else 0.
            carry = (z_lo != np.uint64(0)).astype(np.uint64)
            r = z_hi + m_hi + carry
        else:
            z = aw * bw
            # (z * p_inv) intentionally wraps mod 2⁶⁴; only its low bits (masked) are used.
            t = (z * p_inv) & np.uint64((1 << word_bits) - 1)
            r = (z + t * q) >> word_bits
        return np.where(r < q, r, r - q)


class Montgomery:
    """Montgomery arithmetic for a specific (dtype, modulus) pair.

    The public API accepts either Python / NumPy integer scalars or array-like
    integer sequences. Scalar inputs return Python ints; non-scalar inputs
    return ndarrays with the storage dtype for this context.
    """

    __slots__ = ("_np_dtype", "modulus", "p_inv", "r2", "r_inv", "radix", "word_bits")

    def __init__(self, dtype: DType, modulus: int):
        bits = dtype_word_bits(dtype)
        r = 1 << bits
        self.modulus = modulus
        self.word_bits = bits
        self.radix = r
        self.p_inv = (r - pow(modulus, -1, r)) % r
        self.r2 = pow(r, 2, modulus)
        self.r_inv = pow(r, -1, modulus)
        self._np_dtype = to_np_dtype(dtype)

    def _as_int_array(self, values: Scalar | ArrayLike) -> np.ndarray:
        """Convert input values to Python-int object storage for exact arithmetic."""
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 0:
            return np.asarray(int(arr.item()), dtype=object)
        flat = np.fromiter((int(x) for x in arr.flat), dtype=object, count=arr.size)
        return flat.reshape(arr.shape)

    def _reduce(self, values: Scalar | ArrayLike) -> np.ndarray:
        # Storage is uint64 words for both the 32- and 64-bit paths; the 64-bit path
        # keeps products within uint64 via limb splitting (see ``_mul_wide``).
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.unsignedinteger):
            return arr.astype(np.uint64, copy=False) % np.uint64(self.modulus)

        # Signed or object (possibly > 64-bit or negative) inputs: reduce exactly with
        # Python bigints, then narrow into uint64 (values now lie in [0, modulus)).
        reduced = self._as_int_array(values) % self.modulus
        return np.asarray(reduced, dtype=np.uint64)

    def _to_output(self, values: np.ndarray, *, inputs: tuple[object, ...]) -> np.ndarray | int:
        if all(np.isscalar(value) for value in inputs):
            return int(np.asarray(values).item())
        return np.asarray(values, dtype=self._np_dtype)

    @overload
    def mul(self, a: Scalar, b: Scalar) -> int: ...

    @overload
    def mul(self, a: ArrayLike, b: Scalar | ArrayLike) -> np.ndarray: ...

    @overload
    def mul(self, a: Scalar, b: ArrayLike) -> np.ndarray: ...

    def mul(self, a: Scalar | ArrayLike, b: Scalar | ArrayLike) -> np.ndarray | int:
        aw = np.asarray(a, dtype=np.uint64)
        bw = np.asarray(b, dtype=np.uint64)
        reduced = mul(aw, bw, np.asarray(self.modulus, dtype=np.uint64), np.asarray(self.p_inv, dtype=np.uint64), self.word_bits)
        return self._to_output(reduced, inputs=(a, b))

    @overload
    def add(self, a: Scalar, b: Scalar) -> int: ...

    @overload
    def add(self, a: ArrayLike, b: Scalar | ArrayLike) -> np.ndarray: ...

    @overload
    def add(self, a: Scalar, b: ArrayLike) -> np.ndarray: ...

    def add(self, a: Scalar | ArrayLike, b: Scalar | ArrayLike) -> np.ndarray | int:
        aw = np.asarray(a, dtype=np.uint64)
        bw = np.asarray(b, dtype=np.uint64)
        return self._to_output(add(aw, bw, np.asarray(self.modulus, dtype=np.uint64)), inputs=(a, b))

    @overload
    def sub(self, a: Scalar, b: Scalar) -> int: ...

    @overload
    def sub(self, a: ArrayLike, b: Scalar | ArrayLike) -> np.ndarray: ...

    @overload
    def sub(self, a: Scalar, b: ArrayLike) -> np.ndarray: ...

    def sub(self, a: Scalar | ArrayLike, b: Scalar | ArrayLike) -> np.ndarray | int:
        aw = np.asarray(a, dtype=np.uint64)
        bw = np.asarray(b, dtype=np.uint64)
        return self._to_output(sub(aw, bw, np.asarray(self.modulus, dtype=np.uint64)), inputs=(a, b))

    @overload
    def neg(self, a: Scalar) -> int: ...

    @overload
    def neg(self, a: ArrayLike) -> np.ndarray: ...

    def neg(self, a: Scalar | ArrayLike) -> np.ndarray | int:
        aw = np.asarray(a, dtype=np.uint64)
        q = np.uint64(self.modulus)
        reduced = np.where(aw == 0, aw, q - aw)
        return self._to_output(reduced, inputs=(a,))

    @overload
    def reduce(self, values: Scalar) -> int: ...

    @overload
    def reduce(self, values: ArrayLike) -> np.ndarray: ...

    def reduce(self, values: Scalar | ArrayLike) -> np.ndarray | int:
        return self._to_output(self._reduce(values), inputs=(values,))

    @overload
    def encode(self, values: Scalar) -> int: ...

    @overload
    def encode(self, values: ArrayLike) -> np.ndarray: ...

    def encode(self, values: Scalar | ArrayLike) -> np.ndarray | int:
        vw = self._reduce(values)
        q = np.asarray(self.modulus, dtype=np.uint64)
        if self.word_bits == 64:
            r2 = np.asarray(self.r2, dtype=np.uint64)
            encoded = mul(vw, r2, q, np.asarray(self.p_inv, dtype=np.uint64), self.word_bits)
        else:
            encoded = (vw * np.uint64(self.radix % self.modulus)) % q
        return self._to_output(encoded, inputs=(values,))

    @overload
    def decode(self, values: Scalar) -> int: ...

    @overload
    def decode(self, values: ArrayLike) -> np.ndarray: ...

    def decode(self, values: Scalar | ArrayLike) -> np.ndarray | int:
        vw = np.asarray(values, dtype=np.uint64)
        q = np.asarray(self.modulus, dtype=np.uint64)
        if self.word_bits == 64:
            one = np.asarray(1, dtype=np.uint64)
            decoded = mul(vw, one, q, np.asarray(self.p_inv, dtype=np.uint64), self.word_bits)
        else:
            decoded = (vw * np.uint64(self.r_inv)) % q
        return self._to_output(decoded, inputs=(values,))


@cache
def context(dtype: DType, modulus: int) -> Montgomery:
    """Return a cached Montgomery context for the given (dtype, modulus) pair."""
    return Montgomery(dtype, modulus)


__all__ = ["Montgomery", "context", "add", "sub", "mul"]
