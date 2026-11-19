import random
from collections.abc import Callable, Sequence
from functools import cache
from math import pi, cos, sin

from flint import fmpz_poly


from ren.dtype import DType, dtypes
from ren.poly import Poly
from ren.ir.node import Node, ParamArg
from ren.ring import RingSpec
from ren.ckks import Ciphertext, encrypt, encode
from ren.ckks.ckks import Plaintext
from ren.ckks.utils import get_prev_prime
from ren.ckks.context import context


_PRIME_SIZE_PRESETS: dict[DType, dict[str, list[int]]] = {
    dtypes.uint32: {
        "large": [31] * 22,
        "mixed": [31] + [29] * 10 + [31, 31],
    },
    dtypes.uint64: {
        "large": [60] * 22,
        "mixed": [60] + [40] * 10 + [60, 60],
    },
}


def _ring(n: int, prime_sizes: list[int], *, dtype: DType = dtypes.uint64) -> RingSpec:
    """
    Test helper: construct an NTT/RNS-friendly ring for the given degree and prime sizes.

    Args:
        n: Ring degree (must be a power of two)
        prime_sizes: List of bit sizes for each prime (e.g., [60, 40, 40, 60])
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"ring requires n to be a positive power of two, got n={n}")

    bounds = {}

    def fetch_prime(bit_count):
        bound = bounds.get(bit_count, 1 << bit_count)
        p = get_prev_prime(m=2 * n, bound=bound)
        bounds[bit_count] = p
        return p

    primes = tuple(fetch_prime(bit_count) for bit_count in prime_sizes)
    return RingSpec(n=n, moduli=primes, dtype=dtype)


@cache
def large_prime_ring(n: int, dtype: DType = dtypes.uint64) -> RingSpec:
    """Test helper: construct a ring with a wide RNS chain for the requested dtype."""
    return _ring(n, _PRIME_SIZE_PRESETS[dtype]["large"], dtype=dtype)


@cache
def mixed_prime_ring(n: int, dtype: DType = dtypes.uint64) -> RingSpec:
    """Test helper: construct a mixed-width RNS chain for the requested dtype."""
    return _ring(n, _PRIME_SIZE_PRESETS[dtype]["mixed"], dtype=dtype)


class Leaf(Poly):
    def __init__(self, value: int):
        super().__init__([value, 0, 0, 0], large_prime_ring(4))
        self.value = value


def rand_int_arr(*, size: int, lbound: int = 0, bound: int = 100):
    """
    Random integer array. Includes lower bound but not upper bound.
    """
    return [random.randrange(lbound, bound) for _ in range(size)]


def random_complex_with_norm(norm: float) -> complex:
    """Returns a random complex number with the given norm."""
    arg = random.uniform(0, 2 * pi)
    return norm * cos(arg) + norm * sin(arg) * 1.0j


def random_values_with_norm(n: int, *, norm: float) -> list[complex]:
    """Returns a list of n random complex numbers with the given norm."""
    return [random_complex_with_norm(norm) for _ in range(n)]


def random_plaintext(*, msg_bound: float, level: int | None = None) -> Plaintext:
    """Returns a plaintext of random values with norm equal to msg_bound."""
    ctx = context()
    values = random_values_with_norm(ctx.params.slot_count, norm=msg_bound)
    return encode(values, msg_bound=msg_bound, level=level)


def random_ciphertext(*, msg_bound: float, level: int | None = None) -> Ciphertext:
    """Returns a ciphertext of random values with norm equal to msg_bound."""
    plaintext = random_plaintext(msg_bound=msg_bound, level=level)
    return encrypt(plaintext)


def neg_convolve_naive(arr1: list[int], arr2: list[int], *, n: int, q: int | None = None):
    """
    Perform a naive negacyclic convolution of arr1 and arr2.
    If modulus is given, reduce all results modulo modulus.
    """
    result = [0] * n
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            k = (i + j) % n
            sign = 1 if (i + j) < n else -1
            result[k] += sign * arr1[i] * arr2[j]
    if q is not None:
        result = [x % q for x in result]
    return result


def neg_convolve(arr1: list[int], arr2: list[int], *, n: int, q: int) -> list[int]:
    """Exact negacyclic convolution modulo q, using an independent FLINT product."""
    assert len(arr1) <= n and len(arr2) <= n, "inputs must have at most n coefficients"
    product = fmpz_poly(arr1) * fmpz_poly(arr2)
    return [int(product[i] - product[i + n]) % q for i in range(n)]


def eval_scalar(build: Callable[..., Node], columns: Sequence[Sequence[int]], dtype: DType) -> list[int]:
    """Evaluate a scalar graph elementwise over input columns on the default device.

    uint128 intermediates are decomposed only when the backend requires it.
    """
    import numpy as np

    from ren.buffer import Buffer, BufferSpec
    from ren.device import Device
    from ren.ir import Op
    from ren.kernel import KernelInfo

    count = len(columns[0])
    gidx = Node(Op.GIDX, dtypes.index)
    value = build(*(Node(Op.PARAM, dtype, arg=ParamArg(i + 1, count, Device.DEFAULT)).index(gidx).load() for i in range(len(columns))))
    out_dtype = value.dtype
    store = Node(Op.PARAM, out_dtype, arg=ParamArg(0, count, Device.DEFAULT)).index(gidx).store(value)
    ring = mixed_prime_ring(count, out_dtype)[:1]
    ast = Node(Op.SINK, dtypes.void, (store,), KernelInfo(Device.DEFAULT, ring))

    def buffer(dt: DType, data: Sequence[int] | None = None) -> Buffer:
        buf = Buffer(BufferSpec(dt, count, Device.DEFAULT)).ensure_allocated()
        if data is not None:
            buf.copyin(memoryview(np.array(data, dtype=dt.fmt).tobytes()))
        return buf

    bufs = [buffer(out_dtype), *(buffer(dtype, column) for column in columns)]
    Device[Device.DEFAULT].make_kernel(ast)(bufs)
    out = memoryview(bytearray(bufs[0].nbytes))
    bufs[0].copyout(out)
    return [int(v) for v in np.frombuffer(out, dtype=out_dtype.fmt)]
