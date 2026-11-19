import numpy as np
import pytest

from ren.dtype import dtypes
from ren.ir import Node
from ren.montgomery import context
from tests.utils import eval_scalar, mixed_prime_ring


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_modular_constant_preserves_cast_chain(dtype):
    value = -(1 << 200) + (1 << 64) + 7
    p = mixed_prime_ring(128, dtype)[:1].q
    constant = Node.const(value).cast(dtypes.uint32).cast(dtype)
    expected = (value % (1 << 32)) * pow(2, dtype.itemsize * 8, p) % p
    assert eval_scalar(lambda x: x + constant.mod_reduce(None), [[0] * 128], dtype) == [expected] * 128


@pytest.mark.parametrize("dtype,q", [(dtypes.uint32, 2147483647), (dtypes.uint64, 9223372036854775783)])
def test_arithmetic_matches_exact_values(dtype, q):
    mp = context(dtype, q)
    values = [0, 1, q - 2, q - 1]
    r_inv = pow(1 << (dtype.itemsize * 8), -1, q)
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    array = np.asarray(values, dtype=np_dtype)

    for operation, expected in (
        (mp.add, [[(a + b) % q for b in values] for a in values]),
        (mp.sub, [[(a - b) % q for b in values] for a in values]),
        (mp.mul, [[a * b * r_inv % q for b in values] for a in values]),
    ):
        np.testing.assert_array_equal(operation(array[:, None], array), np.asarray(expected, dtype=np_dtype), strict=True)
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                actual = operation(a, b)
                assert isinstance(actual, int) and actual == expected[i][j]


@pytest.mark.parametrize("dtype,q", [(dtypes.uint32, 2147483647), (dtypes.uint64, 9223372036854775783)])
def test_conversions_match_exact_values(dtype, q):
    mp = context(dtype, q)
    radix = 1 << (dtype.itemsize * 8)
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    values = [0, 1, -1, q - 1, q, q + 1, -q - 1, 2**130 + 123, -(2**130) + 123, np.int64(-1), np.uint64(2**64 - 1)]

    for value in values:
        encoded = mp.encode(value)
        decoded = mp.decode(encoded)
        assert isinstance(encoded, int) and encoded == int(value) * radix % q
        assert isinstance(decoded, int) and decoded == int(value) % q

    for array in (
        values,
        np.asarray(values, dtype=object).reshape(1, -1),
        np.asarray([-1, 0, 1], dtype=np.int64),
        np.asarray([0, 1, 2**64 - 1], dtype=np.uint64),
        np.asarray(-1, dtype=np.int64),
    ):
        source = np.asarray(array, dtype=object)
        expected = np.asarray([int(value) * radix % q for value in source.flat], dtype=np_dtype).reshape(source.shape)
        reduced = np.asarray([int(value) % q for value in source.flat], dtype=np_dtype).reshape(source.shape)
        encoded = mp.encode(array)
        np.testing.assert_array_equal(mp.reduce(array), reduced, strict=True)
        np.testing.assert_array_equal(encoded, expected, strict=True)
        np.testing.assert_array_equal(mp.decode(encoded), reduced, strict=True)
