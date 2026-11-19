import numpy as np
import ren.backends.python as python_backend
from ren.dtype import dtypes, to_np_dtype
from ren.poly import Poly
from ren.tables import _ntt_params
from tests.utils import _ring, large_prime_ring
import pytest


@pytest.mark.parametrize("bits", [1, 2, 3, 8, 17, 32, 63, 64])
def test_python_bit_reverse(bits):
    dtype = to_np_dtype(dtypes.index)
    limits = np.iinfo(dtype)
    values = np.random.default_rng(42).integers(limits.min, limits.max, size=(4, 8), dtype=dtype)
    values[0] = [0, 1, 2, 3, -1, -2, limits.min, limits.max]
    for source in (values, values[:, ::2], values[:0], np.asarray(-2, dtype=dtype), np.arange(1 << min(bits, 8), dtype=dtype)):
        before = source.copy()
        source.setflags(write=False)
        expected = []
        for value in source.flat:
            reversed_value = int(f"{int(value) & ((1 << bits) - 1):0{bits}b}"[::-1], 2)
            expected.append(reversed_value if reversed_value <= limits.max else reversed_value - (1 << 64))

        actual = python_backend._bit_reverse(source, bits)

        np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype).reshape(source.shape), strict=True)
        np.testing.assert_array_equal(source, before, strict=True)


@pytest.mark.parametrize("log_n", range(10, 18))
def test_ntt_rns_roundtrip(log_n):
    n = 1 << log_n
    ring = large_prime_ring(n)
    p = Poly.uniform_on_interval(ring, bound=49)
    p = p + Poly([50 for _ in range(n)], ring)

    p_ntt = p.ntt()
    p_ntt.realize()
    p_back = p_ntt.intt()

    assert list(p) == list(p_back)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("n,prime_count", [(1, 1), (1, 5), (2, 1), (8, 5), (32, 3)])
def test_python_ntt_matches_direct_transform(monkeypatch, dtype, n, prime_count):
    # Exercise full and partial batches, and degrees larger than the batch budget.
    monkeypatch.setattr(python_backend, "_NTT_BATCH_COEFFS", 16)
    ring = _ring(n, [dtype.itemsize * 8 - 1] * prime_count, dtype=dtype)
    np_dtype = to_np_dtype(dtype)
    max_word = np.iinfo(np_dtype).max
    rows = np.random.default_rng(42).integers(0, max_word, size=(prime_count, n), dtype=np_dtype)
    expected_ntt, expected_intt = [], []
    for row, q in zip(rows, ring.moduli):
        edges = [q + 1, q - 1, 0, max_word, 1, q]
        row[: min(n, len(edges))] = edges[:n]
        values = [int(value) for value in row]
        root = _ntt_params(n, q).root
        bits = n.bit_length() - 1
        exponents = [2 * int(f"{k:0{bits}b}"[::-1], 2) + 1 for k in range(n)]
        expected_ntt.append([sum(value * pow(root, exponent * j, q) for j, value in enumerate(values)) % q for exponent in exponents])
        expected_intt.append([
            sum(value * pow(root, -exponent * j, q) for value, exponent in zip(values, exponents)) * pow(n, -1, q) % q for j in range(n)
        ])

    rows.setflags(write=False)
    backend = python_backend.PythonBackend()
    actual_ntt = backend._handle_ntt(ring, rows)
    actual_intt = backend._handle_intt(ring, rows)

    np.testing.assert_array_equal(actual_ntt, np.asarray(expected_ntt, dtype=np_dtype).reshape(-1), strict=True)
    np.testing.assert_array_equal(actual_intt, np.asarray(expected_intt, dtype=np_dtype).reshape(-1), strict=True)
