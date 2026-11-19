"""Tests for RingTables prefix-view optimization."""

from math import prod

import numpy as np
import pytest

import ren.montgomery as mont
from ren.buffer import Buffer
from ren.dtype import dtypes, to_np_dtype
from ren.ring import RingSpec
from ren.tables import Tables
from tests.utils import _ring


def _read_buf(buf: Buffer) -> np.ndarray:
    np_dtype = to_np_dtype(buf.dtype)
    out = np.empty(buf.size, dtype=np_dtype)
    buf.copyout(memoryview(out).cast("B"))
    return out


P0, P1, P2, P3, P4 = 17, 41, 73, 89, 97
N = 4


def setup_function():
    Tables.clear()


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64], ids=["uint32", "uint64"])
def test_prefix_view_sharing(dtype):
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4), dtype=dtype)
    prefix_ring = RingSpec(n=N, moduli=(P0, P1, P2), dtype=dtype)

    tables = Tables("python")
    full_rt = tables.ring(full_ring)
    prefix_rt = tables.ring(prefix_ring)

    assert prefix_rt._parent is full_rt
    assert prefix_rt._prime_offset == 0

    for attr in ("primes", "mont_radix32", "mont_r2", "n_inv", "mont_p_inv"):
        parent_buf = getattr(full_rt, attr)
        child_buf = getattr(prefix_rt, attr)
        assert child_buf.dtype is dtype
        assert child_buf.base is parent_buf.base
        np.testing.assert_array_equal(_read_buf(child_buf), _read_buf(parent_buf)[:3])

    for attr in ("ntt_psis", "ntt_psis_inv"):
        parent_buf = getattr(full_rt, attr)
        child_buf = getattr(prefix_rt, attr)
        assert child_buf.dtype is dtype
        assert child_buf.base is parent_buf.base
        np.testing.assert_array_equal(_read_buf(child_buf), _read_buf(parent_buf)[: N * 3])


def test_interior_slice_view():
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4))
    interior_ring = RingSpec(n=N, moduli=(P2, P3, P4))

    tables = Tables("python")
    full_rt = tables.ring(full_ring)
    interior_rt = tables.ring(interior_ring)

    assert interior_rt._parent is full_rt
    assert interior_rt._prime_offset == 2

    np.testing.assert_array_equal(_read_buf(interior_rt.primes), _read_buf(full_rt.primes)[2:5])
    np.testing.assert_array_equal(_read_buf(interior_rt.ntt_psis), _read_buf(full_rt.ntt_psis)[N * 2 : N * 5])
    np.testing.assert_array_equal(_read_buf(interior_rt.ntt_psis_inv), _read_buf(full_rt.ntt_psis_inv)[N * 2 : N * 5])
    np.testing.assert_array_equal(_read_buf(interior_rt.mont_radix32), _read_buf(full_rt.mont_radix32)[2:5])
    np.testing.assert_array_equal(_read_buf(interior_rt.n_inv), _read_buf(full_rt.n_inv)[2:5])
    np.testing.assert_array_equal(_read_buf(interior_rt.mont_p_inv), _read_buf(full_rt.mont_p_inv)[2:5])


def test_non_contiguous_independence():
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4))
    non_contig_ring = RingSpec(n=N, moduli=(P0, P2, P4))

    tables = Tables("python")
    tables.ring(full_ring)
    non_contig_rt = tables.ring(non_contig_ring)

    assert non_contig_rt._parent is None
    np.testing.assert_array_equal(_read_buf(non_contig_rt.primes), [P0, P2, P4])


def test_creation_order_sub_first():
    prefix_ring = RingSpec(n=N, moduli=(P0, P1, P2))
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4))

    tables = Tables("python")
    prefix_rt = tables.ring(prefix_ring)
    full_rt = tables.ring(full_ring)

    assert prefix_rt._parent is None
    assert full_rt._parent is None

    np.testing.assert_array_equal(_read_buf(prefix_rt.primes), [P0, P1, P2])
    np.testing.assert_array_equal(_read_buf(full_rt.primes), [P0, P1, P2, P3, P4])


def test_rotation_factors_reuse_equivalent_offsets():
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4))
    interior_ring = RingSpec(n=N, moduli=(P2, P3, P4))

    tables = Tables("python")
    full_rt = tables.ring(full_ring)
    interior_rt = tables.ring(interior_ring)

    k = 2 * N - 1
    for offset in (-3, -1, 0, 1, 3):
        factors = interior_rt.get_rotation_factors(k + offset * 2 * N)
        assert factors is interior_rt.get_rotation_factors(k)
        assert factors.base is full_rt.get_rotation_factors(k)

    assert set(interior_rt._rot_cache) == set(full_rt._rot_cache) == {k}


def test_rotation_factors_view():
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4))
    prefix_ring = RingSpec(n=N, moduli=(P0, P1, P2))

    tables = Tables("python")
    full_rt = tables.ring(full_ring)
    prefix_rt = tables.ring(prefix_ring)

    full_rot = full_rt.get_rotation_factors(3)
    prefix_rot = prefix_rt.get_rotation_factors(3)

    assert prefix_rot.base is full_rot.base
    np.testing.assert_array_equal(_read_buf(prefix_rot), _read_buf(full_rot)[: N * 3])


def test_rotation_factors_lazy_parent():
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4))
    prefix_ring = RingSpec(n=N, moduli=(P0, P1, P2))

    tables = Tables("python")
    full_rt = tables.ring(full_ring)
    prefix_rt = tables.ring(prefix_ring)

    k = 7
    assert k not in full_rt._rot_cache

    prefix_rot = prefix_rt.get_rotation_factors(k)
    assert k in full_rt._rot_cache

    full_vals = _read_buf(full_rt.get_rotation_factors(k))
    np.testing.assert_array_equal(_read_buf(prefix_rot), full_vals[: N * 3])


def test_chained_views():
    big_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4))
    medium_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3))
    small_ring = RingSpec(n=N, moduli=(P0, P1))

    tables = Tables("python")
    tables.ring(big_ring)
    medium_rt = tables.ring(medium_ring)
    small_rt = tables.ring(small_ring)

    assert medium_rt._parent is not None
    assert small_rt._parent is not None

    big_primes = _read_buf(tables.ring(big_ring).primes)
    np.testing.assert_array_equal(_read_buf(medium_rt.primes), big_primes[:4])
    np.testing.assert_array_equal(_read_buf(small_rt.primes), big_primes[:2])


def test_parent_views_do_not_cross_dtype():
    full_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3, P4), dtype=dtypes.uint64)
    prefix_ring = RingSpec(n=N, moduli=(P0, P1, P2), dtype=dtypes.uint32)

    tables = Tables("python")
    tables.ring(full_ring)
    prefix_rt = tables.ring(prefix_ring)

    assert prefix_rt._parent is None
    assert prefix_rt.primes.dtype is dtypes.uint32


def test_base_change_tables_dst_src_indices_metadata():
    src_ring = RingSpec(n=N, moduli=(P1, P2, P3, P4))
    dst_ring = RingSpec(n=N, moduli=(P0, P1, P3))

    tables = Tables("python")
    rt = tables.base_change(src_ring, dst_ring)

    missing = np.iinfo(np.uint32).max
    np.testing.assert_array_equal(_read_buf(rt.dst_src_indices), [missing, 0, 2])


@pytest.mark.parametrize("dtype,bits", [(dtypes.uint32, 31), (dtypes.uint64, 63)])
def test_base_change_r2_tables_match_exact_values(dtype, bits):
    primes = _ring(N, [bits] * 5, dtype=dtype).moduli
    src, dst = primes[:3], (primes[4], primes[3])
    tables = Tables("python").base_change(RingSpec(N, src, dtype), RingSpec(N, dst, dtype))
    partial_products = [prod(p for j, p in enumerate(src) if i != j) for i in range(len(src))]
    shift = 2 * dtype.itemsize * 8
    expected_mi = [(partial << shift) % q for q in dst for partial in partial_products]
    expected_prod = [(prod(src) << shift) % q for q in dst]

    np.testing.assert_array_equal(_read_buf(tables.mi_r2_scaled), np.asarray(expected_mi, dtype=to_np_dtype(dtype)), strict=True)
    np.testing.assert_array_equal(_read_buf(tables.prod_r2_scaled), np.asarray(expected_prod, dtype=to_np_dtype(dtype)), strict=True)


def test_rescale_tables_non_quotient_metadata():
    src_ring = RingSpec(n=N, moduli=(P1, P2, P3, P4), dtype=dtypes.uint32)
    dst_ring = RingSpec(n=N, moduli=(P0, P1, P3), dtype=dtypes.uint32)

    tables = Tables("python")
    rt = tables.rescale(src_ring, dst_ring)

    missing = np.iinfo(np.uint32).max
    np.testing.assert_array_equal(_read_buf(rt.removed_src_indices), [1, 3])
    np.testing.assert_array_equal(_read_buf(rt.dst_src_indices), [missing, 0, 2])

    added_prod = P0
    expected_removed = [mont.context(src_ring.dtype, modulus).encode(added_prod) for modulus in (P2, P4)]
    expected_dst = [mont.context(dst_ring.dtype, modulus).encode(added_prod) for modulus in dst_ring.moduli]
    np.testing.assert_array_equal(_read_buf(rt.added_prod_removed_mont), expected_removed)
    np.testing.assert_array_equal(_read_buf(rt.added_prod_dst_mont), expected_dst)


def test_rescale_tables_quotient_added_prod_is_one():
    src_ring = RingSpec(n=N, moduli=(P0, P1, P2, P3), dtype=dtypes.uint32)
    dst_ring = RingSpec(n=N, moduli=(P0, P1), dtype=dtypes.uint32)

    tables = Tables("python")
    rt = tables.rescale(src_ring, dst_ring)

    one_removed = [mont.context(src_ring.dtype, modulus).encode(1) for modulus in (P2, P3)]
    one_dst = [mont.context(dst_ring.dtype, modulus).encode(1) for modulus in dst_ring.moduli]
    np.testing.assert_array_equal(_read_buf(rt.added_prod_removed_mont), one_removed)
    np.testing.assert_array_equal(_read_buf(rt.added_prod_dst_mont), one_dst)


def test_rescale_tables_reuse_base_change_buffers():
    src_ring = RingSpec(n=N, moduli=(P1, P2, P3, P4), dtype=dtypes.uint32)
    dst_ring = RingSpec(n=N, moduli=(P0, P1, P3), dtype=dtypes.uint32)
    removed_ring = RingSpec(n=N, moduli=(P2, P4), dtype=src_ring.dtype)

    tables = Tables("python")
    rt = tables.rescale(src_ring, dst_ring)
    bc = tables.base_change(removed_ring, dst_ring)

    assert rt.inv_src is bc.inv_src
    assert rt.mi_r2_scaled is bc.mi_r2_scaled
    assert rt.prod_r2_scaled is bc.prod_r2_scaled


def test_conversion_table_view_with_63_bit_parent():
    ring = _ring(8, [63, 60, 40])
    tables = Tables("python")
    parent = tables.ring(ring)
    child = tables.ring(ring[1:])
    actual = _read_buf(child.mont_radix32)
    np.testing.assert_array_equal(actual, [(1 << 96) % q for q in ring.moduli[1:]])
    np.testing.assert_array_equal(actual, _read_buf(parent.mont_radix32)[1:])
    assert child.mont_radix32.base is parent.mont_radix32.base
