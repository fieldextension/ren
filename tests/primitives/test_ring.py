import pickle

import pytest

from ren.dtype import DType, dtypes
from ren.ring import RingSpec


def test_ring_cache_key_distinguishes_dtype():
    ring_u32 = RingSpec(n=4, moduli=(17,), dtype=dtypes.uint32)
    ring_u64 = RingSpec(n=4, moduli=(17,), dtype=dtypes.uint64)

    assert ring_u32 is RingSpec(n=4, moduli=(17,), dtype=dtypes.uint32)
    assert ring_u64 is RingSpec(n=4, moduli=(17,), dtype=dtypes.uint64)
    assert ring_u32 is not ring_u64
    assert ring_u32 != ring_u64


def test_ring_slice_preserves_dtype():
    ring = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtypes.uint64)
    sliced = ring[:2]

    assert sliced.moduli == (17, 41)
    assert sliced.dtype is dtypes.uint64


@pytest.mark.parametrize(
    ("dtype", "q"),
    [
        (dtypes.uint32, (1 << 31) + 1),
        (dtypes.uint64, (1 << 63) + 1),
    ],
)
def test_ring_rejects_moduli_outside_dtype_budget(dtype, q):
    with pytest.raises(ValueError, match="All moduli must be less than"):
        RingSpec(n=4, moduli=(q,), dtype=dtype)


def test_is_quotient_of_requires_matching_dtype():
    ring_u32 = RingSpec(n=4, moduli=(17, 41), dtype=dtypes.uint32)
    ring_u64 = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtypes.uint64)

    assert ring_u32.is_quotient_of(RingSpec(n=4, moduli=(17, 41, 73), dtype=dtypes.uint32))
    assert not ring_u32.is_quotient_of(ring_u64)


def test_is_quotient_of_requires_matching_degree():
    ring_n4 = RingSpec(n=4, moduli=(17, 97))
    ring_n8 = RingSpec(n=8, moduli=(17, 97, 113))

    assert ring_n4.is_quotient_of(RingSpec(n=4, moduli=(17, 97, 113)))
    assert not ring_n4.is_quotient_of(ring_n8)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_dtype_pickle_reinterns_to_canonical(dtype):
    # Pickling an interned DType must return the canonical instance, not a stray
    # copy; otherwise identity-based checks like ``dtype in dtypes.uints`` fail.
    restored = pickle.loads(pickle.dumps(dtype))

    assert restored is dtype
    assert restored in dtypes.uints


def test_dtype_pickle_reinterns_unregistered_instance():
    # An instance built directly (still interned via CacheMeta) round-trips to
    # the same canonical object.
    dtype = DType(8, "unsigned long long", "Q")
    assert pickle.loads(pickle.dumps(dtype)) is dtypes.uint64


def test_ring_pickle_reinterns_to_canonical():
    ring = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtypes.uint64)
    restored = pickle.loads(pickle.dumps(ring))

    assert restored is ring
    assert restored == ring
    assert restored.dtype is dtypes.uint64
