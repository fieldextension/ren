import pickle

import pytest

from ren.dtype import dtypes
from ren.poly import Poly
from ren.ring import Domain
from tests.utils import large_prime_ring, mixed_prime_ring


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(bytes, id="bytes"),
        pytest.param(lambda data: memoryview(data).cast("I"), id="typed-view"),
        pytest.param(lambda data: memoryview(data).cast("B", shape=[2, len(data) // 2]), id="byte-matrix"),
    ],
)
def test_poly_tobytes_roundtrip_coeff_domain(dtype, wrap):
    ring = mixed_prime_ring(1024, dtype=dtype)
    p = Poly.uniform(ring)

    raw = p.tobytes()
    decoded = Poly.frombytes(wrap(raw), ring=ring, domain=Domain.COEFF)

    assert list(decoded) == list(p)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_poly_tobytes_roundtrip_ntt_domain(dtype):
    ring = mixed_prime_ring(1024, dtype=dtype)
    p = Poly.uniform(ring).ntt()

    raw = p.tobytes()
    decoded = Poly.frombytes(raw, ring=ring, domain=Domain.NTT)

    assert list(decoded) == list(p)


def test_poly_const_tobytes_roundtrip():
    ring = mixed_prime_ring(1024)
    p = Poly.const(11, ring)

    raw = p.tobytes()
    decoded = Poly.frombytes(raw, ring=ring, domain=Domain.COEFF)

    assert list(decoded) == list(p)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_poly_pickle_roundtrip_coeff_domain(dtype):
    ring = mixed_prime_ring(1024, dtype=dtype)
    p = Poly.uniform(ring)

    decoded = pickle.loads(pickle.dumps(p))

    assert decoded.domain == Domain.COEFF
    assert list(decoded) == list(p)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_poly_pickle_roundtrip_ntt_domain(dtype):
    ring = mixed_prime_ring(1024, dtype=dtype)
    p = Poly.uniform(ring).ntt()

    decoded = pickle.loads(pickle.dumps(p))

    assert decoded.domain == Domain.NTT
    assert list(decoded) == list(p)


@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
@pytest.mark.parametrize("make_poly", [lambda ring: Poly.const(7, ring), lambda ring: Poly.const(3, ring) * 2 + 1], ids=["literal", "expression"])
def test_poly_pickle_roundtrip_const(domain, make_poly):
    ring = mixed_prime_ring(1024)
    p = make_poly(ring)
    if domain is Domain.NTT:
        p = p.ntt()
    payload = pickle.dumps(p)
    decoded = pickle.loads(payload)

    assert len(payload) < ring.nbytes
    assert not p.is_realized
    assert decoded.domain is domain
    assert list(decoded) == list(p) == [7] + [0] * (ring.n - 1)


def test_poly_pickle_materializes_lazy_graph():
    # Pickling is a materialization boundary on the source side: serializing an
    # unrealized expression realizes it in place (so it can be stored as concrete
    # bytes), and the round-trip reconstructs to the same value. The decoded
    # poly's realization state is left to the scheduler and is device-dependent,
    # so it is intentionally not asserted here.
    ring = mixed_prime_ring(1024)
    a = Poly.uniform(ring)
    b = Poly.uniform(ring)
    expr = a * b + a
    assert not expr.is_realized

    decoded = pickle.loads(pickle.dumps(expr))

    assert expr.is_realized
    assert list(decoded) == list(expr)


@pytest.mark.parametrize("mixed_domain", [False, True])
def test_pickle_device_free_constant_leaves_write_pending(mixed_domain):
    ring = large_prime_ring(4)
    target = Poly([1, 2, 3, 4], ring).realize()
    value = Poly.const(7, ring)
    if mixed_domain:
        value = value.ntt() + Poly.const(2, ring)
    output = Poly(value.node.after(target.node.store(Poly.const(9, ring).node)))

    restored = pickle.loads(pickle.dumps(output))

    assert target.tolist() == [1, 2, 3, 4]
    assert restored.domain is Domain.COEFF
    assert restored.tolist() == [9 if mixed_domain else 7, 0, 0, 0]
