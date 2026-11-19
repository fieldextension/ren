from fractions import Fraction
import gc
import weakref
import pytest
from math import prod, sqrt

import numpy as np

import ren.poly as poly_module
from ren import function, realize
from ren.allocator import BufferSpec
from ren.buffer import Buffer
from ren.dtype import dtypes
from ren.engine.realize import call_op
from ren.engine.schedule import schedule
from ren.ir import Node
from ren.ir.node import from_rns_limb_count
from ren.poly import Poly
from ren.ir import Op
from ren.polydist import DiscreteGaussian
from ren.ring import Domain, Residue, RingSpec
from tests.utils import _ring, large_prime_ring, mixed_prime_ring, neg_convolve

RING_DEGREES = [2**2, 2**10, 2**16]
DEGREE_DTYPE_CASES = [pytest.param(n, dtypes.uint64, id=f"{n}-uint64") for n in RING_DEGREES] + [pytest.param(256, dtypes.uint32, id="256-uint32")]


def test_poly_from_buffer_validates_storage():
    ring = mixed_prime_ring(1024)
    buffer = Buffer(BufferSpec(dtype=ring.dtype, size=1, device="python"))

    with pytest.raises(ValueError, match="does not match"):
        Poly._from_buffer(buffer, ring, Domain.COEFF)


def test_poly_from_buffer_requires_allocated_storage():
    ring = mixed_prime_ring(1024)
    buffer = Buffer(BufferSpec(dtype=ring.dtype, size=ring.storage_size, device="python"))

    with pytest.raises(ValueError, match="must be allocated"):
        Poly._from_buffer(buffer, ring, Domain.COEFF)


@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
def test_poly_from_buffer_preserves_value_and_releases_ownership(domain):
    source = Poly([1, 2, 3, 4], large_prime_ring(4))
    source = (source.ntt() if domain is Domain.NTT else source).realize()
    buffer = source.node.base.buffer
    buffer_ref = weakref.ref(buffer)

    wrapped = Poly._from_buffer(buffer, source.ring, domain)
    assert wrapped.node is not source.node
    assert wrapped.node.base.buffer is buffer
    assert (wrapped.ring, wrapped.domain, wrapped.device) == (source.ring, domain, source.device)
    assert wrapped.is_realized and wrapped.tolist() == source.tolist()
    del source, buffer
    gc.collect()
    assert buffer_ref() is wrapped.node.base.buffer

    del wrapped
    gc.collect()
    assert buffer_ref() is None


def test_poly_accepts_node():
    ring = large_prime_ring(4)
    source = Poly([1, 2, 3, 4], ring)

    assert Poly(source.node).node is source.node
    assert Poly(source.node, ring).node is source.node


def test_poly_attaches_ring_to_ringless_node():
    ring = large_prime_ring(4)
    node = Node.const(3)

    poly = Poly(node, ring)

    assert poly.node.src[0] is node
    assert poly.ring == ring
    assert poly.domain is Domain.COEFF
    assert poly.tolist() == [3, 0, 0, 0]


def test_poly_preserves_domain_when_attaching_ring_to_ringless_node():
    ring = large_prime_ring(4)
    node = Node(Op.NTT, dtype=ring.dtype, src=(Node.const(3),))

    poly = Poly(node, ring)

    assert poly.node.src[0] is node
    assert poly.ring == ring
    assert poly.domain is Domain.NTT


def test_poly_rejects_ringless_node_with_mismatched_dtype():
    ring = large_prime_ring(4, dtype=dtypes.uint32)

    with pytest.raises(ValueError, match="node dtype .* does not match ring dtype"):
        Poly(Node.const(3, dtype=dtypes.uint64), ring)


@pytest.mark.parametrize("op", [Op.BUFFER, Op.BUFFER_VIEW])
@pytest.mark.parametrize("delta", [-1, 1])
def test_poly_rejects_ringless_buffer_with_mismatched_size(op, delta):
    ring = large_prime_ring(4)
    size = ring.storage_size + delta
    node = Node.new_buffer("python", size=size, dtype=ring.dtype)
    if op is Op.BUFFER_VIEW:
        source = Node.new_buffer("python", size=ring.storage_size + 2, dtype=ring.dtype)
        node = Node(Op.BUFFER_VIEW, ring.dtype, (source,), (size, 1))

    with pytest.raises(ValueError, match="buffer size .* does not match ring storage size"):
        Poly(node, ring)


def test_poly_rejects_conflicting_node_ring():
    source = Poly([1, 2, 3, 4], large_prime_ring(4))

    with pytest.raises(ValueError, match="ring does not match node ring"):
        Poly(source.node, mixed_prime_ring(4))


def centered_coeffs(poly: Poly):
    q = poly.ring.q
    return [((c + q // 2) % q) - q // 2 for c in list(poly)]


def rounded_rescale_coeffs(poly: Poly, target_moduli: tuple[int, ...]) -> list[int]:
    src_moduli = set(poly.ring.moduli)
    dst_moduli = set(target_moduli)
    removed_product = prod(modulus for modulus in poly.ring.moduli if modulus not in dst_moduli)
    added_product = prod(modulus for modulus in target_moduli if modulus not in src_moduli)
    target_product = prod(target_moduli)
    expected = []
    for coefficient in poly:
        scaled = coefficient * added_product
        remainder = scaled % removed_product
        centered_remainder = remainder - removed_product if remainder > removed_product // 2 else remainder
        expected.append(((scaled - centered_remainder) // removed_product) % target_product)
    return expected


@pytest.mark.parametrize("n", RING_DEGREES)
def test_realize(n: int):
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)
    assert len(list(p)) == n


def test_realize_does_not_mutate_realized_input_poly():
    ring = large_prime_ring(64)
    x = Poly([((i * 17 + 3) % 97) for i in range(ring.n)], ring)
    x.realize()
    before_node = x.node
    before_coeffs = x.tolist()

    y = x
    y = y + x
    y = y.contiguous()
    y = y + 1
    y = y + x
    y = y._to_ring(ring)
    y = y + x
    y = y.rotate(7)
    y = y._to_ring(ring)
    y = y.rotate(3)
    y = y.contiguous()
    y = y + x
    y = y + x
    y = y._to_ring(ring)
    y.realize()

    # Realizing a derived expression must not rewrite the already-realized source poly.
    assert x.node is before_node
    assert x.tolist() == before_coeffs


def test_realize_constant_does_not_reschedule(monkeypatch):
    p = Poly.const(3, large_prime_ring(4)).realize()

    def unexpected_schedule(root):
        pytest.fail("a constant needs no schedule")

    with monkeypatch.context() as patch:
        patch.setattr(poly_module, "schedule", unexpected_schedule)
        assert p.realize() is p
        realize(p, [p])
    assert not p.is_realized
    assert p.tolist() == [3, 0, 0, 0]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
def test_realize_constant_expression(dtype, domain):
    ring = RingSpec(n=4, moduli=(17, 41), dtype=dtype)
    p = Poly.const(3, ring) * 2 + 1
    p = p.ntt() if domain is Domain.NTT else p.intt()
    p.realize()

    assert p.domain is domain
    assert p.tolist() == [7, 0, 0, 0]


def test_polynomial_constant_after_scalar_overflow():
    ring = RingSpec(n=4, moduli=(17, 41), dtype=dtypes.uint64)
    maximum = (1 << 64) - 1
    scalar = (Node.const(maximum, dtypes.uint64) + 1).cast(dtypes.bool).cast(dtypes.uint64)
    p = Poly(scalar, ring)
    assert p.tolist() == p.clone().realize().tolist() == [0] * ring.n
    assert (Poly.const(maximum, ring) + 1).tolist() == [(1 << 64) % ring.q, 0, 0, 0]


@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
@pytest.mark.parametrize("cast_dtype", [dtypes.weakint, dtypes.uint32, dtypes.uint64])
def test_polynomial_constant_preserves_explicit_cast(domain, cast_dtype):
    ring = mixed_prime_ring(128)[:3]
    value = -(1 << 200) + (1 << 32) + 7
    expected = value if cast_dtype is dtypes.weakint else value % (1 << (8 * cast_dtype.itemsize))
    constant = Poly(Node.const(value, cast_dtype).as_ring(ring, domain))
    data = Poly(list(range(ring.n)), ring)
    data = data.ntt() if domain is Domain.NTT else data
    result = (data + constant).realize().intt()
    assert result.tolist() == [expected % ring.q, *range(1, ring.n)]


def test_device_free_after_requires_storage_at_kernel_boundary():
    ring = large_prime_ring(4)
    target = Poly([1, 2, 3, 4], ring).realize()
    write = target.node.store(Poly.const(9, ring).node)
    value = Poly(Poly.const(7, ring).node.after(write))

    assert value.device is None
    realize(value, target + value * 0)
    assert value.tolist() == [7, 0, 0, 0]
    assert target.tolist() == [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match="kernel boundary requires storage"):
        (target + value).realize()
    assert (Poly(target.node.after(write)) + 7).realize().tolist() == [16, 0, 0, 0]


def test_realize_written_buffer_runs_store_once():
    ring = large_prime_ring(4)
    target = Poly([1, 2, 3, 4], ring).realize()
    written = Poly(target.node.after(target.node.store(Poly.const(9, ring).node)))
    written.realize()
    assert target.tolist() == [9, 0, 0, 0]

    Poly(target.node.after(target.node.store(Poly.const(3, ring).node))).realize()
    assert (written + 1).realize().tolist() == [4, 0, 0, 0]


@pytest.mark.parametrize("enabled", [False, True])
def test_realize_restores_gc_after_scheduling_failure(monkeypatch, enabled):
    source = Poly([1, 2, 3, 4], large_prime_ring(4))

    def fail(root):
        assert not gc.isenabled()
        raise RuntimeError("scheduling failed")

    monkeypatch.setattr(poly_module, "schedule", fail)
    was_enabled = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        with pytest.raises(RuntimeError, match="scheduling failed"):
            realize(source)
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if was_enabled else gc.disable)()


def test_realize_releases_old_graph_before_execution(monkeypatch):
    p = (Poly([1, 2, 3, 4], large_prime_ring(4)) + 1) * 3
    old_node = weakref.ref(p.node)
    run = poly_module.run_exec_units

    def execute(units):
        assert old_node() is None
        return run(units)

    monkeypatch.setattr(poly_module, "run_exec_units", execute)
    p.realize()
    assert p.tolist() == [6, 6, 9, 12]


def test_clone_outputs_have_independent_buffers():
    ring = large_prime_ring(4)
    p = Poly([1, 2, 3, 4], ring)

    c1 = p.clone()
    c2 = p.clone()
    realize(c1, c2)

    assert c1.tolist() == [1, 2, 3, 4]
    assert c2.tolist() == [1, 2, 3, 4]
    assert c1.node.base.buffer is not c2.node.base.buffer


@pytest.mark.parametrize("consume", [lambda value: value, function(lambda value: value)], ids=["eager", "function"])
def test_clone_with_unresolved_domain(consume):
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring)
    source = consume(x + x.ntt())
    assert source.domain is None

    clone = source.clone().realize()
    source.realize()

    assert clone.tolist() == [2, 4, 6, 8]
    assert clone.domain is Domain.NTT
    assert clone.node.base.buffer.base.nbytes == ring.nbytes
    assert clone.node.base.buffer.base is not source.node.base.buffer.base


def test_realized_poly_wrappers_are_not_tracked_for_node_rewrites():
    ring = large_prime_ring(4)
    base = Poly([1, 2, 3, 4], ring).realize()
    gc.collect()
    tracked = len(poly_module._pending_polys)

    wrappers = [Poly(base.node) for _ in range(64)]

    assert wrappers
    assert len(poly_module._pending_polys) == tracked

    unrelated = [base.rotate(i + 1) for i in range(4)]
    assert len(poly_module._pending_polys) == tracked + len(unrelated)

    lazy = base + 1
    assert len(poly_module._pending_polys) == tracked + len(unrelated) + 1
    lazy.realize()
    assert len(poly_module._pending_polys) == tracked + len(unrelated)


def test_binary_op_device_mismatch_raises_cleanly():
    ring = large_prime_ring(64)
    p_python = Poly([1, 2, 3, 4] + [0] * 60, ring, device="python")
    p_cuda = p_python.to("cuda:0")

    with pytest.raises(ValueError, match=r"polynomial device mismatch in binary op: python != cuda:0"):
        _ = p_python + p_cuda


@pytest.mark.parametrize("n", RING_DEGREES)
def test_zero(n: int):
    ring = mixed_prime_ring(n)
    p = Poly.zero(ring)
    coeffs = list(p)
    assert coeffs == [0] * n


def test_rand():
    ring = mixed_prime_ring(65536)

    # test with bound first
    b = 100
    q = ring.q
    p = Poly.uniform_on_interval(ring, bound=b)
    coeffs = list(p)
    assert all(c <= b or c >= q - b for c in coeffs)
    for i in range(0, b + 1):
        assert i in coeffs  # this has a (2b/(2b+1))^65536 chance of failing (1 in 2^471)
    for i in range(1, b + 1):
        assert q - i in coeffs  # this has a (2b/(2b+1))^65536 chance of failing (1 in 2^471)

    # test without bound
    p = Poly.uniform(ring)
    coeffs = list(p)
    coeffs_scaled = [int((c * 100) // q) for c in coeffs]
    assert all(0 <= c < 100 for c in coeffs_scaled)
    for i in range(100):
        assert i in coeffs_scaled  # this has a 0.99^65536 chance of failing (1 in 2^950)


def test_ternary():
    n = 65536
    ring = mixed_prime_ring(n)
    p = Poly.ternary(ring)
    coeffs = list(p)
    zero_count = coeffs.count(0)
    one_count = coeffs.count(1)
    negative_one_count = coeffs.count(ring.q - 1)
    assert zero_count + one_count + negative_one_count == n
    assert 0.48 * n <= zero_count <= 0.52 * n  # 1 in 2^80 chance of failing
    assert 0.23 * n <= one_count <= 0.27 * n  # 1 in 2^104 chance of failing
    assert 0.23 * n <= negative_one_count <= 0.27 * n  # 1 in 2^104 chance of failing


def test_sparse_ternary():
    n = 65536
    ring = mixed_prime_ring(n)
    p = Poly.sparse_ternary(ring, h=192)
    coeffs = list(p)
    zero_count = coeffs.count(0)
    one_count = coeffs.count(1)
    negative_one_count = coeffs.count(ring.q - 1)
    assert one_count + negative_one_count == 192
    assert zero_count + one_count + negative_one_count == n
    assert 36 <= one_count <= 156  # 1 in 2^62 chance of failing
    assert 36 <= negative_one_count <= 156  # 1 in 2^62 chance of failing


def test_discrete_gaussian_with_only_zero_in_support():
    ring = mixed_prime_ring(64)
    dist = DiscreteGaussian(ring.n, Fraction(1, 20))
    assert list(Poly.rand(dist, ring)) == [0] * ring.n


def test_discrete_gaussian():
    n = 65536
    ring = mixed_prime_ring(n)
    dist = DiscreteGaussian(n=n, stdev=Fraction(16, 5))
    p = Poly.rand(dist, ring)
    coeffs = list(p)
    zero_count = coeffs.count(0)
    one_count = coeffs.count(1)
    negative_one_count = coeffs.count(ring.q - 1)
    seven_count = coeffs.count(7)
    ten_count = coeffs.count(10)
    twenty_six_count = coeffs.count(26)
    maximum = max(min(x, ring.q - x) for x in coeffs)
    assert sum(min(x, ring.q - x) for x in coeffs) <= dist.coeff_1_norm_bound()
    # chance of failing should be around 2^-64
    assert 7390 <= zero_count <= 8950
    assert 6970 <= one_count <= 8500
    assert 6970 <= negative_one_count <= 8500
    assert 430 <= seven_count <= 910
    assert 3 <= ten_count <= 140
    assert twenty_six_count <= 1
    assert maximum <= 33


def test_large_discrete_gaussian():
    n = 65536
    stdev = Fraction(1 << 70, 1)
    ring = mixed_prime_ring(n)
    dist = DiscreteGaussian(n=n, stdev=stdev)
    p = Poly.rand(dist, ring)
    coeffs = list(p)
    coeffs_signed = [coeff if 2 * coeff < ring.q else coeff - ring.q for coeff in coeffs]
    # The construction reproduces stdev exactly (up to the negligible base-table
    # approximation), so for stdev >> 1 the coefficients are effectively N(0, stdev^2).
    mean = sum(coeffs_signed) / n
    rms = sqrt(sum(x * x for x in coeffs_signed) / n)
    max_abs = max(abs(x) for x in coeffs_signed)
    assert sum(map(abs, coeffs_signed)) <= dist.coeff_1_norm_bound()
    # The relative sampling error of the RMS is about 1/sqrt(2n), or 0.28%.
    assert 0.97 * stdev <= rms <= 1.03 * stdev
    # sample mean ~ N(0, stdev^2 / n), sd = stdev / 256
    assert abs(mean) <= 0.04 * stdev
    # max of n Gaussians concentrates around sqrt(2 ln n) ~ 4.7 stdev
    assert 3 * stdev <= max_abs <= 11 * stdev


def test_random_repeatability():
    n = 65536
    ring = mixed_prime_ring(n)
    p = Poly.ternary(ring)
    q = p + 0
    q.realize()
    assert not p.is_realized  # Reading p must replay the same random draw.
    assert list(p) == list(q)
    r = Poly.ternary(ring)
    assert list(p) != list(r)


@pytest.mark.parametrize("n,dtype", DEGREE_DTYPE_CASES)
def test_add(n: int, dtype):
    ring = mixed_prime_ring(n, dtype=dtype)
    p1 = Poly.uniform(ring)
    p2 = Poly.uniform(ring)
    p3 = p1 + p2
    assert list(p3) == [(a + b) % ring.q for a, b in zip(list(p1), list(p2))]


@pytest.mark.parametrize("n", RING_DEGREES)
def test_sub(n: int):
    ring = mixed_prime_ring(n)
    p1 = Poly.uniform(ring)
    p2 = Poly.uniform(ring)
    p3 = p1 - p2
    assert list(p3) == [(a - b) % ring.q for a, b in zip(list(p1), list(p2))]


@pytest.mark.parametrize("n", RING_DEGREES)
def test_neg(n: int):
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)
    p_neg = -p
    assert list(p_neg) == [(-c) % ring.q for c in list(p)]


@pytest.mark.parametrize("n,dtype", DEGREE_DTYPE_CASES)
def test_mul(n: int, dtype):
    ring = mixed_prime_ring(n, dtype=dtype)
    p1 = Poly.uniform(ring)
    p2 = Poly.uniform(ring)
    p3 = p1 * p2
    assert list(p3) == neg_convolve(list(p1), list(p2), n=ring.n, q=ring.q)


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("galois_elt", [-1, 3])
def test_automorphism(n: int, galois_elt):
    """Test Galois automorphism (rotation permutation)."""
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)
    p_auto = p.automorphism(galois_elt)

    # Verify automorphism: X -> X^k maps coeff[i] to coeff[(i * k) mod n]
    # with negation when (i * k) // n is odd
    expected = [0] * n
    for i, c in enumerate(list(p)):
        new_idx = (i * galois_elt) % n
        sgn = -1 if (i * galois_elt) % (2 * n) >= n else 1
        expected[new_idx] = (expected[new_idx] + sgn * c) % ring.q

    assert list(p_auto) == expected


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("galois_elt", [-1, 3])
def test_automorphism_ntt(n: int, galois_elt):
    """Test Galois automorphism (rotation permutation) in NTT domain."""
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)
    p_ntt_auto = p.ntt().automorphism(galois_elt).intt()
    p_coeff_auto = p.automorphism(galois_elt)
    assert list(p_ntt_auto) == list(p_coeff_auto)


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("k", [-314159, -16, 0, 3, 161038])
def test_rotate(n: int, k: int):
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)
    p_rotated = p.rotate(k)
    coeffs = list(p)
    s = k % n
    if (k % (2 * n)) < n:
        expected = [-coeff % ring.q for coeff in coeffs[n - s :]] + coeffs[: n - s]
    else:
        expected = coeffs[n - s :] + [-coeff % ring.q for coeff in coeffs[: n - s]]
    assert list(p_rotated) == expected


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("num_b", [1, 2, 3, 4])
@pytest.mark.parametrize("remove_from_front", [True, False])
def test_rescale(n: int, num_b: int, remove_from_front: bool):
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)
    target_moduli = ring.moduli[num_b:] if remove_from_front else ring.moduli[:-num_b]
    p_rescaled = p.rescale(target_moduli)
    assert list(p_rescaled) == rounded_rescale_coeffs(p, target_moduli)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("remove_from_front", [False, True])
@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
def test_rescale_single_prime_rounding_boundaries(dtype, remove_from_front: bool, domain: Domain):
    bits = 31 if dtype is dtypes.uint32 else 63
    ring = _ring(16, [bits, bits - 11, bits], dtype=dtype)
    removed = ring.moduli[0 if remove_from_front else -1]
    target_moduli = ring.moduli[1:] if remove_from_front else ring.moduli[:-1]
    half = removed // 2
    coefficients = [
        half - 1,
        half,
        half + 1,
        -half,
        -half - 1,
        removed + half,
        removed + half + 1,
        -removed - half,
        -removed - half - 1,
        0,
        1,
        -1,
        removed - 1,
        removed,
        removed + 1,
        -removed,
    ]
    source = Poly(coefficients, ring)
    if domain is Domain.NTT:
        source = source.ntt().realize()

    expected = [((coefficient + half) // removed) % prod(target_moduli) for coefficient in coefficients]
    assert source.rescale(target_moduli).tolist() == expected


@pytest.mark.parametrize("n", RING_DEGREES)
def test_rescale_non_contiguous_subset(n: int):
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)
    target_moduli = ring.moduli[::2]
    assert list(p.rescale(target_moduli)) == rounded_rescale_coeffs(p, target_moduli)


def test_rescale_chain_preserves_intermediate_rounding():
    ring = mixed_prime_ring(256, dtype=dtypes.uint32)[:6]
    middle_ring = ring[2:4]
    low_ring = ring[:2]
    coeffs = [(ring.q // 3 + index * (ring.q // 257)) % ring.q for index in range(ring.n)]

    middle = Poly(coeffs, ring).rescale(middle_ring.moduli).realize()
    expected = list(middle.rescale(low_ring.moduli))
    direct = list(Poly(coeffs, ring).rescale(low_ring.moduli))
    actual = list(Poly(coeffs, ring).rescale(middle_ring.moduli).rescale(low_ring.moduli))

    assert direct != expected
    assert actual == expected


@pytest.mark.parametrize("n", RING_DEGREES)
def test_rescale_mixed_basis(n: int):
    full_ring = mixed_prime_ring(n)
    ring = RingSpec(n=n, moduli=full_ring.moduli[:4], dtype=full_ring.dtype)
    added = next(modulus for modulus in large_prime_ring(n).moduli if modulus not in ring.moduli)
    target_moduli = (added, ring.moduli[0], ring.moduli[2])
    p = Poly.uniform(ring)
    assert list(p.rescale(target_moduli)) == rounded_rescale_coeffs(p, target_moduli)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("residue", [Residue.BALANCED, Residue.NONNEGATIVE])
def test_base_change_from_larger_source_primes(dtype, residue):
    bits = (31, 20) if dtype is dtypes.uint32 else (60, 40)
    full = _ring(1024, [bits[0]] * 2 + [bits[1]] * 3, dtype=dtype)
    source_ring = full[:2]
    values = [((index * 17 + 23) * source_ring.q // 2053) % source_ring.q for index in range(full.n)]
    source = Poly(values, source_ring).realize()
    representatives = [value - source_ring.q if residue is Residue.BALANCED and value > source_ring.q // 2 else value for value in values]

    for destination in (full, full[2:], RingSpec(full.n, (full.moduli[3], full.moduli[0]), dtype)):
        actual = source._to_ring(destination, residue=residue).realize()
        assert actual.tolist() == [value % destination.q for value in representatives]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("residue", [Residue.BALANCED, Residue.NONNEGATIVE])
def test_base_change_accumulates_many_wide_partials(dtype, residue):
    bits = 31 if dtype is dtypes.uint32 else 60
    full = _ring(1024, [bits] * 23, dtype=dtype)
    source_ring, destination = full[:20], full[20:]
    values = [((index * 17 + 23) * source_ring.q // 2053) % source_ring.q for index in range(full.n)]
    source = Poly(values, source_ring).realize()
    representatives = [value - source_ring.q if residue is Residue.BALANCED and value > source_ring.q // 2 else value for value in values]

    actual = source._to_ring(destination, residue=residue).realize()
    assert actual.tolist() == [value % destination.q for value in representatives]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_rescale_from_larger_removed_primes(dtype):
    bits = (31, 20) if dtype is dtypes.uint32 else (60, 40)
    full = _ring(1024, [bits[0]] * 2 + [bits[1]] * 3, dtype=dtype)
    ring = full[:4]
    values = [((index * 17 + 23) * ring.q // 2053) % ring.q for index in range(ring.n)]
    source = Poly(values, ring).realize()

    for destination in (ring[2:], RingSpec(ring.n, (full.moduli[4], full.moduli[2]), dtype)):
        assert source.rescale(destination.moduli).tolist() == rounded_rescale_coeffs(source, destination.moduli)


@pytest.mark.parametrize("n", RING_DEGREES)
def test_to_ring_identity(n: int):
    ring = mixed_prime_ring(n)
    p1 = Poly.uniform(ring)
    p2 = p1._to_ring(ring)
    assert list(p2) == list(p1)


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("sl", [slice(1, -1)])
def test_to_ring_slice(n: int, sl: slice):
    ring = mixed_prime_ring(n)
    new_ring = ring[sl]
    p1 = Poly.uniform(ring)
    p2 = p1.to_quotient_ring(new_ring)
    assert list(p2) == [c % new_ring.q for c in list(p1)]


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("sl", [slice(None, 2)])
def test_to_ring_extend(n: int, sl: slice):
    ring_full = mixed_prime_ring(n)
    ring_small = ring_full[sl]
    p_small = Poly.uniform(ring_small)
    p_full = p_small.lift_to_ring(ring_full)
    assert centered_coeffs(p_full) == centered_coeffs(p_small)


@pytest.mark.parametrize("n,dtype", DEGREE_DTYPE_CASES)
def test_to_ring_arbitrary(n: int, dtype):
    ring = mixed_prime_ring(n, dtype=dtype)
    p1 = Poly.uniform(ring)
    new_ring = large_prime_ring(n, dtype=dtype)
    assert new_ring.q // ring.q > 2**240
    p2 = p1._to_ring(new_ring)
    assert centered_coeffs(p1) == centered_coeffs(p2)


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("sl", [slice(None, 3)])
def test_to_ring_drop_and_extend(n: int, sl: slice):
    """Test that drop + extend is lossless for values within the smaller range."""
    ring_full = mixed_prime_ring(n)
    ring_small = ring_full[sl]
    p_small = Poly.uniform(ring_full).to_quotient_ring(ring_small)
    p_roundtrip = p_small.lift_to_ring(ring_full).to_quotient_ring(ring_small)
    assert list(p_small) == list(p_roundtrip)


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("sl", [slice(None, None, 2)])  # non-contiguous subset
def test_to_ring_subset(n: int, sl: slice):
    """Test non-contiguous subset: [p0,p1,p2,p3,p4,...] -> [p0,p2,p4,...]."""
    ring = mixed_prime_ring(n)
    p1 = Poly.uniform(ring)
    new_ring = ring[sl]
    p2 = p1.to_quotient_ring(new_ring)
    assert list(p2) == [c % new_ring.q for c in list(p1)]


@pytest.mark.parametrize("n", RING_DEGREES)
@pytest.mark.parametrize("sl", [slice(None, None, 2)])  # non-contiguous subset
def test_to_ring_subset_from_ntt(n, sl: slice):
    """Test subset from NTT domain - should be domain-agnostic."""
    ring = mixed_prime_ring(n)
    p1 = Poly.uniform(ring).ntt()  # Start in NTT domain
    new_ring = ring[sl]
    p2 = p1.to_quotient_ring(new_ring)
    assert list(p2) == [c % new_ring.q for c in list(p1)]  # list() calls intt() internally


@pytest.mark.parametrize("device", ["cpu", " PYTHON "])
def test_poly_to_normalizes_device_aliases(device):
    poly = Poly([1, 2, 3, 4], large_prime_ring(4), device="python")
    moved = poly.to(device)
    assert moved.device == "python"
    assert moved is poly
    assert (poly + moved).tolist() == [2, 4, 6, 8]


def test_poly_to_inserts_copy_ir():
    ring = mixed_prime_ring(4)
    p = Poly([1, 2, 3, 4], ring, device="python")
    p2 = p.to("cuda:0")

    # Device annotations
    assert p.device == "python"
    assert p2.device == "cuda:0"

    # IR shape: COPY(TO_RNS(BUFFER), DEVICE("cuda:0"))
    # Ring/domain metadata propagates through COPY automatically
    n = p2.node
    assert n.op is Op.COPY
    src_node, dev_node = n.src
    assert dev_node.op is Op.DEVICE
    assert dev_node.arg == "cuda:0"

    # Coefficients stay compact until the target backend materializes RNS.
    assert src_node.op is Op.TO_RNS
    src_buf = src_node.src[0]
    assert src_buf.op is Op.BUFFER
    assert src_buf.dtype is dtypes.uint32
    assert src_buf.device == "python"

    # Ring/domain propagate through COPY
    assert n.ring == p.ring
    assert n.domain == p.domain


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_poly_coefficients_materialize_to_rns(dtype):
    ring = mixed_prime_ring(16, dtype=dtype)
    coeffs = np.array([0, 1, -1, 17, -29, 2**20, -(2**20), 1234567] * 2, dtype=np.int64)

    p = Poly(coeffs, ring)

    assert p.domain == Domain.COEFF
    assert list(p) == [int(c) % ring.q for c in coeffs]


def test_poly_accepts_wide_signed_coefficients():
    ring = mixed_prime_ring(4)
    coeffs = [0, 2**100, -(2**100), 4 + 10 * ring.q]

    p = Poly(coeffs, ring)

    assert list(p) == [c % ring.q for c in coeffs]


@pytest.mark.parametrize(
    "np_dtype,bits",
    [(np.float64, bits) for bits in [0, 31, 32, 53, 63, 64, 100, 320, 1023]]
    + [(np.float16, 10), (np.float32, 40), (np.float32, 100), (np.longdouble, 200)],
)
def test_float_coefficient_limbs_match_python_integers(np_dtype, bits: int):
    rng = np.random.default_rng(954)
    data = np.rint(np.ldexp(rng.uniform(-1, 1, 64).astype(np_dtype), bits))
    boundary = np_dtype(2) ** bits
    data[:8] = np.rint([0, -0.0, boundary, -boundary, np.nextafter(boundary, np_dtype(0)), np.nextafter(-boundary, np_dtype(0)), 1, -1])
    integers = list(map(int, data))
    signed_bits = max((value if value >= 0 else ~value).bit_length() + 1 for value in integers)
    width = 4 * ((signed_bits + 31) // 32)

    for coefficients in (data, data.astype(data.dtype.newbyteorder(">"))):
        limbs = poly_module._pack_signed_u32_limbs(coefficients)
        assert limbs.T.copy().tobytes() == b"".join(value.to_bytes(width, "little", signed=True) for value in integers)


@pytest.mark.parametrize("value", [1.5, np.nan, np.inf, -np.inf])
def test_float_coefficient_packing_rejects_nonintegers(value: float):
    with pytest.raises(ValueError, match="finite integers"):
        Poly(np.array([value, 0, 0, 0]), mixed_prime_ring(4))


@pytest.mark.parametrize(
    "shape,error",
    [
        ((3,), "data length .* does not match degree 4"),
        ((), "one-dimensional"),
        ((2, 2), "one-dimensional"),
        ((4, 1), "one-dimensional"),
        ((4, 2), "one-dimensional"),
    ],
)
def test_poly_coefficients_reject_invalid_shape(shape, error):
    with pytest.raises(ValueError, match=error):
        Poly(np.zeros(shape), mixed_prime_ring(4))


@pytest.mark.parametrize("device", ["python", pytest.param("cuda:0", marks=pytest.mark.requires_cuda)])
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_float_coefficients_materialize_to_rns(device: str, dtype):
    ring = mixed_prime_ring(4, dtype=dtype)
    for values in ([0, 1, -1, 2**40], [0, 2**63, -(2**63), 2**100]):
        data = np.array(values, dtype=np.float64)
        p = Poly(data, ring, device=device)
        data.fill(0)

        assert list(p) == [value % ring.q for value in values]


def test_poly_constructor_rejects_complex_coefficients():
    with pytest.raises(TypeError, match="must be integers"):
        Poly(np.full(4, 1 + 1j), mixed_prime_ring(4))


def test_poly_coefficients_schedule_to_rns():
    ring = mixed_prime_ring(16)
    p = Poly(np.arange(ring.n, dtype=np.int64), ring)

    linear, _ = schedule(Node.sink(p.node))

    assert any(call_op(call) is Op.TO_RNS for call in linear.src)


FROM_RNS_DEVICES = ["python", pytest.param("cuda:0", marks=pytest.mark.requires_cuda)]


def test_poly_coefficients_schedule_from_rns():
    """Reading coefficients of a computed poly schedules an Op.FROM_RNS kernel.

    A computed source (here a sum) drives the real reconstruction; the scheduler
    materializes the RNS input internally (conversion sources are bufferized).
    """
    ring = mixed_prime_ring(16)
    p = Poly(np.arange(ring.n, dtype=np.int64), ring)

    frn = (p + p).node.from_rns()
    linear, _ = schedule(Node.sink(frn))

    assert any(call_op(call) is Op.FROM_RNS for call in linear.src)


@pytest.mark.parametrize("device", FROM_RNS_DEVICES)
def test_from_rns_materializes_raw_signed_limbs(device):
    """FROM_RNS on `device` produces little-endian signed two's-complement uint32 limbs.

    Exercises the per-device kernel (the python handler and the CUDA
    ren_from_rns_limbs kernel) over a computed poly in a single realize, then
    decodes the limbs back to the balanced [-Q/2, Q/2) representative -- the
    same signed encoding TO_RNS consumes.
    """
    ring = mixed_prime_ring(16)
    q = ring.q
    coeffs = np.array([0, 1, -1, 17, -29, 2**20, -(2**20), 1234567] * 2, dtype=np.int64)
    p = Poly(coeffs, ring, device=device)

    frn = (p + p).node.from_rns()
    limb_count = from_rns_limb_count(ring)
    assert frn.dtype is dtypes.uint32

    fr = Poly(frn).to("python").contiguous()
    fr.realize()

    raw = np.array(fr.node.buffer.as_typed_buffer(), copy=False)
    assert raw.dtype == np.uint32
    raw = raw.reshape(limb_count, ring.n).astype(object)

    x = np.zeros(ring.n, dtype=object)
    for limb in reversed(list(raw)):
        x = x * (1 << 32) + limb
    modulus = 1 << (32 * limb_count)
    neg_threshold = modulus >> 1
    x = np.where(x >= neg_threshold, x - modulus, x)

    expected = [((2 * int(c)) % q + q // 2) % q - q // 2 for c in coeffs]
    assert x.tolist() == expected


@pytest.mark.parametrize("n,dtype", DEGREE_DTYPE_CASES)
def test_ntt_intt_roundtrip(n: int, dtype):
    """Test that NTT -> INTT is identity."""
    ring = mixed_prime_ring(n, dtype=dtype)
    p = Poly.uniform(ring)
    # Materialize the forward transform so the inverse can't cancel it in the graph.
    p_ntt = p.ntt().realize()
    p_back = p_ntt.intt()
    assert list(p_back) == list(p)


@pytest.mark.requires_cuda
def test_u32_cuda_dense_convolution_ntt_layout():
    n = 65536
    ring = mixed_prime_ring(n, dtype=dtypes.uint32)
    coeffs = [((i * 17 + 3) % 2001) - 1000 for i in range(n)]
    p = Poly(coeffs, ring)

    terms = ((7, 23), (5, 1234), (-11, 40000))
    sparse_coeffs = [0] * n
    for c, k in terms:
        sparse_coeffs[k] = c % ring.q
    q = Poly(sparse_coeffs, ring)

    actual = list(p * q)

    def expected_at(i: int) -> int:
        total = 0
        for c, k in terms:
            raw = i - k
            src = raw % n
            sign = -1 if raw % (2 * n) >= n else 1
            total += c * sign * coeffs[src]
        return total % ring.q

    assert actual == [expected_at(i) for i in range(n)]


@pytest.mark.parametrize("n", [256])
@pytest.mark.parametrize("num_b", [1, 2])
@pytest.mark.parametrize("remove_from_front", [True, False])
def test_u32_rescale(n: int, num_b: int, remove_from_front: bool):
    ring = mixed_prime_ring(n, dtype=dtypes.uint32)
    p = Poly.uniform(ring)
    target_moduli = ring.moduli[num_b:] if remove_from_front else ring.moduli[:-num_b]
    p_rescaled = p.rescale(target_moduli)
    assert list(p_rescaled) == rounded_rescale_coeffs(p, target_moduli)


@pytest.mark.parametrize("n", RING_DEGREES)
def test_scale_by_primes(n: int):
    """Test per-prime limb scaling (used in CKKS keyswitching)."""
    ring = mixed_prime_ring(n)
    p = Poly.uniform(ring)

    scalars = tuple(i + 2 for i in range(len(ring.moduli)))
    scaled = list(p.scale_by_primes(scalars))

    # The residues at all primes uniquely identify each coefficient in [0, Q).
    for actual, coefficient in zip(scaled, p, strict=True):
        assert 0 <= actual < ring.q
        for prime, scalar in zip(ring.moduli, scalars, strict=True):
            assert actual % prime == coefficient * scalar % prime


def test_const():
    const = 123456789
    ring = mixed_prime_ring(256)
    c = Poly.const(const, ring)
    assert list(c) == [const % ring.q] + (ring.n - 1) * [0]


def test_poly_plus_const():
    const = 123456789
    ring = mixed_prime_ring(256)
    p = Poly.uniform(ring)
    c = Poly.const(const, ring)
    expected = list(p)
    expected[0] = (expected[0] + const) % ring.q
    assert list(p + c) == expected
    assert list(c + p) == expected


def test_const_plus_const():
    left, right = (123456789, 987654321)
    ring = mixed_prime_ring(256)
    c1 = Poly.const(left, ring)
    c2 = Poly.const(right, ring)
    assert list(c1 + c2) == [(left + right) % ring.q] + (ring.n - 1) * [0]


def test_poly_minus_const():
    const = 123456789
    ring = mixed_prime_ring(256)
    p = Poly.uniform(ring)
    c = Poly.const(const, ring)
    expected = list(p)
    expected[0] = (expected[0] - const) % ring.q
    assert list(p - c) == expected


def test_const_minus_poly():
    const = 123456789
    ring = mixed_prime_ring(256)
    p = Poly.uniform(ring)
    c = Poly.const(const, ring)
    expected = [-coeff % ring.q for coeff in list(p)]
    expected[0] = (const + expected[0]) % ring.q
    assert list(c - p) == expected


def test_const_minus_const():
    left, right = (123456789, 987654321)
    ring = mixed_prime_ring(256)
    c1 = Poly.const(left, ring)
    c2 = Poly.const(right, ring)
    expected = [(left - right) % ring.q] + (ring.n - 1) * [0]
    assert list(c1 - c2) == expected


def test_poly_times_const():
    const = 123456789
    ring = mixed_prime_ring(256)
    p = Poly.uniform(ring)
    c = Poly.const(const, ring)
    expected = [(coeff * const) % ring.q for coeff in list(p)]
    assert list(p * c) == expected
    assert list(c * p) == expected


def test_const_times_const():
    left, right = (123456789, 987654321)
    ring = mixed_prime_ring(256)
    c1 = Poly.const(left, ring)
    c2 = Poly.const(right, ring)
    expected = [(left * right) % ring.q] + (ring.n - 1) * [0]
    assert list(c1 * c2) == expected


@pytest.mark.parametrize("const", [123456789, 987654321])
def test_const_neg(const: int):
    ring = mixed_prime_ring(256)
    c = Poly.const(const, ring)
    expected = [-const % ring.q] + (ring.n - 1) * [0]
    assert list(-c) == expected


@pytest.mark.parametrize("const", [3, -4])
def test_const_automorphism(const: int):
    galois_elt = 123
    ring = mixed_prime_ring(256)
    c = Poly.const(const, ring)
    assert list(c.automorphism(galois_elt)) == [const % ring.q] + (ring.n - 1) * [0]


def test_const_rescale():
    const = 3**100
    ring = mixed_prime_ring(256)
    p1 = Poly.const(const, ring)
    p2 = p1.rescale(ring.moduli[:-2])
    b = prod(ring.moduli[-2:])
    assert list(p2) == [round(Fraction(const, b)) % p2.ring.q] + (ring.n - 1) * [0]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
@pytest.mark.parametrize(
    "target_moduli",
    [(17, 41), (41, 73), (73, 17), (17, 89), (89,)],
    ids=["prefix", "suffix", "non-contiguous-reordered", "mixed", "disjoint"],
)
def test_const_rescale_matches_materialized(dtype, domain: Domain, target_moduli: tuple[int, ...]):
    ring = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtype)
    target = RingSpec(n=ring.n, moduli=target_moduli, dtype=dtype)
    halfway = ring.q // (2 * target.q)
    values = (0, halfway, halfway + 1, -halfway, -halfway - 1, 2000, -2000, ring.q + 2000, 3**100)

    for value in values:
        constant = Poly.const(value, ring)
        materialized = Poly([value, 0, 0, 0], ring)
        if domain is Domain.NTT:
            constant, materialized = constant.ntt(), materialized.ntt()

        result = constant.rescale(target_moduli)
        expected = round(Fraction(value * target.q, ring.q)) % target.q
        assert result.node.const_val is not None
        assert result.node.const_val % target.q == expected

        linear, becomes_map = schedule(Node.sink(result.node))
        assert linear.op is Op.LINEAR and not linear.src
        folded = becomes_map[result.node]
        assert folded.base.op is Op.CONST
        assert folded.ring == target
        assert folded.domain is domain
        assert list(result) == list(materialized.rescale(target_moduli)) == [expected, 0, 0, 0]


@pytest.mark.parametrize("sign", [1, -1])
def test_const_rescale_chain_preserves_intermediate_rounding(sign: int):
    ring = RingSpec(n=4, moduli=(17, 41, 73))
    value = sign * 4488
    constant = Poly.const(value, ring)
    result = constant.rescale((41,)).rescale((17,))
    direct = constant.rescale((17,))
    materialized = Poly([value, 0, 0, 0], ring).rescale((41,)).realize()

    assert result.node.const_val == sign * 2
    assert direct.node.const_val == sign
    assert list(result) == list(materialized.rescale((17,))) == [(sign * 2) % 17, 0, 0, 0]
    linear, _ = schedule(Node.sink(result.node))
    assert linear.op is Op.LINEAR and not linear.src


@pytest.mark.parametrize("k", [-5, 0, 1, 12, 200, 256, 257, -256])
def test_const_rotate(k: int):
    const = 3
    ring = mixed_prime_ring(256)
    q = ring.q
    c = Poly.const(const, ring)
    result = list(c.rotate(k))
    pos = k % ring.n
    sign = -1 if k % (2 * ring.n) >= ring.n else 1
    assert result == [0] * pos + [(sign * const) % q] + [0] * (ring.n - pos - 1)


def test_const_to_ring_identity():
    const = 123456789
    ring = mixed_prime_ring(256)
    p1 = Poly.const(const, ring)
    p2 = p1._to_ring(ring)
    assert list(p2) == [const % ring.q] + (ring.n - 1) * [0]


def test_const_to_ring_slice():
    const = 3**200
    ring = mixed_prime_ring(256)
    p1 = Poly.const(const, ring)
    new_ring = ring[1:-1]
    new_q = new_ring.q
    p2 = p1.to_quotient_ring(new_ring)
    assert list(p2) == [const % new_q] + (ring.n - 1) * [0]


def test_const_to_ring_extend():
    const = 3**200
    ring_full = mixed_prime_ring(256)
    ring_small = ring_full[1:-1]
    p_small = Poly.const(const, ring_small)
    p_full = p_small.lift_to_ring(ring_full)
    centered_const = ((const + ring_small.q // 2) % ring_small.q) - ring_small.q // 2
    assert list(p_full) == [centered_const % ring_full.q] + (ring_full.n - 1) * [0]


@pytest.mark.parametrize(
    "value, residue, expected",
    [
        (-1, Residue.BALANCED, 696),
        (16, Residue.BALANCED, 696),
        (-1, Residue.NONNEGATIVE, 16),
        (16, Residue.NONNEGATIVE, 16),
        (8, Residue.BALANCED, 8),
        (9, Residue.BALANCED, 689),
        (33, Residue.NONNEGATIVE, 16),
    ],
)
def test_const_lift_to_ring(value: int, residue: Residue, expected: int):
    ring = RingSpec(n=4, moduli=(17,))
    target = RingSpec(n=4, moduli=(17, 41))
    result = Poly.const(value, ring).lift_to_ring(target, residue)
    materialized = Poly([value, 0, 0, 0], ring).lift_to_ring(target, residue)

    assert result.tolist() == materialized.tolist() == [expected, 0, 0, 0]
    linear, becomes_map = schedule(Node.sink(result.node))
    folded = becomes_map[result.node]
    assert not linear.src
    assert folded.base.op is Op.CONST
    assert folded.ring == target
    assert Poly(folded).tolist() == [expected, 0, 0, 0]


def test_const_lift_after_quotient_preserves_intermediate_modulus():
    ring = RingSpec(n=4, moduli=(17, 41))
    quotient = ring[:1]
    result = Poly.const(33, ring).to_quotient_ring(quotient).lift_to_ring(ring, Residue.BALANCED)
    materialized = Poly([33, 0, 0, 0], ring).to_quotient_ring(quotient).realize()

    assert result.tolist() == materialized.lift_to_ring(ring, Residue.BALANCED).tolist() == [696, 0, 0, 0]
    linear, becomes_map = schedule(Node.sink(result.node))
    folded = becomes_map[result.node]
    assert not linear.src
    assert folded.base.op is Op.CONST
    assert folded.ring == ring
    assert Poly(folded).tolist() == [696, 0, 0, 0]


def test_const_to_ring_arbitrary():
    const = 3**200
    ring = mixed_prime_ring(256)
    p1 = Poly.const(const, ring)
    new_ring = large_prime_ring(ring.n)
    p2 = p1._to_ring(new_ring)
    centered_const = ((const + ring.q // 2) % ring.q) - ring.q // 2
    assert list(p2) == [centered_const % new_ring.q] + (ring.n - 1) * [0]


@pytest.mark.parametrize("k", [-5, 0, 1, 12, 200, 256, 257, -256])
def test_monomial(k: int):
    const = 3
    ring = mixed_prime_ring(256)
    q = ring.q
    m = Poly.monomial(const, k, ring)
    pos = k % ring.n
    sign = -1 if k % (2 * ring.n) >= ring.n else 1
    assert list(m) == [0] * pos + [(sign * const) % q] + [0] * (ring.n - pos - 1)


def test_monomial_plus_const():
    ring = mixed_prime_ring(256)
    m = Poly.monomial(3, 10, ring)
    c = Poly.const(4, ring)
    expected = [4] + [0] * 9 + [3] + [0] * (ring.n - 2 - 9)
    assert list(m + c) == expected
    assert list(c + m) == expected
    m_const = Poly.monomial(3, 0, ring)
    expected = [7] + (ring.n - 1) * [0]
    assert list(c + m_const) == expected
    assert list(m_const + c) == expected


def test_monomial_plus_monomial():
    ring = mixed_prime_ring(256)
    m1 = Poly.monomial(3, 10, ring)
    m2 = Poly.monomial(4, 2, ring)
    expected = [0] * 2 + [4] + [0] * 7 + [3] + [0] * (ring.n - 2 - 9)
    assert list(m1 + m2) == expected
    assert list(m2 + m1) == expected
    m3 = Poly.monomial(5, 2, ring)
    expected = [0] * 2 + [9] + [0] * (ring.n - 3)
    assert list(m2 + m3) == expected
    assert list(m3 + m2) == expected
    m4 = Poly.monomial(6, 260, ring)
    expected = [0, 0, 5, 0, ring.q - 6] + [0] * (ring.n - 5)
    assert list(m3 + m4) == expected
    assert list(m4 + m3) == expected


def test_monomial_plus_poly():
    ring = mixed_prime_ring(256)
    p = Poly.uniform(ring)
    c = Poly.monomial(3, 10, ring)
    expected = list(p)
    expected[10] += 3
    expected[10] %= ring.q
    assert list(p + c) == expected
    assert list(c + p) == expected


def test_monomial_minus_const():
    ring = mixed_prime_ring(256)
    q = ring.q
    m = Poly.monomial(3, 10, ring)
    c = Poly.const(4, ring)
    expected = [q - 4] + [0] * 9 + [3] + [0] * (ring.n - 2 - 9)
    assert list(m - c) == expected
    assert list(c - m) == [-coeff % q for coeff in expected]
    m_const = Poly.monomial(3, 0, ring)
    expected = [1] + (ring.n - 1) * [0]
    assert list(c - m_const) == expected
    assert list(m_const - c) == [-coeff % q for coeff in expected]


def test_monomial_minus_monomial():
    ring = mixed_prime_ring(256)
    q = ring.q
    m1 = Poly.monomial(3, 10, ring)
    m2 = Poly.monomial(4, 2, ring)
    expected = [0] * 2 + [q - 4] + [0] * 7 + [3] + [0] * (ring.n - 2 - 9)
    assert list(m1 - m2) == expected
    assert list(m2 - m1) == [-coeff % q for coeff in expected]
    m3 = Poly.monomial(5, 2, ring)
    expected = [0] * 2 + [q - 1] + [0] * (ring.n - 3)
    assert list(m2 - m3) == expected
    assert list(m3 - m2) == [-coeff % q for coeff in expected]
    m4 = Poly.monomial(6, 260, ring)
    expected = [0, 0, 5, 0, 6] + [0] * (ring.n - 5)
    assert list(m3 - m4) == expected
    assert list(m4 - m3) == [-coeff % q for coeff in expected]


def test_monomial_minus_poly():
    ring = mixed_prime_ring(256)
    p = Poly.uniform(ring)
    c = Poly.monomial(3, 10, ring)
    expected = list(p)
    expected[10] -= 3
    expected[10] %= ring.q
    assert list(p - c) == expected
    assert list(c - p) == [-coeff % ring.q for coeff in expected]


def test_monomial_times_const():
    ring = mixed_prime_ring(256)
    c = Poly.monomial(3, 10, ring)
    expected = [0] * 10 + [12] + [0] * (ring.n - 10 - 1)
    assert list(4 * c) == expected
    assert list(c * 4) == expected


def test_monomial_times_monomial():
    ring = mixed_prime_ring(256)
    c1 = Poly.monomial(3, 5, ring)
    c2 = Poly.monomial(4, 6, ring)
    c3 = Poly.monomial(12, 11, ring)
    assert list(c1 * c2) == list(c3)
    assert list(c2 * c1) == list(c3)


def test_monomial_times_poly():
    ring = mixed_prime_ring(256)
    q = ring.q
    p = Poly.uniform(ring)
    c = Poly.monomial(3, 10, ring)
    expected = [-3 * coeff % q for coeff in list(p)[-10:]] + [3 * coeff % q for coeff in list(p)[:-10]]
    assert list(c * p) == expected
    assert list(p * c) == expected


def test_monomial_neg():
    ring = mixed_prime_ring(256)
    c1 = Poly.monomial(3, 10, ring)
    c2 = Poly.monomial(-3, 10, ring)
    assert list(-c1) == list(c2)
    assert list(c1) == list(-c2)


@pytest.mark.parametrize("exponent", [3, -4, 0, 100, 300])
def test_monomial_automorphism(exponent: int):
    galois_elt = 123
    ring = mixed_prime_ring(256)
    q = ring.q
    c = Poly.monomial(3, exponent, ring)
    nonzero_index = (exponent * galois_elt) % ring.n
    flipped = ((exponent * galois_elt) % (2 * ring.n)) >= ring.n
    coeff = q - 3 if flipped else 3
    expected = [0] * nonzero_index + [coeff] + [0] * (ring.n - 1 - nonzero_index)
    assert list(c.automorphism(galois_elt)) == expected


def test_monomial_rescale():
    const = 3**100
    ring = mixed_prime_ring(256)
    p1 = Poly.monomial(const, 10, ring)
    p2 = p1.rescale(ring.moduli[:-2])
    b = prod(ring.moduli[-2:])
    assert list(p2) == [0] * 10 + [round(Fraction(const, b)) % p2.ring.q] + (ring.n - 1 - 10) * [0]


@pytest.mark.parametrize("r", [-200, -5, 0, 6, 10, 100, 300])
def test_monomial_rotate(r: int):
    ring = mixed_prime_ring(256)
    p = Poly.monomial(3, 10, ring)
    assert list(p.rotate(r)) == list(Poly.monomial(3, 10 + r, ring))


def test_equals():
    n = 4
    ring = RingSpec(n, (17, 41))
    ring2 = RingSpec(n, (17, 73))

    # Test that two polys with different rings and same coefficients are not equal
    p1 = Poly([0, 1, 0, 1], ring)
    p2 = Poly([0, 1, 0, 1], ring2)
    assert not p1.equals(p2)

    # Test that two polys with the same ring and different coefficients are not equal
    p3 = Poly([0, 1, 0, 1], ring)
    p4 = Poly([0, 2, 0, 1], ring)
    assert not p3.equals(p4)

    # Test that two polys with the same ring and same coefficients are equal
    p5 = Poly([0, 1, 0, 1], ring)
    p6 = Poly([0, 1, 0, 1], ring)
    assert p5.equals(p6)

    # Test that equality works across domains
    p7 = Poly([0, 1, 0, 1], ring)
    p8 = Poly([0, 1, 0, 1], ring).ntt()
    assert p7.equals(p8)

    p9 = Poly([1, 0, 0, 0], ring).ntt()
    p10 = Poly([1, 1, 1, 1], ring)
    assert not p9.equals(p10)

    # Test that coefficient overflows are handled
    p11 = Poly([1, 2, 3, 4], ring)
    p12 = Poly([1, 2, 3, 4 + 10 * ring.q], ring)
    assert p11.equals(p12)


@pytest.mark.parametrize("convert", [Poly.ntt, Poly.intt], ids=["ntt", "intt"])
def test_domain_conversion_preserves_realized_poly(convert):
    poly = convert(Poly([1, 2, 3, 4], large_prime_ring(4))).realize()

    assert convert(poly) is poly


@pytest.mark.parametrize("domain", [Domain.NTT, Domain.COEFF])
def test_domain_conversion_preserves_lazy_request(domain):
    ring = mixed_prime_ring(4)
    coeffs = [ring.moduli[-1] * i for i in range(1, 5)]
    source = Poly(coeffs, ring)
    if domain is Domain.NTT:
        result = source.ntt().rescale(ring.moduli[:-1]).ntt()
        expected = [1, 2, 3, 4]
    else:
        result = (source * source).intt()
        expected = neg_convolve(coeffs, coeffs, n=ring.n, q=ring.q)

    result.realize()
    assert result.domain is domain
    assert result.tolist() == expected


def test_outputs_keep_independent_domains():
    ring = large_prime_ring(4)

    def split_domains(x: Poly) -> tuple[Poly, Poly]:
        return x, x.ntt()

    coeff, ntt = split_domains(Poly([1, 2, 3, 4], ring, device="python"))
    realize(coeff, ntt)

    assert coeff.domain is Domain.COEFF
    assert ntt.domain is Domain.NTT
    assert coeff.tolist() == ntt.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "dtype,prime_sizes",
    [
        (dtypes.uint32, [31]),
        (dtypes.uint32, [29, 28]),
        (dtypes.uint32, [31, 30]),
        (dtypes.uint64, [17]),
        (dtypes.uint64, [16, 16]),
        (dtypes.uint64, [21, 21, 40]),
        (dtypes.uint64, [40, 39]),
        (dtypes.uint64, [60, 40]),
        (dtypes.uint64, [62, 40]),
        (dtypes.uint64, [63, 62, 40]),
        pytest.param(dtypes.uint32, [30] * 112, id="large-basis"),
    ],
)
@pytest.mark.parametrize("input_bits", [32, 64, None], ids=["one-limb", "two-limb", "wide"])
def test_signed_rns_conversion_at_word_boundary(dtype, prime_sizes, input_bits):
    ring = _ring(32, prime_sizes, dtype=dtype)
    half = ring.q // 2
    if input_bits is None:
        values = [-half, -half + 1, -(1 << 64), -1, 0, 1, 1 << 64, half]
        values += [-q + 1 for q in ring.moduli[:3]] + [q - 1 for q in ring.moduli[:3]]
        values += [-(1 << 32), 1 << 32, -(1 << 31), (1 << 31) - 1]
    else:
        bound = 1 << (input_bits - 1)
        values = [-bound, -bound + 1, -1, 0, 1, bound - 1]
    values += [0] * (ring.n - len(values))
    p = Poly(values, ring).realize()
    raw = np.array(p.node.buffer.as_typed_buffer()).reshape(len(ring.moduli), ring.n)
    radix = 1 << (8 * dtype.itemsize)
    np.testing.assert_array_equal(raw, [[value * radix % q for value in values] for q in ring.moduli])
    assert p.to_centered_list() == [(value + half) % ring.q - half for value in values]
