import gc
import weakref
from typing import assert_type
import pytest

import ren.engine.rules as rules
from ren.engine.pattern import rewrite
from ren.ir import Node, Op
from ren.ir.node import ParamArg, RandomArg
from ren.dtype import DType, dtypes
from ren.polydist import Ternary
from ren.ring import RingSpec, Domain


def test_cached_node():
    class Child(Node):
        pass

    n1 = assert_type(Node(Op.ADD, dtype=dtypes.void), Node)
    n2 = assert_type(type(n1)(Op.ADD), Node)
    child = assert_type(Child(Op.ADD), Child)

    assert n1 is n2
    assert child is Child(Op.ADD) and child is not n1


def test_node_removed_from_cache_on_del():
    Node._cache.clear()

    n = Node(Op.ADD)
    key = Node.__cache_key__(n.op, n.dtype, n.src, n.arg)
    assert key in Node._cache

    w = weakref.ref(n)
    del n
    gc.collect()

    assert w() is None
    assert key not in Node._cache


def test_toposort_orders_shared_dependencies_once():
    shared = Node.alloc("python", 1, dtypes.uint64)
    left = Node(Op.ADD, src=(shared,), arg="left")
    right = Node(Op.ADD, src=(shared,), arg="right")
    root = left.after(right)

    order = root.toposort()

    assert list(order) == [shared, left, right, root]
    assert len(order) == 4
    assert shared in order


def test_uncached_node_does_not_evict_cached_twin():
    Node._cache.clear()
    uncached = Node._uncached(Op.CONST, dtypes.weakint, (), 12345)
    cached = Node(uncached.op, uncached.dtype, uncached.src, uncached.arg)
    assert cached is not uncached

    del uncached
    gc.collect()

    assert Node(cached.op, cached.dtype, cached.src, cached.arg) is cached


def test_substitute_keeps_shared_call_body_opaque():
    source = Node.new_buffer("python", size=4, dtype=dtypes.uint64)
    param = Node.param(0, source)
    body = Node(Op.BUFFER_VIEW, source.dtype, src=(param,), arg=(2, 0))
    output = Node.new_buffer("python", size=2, dtype=source.dtype)
    call = Node(Op.CALL, src=(body, output, source))

    assert body not in call.toposort()
    assert body in call.toposort(enter_calls=True)
    result = Node.sink(body, call).substitute({param: source})

    assert result.src[0].src[0] is source
    assert result.src[1].src[0] is body


def test_map_stops_at_replaced_input_graph():
    source = Node.new_buffer("python", size=4, dtype=dtypes.uint64)
    param = Node.param(0, source)
    for _ in range(100):
        source = -source
    root = -source
    visited: list[Node] = []

    def bind(node: Node) -> Node | None:
        visited.append(node)
        return param if node is source else None

    result = root.map(bind)

    assert result.src == (param,)
    assert visited == [root, source]


def test_ring_as_ring_and_propagation():
    buf = Node.new_buffer("python", size=4, dtype=dtypes.uint64)
    ring = RingSpec(n=4, moduli=(17,))
    a = buf.as_ring(ring, Domain.COEFF)
    b = buf.as_ring(ring, Domain.COEFF)

    assert a.ring == ring
    assert b.ring == ring

    add = Node(Op.ADD, dtype=a.dtype, src=(a, b))
    assert add.ring == ring


def test_ring_mismatch_raises():
    buf = Node.new_buffer("python", size=4, dtype=dtypes.uint64)
    ring1 = RingSpec(n=4, moduli=(17,))
    ring2 = RingSpec(n=4, moduli=(257,))

    a = buf.as_ring(ring1, Domain.COEFF)
    b = buf.as_ring(ring2, Domain.COEFF)

    with pytest.raises(RuntimeError):
        _ = Node(Op.ADD, dtype=a.dtype, src=(a, b)).ring


def test_domain_as_ring_ntt_intt():
    buf = Node.new_buffer("python", size=4, dtype=dtypes.uint64)
    ring = RingSpec(n=4, moduli=(17,))
    coeff = buf.as_ring(ring, Domain.COEFF)

    assert coeff.domain is Domain.COEFF

    ntt = Node(Op.NTT, dtype=coeff.dtype, src=(coeff,))
    assert ntt.domain is Domain.NTT

    intt = Node(Op.INTT, dtype=ntt.dtype, src=(ntt,))
    assert intt.domain is Domain.COEFF


def test_domain_propagation_through_add():
    buf = Node.new_buffer("python", size=4, dtype=dtypes.uint64)
    ring = RingSpec(n=4, moduli=(17,))
    coeff = buf.as_ring(ring, Domain.COEFF)
    ntt = Node(Op.NTT, dtype=coeff.dtype, src=(coeff,))

    add = Node(Op.ADD, dtype=coeff.dtype, src=(ntt, ntt))
    assert add.domain is Domain.NTT


@pytest.mark.parametrize("target_domain", [Domain.COEFF, Domain.NTT])
def test_store_converts_value_to_target_domain(target_domain: Domain):
    ring = RingSpec(n=4, moduli=(17,))
    value_domain = Domain.NTT if target_domain is Domain.COEFF else Domain.COEFF
    target = Node.new_buffer("python", size=4, dtype=dtypes.uint64).as_ring(ring, target_domain)
    value = Node.new_buffer("python", size=4, dtype=dtypes.uint64).as_ring(ring, value_domain)

    store = rewrite(target.store(value), rules.inject_domains)

    assert store.src[0] is target
    assert store.src[1].op is (Op.NTT if target_domain is Domain.NTT else Op.INTT)
    assert store.src[1].src[0] is value


def test_after_dependencies_do_not_change_value_metadata():
    ring = RingSpec(n=4, moduli=(17,))
    value = Node.new_buffer("python", size=4, dtype=dtypes.uint64).as_ring(ring, Domain.COEFF)
    dependency = Node.new_buffer("python", size=4, dtype=dtypes.uint64).as_ring(RingSpec(n=4, moduli=(41,)), Domain.NTT)

    after = value.after(dependency)

    assert after.ring == ring
    assert after.domain is Domain.COEFF
    assert rewrite(after, rules.inject_domains) is after


@pytest.mark.parametrize("input_domain", [Domain.COEFF, Domain.NTT])
def test_domain_propagation_through_copy(input_domain):
    """to() preserves domain in IR structure."""
    buf = Node.new_buffer("python", size=4, dtype=dtypes.uint64)
    ring = RingSpec(n=4, moduli=(17,))
    p = buf.as_ring(ring, Domain.COEFF)

    if input_domain == Domain.NTT:
        p = Node(Op.NTT, dtype=p.dtype, src=(p,))

    assert p.domain == input_domain

    p2 = p.copy_to_device("cuda:0")

    # Domain is preserved via propagation through COPY
    assert p2.domain == input_domain
    assert p2.op is Op.COPY


def test_empty_like_requests_fresh_storage_with_same_logical_metadata():
    ring = RingSpec(n=4, moduli=(17, 257))
    key = Node.new_buffer("python", size=8, dtype=dtypes.uint32)
    node = Node(Op.RANDOM, dtype=dtypes.uint64, src=(key,), arg=RandomArg(Ternary(ring.n), 0)).as_ring(ring, Domain.NTT)

    empty = node.empty_like()

    assert empty.ring == ring
    assert empty.domain is Domain.NTT
    assert empty.base.op is Op.ALLOC
    assert empty.base is not node.empty_like().base
    assert empty.base.size == ring.n * len(ring.moduli)
    assert empty.device == "python"


def test_clone_uses_fresh_allocation_store_after_graph():
    ring = RingSpec(n=4, moduli=(17,))
    node = Node.new_buffer("python", size=4, dtype=dtypes.uint64).as_ring(ring, Domain.COEFF)

    clone = node.clone()

    assert clone.ring == ring
    assert clone.domain is Domain.COEFF
    assert clone.op is Op.AFTER
    assert clone.src[0].ring == ring
    assert clone.src[0].base.op is Op.ALLOC
    assert clone.src[0].base is not node.base
    assert clone.src[1].op is Op.STORE
    assert clone.src[1].src[1] is node


def test_clone_to_different_device_preserves_metadata():
    ring = RingSpec(n=4, moduli=(17,))
    node = Node.new_buffer("python", size=4, dtype=dtypes.uint64).as_ring(ring, Domain.COEFF)

    clone = node.clone("cuda")

    assert clone.device == "cuda:0"
    assert clone.ring == ring
    assert clone.domain is Domain.COEFF


def test_bufferize_to_store_uses_logical_empty_like_target():
    ring = RingSpec(n=4, moduli=(17,))
    key = Node.new_buffer("python", size=8, dtype=dtypes.uint32)
    node = Node(Op.RANDOM, dtype=dtypes.uint64, src=(key,), arg=RandomArg(Ternary(ring.n), 0)).as_ring(ring, Domain.COEFF)

    lowered = rewrite(node.bufferize(), rules.bufferize_to_store)

    assert lowered.op is Op.AFTER
    assert lowered.ring == ring
    assert lowered.domain is Domain.COEFF
    assert lowered.src[0].ring == ring
    assert lowered.src[0].base.op is Op.ALLOC
    assert lowered.src[1].op is Op.STORE
    assert lowered.src[1].src[0] is lowered.src[0]


def test_replace_unchanged_returns_self_for_uncached_nodes():
    buf = Node.new_buffer("python", size=4, dtype=dtypes.uint64)

    assert buf.replace(src=buf.src) is buf
    assert buf.replace(arg=buf.arg) is buf
    assert buf.replace(dtype=dtypes.uint32) is not buf


def _scalar(arg: int, dtype: DType = dtypes.uint64) -> Node:
    return Node(Op.PARAM, dtype, arg=ParamArg(arg, 4, "python")).index(Node(Op.GIDX, dtypes.index, arg=4)).load()


def test_comparisons_build_bool_nodes():
    x, y = _scalar(1), _scalar(2)
    for cmp in (x < y, x > y, x <= y, x >= y, x.eq(y), x.ne(y)):
        assert cmp.dtype is dtypes.bool


def test_node_truth_requires_constant_boolean():
    five = Node.const(5, dtypes.uint64)
    assert bool(five < 3) is False
    assert bool(five >= 3) is True
    with pytest.raises(ValueError, match="non-constant"):
        bool(_scalar(1) < 3)
    with pytest.raises(TypeError, match="boolean"):
        bool(five)


def test_comparison_literals_and_reflection_use_node_dtype():
    x = _scalar(1, dtypes.uint32)
    five = Node.const(5, dtypes.uint32)
    assert (x < 5) is (x < five)
    assert (5 < x) is (x > 5) is (five < x)
    assert (5 >= x) is (x <= 5)
    assert x.eq(5).src[1] is five
    assert (x << 3).op is Op.SHL and (x << 3).src == (x, Node.const(3, dtypes.uint32)) and (x << 3).dtype is dtypes.uint32
    assert (x >> 3).op is Op.SHR and (x >> 3).dtype is dtypes.uint32


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64, dtypes.index])
def test_integer_operator_literals_use_raw_node_dtype(dtype):
    x = _scalar(1, dtype)
    three = Node.const(3, dtype)
    for result, op in ((x % 3, Op.MOD), (x // 3, Op.IDIV), (x & 3, Op.AND)):
        assert result.op is op and result.dtype is dtype and result.src == (x, three)
    for result, op in ((3 % x, Op.MOD), (3 // x, Op.IDIV), (3 & x, Op.AND)):
        assert result.op is op and result.dtype is dtype and result.src == (three, x)


def test_node_equality_and_hash_stay_identity_based():
    x, y = _scalar(1), _scalar(2)
    assert (x == x) is True and (x == y) is False
    assert (x != y) is True and (x != x) is False
    assert hash(x) == hash(_scalar(1)) and _scalar(1) is x
    assert {x: "x", y: "y"}[_scalar(2)] == "y"
    assert x in {x, y} and len({x, y, _scalar(1)}) == 2


def test_exact_literal_is_shared_across_rings_and_devices():
    from ren.poly import Poly

    value = (1 << 200) + 7
    literal = Node.const(value)
    assert literal.dtype is dtypes.weakint and literal.src == ()
    for dtype in (dtypes.uint32, dtypes.uint64):
        ring = RingSpec(8, (17, 97), dtype=dtype)
        virtual = Poly.const(value, ring)
        assert virtual.node.src[0] is literal and virtual.node.const_val == value
        assert virtual.device is None
        for device in ("python", "cuda:0"):
            p = Poly.const(value, ring, device)
            assert {n for n in p.node.toposort() if n.op is Op.CONST} == {literal}
            assert p.node.dtype is dtype and p.ring == ring and p.device == device


@pytest.mark.parametrize("value", [(1 << 200) + 7, -(1 << 200) - 7])
def test_literal_cast_has_word_semantics(value):
    literal = Node.const(value)
    for dtype in (dtypes.uint32, dtypes.uint64, dtypes.uint128):
        typed = literal.cast(dtype)
        assert typed.src == (literal,) and typed.op is Op.CAST
        assert typed.const_val == value % (1 << (dtype.itemsize * 8))


@pytest.mark.parametrize("value", [2.5, "3"])
def test_integer_literal_rejects_non_integers(value):
    with pytest.raises(TypeError):
        Node.const(value)
