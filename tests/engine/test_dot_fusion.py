from functools import reduce
from importlib import import_module
from operator import add

import pytest

from ren.poly import Poly, realize
from ren.backends.cuda.renderer import CUDARenderer, KernelSpec
from ren.context import Context
from ren.dtype import dtypes
from ren.engine.pattern import rewrite
from ren.engine.rules.dot_fusion import dot_fusion, dot_steps
from ren.engine.codegen import lower_kernel
from ren.backends.cuda import SUPPORTED_DTYPES
from ren.engine.schedule import schedule
from ren.ir import Op
from ren.ir.node import Node, ParamArg
from ren.kernel import KernelInfo, get_output_slots
from ren.ring import Domain
from tests.utils import large_prime_ring


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_dot_steps_preserves_prefixes(dtype):
    ring = large_prime_ring(128, dtype)[:3]
    idx = Node(Op.GIDX, dtypes.index)
    loads = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")).index(idx).load() for i in range(5)]
    p0, p1, p2, p3 = [(loads[0] * x).mod_reduce(None) for x in loads[1:]]
    first = (p0 + p1).mod_reduce(None)
    second = p2 + first
    value = (second + p3).mod_reduce(None)

    assert dot_steps(p0) == [p0.src[0], p0]
    steps = dot_steps(value)
    assert steps is not None
    assert steps == [p2.src[0], p2, p0.src[0], p0, p1.src[0], p1, first.src[0], first, second, p3.src[0], p3, value.src[0], value]
    inner = (p1 + p2).mod_reduce(None)
    right = p0 + inner
    steps = dot_steps(right)
    assert steps is not None
    assert steps == [p0.src[0], p0, p1.src[0], p1, p2.src[0], p2, inner.src[0], inner, right]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_dot_steps_rejects_unsupported_ops(dtype):
    ring = large_prime_ring(128, dtype)[:3]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(5)]
    loads = [s.index(idx).load() for s in slots]
    p0, p1 = [(loads[0] * x).mod_reduce(None) for x in loads[1:3]]
    for value in ((p0 + p1) // 3, (p0 + p1) & 3, (p0 + p1).cast(dtypes.index)):
        assert dot_steps(value) is None


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize(
    "offsets,target_offset,fused",
    [((0, 0), 0, True), ((128, 128), 0, True), ((1, 1), 0, False), ((-128, -128), 0, False), ((0, 128), 0, False), ((0, 0), 1, False)],
)
def test_dot_fusion_requires_compatible_indices(offsets, target_offset, fused, scaled):
    ring = large_prime_ring(128)[:1]
    inputs = [Node.new_buffer("cuda:0", 3 * ring.n, ring.dtype) for _ in range(3)]
    outputs = [Node.new_buffer("cuda:0", ring.storage_size, ring.dtype) for _ in range(2)]
    target, *params = [Node.param(i, buf) for i, buf in enumerate((outputs[0], *inputs))]
    idx = Node(Op.GIDX, dtypes.index)
    loads = [param.index(idx + offset if offset else idx).load() for param, offset in zip(params, (offsets[0], offsets[1], offsets[1]))]
    value = ((loads[0] * loads[1]).mod_reduce(None) + (loads[0] * loads[2]).mod_reduce(None)).mod_reduce(None)
    if scaled:
        value = (value * 3).mod_reduce(None)
    body = Node.sink(target.index(idx + target_offset if target_offset else idx).store(value)).replace(arg=KernelInfo("cuda:0", ring, Domain.NTT))
    root = Node.sink(*(body.call(output, *inputs) for output in outputs))

    with Context(DOT_TILE=8):
        assert (dot_fusion(root) is not None) is fused


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_dot_fusion_accepts_scalar_coefficients(dtype):
    ring = large_prime_ring(128, dtype)[:3]
    x, y = [Node.new_buffer("cuda:0", ring.storage_size, dtype).as_ring(ring, Domain.NTT) for _ in range(2)]
    outputs = x * 3 + y * 5, x * 7 + y * 11

    with Context(DOT_TILE=8):
        linear, _ = schedule(Node.sink(*outputs))

    assert len(linear.src) == 1
    spec = KernelSpec(linear.src[0].src[0])
    assert len(spec.stores) == 2
    CUDARenderer("sm_120").render(lower_kernel(spec.ast, SUPPORTED_DTYPES))

    a, b = list(range(ring.n)), list(reversed(range(ring.n)))
    px, py = [Poly(values, ring).ntt().realize() for values in (a, b)]
    actual = px * 3 + py * 5, px * 7 + py * 11
    with Context(DOT_TILE=8):
        realize(*actual)
    assert [p.tolist() for p in actual] == [[(c * x + d * y) % ring.q for x, y in zip(a, b)] for c, d in ((3, 5), (7, 11))]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_dot_fusion_reuses_loads_with_mixed_scalar_outputs(dtype):
    ring = large_prime_ring(128, dtype)[:3]
    x, y, a, b, c, d = [Node.new_buffer("cuda:0", ring.storage_size, dtype).as_ring(ring, Domain.NTT) for _ in range(6)]
    outputs = x * a + y * b, x * c + y * d, x * 3 + y * 5

    with Context(DOT_TILE=8):
        linear, _ = schedule(Node.sink(*outputs))

    assert len(linear.src) == 1
    spec = KernelSpec(linear.src[0].src[0])
    assert len(spec.stores) == 3
    # Each of the six inputs is loaded once, including x and y shared by all outputs.
    kernels = CUDARenderer("sm_120").render(lower_kernel(spec.ast, SUPPORTED_DTYPES))
    assert sum(kernel.source.count("__ldg(&buf") for kernel in kernels) == 6


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("shape", ["scaled", "nested", "shared"])
def test_dot_fusion_preserves_composite_outputs(dtype, shape):
    ring = large_prime_ring(128, dtype)[:3]

    def outputs(x, y, a, b, c, d):
        left, right = x * a + y * b, x * c + y * d
        return {
            "scaled": (left * 3, right * 5),
            "nested": (left + right, left + x * d + y * c),
            "shared": ((left + right) * (left + x * d), (left + right) * 3),
        }[shape]

    inputs = [Node.new_buffer("cuda:0", ring.storage_size, dtype).as_ring(ring, Domain.NTT) for _ in range(6)]
    with Context(DOT_TILE=8):
        linear, _ = schedule(Node.sink(*outputs(*inputs)))
    assert len(linear.src) == 1
    ast = linear.src[0].src[0]
    assert len(ast.src) == 2
    kernels = CUDARenderer("sm_120").render(lower_kernel(ast, SUPPORTED_DTYPES))
    assert sum(kernel.source.count("__ldg(&buf") for kernel in kernels) == 6

    inputs = [Poly.uniform(ring).ntt().realize() for _ in range(6)]
    with Context(DOT_TILE=0):
        expected = outputs(*inputs)
        realize(*expected)
    with Context(DOT_TILE=8):
        actual = outputs(*inputs)
        realize(*actual)
    assert [p.tobytes() for p in actual] == [p.tobytes() for p in expected]


def test_dot_fusion_recognizes_repeated_body_once(monkeypatch):
    ring = large_prime_ring(128)[:3]
    bufs = [Node.new_buffer("cuda:0", ring.storage_size, ring.dtype) for _ in range(5)]
    target, *inputs = [Node.param(i, buf) for i, buf in enumerate(bufs[:4])]
    idx = Node(Op.GIDX, dtypes.index)
    x, a, b = [param.index(idx).load() for param in inputs]
    value = ((x * a).mod_reduce(None) + (x * b).mod_reduce(None)).mod_reduce(None)
    body = Node.sink(target.index(idx).store(value)).replace(arg=KernelInfo("cuda:0", ring, Domain.NTT))
    root = Node.sink(body.call(*bufs[:4]), body.call(bufs[4], *bufs[1:4]))

    fusion_module = import_module("ren.engine.rules.dot_fusion")
    original = type(fusion_module._operands).match
    budget = len(value.toposort(gate=lambda n: n.op not in {Op.INDEX, Op.DEVICE}))
    calls = 0

    def bounded(self, node):
        nonlocal calls
        if self is fusion_module._operands:
            calls += 1
            assert calls <= budget, "fusion analyzed the same dot body more than once"
        return original(self, node)

    monkeypatch.setattr(type(fusion_module._operands), "match", bounded)
    with Context(DOT_TILE=8):
        assert dot_fusion(root) is not None


def test_dot_steps_visits_shared_dag_once(monkeypatch):
    ring = large_prime_ring(128)[:3]
    idx = Node(Op.GIDX, dtypes.index)
    loads = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")).index(idx).load() for i in range(4)]
    p, q, value = [(loads[0] * x).mod_reduce(None) for x in loads[1:]]
    for _ in range(16):
        value = (value + p) + (value + q)

    # Bound recognition by DAG size so repeated shared subtrees cannot expand
    # exponentially, without relying on wall-clock timing.
    budget = len(value.toposort())
    fusion_module = import_module("ren.engine.rules.dot_fusion")
    original = type(fusion_module._operands).match
    calls = 0

    def bounded(self, node):
        nonlocal calls
        if self is fusion_module._operands:
            calls += 1
            assert calls <= budget, "dot recognition repeatedly expanded shared subtrees"
        return original(self, node)

    monkeypatch.setattr(type(fusion_module._operands), "match", bounded)
    steps = dot_steps(value)
    assert steps is not None
    assert steps[-1] is value
    assert len(set(steps)) == len(steps) <= budget


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize(
    "rows",
    [
        (((0, 1), (4, 5)), ((1, 2), (4, 6)), ((2, 3), (4, 7))),
        (((7, 6), (5, 3), (3, 0)), ((4, 3), (1, 5), (3, 5)), ((1, 5), (5, 7), (5, 2)), ((0, 6), (4, 6), (7, 0))),
    ],
)
def test_dot_body_reuses_indirectly_shared_loads(dtype, rows):
    ring = large_prime_ring(128, dtype)[:3]
    idx = Node(Op.GIDX, dtypes.index)
    params = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(len(rows) + 8)]
    loads = [param.index(idx).load() for param in params[len(rows) :]]
    values = []
    for row in rows:
        products = [(loads[a] * loads[b]).mod_reduce(None) for a, b in row]
        values.append(reduce(lambda a, b: (a + b).mod_reduce(None), products))
    body = Node.sink(*(param.index(idx).store(value) for param, value in zip(params, values))).replace(arg=KernelInfo("cuda:0", ring, Domain.NTT))
    kernels = CUDARenderer("sm_120").render(lower_kernel(body, SUPPORTED_DTYPES))
    # Shared inputs are loaded once across rows with different products.
    assert sum(kernel.source.count("__ldg(&buf") for kernel in kernels) == 8


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_dot_fusion_accepts_long_single_output(dtype):
    ring = large_prime_ring(128, dtype)[:3]

    def fresh():
        return Node.new_buffer("cuda:0", ring.storage_size, dtype).as_ring(ring, Domain.NTT)

    x = fresh()
    output = reduce(add, [x * fresh() for _ in range(600)])
    with Context(DOT_TILE=0):
        plain, _ = schedule(Node.sink(output))
    with Context(DOT_TILE=8):
        tiled, _ = schedule(Node.sink(output))
    plain_kernels = [n for n in plain.toposort() if n.op is Op.CALL]
    tiled_kernels = [n for n in tiled.toposort() if n.op is Op.CALL]
    assert len(plain_kernels) == len(tiled_kernels) == 1
    assert tiled_kernels[0].src[0] is plain_kernels[0].src[0]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("nested", [False, True])
def test_dot_fusion_projects_all_digit_segments(dtype, nested):
    ring = large_prime_ring(128, dtype)[:6]

    def fresh():
        return Node.new_buffer("cuda:0", ring.storage_size, dtype).as_ring(ring, Domain.NTT)

    x = fresh()
    terms = [(x * fresh(), fresh() * fresh()) for _ in range(2)]
    outputs = []
    for start in (0, 2, 4):
        inner = ring[max(0, start - 1) :] if nested else ring
        outputs.extend((a.to_ring(inner) + b.to_ring(inner)).to_ring(ring[start : start + 2]) for a, b in terms)
    with Context(DOT_TILE=8):
        linear, _ = schedule(Node.sink(*outputs))
    kernels = linear.src
    assert len(kernels) == 3
    assert {k.src[0].arg.ring for k in kernels} == {ring[start : start + 2] for start in (0, 2, 4)}
    for kernel in kernels:
        spec = KernelSpec(kernel.src[0])
        assert len(spec.stores) == 2
        CUDARenderer("sm_120").render(lower_kernel(spec.ast, SUPPORTED_DTYPES))


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("views", [False, True])
@pytest.mark.parametrize("boundary", [None, "bufferize", "contiguous"])
def test_dot_fusion_keeps_outputs_and_dependencies(dtype, views, boundary, schedule_cache):
    ring = large_prime_ring(128, dtype)[:3]

    def fresh():
        return Node.new_buffer("cuda:0", ring.storage_size, dtype).as_ring(ring, Domain.NTT)

    source_ring = large_prime_ring(128, dtype)[:4]
    xs = [
        Node.new_buffer("cuda:0", source_ring.storage_size, dtype).as_ring(source_ring, Domain.NTT).to_ring(ring) if views else fresh()
        for _ in range(4)
    ]
    outputs = [reduce(add, [x * fresh() for x in xs[:terms]]) for terms in (4, 3, 2, 4)]
    # A later dot depends on the first output, and must never join its tile.
    left, right = (getattr(n, boundary)() if boundary else n for n in (outputs[0], outputs[-1]))
    downstream = left * fresh() + right * fresh()
    root = Node.sink(*outputs, downstream)
    rewrites = []
    with Context(DOT_TILE=4):
        fused_root, _ = schedule(root, trace=lambda name, before, after: rewrites.append((name, before, after)))
    fused_root = next(after for name, _, after in reversed(rewrites) if name == "dot_fusion")
    # Fusion operates on abstract storage; concrete buffers are bound afterward.
    assert all(n.op is not Op.BUFFER for n in fused_root.toposort(enter_calls=True))
    graph_tiles = [n for n in fused_root.toposort() if n.op is Op.CALL and n.src[0].op is Op.SINK and len(n.src[0].src) > 1]
    assert len(graph_tiles) == 1 and len(graph_tiles[0].src[0].src) == 4
    assert all(n.ring is None for n in graph_tiles[0].src[0].toposort())
    assert any(name == "dot_fusion" and before.op is Op.CALL for name, before, after in rewrites)
    with Context(DOT_TILE=0):
        plain, _ = schedule(root)
    with Context(DOT_TILE=4):
        tiled, becomes = schedule(root)
    multi = [call for call in tiled.src if len(get_output_slots(call.src[0])) > 1]
    assert len(multi) == 1 and len(get_output_slots(multi[0].src[0])) == 4
    assert len(tiled.src) == len(plain.src) - 3
    assert {multi[0].src[i + 1].base for i in get_output_slots(multi[0].src[0])} == {becomes[n].base for n in outputs}
    assert tiled.src.index(multi[0]) < len(tiled.src) - 1
    assert all(n.op is not Op.AS_RING for n in multi[0].src[0].toposort())


def test_dot_fusion_keeps_all_dependencies_of_shared_input():
    ring = large_prime_ring(128)[:3]

    def fresh():
        return Node.new_buffer("cuda:0", ring.storage_size, ring.dtype).as_ring(ring, Domain.NTT)

    x, y = fresh(), fresh()
    with Context(DOT_TILE=0):
        root, _ = schedule(Node.sink(x * fresh() + y * fresh(), x * fresh() + y * fresh()))
    kernels = [n for n in root.toposort() if n.op is Op.CALL]
    assert len(kernels) == 2
    idx = Node(Op.GIDX, dtypes.index)
    target, source = (Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(2))
    copy_body = Node(Op.SINK, src=(target.index(idx).store(source.index(idx).load()),), arg=kernels[0].src[0].arg)
    markers = [Node(Op.CALL, src=(copy_body, fresh().base, x.base)) for _ in kernels]
    marked = [k.replace(src=(k.src[0], *(x.base.after(marker) if s.base is x.base else s for s in k.src[1:]))) for k, marker in zip(kernels, markers)]
    root = root.substitute(dict(zip(kernels, marked)))
    for tile_size in (0, 1):
        with Context(DOT_TILE=tile_size):
            assert dot_fusion(root) is None
    with Context(DOT_TILE=8):
        fusion = dot_fusion(root)
        assert fusion is not None
        fused = rewrite(root, fusion, bottom_up=True)
    order = list(fused.toposort())
    tiles = [n for n in order if n.op is Op.CALL and len(n.src[0].src) > 1]
    assert len(tiles) == 1
    assert all(marker in order and order.index(marker) < order.index(tiles[0]) for marker in markers)
    assert all(old not in order for old in marked)
