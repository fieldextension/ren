from ren.engine.codegen import lower_kernel
import re
from contextlib import nullcontext
from functools import reduce
from operator import add
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

import ren
from ren.backends.cuda import SUPPORTED_DTYPES
from ren.backends.cuda.renderer import CUDARenderer
from ren.buffer import Buffer, BufferSpec
from ren.context import Context
from ren.dtype import dtypes
from ren.device import Device
from ren.engine.realize import ExecUnit
from ren.ir import Op
from ren.ir.node import Node, ParamArg
from ren.kernel import KernelInfo, KernelSpec
from ren.poly import Poly
from ren.ring import Domain
from tests.utils import large_prime_ring


@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("case", ["mixed_operands", "pointer_operand", "load_arity", "wrong_axis"])
def test_generated_kernel_rejects_invalid_scalar_ir(outputs, case):
    ring = large_prime_ring(128, dtypes.uint32)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(outputs + 1)]
    value = slots[-1].index(idx).load()
    if case == "mixed_operands":
        value = value + Node.const(1, dtypes.index)
    elif case == "pointer_operand":
        value = value + slots[-1]
    elif case == "load_arity":
        value = value.replace(src=(*value.src, idx))
    else:
        value = value.substitute({idx: idx.replace(arg=1)})
    ast = Node(Op.SINK, src=tuple(s.index(idx).store(value) for s in slots[:-1]), arg=KernelInfo("cuda:0", ring, Domain.NTT))
    with pytest.raises(ValueError, match="Invalid IR node"):
        lower_kernel(ast, SUPPORTED_DTYPES)


@pytest.mark.parametrize(
    "case",
    [
        "ntt_ntt",
        "ntt_dot",
        "dot_ntt",
        "duplicate_output",
        "read_output",
        "index_operand",
        "wrong_dtype",
        "wrong_value_dtype",
        "wrong_axis",
        "wrong_ring",
        "hidden_store",
    ],
)
def test_cuda_kernel_rejects_unsupported_outputs_before_compilation(case, monkeypatch):
    pytest.importorskip("cuda.bindings")

    from ren.backends.cuda import api as cuda
    from ren.backends.cuda.backend import CUDABackend

    ring = large_prime_ring(128, dtypes.uint32)[:3]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(6)]
    a, b, c, d = [s.index(idx).load() for s in slots[1:5]]
    dot = ((a * b).mod_reduce(None) + (c * d).mod_reduce(None)).mod_reduce(None)
    stores = [slots[i].index(idx).store(dot) for i in (0, 5)]
    info = KernelInfo("cuda:0", ring, Domain.NTT)
    if case in {"ntt_ntt", "ntt_dot", "dot_ntt"}:
        for i, kind in enumerate(case.split("_")):
            if kind == "ntt":
                stores[i] = slots[5 * i].store(Node(Op.NTT, ring.dtype, (slots[1],)))
    elif case == "duplicate_output":
        stores[1] = slots[0].index(idx).store((dot + (a * c).mod_reduce(None)).mod_reduce(None))
    elif case == "read_output":
        stores[1] = stores[1].substitute({slots[1]: slots[0]})
    elif case == "index_operand":
        offset = Node.const(ring.n, dtypes.index)
        value = (slots[1].index(idx + offset).load() * offset).mod_reduce(None)
        stores = [slots[i].index(idx).store(value) for i in (0, 5)]
    elif case == "wrong_dtype":
        stores[1] = stores[1].substitute({slots[5]: Node(Op.PARAM, dtypes.uint64, arg=ParamArg(5, ring.storage_size, "cuda:0"))})
    elif case == "wrong_value_dtype":
        stores[1] = slots[5].index(idx).store(dot.replace(dtype=dtypes.bool))
    elif case == "wrong_axis":
        stores[1] = stores[1].substitute({idx: Node(Op.GIDX, dtypes.index, arg=1)})
    elif case == "wrong_ring":
        stores[1] = slots[5].index(idx).store(dot.replace(arg=ring[:2]))
    elif case == "hidden_store":
        stores = [slots[0].index(idx).store(stores[1])]
    ast = Node(Op.SINK, src=tuple(stores), arg=info)
    backend = object.__new__(CUDABackend)
    backend.context = Mock()
    backend.renderer = CUDARenderer("sm_120")
    backend._make_kernel_cache = {}
    compile_mock = Mock()
    monkeypatch.setattr(backend, "_compile", compile_mock)
    activate = Mock()
    monkeypatch.setattr(cuda, "ctx_set_current", activate)

    with pytest.raises((ValueError, NotImplementedError), match="Invalid IR node|kernel|STORE|MOD_REDUCE"):
        backend.make_kernel(ast)
    activate.assert_not_called()
    compile_mock.assert_not_called()
    assert not backend._make_kernel_cache


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("axis", [None, 0])
def test_cuda_dot_body_owns_arguments_outputs_and_cache(dtype, axis, monkeypatch):
    pytest.importorskip("cuda.bindings")

    from ren.backends.cuda import api as cuda
    from ren.backends.cuda.backend import CUDABackend

    ring = large_prime_ring(128, dtype)[:3]
    idx = Node(Op.GIDX, dtypes.index, arg=axis)
    slots = {i: Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in (0, 2, 4, 5, 7, 9)}
    a, b, c, d = [slots[i].index(idx).load() for i in (0, 2, 5, 9)]
    ab, cd, ac, bd, ad, bc = [(x * y).mod_reduce(None) for x, y in ((a, b), (c, d), (a, c), (b, d), (a, d), (b, c))]
    first_store = slots[4].index(idx).store((ab + cd).mod_reduce(None))
    asts = [
        Node(Op.SINK, src=(first_store, slots[7].index(idx).store(value)), arg=KernelInfo("cuda:0", ring, Domain.NTT))
        for value in ((ac + bd).mod_reduce(None), (ad + bc).mod_reduce(None))
    ]
    spec = KernelSpec(asts[0])
    assert spec.argument_slots == (0, 2, 4, 5, 7, 9)
    assert spec.output_slots == (4, 7)

    backend = object.__new__(CUDABackend)
    backend.context = Mock()
    backend.renderer = Mock(wraps=CUDARenderer("sm_120"))
    backend._make_kernel_cache = {}
    compile_mock = Mock(side_effect=lambda rendered: object())
    monkeypatch.setattr(backend, "_compile", compile_mock)
    monkeypatch.setattr(backend, "_launch_dims", Mock(return_value=((1,), (128,))))
    monkeypatch.setattr(cuda, "ctx_set_current", Mock())

    prepare = Mock(wraps=backend._build_launch_specs)
    monkeypatch.setattr(backend, "_build_launch_specs", prepare)

    first, repeated, changed = [backend.make_kernel(ast) for ast in (asts[0], asts[0], asts[1])]
    assert backend.renderer.render.call_count == compile_mock.call_count == prepare.call_count == 2
    assert first.specs[0].func is repeated.specs[0].func
    assert changed.specs[0].func is not first.specs[0].func
    rendered, changed_rendered = [call.args[0] for call in compile_mock.call_args_list]
    assert rendered.cache_key != changed_rendered.cache_key
    params = rendered.source.split("void ren_generated_", 1)[1].split("(", 1)[1].split(")", 1)[0]
    assert tuple(map(int, re.findall(r"\bbuf(\d+)\b", params))) == first.specs[0].buf_indices == spec.argument_slots
    assert f"const {dtype.name}* __restrict__ buf0" in params
    assert f"{dtype.name}* buf4" in params and f"{dtype.name}* buf7" in params

    bufs = [SimpleNamespace(_buf=100 + i) for i in range(10)]
    first.specs[0].bind_bufs(cast(list[Buffer], bufs))
    repeated.specs[0].bind_bufs(cast(list[Buffer], [SimpleNamespace(_buf=200 + i) for i in range(10)]))
    assert [arg.value for arg in first.specs[0].args[:-1]] == [100 + i for i in spec.argument_slots]
    assert [arg.value for arg in repeated.specs[0].args[:-1]] == [200 + i for i in spec.argument_slots]
    assert all(a is not b for a, b in zip(first.specs[0].args, repeated.specs[0].args, strict=True))


@pytest.mark.parametrize("case", ["slot", "split_mul", "split_neg"])
def test_cuda_rejects_invalid_after_definition_boundary(case, monkeypatch):
    pytest.importorskip("cuda.bindings")

    from ren.backends.cuda import api as cuda
    from ren.backends.cuda.backend import CUDABackend

    ring = large_prime_ring(128, dtypes.uint32)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    output, x, y = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(3)]
    a, b = x.index(idx).load(), y.index(idx).load()
    raw = -a if case == "split_neg" else a * b
    value = raw.after(raw).mod_reduce(None) if case != "slot" else (a.after(x) * b).mod_reduce(None)
    ast = Node(Op.SINK, src=(output.index(idx).store(value),), arg=KernelInfo("cuda:0", ring, Domain.NTT))
    backend = object.__new__(CUDABackend)
    backend.renderer = CUDARenderer("sm_120")
    backend._make_kernel_cache = {}
    activate = Mock()
    monkeypatch.setattr(cuda, "ctx_set_current", activate)
    with pytest.raises(ValueError):
        backend.make_kernel(ast)
    activate.assert_not_called()
    assert not backend._make_kernel_cache


def test_after_dependency_order_controls_source_and_cache(monkeypatch):
    pytest.importorskip("cuda.bindings")

    from ren.backends.cuda import api as cuda
    from ren.backends.cuda.backend import CUDABackend

    ring = large_prime_ring(128, dtypes.uint32)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(4)]
    a, b, c = [s.index(idx).load() for s in slots[1:]]
    asts = []
    for first, second in ((b, slots[3]), (c, slots[2])):
        f = (a * first).mod_reduce(None)
        g = (a * second.index(idx.after(f)).load()).mod_reduce(None)
        # g depends on f even though g is the left operand.
        value = (g + f).mod_reduce(None)
        asts.append(Node(Op.SINK, src=(slots[0].index(idx).store(value),), arg=KernelInfo("cuda:0", ring, Domain.NTT)))
    ab, ac = (a * b).mod_reduce(None), (a * c).mod_reduce(None)
    inline = asts[0].replace(src=(slots[0].index(idx).store((ab + ac).mod_reduce(None)),))
    backend = object.__new__(CUDABackend)
    backend.context = Mock()
    backend.renderer = CUDARenderer("sm_120")
    backend._make_kernel_cache = {}
    compile_mock = Mock(side_effect=lambda _: object())
    monkeypatch.setattr(backend, "_compile", compile_mock)
    monkeypatch.setattr(backend, "_launch_dims", Mock(return_value=((1,), (128,))))
    monkeypatch.setattr(cuda, "ctx_set_current", Mock())
    first = backend.make_kernel(asts[0])
    repeated = backend.make_kernel(asts[0])
    backend.make_kernel(asts[1])
    backend.make_kernel(inline)
    assert compile_mock.call_count == 3
    assert first.specs[0].func is repeated.specs[0].func
    forward, reverse, inline_source = [call.args[0].source for call in compile_mock.call_args_list]
    assert len({forward, reverse, inline_source}) == 3
    for source, first, second in ((forward, 2, 3), (reverse, 3, 2)):
        assert source.index(f"__ldg(&buf{first}[") < source.index(f"__ldg(&buf{second}[")


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_cuda_materializes_shared_load_without_scheduling(dtype):
    ring = large_prime_ring(128, dtype)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    a, b, x = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(3)]
    value = x.index(idx).load()
    stores = (a.index(idx).store(value), b.index(idx).store((value + value).mod_reduce(None)))
    body = Node(Op.SINK, src=stores, arg=KernelInfo("cuda:0", ring, Domain.NTT))
    (rendered,) = CUDARenderer("sm_120").render(lower_kernel(body, SUPPORTED_DTYPES))
    assert rendered.source.count("__ldg(&buf2[") == 1


@pytest.mark.requires_cuda
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("chained", [False, True])
def test_cuda_single_output_scalar_after_executes_in_dependency_order(cuda_device, dtype, chained):
    device = cuda_device.device
    ring = large_prime_ring(128, dtype)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(4)]
    a, b = [s.index(idx).load() for s in slots[1:3]]
    first = (a * b).mod_reduce(None)
    second = (a * slots[3].index(idx.after(first)).load()).mod_reduce(None)
    # first has no data consumer. Its definition must follow from AFTER alone.
    value = a.after(second) if chained else second
    ast = Node(Op.SINK, src=(slots[0].index(idx).store(value),), arg=KernelInfo(device, ring, Domain.NTT))
    (rendered,) = CUDARenderer("sm_120").render(lower_kernel(ast, SUPPORTED_DTYPES))
    assert rendered.source.index("__ldg(&buf2[") < rendered.source.index("__ldg(&buf3[")
    if chained:
        assert rendered.source.count("__ldg(&buf1[idx])") == 1
    backend = Device[device]
    bufs = [Buffer(BufferSpec(dtype=dtype, size=ring.storage_size, device=device)) for _ in slots]
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    arrays = [np.full(ring.storage_size, x, dtype=np_dtype) for x in (3, 5, 7)]
    for buf, values in zip(bufs[1:], arrays):
        buf.copyin(memoryview(values).cast("B"))
    ExecUnit(backend.make_kernel(ast), Node(Op.CALL, src=(ast, *(Node.from_buffer(buf) for buf in bufs))))()
    backend.synchronize()
    expected = 3 if chained else (3 * 7 * pow(1 << (dtype.itemsize * 8), -1, ring.moduli[0])) % ring.moduli[0]
    assert bytes(bufs[0].as_buffer()) == np.full(ring.storage_size, expected, dtype=np_dtype).tobytes()


@pytest.mark.requires_cuda
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("case", ["add", "defined_add", "load", "mul", "sub", "neg", 0, 3, -3, 2**80 + 3])
def test_cuda_mod_reduce_consumes_after_value(cuda_device, dtype, case):
    device = cuda_device.device
    ring = large_prime_ring(128, dtype)[:3]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(4)]
    a, b, c = [s.index(idx).load() for s in slots[1:]]
    dependency = (b * c).mod_reduce(None)
    if isinstance(case, int):
        value = Node(Op.MOD_REDUCE, dtype, (Node.const(case),))
    else:
        value = {
            "add": a + b,
            "defined_add": a + b,
            "load": a,
            "mul": (a * b).mod_reduce(None),
            "sub": (a - b).mod_reduce(None),
            "neg": (-a).mod_reduce(None),
        }[case]
    defined = value.after(dependency).after(c)
    reduced = defined.mod_reduce(None)
    result = reduced
    if case == "defined_add":
        defined = value.after(dependency, value)
        reduced = value.mod_reduce(None)
        result = reduced.after(defined)
    ast = Node(Op.SINK, src=(slots[0].index(idx).store(result),), arg=KernelInfo(device, ring, Domain.NTT))
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    left = np.concatenate([np.resize(np.array([0, 1, p - 1, p - 2], dtype=np_dtype), ring.n) for p in ring.moduli])
    right = np.concatenate([np.resize(np.array([0, p - 1, p - 1, 2], dtype=np_dtype), ring.n) for p in ring.moduli])
    bufs = [Buffer(BufferSpec(dtype=dtype, size=ring.storage_size, device=device)) for _ in slots]
    for buf, data in zip(bufs[1:], (left, right, np.full(ring.storage_size, 7, dtype=np_dtype))):
        buf.copyin(memoryview(data).cast("B"))
    backend = Device[device]
    ExecUnit(backend.make_kernel(ast), Node(Op.CALL, src=(ast, *(Node.from_buffer(buf) for buf in bufs))))()
    backend.synchronize()
    expected = []
    radix = 1 << (dtype.itemsize * 8)
    for p, xs, ys in zip(ring.moduli, left.reshape(-1, ring.n), right.reshape(-1, ring.n)):
        for x, y in zip(xs, ys):
            x, y = int(x), int(y)
            expected.append(
                (
                    case * radix
                    if isinstance(case, int)
                    else {"add": x + y, "defined_add": x + y, "load": x, "mul": x * y * pow(radix, -1, p), "sub": x - y, "neg": -x}[case]
                )
                % p
            )
    assert bytes(bufs[0].as_buffer()) == np.array(expected, dtype=np_dtype).tobytes()


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("op", [Op.MUL, Op.SUB, Op.NEG])
@pytest.mark.parametrize("boundary", ["wrapped", "nested", "dependency"])
def test_cuda_rejects_split_modular_after(dtype, op, boundary):
    ring = large_prime_ring(128, dtype)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(3)]
    a, b = [s.index(idx).load() for s in slots[1:]]
    raw = {Op.MUL: a * b, Op.SUB: a - b, Op.NEG: -a}[op]
    if boundary == "dependency":
        value = raw.mod_reduce(None).after(raw)
    else:
        value = raw.after(a)
        if boundary == "nested":
            value = value.after(b)
        value = value.mod_reduce(None)
    ast = Node(Op.SINK, src=(slots[0].index(idx).store(value),), arg=KernelInfo("cuda:0", ring, Domain.NTT))
    with pytest.raises(ValueError, match="AFTER cannot split a modular instruction"):
        CUDARenderer("sm_120").render(lower_kernel(ast, SUPPORTED_DTYPES))


@pytest.mark.requires_cuda
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_shared_lazy_prefix_can_feed_reduced_and_longer_outputs(dtype):
    ring = large_prime_ring(128, dtype)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")) for i in range(6)]
    a, *xs = [s.index(idx).load() for s in slots[2:]]
    products = [(a * x).mod_reduce(None) for x in xs]
    prefix = products[0] + products[1]
    values = [prefix.mod_reduce(None), (prefix + products[2]).mod_reduce(None)]
    ast = Node(Op.SINK, src=tuple(s.index(idx).store(v) for s, v in zip(slots, values)), arg=KernelInfo(Device.DEFAULT, ring, Domain.NTT))
    spec = KernelSpec(ast)
    # The reduced output and longer row must share the same addition.
    assert sum(n.op is Op.ADD for n in spec.program) == 2
    CUDARenderer("sm_120").render(lower_kernel(spec.ast, SUPPORTED_DTYPES))
    bufs = [Buffer(BufferSpec(dtype=dtype, size=ring.storage_size, device=Device.DEFAULT)) for _ in slots]
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    rng = np.random.default_rng(7654)
    arrays = [rng.integers(0, ring.moduli[0], ring.n, dtype=np_dtype) for _ in slots[2:]]
    for buf, data in zip(bufs[2:], arrays):
        buf.copyin(memoryview(data))
    backend = Device[Device.DEFAULT]
    ExecUnit(backend.make_kernel(ast), Node(Op.CALL, src=(ast, *(Node.from_buffer(buf) for buf in bufs))))()
    backend.synchronize()
    p = ring.moduli[0]
    r_inv = pow(1 << (dtype.itemsize * 8), -1, p)
    for output, terms in zip(bufs, (2, 3)):
        expected = np.array([sum(int(arrays[0][i]) * int(x[i]) * r_inv for x in arrays[1 : terms + 1]) % p for i in range(ring.n)], dtype=np_dtype)
        assert bytes(output.as_buffer()) == expected.tobytes()


@pytest.mark.requires_cuda
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("graph", [0, 1, "queue"])
@pytest.mark.parametrize("terms", [4, 20])
@pytest.mark.parametrize("function", [False, True])
def test_sparse_dot_tile_gpu_replay(dtype, prune, graph, terms, function, monkeypatch):
    from ren.backends.cuda.graph import CUDAQueueGraphRunner

    if graph == "queue":
        from ren.backends.cuda import api as cuda

        def fail_graph_instantiate(_graph, _stream):
            raise RuntimeError("forced graph failure")

        monkeypatch.setattr(cuda, "graph_instantiate", fail_graph_instantiate)

    ring = large_prime_ring(128, dtype)[:3]
    rng = np.random.default_rng(987)
    # The two middle rows omit different inputs, so their reductions share only some terms.
    rows = [tuple(b for b in range(terms) if b != skip) for skip in (-1, 1, 2, -1)]
    weights = [[Poly(rng.integers(-100, 100, ring.n), ring, "cuda:0").ntt() for _ in row] for row in rows]
    ren.realize(*(p for row in weights for p in row))

    def compute(*xs):
        outputs = [reduce(add, [w * xs[2 * b + c] for w, b in zip(weights[g], row)]) for g, row in enumerate(rows) for c in range(2)]
        # Both the original and rescaled outputs must survive the next replay.
        return tuple(outputs + [p.rescale(ring.moduli[:-1]) for p in outputs])

    traced = ren.function(compute) if function else compute
    baseline = ren.jit(traced, prune=prune)
    tiled = ren.jit(traced, prune=prune)
    retained = None
    retained_bytes = None
    for iteration in range(3):
        xs = [Poly(rng.integers(-100, 100, ring.n), ring, "cuda:0").ntt() for _ in range(2 * terms)]
        ren.realize(*xs)
        with pytest.warns(UserWarning, match="falling back to queue mode") if graph == "queue" and iteration == 0 else nullcontext():
            with Context(DOT_TILE=0, GRAPH=int(graph != 0)):
                reference = baseline(*xs)
            with Context(DOT_TILE=8, GRAPH=int(graph != 0)):
                actual = tiled(*xs)
        actual_bytes = tuple(bytes(p.data()) for p in actual)
        assert actual_bytes == tuple(bytes(p.data()) for p in reference)
        if retained is not None:
            assert tuple(bytes(p.data()) for p in retained) == retained_bytes
        retained = actual
        retained_bytes = actual_bytes
    for fn, expected_outputs in ((baseline, 1), (tiled, 8)):
        plan = next(iter(fn._plans.values()))
        assert max(len(unit.call.src[0].src) for unit in plan.units if unit.root_op is Op.SINK) == expected_outputs
        runner = plan.runner
        if graph == 0:
            assert runner is None
        else:
            assert isinstance(runner, CUDAQueueGraphRunner)
            assert (runner._graph_exec is not None) == (graph == 1)
