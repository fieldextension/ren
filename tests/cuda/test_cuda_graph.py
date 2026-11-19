import ctypes
import gc
from array import array
import weakref
from collections.abc import Sequence
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

pytest.importorskip("cuda.bindings")

from ren.allocator import BufferSpec
import ren
from ren.backends.cuda import SUPPORTED_DTYPES
from ren.backends.cuda import api as cuda_api
from ren.backends.cuda import graph as cuda_graph
from ren.backends.cuda.backend import CUDABackend, CUDAKernelProgram, CUDAKernelSpec
from ren.backends.cuda.queue import Signal
from ren.device import Device
from ren.backends.cuda.graph import CUDAQueueGraphRunner, _compute_unit_deps
from ren.backends.cuda.renderer import CUDARenderer, RenderedKernel
from ren.buffer import Buffer
from ren.context import Context
from ren.dtype import dtypes
from ren.engine.realize import ExecUnit
from ren.engine.codegen import lower_kernel
from ren.ir import Op
from ren.kernel import KernelInfo
from ren.ir.node import Node, ParamArg
from ren.poly import Poly
from ren.ring import Domain

from tests.utils import _ring


def _buf(size: int, *, offset: int = 0, base: Buffer | None = None) -> Buffer:
    spec = BufferSpec(dtype=dtypes.uint8, size=size, device="python")
    return Buffer(spec, base=base, offset=offset) if base is not None else Buffer(spec)


def _unit(op: Op, bufs: list[Buffer], writes: tuple[int, ...] = (0,)) -> ExecUnit:
    args = tuple(Node.from_buffer(buf) for buf in bufs)
    body = Node.sink(*(Node.param(i, args[i]).store(Node.const(0, args[i].dtype)) for i in writes)) if op is Op.SINK else Node(op)
    return ExecUnit(program=lambda bufs: None, call=Node(Op.CALL, src=(body, *args)))


def _unit_deps(units: Sequence[ExecUnit]) -> list[list[int]]:
    return _compute_unit_deps(units, [unit.resolve(()) for unit in units])


def test_compute_unit_deps_tracks_every_output():
    x, a, b, y = [_buf(8) for _ in range(4)]
    units = [
        _unit(Op.SINK, [a, x, b], (0, 2)),
        _unit(Op.SINK, [y, b]),
    ]

    assert _unit_deps(units) == [[], [0]]


def test_compute_unit_deps_ignores_disjoint_views_of_same_arena():
    arena = _buf(256)
    left = _buf(64, offset=0, base=arena)
    right = _buf(64, offset=128, base=arena)
    src = _buf(64)

    deps = _unit_deps([
        _unit(Op.COPY, [left, src]),
        _unit(Op.SINK, [right, src]),
    ])

    assert deps == [[], []]


def test_compute_unit_deps_tracks_overlapping_views_of_same_arena():
    arena = _buf(256)
    written = _buf(96, offset=32, base=arena)
    read = _buf(96, offset=64, base=arena)
    src = _buf(96)

    deps = _unit_deps([
        _unit(Op.COPY, [written, src]),
        _unit(Op.SINK, [read, src]),
    ])

    assert deps == [[], [0]]


def test_compute_unit_deps_keeps_unoverwritten_read_ranges():
    arena = _buf(256)
    full = _buf(128, offset=0, base=arena)
    left = _buf(64, offset=0, base=arena)
    right = _buf(64, offset=64, base=arena)
    src = _buf(128)

    deps = _unit_deps([
        _unit(Op.SINK, [_buf(128), full]),
        _unit(Op.COPY, [left, src]),
        _unit(Op.COPY, [right, src]),
    ])

    assert deps == [[], [0], [0]]


def test_generated_kernel_name_is_stable():
    """A domain-only difference must not churn the symbol name or rendered source."""
    ring = _ring(8, [60, 60])  # uint64
    gidx = Node(Op.GIDX, dtypes.index)
    in_slot = Node(Op.PARAM, ring.dtype, arg=ParamArg(0, ring.storage_size, "cuda:0"))
    out_slot = Node(Op.PARAM, ring.dtype, arg=ParamArg(1, ring.storage_size, "cuda:0"))
    load = in_slot.index(gidx).load()
    root = Node(Op.MOD_REDUCE, ring.dtype, src=(load,))
    store = out_slot.index(gidx).store(root)

    renderer = CUDARenderer("sm_89")

    def render(domain: Domain):
        body = Node(Op.SINK, src=(store,), arg=KernelInfo("cuda:0", ring, domain))
        spec = lower_kernel(body, SUPPORTED_DTYPES)
        (kernel,) = renderer.render(spec)
        return kernel

    a, b = render(Domain.COEFF), render(Domain.NTT)
    assert a.name == "&ren_generated_u64"
    assert a.source == b.source and a.cache_key == b.cache_key


def _delay(backend: CUDABackend) -> Signal:
    cuda_api.ctx_set_current(backend.context)
    delay = backend._compile(
        RenderedKernel(
            "__attribute__((global)) void delay() { unsigned long long start = clock64(); while (clock64() - start < 1000000000ULL) {} }",
            "&delay",
        )
    )
    done = Signal()
    backend.synchronize()
    cuda_api.launch_kernel(delay, (1,), (1,), [], backend.compute_queue.stream)
    backend.compute_queue.signal(done)

    return done


@pytest.mark.requires_cuda
@pytest.mark.parametrize("fallback", [False, True], ids=["graph", "queue"])
def test_jit_reuses_allocations_with_gpu_work_pending(monkeypatch, fallback):
    ring = _ring(64, [60])
    backend = Device["cuda:0"]
    assert isinstance(backend, CUDABackend)
    if fallback:
        monkeypatch.setattr(cuda_api, "graph_instantiate", Mock(side_effect=RuntimeError("forced graph failure")))

    @ren.jit
    def add_one(poly):
        return poly + 1

    x = Poly([2] * ring.n, ring, device="cuda:0").realize()
    with pytest.warns(UserWarning, match="falling back to queue mode") if fallback else nullcontext():
        add_one(x)
    retained = add_one(x)
    # Reserve all storage before the delay: cudaMalloc can synchronize the device.
    reserve = [Buffer(x.node.base.buffer.spec).ensure_allocated() for _ in range(40)]
    del reserve
    done = _delay(backend)

    outputs = []
    try:
        for _ in range(16):
            temporary = add_one(x)
            address = temporary.node.base.buffer._buf
            buffer_ref = weakref.ref(temporary.node.base.buffer)
            outputs.append(add_one(temporary))
            del temporary
            assert buffer_ref() is None
            # Reuse the released input on the copy queue while its readers are pending.
            replacement = Poly([9] * ring.n, ring, device="cuda:0").realize()
            assert replacement.node.base.buffer._buf == address
            del replacement
        assert cuda_api._drv.cuEventQuery(done._event)[0] == cuda_api._drv.CUresult.CUDA_ERROR_NOT_READY
    finally:
        backend.synchronize()
        done.close()
    assert retained.tolist() == [3] + [2] * (ring.n - 1)
    assert all(output.tolist() == [4] + [2] * (ring.n - 1) for output in outputs)


@pytest.mark.requires_multi_gpu
def test_jit_collection_preserves_active_cuda_context():
    source, destination = Device["cuda:0"], Device["cuda:1"]
    assert isinstance(source, CUDABackend) and isinstance(destination, CUDABackend)
    ring = _ring(64, [60])
    jitted = ren.jit(lambda x: x + 1)
    jitted(Poly([1] * ring.n, ring, device=source.device))
    source.synchronize()
    cycle: list[object] = [jitted]
    cycle.append(cycle)
    del jitted, cycle

    cuda_api.ctx_set_current(destination.context)
    gc.collect()
    signal = Signal()
    try:
        destination.compute_queue.signal(signal)
        signal.synchronize()
    finally:
        signal.close()


@pytest.mark.requires_multi_gpu
@pytest.mark.parametrize("fallback", [False, True], ids=["graph", "queue"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_cross_device_copies_release_source_before_destination_compute(monkeypatch, fallback, dynamic):
    source, destination = Device["cuda:0"], Device["cuda:1"]
    assert isinstance(source, CUDABackend) and isinstance(destination, CUDABackend)
    if fallback:
        monkeypatch.setattr(cuda_api, "graph_instantiate", Mock(side_effect=RuntimeError("forced graph failure")))

    rendered = RenderedKernel(
        """__attribute__((global)) void delayed_add(unsigned int *out, const unsigned int *a, const unsigned int *b,
                                                   unsigned long long cycles) {
          unsigned long long start = clock64();
          while (clock64() - start < cycles) {}
          for (int i = 0; i < 4; ++i) out[i] = a[i] + b[i];
        }""",
        "&delayed_add",
    )
    cuda_api.ctx_set_current(destination.context)
    kernel = destination._compile(rendered)
    done = Signal()
    inputs = [Buffer(BufferSpec(dtypes.uint32, 8, source.device)).ensure_allocated() for _ in range(2)]
    left, right, output = [Buffer(BufferSpec(dtypes.uint32, 4, destination.device)).ensure_allocated() for _ in range(3)]
    input_node = Node.from_buffer(inputs[0])
    if dynamic:
        input_node = Node.param(0, input_node)
    a, b = [Node(Op.BUFFER_VIEW, dtypes.uint32, src=(input_node,), arg=(4, offset)) for offset in (0, 4)]

    def copy(dst, src):
        return ExecUnit(None, Node.param(1, src).copy_to_device(destination.device).call(Node.from_buffer(dst), src))

    def compute(dst, a, b, cycles):
        unit = _unit(Op.SINK, [dst, a, b])
        spec = CUDAKernelSpec(kernel, (1,), (1,), 0, [ctypes.c_void_p() for _ in range(3)] + [ctypes.c_ulonglong(cycles)], (0, 1, 2))
        return ExecUnit(CUDAKernelProgram([spec], destination), unit.call)

    # Delay the second copy, then keep computing after both copies have finished.
    units = (copy(left, a), compute(right, left, left, 100000000), copy(right, b), compute(output, left, right, 1000000000))
    with pytest.warns(UserWarning, match="falling back to queue mode") if fallback else nullcontext():
        runner = CUDAQueueGraphRunner(units, [inputs[0]])
    assert (runner._graph_exec is None) == fallback
    try:
        for iteration in range(3):
            src = inputs[iteration % 2] if dynamic else inputs[0]
            values = array("I", range(iteration * 10, iteration * 10 + 8))
            src.copyin(memoryview(values))
            runner([src])
            destination.compute_queue.signal(done)
            # Reuse the source while the destination is still computing on its copies.
            src.copyin(memoryview(array("I", [99] * 8)))
            source.synchronize()
            assert cuda_api._drv.cuEventQuery(done._event)[0] == cuda_api._drv.CUresult.CUDA_ERROR_NOT_READY
            assert list(output.as_typed_buffer()) == [values[i] + values[i + 4] for i in range(4)]
    finally:
        source.synchronize()
        destination.synchronize()
        runner.close()
        done.close()


@pytest.mark.requires_multi_gpu
@pytest.mark.parametrize("mode", ["graph", "queue", "eager"])
def test_cross_device_copy_protects_reused_source(monkeypatch, mode):
    ring = _ring(64, [60])
    source, destination = Device["cuda:0"], Device["cuda:1"]
    assert isinstance(source, CUDABackend) and isinstance(destination, CUDABackend)
    if mode == "queue":
        monkeypatch.setattr(cuda_api, "graph_instantiate", Mock(side_effect=RuntimeError("forced graph failure")))

    @ren.jit
    def produce(poly):
        return poly + 1

    def copy(poly):
        return poly.to("cuda:1") + 1

    consume = copy if mode == "eager" else ren.jit(copy)
    a, b = [Poly([value] * ring.n, ring, device="cuda:0").realize() for value in (2, 9)]
    with pytest.warns(UserWarning, match="falling back to queue mode") if mode == "queue" else nullcontext():
        consume(produce(a)).realize()
    consume(produce(a)).realize()
    for backend in (source, destination):
        spec = BufferSpec(ring.dtype, ring.storage_size, backend.device)
        reserve = [Buffer(spec).ensure_allocated() for _ in range(8)]
        del reserve
    source.synchronize()
    done = _delay(destination)
    outputs, replacements = [], []
    try:
        for _ in range(3):
            temporary = produce(a)
            address = temporary.node.base.buffer._buf
            outputs.append(consume(temporary).realize())
            del temporary
            cuda_api.ctx_set_current(destination.context)
            assert cuda_api._drv.cuEventQuery(done._event)[0] == cuda_api._drv.CUresult.CUDA_ERROR_NOT_READY
            replacements.append(produce(b))
            assert replacements[-1].node.base.buffer._buf == address
    finally:
        source.synchronize()
        destination.synchronize()
        done.close()
    assert all(output.tolist() == [4] + [2] * (ring.n - 1) for output in outputs)
    assert all(replacement.tolist() == [10] + [9] * (ring.n - 1) for replacement in replacements)


@pytest.mark.requires_multi_gpu
@pytest.mark.parametrize("fallback", [False, True], ids=["graph", "queue"])
def test_bidirectional_replays_protect_reused_sources(monkeypatch, fallback):
    ring = _ring(64, [60])
    zero, one = Device["cuda:0"], Device["cuda:1"]
    assert isinstance(zero, CUDABackend) and isinstance(one, CUDABackend)
    backends = (zero, one)
    if fallback:
        monkeypatch.setattr(cuda_api, "graph_instantiate", Mock(side_effect=RuntimeError("forced graph failure")))

    @ren.jit
    def to_one(poly):
        return poly.to("cuda:1") + 1

    @ren.jit
    def to_zero(poly):
        return poly.to("cuda:0") + 1

    transfers = (to_one, to_zero)
    producers = (ren.jit(lambda x: x + 1), ren.jit(lambda x: x + 1))
    seeds = [Poly([2] * ring.n, ring, device=backend.device).realize() for backend in backends]
    overwrite = [Poly([98, *([99] * (ring.n - 1))], ring, device=backend.device).realize() for backend in backends]
    with pytest.warns(UserWarning, match="falling back to queue mode") if fallback else nullcontext():
        for transfer, producer, seed in zip(transfers, producers, seeds, strict=True):
            transfer(producer(seed))
    for transfer in transfers:
        runner = next(iter(transfer._plans.values())).runner
        assert isinstance(runner, CUDAQueueGraphRunner)
        assert (runner._graph_exec is None) == fallback
    current = producers[0](seeds[0])
    for backend in backends:
        # Allocate before launch so the work stays pending during the reuse check.
        reserve = [Buffer(BufferSpec(ring.dtype, ring.storage_size, backend.device)).ensure_allocated() for _ in range(16)]
        del reserve
        backend.synchronize()
    signals = [_delay(backend) for backend in backends]
    replacements = []
    copied = []
    try:
        for iteration, transfer in enumerate(transfers * 4):
            source = iteration % 2
            address = current.node.base.buffer._buf
            result = transfer(current)
            copied.append(result)
            del current
            replacement = producers[source](overwrite[source])
            assert replacement.node.base.buffer._buf == address
            replacements.append(replacement)
            current = producers[1 - source](result)
            del result
        assert all(cuda_api._drv.cuEventQuery(signal._event)[0] == cuda_api._drv.CUresult.CUDA_ERROR_NOT_READY for signal in signals)
    finally:
        for backend, signal in zip(backends, signals, strict=True):
            backend.synchronize()
            signal.close()
        for transfer in transfers:
            transfer.clear_jit()
    assert current.tolist() == [19] + [2] * (ring.n - 1)
    assert all(replacement.tolist() == [99] * ring.n for replacement in replacements)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("fallback", [False, True], ids=["graph", "queue"])
@pytest.mark.parametrize("source_device", ["python", "cuda:0"])
def test_graph_runner_copy_then_compute(monkeypatch: pytest.MonkeyPatch, fallback: bool, source_device: str):
    ring = _ring(64, [60])

    @ren.jit
    def copy_and_add(poly):
        return poly.to("cuda:0") + 1

    build_sync = Mock(wraps=cuda_graph._build_queue_sync)
    monkeypatch.setattr(cuda_graph, "_build_queue_sync", build_sync)
    if fallback:
        monkeypatch.setattr(cuda_api, "graph_instantiate", Mock(side_effect=RuntimeError("forced graph failure")))

    with pytest.warns(UserWarning, match="falling back to queue mode") if fallback else nullcontext():
        copy_and_add(Poly([0] * ring.n, ring, device=source_device))

    plan = next(iter(copy_and_add._plans.values()))
    runner = plan.runner
    assert isinstance(runner, CUDAQueueGraphRunner)
    assert (runner._graph_exec is None) == fallback
    assert build_sync.call_count == int(fallback)

    inputs = [Poly([value] * ring.n, ring, device=source_device).realize() for value in (2, 5, 9)]
    record = Mock(wraps=cuda_api.event_record)
    synchronize = Mock(wraps=cuda_api.event_synchronize)
    monkeypatch.setattr(cuda_api, "event_record", record)
    monkeypatch.setattr(cuda_api, "event_synchronize", synchronize)
    outputs = [copy_and_add(poly) for poly in inputs]
    if not fallback:
        assert record.call_count == synchronize.call_count == (len(inputs) if source_device == "python" else 0)
    for output, value in zip(outputs, (2, 5, 9), strict=True):
        assert output.tolist() == [value + 1] + [value] * (ring.n - 1)
    copy_and_add.clear_jit()


@pytest.mark.requires_cuda
def test_graph_runner_static_host_copy_and_kernel():
    ring = _ring(64, [60])
    source = Poly([3] * ring.n, ring, device="python").realize()
    one = Poly.const(1, ring, device="cuda:0")

    @ren.jit(static_output=True)
    def copy_and_add(scalar):
        return source.to("cuda:0") + scalar

    copy_and_add(one)
    runner = next(iter(copy_and_add._plans.values())).runner
    assert isinstance(runner, CUDAQueueGraphRunner)
    assert runner._graph_exec is not None
    gc.collect()
    for _ in range(3):
        assert copy_and_add(one).tolist() == [4] + [3] * (ring.n - 1)
    copy_and_add.clear_jit()


@pytest.mark.requires_cuda
@pytest.mark.parametrize("mode", ["graph", "queue"])
@pytest.mark.parametrize("source_device", ["python", pytest.param("cuda:1", marks=pytest.mark.requires_multi_gpu)])
def test_mixed_copy_compute_replays_fresh_inputs(mode, source_device, monkeypatch: pytest.MonkeyPatch):
    ring = _ring(64, [60])

    if mode == "queue":

        def fail_graph_instantiate(_graph, _stream):
            raise RuntimeError("forced graph failure")

        monkeypatch.setattr(cuda_api, "graph_instantiate", fail_graph_instantiate)

    @ren.jit
    def add_one(poly):
        return poly.to("cuda:0") + 1

    retained = None
    with Context(GRAPH=1):
        for iteration in range(3):
            values = [iteration * 100 + i for i in range(ring.n)]
            source = Poly(values, ring, source_device).realize()
            with pytest.warns(UserWarning, match="falling back to queue mode") if mode == "queue" and iteration == 0 else nullcontext():
                result = add_one(source)
            assert result.tolist() == [values[0] + 1, *values[1:]]
            if retained is not None:
                assert retained.tolist() == [1, *range(1, ring.n)]
            else:
                retained = result

    plan = next(iter(add_one._plans.values()))
    assert {unit.root_op for unit in plan.units} == {Op.COPY, Op.SINK}
    runner = plan.runner
    assert isinstance(runner, CUDAQueueGraphRunner)
    assert (runner._graph_exec is not None) == (mode == "graph")


@pytest.mark.requires_cuda
def test_graph_runner_does_not_retain_dynamic_input_buffers():
    ring = _ring(64, [60])

    @ren.jit
    def add_one(poly):
        return poly + 1

    def run() -> weakref.ReferenceType[Buffer]:
        poly = Poly([0] * ring.n, ring).realize()
        ref = weakref.ref(poly.node.base.buffer)
        add_one(poly)
        return ref

    for _ in range(2):
        ref = run()
        gc.collect()
        assert ref() is None

    plan = next(iter(add_one._plans.values()))
    runner = plan.runner
    assert isinstance(runner, CUDAQueueGraphRunner)
    assert runner._graph_exec is not None


@pytest.mark.requires_cuda
def test_graph_runner_resolves_shared_bindings_once_per_replay(monkeypatch: pytest.MonkeyPatch):
    ring = _ring(64, [60])
    captures = []

    @ren.function
    def double(poly):
        return poly + poly

    @ren.jit
    def jitted(poly):
        captures.append(None)
        doubled = double(poly).realize()
        return doubled, (poly * 3).realize()

    outputs = [jitted(Poly([1] * ring.n, ring))]

    plan = next(iter(jitted._plans.values()))
    assert isinstance(plan.runner, CUDAQueueGraphRunner)
    assert plan.runner._graph_exec is not None

    resolved = []
    resolve = cuda_graph.resolve_buffer

    def tracked_resolve(node, bindings):
        resolved.append(node)
        return resolve(node, bindings)

    monkeypatch.setattr(cuda_graph, "resolve_buffer", tracked_resolve)
    for value in (3, 5):
        resolved.clear()
        outputs.append(jitted(Poly([value] * ring.n, ring)))
        assert resolved and len(resolved) == len(set(resolved))

    assert len(captures) == 1
    for value, (doubled, tripled) in zip((1, 3, 5), outputs, strict=True):
        assert doubled.tolist() == [2 * value] * ring.n
        assert tripled.tolist() == [3 * value] * ring.n


@pytest.mark.requires_cuda
@pytest.mark.parametrize("source_device", ["python", "cuda:0"])
def test_graph_runner_rebinds_copy_views(source_device, cuda_device, monkeypatch: pytest.MonkeyPatch):
    source_spec = BufferSpec(dtype=dtypes.uint8, size=16, device=source_device)
    destination_spec = BufferSpec(dtype=dtypes.uint8, size=12, device=cuda_device.device)
    source = Node.param(0, Node.from_buffer(Buffer(source_spec)))
    destination = Node.param(1, Node.from_buffer(Buffer(destination_spec)))

    def view(node, size, offset):
        return Node(Op.BUFFER_VIEW, node.dtype, src=(node,), arg=(size, offset))

    inner = view(source, 12, 2)
    left, right = view(inner, 4, 1), view(inner, 4, 5)
    units: tuple[ExecUnit, ...] = tuple(
        ExecUnit(lambda bufs: None, Node(Op.CALL, src=(Node.param(1, src).copy_to_device(cuda_device.device), view(destination, 4, offset), src)))
        for offset, src in ((0, left), (4, left), (8, right))
    )
    runner = CUDAQueueGraphRunner(units, [Buffer(source_spec), Buffer(destination_spec)])
    assert runner._graph_exec is not None
    resolved = []
    resolve = cuda_graph.resolve_buffer

    def tracked_resolve(node, bindings):
        resolved.append(node)
        return resolve(node, bindings)

    monkeypatch.setattr(cuda_graph, "resolve_buffer", tracked_resolve)
    outputs = []
    try:
        for seed, offset in ((0, 1), (100, 5)):
            storage = Buffer(BufferSpec(dtype=dtypes.uint8, size=32, device=source_device))
            storage.copyin(memoryview(bytes(range(seed, seed + 32))))
            output = Buffer(destination_spec)
            resolved.clear()
            runner([storage.view(16, offset), output])
            cuda_device.synchronize()
            assert resolved and len(resolved) == len(set(resolved))
            start = seed + offset + 3
            outputs.append((output, list(range(start, start + 4)) * 2 + list(range(start + 4, start + 8))))

        with pytest.raises(ValueError, match="does not match its parameter"):
            runner([storage.view(15, 0), output])
        with pytest.raises(RuntimeError, match="missing execution buffer"):
            runner([])
        for output, expected in outputs:
            assert list(output.as_buffer()) == expected
    finally:
        runner.close()


@pytest.mark.requires_cuda
def test_graph_runner_queue_fallback_does_not_retain_dynamic_input_buffers(monkeypatch: pytest.MonkeyPatch):
    ring = _ring(64, [60])

    @ren.jit
    def add_one(poly):
        return poly + 1

    def fail_graph_instantiate(_graph, _stream):
        raise RuntimeError("forced graph failure")

    monkeypatch.setattr(cuda_api, "graph_instantiate", fail_graph_instantiate)
    with pytest.warns(UserWarning, match="falling back to queue mode"):
        add_one(Poly([0] * ring.n, ring))

    plan = next(iter(add_one._plans.values()))
    runner = plan.runner
    assert isinstance(runner, CUDAQueueGraphRunner)
    assert runner._graph_exec is None

    def replay() -> weakref.ReferenceType[Buffer]:
        poly = Poly([1] * ring.n, ring).realize()
        ref = weakref.ref(poly.node.base.buffer)
        assert add_one(poly).tolist()[0] == 2
        return ref

    ref = replay()
    gc.collect()

    assert ref() is None
