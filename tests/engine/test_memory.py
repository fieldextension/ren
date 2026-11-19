import pytest

from ren.buffer import Buffer
from ren.allocator import BufferSpec
from ren.dtype import dtypes
from ren.device import Device
from ren.engine.memory import plan_memory
from ren.ir.node import Node, ParamArg
from ren.kernel import KernelInfo
from ren.ring import Domain
from ren.engine.schedule import schedule
from ren.engine.realize import make_exec_units, unwrap_view
from ren.ir import Op
from ren.poly import Poly, realize
from tests.utils import large_prime_ring


def plan_buffers(buffers):
    body = Node(Op.SINK)
    return plan_memory(Node(Op.LINEAR, src=tuple(Node(Op.CALL, src=(body, *(Node.from_buffer(buf) for buf in bufs))) for bufs in buffers)))


def test_memory_planner_basic():
    spec = BufferSpec(dtype=dtypes.uint8, size=16, device="python")

    # Unallocated buffers are eligible for shared storage.
    buf0 = Buffer(spec)
    buf1 = Buffer(spec)
    buf2 = Buffer(spec)

    # Simulate kernel sequence where buf0 dies before buf2 is needed
    buffers = [
        [buf0, buf1],  # kernel 0: buf0 last use
        [buf1, buf2],  # kernel 1: buf2 first use, buf0 is dead
    ]

    assigned = plan_buffers(buffers)

    first, live, reused = (assigned[buf] for buf in (buf0, buf1, buf2))
    assert first.base is live.base is reused.base
    # buf0 and buf2 reuse a range because their lifetimes don't overlap.
    assert first._offset == reused._offset
    assert first.nbytes == live.nbytes == reused.nbytes == buf0.nbytes
    # buf1 overlaps both lifetimes and must occupy a disjoint range.
    assert first._offset + first.nbytes <= live._offset or live._offset + live.nbytes <= first._offset


def test_memory_planner_peak_sizing():
    """Arena should be sized at peak concurrent usage, not total of all buffers."""
    # 3 buffers of 1024 bytes each, but non-overlapping lifetimes:
    # buf0 lives in kernel 0 only, buf1 lives in kernels 0-1, buf2 lives in kernel 2 only
    # Peak concurrent = buf0 + buf1 = 2048, not buf0 + buf1 + buf2 = 3072
    spec = BufferSpec(dtype=dtypes.uint8, size=1024, device="python")
    buf0 = Buffer(spec)
    buf1 = Buffer(spec)
    buf2 = Buffer(spec)

    buffers = [
        [buf0, buf1],  # kernel 0: buf0 and buf1 alive
        [buf1],  # kernel 1: buf0 dead, buf1 alive
        [buf2],  # kernel 2: buf1 dead, buf2 alive (reuses buf0's slot)
    ]

    assigned = plan_buffers(buffers)

    # All 3 buffers should be assigned into the arena
    assert buf0 in assigned and buf1 in assigned and buf2 in assigned

    # All should share the same arena base
    arena = assigned[buf0].base
    assert assigned[buf1].base is arena
    assert assigned[buf2].base is arena

    # Arena should be sized at peak (2 buffers), not total (3 buffers)
    # With python device, min_block=1, so arena should be exactly 2048
    assert arena.nbytes == 2048, f"expected arena=2048, got {arena.nbytes}"


def test_memory_planning_preserves_after_ast(monkeypatch):
    ring = large_prime_ring(128, dtypes.uint32)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "python")) for i in range(2)]
    value = slots[1].index(idx).load()
    ast = Node(Op.SINK, src=(slots[0].index(idx).store(value.after(value)),), arg=KernelInfo("python", ring, Domain.NTT))
    bufs = [Buffer(BufferSpec(dtype=ring.dtype, size=ring.storage_size, device="python")) for _ in slots]
    call = Node(Op.CALL, src=(ast, *(Node.from_buffer(buf) for buf in bufs)))
    linear = Node(Op.LINEAR, src=(call,))
    assigned = plan_memory(linear)
    compiled = []

    def compile(body):
        compiled.append(body)
        return lambda bufs: None

    monkeypatch.setattr(Device["python"], "make_kernel", compile)
    (planned,) = make_exec_units(linear, assigned)
    assert compiled == [ast] and planned.call.src[0] is ast
    assert all(a is not b for a, b in zip(planned.resolve(), bufs, strict=True))


def test_memory_planner_optimizes_scheduled_intermediates():
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring, device="python")
    output = (x + x) * (x + 1)
    linear, becomes = schedule(Node.sink(output.node))
    output_buffer = becomes[output.node].base.buffer

    assigned = plan_memory(linear, {output_buffer})

    assert output_buffer not in assigned
    assert assigned
    assert all(view.base is not view for view in assigned.values())


def test_memory_planner_optimizes_buffer_view_storage():
    ring = large_prime_ring(64)
    small_ring = ring[:2]
    x = Poly([1] * ring.n, ring, device="python")
    viewed = (x + x).contiguous().to_quotient_ring(small_ring)
    linear, _ = schedule(Node.sink((viewed + viewed).node))
    binding = next(source for call in linear.src for source in call.inputs if source.op is Op.BUFFER_VIEW)
    source = unwrap_view(binding)[0].buffer

    assigned = plan_memory(linear)

    assert source in assigned
    units = make_exec_units(linear, assigned)
    unit, slot = next((unit, slot) for unit in units for slot, source in enumerate(unit.call.inputs) if source.op is Op.BUFFER_VIEW)
    view = unit.resolve()[slot]
    assert view.base is assigned[source].base
    assert view.size == binding.arg[0]
    assert view.offset == assigned[source].offset + binding.arg[1]


def test_memory_planner_does_not_capture_live_lazy_intermediates():
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring, device="python")
    shared = x + x
    left, right = shared * 2, shared * 3

    realize(left, right)

    assert list(left) == [4, 8, 12, 16]
    assert list(right) == [6, 12, 18, 24]
    assert list(shared) == [2, 4, 6, 8]
    assert list(shared + shared) == [4, 8, 12, 16]


@pytest.mark.parametrize("kind", ["direct", "view"])
def test_memory_planner_excludes_unbound_parameters(kind):
    output = Node.new_buffer("python", 16, dtypes.uint8)
    parameter = Node(Op.PARAM, output.dtype, arg=ParamArg(0, 16 if kind == "direct" else 32, "python"))
    source = parameter
    if kind == "view":
        source = Node(Op.BUFFER_VIEW, output.dtype, src=(parameter,), arg=(16, 8))
    call = Node(Op.CALL, src=(Node(Op.SINK), output, source))

    assigned = plan_memory(Node(Op.LINEAR, src=(call,)))

    assert set(assigned) == {output.buffer}
