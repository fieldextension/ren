import numpy as np
import pytest

from ren.buffer import Buffer, BufferSpec
from ren.device import Device
from ren.dtype import dtypes
from ren.engine.realize import ExecUnit
from ren.ir import Node, Op
from ren.ir.node import ParamArg
from ren.kernel import KernelInfo
from ren.ring import Domain
from tests.utils import large_prime_ring


@pytest.fixture(params=["python", pytest.param("cuda:0", marks=pytest.mark.requires_cuda)])
def device(request):
    return request.param


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("outputs", [1, 2])
def test_generated_kernel_indexing_and_conditionals(device, dtype, outputs):
    ring = large_prime_ring(128, dtype)[:3]
    idx = Node(Op.GIDX, dtypes.index)
    sizes = [ring.storage_size] * outputs + [ring.storage_size + ring.n, ring.storage_size]
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, size, device)) for i, size in enumerate(sizes)]
    left = slots[-2].index(idx + ring.n).load()
    # Rotate within each limb while the other input selects a contiguous segment.
    n = Node.const(ring.n, dtypes.index)
    limb = Node(Op.IDIV, dtypes.index, src=(idx, n))
    rotated = limb * n + Node(Op.MOD, dtypes.index, src=(idx + 1, n))
    right = slots[-1].index(rotated).load()
    at_start = Node(Op.CMPEQ, dtypes.bool, src=(Node(Op.MOD, dtypes.index, src=(idx, n)), Node.const(0, dtypes.index)))
    value = Node(Op.WHERE, dtype, src=(at_start, left, (left + right).mod_reduce(None)))
    values = [value, (-value).mod_reduce(None)]
    ast = Node(Op.SINK, src=tuple(s.index(rotated).store(v) for s, v in zip(slots[:outputs], values)), arg=KernelInfo(device, ring, Domain.COEFF))
    backend = Device[device]
    kernel = backend.make_kernel(ast)
    bufs = [Buffer(BufferSpec(dtype=dtype, size=size, device=device)) for size in sizes]
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    a = np.concatenate([np.arange(ring.n, dtype=np_dtype) + p - ring.n for p in ring.moduli])
    b = np.tile(np.arange(ring.n, dtype=np_dtype), len(ring.moduli))
    bufs[-2].copyin(memoryview(np.concatenate([np.zeros(ring.n, dtype=np_dtype), a])).cast("B"))
    bufs[-1].copyin(memoryview(b).cast("B"))
    ExecUnit(kernel, ast.call(*(Node.from_buffer(buf) for buf in bufs)))()
    backend.synchronize()
    expected = []
    for p, row in zip(ring.moduli, a.reshape(-1, ring.n)):
        expected.extend(int(x) if i == 0 else (int(x) + (i + 1) % ring.n) % p for i, x in enumerate(row))
    negative = [(-x) % p for p, row in zip(ring.moduli, np.array(expected).reshape(-1, ring.n)) for x in map(int, row)]
    for buf, result in zip(bufs[:outputs], (expected, negative)):
        scattered = np.roll(np.array(result, dtype=np_dtype).reshape(-1, ring.n), 1, axis=1)
        assert bytes(buf.as_buffer()) == scattered.tobytes()


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("start", [0, 1, 2])
def test_generic_multi_output_kernel_executes_load_and_add(device, dtype, materialized, start):
    backend = Device[device]
    source_ring = large_prime_ring(128, dtype)[:3]
    ring = source_ring[start : start + 1]
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, (source_ring if i in (1, 2) else ring).storage_size, device)) for i in range(4)]
    load_idx = idx + start * ring.n if start else idx
    a, b = (s.index(load_idx).load() for s in slots[1:3])
    value = (a + b).mod_reduce(None)
    stores = (slots[0].index(idx).store(a), slots[3].index(idx).store(value))
    ast = Node(Op.SINK, src=stores, arg=KernelInfo(device, ring, Domain.NTT))
    if materialized:
        defined = a.after(a)
        ast = ast.replace(
            src=(slots[0].index(idx).store(defined), slots[3].index(idx).store((defined + b).mod_reduce(None).after(defined).mod_reduce(None)))
        )
    kernel = backend.make_kernel(ast)
    bufs = [Buffer(BufferSpec(dtype=dtype, size=slot.arg.size, device=device)) for slot in slots]
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    left = np.arange(source_ring.storage_size, dtype=np_dtype)
    right = np.full(source_ring.storage_size, ring.moduli[0] - 1, dtype=np_dtype)
    bufs[1].copyin(memoryview(left).cast("B"))
    bufs[2].copyin(memoryview(right).cast("B"))
    ExecUnit(kernel, Node(Op.CALL, src=(ast, *(Node.from_buffer(buf) for buf in bufs))))()
    backend.synchronize()
    left, right = (values[start * ring.n : (start + 1) * ring.n] for values in (left, right))
    assert bytes(bufs[0].as_buffer()) == left.tobytes()
    assert bytes(bufs[3].as_buffer()) == ((left + right) % ring.moduli[0]).tobytes()


def test_generated_outputs_keep_their_own_dtype(device):
    ring = large_prime_ring(128, dtypes.uint64)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    types = (dtypes.uint64, dtypes.uint32, dtypes.uint64)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, device)) for i, dtype in zip((0, 2, 3), types)]
    value = slots[0].index(idx).load()
    stores = (slots[1].index(idx).store(value.cast(dtypes.uint32)), slots[2].index(idx).store(value >> 32))
    ast = Node(Op.SINK, src=stores, arg=KernelInfo(device, ring, Domain.NTT))
    source = np.arange(ring.storage_size, dtype=np.uint64) * (2**32 + 3) + (2**32 - 1)
    buffers = [Buffer(BufferSpec(dtype=dtype, size=ring.storage_size, device=device)) for dtype in types]
    buffers[0].copyin(memoryview(source).cast("B"))
    backend = Device[device]
    bindings = (buffers[0], buffers[0], *buffers[1:])
    ExecUnit(backend.make_kernel(ast), ast.call(*(Node.from_buffer(buf) for buf in bindings)))()
    backend.synchronize()
    assert bytes(buffers[1].as_buffer()) == source.astype(np.uint32).tobytes()
    assert bytes(buffers[2].as_buffer()) == (source >> 32).tobytes()
