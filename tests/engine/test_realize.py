from array import array
import gc
import tracemalloc
import weakref

import numpy as np
import pytest

from ren.backends.python import PythonBackend
from ren.buffer import Buffer, BufferSpec
from ren.dtype import DType, dtypes, to_np_dtype
from ren.engine.realize import buffer_binding, resolve_buffer
from ren.ir import Node, Op
from ren.kernel import KernelInfo
from ren.ring import RingSpec
from ren.utils import disable_gc
from tests.utils import large_prime_ring


@pytest.mark.parametrize("readonly", [False, True], ids=["success", "failed_store"])
@disable_gc()
def test_python_kernel_releases_input_storage(readonly):
    ring = RingSpec(n=8, moduli=(17,))
    spec = BufferSpec(ring.dtype, ring.storage_size, "python")
    prototype = Node.new_buffer("python", ring.storage_size, ring.dtype)
    target, operand = (Node.param(i, prototype) for i in range(2))
    idx = Node(Op.GIDX, dtypes.index)
    value = operand.index(idx).load()
    body = Node.sink(target.index(idx).store(value + value)).replace(arg=KernelInfo("python", ring))
    kernel = PythonBackend().make_kernel(body)

    def execute():
        data = np.arange(ring.n, dtype=to_np_dtype(ring.dtype))
        source = Buffer(spec).allocate(opaque=data)
        output = Buffer(spec).allocate(opaque=bytes(source.nbytes) if readonly else None)
        if readonly:
            with pytest.raises(ValueError, match="read-only"):
                kernel([output, source])
        else:
            kernel([output, source])
        return weakref.ref(data), output

    source_ref, output = execute()
    assert source_ref() is None
    assert list(output.as_typed_buffer()) == ([0] * ring.n if readonly else [2 * i for i in range(ring.n)])


def test_python_kernel_bounds_temporary_memory():
    ring = RingSpec(n=4096, moduli=(65537,), dtype=dtypes.uint64)
    spec = BufferSpec(ring.dtype, ring.storage_size, "python")
    prototype = Node.new_buffer("python", ring.storage_size, ring.dtype)
    target, operand = (Node.param(i, prototype) for i in range(2))
    idx = Node(Op.GIDX, dtypes.index)
    value = operand.index(idx).load()
    for _ in range(32):
        value = value + value
    body = Node.sink(target.index(idx).store(value)).replace(arg=KernelInfo("python", ring))
    kernel = PythonBackend().make_kernel(body)
    source = Buffer(spec).allocate(opaque=np.arange(ring.n, dtype=to_np_dtype(ring.dtype)))
    output = Buffer(spec).allocate()

    tracemalloc.start()
    try:
        kernel([output, source])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert list(output.as_typed_buffer()) == [i * (1 << 32) for i in range(ring.n)]
    # Allow indexing and arithmetic temporaries without retaining the 32-step chain.
    assert peak < 8 * source.nbytes


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_python_kernel_preserves_modular_and_word_arithmetic(dtype):
    ring = large_prime_ring(4, dtype)[:2]
    spec = BufferSpec(dtype, ring.storage_size, "python")
    prototype = Node.new_buffer("python", ring.storage_size, dtype)
    target, left, right = (Node.param(i, prototype) for i in range(3))
    idx = Node(Op.GIDX, dtypes.index)
    product = left.index(idx).load() * right.index(idx).load()
    constant = -(1 << 130) + 7
    encoded = Node(Op.MOD_REDUCE, dtype, src=(Node.const(constant),))
    value = product + product.mod_reduce(None) + encoded
    body = Node.sink(target.index(idx).store(value)).replace(arg=KernelInfo("python", ring))
    kernel = PythonBackend().make_kernel(body)
    left_values = [a for q in ring.moduli for a in (q - 1, 0, 1, q - 2)]
    right_values = [b for q in ring.moduli for b in (q - 2, q - 1, 3, q - 1)]
    inputs = [Buffer(spec).allocate(opaque=np.asarray(values, dtype=to_np_dtype(dtype))) for values in (left_values, right_values)]
    output = Buffer(spec).allocate()

    kernel([output, *inputs])

    radix = 1 << (dtype.itemsize * 8)
    expected = []
    for i, (a, b) in enumerate(zip(left_values, right_values)):
        q = ring.moduli[i // ring.n]
        expected.append((a * b + a * b * pow(radix, -1, q) % q + constant * radix % q) % radix)
    assert list(output.as_typed_buffer()) == expected


def test_reusable_multi_output_call_allocates_each_output_per_invocation():
    from ren.engine.pattern import rewrite
    from ren.engine.rules.callify import resolve_calls
    from ren.kernel import KernelInfo

    source = Node.new_buffer("python", 4, dtypes.uint32)
    left, right = (Node.alloc("python", 4, dtypes.uint32) for _ in range(2))
    out0, operand, out1 = (Node.param(i, source) for i in range(3))
    body = Node.sink(out0.store(operand), out1.store(operand)).replace(arg=KernelInfo("python"))
    call = body.call(left, Node.param(0, source), right)
    graph = Node.sink(left.after(call), right.after(call))
    first, second = (rewrite(graph.call(source), resolve_calls).src[0].src[1] for _ in range(2))

    assert first.src[2] is second.src[2] is source
    assert first.src[1] is not second.src[1]
    assert first.src[3] is not second.src[3]


def _view(source: Node, size: int, offset: int, dtype: DType | None = None) -> Node:
    return Node(Op.BUFFER_VIEW, dtype or source.dtype, src=(source,), arg=(size, offset))


def test_view_keeps_storage_alive_until_released():
    source = Node.new_buffer("python", 16, dtypes.uint32)
    view = _view(source, 8, 3)
    storage = weakref.ref(source.buffer)
    assert not view.buffer.is_allocated()
    assert view.buffer.base is source.buffer

    del source
    gc.collect()
    assert storage() is not None

    del view
    gc.collect()
    assert storage() is None


def test_view_binding_releases_producer_inputs():
    source, output = (Node.new_buffer("python", 16, dtypes.uint32) for _ in range(2))
    source_ref = weakref.ref(source.buffer)
    target, operand = Node.param(0, output), Node.param(1, source)
    producer = Node(Op.CALL, src=(Node.sink(target.store(operand)), output, source))
    binding = buffer_binding(_view(output.after(producer), 8, 3))
    del source, producer
    gc.collect()

    assert source_ref() is None
    resolved = resolve_buffer(binding, [])
    assert resolved.base is output.buffer and resolved.size == 8 and resolved.offset == 3


def test_nested_parameter_views_rebind_to_current_storage():
    prototype = Node.new_buffer("python", 16, dtypes.uint32)
    param = Node.param(0, prototype)
    inner = _view(param, 8, 3)
    binding = _view(inner, 3, 2)

    for seed, input_offset in ((0, 0), (100, 4)):
        source = Buffer(BufferSpec(dtype=dtypes.uint32, size=32, device="python"))
        source.copyin(memoryview(array("I", range(seed, seed + 32))).cast("B"))
        bound = source.view(16, input_offset)
        result = resolve_buffer(binding, [bound])
        assert result.base is source
        assert result.size == 3
        assert result._offset == (input_offset + 5) * dtypes.uint32.itemsize
        assert result.as_buffer().cast("I").tolist() == list(range(seed + input_offset + 5, seed + input_offset + 8))


def test_bare_binding_preserves_identity_and_explicit_view_remains_a_view():
    node = Node.new_buffer("python", 16, dtypes.uint32)
    param = Node.param(0, node)
    source = node.buffer
    assert resolve_buffer(param, [source]) is source
    call = Node(Op.CALL, src=(Node.param(1, node).copy_to_device("python"), node, node))
    assert resolve_buffer(node.after(call), []) is source
    viewed = resolve_buffer(_view(param, 16, 0), [source])
    assert viewed is not source
    assert viewed._base is source


@pytest.mark.parametrize("size,offset", [(4, 2), (-1, 0), (1, -1)])
def test_nested_view_must_fit_its_immediate_parent(size: int, offset: int):
    node = Node.new_buffer("python", 16, dtypes.uint32)
    binding = _view(_view(Node.param(0, node), 4, 3), size, offset)
    with pytest.raises(AssertionError, match="invalid buffer view range"):
        resolve_buffer(binding, [node.buffer])


def test_nested_view_cannot_hide_invalid_inner_range():
    node = Node.new_buffer("python", 16, dtypes.uint32)
    binding = _view(_view(Node.param(0, node), 16, 4), 1, 0)
    with pytest.raises(AssertionError, match="invalid buffer view range"):
        resolve_buffer(binding, [node.buffer])


def test_nested_view_cannot_hide_inner_dtype_change():
    node = Node.new_buffer("python", 16, dtypes.uint32)
    binding = _view(_view(Node.param(0, node), 8, 0, dtypes.uint64), 1, 0, dtypes.uint32)
    with pytest.raises(ValueError, match="buffer view dtype"):
        resolve_buffer(binding, [node.buffer])


@pytest.mark.parametrize("size,dtype,device", [(15, dtypes.uint32, "python"), (16, dtypes.uint64, "python"), (16, dtypes.uint32, "cuda:0")])
def test_view_parameter_still_validates_bound_storage(size: int, dtype: DType, device: str):
    node = Node.new_buffer("python", 16, dtypes.uint32)
    source = Buffer(BufferSpec(size=size, dtype=dtype, device=device))
    with pytest.raises(ValueError, match="does not match its parameter"):
        resolve_buffer(_view(Node.param(0, node), 1, 0), [source])


def test_view_parameter_requires_a_binding():
    node = Node.new_buffer("python", 16, dtypes.uint32)
    with pytest.raises(RuntimeError, match="missing execution buffer"):
        resolve_buffer(_view(Node.param(0, node), 1, 0), [])
