import random
import numpy as np
import pytest

from ren.buffer import Buffer, BufferSpec
from ren.backends.cuda import SUPPORTED_DTYPES as CUDA_SUPPORTED_DTYPES
from ren.backends.python import SUPPORTED_DTYPES as PYTHON_SUPPORTED_DTYPES
from ren.device import Device
from ren.dtype import dtypes
from ren.engine.codegen import lower_kernel
from ren.engine.rules.modular import mod_add, mod_sub, mont_mul
from ren.ir import Node, Op
from ren.ir.node import ParamArg
from ren.kernel import KernelInfo
from ren.ring import Domain
from ren.ckks.utils import get_prev_prime
from tests.utils import eval_scalar, mixed_prime_ring


@pytest.mark.parametrize("supported", [PYTHON_SUPPORTED_DTYPES, CUDA_SUPPORTED_DTYPES])
def test_lower_kernel_folds_table_lookup(supported):
    ring = mixed_prime_ring(128, dtypes.uint64)[:1]
    table = Node(Op.CONST_ARRAY, ring.dtype, arg=(17, 97))
    output = Node(Op.PARAM, ring.dtype, arg=ParamArg(0, ring.storage_size, "python"))
    value = table.index(Node.const(1, dtypes.index)).load()
    ast = Node.sink(output.index(Node(Op.GIDX, dtypes.index)).store(value)).replace(arg=KernelInfo("python", ring, Domain.NTT))
    assert lower_kernel(ast, supported).stores[0].src[1] is Node.const(97, ring.dtype)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_constant_table_lookup_executes(dtype):
    maximum = (1 << (8 * dtype.itemsize)) - 1
    table = Node(Op.CONST_ARRAY, dtype, arg=(17, maximum))

    def build(index):
        fixed = table.index(Node.const(3, dtypes.index) - 2).load()
        return table.index(index.cast(dtypes.index)).load() + fixed

    assert eval_scalar(build, [[0, 1] * 64], dtype) == [16, maximum - 1] * 64


@pytest.mark.parametrize("specialized", [False, True])
def test_wide_scalar_support_does_not_allow_wide_buffers(specialized):
    ring = mixed_prime_ring(128, dtypes.uint64)[:1]
    source = Node(Op.PARAM, dtypes.uint128, arg=ParamArg(1, ring.storage_size, "cuda:0"))
    output = Node(Op.PARAM, ring.dtype, arg=ParamArg(0, ring.storage_size, "cuda:0"))
    idx = Node(Op.GIDX, dtypes.index)
    value = Node(Op.NTT, ring.dtype, src=(source,)) if specialized else source.index(idx).load().cast(ring.dtype)
    ast = Node.sink(output.index(idx).store(value)).replace(arg=KernelInfo("cuda:0", ring, Domain.NTT))
    with pytest.raises(ValueError, match=r"Invalid IR node: Op\.PARAM \(uint128\)"):
        lower_kernel(ast, CUDA_SUPPORTED_DTYPES)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_modular_lowering_executes_mixed_prime_arithmetic(dtype):
    ring = mixed_prime_ring(128, dtype)[:3]
    device = Device.DEFAULT
    idx = Node(Op.GIDX, dtypes.index)
    slots = [Node(Op.PARAM, dtype, arg=ParamArg(i, ring.storage_size, device)) for i in range(6)]
    a, b = (slot.index(idx).load() for slot in slots[-2:])
    values = [(a + b).mod_reduce(None), (a - b).mod_reduce(None), (a * b).mod_reduce(None), (-a).mod_reduce(None)]
    ast = Node.sink(*(slot.index(idx).store(value) for slot, value in zip(slots, values))).replace(arg=KernelInfo(device, ring, Domain.NTT))
    backend = Device[device]
    lowered = lower_kernel(ast, CUDA_SUPPORTED_DTYPES if Device.is_cuda(device) else PYTHON_SUPPORTED_DTYPES).ast
    assert all(n.op is not Op.MOD_REDUCE for n in lowered.toposort())
    np_dtype = dtype.fmt
    xs = [v for p in ring.moduli for v in ([0, 1, p - 1, p - 2] * 32)]
    ys = [v for p in ring.moduli for v in ([p - 1, p - 2, p - 1, 2] * 32)]
    buffers = [Buffer(BufferSpec(dtype, ring.storage_size, device)).ensure_allocated() for _ in slots]
    for buffer, data in zip(buffers[-2:], (xs, ys)):
        buffer.copyin(memoryview(np.array(data, dtype=np_dtype)).cast("B"))
    backend.make_kernel(ast)(buffers)
    backend.synchronize()
    expected = [[], [], [], []]
    radix = 1 << (dtype.itemsize * 8)
    for i, (x, y) in enumerate(zip(xs, ys)):
        p = ring.moduli[i // ring.n]
        for row, value in zip(expected, (x + y, x - y, x * y * pow(radix, -1, p), -x)):
            row.append(value % p)
    for buffer, row in zip(buffers, expected):
        assert bytes(buffer.as_buffer()) == np.array(row, dtype=np_dtype).tobytes()


def _modulus_cases():
    # Largest supported moduli (p < R/2) plus typical NTT primes, for both word sizes.
    return [
        (dtypes.uint32, get_prev_prime(m=2, bound=1 << 31)),
        (dtypes.uint32, get_prev_prime(m=1 << 17, bound=1 << 30)),
        (dtypes.uint64, get_prev_prime(m=2, bound=1 << 63)),
        (dtypes.uint64, get_prev_prime(m=1 << 17, bound=1 << 60)),
    ]


def _operands(p: int, word_bits: int, count: int = 128) -> tuple[list[int], list[int]]:
    """Pair full-word inputs, including R - 1, with residues below p."""
    edges_a = [0, 1, 2, p - 1, p, p + 1, (1 << word_bits) - 1]
    edges_b = [0, 1, 2, p // 2, p - 2, p - 1]
    pairs = [(a, b) for a in edges_a for b in edges_b]
    rng = random.Random(p)
    pairs += [(rng.getrandbits(word_bits), rng.randrange(p)) for _ in range(count - len(pairs))]
    return [a for a, _ in pairs], [b for _, b in pairs]


@pytest.mark.parametrize(("dtype", "p"), _modulus_cases())
def test_mont_mul_expression_matches_exact_reduction(dtype, p):
    bits = dtype.itemsize * 8
    a, b = _operands(p, bits)

    def build(x, y):
        return mont_mul(x, y, Node.const(p, dtype), Node.const(-pow(p, -1, 1 << bits) % (1 << bits), dtype))

    r_inv = pow(1 << bits, -1, p)
    assert eval_scalar(build, [a, b], dtype) == [x * y * r_inv % p for x, y in zip(a, b)]


@pytest.mark.parametrize(("dtype", "p"), _modulus_cases())
def test_mod_add_sub_expressions_match_exact_reduction(dtype, p):
    a, b = _operands(p, dtype.itemsize * 8)
    a = [x % p for x in a]
    assert eval_scalar(lambda x, y: mod_add(x, y, Node.const(p, dtype)), [a, b], dtype) == [(x + y) % p for x, y in zip(a, b)]
    assert eval_scalar(lambda x, y: mod_sub(x, y, Node.const(p, dtype)), [a, b], dtype) == [(x - y) % p for x, y in zip(a, b)]
