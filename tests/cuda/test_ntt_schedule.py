from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ren.backends.cuda.kernels import load_kernel
from ren.backends.cuda.ntt import NTTLaunchConfig, _build_merge_ntt_schedule, _select_ntt_schedule
from ren.backends.cuda.renderer import CUDARenderer
from ren.dtype import dtypes
from ren.ir import Op
from ren.ir.node import Node, ParamArg
from ren.kernel import KernelInfo, KernelSpec
from tests.utils import _ring


@pytest.mark.parametrize("arch", [None, "sm_89", "sm_120"])
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("logn", [1, 10, 11, 12, 15, 16, 17])
@pytest.mark.parametrize("inverse", [False, True])
def test_ntt_schedule_kernel_contract(arch, dtype, k, logn, inverse):
    ring = _ring(1 << logn, [31] * k, dtype=dtype)
    schedule = _select_ntt_schedule(arch, ring, inverse=inverse)
    slot = Node(Op.PARAM, dtype, arg=ParamArg(0, ring.storage_size, "cuda:0"))
    output = Node(Op.PARAM, dtype, arg=ParamArg(1, ring.storage_size, "cuda:0"))
    spec = KernelSpec(Node(Op.SINK, src=(output.store(Node(Op.INTT if inverse else Op.NTT, dtype, (slot,))),), arg=KernelInfo("cuda:0", ring)))
    rendered = CUDARenderer(arch).render(spec)
    assert isinstance(rendered, tuple)
    assert [kernel.name for kernel in rendered] == ["&" + phase.kernel_name for phase in schedule]
    assert all(kernel.source == load_kernel("ntt.cu") for kernel in rendered)
    assert sum(phase.outer_count for phase in schedule) == logn
    forward_order = schedule[::-1] if inverse else schedule
    next_stage = 0
    for phase in forward_order:
        assert phase.logm == next_stage
        next_stage += phase.outer_count

    specialized = arch == "sm_120" and logn == 16 and (dtype is dtypes.uint32 or k >= 5)
    direction = "inverse" if inverse else "forward"
    for i, phase in enumerate(schedule):
        if specialized:
            assert phase.kernel_name == f"ren_ntt_{direction}_radix8_phase{i}<{dtype.name},{str(dtype is dtypes.uint64).lower()}>"
            values = 512 if i == (0 if inverse else 1) else 8192 // dtype.itemsize
            assert phase.block[0] * phase.block[1] * 8 == values
            assert phase.shared_bytes == values * dtype.itemsize
            assert phase.grid_xy[0] * phase.grid_xy[1] * values == ring.n
        else:
            assert phase.kernel_name == f"ren_ntt_{direction}_merge<{dtype.name}>"
            # These are the merge kernel's thread-group and coverage contracts.
            gx, gy = phase.grid_xy
            bx, by = phase.block
            assert by == 1 << (phase.outer_count - 1)
            assert gy == 1 << phase.logm
            assert gx * gy * bx * by * 2 == ring.n
            assert phase.shared_bytes == 2 * bx * by * dtype.itemsize


@pytest.mark.parametrize("arch", [None, "sm_89", "sm_120"])
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("k,use_u64_radix8", [(1, False), (3, False), (4, False), (5, True), (6, True), (7, True), (8, True), (9, True), (30, True)])
@pytest.mark.parametrize("inverse", [False, True])
def test_ntt_schedule_modulus_count_boundary(arch, dtype, k, use_u64_radix8, inverse):
    ring = _ring(65536, [31] * k, dtype=dtype)
    schedule = _select_ntt_schedule(arch, ring, inverse=inverse)
    if arch != "sm_120" or (dtype is dtypes.uint64 and not use_u64_radix8):
        assert schedule == _build_merge_ntt_schedule(ring, inverse=inverse)
    else:
        direction = "inverse" if inverse else "forward"
        assert [phase.kernel_name for phase in schedule] == [
            f"ren_ntt_{direction}_radix8_phase{i}<{dtype.name},{str(dtype is dtypes.uint64).lower()}>" for i in range(2)
        ]


@pytest.mark.parametrize("dtype,k", [(dtypes.uint32, 1), (dtypes.uint64, 1), (dtypes.uint64, 5)])
@pytest.mark.parametrize("inverse", [False, True])
def test_selected_schedule_keeps_original_merge_baseline(dtype, k, inverse):
    ring = _ring(65536, [31] * k, dtype=dtype)
    baseline = _build_merge_ntt_schedule(ring, inverse=inverse)
    direction = "inverse" if inverse else "forward"
    name = f"ren_ntt_{direction}_merge<{dtype.name}>"
    expected = (
        NTTLaunchConfig(name, (128, 1), (4, 64), 512 * dtype.itemsize, 0, 7),
        NTTLaunchConfig(name, (1, 128), (1, 256), 512 * dtype.itemsize, 7, 9),
    )
    assert baseline == (expected[::-1] if inverse else expected)
    _select_ntt_schedule("sm_120", ring, inverse=False)
    _select_ntt_schedule("sm_120", ring, inverse=True)
    assert _build_merge_ntt_schedule(ring, inverse=inverse) == baseline


@pytest.mark.parametrize("arch", ["sm_89", "sm_120"])
@pytest.mark.parametrize("dtype,k", [(dtypes.uint32, 1), (dtypes.uint64, 4), (dtypes.uint64, 5)])
@pytest.mark.parametrize("inverse", [False, True])
def test_ntt_make_kernel_reuses_backend_caches(arch, dtype, k, inverse, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("cuda.bindings")
    from ren.backends.cuda import api as cuda
    from ren.backends.cuda.backend import CUDABackend

    ring = _ring(65536, [31] * k, dtype=dtype)
    schedule = _select_ntt_schedule(arch, ring, inverse=inverse)
    in_slot, out_slot = (Node.param(i, Node.new_buffer("cuda:0", ring.storage_size, ring.dtype)) for i in (0, 1))
    computation = Node(Op.INTT if inverse else Op.NTT, ring.dtype, (in_slot,))
    ast = Node(Op.SINK, dtypes.void, (out_slot.store(computation),), KernelInfo("cuda:0", ring))
    names = list(dict.fromkeys("&" + phase.kernel_name for phase in schedule))
    load_module = Mock(side_effect=lambda cubin: object())
    get_function = Mock(side_effect=lambda module, name: object())
    monkeypatch.setattr(cuda, "module_load_data", load_module)
    monkeypatch.setattr(cuda, "module_get_function", get_function)
    monkeypatch.setattr(cuda, "ctx_set_current", Mock())
    backends = [object.__new__(CUDABackend) for _ in range(2)]
    backend_funcs = []
    for backend in backends:
        backend.arch = arch
        backend.context = Mock()
        backend._kernel_cache = {}
        backend._make_kernel_cache = {}
        render_mock = Mock(wraps=CUDARenderer(arch).render)
        backend.renderer = Mock(render=render_mock)
        tables = SimpleNamespace(**{
            name: SimpleNamespace(_buf=i) for i, name in enumerate(("ntt_psis", "ntt_psis_inv", "primes", "mont_p_inv", "n_inv"), 1)
        })
        backend.tables = Mock(ring=Mock(return_value=tables))
        compile_mock = Mock(return_value=(b"cubin", b"kernel"))
        backend.compiler = Mock(compile=compile_mock)
        first, second = backend.make_kernel(ast), backend.make_kernel(ast)
        funcs = tuple(launch.func for launch in first.specs)
        assert tuple(launch.func for launch in second.specs) == funcs
        backend_funcs.append(funcs)
        render_mock.assert_called_once()
        backend.tables.ring.assert_called_once()
        assert [call.args[1] for call in compile_mock.call_args_list] == names
    assert backend_funcs[0] != backend_funcs[1]
    assert load_module.call_count == get_function.call_count == len(backends) * len(names)


@pytest.mark.parametrize("dtype,k", [(dtypes.uint32, 2), (dtypes.uint64, 4), (dtypes.uint64, 5)])
@pytest.mark.parametrize("inverse", [False, True])
def test_launch_builder_preserves_functions_and_owns_mutable_args(dtype, k, inverse, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("cuda.bindings")
    from ren.backends.cuda.backend import CUDABackend

    ring = _ring(65536, [31] * k, dtype=dtype)
    op = Op.INTT if inverse else Op.NTT
    slot = Node(Op.PARAM, ring.dtype, arg=ParamArg(0, ring.storage_size, "cuda:0"))
    specs = [
        KernelSpec(
            Node(
                Op.SINK,
                src=(Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "cuda:0")).store(Node(op, ring.dtype, (slot,))),),
                arg=KernelInfo("cuda:0", ring),
            )
        )
        for i in (1, 2)
    ]
    schedule = _select_ntt_schedule("sm_120", ring, inverse=inverse)
    backend = object.__new__(CUDABackend)
    backend.arch = "sm_120"
    tables = SimpleNamespace(**{
        name: SimpleNamespace(_buf=i) for i, name in enumerate(("ntt_psis", "ntt_psis_inv", "primes", "mont_p_inv", "n_inv"), 1)
    })
    monkeypatch.setattr(backend, "tables", SimpleNamespace(ring=lambda ring: tables), raising=False)
    # No renderer or compiler is supplied: builders must preserve the functions.
    build = backend._build_launch_specs
    funcs = tuple(Mock() for _ in schedule)
    first, second = (build(spec, funcs) for spec in specs)
    for i, (a, b, func, phase) in enumerate(zip(first, second, funcs, schedule, strict=True)):
        assert a.func is b.func is func
        assert a.grid == b.grid == (*phase.grid_xy, k)
        assert a.block == b.block == phase.block
        assert a.shared_mem == b.shared_mem == phase.shared_bytes
        assert a.buf_indices == ((0, 1) if i == 0 else (1, 1))
        assert b.buf_indices == ((0, 2) if i == 0 else (2, 2))
        assert [arg.value for arg in a.args[-3:]] == [16, phase.logm, phase.outer_count]
        a.args[0].value = 100 + i
        assert b.args[0].value is None
    assert first[0].args[0].value == 100
    assert first[1].args[0].value == 101
    with pytest.raises(ValueError):
        build(specs[0], funcs[:-1])


@pytest.mark.parametrize(
    "arch,n,dtype,prime_sizes,arithmetic",
    [
        ("sm_120", 65536, dtypes.uint64, [61] * 4, "merge"),
        ("sm_120", 65536, dtypes.uint64, [61] * 5, "true"),
        ("sm_120", 65536, dtypes.uint64, [40, 60, 61, 40, 61], "true"),
        ("sm_120", 65536, dtypes.uint64, [62] * 5, "true"),
        ("sm_120", 65536, dtypes.uint64, [61, 61, 62, 61, 61], "true"),
        ("sm_120", 65536, dtypes.uint64, [63, 62, 62, 62, 62], "false"),
        ("sm_120", 65536, dtypes.uint64, [62, 62, 63, 62, 62], "false"),
        ("sm_120", 65536, dtypes.uint64, [62, 62, 62, 62, 63], "false"),
        ("sm_120", 65536, dtypes.uint64, [63] * 5, "false"),
        ("sm_120", 65536, dtypes.uint32, [29] * 5, "true"),
        ("sm_120", 65536, dtypes.uint32, [30], "true"),
        ("sm_120", 65536, dtypes.uint32, [29, 30, 30], "true"),
        ("sm_120", 65536, dtypes.uint32, [31, 30, 30], "false"),
        ("sm_120", 65536, dtypes.uint32, [30, 31, 30], "false"),
        ("sm_120", 65536, dtypes.uint32, [30, 30, 31], "false"),
        ("sm_120", 65536, dtypes.uint32, [31] * 5, "false"),
        ("sm_89", 65536, dtypes.uint64, [61] * 5, "merge"),
        (None, 65536, dtypes.uint64, [61] * 5, "merge"),
        ("sm_120", 32768, dtypes.uint64, [61] * 5, "merge"),
        ("sm_120", 131072, dtypes.uint64, [61] * 5, "merge"),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_ntt_schedule_lazy_prime_boundary(arch, n, dtype, prime_sizes, arithmetic, inverse):
    """A single wide prime disables lazy arithmetic without changing the radix schedule."""
    ring = _ring(n, prime_sizes, dtype=dtype)
    schedule = _select_ntt_schedule(arch, ring, inverse=inverse)
    if arithmetic == "merge":
        assert schedule == _build_merge_ntt_schedule(ring, inverse=inverse)
    else:
        direction = "inverse" if inverse else "forward"
        assert [phase.kernel_name for phase in schedule] == [f"ren_ntt_{direction}_radix8_phase{i}<{dtype.name},{arithmetic}>" for i in range(2)]
