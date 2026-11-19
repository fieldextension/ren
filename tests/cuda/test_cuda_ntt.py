import ctypes
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cuda.bindings")

from ren.allocator import BufferSpec
from ren.backends.cuda import api as cuda
from ren.backends.cuda.backend import CUDABackend, CUDAKernelSpec
from ren.backends.cuda.kernels import load_kernel
from ren.backends.cuda.ntt import _build_merge_ntt_schedule
from ren.backends.cuda.renderer import RenderedKernel
from ren.buffer import Buffer
from ren.dtype import dtypes
from ren.ir import Op
from ren.ir.node import Node, ParamArg
from ren.kernel import KernelInfo, KernelSpec
from ren.ring import RingSpec
from tests.utils import _ring


@pytest.mark.requires_cuda
@pytest.mark.parametrize(
    "n,moduli,dtype,lazy",
    [
        (65536, (2147352577,), dtypes.uint32, False),
        (65536, (2013265921, 2147352577, 469762049), dtypes.uint32, False),
        (65536, (469762049, 2147352577, 2013265921), dtypes.uint32, False),
        (32768, (2147352577,), dtypes.uint32, False),
        (65536, (2147352577,), dtypes.uint64, False),
        (65536, _ring(65536, [40, 60, 63, 63]).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [40, 60, 63, 60, 63]).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [40, 60, 63, 60, 63]).moduli[::-1], dtypes.uint64, False),
        (65536, _ring(65536, [63] * 5).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [40, 60, 63] * 2).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [63] * 7).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [40, 60, 63, 40, 60, 63, 60, 63]).moduli, dtypes.uint64, False),
        (32768, _ring(32768, [63] * 5).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [29] * 5, dtype=dtypes.uint32).moduli, dtypes.uint32, True),
        (65536, _ring(65536, [30], dtype=dtypes.uint32).moduli, dtypes.uint32, True),
        (65536, _ring(65536, [29, 30, 30], dtype=dtypes.uint32).moduli, dtypes.uint32, True),
        (65536, _ring(65536, [61] * 4).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [61] * 5).moduli, dtypes.uint64, True),
        (65536, _ring(65536, [40, 60, 61] * 2).moduli, dtypes.uint64, True),
        (65536, _ring(65536, [61] * 7).moduli, dtypes.uint64, True),
        (65536, _ring(65536, [40, 60, 61, 40, 60, 61, 60, 61]).moduli, dtypes.uint64, True),
        (65536, _ring(65536, [62] * 5).moduli, dtypes.uint64, True),
        (65536, _ring(65536, [61, 61, 62, 61, 61]).moduli, dtypes.uint64, True),
        (65536, _ring(65536, [63, 61, 61, 61, 61]).moduli, dtypes.uint64, False),
        (65536, _ring(65536, [61, 61, 61, 61, 63]).moduli, dtypes.uint64, False),
        (32768, _ring(32768, [61] * 5).moduli, dtypes.uint64, False),
    ],
)
@pytest.mark.parametrize("pattern", ["zero", "one", "max", "alternating", "impulse", "random"])
@pytest.mark.parametrize("inverse", [False, True])
def test_ntt_matches_merge_and_roundtrip(cuda_device: CUDABackend, n, moduli, dtype, lazy, pattern, inverse):
    """Exercise raw kernels so IR cannot cancel the forward/inverse pair."""
    backend = cuda_device
    ring = RingSpec(n, moduli, dtype=dtype)
    source = load_kernel("ntt.cu")
    coeff_type = "unsigned int" if dtype is dtypes.uint32 else "unsigned long long"
    direction = "inverse" if inverse else "forward"
    merge = backend._compile(RenderedKernel(source, f"&ren_ntt_{direction}_merge<{coeff_type}>"))
    slot = Node(Op.PARAM, dtype, arg=ParamArg(0, ring.storage_size, backend.device))
    output = Node(Op.PARAM, dtype, arg=ParamArg(1, ring.storage_size, backend.device))
    spec, recovery_spec = (
        KernelSpec(Node(Op.SINK, src=(output.store(Node(op, dtype, (slot,))),), arg=KernelInfo(backend.device, ring)))
        for op in ((Op.INTT, Op.NTT) if inverse else (Op.NTT, Op.INTT))
    )
    funcs = tuple(backend._compile(kernel) for kernel in backend.renderer.render(spec))
    recovery_funcs = tuple(backend._compile(kernel) for kernel in backend.renderer.render(recovery_spec))
    launches = backend._build_launch_specs(spec, funcs)
    recovery_launches = backend._build_launch_specs(recovery_spec, recovery_funcs)
    assert len(launches) == len(recovery_launches) == 2
    phase0, phase1 = launches
    use_radix8 = backend.arch == "sm_120" and n == 65536 and (dtype is dtypes.uint32 or len(moduli) >= 5)
    assert (int(phase0.func) != int(merge)) == use_radix8
    assert (int(phase1.func) != int(merge)) == use_radix8
    if use_radix8:
        for i, phase in enumerate(launches):
            name = f"&ren_ntt_{direction}_radix8_phase{i}<{coeff_type},{str(lazy).lower()}>"
            assert int(phase.func) == int(backend._compile(RenderedKernel(source, name)))
        large, small = (phase1, phase0) if inverse else (phase0, phase1)
        grid, threads = (32, 256) if dtype is dtypes.uint32 else (64, 128)
        assert large.grid == (grid, 1, len(moduli)) and large.block == (threads, 1) and large.shared_mem == 8192
        assert small.grid == (1, 128, len(moduli)) and small.block == (64, 1) and small.shared_mem == 512 * dtype.itemsize
    baseline = _build_merge_ntt_schedule(ring, inverse=inverse)
    baseline_phase0, baseline_phase1 = (
        replace(
            launch,
            func=merge,
            grid=(*phase.grid_xy, len(moduli)),
            block=phase.block,
            shared_mem=phase.shared_bytes,
            args=launch.args[:-2] + [ctypes.c_int(phase.logm), ctypes.c_int(phase.outer_count)],
        )
        for launch, phase in zip(launches, baseline, strict=True)
    )
    bufs = [Buffer(BufferSpec(dtype, ring.storage_size, backend.device)).allocate() for _ in range(4)]
    np_dtype = np.uint32 if dtype is dtypes.uint32 else np.uint64
    q = np.array(moduli, dtype=np_dtype)[:, None]
    data = np.zeros((len(moduli), n), dtype=np_dtype)
    if pattern == "one":
        data.fill(1)
    elif pattern == "max":
        data[:] = q - 1
    elif pattern == "alternating":
        data[:, 1::2] = q - 1
    elif pattern == "impulse":
        data[:, [0, 1, n // 2, n - 1]] = q - 1
    elif pattern == "random":
        data = np.random.default_rng(3000).integers(0, q, size=data.shape, dtype=np_dtype)
    bufs[0].copyin(memoryview(data).cast("B"))
    backend.synchronize()

    def run(launch: CUDAKernelSpec, src: int, dst: int) -> None:
        args = [ctypes.c_void_p(int(bufs[i]._buf)) for i in (src, dst)] + launch.args[2:]
        cuda.ctx_set_current(backend.context)
        cuda.launch_kernel(launch.func, launch.grid, launch.block, args, backend.compute_queue.stream, launch.shared_mem)

    def read(index: int):
        backend.synchronize()
        return np.frombuffer(bufs[index].as_buffer(), dtype=np_dtype).copy()

    run(baseline_phase0, 0, 1)
    intermediate = read(1)
    run(baseline_phase1, 1, 2)
    expected = read(2)
    run(phase0, 0, 3)
    np.testing.assert_array_equal(read(3), intermediate)
    assert np.all(intermediate.reshape(len(moduli), n) < q)
    np.testing.assert_array_equal(read(0), data.ravel())
    run(phase1, 3, 1)
    np.testing.assert_array_equal(read(1), expected)
    np.testing.assert_array_equal(read(3), intermediate)
    run(phase1, 3, 3)
    np.testing.assert_array_equal(read(3), expected)
    bufs[1].copyin(memoryview(data).cast("B"))
    backend.synchronize()
    run(phase0, 1, 1)
    np.testing.assert_array_equal(read(1), intermediate)
    run(phase1, 1, 1)
    np.testing.assert_array_equal(read(1), expected)
    assert np.all(expected.reshape(len(moduli), n) < q)
    run(recovery_launches[0], 1, 3)
    run(recovery_launches[1], 3, 3)
    np.testing.assert_array_equal(read(3), data.ravel())
