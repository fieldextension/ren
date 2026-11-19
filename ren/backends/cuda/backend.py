from fractions import Fraction
from typing import Any, ClassVar
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property
import ctypes
import os


from ren.backend import Backend
from ren.buffer import Buffer
from ren.dtype import DType, dtypes
from ren.engine import codegen
from ren.kernel import KernelSpec
from ren.ir import Op
from ren.ir.node import Node
from ren.context import DEBUG
from ren.polydist import DiscreteGaussian, SparseTernary, Ternary, UniformBounded, UniformModQ
from ren.random import ChaChaRng
from ren.tables import Tables

from . import api as cuda
from . import SUPPORTED_DTYPES
from .compiler import CUDACompiler
from .allocator import CUDAAllocator
from .graph import CUDAQueueGraphRunner
from .kernels import get_headers
from .ntt import _select_ntt_schedule
from .renderer import CUDARenderer, RenderedKernel
from .queue import ComputeQueue, CopyQueue, Signal
from ren.engine.pattern import pat, rules


def _coeff_ctype(dtype: DType) -> type[ctypes._SimpleCData]:
    if dtype is dtypes.uint32:
        return ctypes.c_uint
    if dtype is dtypes.uint64:
        return ctypes.c_ulonglong
    raise TypeError(f"unsupported CUDA coeff dtype {dtype}")


def _validate_chacha_blocks(blocks: int) -> None:
    if blocks > 1 << 32:
        raise ValueError("Random operation exceeds the ChaCha20 block counter")


@dataclass
class CUDAKernelSpec:
    func: Any
    grid: tuple[int, ...]
    block: tuple[int, ...]
    shared_mem: int
    args: list[Any]
    buf_indices: tuple[int, ...]

    def bind_bufs(self, bufs: list[Buffer]) -> None:
        for i, buf_idx in enumerate(self.buf_indices):
            self.args[i].value = int(bufs[buf_idx]._buf)


@dataclass
class CUDAKernelProgram:
    specs: list[CUDAKernelSpec]
    backend: CUDABackend

    def __call__(self, bufs: list[Buffer]) -> None:
        for spec in self.specs:
            self.backend._launch_spec(spec, bufs)


class CUDABackend(Backend):
    arch: str
    device: str
    graph = CUDAQueueGraphRunner
    _all_backends: ClassVar[list[CUDABackend]] = []

    def __init__(self, device: str):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        self.device = device

        cuda.init(0)

        self.device_id = device_id
        self.cu_device = cuda.device_get(device_id)
        self.arch = cuda.get_arch(device_id)
        self.compiler = CUDACompiler(self.arch, headers=get_headers())
        self.context = cuda.ctx_create(self.cu_device)

        cuda.ctx_set_current(self.context)

        # Enable peer access against already-open CUDA backends so inter-device
        # copies can use direct device pointers when the hardware supports it.
        for other in CUDABackend._all_backends:
            if cuda.device_can_access_peer(self.device_id, other.device_id):
                cuda.ctx_set_current(other.context)
                cuda.ctx_enable_peer_access(self.context)
                cuda.ctx_set_current(self.context)
                cuda.ctx_enable_peer_access(other.context)
                if DEBUG >= 2:
                    print(f"P2P enabled: {device} <-> {other.device}")

        CUDABackend._all_backends.append(self)

        # Keep compute and copy on separate queues and use one signal per queue
        # for cross-queue waits. That preserves overlap without introducing a
        # full graph dependency tracker here.
        self.compute_queue = ComputeQueue()
        self.copy_queue = CopyQueue()
        self._copy_signal = Signal()
        self._compute_signal = Signal()

        self.sm_count = cuda.device_get_sm_count(device_id)
        self._allocator = CUDAAllocator(self)
        self.renderer = CUDARenderer(self.arch)
        self.tables = Tables(device)
        self._block_size_cache: dict[tuple, int] = {}
        self._kernel_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._make_kernel_cache: dict[Node, list[CUDAKernelSpec]] = {}
        # We can't allocate buffers in the constructor, so set default rng lazily
        self._default_rng: ChaChaRng | None = None
        if DEBUG >= 1:
            print(f"cuda backend init device={device} arch={self.arch}")

    @property
    def allocator(self) -> CUDAAllocator:
        return self._allocator

    def set_default_rng(self, rng: ChaChaRng):
        if not isinstance(rng, ChaChaRng):
            raise TypeError("rng must be a ChaChaRng")
        if rng.device != self.device:
            raise ValueError(f"RNG device mismatch: {rng.device} != {self.device}")
        self._default_rng = rng

    def default_rng(self) -> ChaChaRng:
        if self._default_rng is None:
            self.set_default_rng(ChaChaRng(self.device, os.urandom(ChaChaRng.seed_size)))
        assert self._default_rng is not None
        return self._default_rng

    @classmethod
    def device_ptr(cls, buf: Buffer, offset_elements: int = 0, ctype=None):
        if ctype is None:
            ctype = _coeff_ctype(buf.dtype)
        return ctypes.c_void_p(int(buf._buf) + offset_elements * ctypes.sizeof(ctype))

    def _launch_spec(self, spec: CUDAKernelSpec, bufs: list[Buffer]) -> None:
        spec.bind_bufs(bufs)
        cuda.ctx_set_current(self.context)
        self.compute_queue.launch(spec.func, spec.grid, spec.block, spec.args, spec.shared_mem)

    def _launch_dims(self, func: Any, size: int, shared_mem: int = 0) -> tuple[tuple[int], tuple[int]]:
        block_hint = self._get_optimal_block_size(func, shared_mem)
        grid_hint = (size + block_hint - 1) // block_hint
        waves = (grid_hint + self.sm_count - 1) // self.sm_count
        # decrease the block size to try to launch full waves
        denom = 32 * waves * self.sm_count
        warps_per_block = (size + denom - 1) // denom
        block_hint_2 = 32 * warps_per_block
        # stalls tend to increase below 4 warps / sm
        block = max(min(block_hint, 128), block_hint_2)
        grid = (size + block - 1) // block
        return (grid,), (block,)

    def _get_optimal_block_size(self, func: Any, shared_mem_bytes: int = 0) -> int:
        cache_key = (id(func), shared_mem_bytes)
        if cache_key in self._block_size_cache:
            return self._block_size_cache[cache_key]

        _, block = cuda.occupancy_max_potential_block_size(func, shared_mem_bytes)
        self._block_size_cache[cache_key] = block
        if DEBUG >= 2:
            print(f"occupancy: block={block} smem={shared_mem_bytes}")
        return block

    def synchronize(self) -> None:
        cuda.ctx_set_current(self.context)
        self.compute_queue.synchronize()
        self.copy_queue.synchronize()

    def finalize(self) -> None:
        try:
            cuda.ctx_set_current(self.context)
            self.synchronize()
            self.allocator.free_cache()
            self.allocator.free_staging()
            # Close queue-owned handles before releasing the device's primary
            # context reference.
            self._copy_signal.close()
            self._compute_signal.close()
            self.compute_queue.close()
            self.copy_queue.close()
            cuda.primary_ctx_release(self.cu_device)
        except Exception:
            pass
        if self in CUDABackend._all_backends:
            CUDABackend._all_backends.remove(self)

    def _get_input_slot(self, node: Node) -> int:
        slot = node.base
        if slot.op is Op.PARAM:
            return slot.arg.slot
        raise RuntimeError(f"Expected PARAM, got {slot.op}")

    def make_kernel(self, ast: Node) -> CUDAKernelProgram:
        # The body, its order and definition boundaries determine rendering and launch planning.
        if (launches := self._make_kernel_cache.get(ast)) is None:
            spec = codegen.lower_kernel(ast, SUPPORTED_DTYPES)
            rendered = self.renderer.render(spec)
            cuda.ctx_set_current(self.context)
            funcs = tuple(self._compile(kernel) for kernel in rendered)
            launches = self._make_kernel_cache[ast] = self._build_launch_specs(spec, funcs)
        # Each execution unit owns its argument cells; graph replay patches them in place.
        return CUDAKernelProgram([replace(launch, args=[type(arg)(arg.value) for arg in launch.args]) for launch in launches], self)

    def _compile(self, rendered: RenderedKernel) -> Any:
        cached = self._kernel_cache.get(rendered.cache_key)
        if cached is not None:
            return cached[1]

        cubin, lowered_name = self.compiler.compile(rendered.source, rendered.name)
        # A module belongs to the context it is loaded into, and the cache
        # below is per backend: load it into this backend's context.
        cuda.ctx_set_current(self.context)
        module = cuda.module_load_data(cubin)
        func = cuda.module_get_function(module, lowered_name)
        self._kernel_cache[rendered.cache_key] = (module, func)

        if DEBUG >= 1:
            print(f"compiled kernel: name={rendered.name} lowered={lowered_name.decode()}")
        if DEBUG >= 2:
            print(rendered.source)

        return func

    @cached_property
    def _launch_handlers(self):
        return rules(
            [
                (pat().sink(pat().store(pat.node(op))), build)
                for op, build in (
                    (Op.NTT, self._build_ntt_launch),
                    (Op.INTT, self._build_intt_launch),
                    (Op.RESCALE, self._build_rescale_launch),
                    (Op.TO_RING, self._build_to_ring_launch),
                    (Op.RANDOM, self._build_random_launch),
                    (Op.TO_RNS, self._build_to_rns_launch),
                    (Op.FROM_RNS, self._build_from_rns_launch),
                )
            ]
            + [(pat(Op.SINK), self._build_generated_launch)]
        )

    def _build_launch_specs(self, spec: KernelSpec, funcs: tuple[Callable, ...]) -> list[CUDAKernelSpec]:
        build = self._launch_handlers.dispatch(spec.ast)
        assert build is not None, f"Invalid kernel received by cuda device {spec.ast.op}"
        return build(spec, funcs)

    def _build_generated_launch(self, spec: KernelSpec, funcs: tuple[Callable, ...]) -> list[CUDAKernelSpec]:
        (func,) = funcs
        ring = spec.ring
        size = ring.n * len(ring.moduli)
        grid, block = self._launch_dims(func, size)
        all_slots = spec.argument_slots
        args: list[Any] = [ctypes.c_void_p() for _ in all_slots]
        args.append(ctypes.c_ulonglong(size))
        buf_indices = all_slots
        return [
            CUDAKernelSpec(
                func=func,
                grid=grid,
                block=block,
                shared_mem=0,
                args=args,
                buf_indices=buf_indices,
            )
        ]

    def _build_ntt_launch(self, spec: KernelSpec, funcs: tuple[Any, ...], *, node: Node) -> list[CUDAKernelSpec]:
        """Apply the forward schedule to the renderer's compiled kernels."""
        out_slot = spec.output_slots[0]
        in_slot = self._get_input_slot(node.src[0])
        ring = spec.ring
        n_power = ring.n.bit_length() - 1
        num_primes = len(ring.moduli)
        schedule = _select_ntt_schedule(self.arch, ring, inverse=False)

        rt = self.tables.ring(ring)
        psis_ptr = ctypes.c_void_p(int(rt.ntt_psis._buf))
        primes_ptr = ctypes.c_void_p(int(rt.primes._buf))
        p_inv_ptr = ctypes.c_void_p(int(rt.mont_p_inv._buf))

        specs: list[CUDAKernelSpec] = []
        for i, (phase, func) in enumerate(zip(schedule, funcs, strict=True)):
            args: list[Any] = [
                ctypes.c_void_p(),
                ctypes.c_void_p(),
                psis_ptr,
                primes_ptr,
                p_inv_ptr,
                ctypes.c_int(n_power),
                ctypes.c_int(phase.logm),
                ctypes.c_int(phase.outer_count),
            ]
            buf_indices = (in_slot, out_slot) if i == 0 else (out_slot, out_slot)
            specs.append(
                CUDAKernelSpec(
                    func=func,
                    grid=(*phase.grid_xy, num_primes),
                    block=phase.block,
                    shared_mem=phase.shared_bytes,
                    args=args,
                    buf_indices=buf_indices,
                )
            )
        return specs

    def _build_intt_launch(self, spec: KernelSpec, funcs: tuple[Any, ...], *, node: Node) -> list[CUDAKernelSpec]:
        """Apply the inverse schedule to the renderer's compiled kernels."""
        out_slot = spec.output_slots[0]
        in_slot = self._get_input_slot(node.src[0])
        ring = spec.ring
        n_power = ring.n.bit_length() - 1
        num_primes = len(ring.moduli)
        schedule = _select_ntt_schedule(self.arch, ring, inverse=True)

        rt = self.tables.ring(ring)
        psis_inv_ptr = ctypes.c_void_p(int(rt.ntt_psis_inv._buf))
        primes_ptr = ctypes.c_void_p(int(rt.primes._buf))
        p_inv_ptr = ctypes.c_void_p(int(rt.mont_p_inv._buf))
        n_inv_ptr = ctypes.c_void_p(int(rt.n_inv._buf))

        specs: list[CUDAKernelSpec] = []
        for i, (phase, func) in enumerate(zip(schedule, funcs, strict=True)):
            args: list[Any] = [
                ctypes.c_void_p(),
                ctypes.c_void_p(),
                psis_inv_ptr,
                primes_ptr,
                p_inv_ptr,
                n_inv_ptr,
                ctypes.c_int(n_power),
                ctypes.c_int(phase.logm),
                ctypes.c_int(phase.outer_count),
            ]
            buf_indices = (in_slot, out_slot) if i == 0 else (out_slot, out_slot)
            specs.append(
                CUDAKernelSpec(
                    func=func,
                    grid=(*phase.grid_xy, num_primes),
                    block=phase.block,
                    shared_mem=phase.shared_bytes,
                    args=args,
                    buf_indices=buf_indices,
                )
            )
        return specs

    def _build_random_launch(self, spec: KernelSpec, funcs: tuple[Any, ...], *, node: Node) -> list[CUDAKernelSpec]:
        out_slot = spec.output_slots[0]
        (func,) = funcs
        dist = node.arg.dist
        stream_id = node.arg.stream_id
        key_slot = self._get_input_slot(node.src[0])
        stream_id = int(stream_id)
        match dist:
            case DiscreteGaussian(n=_, stdev=stdev):
                return self._build_random_discrete_gaussian_launch(spec, func, out_slot, key_slot, stream_id, stdev)
            case SparseTernary():
                return self._build_random_sparse_ternary_launch(spec, func, out_slot, key_slot, stream_id, dist)
            case Ternary():
                return self._build_random_ternary_launch(spec, func, out_slot, key_slot, stream_id)
            case UniformBounded(n=_, bound=bound):
                return self._build_random_uniform_bounded_launch(spec, func, out_slot, key_slot, stream_id, bound)
            case UniformModQ():
                return self._build_random_uniform_mod_q_launch(spec, func, out_slot, key_slot, stream_id)
            case _:
                raise NotImplementedError(f"Unsupported distribution: {dist}")

    def _build_random_sparse_ternary_launch(
        self,
        spec: KernelSpec,
        func: Any,
        out_slot: int,
        key_slot: int,
        stream_id: int,
        dist: SparseTernary,
    ) -> list[CUDAKernelSpec]:
        ring = spec.ring
        rt = self.tables.ring(ring)
        _validate_chacha_blocks((dist.h + 15) // 16)
        num_threads = 256
        args: list[Any] = [
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_void_p(int(rt.primes._buf)),
            ctypes.c_void_p(int(rt.mont_p_inv._buf)),
            ctypes.c_void_p(int(rt.mont_r2._buf)),
            ctypes.c_ulonglong(stream_id),
            ctypes.c_ulonglong(ring.n),
            ctypes.c_ulonglong(len(ring.moduli)),
            ctypes.c_ulonglong(dist.h),
        ]
        return [
            CUDAKernelSpec(
                func=func,
                grid=(1,),
                block=(num_threads,),
                shared_mem=0,
                args=args,
                buf_indices=(out_slot, key_slot),
            )
        ]

    def _build_random_ternary_launch(self, spec: KernelSpec, func: Any, out_slot: int, key_slot: int, stream_id: int) -> list[CUDAKernelSpec]:
        ring = spec.ring
        rt = self.tables.ring(ring)
        _validate_chacha_blocks(ring.n)
        num_threads = ring.n
        args: list[Any] = [
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_void_p(int(rt.primes._buf)),
            ctypes.c_void_p(int(rt.mont_p_inv._buf)),
            ctypes.c_void_p(int(rt.mont_r2._buf)),
            ctypes.c_ulonglong(stream_id),
            ctypes.c_ulonglong(ring.n),
            ctypes.c_ulonglong(len(ring.moduli)),
        ]
        grid, block = self._launch_dims(func, num_threads)
        return [CUDAKernelSpec(func=func, grid=grid, block=block, shared_mem=0, args=args, buf_indices=(out_slot, key_slot))]

    def _build_random_uniform_bounded_launch(
        self, spec: KernelSpec, func: Any, out_slot: int, key_slot: int, stream_id: int, bound: int
    ) -> list[CUDAKernelSpec]:
        if bound >= 1 << 62:
            raise ValueError("CUDA uniform bound must be less than 2^62")
        ring = spec.ring
        rt = self.tables.ring(ring)
        _validate_chacha_blocks(ring.n)
        num_threads = ring.n
        args: list[Any] = [
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_void_p(int(rt.primes._buf)),
            ctypes.c_void_p(int(rt.mont_p_inv._buf)),
            ctypes.c_void_p(int(rt.mont_r2._buf)),
            ctypes.c_ulonglong(stream_id),
            ctypes.c_ulonglong(bound),
            ctypes.c_ulonglong(ring.n),
            ctypes.c_ulonglong(len(ring.moduli)),
        ]
        grid, block = self._launch_dims(func, num_threads)
        return [CUDAKernelSpec(func=func, grid=grid, block=block, shared_mem=0, args=args, buf_indices=(out_slot, key_slot))]

    def _build_random_uniform_mod_q_launch(self, spec: KernelSpec, func: Any, out_slot: int, key_slot: int, stream_id: int) -> list[CUDAKernelSpec]:
        ring = spec.ring
        rt = self.tables.ring(ring)
        _validate_chacha_blocks(ring.n * len(ring.moduli))
        num_threads = ring.n * len(ring.moduli)
        args: list[Any] = [
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_void_p(int(rt.primes._buf)),
            ctypes.c_ulonglong(stream_id),
            ctypes.c_ulonglong(ring.n),
            ctypes.c_ulonglong(len(ring.moduli)),
        ]
        grid, block = self._launch_dims(func, num_threads)
        return [CUDAKernelSpec(func=func, grid=grid, block=block, shared_mem=0, args=args, buf_indices=(out_slot, key_slot))]

    def _build_random_discrete_gaussian_launch(
        self, spec: KernelSpec, func: Any, out_slot: int, key_slot: int, stream_id: int, stdev: Fraction
    ) -> list[CUDAKernelSpec]:
        if self.tables.dists.gaussian_max_coefficient(stdev) >= 1 << 127:
            raise ValueError("Gaussian coefficient exceeds the signed 128-bit CUDA range")
        plan = self.tables.dists.discrete_gaussian(stdev)
        ring = spec.ring
        rt = self.tables.ring(ring)
        num_threads = ring.n
        match plan.sample_combination_table:
            case None:
                _validate_chacha_blocks(ring.n)
                args: list[Any] = [
                    ctypes.c_void_p(),
                    ctypes.c_void_p(),
                    ctypes.c_void_p(int(rt.primes._buf)),
                    ctypes.c_void_p(int(rt.mont_p_inv._buf)),
                    ctypes.c_void_p(int(rt.mont_r2._buf)),
                    ctypes.c_ulonglong(stream_id),
                    ctypes.c_void_p((int(plan.base_sampler_table._buf))),
                    ctypes.c_ulonglong(ring.n),
                    ctypes.c_ulonglong(len(ring.moduli)),
                ]
                grid, block = self._launch_dims(func, num_threads)
            case Buffer():
                _validate_chacha_blocks(ring.n << (plan.sample_combination_table.size - 1))
                args: list[Any] = [
                    ctypes.c_void_p(),
                    ctypes.c_void_p(),
                    ctypes.c_void_p(int(rt.primes._buf)),
                    ctypes.c_void_p(int(rt.mont_p_inv._buf)),
                    ctypes.c_void_p(int(rt.mont_r2._buf)),
                    ctypes.c_ulonglong(stream_id),
                    ctypes.c_void_p((int(plan.base_sampler_table._buf))),
                    ctypes.c_void_p((int(plan.sample_combination_table._buf))),
                    ctypes.c_ulonglong(ring.n),
                    ctypes.c_ulonglong(len(ring.moduli)),
                    ctypes.c_ulonglong(plan.sample_combination_table.size),
                ]
                # The kernel uses warp shuffle to combine base samples, so it can't use more
                # than 2^5 threads per coefficient.  Each thread generates 2 base samples.
                if plan.sample_combination_table.size < 1 or plan.sample_combination_table.size > 6:
                    raise ValueError("Number of merge steps must be between 1 and 6")
                # Block size 128 is usually a decent choice, haven't actually profiled
                block = (1 << (plan.sample_combination_table.size - 1), 1 << (8 - plan.sample_combination_table.size))
                grid = ((ring.n + block[1] - 1) // block[1],)

        return [CUDAKernelSpec(func=func, grid=grid, block=block, shared_mem=0, args=args, buf_indices=(out_slot, key_slot))]

    def _build_to_rns_launch(self, spec: KernelSpec, funcs: tuple[Any, ...], *, node: Node) -> list[CUDAKernelSpec]:
        out_slot = spec.output_slots[0]
        (func,) = funcs
        ring = spec.ring
        rt = self.tables.ring(ring)
        in_slot = self._get_input_slot(node.src[0])
        launch_size = ring.n * len(ring.moduli)
        grid, block = self._launch_dims(func, launch_size)
        return [
            CUDAKernelSpec(
                func=func,
                grid=grid,
                block=block,
                shared_mem=0,
                args=[
                    ctypes.c_void_p(),
                    ctypes.c_void_p(),
                    ctypes.c_ulonglong(ring.n),
                    ctypes.c_ulonglong(len(ring.moduli)),
                    ctypes.c_void_p(int(rt.primes._buf)),
                    ctypes.c_void_p(int(rt.mont_p_inv._buf)),
                    ctypes.c_void_p(int(rt.mont_r2._buf)),
                    ctypes.c_void_p(int(rt.mont_radix32._buf)),
                ],
                buf_indices=(in_slot, out_slot),
            )
        ]

    def _build_from_rns_launch(self, spec: KernelSpec, funcs: tuple[Any, ...], *, node: Node) -> list[CUDAKernelSpec]:
        out_slot = spec.output_slots[0]
        (func,) = funcs
        ring = spec.ring
        rt = self.tables.ring(ring)
        in_slot = self._get_input_slot(node.src[0])
        launch_size = ring.n  # one thread per coefficient
        grid, block = self._launch_dims(func, launch_size)
        return [
            CUDAKernelSpec(
                func=func,
                grid=grid,
                block=block,
                shared_mem=0,
                args=[
                    ctypes.c_void_p(),  # src (RNS Montgomery, in_slot)
                    ctypes.c_void_p(),  # dst (signed u32 limbs, out_slot)
                    ctypes.c_ulonglong(ring.n),
                    ctypes.c_void_p(int(rt.primes._buf)),
                    ctypes.c_void_p(int(rt.mont_p_inv._buf)),
                    ctypes.c_void_p(int(rt.mont_r2._buf)),
                    ctypes.c_void_p(int(rt.mont_radix32._buf)),
                    ctypes.c_void_p(int(rt.crt_inverses._buf)),
                    ctypes.c_void_p(int(rt.crt_partial_prods._buf)),
                    ctypes.c_void_p(int(rt.crt_q._buf)),
                    ctypes.c_void_p(int(rt.crt_q_half._buf)),
                ],
                buf_indices=(in_slot, out_slot),
            )
        ]

    def _build_rescale_launch(self, spec: KernelSpec, funcs: tuple[Any, ...], *, node: Node) -> list[CUDAKernelSpec]:
        out_slot = spec.output_slots[0]
        (func,) = funcs
        src_ring = spec.ring
        dst_ring = node.arg

        n = src_ring.n

        src_rt = self.tables.ring(src_ring)
        dst_rt = self.tables.ring(dst_ring)
        rescale_rt = self.tables.rescale(src_ring, dst_ring)

        args: list[Any] = [
            ctypes.c_void_p(),
            ctypes.c_void_p(),
            ctypes.c_ulonglong(n),
            ctypes.c_void_p(int(src_rt.primes._buf)),
            ctypes.c_void_p(int(dst_rt.primes._buf)),
            ctypes.c_void_p(int(src_rt.mont_p_inv._buf)),
            ctypes.c_void_p(int(dst_rt.mont_p_inv._buf)),
            ctypes.c_void_p(int(rescale_rt.removed_src_indices._buf)),
            ctypes.c_void_p(int(rescale_rt.dst_src_indices._buf)),
            ctypes.c_void_p(int(rescale_rt.inv_src._buf)),
            ctypes.c_void_p(int(rescale_rt.mi_r2_scaled._buf)),
            ctypes.c_void_p(int(rescale_rt.prod_r2_scaled._buf)),
            ctypes.c_void_p(int(rescale_rt.inv_prod_mont._buf)),
        ]
        if not dst_ring.is_quotient_of(src_ring):
            args.extend([
                ctypes.c_void_p(int(rescale_rt.added_prod_removed_mont._buf)),
                ctypes.c_void_p(int(rescale_rt.added_prod_dst_mont._buf)),
            ])

        grid, block = self._launch_dims(func, n)
        in_slot = self._get_input_slot(node.src[0])
        return [
            CUDAKernelSpec(
                func=func,
                grid=grid,
                block=block,
                shared_mem=0,
                args=args,
                buf_indices=(in_slot, out_slot),
            )
        ]

    def _build_to_ring_launch(self, spec: KernelSpec, funcs: tuple[Any, ...], *, node: Node) -> list[CUDAKernelSpec]:
        out_slot = spec.output_slots[0]
        (func,) = funcs
        src_ring = spec.ring
        dst_ring = node.arg.ring
        in_slot = self._get_input_slot(node.src[0])

        n_coeffs = src_ring.n
        num_src = len(src_ring.moduli)
        num_dst = len(dst_ring.moduli)

        is_prefix_extend = num_dst > num_src and dst_ring.moduli[:num_src] == src_ring.moduli
        is_permutation = set(dst_ring.moduli) <= set(src_ring.moduli)

        if is_permutation:
            # Pure reindexing among shared primes: no CRT work required.
            launch_size = n_coeffs * num_dst
            grid, block = self._launch_dims(func, launch_size)
            args: list[Any] = [
                ctypes.c_void_p(),
                ctypes.c_void_p(),
                ctypes.c_ulonglong(n_coeffs),
            ]
        elif is_prefix_extend:
            bc = self.tables.base_change(src_ring, dst_ring)
            dst_rt = self.tables.ring(dst_ring)
            # Prefix extension keeps the existing limbs and synthesizes the new
            # ones with Montgomery tables.
            launch_size = n_coeffs * num_dst
            grid, block = self._launch_dims(func, launch_size)
            args = [
                ctypes.c_void_p(),
                ctypes.c_void_p(),
                ctypes.c_ulonglong(n_coeffs),
                ctypes.c_void_p(int(dst_rt.primes._buf)),
                ctypes.c_void_p(int(dst_rt.mont_p_inv._buf)),
                ctypes.c_void_p(int(bc.inv_src._buf)),
                ctypes.c_void_p(int(bc.mi_r2_scaled._buf)),
                ctypes.c_void_p(int(bc.prod_r2_scaled._buf)),
            ]
        else:
            bc = self.tables.base_change(src_ring, dst_ring)
            dst_rt = self.tables.ring(dst_ring)
            src_rt = self.tables.ring(src_ring)
            # General base change may mix copied limbs and lifted limbs, so the
            # kernel receives both the shared-prime index map and the full CRT
            # tables for the remaining destinations.
            grid, block = self._launch_dims(func, n_coeffs)
            args = [
                ctypes.c_void_p(),
                ctypes.c_void_p(),
                ctypes.c_ulonglong(n_coeffs),
                ctypes.c_void_p(int(src_rt.primes._buf)),
                ctypes.c_void_p(int(dst_rt.primes._buf)),
                ctypes.c_void_p(int(src_rt.mont_p_inv._buf)),
                ctypes.c_void_p(int(dst_rt.mont_p_inv._buf)),
                ctypes.c_void_p(int(bc.dst_src_indices._buf)),
                ctypes.c_void_p(int(bc.inv_src._buf)),
                ctypes.c_void_p(int(bc.mi_r2_scaled._buf)),
                ctypes.c_void_p(int(bc.prod_r2_scaled._buf)),
            ]

        return [
            CUDAKernelSpec(
                func=func,
                grid=grid,
                block=block,
                shared_mem=0,
                args=args,
                buf_indices=(in_slot, out_slot),
            )
        ]


__all__ = ["CUDABackend"]
