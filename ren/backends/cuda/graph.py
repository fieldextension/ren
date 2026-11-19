import bisect
import ctypes
import traceback
import warnings
from collections.abc import Sequence
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ren.buffer import Buffer
from ren.context import DEBUG
from ren.device import Device
from ren.engine.realize import ExecUnit, resolve_buffer, unwrap_view
from ren.ir import Node, Op
from ren.kernel import get_output_slots

from . import api as cuda
from .queue import Signal

if TYPE_CHECKING:
    from .backend import CUDABackend, CUDAKernelProgram, CUDAKernelSpec


def _safe_destroy(handle: Any, destroy_fn: Any) -> None:
    if handle is not None:
        try:
            destroy_fn(handle)
        except Exception:
            pass


class _IntervalPartition:
    """Disjoint byte-intervals over a single buffer base, kept sorted by start.

    Each interval carries a list of unit indices (the writer, or the readers
    since the last write). Overlap queries, region clears, and inserts touch
    only the intervals that actually overlap the accessed range (located by
    bisection) instead of rescanning every interval recorded for the base.
    """

    __slots__ = ("starts", "ends", "ids")

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.ids: list[list[int]] = []

    def _split_at(self, x: int) -> None:
        """Ensure an interval boundary exists at offset ``x`` (splitting if inside one)."""
        i = bisect.bisect_right(self.starts, x) - 1
        if i >= 0 and self.starts[i] < x < self.ends[i]:
            end = self.ends[i]
            self.ends[i] = x
            self.starts.insert(i + 1, x)
            self.ends.insert(i + 1, end)
            self.ids.insert(i + 1, list(self.ids[i]))

    def overlapping(self, start: int, end: int) -> list[list[int]]:
        """Id-lists of every interval overlapping ``[start, end)``."""
        # Disjoint + sorted by start => ends ascending; the first interval with
        # end > start is the leftmost that can overlap.
        i = bisect.bisect_right(self.ends, start)
        starts = self.starts
        n = len(starts)
        out: list[list[int]] = []
        while i < n and starts[i] < end:
            out.append(self.ids[i])
            i += 1
        return out

    def clear(self, start: int, end: int) -> None:
        """Drop the ``[start, end)`` region, trimming intervals that straddle it."""
        self._split_at(start)
        self._split_at(end)
        lo = bisect.bisect_left(self.starts, start)
        hi = bisect.bisect_left(self.starts, end)
        if hi > lo:
            del self.starts[lo:hi]
            del self.ends[lo:hi]
            del self.ids[lo:hi]

    def assign(self, start: int, end: int, idx: int) -> None:
        """Make ``idx`` the sole owner of ``[start, end)`` (write hazard update)."""
        self.clear(start, end)
        i = bisect.bisect_left(self.starts, start)
        self.starts.insert(i, start)
        self.ends.insert(i, end)
        self.ids.insert(i, [idx])

    def add(self, start: int, end: int, idx: int) -> None:
        """Record ``idx`` as reading every offset in ``[start, end)``, filling gaps."""
        self._split_at(start)
        self._split_at(end)
        i = bisect.bisect_left(self.starts, start)
        cur = start
        while cur < end:
            if i < len(self.starts) and self.starts[i] < end:
                seg_start = self.starts[i]
                if seg_start > cur:
                    self.starts.insert(i, cur)
                    self.ends.insert(i, seg_start)
                    self.ids.insert(i, [idx])
                    i += 1
                self.ids[i].append(idx)
                cur = self.ends[i]
                i += 1
            else:
                self.starts.insert(i, cur)
                self.ends.insert(i, end)
                self.ids.insert(i, [idx])
                cur = end
                i += 1


def _compute_unit_deps(units: Sequence[ExecUnit], resolved_units: Sequence[Sequence[Buffer]]) -> list[list[int]]:
    """Return per-unit hazard dependencies (RAW/WAR/WAW) from captured buffers.

    Hazards are tracked per buffer base as disjoint, sorted byte-interval
    partitions, so each access inspects only the intervals it overlaps rather
    than every interval ever recorded for the base. Dependency lists are
    returned sorted ascending.
    """
    writers: dict[int, _IntervalPartition] = defaultdict(_IntervalPartition)
    readers: dict[int, _IntervalPartition] = defaultdict(_IntervalPartition)
    deps: list[list[int]] = [[] for _ in units]

    for idx, (unit, bufs) in enumerate(zip(units, resolved_units, strict=True)):
        writes = get_output_slots(unit.call.src[0])
        cur_deps: set[int] = set()

        for i, buf in enumerate(bufs):
            key = id(buf.base)
            start = buf._offset
            end = start + buf.nbytes
            for ids in writers[key].overlapping(start, end):
                cur_deps.update(ids)
            if i in writes:
                for ids in readers[key].overlapping(start, end):
                    cur_deps.update(ids)

        deps[idx] = sorted(cur_deps)

        for i, buf in enumerate(bufs):
            key = id(buf.base)
            start = buf._offset
            end = start + buf.nbytes
            if i in writes:
                writers[key].assign(start, end, idx)
                readers[key].clear(start, end)
            else:
                readers[key].add(start, end, idx)

    return deps


def _queue_id(unit: ExecUnit) -> int:
    return 1 if unit.root_op is Op.COPY else 0


def _build_queue_sync(
    units: Sequence[ExecUnit], deps: list[list[int]], queue_ids: list[int]
) -> tuple[list[list[Signal]], list[Signal | None], list[Signal]]:
    """Build cross-queue wait/signal sets using per-queue high-water marks."""
    queue_access: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(lambda: -1))
    needs_signal: set[int] = set()
    raw_waits: list[list[int]] = [[] for _ in units]

    for idx, my_deps in enumerate(deps):
        my_q = queue_ids[idx]
        waits: list[int] = []
        for dep_idx in my_deps:
            dep_q = queue_ids[dep_idx]
            if dep_q != my_q and queue_access[my_q][dep_q] < dep_idx:
                waits.append(dep_idx)
                queue_access[my_q][dep_q] = dep_idx
                needs_signal.add(dep_idx)
        raw_waits[idx] = waits

    sig_map = {uid: Signal() for uid in needs_signal}
    unit_signals: list[Signal | None] = [sig_map.get(i) for i in range(len(units))]
    unit_waits: list[list[Signal]] = [[sig_map[d] for d in ws] for ws in raw_waits]
    return unit_waits, unit_signals, list(sig_map.values())


def _copy_to_host_ptr(dst_ptr: int, src: memoryview) -> None:
    nbytes = src.nbytes
    if nbytes == 0:
        return
    if src.contiguous and not src.readonly:
        src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(src))
        ctypes.memmove(dst_ptr, src_ptr, nbytes)
    else:
        ctypes.memmove(dst_ptr, bytes(src), nbytes)


@dataclass(slots=True)
class _KernelNodeState:
    node: Any
    params: Any
    param_array: Any  # prevent GC of pointer chain that kernelParams references
    spec: CUDAKernelSpec
    dynamic_args: list[tuple[int, int]]


@dataclass(slots=True)
class _CopyNodeState:
    node: Any
    params: Any
    staging: Any | None
    memcpy_ctx: Any | None
    dst_idx: int | None
    src_idx: int | None


class CUDAQueueGraphRunner:
    """Dual-mode CUDA runner: queue (streams + events) or graph (CUDA graph + selective param updates)."""

    def __init__(self, units: tuple[ExecUnit, ...], bindings: list[Buffer]):
        from .backend import CUDAKernelProgram

        if not units:
            raise ValueError("empty execution plan")

        device = units[0].device
        assert Device.is_cuda(device), f"{device!r} is not a CUDA device"
        backend = Device[device]
        resolved_units = [[buf.ensure_allocated() for buf in unit.resolve(bindings)] for unit in units]
        prepared: list[ExecUnit[CUDAKernelProgram] | ExecUnit[None]] = []
        for unit, bufs in zip(units, resolved_units, strict=True):
            if unit.device != backend.device:
                raise ValueError(f"unit device {unit.device} != backend {backend.device}")
            if unit.root_op is Op.SINK:
                if not isinstance(unit.program, CUDAKernelProgram):
                    raise ValueError("unsupported program type for SINK unit")
                prepared.append(ExecUnit(unit.program, unit.call))
            elif unit.root_op is Op.COPY:
                if not _is_supported_copy(bufs, backend):
                    raise ValueError("unsupported COPY unit")
                prepared.append(ExecUnit(None, unit.call))
            else:
                raise ValueError(f"unsupported op {unit.root_op}")

        self.backend: CUDABackend = backend
        self.units = tuple(prepared)
        deps = _compute_unit_deps(units, resolved_units)
        self._owned_signals: list[Signal] = []
        self._kernel_nodes: list[_KernelNodeState] = []
        self._copy_nodes: list[_CopyNodeState] = []
        self._binding_indices: dict[Node, int] = {}
        self._copy_signals: dict[CUDABackend, Signal] = {}
        self._staging_bufs: list[Any] = []
        self._staging_event: Signal | None = None
        self._graph: Any | None = None
        self._graph_exec: Any | None = None
        self._closed: bool = False

        self._try_build_graph(resolved_units, deps)
        if self._graph_exec is None:
            self.queue_ids = [_queue_id(unit) for unit in units]
            self._has_copy_ops = 1 in self.queue_ids
            self.unit_waits, self.unit_signals, signals = _build_queue_sync(units, deps, self.queue_ids)
            self._owned_signals.extend(signals)
            if self._has_copy_ops:
                self._entry_signal = Signal()
                self._exit_signal = Signal()
                self._owned_signals.extend([self._entry_signal, self._exit_signal])

        self._device_bindings = tuple(idx for node, idx in self._binding_indices.items() if node.device.startswith("cuda:"))

    def _binding_index(self, binding: Node) -> int | None:
        if unwrap_view(binding)[0].op is not Op.PARAM:
            return None
        return self._binding_indices.setdefault(binding, len(self._binding_indices))

    def _add_kernel_nodes(self, dep_nodes: list[Any], unit: ExecUnit[CUDAKernelProgram], bufs: list[Buffer]) -> Any:
        prev = None
        for spec in unit.program.specs:
            spec.bind_bufs(bufs)
            params, param_array = cuda.make_kernel_node_params(spec.func, spec.grid, spec.block, spec.args, spec.shared_mem)
            node = cuda.graph_add_kernel_node(self._graph, dep_nodes if prev is None else [prev], params)
            dynamic_args = [
                (arg_pos, binding_idx)
                for arg_pos, buf_idx in enumerate(spec.buf_indices)
                if (binding_idx := self._binding_index(unit.call.src[buf_idx + 1])) is not None
            ]
            # CUDA copies arguments when adding the node; retain only state needed for replay updates.
            if dynamic_args:
                self._kernel_nodes.append(_KernelNodeState(node=node, params=params, param_array=param_array, spec=spec, dynamic_args=dynamic_args))
            prev = node
        if prev is None:
            raise ValueError("SINK unit has no kernel specs")
        return prev

    def _add_copy_node(self, dep_nodes: list[Any], src_copies: dict[CUDABackend, list[Any]], unit: ExecUnit[None], bufs: list[Buffer]) -> Any:
        dst, src = bufs[:2]
        dst_binding, src_binding = unit.call.src[1:3]
        dst_idx, src_idx = self._binding_index(dst_binding), self._binding_index(src_binding)

        staging = None
        src_dev = Device[src.device] if Device.is_cuda(src.device) else self.backend
        memcpy_ctx = src_dev.context
        if Device.is_cuda(src.device):
            params = cuda.make_memcpy_dtod_params(int(dst._buf), int(src._buf), dst.nbytes)
        else:
            staging = cuda.mem_host_alloc(dst.nbytes)
            self._staging_bufs.append(staging)
            _copy_to_host_ptr(int(staging), src.as_buffer())
            params = cuda.make_memcpy_htod_params(int(dst._buf), int(staging), dst.nbytes)

        node = cuda.graph_add_memcpy_node(self._graph, dep_nodes, params, ctx=memcpy_ctx)
        if src_dev is not self.backend:
            src_copies.setdefault(src_dev, []).append(node)
        if dst_idx is not None or src_idx is not None:
            self._copy_nodes.append(
                _CopyNodeState(node=node, params=params, staging=staging, memcpy_ctx=memcpy_ctx, dst_idx=dst_idx, src_idx=src_idx)
            )
        return node

    def _destroy_graph(self) -> None:
        _safe_destroy(self._graph_exec, cuda.graph_exec_destroy)
        _safe_destroy(self._graph, cuda.graph_destroy)
        self._graph_exec = self._graph = None
        for ptr in self._staging_bufs:
            _safe_destroy(ptr, cuda.mem_host_free)
        self._staging_bufs.clear()
        self._kernel_nodes.clear()
        self._copy_nodes.clear()
        self._binding_indices.clear()
        self._copy_signals.clear()

    def _update_kernel_node(self, state: _KernelNodeState, addresses: dict[int, int]) -> None:
        changed = False
        for arg_pos, binding_idx in state.dynamic_args:
            addr = addresses[binding_idx]
            if state.spec.args[arg_pos].value != addr:
                state.spec.args[arg_pos].value = addr
                changed = True
        if changed:
            cuda.graph_exec_kernel_node_set_params(self._graph_exec, state.node, state.params)

    def _update_copy_node(self, state: _CopyNodeState, buffers: list[Buffer], addresses: dict[int, int]) -> None:
        if state.staging is not None:
            if state.src_idx is not None:
                _copy_to_host_ptr(int(state.staging), buffers[state.src_idx].as_buffer())
        elif state.src_idx is not None:
            state.params.srcDevice = addresses[state.src_idx]
        if state.dst_idx is not None:
            state.params.dstDevice = addresses[state.dst_idx]
        cuda.graph_exec_memcpy_node_set_params(self._graph_exec, state.node, state.params, ctx=state.memcpy_ctx)

    def _try_build_graph(self, resolved_units: list[list[Buffer]], deps: list[list[int]]) -> None:
        cuda.ctx_set_current(self.backend.context)
        src_copies: dict[CUDABackend, list[Any]] = {}
        try:
            self._graph = cuda.graph_create()
            unit_nodes: list[Any] = []

            for idx, unit in enumerate(self.units):
                dep_nodes = [unit_nodes[d] for d in deps[idx]]
                if unit.program is not None:
                    node = self._add_kernel_nodes(dep_nodes, unit, resolved_units[idx])
                else:
                    node = self._add_copy_node(dep_nodes, src_copies, unit, resolved_units[idx])
                unit_nodes.append(node)

            # Allow source reuse as soon as all its copies finish.
            for src_dev, copy_nodes in src_copies.items():
                sig = Signal()
                self._copy_signals[src_dev] = sig
                self._owned_signals.append(sig)
                cuda.graph_add_event_record_node(self._graph, copy_nodes, sig._event)
            self._graph_exec = cuda.graph_instantiate(self._graph, self.backend.compute_queue.stream)
            # Only dynamic H2D staging writes can race with an in-flight launch's DMA read.
            if any(s.staging is not None and s.src_idx is not None for s in self._copy_nodes):
                self._staging_event = Signal()
                self._owned_signals.append(self._staging_event)
                self._staging_event.record(self.backend.compute_queue.stream)
            if DEBUG >= 1:
                print(f"graph runner: instantiated {len(self.units)} units ({sum(u.root_op is Op.COPY for u in self.units)} copies)")
        except Exception as e:
            tb = traceback.format_exc()
            warnings.warn(f"CUDA graph construction failed, falling back to queue mode: {e}\n{tb}", stacklevel=2)
            self._destroy_graph()

    def _dispatch_copy_queue(self, bufs: list[Buffer]) -> None:
        dst, src = bufs[:2]
        if not Device.is_cuda(src.device):
            self.backend.allocator._enqueue_copyin(dst._buf, src.as_buffer())
            return
        src_dev = Device[src.device]
        if src_dev is not self.backend:
            cuda.ctx_set_current(src_dev.context)
            src_dev.compute_queue.signal(src_dev._compute_signal)
            cuda.ctx_set_current(self.backend.context)
            self.backend.copy_queue.wait(src_dev._compute_signal)
        self.backend.copy_queue.copy_dtod(int(dst._buf), int(src._buf), dst.nbytes)
        if src_dev is not self.backend:
            self.backend.copy_queue.signal(self.backend._copy_signal)
            cuda.ctx_set_current(src_dev.context)
            src_dev.compute_queue.wait(self.backend._copy_signal)
            cuda.ctx_set_current(self.backend.context)

    def _run_queue(self, bindings: list[Buffer]) -> None:
        cuda.ctx_set_current(self.backend.context)
        queues = (self.backend.compute_queue, self.backend.copy_queue)

        if self._has_copy_ops:
            self.backend.compute_queue.signal(self._entry_signal)
            self.backend.copy_queue.wait(self._entry_signal)

        for idx, unit in enumerate(self.units):
            q = queues[self.queue_ids[idx]]
            bufs = [buf.ensure_allocated() for buf in unit.resolve(bindings)]
            for sig in self.unit_waits[idx]:
                q.wait(sig)

            if unit.program is not None:
                for spec in unit.program.specs:
                    self.backend._launch_spec(spec, bufs)
            else:
                self._dispatch_copy_queue(bufs)

            if (sig := self.unit_signals[idx]) is not None:
                q.signal(sig)

        if self._has_copy_ops:
            self.backend.copy_queue.signal(self._exit_signal)
            self.backend.compute_queue.wait(self._exit_signal)

    def _run_graph(self, bindings: list[Buffer]) -> None:
        assert self._graph_exec is not None
        cuda.ctx_set_current(self.backend.context)
        if self._staging_event is not None:
            self._staging_event.synchronize()

        buffers = [resolve_buffer(node, bindings).ensure_allocated() for node in self._binding_indices]
        addresses = {idx: int(buffers[idx]._buf) for idx in self._device_bindings}

        for src_dev in self._copy_signals:
            cuda.ctx_set_current(src_dev.context)
            src_dev.compute_queue.signal(src_dev._compute_signal)

        cuda.ctx_set_current(self.backend.context)
        for src_dev in self._copy_signals:
            self.backend.compute_queue.wait(src_dev._compute_signal)

        for state in self._kernel_nodes:
            self._update_kernel_node(state, addresses)

        for state in self._copy_nodes:
            self._update_copy_node(state, buffers, addresses)

        cuda.graph_launch(self._graph_exec, self.backend.compute_queue.stream)
        if self._staging_event is not None:
            self._staging_event.record(self.backend.compute_queue.stream)
        if self._copy_signals:
            for src_dev, sig in self._copy_signals.items():
                cuda.ctx_set_current(src_dev.context)
                src_dev.compute_queue.wait(sig)
            cuda.ctx_set_current(self.backend.context)

    def __call__(self, bindings: list[Buffer]) -> None:
        if self._graph_exec is not None:
            self._run_graph(bindings)
        else:
            self._run_queue(bindings)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            cuda.ctx_push_current(self.backend.context)
        except Exception:
            # Context already destroyed (e.g. backend finalized first during shutdown).
            # CUDA freed all context-owned resources; just prevent Signal.__del__ from
            # trying to destroy already-freed events.
            for sig in self._owned_signals:
                sig._event = 0
            self._owned_signals.clear()
            return
        try:
            self._destroy_graph()
            for sig in self._owned_signals:
                _safe_destroy(sig, Signal.close)
            self._owned_signals.clear()
        finally:
            cuda.ctx_pop_current()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _is_supported_copy(bufs: list[Buffer], backend: CUDABackend) -> bool:
    if len(bufs) < 2:
        return False
    return bufs[0].device == backend.device


__all__ = ["CUDAQueueGraphRunner"]
