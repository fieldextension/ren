from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ren.buffer import Buffer
from ren.device import Device
from ren.ir import Op
from ren.ir.node import Node
from ren.kernel import KernelInfo
from ren.prof import trace


class Program(Protocol):
    """Executable over a positional list of buffers."""

    def __call__(self, bufs: list[Buffer], /) -> None: ...


@dataclass(frozen=True, slots=True, eq=False)
class ExecUnit[P: Program | None = Program]:
    """Compiled operation whose buffer bindings live in CALL IR."""

    program: P
    call: Node

    def __post_init__(self) -> None:
        if self.call.op is not Op.CALL or len(self.call.src) < 2:
            raise ValueError("execution unit requires a CALL with an output binding")

    @property
    def root_op(self) -> Op:
        return self.call.src[0].op

    @property
    def device(self) -> str:
        return self.call.device

    def resolve(self, bindings: Sequence[Buffer] = ()) -> list[Buffer]:
        return call_buffers(self.call, bindings)

    def __call__[Q: Program](self: ExecUnit[Q], bindings: Sequence[Buffer] = ()) -> None:
        self.program([buf.ensure_allocated() for buf in self.resolve(bindings)])


def rewrite_buffers(root: Node, replace_buffer: Callable[[Node, Buffer], Node | None]) -> Node:
    """Rewrite concrete buffer bindings while leaving callable bodies opaque."""
    return root.map(
        lambda node: (
            replace_buffer(node, node.buffer)
            if node.op is Op.BUFFER or (node.op is Op.BUFFER_VIEW and unwrap_view(node)[0].op is Op.BUFFER)
            else None
        )
    )


def buffer_binding(node: Node) -> Node:
    """Collapse scheduling wrappers into a self-contained buffer binding."""
    if node.op in {Op.AS_RING, Op.AFTER}:
        return buffer_binding(node.src[0])
    if node.op is Op.BUFFER_VIEW:
        source = buffer_binding(node.src[0])
        return node if source is node.src[0] else node.replace(src=(source,))
    return node


def unwrap_view(node: Node) -> tuple[Node, int, int]:
    """Return base storage, size, and element offset, validating each view."""
    node = buffer_binding(node)
    if node.op is Op.BUFFER_VIEW:
        base, parent_size, parent_offset = unwrap_view(node.src[0])
        size, offset = node.arg
        if node.dtype != base.dtype:
            raise ValueError(f"buffer view dtype {node.dtype} != source dtype {base.dtype}")
        assert offset >= 0 and size >= 0 and offset + size <= parent_size, "invalid buffer view range"
        return base, size, parent_offset + offset
    if node.op not in {Op.BUFFER, Op.PARAM}:
        raise TypeError(f"expected buffer binding, got {node.op}")
    return node, node.size, 0


def resolve_buffer(node: Node, bindings: Sequence[Buffer]) -> Buffer:
    """Resolve a concrete or positional buffer expression."""
    if node.op is Op.PARAM:
        if node.arg.slot >= len(bindings):
            raise RuntimeError(f"missing execution buffer for parameter {node.arg.slot}")
        source = bindings[node.arg.slot]
        spec = source.spec
        if node.dtype != spec.dtype or node.arg.size != spec.size or node.arg.device != spec.device:
            raise ValueError(f"execution buffer {node.arg.slot} does not match its parameter")
        return source
    if node.op in {Op.AS_RING, Op.AFTER}:
        return resolve_buffer(node.src[0], bindings)
    if node.op is Op.BUFFER_VIEW:
        source = resolve_buffer(node.src[0], bindings)
        if node.dtype != source.dtype:
            raise ValueError(f"buffer view dtype {node.dtype} != source dtype {source.dtype}")
        return source.view(*node.arg)
    if node.op is Op.BUFFER:
        return node.buffer
    raise TypeError(f"expected buffer binding, got {node.op}")


def call_buffers(call: Node, bindings: Sequence[Buffer] = ()) -> list[Buffer]:
    """Resolve a CALL's positional buffer arguments."""
    return [src.buffer if src.op is Op.BUFFER else resolve_buffer(src, bindings) for src in call.inputs]


def call_op(call: Node) -> Op:
    """Return the operation computed by a CALL body."""
    body = call.src[0]
    if body.op is Op.SINK and body.src and body.src[0].op is Op.STORE:
        return body.src[0].src[1].op
    return body.op


# Scheduled calls collected by the active JIT capture.
capturing: dict[object, list[Node]] = {}


def _lower_compute(call: Node) -> ExecUnit:
    """Lower a compute call with a SINK body to an ExecUnit."""
    ast = call.src[0]
    assert ast.op is Op.SINK, f"expected compute SINK, got {ast.op}"
    info = ast.arg
    assert isinstance(info, KernelInfo), "compute SINK must carry KernelInfo metadata"
    backend = Device[info.device]
    kernel_name = f"kernel_{call_op(call).name.lower()}"
    with trace(f"make_kernel:{kernel_name}", device=info.device):
        program = backend.make_kernel(ast)
    return ExecUnit(program=program, call=call)


def _lower_copy(call: Node) -> ExecUnit:
    """Lower a copy call to an ExecUnit."""
    dst, src = call.src[1:3]
    dst_alloc = Device[dst.device].allocator
    same_type = dst.device.split(":")[0] == src.device.split(":")[0]

    # Use transfer if: allocator supports it AND same device type
    if dst_alloc.supports_transfer and same_type:

        def run(bufs: list[Buffer]) -> None:
            dst, src = bufs
            dst._allocator.transfer(dst._buf, src._buf, dst.nbytes, Device[src.device], Device[dst.device])

    else:
        # Host-mediated fallback
        def run(bufs: list[Buffer]) -> None:
            dst, src = bufs
            host = memoryview(bytearray(src.nbytes))
            src.copyout(host)
            dst.copyin(host)

    return ExecUnit(program=run, call=call)


# Dispatch table: scheduled root op -> lowerer function
_lowerers = {
    Op.SINK: _lower_compute,  # compute kernels use SINK(STORE, ...)
    Op.COPY: _lower_copy,
}


def make_exec_units(linear: Node, assigned: dict[Buffer, Buffer] | None = None) -> list[ExecUnit]:
    """Lower a LINEAR schedule into executable units."""
    assert linear.op is Op.LINEAR
    with trace("make_exec_units"):
        if assigned:
            replacement_nodes: dict[Buffer, Node] = {}

            def replace_assigned(_node: Node, buf: Buffer) -> Node | None:
                if (replacement := assigned.get(buf)) is None:
                    return None
                if (node := replacement_nodes.get(replacement)) is None:
                    node = replacement_nodes[replacement] = Node.from_buffer(replacement)
                return node

            linear = rewrite_buffers(linear, replace_assigned)
        return [_lowerers[call.src[0].op](call) for call in linear.src]


def run_exec_units(units: list[ExecUnit]) -> None:
    """Execute and consume units, releasing intermediates after their last use."""
    idx = 0

    while units:
        unit = units.pop(0)
        with trace(f"exec_unit[{idx}]", device=unit.device):
            unit()
        idx += 1
