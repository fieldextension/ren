from collections.abc import Callable

from ren.kernel import KernelInfo
from ren.engine.pattern import Gate, RuleSet, pat, rewrite, rules
from ren.ir import GroupOp, Op
from ren.ir.node import Node
from ren.dtype import dtypes

from .indexing import is_conversion_store_value
from .lowering import lower_ops


def add_buffer(bindings: dict[Node, tuple[Node, Node]], dep: Node) -> Node:
    """Add a buffer argument and return its parameter, preserving all producer dependencies."""
    buf = dep.base
    if buf not in bindings:
        bindings[buf] = (Node.param(len(bindings), buf), dep)
    param, prior = bindings[buf]
    if prior is not dep and dep is not buf:
        bindings[buf] = (param, dep if prior is buf else buf.after(prior, dep))
    return param


def _parameterize(bindings: dict[Node, tuple[Node, Node]]) -> RuleSet:
    """Replace buffer references with positional parameters in kernel IR.

    Storage inputs become positional PARAM nodes before kernel lowering.
    Their ring/domain annotations remain available to the lowering passes.
    AFTER nodes retain producer dependencies in the CALL arguments.

    Example:
        STORE(INDEX(BUFFER_0, gidx), LOAD(INDEX(BUFFER_1, gidx)))
      -> STORE(INDEX(PARAM(0), gidx), LOAD(INDEX(PARAM(1), gidx)))
    with bindings recording {BUFFER_0: (PARAM(0), BUFFER_0), BUFFER_1: (PARAM(1), dep_of_BUFFER_1)}.
    """

    def parameter(node: Node) -> Gate:
        param = add_buffer(bindings, node)
        return Gate(param.as_ring(node.ring, node.domain) if node.ring is not None else param)

    def handle_after(after: Node) -> Gate | None:
        if after._device is None:
            raise RuntimeError("AFTER at a kernel boundary requires storage")
        if after.base.op in GroupOp.Defines | {Op.BUFFER_VIEW}:
            return parameter(after)
        return None

    return rules([
        (pat.node(GroupOp.Defines | {Op.BUFFER_VIEW, Op.BUFFERIZE}), parameter),
        (pat.after(Op.AFTER), handle_after),
    ])


def kernelize(prepare_kernel: Callable[[Node], Node], reuse_outputs: RuleSet) -> RuleSet:
    """Turn each top-level STORE into a compute or copy CALL, with the output buffer in slot 0."""

    def split_store(store: Node) -> Node | None:
        target, value = store.src
        out_buf = (target.src[0] if target.op is Op.INDEX else target).base
        if out_buf.op not in GroupOp.Defines:
            return None

        conv = value.base
        conv_src = conv.src[0] if is_conversion_store_value(value) and conv.src else None
        ring = conv_src.ring if conv_src is not None and conv_src.ring is not None else value.ring
        assert ring is not None or conv.op is Op.COPY

        domain = conv_src.domain if conv_src is not None else value.domain
        bindings: dict[Node, tuple[Node, Node]] = {}
        add_buffer(bindings, out_buf)

        lowered = rewrite(store, lower_ops(store.device) + reuse_outputs + _parameterize(bindings), bottom_up=True)
        stored, target_base = lowered.src[1].base, lowered.src[0].base
        if target_base.op is Op.PARAM and stored.op is Op.PARAM:
            lowered = stored.copy_to_device(target_base.device)
        elif stored.op is Op.COPY:
            lowered = stored
        else:
            body = Node(Op.SINK, dtypes.void, src=(lowered,), arg=KernelInfo(device=store.device, ring=ring, domain=domain))
            lowered = prepare_kernel(body)

        return lowered.call(*(dep for _, dep in bindings.values()))

    return rules([(pat.store(Op.STORE), split_store)])
