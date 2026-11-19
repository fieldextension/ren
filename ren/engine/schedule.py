from collections import OrderedDict
from collections.abc import Callable, Container
from functools import cache

from ren.context import DEBUG, DOT_TILE
from ren.engine import rules
from ren.engine.pattern import rewrite
from ren.engine.realize import buffer_binding
from ren.engine.rules.callify import transform_to_call
from ren.ir import Op
from ren.ir.node import Node, ParamArg
from ren.ir.op import GroupOp

# Trace callback receives (pass_name, from_node, to_node) for each node replacement.
# schedule() always supplies pass names, while the lower-level rewrite engine
# treats the name as optional metadata.
TraceCallback = Callable[[str | None, Node, Node], None]


def _assert_no_ops(root: Node, forbidden: Container[Op], stage: str) -> None:
    """Validate that no forbidden ops remain after a pipeline stage."""
    if not DEBUG:
        return
    for n in root.toposort():
        assert n.op not in forbidden, f"{n.op} should not exist after {stage}"


_prepare_cache: OrderedDict[Node, Node] = OrderedDict()


def _prepare_function(body: Node, trace: TraceCallback | None) -> Node:
    if (prepared := _prepare_cache.get(body)) is not None:
        _prepare_cache.move_to_end(body)
        return prepared
    prepared = _prepare_body(body, trace)
    _prepare_cache[body] = prepared
    if len(_prepare_cache) > 256:
        _prepare_cache.popitem(last=False)
    return prepared


_simplify_rules = rules.algebraic + rules.ring + rules.const + rules.copy


def _prepare_body(root: Node, trace: TraceCallback | None) -> Node:
    root = rewrite(root, _simplify_rules, name="simplify", trace=trace)
    prepare = rules.inject_domains + rules.prepare_calls(lambda body: _prepare_function(body, trace))
    return rewrite(root, prepare, name="prepare", trace=trace)


_schedule_cache: OrderedDict[tuple[Node, int], tuple[Node, tuple[Node, ...]]] = OrderedDict()
_store_rules = rules.bufferize_to_store + rules.buffer_views + rules.copy + rules.algebraic + rules.ring + rules.domain


def _prepare_kernel(root: Node, trace: TraceCallback | None) -> Node:
    _assert_no_ops(root, GroupOp.Lowered, "lower_ops")

    # Analyze polynomial bounds before expanding constants and indexing views.
    analysis = rules.analyze_ring_lowering(root)
    root = rewrite(root, rules.lower_ring(analysis), bottom_up=True, name="lower_ring", trace=trace)
    root = rewrite(root, rules.indexify, name="indexify", trace=trace)
    # Quotient views need the ring to compute their offsets in indexify.
    return rewrite(root, rules.strip_ring, name="strip_ring", trace=trace)


def _schedule_body(root: Node, trace: TraceCallback | None) -> tuple[Node, tuple[Node, ...]]:
    allocations = (node for node in root.toposort() if node.op is Op.ALLOC)
    root = root.substitute({node: node.replace(arg=ParamArg(-slot - 1, node.arg.size, node.arg.device)) for slot, node in enumerate(allocations)})
    key = (root, DOT_TILE.value)
    if (scheduled := _schedule_cache.get(key)) is not None:
        _schedule_cache.move_to_end(key)
        return scheduled

    # First decide materialization boundaries, then lower them to STORE/AFTER.
    bplan = rules.bufferize_plan(root)
    reuse_outputs = rules.reuse_outputs(root)
    root = rewrite(root, rules.bufferize(bplan) + reuse_outputs, bottom_up=True, name="bufferize", trace=trace)
    prepare_kernel = cache(lambda body: _prepare_kernel(body, trace))
    root = rewrite(root, _store_rules + rules.kernelize(prepare_kernel, reuse_outputs), name="kernelize", trace=trace)
    if (fusion := rules.dot_fusion(root)) is not None:
        # Replace before descending so dependencies introduced by a merged kernel are visited too.
        root = rewrite(root, fusion, bottom_up=True, name="dot_fusion", trace=trace)

    scheduled = Node(Op.LINEAR, src=tuple(node for node in root.toposort() if node.op is Op.CALL)), root.src
    _schedule_cache[key] = scheduled
    if len(_schedule_cache) > 256:
        _schedule_cache.popitem(last=False)
    return scheduled


def schedule(root: Node, *, trace: TraceCallback | None = None) -> tuple[Node, dict[Node, Node]]:
    """Schedule a computation graph into an ordered LINEAR of CALLs."""
    assert root.op is Op.SINK, f"schedule expects SINK root, got {root.op}"

    original_outputs = root.src
    outputs = tuple(value.bufferize() if value._device is not None and not value.has_buffer_identity else value for value in original_outputs)
    root = _prepare_body(root.replace(src=outputs), trace)
    prepared_outputs = root.src
    root = rewrite(root, rules.schedule_calls(lambda body: _schedule_body(body, trace)[0]), enter_calls=True, name="schedule_calls", trace=trace)
    call = transform_to_call(root)
    if trace is not None:
        trace("callify", root, call)

    linear, output_nodes = _schedule_body(call.src[0], trace)
    root = rewrite(Node.sink(linear, *output_nodes).call(*call.inputs), rules.resolve_calls, name="resolve_calls", trace=trace)
    linear, output_nodes = root.src[0], root.src[1:]
    bindings: dict[Node, Node] = {}

    def binding(node: Node) -> Node:
        if node not in bindings:
            bindings[node] = buffer_binding(node)
        return bindings[node]

    linear = Node(
        Op.LINEAR, src=tuple(node.src[0].call(*(binding(source) for source in node.inputs)) for node in linear.toposort() if node.op is Op.CALL)
    )
    if DEBUG:
        for kernel in linear.src:
            assert all(n.op is not Op.BUFFER for n in kernel.src[0].toposort(enter_calls=True)), "CALL body must not contain BUFFER nodes"

    if DEBUG >= 2:
        print(f"schedule: ops={len(linear.src)}")

    outputs: dict[Node, Node] = {}
    for original, prepared, value in zip(original_outputs, prepared_outputs, output_nodes, strict=True):
        ring, domain = prepared.ring, prepared.domain
        assert ring is not None and domain is not None, "materialized output missing ring/domain metadata"
        # Return constants directly without changing storage needed by execution.
        base = prepared.base
        outputs[original] = (base if base.op is Op.CONST else binding(value)).as_ring(ring, domain)
    return linear, outputs
