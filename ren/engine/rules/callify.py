from collections.abc import Callable

from ren.engine.pattern import Gate, RuleSet, pat, rules
from ren.ir import Node, Op


def prepare_calls(prepare_body: Callable[[Node], Node]) -> RuleSet:
    prepared: set[Node] = set()

    def prepare(call: Node) -> Gate | None:
        body, inputs = call.src[0], call.inputs
        if call in prepared or body.op is not Op.SINK or body.arg is not None:
            return None

        bindings = list(inputs)

        def resolve(node: Node) -> Node | None:
            if node.op is not Op.AS_RING or (param := node.src[0]).op is not Op.PARAM:
                return None
            if param.arg.slot >= len(inputs):
                raise RuntimeError(f"ren.function: missing argument for parameter {param.arg.slot}")
            source = bindings[param.arg.slot]
            return node.replace(arg=(source.ring, source.domain)) if source.ring is not None else None

        body = prepare_body(body.map(resolve))
        for output in body.src:
            target, store = output.src
            value = store.src[1]
            assert value.ring is not None and value.domain is not None, "function output missing ring/domain metadata"
            bindings[target.arg.slot] = inputs[target.arg.slot].as_ring(value.ring, value.domain)
        result = body.call(*bindings)
        prepared.add(result)
        return Gate(result)

    def output_domain(value: Node, call: Node) -> Node | None:
        if call not in prepared:
            return None
        for output in call.src[0].src:
            binding = call.inputs[output.src[0].arg.slot]
            if binding.base is value.base:
                return binding.after(call)
        return None

    return rules([
        (pat.call(Op.CALL), prepare),
        (pat(Op.AFTER, (pat.value(Op.AS_RING, pat(Op.ALLOC)), pat.call(Op.CALL))), output_domain),
    ])


def transform_to_call(root: Node) -> Node:
    """Parameterize a root SINK so its schedule can be reused."""
    assert root.op is Op.SINK

    body = root.replace(src=tuple(value.src[0] if value.op is Op.CONTIGUOUS else value for value in root.src))
    bindings: list[Node] = []

    def parameterize(node: Node) -> Node | None:
        if node.op not in {Op.BUFFER, Op.BUFFER_VIEW}:
            return None
        slot = len(bindings)
        bindings.append(node)
        return Node.param(slot, node)

    parameterized = body.map(parameterize)
    return parameterized.call(*bindings)


def schedule_calls(schedule_body: Callable[[Node], Node]) -> RuleSet:
    def schedule(call: Node) -> Gate | None:
        body = call.src[0]
        if body.op is not Op.SINK or body.arg is not None:
            return None
        return Gate(schedule_body(body).call(*call.inputs))

    return rules([(pat.call(Op.CALL), schedule)])


def _instantiate_call(call: Node) -> Node | None:
    body, sources = call.src[0], call.inputs
    if body.op not in {Op.SINK, Op.LINEAR} or body.arg is not None:
        return None

    def bind(node: Node) -> Node | None:
        if node.op is Op.PARAM:
            if node.arg.slot >= len(sources):
                raise RuntimeError(f"ren.function: missing argument for parameter {node.arg.slot}")
            source = sources[node.arg.slot]
            if (node.dtype, node.arg.size, node.arg.device) != (source.dtype, source._storage_size(), source.device):
                raise ValueError(f"ren.function: argument {node.arg.slot} does not match its parameter")
            return source
        if node.op is Op.ALLOC:
            return Node.new_buffer(node.device, node.size, node.dtype)
        return None

    return body.map(bind)


resolve_calls = rules([(pat.call(Op.CALL), _instantiate_call)])
