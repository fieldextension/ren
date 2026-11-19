from ren.dtype import DType, dtypes
from ren.engine import pattern, rules
from ren.engine.pattern import pat, rewrite
from ren.ir import Node, Op, spec
from ren.ir.op import GroupOp
from ren.ir.spec import validate
from ren.kernel import KernelSpec


_constant_rules = rules.const_casts + rules.const_loads
_specialized = pattern.rules([
    (pat().sink(pat().store(pat(spec.specialized_ops))), lambda: True),
])


def lower_kernel(ast: Node, supported: frozenset[DType]) -> KernelSpec:
    """Validate a scheduled kernel and lower its arithmetic for the target."""
    kernel = KernelSpec(ast)
    if replacements := {n: out for n in kernel.program if (out := _constant_rules.match(n)) is not None}:
        kernel = KernelSpec(ast.substitute(replacements))
    if kernel.ast.op is not Op.SINK:
        raise ValueError("kernel requires a SINK")
    specialized = _specialized.match(kernel.ast) is True
    validate(kernel.program, spec.specialized if specialized else spec.modular)
    slots = kernel.argument_slots
    if len(set(slots)) != len(slots):
        raise ValueError("kernel requires one PARAM definition per slot")
    if {n for n in kernel.program if n.op is Op.STORE} != set(kernel.stores):
        raise ValueError("kernel must contain only top-level STOREs")

    if not specialized:
        outputs = set(kernel.output_slots)
        if len(outputs) != len(kernel.stores):
            raise ValueError("generated kernel requires distinct output slots")
        definitions = {s for n in kernel.program if n.op is Op.AFTER for s in n.src[1:]}
        for node in kernel.program:
            if node.op is Op.LOAD:
                buffer = node.src[0].src[0]
                if buffer.op is Op.PARAM and buffer.arg.slot in outputs:
                    raise ValueError("generated kernel cannot read its output slots")
            if node.op is Op.MOD_REDUCE:
                (value,) = node.src
                if value.base.op in GroupOp.ReducedInputs and (value.op is Op.AFTER or value in definitions):
                    raise ValueError("AFTER cannot split a modular instruction")

        if any(n.op is Op.MOD_REDUCE for n in kernel.program):
            kernel = KernelSpec(rules.lower_modular(kernel))
        if dtypes.uint128 not in supported and any(n.dtype is dtypes.uint128 for n in kernel.program):
            kernel = KernelSpec(rewrite(kernel.ast, rules.lower_dtypes() + rules.fold))
        validate(kernel.program, spec.scalar)

    if not {n.dtype for n in kernel.program} <= supported | {dtypes.weakint, dtypes.void}:
        raise ValueError("Unsupported kernel dtype")
    return kernel
