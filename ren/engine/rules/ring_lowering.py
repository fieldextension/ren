from dataclasses import dataclass

from ren.dtype import dtypes
from ren.engine.pattern import RuleSet, pat, rules
from ren.ir import Op
from ren.ir.op import GroupOp
from ren.ir.node import Node
from ren.ring import Domain, RingSpec


@dataclass(frozen=True, slots=True)
class RingLoweringAnalysis:
    overflow_add_reduce_edges: frozenset[tuple[Node, Node]]


def _is_view_over_compute(n: Node) -> bool:
    return n.view_offset is not None and not n.has_buffer_identity


def _max_safe_bound(ring: RingSpec) -> int:
    max_prime_bits = max(p.bit_length() for p in ring.moduli)
    coeff_bits = ring.dtype.itemsize * 8
    return max(2, 1 << (coeff_bits - max_prime_bits - 1))


def analyze_ring_lowering(root: Node) -> RingLoweringAnalysis:
    """Analyze additive bound growth to find edges needing overflow reduction.

    Walks the DAG bottom-up, tracking how many unreduced additions accumulate
    at each node. When an ADD's bound exceeds the safe limit for its ring
    (based on modulus bit-width), the (ADD, child) edges contributing growth
    are flagged for MOD_REDUCE insertion by lower_ring.

    Example:
        ADD(ADD(a, b), c) where bound(ADD(a,b))=2, bound(c)=1 -> total=3
      If 3 > max_safe, edges (outer_ADD, inner_ADD) are flagged.
    """
    bounds: dict[Node, int] = {}
    max_safe: dict[RingSpec, int] = {}

    def visit(node: Node) -> int:
        base = node.base
        if base.op is Op.MOD_REDUCE:
            return 1
        if base.op is Op.ADD:
            return sum(bounds.get(s.base, 1) for s in base.src)
        if base.op in {Op.COPY, Op.CONTIGUOUS} or (base.view_offset is not None):
            assert base.src, f"{base.op} must have a source for ring lowering analysis"
            return bounds.get(base.src[0].base, 1)
        return 1

    for node in root.toposort():
        bounds[node] = visit(node)
    base_bounds = {n: v for n, v in bounds.items() if n is n.base}

    overflow_add_reduce_edges: set[tuple[Node, Node]] = set()
    for add, add_bound in base_bounds.items():
        if add.op is not Op.ADD:
            continue
        ring = add.ring
        if ring is None:
            continue
        limit = max_safe.setdefault(ring, _max_safe_bound(ring))
        if add_bound <= limit:
            continue
        for child in add.src:
            child_base = child.base
            if base_bounds.get(child_base, 1) > 1:
                overflow_add_reduce_edges.add((add, child_base))

    return RingLoweringAnalysis(overflow_add_reduce_edges=frozenset(overflow_add_reduce_edges))


def _lower_const_poly(n: Node, c: Node) -> Node | None:
    """Lower ring-constant polynomial semantics to explicit indexing ops.

    A scalar CONST inside a ring has polynomial meaning that depends on domain:
      - NTT:   c broadcasts to every slot -> [c, c, c, ...]  (just MOD_REDUCE)
      - COEFF: c is the degree-0 coefficient -> [c, 0, 0, ...]  (conditional on index)

    This pass replaces AS_RING(CONST(c)) with the explicit form, including
    per-limb modular reduction so backends see canonical values. The result is
    re-wrapped in AS_RING since MOD_REDUCE/WHERE do not carry domain.

    Example (COEFF, n=4):
        AS_RING(CONST(7), ring=R, domain=COEFF)
      -> AS_RING(WHERE((GIDX & 3) == 0, MOD_REDUCE(7, R), MOD_REDUCE(0, R)), R, COEFF)

    Example (NTT):
        AS_RING(CONST(7), ring=R, domain=NTT)
      -> AS_RING(MOD_REDUCE(7, R), R, NTT)
    """
    ring, dom = n.ring, n.domain
    assert ring is not None and dom is not None, "AS_RING must carry ring and domain"

    reduced = c.mod_reduce(ring)
    if dom is Domain.NTT:
        lowered = reduced
    elif dom is Domain.COEFF:
        idx = Node(Op.GIDX, dtypes.index)
        coeff_idx = idx & (ring.n - 1)
        zero = Node.const(0).mod_reduce(ring)
        lowered = coeff_idx.eq(0).where(reduced, zero)
    else:
        return None

    # Re-anchor ring/domain: mod_reduce / WHERE carry ring but not domain.
    return lowered.as_ring(ring, dom)


def lower_ring(analysis: RingLoweringAnalysis) -> RuleSet:
    """Expand polynomial constants and insert required modular reductions.

    Run with bottom_up=True so multiplication consumes scalar factors before
    their polynomial representation is expanded. Other uses of a shared
    constant keep their polynomial meaning.

    Inserts MOD_REDUCE between a parent and child when:
      - The child is SUB/NEG/MUL (always need reduced inputs)
      - The parent is SUB/NEG/MUL (inputs to these must be reduced)
      - A STORE's value operand (slot 1)
      - An ADD edge flagged by analyze_ring_lowering for overflow

    Skips nodes that are already reduced or inherently canonical (BUFFER, LOAD, CONST,
    conversion ops).

    Example:
        MUL(ADD(a, b), c)
      -> MUL(MOD_REDUCE(ADD(a, b)), MOD_REDUCE(c))
    """
    overflow_add_reduce_edges = analysis.overflow_add_reduce_edges

    reduced: dict[tuple[Node, RingSpec], Node] = {}

    def reduced_value(value: Node, ring: RingSpec) -> Node:
        key = (value.base, ring)
        if key not in reduced:
            reduced[key] = value.base.mod_reduce(ring)
        return reduced[key]

    def reduce_edge(parent: Node, child: Node, src_idx: int) -> Node:
        ring = child.ring
        if ring is None:
            return child
        child_base = child.base
        if (
            child_base.op in GroupOp.Defines
            or child_base.op in {Op.MOD_REDUCE, Op.CONST, Op.RANDOM, Op.BUFFER_VIEW, Op.LOAD}
            or (child_base.op in GroupOp.Conversion and not _is_view_over_compute(child_base))
        ):
            return child

        should_reduce = (
            (child_base.op in GroupOp.ReducedInputs and parent.op is not Op.MOD_REDUCE)
            or (parent.op in GroupOp.ReducedInputs)
            or (parent.op is Op.STORE and src_idx == 1)
            or ((parent.base, child_base) in overflow_add_reduce_edges)
        )

        if not should_reduce:
            return child

        if _is_view_over_compute(child_base):
            src = child_base.src[0]
            if src.base.op is Op.MOD_REDUCE:
                return child
            src_ring = src.ring
            assert src_ring is not None, f"view-over-compute source missing ring for {child_base.op}"
            reduced_src = reduced_value(src, src_ring)
            return child_base.replace(src=(reduced_src, *child_base.src[1:]))

        return reduced_value(child_base, ring)

    def lower_node(n: Node) -> Node | None:
        src = n.inputs
        if not src:
            return None
        new_src = tuple(reduce_edge(n, s, i) for i, s in enumerate(src))
        return n.replace(src=(n.src[0], *new_src) if n.op is Op.CALL else new_src)

    return rules([
        (
            pat.n(Op.MUL, [pat.x, pat.c(Op.AS_RING, pat(Op.CONST))]),
            lambda n, x, c: n.replace(src=(x, c.src[0].mod_reduce(c.ring))),
        ),
        (pat.n(Op.AS_RING, pat.c(Op.CONST)), _lower_const_poly),
        (pat.n(), lower_node),
    ])
