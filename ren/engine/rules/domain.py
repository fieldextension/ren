from ren.engine.pattern import pat, rules
from ren.ir import Op
from ren.ir.node import Node
from ren.ir.op import GroupOp
from ren.ring import Domain


# Domain propagation and NTT <-> COEFF simplification.
# Resolves unannotated STORE destinations from their values, eliminates
# redundant NTT/INTT pairs, pushes domain conversions through
# linear/transparent ops to expose more cancellation opportunities, and
# drops conversions when the input is already in the target domain.
#
# Examples:
#   NTT(INTT(x))             -> x              (round-trip elimination)
#   INTT(NTT(x))             -> x
#   NTT(NEG(x))              -> NEG(NTT(x))    (commute through linear NEG)
#   INTT(NTT(x) + y)         -> x + INTT(y)    (commute through ADD/SUB to cancel a conversion)
#   NTT(INTT(x) + y)         -> x + NTT(y)
#   NTT(x) where x is in NTT    -> x              (already in target domain)
def _commute_domain(n: Node, v: Node) -> Node | None:
    """n(v(x, ...)) -> v(n(x), ...) for matched linear ops."""
    if v.src[0].domain == n.domain:
        return None  # No mismatch to fix
    return v.replace(src=tuple(n.replace(src=(s,)) for s in v.src))


def _store_domain(target: Node, value: Node) -> Node | None:
    if target.domain is None and value.domain is not None:
        target = target.as_ring(target.arg[0], value.domain)
        return target.after(target.store(value))
    return None


simplify = rules([
    (pat(Op.AFTER, (pat.target(Op.AS_RING), pat(Op.STORE, (pat.target, pat.value)))), _store_domain),
    # round-trip elimination
    (pat(Op.NTT, pat(Op.INTT, pat.x)), lambda x: x),
    (pat(Op.INTT, pat(Op.NTT, pat.x)), lambda x: x),
    # domain ops commute through transparent ops (enables round-trip/redundant elimination)
    (pat.n({Op.NTT, Op.INTT}, (pat.v(Op.NEG),)), _commute_domain),
    (pat.n({Op.NTT, Op.INTT}, (pat.v(Op.MUL, [pat(), pat(Op.AS_RING, pat(Op.CONST))]),)), _commute_domain),
    # ADD/SUB commute only when an opposite transform cancels.
    (pat.n(Op.INTT, (pat.v({Op.ADD, Op.SUB}, [pat(Op.NTT), pat()]),)), _commute_domain),
    (pat.n(Op.NTT, (pat.v({Op.ADD, Op.SUB}, [pat(Op.INTT), pat()]),)), _commute_domain),
    # Domain requests are idempotent.
    (pat.n({Op.NTT, Op.INTT}, pat.x), lambda n, x: x if x.domain == n.domain else None),
])


def _input_domain(n: Node) -> Domain | None:
    """Domain selected for this op's inputs by the current placement rules."""
    if n.op is Op.STORE:
        return n.src[0].domain
    if n.op is Op.MUL:
        for i, source in enumerate(n.src):
            if source.op is Op.AS_RING and source.src[0].op is Op.CONST:
                return n.src[1 - i].domain
        return Domain.NTT
    if n.op is Op.ROTATE:
        return Domain.NTT
    if n.op in {Op.RESCALE, Op.FROM_RNS}:
        return Domain.COEFF
    if n.op is Op.TO_RING:
        src_ring = n.src[0].ring
        if src_ring is not None and not n.arg.ring.is_quotient_of(src_ring):
            return Domain.COEFF
    if n.op not in GroupOp.ALU:
        return None

    domains = (source.domain for source in n.src if source.domain is not None)
    first = next(domains, None)
    return Domain.NTT if any(domain != first for domain in domains) else None


def _to_domain(n: Node, domain: Domain) -> Node:
    return Node(Op.NTT if domain == Domain.NTT else Op.INTT, dtype=n.dtype, src=(n,))


def _inject_domain(n: Node) -> Node | None:
    """Insert NTT/INTT nodes where domain mismatches occur."""
    if (target := _input_domain(n)) is None:
        return None

    new_src = tuple(s if s.domain in (target, None) else _to_domain(s, target) for s in n.src)
    return n.replace(src=new_src)


# Insert NTT/INTT conversions where the selected implementation's domain
# doesn't match its operands. Polynomial MUL and ROTATE use NTT; constant MUL
# preserves the other operand's domain. RESCALE and ring extension use
# coefficients; mixed ALU inputs prefer NTT.
# Stores use the destination's domain.
#
# Example:
#   MUL(x:COEFF, y:COEFF)
#     -> MUL(NTT(x), NTT(y))          (selected implementation uses NTT)
#
#   RESCALE(x:NTT)
#     -> RESCALE(INTT(x))             (selected implementation uses coefficients)
inject = rules([(pat.n(), _inject_domain)])
