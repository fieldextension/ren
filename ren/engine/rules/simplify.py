from ren.dtype import dtypes
from ren.engine.pattern import pat, rules
from ren.ir import Op
from ren.ir.op import GroupOp
from ren.ir.node import Node, from_rns_limb_count, python_alu
from ren.ring import Domain


def _fold_const(n: Node) -> Node | None:
    return n.const_like(value) if (value := n.const_val) is not None else None


const_loads = rules([
    (pat.n(Op.LOAD, pat(Op.CONST_ARRAY).index(pat().const())), _fold_const),
])


# Fold each operation at its own width before consuming its result.
fold = const_loads + rules([
    (pat.n((*python_alu, Op.CAST), pat(Op.CONST).cast()), _fold_const),
    # neutral elements
    (pat.x + 0, lambda x: x),
    (pat.x - 0, lambda x: x),
    (pat.x * 1, lambda x: x),
    (pat.x * 0, lambda x: 0),
    (pat({Op.SHL, Op.SHR}, (pat.x, pat().const(0))), lambda x: x),
    (pat.x < pat.x, lambda x: Node.const(0, dtypes.bool)),
    (pat.x(dtype=dtypes.uints) < 0, lambda x: Node.const(0, dtypes.bool)),
    (pat.x.eq(pat.x), lambda x: Node.const(1, dtypes.bool)),
    (pat.c.const().where(pat.yes, pat.no), lambda c, yes, no: yes if c.const_val else no),
    (pat().where(pat.x, pat.x), lambda x: x),
])


def _should_flip(x: Node, y: Node) -> bool:
    """Determine if operands should be flipped for canonical ordering.

    Constants always go right, then use tuplize for deterministic ordering.

    NOTE: This could be simplified by relying on Op enum ordering,
    where CONST is placed late in the enum so it naturally sorts right via tuplize.
    For now we're explicit to avoid implicit coupling to enum order.
    """
    x_const = x.const_val is not None
    y_const = y.const_val is not None
    if x_const != y_const:
        return x_const  # Flip if x is const (move it right)
    return x.tuplize > y.tuplize


# Algebraic simplification rules.
#
# Canonicalizes and simplifies arithmetic over the DAG:
#   - Neutral element elimination:  x + 0 -> x,  x * 1 -> x,  x * 0 -> 0
#   - Canonical operand ordering:   CONST(3) + x  ->  x + CONST(3)
#   - Negation normalization:       (-x) + y -> y - x,  -(x - y) -> y - x
#   - Self-cancellation:            x - x -> 0,  -(-x) -> x
#   - Term combination:             x + x -> x*2,  x*c1 + x*c2 -> x*(c1+c2)
#   - Constant reassociation:       (x + 3) + 5 -> x + 8,  (x * 3) * 5 -> x * 15
#   - Constant bubbling:            (x + 3) + y -> (x + y) + 3
#
# Example:
#       ADD                ADD
#      /   \     ->        /   \
#   MUL(x,2) x       MUL(x,3)  (term combination via x*2 + x -> x*3)
algebraic = fold + rules([
    # fold arithmetic constants before rules that create or reassociate them
    (pat.n({Op.ADD, Op.SUB, Op.MUL, Op.IDIV, Op.NEG, Op.MOD_REDUCE}, pat().const()), _fold_const),
    # canonical ordering: constants go right, then smaller tuplize goes left
    (pat.n(GroupOp.Commutative, (pat.x, pat.y)), lambda n, x, y: n.replace(src=(y, x)) if _should_flip(x, y) else None),
    # canonicalize negation/sub
    (-pat.x + pat.y, lambda x, y: y - x),  # Also handles pat.x + -pat.y due to commutativity
    (0 - pat.y, lambda y: -y),
    (pat.x - pat.x, lambda x: 0),
    (pat.x * -1, lambda x: -x),
    (-(-pat.x), lambda x: x),
    (-(pat.x - pat.y), lambda x, y: y - x),
    (pat.x - (-pat.y), lambda x, y: x + y),
    (-pat.x - pat.y, lambda x, y: -(x + y)),
    ((-pat.x) * -1, lambda x: x),
    # (x * -1) * -1 is handled by (x * c1) * c2 -> x * (c1*c2), then x * 1 -> x
    ((-pat.x) * (-pat.y), lambda x, y: x * y),
    (-(pat.x * pat.c.const()), lambda x, c: x * (-c)),
    (pat.x + pat.x, lambda x: x * 2),
    # Term combination for chains: (y + x) + x -> y + x*2
    ((pat.y + pat.x) + pat.x, lambda y, x: y + x * 2),
    # (y + x*c0) + x*c1 -> y + x*(c0+c1)
    ((pat.y + pat.x * pat.c0.const()) + pat.x * pat.c1.const(), lambda y, x, c0, c1: y + x * (c0 + c1)),
    # (y + x) + x*c -> y + x*(c+1)
    ((pat.y + pat.x) + pat.x * pat.c.const(), lambda y, x, c: y + x * (c + 1)),
    # (y + x*c) + x -> y + x*(c+1)
    ((pat.y + pat.x * pat.c.const()) + pat.x, lambda y, x, c: y + x * (c + 1)),
    # factor out common terms
    (pat.x * pat.c1.const() + pat.x * pat.c2.const(), lambda x, c1, c2: x * (c1 + c2)),
    (pat.x * pat.c1.const() - pat.x * pat.c2.const(), lambda x, c1, c2: x * (c1 - c2)),
    (pat.x * pat.c1.const() + pat.x, lambda x, c1: x * (c1 + 1)),
    (pat.x - pat.x * pat.c1.const(), lambda x, c1: x * (1 - c1)),
    # pat.x + pat.x * pat.c1.const() is handled by pat.x * pat.c1.const() + pat.x due to commutativity
    # reassociate to fold constants
    ((pat.x + pat.c1.const()) + pat.c2.const(), lambda x, c1, c2: x + (c1 + c2)),
    ((pat.x - pat.c1.const()) - pat.c2.const(), lambda x, c1, c2: x - (c1 + c2)),
    ((pat.x - pat.c1.const()) + pat.c2.const(), lambda x, c1, c2: x + (c2 - c1)),
    ((pat.x + pat.c1.const()) - pat.c2.const(), lambda x, c1, c2: x + (c1 - c2)),
    ((pat.x * pat.c1.const()) * pat.c2.const(), lambda x, c1, c2: x * (c1 * c2)),
    # bubble constants outward
    ((pat.x + pat.c1.const()) + pat.y, lambda x, c1, y: (x + y) + c1),
    ((pat.x - pat.c1.const()) + pat.y, lambda x, c1, y: (x + y) - c1),
    ((pat.x + pat.c1.const()) - pat.y, lambda x, c1, y: (x - y) + c1),
    ((pat.x - pat.c1.const()) - pat.y, lambda x, c1, y: (x - y) - c1),
    ((pat.x * pat.c1.const()) * pat.y, lambda x, c1, y: (x * y) * c1),
])

# Polynomial consumers use the value after an explicit scalar cast.
const_casts = rules([
    (pat.n({Op.AS_RING, Op.MOD_REDUCE}, pat().const().cast(name="c")), lambda n, c: n.replace(src=(Node.const(c.const_val),))),
])


# Constant folding through ring/domain operations.
# When an op's input is a known constant, the output is also a constant
# (possibly with a different value and metadata). CONTIGUOUS of a constant
# is a no-op, and ROTATE of zero is always zero regardless of rotation amount.
#
# Examples:
#   NTT(CONST(c))          -> CONST(c)   (with NTT metadata)
#   RESCALE(CONST(4488))   -> CONST(1)   (Q=17*41*73 -> Q=17)
#   TO_RING(CONST(16))     -> CONST(-1)  (Q=17 -> Q=17*41, balanced)
#   CONTIGUOUS(CONST(c))   -> CONST(c)
#   ROTATE(CONST(0), k)    -> CONST(0)
const = const_casts + rules([
    (pat.n({Op.NTT, Op.INTT, Op.AUTOMORPHISM, Op.TO_RING, Op.RESCALE}, pat().const()), _fold_const),
    (pat(Op.CONTIGUOUS, pat.c.const()), lambda c: c),
    # ROTATE(0, k) -> 0
    (pat.n(Op.ROTATE, pat().const(0)), lambda n: n.const_like(0)),
])

# COPY (device transfer) simplification.
# Materializes device-free values, collapses nested copies into a single hop,
# and eliminates same-device copies.
#
# Examples:
#   COPY(CONST(5), dev:cuda) -> ALLOC + STORE on cuda
#   COPY(COPY(x, dev:A), dev:B) -> COPY(x, dev:B)  (skip intermediate device)
#   COPY(x, dev:D) -> x                             (when x is already on D)
copy = rules([
    (pat(Op.COPY, (pat.x, pat.d)), lambda x, d: x.clone(d.arg) if x._device is None else None),
    (pat(Op.COPY, (pat(Op.COPY, (pat.x, pat())), pat.d2)), lambda x, d2: x.copy_to_device(d2.arg)),
    (pat(Op.COPY, (pat.x, pat.d)), lambda x, d: x if x._device == d.arg else None),
])


# Ring operation simplification.
# Eliminates identity ring conversions and fuses consecutive
# ROTATE/AUTOMORPHISM into a single application.
#
# Examples:
#   TO_RING(TO_RING(x, R1->R2), R2->R3) -> TO_RING(x, R1->R3)  (view-only chain)
#   TO_RING(x, R->R)                    -> x                    (identity)
#   AS_RING(AS_RING(x))                -> AS_RING(x)           (collapse wrapper)
#   RESCALE(x, target=x.ring)          -> x                    (identity rescale)
#   ROTATE(x, 0)                       -> x
#   ROTATE(ROTATE(x, k1), k2)          -> ROTATE(x, k1+k2)
#   AUTOMORPHISM(x, 1)                 -> x                    (sigma_1 = identity)
#   AUTOMORPHISM(AUTOMORPHISM(x, k1), k2) -> AUTOMORPHISM(x, k1*k2 mod 2n)
#   TO_RNS(FROM_RNS(x)) -> x for coefficient-domain residues in the same ring.
def _collapse_to_ring_chain(outer: Node, inner: Node, x: Node) -> Node | None:
    """Collapse TO_RING(TO_RING(x)) when both hops are view-only."""
    if inner.view_offset is None or outer.view_offset is None:
        return None
    return outer.replace(src=(x,))


def _special_rotations(n: Node, x: Node) -> Node | None:
    assert n.ring is not None
    k = n.arg % (2 * n.ring.n)
    if k == 0:
        return x
    elif k == n.ring.n:
        return -x
    else:
        return None


ring = rules([
    (
        pat.n(Op.TO_RNS, pat(Op.FROM_RNS, pat.x)),
        lambda n, x: x if x.domain is Domain.COEFF and x.ring == n.arg.ring and n.arg.limb_count == from_rns_limb_count(n.arg.ring) else None,
    ),
    (pat.outer(Op.TO_RING, pat.inner(Op.TO_RING, pat.x)), _collapse_to_ring_chain),
    (pat.n(Op.TO_RING, pat.x), lambda n, x: x if n.ring == x.ring else None),
    (pat.n(Op.AS_RING, pat(Op.AS_RING, pat.x)), lambda n, x: n.replace(src=(x,))),
    (pat.n(Op.AS_RING, pat.after(Op.AFTER)), lambda n, after: after.src[0].as_ring(*n.arg).after(*after.src[1:])),
    (pat.n(Op.RESCALE, pat.x), lambda n, x: x if n.arg == x.ring else None),  # identity
    (pat.n(Op.ROTATE, pat.x), _special_rotations),
    (pat(Op.AUTOMORPHISM, pat.x, arg=1), lambda x: x),  # identity automorphism
    # ROTATE(ROTATE(x, k1), k2) -> ROTATE(x, k1+k2)
    (pat.outer(Op.ROTATE, pat.inner(Op.ROTATE, pat())), lambda outer, inner: inner.replace(arg=inner.arg + outer.arg)),
    # AUTOMORPHISM(AUTOMORPHISM(x, k1), k2) -> AUTOMORPHISM(x, k1*k2 mod 2n)
    (
        pat.outer(Op.AUTOMORPHISM, pat.inner(Op.AUTOMORPHISM, pat())),
        lambda outer, inner: inner.replace(arg=(inner.arg * outer.arg) % (2 * inner.ring.n)),
    ),
])
