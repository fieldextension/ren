"""Lower uint128 scalar expressions to pairs of uint64 words.

For targets that require decomposition, each uint128 node becomes a (lo, hi)
pair. The 64x64 high product uses 32-bit halves (TAOCP 4.3.1, tinygrad's l2i).
Shared scalar simplification removes zero cross terms after decomposition.
"""

from ren.dtype import dtypes
from ren.engine.pattern import RuleSet, pat, rules
from ren.ir import Op
from ren.ir.node import Node

_WORD = dtypes.uint64
_BITS = 64
_MASK = (1 << _BITS) - 1
_HALF_MASK = (1 << (_BITS // 2)) - 1

type Pair = tuple[Node, Node]


def _mulhi(a: Node, b: Node) -> Node:
    """High word of the full 128-bit product of two uint64 words."""
    half = _BITS // 2
    a_lo, a_hi = a & _HALF_MASK, a >> half
    b_lo, b_hi = b & _HALF_MASK, b >> half
    lo_lo, hi_lo, lo_hi, hi_hi = a_lo * b_lo, a_hi * b_lo, a_lo * b_hi, a_hi * b_hi
    # The bit-32 column: carry out of lo_lo plus the low halves of both cross terms.
    cross = (lo_lo >> half) + (hi_lo & _HALF_MASK) + (lo_hi & _HALF_MASK)
    return hi_hi + (hi_lo >> half) + (lo_hi >> half) + (cross >> half)


def _shift_pair(op: Op, x: Pair, bits: int) -> Pair:
    assert 0 <= bits < 2 * _BITS, f"uint128 shift out of range: {bits}"
    if bits == 0:
        return x
    lo, hi = x
    zero = Node.const(0, _WORD)
    if op is Op.SHL:
        if bits >= _BITS:
            return zero, lo << (bits - _BITS)
        # The two terms occupy disjoint bits, so ADD is the bitwise union.
        return lo << bits, (hi << bits) + (lo >> (_BITS - bits))
    if bits >= _BITS:
        return hi >> (bits - _BITS), zero
    return (lo >> bits) + (hi << (_BITS - bits)), hi >> bits


def _wide(n: Node, pairs: dict[Node, Pair]) -> Pair:
    """(lo, hi) words of a uint128 node, given its rewritten sources."""
    match n.op:
        case Op.CAST if n.src[0].op is Op.CONST:
            value = n.src[0].arg
            return Node.const(value & _MASK, _WORD), Node.const((value >> _BITS) & _MASK, _WORD)
        case Op.CAST if n.src[0].dtype in (dtypes.bool, dtypes.uint32, dtypes.uint64):
            return n.src[0].cast(_WORD), Node.const(0, _WORD)
        case Op.CAST if n.src[0].dtype is dtypes.index:
            x = n.src[0]
            return x.cast(_WORD), (x < 0).where(Node.const(_MASK, _WORD), 0)
        case Op.CAST if n.src[0].dtype is dtypes.uint128:
            return pairs[n.src[0]]
        case Op.SHL | Op.SHR:
            if (bits := n.src[1].const_val) is None:
                raise NotImplementedError(f"uint128 {n.op} requires a constant shift amount")
            return _shift_pair(n.op, pairs[n.src[0]], bits)
        case Op.WHERE:
            (yes_lo, yes_hi), (no_lo, no_hi) = pairs[n.src[1]], pairs[n.src[2]]
            return n.src[0].where(yes_lo, no_lo), n.src[0].where(yes_hi, no_hi)
        case Op.NEG:
            lo, hi = pairs[n.src[0]]
            return -lo, -hi - (lo > 0).cast(_WORD)
        case Op.ADD:
            (a_lo, a_hi), (b_lo, b_hi) = pairs[n.src[0]], pairs[n.src[1]]
            lo = a_lo + b_lo
            return lo, a_hi + b_hi + (lo < a_lo).cast(_WORD)
        case Op.SUB:
            (a_lo, a_hi), (b_lo, b_hi) = pairs[n.src[0]], pairs[n.src[1]]
            return a_lo - b_lo, a_hi - b_hi - (a_lo < b_lo).cast(_WORD)
        case Op.MUL:
            (a_lo, a_hi), (b_lo, b_hi) = pairs[n.src[0]], pairs[n.src[1]]
            return a_lo * b_lo, _mulhi(a_lo, b_lo) + a_lo * b_hi + a_hi * b_lo
    raise NotImplementedError(f"uint128 {n.op} is not supported")


def lower_dtypes() -> RuleSet:
    """Rules for decomposing uint128 arithmetic into uint64 words."""
    pairs: dict[Node, Pair] = {}

    return rules([
        (pat.n(dtype=dtypes.uint128), lambda n: pairs.update({n: _wide(n, pairs)})),
        (
            pat.x(dtype=dtypes.uint128).cast(dtypes.bool),
            lambda x: pairs[x][1].eq(0).where(pairs[x][0] > 0, 1),
        ),
        (
            pat.x(dtype=dtypes.uint128).cast((dtypes.index, dtypes.uint32, dtypes.uint64), name="n"),
            lambda n, x: pairs[x][0].cast(n.dtype),
        ),
        (
            pat.a(dtype=dtypes.uint128).eq(pat.b(dtype=dtypes.uint128)),
            lambda a, b: pairs[a][1].eq(pairs[b][1]).where(pairs[a][0].eq(pairs[b][0]), 0),
        ),
        (
            pat.a(dtype=dtypes.uint128) < pat.b(dtype=dtypes.uint128),
            lambda a, b: pairs[a][1].eq(pairs[b][1]).where(pairs[a][0] < pairs[b][0], pairs[a][1] < pairs[b][1]),
        ),
    ])


__all__ = ["lower_dtypes"]
