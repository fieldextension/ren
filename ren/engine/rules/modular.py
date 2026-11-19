"""Lower coefficient arithmetic to typed scalar operations and indexed constants."""

from ren.dtype import dtypes
from ren.engine.pattern import pat, rules
from ren.ir import Node, Op
from ren.montgomery import context
from ren.kernel import KernelSpec


def mod_add(a: Node, b: Node, p: Node) -> Node:
    """Add two residues modulo p. Their sum fits in one word when p < 2^(word_bits - 1)."""
    s = a + b
    return (s < p).where(s, s - p)


def mod_sub(a: Node, b: Node, p: Node) -> Node:
    """Return (a - b) mod p for 0 <= a, b < p."""
    return (a < b).where(a + p - b, a - b)


def mont_mul(a: Node, b: Node, p: Node, p_inv: Node) -> Node:
    """a * b / R mod p for a*b < p*R and p < R/2, where R = 2^word_bits."""
    word = a.dtype
    wide = dtypes.uint64 if word is dtypes.uint32 else dtypes.uint128
    product = a.cast(wide) * b.cast(wide)
    correction = product.cast(word) * p_inv
    total = product + correction.cast(wide) * p.cast(wide)
    r = (total >> (word.itemsize * 8)).cast(word)
    return (r < p).where(r, r - p)


def lower_modular(kernel: KernelSpec) -> Node:
    ring = kernel.ring
    lowered: dict[Node, Node] = {}
    idx = Node(Op.GIDX, dtypes.index) >> (ring.n.bit_length() - 1)

    def table(values: tuple[int, ...]) -> Node:
        if len(set(values)) == 1:
            return Node.const(values[0], ring.dtype)
        return Node(Op.CONST_ARRAY, ring.dtype, arg=values).index(idx).load()

    p = table(ring.moduli)
    contexts = tuple(context(ring.dtype, q) for q in ring.moduli)
    p_inv = table(tuple(c.p_inv for c in contexts))

    def reduced(x: Node) -> Node:
        return lowered[x] if x.base.op in {Op.LOAD, Op.MOD_REDUCE} else lowered[x] % p

    # Match the original operands so their reductions are still visible.
    expand = rules([
        (pat.c.const(), lambda c: table(tuple(c.arg * ctx.radix % ctx.modulus for ctx in contexts))),
        (pat.x({Op.AFTER, Op.LOAD, Op.MOD_REDUCE}), reduced),
        (pat.a + pat.b, lambda a, b: mod_add(reduced(a), reduced(b), p)),
        (pat.a - pat.b, lambda a, b: mod_sub(reduced(a), reduced(b), p)),
        (pat.a * pat.b, lambda a, b: mont_mul(reduced(a), reduced(b), p, p_inv)),
        (-pat.a, lambda a: mod_sub(Node.const(0, ring.dtype), reduced(a), p)),
    ])
    for n in kernel.program:
        if n.op is Op.MOD_REDUCE:
            (x,) = n.src
            action = expand.dispatch(x)
            lowered[n] = action() if action is not None else (lowered[x] % p).cast(ring.dtype)
        else:
            lowered[n] = n.replace(src=tuple(lowered[s] for s in n.src)) if n.src else n
    return lowered[kernel.ast]
