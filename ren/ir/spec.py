"""Structural rules for compute kernels and type rules for scalar expressions."""

from collections.abc import Iterable

from ren.dtype import dtypes
from ren.engine.pattern import Pat, RuleSet, pat, rules
from ren.ir.node import Node, ParamArg, RandomArg, ToRingArg, ToRnsArg
from ren.ir.op import GroupOp, Op
from ren.kernel import KernelInfo
from ren.ring import RingSpec


def validate(nodes: Iterable[Node], ruleset: RuleSet) -> None:
    for node in nodes:
        if ruleset.match(node) is not True:
            raise ValueError(f"Invalid IR node: {node.op} ({node.dtype.name})")


def _stores(value: Pat) -> RuleSet:
    return rules([
        (pat.store(Op.STORE, (target, value)), lambda store, target: store.dtype is target.dtype is store.src[1].dtype)
        for target in (pat.target(Op.PARAM), pat.target(Op.INDEX, (pat(Op.PARAM), pat(dtype=dtypes.index))))
    ])


_storage = rules([
    (pat.n(Op.SINK, pat(Op.STORE)), lambda n: isinstance(n.arg, KernelInfo) and n.arg.ring is not None and bool(n.src)),
    (pat.n(Op.PARAM, (), dtype=dtypes.uints), lambda n: isinstance(n.arg, ParamArg)),
    (pat.n(Op.DEVICE, (), dtype=dtypes.void), lambda n: isinstance(n.arg, str)),
    (pat.n(Op.GIDX, (), dtype=dtypes.index), lambda n: n.arg in (None, 0)),
    (pat.n(Op.INDEX, (pat.buf({Op.PARAM, Op.CONST_ARRAY}), pat(dtype=dtypes.index))), lambda n, buf: n.dtype is buf.dtype),
])

_scalar = pat(GroupOp.Elementwise | {Op.GIDX, Op.LOAD, Op.CONST, Op.AFTER})
scalar = (
    _storage
    + _stores(_scalar)
    + rules([
        (
            pat.n(Op.CONST_ARRAY, (), dtype=(dtypes.uint32, dtypes.uint64)),
            lambda n: isinstance(n.arg, tuple) and bool(n.arg) and all(isinstance(v, int) and 0 <= v < 1 << (n.dtype.itemsize * 8) for v in n.arg),
        ),
        (pat.n(Op.CONST, (), dtype=dtypes.weakint), lambda n: isinstance(n.arg, int)),
        (pat(Op.CAST, (_scalar,), dtype=(*dtypes.uints, dtypes.uint128, dtypes.index, dtypes.bool)), lambda: True),
        (pat.n(Op.LOAD, (pat.ptr(Op.INDEX),)), lambda n, ptr: n.dtype is ptr.dtype),
        (pat.n(Op.AFTER, _scalar), lambda n: len(n.src) >= 2 and n.dtype is n.src[0].dtype),
        (
            pat.n(Op.BITREVERSE, (_scalar,), dtype=dtypes.index),
            lambda n: n.src[0].dtype is dtypes.index and isinstance(n.arg, int) and 0 < n.arg <= 64,
        ),
        (pat.n(Op.NEG, (_scalar,), dtype=(*dtypes.uints, dtypes.uint128, dtypes.index)), lambda n: n.dtype is n.src[0].dtype),
        (pat.n({Op.CMPEQ, Op.CMPLT}, (_scalar, _scalar), dtype=dtypes.bool), lambda n: n.src[0].dtype is n.src[1].dtype),
        (pat.n(Op.WHERE, (_scalar, _scalar, _scalar)), lambda n: n.src[0].dtype is dtypes.bool and n.dtype is n.src[1].dtype is n.src[2].dtype),
        (pat.n(GroupOp.Binary - {Op.CMPEQ, Op.CMPLT}, (_scalar, _scalar)), lambda n: n.dtype is n.src[0].dtype is n.src[1].dtype),
    ])
)

specialized_ops = frozenset({Op.NTT, Op.INTT, Op.RESCALE, Op.TO_RING, Op.TO_RNS, Op.FROM_RNS, Op.RANDOM})
specialized = (
    _storage
    + _stores(pat(specialized_ops))
    + rules([
        (pat.n({Op.NTT, Op.INTT}, (pat.x(Op.PARAM),)), lambda n, x: n.arg is None and n.dtype is x.dtype),
        (pat.n(Op.RESCALE, (pat.x(Op.PARAM),)), lambda n, x: isinstance(n.arg, RingSpec) and n.dtype is x.dtype is n.arg.dtype),
        (pat.n(Op.TO_RING, (pat.x(Op.PARAM),)), lambda n, x: isinstance(n.arg, ToRingArg) and n.dtype is x.dtype is n.arg.ring.dtype),
        (
            pat.n(Op.TO_RNS, (pat(Op.PARAM, dtype=dtypes.uint32),)),
            lambda n: isinstance(n.arg, ToRnsArg) and n.arg.limb_count > 0 and n.dtype is n.arg.ring.dtype,
        ),
        (pat.n(Op.FROM_RNS, (pat(Op.PARAM),), dtype=dtypes.uint32), lambda n: n.arg is None),
        (pat.n(Op.RANDOM, (pat(Op.PARAM, dtype=dtypes.uint32),), dtype=dtypes.uints), lambda n: isinstance(n.arg, RandomArg)),
    ])
)

modular = scalar + rules([
    (
        pat.n(Op.MOD_REDUCE, (_scalar,), dtype=dtypes.uints),
        lambda n: n.arg is None and n.src[0].dtype in (n.dtype, dtypes.weakint),
    ),
])
