import random
import operator

import numpy as np
import pytest
from typing import assert_type

from ren import tree
from ren.dtype import BoolFmt, DType, FloatFmt, FmtStr, IntFmt, dtypes
from ren.device import Device
from ren.ir import Node
from ren.poly import Poly
from tests.utils import eval_scalar


@pytest.mark.parametrize("fmt", ["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "e", "f", "d", None])
def test_format_predicates_narrow_both_branches(fmt: FmtStr | None):
    if DType.is_int_fmt(fmt):
        assert_type(fmt, IntFmt)
        assert fmt in "bBhHiIlLqQ"
    else:
        assert_type(fmt, BoolFmt | FloatFmt | None)
        if DType.is_float_fmt(fmt):
            assert_type(fmt, FloatFmt)
            assert fmt in "efd"
        else:
            assert_type(fmt, BoolFmt | None)
            assert fmt in ("?", None)


def test_from_py_infers_homogeneous_sequence_dtype():
    assert dtypes.from_py([1, 2, 3]) is dtypes.default_int
    assert dtypes.from_py((1.0, 2.0, 3.0)) is dtypes.default_float


def test_from_py_empty_sequence_defaults_to_float():
    assert dtypes.from_py([]) is dtypes.default_float
    assert dtypes.from_py(()) is dtypes.default_float


def test_from_py_rejects_mixed_sequence_dtype():
    with pytest.raises(RuntimeError, match="Different dtypes"):
        dtypes.from_py([1, 2.0])


_U64_MASK = (1 << 64) - 1
_U128_MASK = (1 << 128) - 1
_WORD_EDGES = [0, 1, 2, (1 << 32) - 1, 1 << 32, (1 << 32) + 1, (1 << 63) - 1, 1 << 63, _U64_MASK - 1, _U64_MASK]


def test_python_executes_deep_scalar_expression():
    def build(x):
        for _ in range(512):
            x = x * 13 + 7
        return x

    inputs = list(range(128))
    expected = inputs
    for _ in range(512):
        expected = [(x * 13 + 7) & _U64_MASK for x in expected]
    with Device.scope("python"):
        assert eval_scalar(build, [inputs], dtypes.uint64) == expected


def _word_columns(dtype, count: int = 128) -> tuple[list[int], list[int]]:
    """Every pair of edge words, then random words, padded to a power-of-two count."""
    bits = dtype.itemsize * 8
    edges = [e for e in _WORD_EDGES if e < 1 << bits] + [(1 << (bits - 1)) - 1, 1 << (bits - 1), (1 << bits) - 1]
    edges = list(dict.fromkeys(edges))
    pairs = [(a, b) for a in edges for b in edges]
    rng = random.Random(bits)
    pairs += [(rng.getrandbits(bits), rng.getrandbits(bits)) for _ in range(count - len(pairs))]
    return [a for a, _ in pairs], [b for _, b in pairs]


def _wide(x: Node) -> Node:
    return x.cast(dtypes.uint128)


def _join(hi: Node, lo: Node) -> Node:
    return (_wide(hi) << 64) + _wide(lo)


# uint128 expressions of two uint64 words a, b, with their exact values mod 2^128.
_WIDE_CASES = {
    "product": (lambda a, b: _wide(a) * _wide(b), lambda a, b: a * b),
    "sum_carry": (lambda a, b: _wide(a) + _wide(b), lambda a, b: a + b),
    "difference_borrow": (lambda a, b: _wide(a) - _wide(b), lambda a, b: a - b),
    "negate": (lambda a, b: -_join(a, b), lambda a, b: -(a << 64 | b)),
    "full_product": (lambda a, b: _join(a, b) * _join(b, a), lambda a, b: (a << 64 | b) * (b << 64 | a)),
    "full_sum": (lambda a, b: _join(a, b) + _join(b, a), lambda a, b: (a << 64 | b) + (b << 64 | a)),
    "full_difference": (lambda a, b: _join(a, b) - _join(b, a), lambda a, b: (a << 64 | b) - (b << 64 | a)),
    "narrow_then_widen": (lambda a, b: _wide(_join(a, b).cast(dtypes.uint64) + 1) * _wide(a), lambda a, b: ((b + 1) & _U64_MASK) * a),
    "where": (
        lambda a, b: (a < b).where(_join(a, b), _wide(a) * _wide(b)),
        lambda a, b: (a << 64 | b) if a < b else a * b,
    ),
    **{f"shl_{k}": (lambda a, b, k=k: _join(a, b) << k, lambda a, b, k=k: (a << 64 | b) << k) for k in (0, 1, 31, 63, 64, 65, 127)},
    **{f"shr_{k}": (lambda a, b, k=k: _join(a, b) >> k, lambda a, b, k=k: (a << 64 | b) >> k) for k in (0, 1, 31, 63, 64, 65, 127)},
}


@pytest.mark.parametrize("case", list(_WIDE_CASES))
def test_uint128_lowering_matches_python_integers(case):
    build, exact = _WIDE_CASES[case]
    a, b = _word_columns(dtypes.uint64)
    expected = [exact(x, y) & _U128_MASK for x, y in zip(a, b)]
    lo = eval_scalar(lambda x, y: build(x, y).cast(dtypes.uint64), [a, b], dtypes.uint64)
    hi = eval_scalar(lambda x, y: (build(x, y) >> 64).cast(dtypes.uint64), [a, b], dtypes.uint64)
    narrow = eval_scalar(lambda x, y: build(x, y).cast(dtypes.uint32), [a, b], dtypes.uint64)
    boolean = eval_scalar(lambda x, y: build(x, y).cast(dtypes.bool).cast(dtypes.uint64), [a, b], dtypes.uint64)
    assert lo == [e & _U64_MASK for e in expected]
    assert hi == [e >> 64 for e in expected]
    assert narrow == [e & 0xFFFFFFFF for e in expected]
    assert boolean == [int(bool(e)) for e in expected]


@pytest.mark.parametrize("value", [0, 1, (1 << 64) - 1, 1 << 64, (1 << 100) + 3, -1, (1 << 132) + 5])
def test_uint128_constants_execute(value):
    a, _ = _word_columns(dtypes.uint64)
    factor = (1 << 63) + 1

    def build(x):
        return (_wide(x) + Node.const(value, dtypes.uint128)) * Node.const(factor, dtypes.uint128)

    expected = [((x + value) * factor) & _U128_MASK for x in a]
    assert eval_scalar(lambda x: build(x).cast(dtypes.uint64), [a], dtypes.uint64) == [e & _U64_MASK for e in expected]
    assert eval_scalar(lambda x: (build(x) >> 64).cast(dtypes.uint64), [a], dtypes.uint64) == [e >> 64 for e in expected]
    # Both operands need their width even when their literals fit in uint64.
    product = Node.const(value, dtypes.uint128) * Node.const(factor, dtypes.uint128)
    assert eval_scalar(lambda x: (product >> 64).cast(dtypes.uint64), [a], dtypes.uint64) == [((value * factor) & _U128_MASK) >> 64] * len(a)


# IR comparison built from Node operators/methods, and the matching Python predicate.
_COMPARISONS = {
    "lt": (lambda x, y: x < y, lambda x, y: x < y),
    "le": (lambda x, y: x <= y, lambda x, y: x <= y),
    "gt": (lambda x, y: x > y, lambda x, y: x > y),
    "ge": (lambda x, y: x >= y, lambda x, y: x >= y),
    "eq": (lambda x, y: x.eq(y), lambda x, y: x == y),
    "ne": (lambda x, y: x.ne(y), lambda x, y: x != y),
}


@pytest.mark.parametrize("name", list(_COMPARISONS))
def test_uint128_comparisons_match_python_integers(name):
    a, b = _word_columns(dtypes.uint64)
    pairs = [((x << 64 | y), (x << 64 | z)) for x, y, z in zip(a, b, reversed(b))] + [((x << 64 | y), (y << 64 | y)) for x, y in zip(a, b)]
    lhs, rhs = [p[0] for p in pairs], [p[1] for p in pairs]
    words = [[v >> 64 for v in lhs], [v & _U64_MASK for v in lhs], [v >> 64 for v in rhs], [v & _U64_MASK for v in rhs]]
    build, exact = _COMPARISONS[name]

    def kernel(ah, al, bh, bl):
        return build(_join(ah, al), _join(bh, bl)).cast(dtypes.uint64)

    assert eval_scalar(kernel, words, dtypes.uint64) == [int(exact(x, y)) for x, y in pairs]


def test_uint128_cast_to_bool_checks_both_words():
    values = [0, 1, 256, _U64_MASK, 1 << 64, (1 << 64) + 1, 1 << 127, _U128_MASK]
    words = [[value >> 64 for value in values], [value & _U64_MASK for value in values]]
    assert eval_scalar(lambda hi, lo: _join(hi, lo).cast(dtypes.bool).cast(dtypes.uint64), words, dtypes.uint64) == [
        int(bool(value)) for value in values
    ]


@pytest.mark.parametrize("value", [0, 256, 1 << 64, _U128_MASK])
def test_uint128_constant_cast_to_bool(value):
    result = Node.const(value, dtypes.uint128).cast(dtypes.bool).cast(dtypes.uint64)
    assert eval_scalar(lambda x: x + result, [[0] * 8], dtypes.uint64) == [int(bool(value))] * 8


@pytest.mark.parametrize("value", [0, 1, _U64_MASK])
def test_uint128_consumes_wrapped_word_constants(value):
    a, _ = _word_columns(dtypes.uint64)
    word = Node.const(value, dtypes.uint64) + 1
    expected = [((value + 1) & _U64_MASK) * x for x in a]
    assert eval_scalar(lambda x: (_wide(word) * _wide(x) >> 64).cast(dtypes.uint64), [a], dtypes.uint64) == [e >> 64 for e in expected]


def test_index_widening_preserves_negative_values():
    inputs = [0, 1, 2, 3]
    expected = [(x - 2) & _U128_MASK for x in inputs]

    def build(x):
        return (x.cast(dtypes.index) - 2).cast(dtypes.uint128)

    assert eval_scalar(lambda x: build(x).cast(dtypes.uint64), [inputs], dtypes.uint64) == [v & _U64_MASK for v in expected]
    assert eval_scalar(lambda x: (build(x) >> 64).cast(dtypes.uint64), [inputs], dtypes.uint64) == [v >> 64 for v in expected]
    constant = Node.const(-1, dtypes.index).cast(dtypes.uint128)
    assert eval_scalar(lambda x: (constant >> 64).cast(dtypes.uint64), [inputs], dtypes.uint64) == [_U64_MASK] * len(inputs)


def test_scalar_ops_accept_numpy_integer_operands():
    assert eval_scalar(lambda x: (x * np.int64(3)) >> np.int64(1), [[0, 1, 2, 3]], dtypes.uint64) == [0, 1, 3, 4]


def test_uint128_consumes_montgomery_product():
    a = list(range(8))

    def build(x):
        product = (Node.const(1, dtypes.uint64).mod_reduce(None) * x).mod_reduce(None)
        return (_wide(product) + _wide(x)).cast(dtypes.uint64)

    assert eval_scalar(build, [a], dtypes.uint64) == [2 * x for x in a]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("name", list(_COMPARISONS))
def test_word_comparisons_execute(dtype, name):
    build, exact = _COMPARISONS[name]
    a, b = _word_columns(dtype)
    limit = b[3]
    assert eval_scalar(lambda x, y: build(x, y).cast(dtype), [a, b], dtype) == [int(exact(x, y)) for x, y in zip(a, b)]
    # Integer literals on either side become constants of the node's dtype.
    assert eval_scalar(lambda x: build(x, limit).cast(dtype), [a], dtype) == [int(exact(x, limit)) for x in a]
    if name not in {"eq", "ne"}:
        assert eval_scalar(lambda x: build(limit, x).cast(dtype), [a], dtype) == [int(exact(limit, x)) for x in a]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_where_selects_and_shifts_execute(dtype):
    a, b = _word_columns(dtype)
    mask = (1 << (dtype.itemsize * 8)) - 1
    assert eval_scalar(lambda x, y: (x < y).where(x << 3, y >> 5), [a, b], dtype) == [(x << 3) & mask if x < y else y >> 5 for x, y in zip(a, b)]
    assert eval_scalar(lambda x, y: x.eq(y).where(7, y), [a, b], dtype) == [7 if x == y else y for x, y in zip(a, b)]
    assert eval_scalar(lambda x, y: (x > y).where(x, 0), [a, b], dtype) == [x if x > y else 0 for x, y in zip(a, b)]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize(
    "case",
    [
        ("add", lambda a, b: a + b, lambda a, b: a + b),
        ("sub", lambda a, b: a - b, lambda a, b: a - b),
        ("mul", lambda a, b: a * b, lambda a, b: a * b),
        ("shl", lambda a, b: a << 7, lambda a, b: a << 7),
        ("shr", lambda a, b: a >> 7, lambda a, b: a >> 7),
    ],
    ids=lambda case: case[0],
)
def test_word_ops_wrap_at_dtype_width(dtype, case):
    _, build, exact = case
    a, b = _word_columns(dtype)
    mask = (1 << (dtype.itemsize * 8)) - 1
    assert eval_scalar(build, [a, b], dtype) == [exact(x, y) & mask for x, y in zip(a, b)]


def test_word_casts_zero_extend_and_keep_low_bits():
    a, _ = _word_columns(dtypes.uint64)
    assert eval_scalar(lambda x: x.cast(dtypes.uint32).cast(dtypes.uint64), [a], dtypes.uint64) == [x & 0xFFFFFFFF for x in a]
    small, _ = _word_columns(dtypes.uint32)
    assert eval_scalar(lambda x: x.cast(dtypes.uint64) * x.cast(dtypes.uint64), [small], dtypes.uint32) == [x * x for x in small]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("op", [operator.mod, operator.floordiv, operator.and_], ids=["mod", "floordiv", "and"])
def test_scalar_integer_operators_execute(dtype, op):
    a, b = _word_columns(dtype)
    b = [max(1, x) for x in b]
    assert eval_scalar(lambda x, y: op(x, y), [a, b], dtype) == [op(x, y) for x, y in zip(a, b)]
    assert eval_scalar(lambda x: op(x, 13), [a], dtype) == [op(x, 13) for x in a]
    assert eval_scalar(lambda y: op(13, y), [b], dtype) == [op(13, y) for y in b]


def test_tree_rebuild_preserves_dtype_identity():
    _, spec = tree.extract(dtypes.uint32, leaf=Poly)
    assert spec.unflatten(()) is dtypes.uint32
