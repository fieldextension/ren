"""Tests for pattern matching and rewrite rules."""

import operator

import pytest

from ren.engine.pattern import Gate, pat, rewrite, rules
from ren.engine.rules import algebraic, domain, fold, ring as ring_rules
from ren.ir import Op
from ren.ir.node import Node, ToRnsArg, from_rns_limb_count
from ren.poly import Poly
from ren.ring import Domain
from tests.utils import large_prime_ring
from ren.dtype import dtypes


def _dummy() -> Node:
    return Node.new_buffer("python", size=1, dtype=dtypes.uint64)


def test_patterns_compare_by_structure():
    left = pat.n((Op.ADD, Op.MUL), pat.x, arg=[])
    right = pat.n({Op.MUL, Op.ADD}, pat.x, arg=[])

    assert left == right
    assert hash(left) == hash(right)


def test_pattern_rejects_empty_op():
    with pytest.raises(ValueError):
        pat(set())


def test_commutative_pattern_deduplicates_equal_orderings():
    pattern = pat.x + pat.x

    assert isinstance(pattern.src, tuple)
    assert len(pattern.src) == 1


def test_multi_op_pattern_dispatches_by_each_op():
    x = _dummy()
    matcher = rules([(pat((Op.ADD, Op.MUL)), lambda: True)])

    assert matcher.match(x + x) is True
    assert matcher.match(x * x) is True
    assert matcher.match(x) is None


def test_op_patterns_dispatch_before_wildcards():
    matcher = rules([
        (pat(), lambda: "wildcard"),
        (pat((Op.BUFFER, Op.CONST)), lambda: "op"),
    ])

    assert matcher.match(_dummy()) == "op"


def test_dispatch_returns_op_handler():
    def handler(node):
        return node

    dispatcher = rules([(pat((Op.BUFFER, Op.CONST)), handler)])

    assert dispatcher.dispatch(_dummy()) is handler


def test_unchanged_result_declines_the_match():
    x = _dummy()
    node = Node(Op.NEG, dtype=dtypes.uint64, src=(x,))
    ruleset = rules([
        (pat.n(Op.NEG), lambda n: n.replace(src=n.src)),
        (pat.n(Op.NEG, pat.x), lambda n, x: x),
    ])

    assert ruleset.match(node) is x
    assert rewrite(node, ruleset) is x


def test_dispatch_binds_structural_matches():
    x, y = _dummy(), _dummy()
    node = Node(Op.SUB, dtype=dtypes.uint64, src=(x, y))
    dispatcher = rules([(pat(Op.SUB, (pat.x, pat.y)), lambda x, y, scale: (x, y, scale))])

    handler = dispatcher.dispatch(node)

    assert handler is not None
    assert handler(scale=3) == (x, y, 3)


@pytest.mark.parametrize("pattern, op", [(7 < pat.x, Op.CMPLT), (7 // pat.x, Op.IDIV), (pat.x.eq(7), Op.CMPEQ)])
def test_scalar_patterns_capture_reversed_operands(pattern, op):
    x = _dummy()
    dtype = dtypes.bool if op in {Op.CMPLT, Op.CMPEQ} else x.dtype
    expr = Node(op, dtype, src=(Node.const(7, x.dtype), x))
    assert rules([(pattern, lambda x: (x,))]).match(expr) == (x,)


def test_cast_and_where_patterns_capture_typed_branch():
    x = _dummy()
    cond, branch = x < 7, x.cast(dtypes.uint32)
    matcher = rules([
        (pat.cond.where(pat.x.cast(dtypes.uint32, name="branch"), 0), lambda cond, x, branch: (cond, x, branch)),
    ])
    assert matcher.match(cond.where(branch, 0)) == (cond, x, branch)
    assert matcher.match(cond.where(x.cast(dtypes.uint128), 0)) is None


def test_const_capture_binds_ring_wrapper():
    wrapped = Node.const(5).as_ring(large_prime_ring(4), Domain.COEFF)
    bare = Node.const(5)
    matcher = rules([(pat.c.const(), lambda c: (c,))])

    assert matcher.match(wrapped) == (wrapped,)
    assert matcher.match(bare) == (bare,)


def test_base_binds_node_under_wrapper():
    x, y = _dummy(), _dummy()
    mul = x * y
    wrapped = mul.after(_dummy())
    stacked = wrapped.as_ring(large_prime_ring(4), Domain.COEFF)
    matcher = rules([(pat.m(Op.MUL, (pat.a, pat.b)).base(), lambda m, a, b: (m, a, b))])

    assert matcher.match(wrapped) == (mul, x, y)
    assert matcher.match(stacked) == (mul, x, y)
    assert matcher.match(mul) == (mul, x, y)
    assert matcher.match(x.after(_dummy())) is None


def test_outer_and_name_bind_wrapper_and_base():
    x, y = _dummy(), _dummy()
    mul = x * y
    wrapped = mul.after(_dummy())
    matcher = rules([(pat.m(Op.MUL, (pat.a, pat.b), outer="w").base(), lambda m, w, a, b: (m, w))])

    assert matcher.match(wrapped) == (mul, wrapped)
    assert matcher.match(mul) == (mul, mul)


def test_rewrite_trace_receives_optional_name():
    x = _dummy()
    expr = Node(Op.ADD, dtype=dtypes.uint64, src=(x, Node.const(0)))
    seen: list[tuple[str | None, Node, Node]] = []

    out = rewrite(expr, rules([(pat.x + 0, lambda x: x)]), name="simplify", trace=lambda name, old, new: seen.append((name, old, new)))

    assert out is x
    assert ("simplify", expr, x) in seen


def test_rewrite_trace_allows_unnamed_rewrites():
    x = _dummy()
    expr = Node(Op.ADD, dtype=dtypes.uint64, src=(x, Node.const(0)))
    seen: list[tuple[str | None, Node, Node]] = []

    out = rewrite(expr, rules([(pat.x + 0, lambda x: x)]), trace=lambda name, old, new: seen.append((name, old, new)))

    assert out is x
    assert (None, expr, x) in seen


def test_rewrite_processes_shared_dependency_once():
    x = _dummy()
    shared = x + 0
    parent = Node(Op.NEG, dtype=dtypes.uint64, src=(shared,))
    root = Node(Op.SINK, dtype=dtypes.void, src=(parent, shared))
    calls = 0

    def remove_zero(x):
        nonlocal calls
        calls += 1
        return x

    out = rewrite(root, rules([(pat.x + 0, remove_zero)]))

    assert out.src[0].src == (x,)
    assert out.src[1] is x
    assert calls == 1


def test_rewrite_reenters_replacement_nodes():
    x = _dummy()
    root = Node(Op.NEG, dtype=dtypes.uint64, src=(x,))
    rewrite_rules = rules([
        (-pat.x, lambda x: x + 0),
        (pat.x + 0, lambda x: x),
    ])

    assert rewrite(root, rewrite_rules) is x


@pytest.mark.parametrize("bottom_up", [False, True])
def test_rewrite_accepts_unchanged_scalar_result(bottom_up: bool):
    value = Node.const(7, dtype=dtypes.uint64)
    rewrite_rules = rules([(pat.n(Op.CONST), lambda n: n.arg)])

    assert rewrite(value, rewrite_rules, bottom_up=bottom_up) is value


@pytest.mark.parametrize("gate_first", [False, True])
@pytest.mark.parametrize("bottom_up", [False, True])
def test_gate_does_not_hide_independent_occurrences(gate_first: bool, bottom_up: bool):
    x = _dummy()
    shared = x + 1
    rewrite_rules = rules([
        (-pat.x, lambda x: Gate(x + 1)),
        (pat.x + 1, lambda x: x * 5),
    ])
    outputs = (-x, shared) if gate_first else (shared, -x)

    result = rewrite(Node.sink(*outputs), rewrite_rules, bottom_up=bottom_up)

    gated, independent = result.src if gate_first else reversed(result.src)
    assert gated is shared
    assert independent is x * 5
    assert rewrite(gated, rewrite_rules, bottom_up=bottom_up) is independent


def test_parent_can_match_inside_gate_and_expose_its_children():
    x = _dummy()
    rewrite_rules = rules([
        (-pat.x, lambda x: Gate((x + 1) * 3)),
        ((pat.x * 3) * 7, lambda x: x + 9),
        (pat.x + 1, lambda x: x * 5),
    ])
    assert rewrite((-x) * 7, rewrite_rules) is (x * 5) + 9


def test_discarding_gate_does_not_protect_recreated_value():
    x = _dummy()
    shared = x + 1
    rewrite_rules = rules([
        (-pat.x, lambda x: Gate(x + 1)),
        (pat.x * 7, lambda x: shared + 9),
        (pat.x + 1, lambda x: x * 5),
    ])
    assert rewrite((-x) * 7, rewrite_rules) is (x * 5) + 9


@pytest.mark.parametrize("terminal", [False, True])
def test_converging_rewrites_apply_once_per_node(terminal):
    leaf = _dummy()
    target = Node(Op.LOAD, src=(leaf,))
    root = Node.sink(Node(Op.NEG, src=(target,)), Node(Op.INTT, src=(target,)), target)
    seen = []

    def action(n):
        seen.append(n)
        if n.op in (Op.NEG, Op.INTT):
            return target
        return Gate(n) if terminal and n is target else None

    matcher = rules([(pat.n, action)])
    for _ in range(2):
        seen.clear()
        assert rewrite(root, matcher, bottom_up=True).src == (target,) * 3
        assert seen.count(target) == 1
        assert seen.count(leaf) == (0 if terminal else 1)


def test_callable_bodies_are_structural_but_opaque_to_rewrite():
    source = Node.new_buffer("python", size=1, dtype=dtypes.uint64)
    param = Node.param(0, source)
    body = Node.sink(Node(Op.ADD, dtypes.uint64, src=(param, Node.const(0, dtypes.uint64))))
    binding = Node(Op.ADD, dtypes.uint64, src=(source, Node.const(0, dtypes.uint64)))
    call = body.call(binding)
    simplify = rules([(pat.x + 0, lambda x: x)])

    assert body not in call.toposort()
    assert body in call.toposort(enter_calls=True)

    result = rewrite(call, simplify)
    assert result.src[0] is body
    assert result.src[1] is source

    result = rewrite(call, simplify, enter_calls=True)
    assert result.src[0] is Node.sink(param)
    assert result.src[1] is source


@pytest.mark.parametrize("call_first", [False, True])
@pytest.mark.parametrize("bottom_up", [False, True])
def test_opaque_body_does_not_hide_shared_nodes_outside_call(call_first: bool, bottom_up: bool):
    source = Node.new_buffer("python", size=1, dtype=dtypes.uint64)
    param = Node.param(0, source)
    body = Node.sink(param + 0)
    call = body.call(source)
    root = Node.sink(call, body) if call_first else Node.sink(body, call)

    result = rewrite(root, rules([(pat.x + 0, lambda x: x)]), bottom_up=bottom_up)
    rewritten_call, rewritten_body = result.src if call_first else reversed(result.src)

    assert rewritten_call.src[0] is body
    assert rewritten_body.src == (param,)


def test_substitute_keeps_nested_parameter_scope():
    source = Node.new_buffer("python", size=1, dtype=dtypes.uint64)
    replacement = Node.new_buffer("python", size=1, dtype=dtypes.uint64)
    param = Node.param(0, source)
    inner_body = Node.sink(param)
    inner = inner_body.call(param)
    outer_body = Node.sink(source.after(inner))

    rebound = outer_body.substitute({param: replacement})
    rebound_inner = rebound.src[0].src[1]

    assert rebound_inner.src[0] is inner_body
    assert rebound_inner.src[1] is replacement
    assert inner_body.src[0] is param


class TestNeutralElements:
    @pytest.mark.parametrize("op,constant", [(operator.add, 0), (operator.sub, 0), (operator.mul, 1)])
    def test_preserves_operand(self, op, constant):
        x = _dummy()
        assert rewrite(op(x, constant), fold) is x

    def test_mul_zero(self):
        x = _dummy()
        assert rewrite(x * 0, fold) is Node.const(0, x.dtype)


class TestConstFolding:
    @pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
    def test_constant_table_lookup(self, dtype):
        maximum = (1 << (8 * dtype.itemsize)) - 1
        table = Node(Op.CONST_ARRAY, dtype, arg=(17, maximum))
        value = table.index(Node.const(3, dtypes.index) - 2).load()
        assert table.const_val is None
        assert value.const_val == maximum
        assert rewrite(value + 1, fold) is Node.const(0, dtype)

    @pytest.mark.parametrize("index", [None, -1, 2])
    def test_constant_table_keeps_unknown_or_invalid_index(self, index):
        table = Node(Op.CONST_ARRAY, dtypes.uint64, arg=(17, 97))
        idx = Node(Op.GIDX, dtypes.index) if index is None else Node.const(index, dtypes.index)
        value = table.index(idx).load()
        assert value.const_val is None
        assert rewrite(value, fold) is value

    def test_constant_table_keeps_index_dependencies(self):
        dependency = Node(Op.GIDX, dtypes.index)
        index = Node.const(1, dtypes.index).after(dependency)
        value = Node(Op.CONST_ARRAY, dtypes.uint64, arg=(17, 97)).index(index).load()
        assert rewrite(value, fold) is value

    @pytest.mark.parametrize(
        "op, args, expected",
        [
            (Op.ADD, (2, 3), 5),
            (Op.SUB, (5, 3), 2),
            (Op.MUL, (2, 3), 6),
            (Op.NEG, (3,), -3),
            (Op.IDIV, (7, 2), 3),
        ],
    )
    def test_exact_integer_arithmetic(self, op, args, expected):
        expr = Node(op, dtypes.weakint, src=tuple(map(Node.const, args)))
        out = rewrite(expr, algebraic)
        assert out.op is Op.CONST and out.arg == expected

    @pytest.mark.parametrize("dtype", [dtypes.weakint, dtypes.uint32, dtypes.uint64])
    def test_cast_after_constant_arithmetic_preserves_width(self, dtype):
        bits = dtype.itemsize * 8 if dtype is not dtypes.weakint else 64
        expr = (Node.const((1 << bits) - 1, dtype) + 1).cast(dtypes.uint128)
        assert rewrite(expr, algebraic).const_val == ((1 << bits) if dtype is dtypes.weakint else 0)

    def test_scalar_comparison_after_shift_overflow(self):
        x = Node.const(1 << 31, dtypes.uint32)
        expr = (x << 1) < x
        assert expr.const_val == 1
        assert rewrite(expr, fold) is Node.const(1, dtypes.bool)

    def test_idiv_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            rewrite(Node.const(1) // 0, algebraic)

    @pytest.mark.parametrize("op, expected", [(operator.floordiv, 3), (operator.mod, 1)])
    def test_typed_division_folds_only_with_nonzero_divisor(self, op, expected):
        x = Node.const(7, dtypes.uint32)
        dead = op(x, 1 << 32)  # zero after conversion to uint32
        assert rewrite(dead, algebraic).const_val is None
        result = Node.const(1, dtypes.bool).where(op(x, 2), dead)
        assert rewrite(result, fold) is Node.const(expected, x.dtype)

    def test_algebraic_folds_constants_it_reassociates(self):
        x = _dummy()
        out = rewrite(x + (Node.const(5) + Node.const(2)), algebraic)
        assert out.op is Op.ADD
        assert out.src == (x, Node.const(7))


class TestNegationCanonicalization:
    def test_neg_plus_y_to_sub(self):
        x, y = _dummy(), _dummy()
        assert rewrite(-x + y, algebraic) is y - x

    def test_x_plus_neg_to_sub(self):
        x, y = _dummy(), _dummy()
        assert rewrite(x + -y, algebraic) is x - y

    def test_zero_minus_y(self):
        y = _dummy()
        assert rewrite(0 - y, algebraic) is -y

    def test_self_cancel(self):
        x = _dummy()
        assert rewrite(x - x, algebraic) is Node.const(0, x.dtype)

    def test_double_negation(self):
        x = _dummy()
        assert rewrite(-(-x), algebraic) is x

    def test_sub_of_negation(self):
        x, y = _dummy(), _dummy()
        out = rewrite(x - (-y), algebraic)
        assert out.op is Op.ADD
        assert set(out.src) == {x, y}


def test_factor_common_term():
    x = _dummy()
    assert rewrite(x * 3 + x * 5, algebraic) is x * 8


class TestReassociationRules:
    def test_add_reassociate_folds_consts(self):
        # (X + C1) + C2 -> X + (C1 + C2)
        # Test: (a + 3) + 5 -> a + 8
        a = _dummy()
        inner = Node(Op.ADD, dtype=dtypes.uint64, src=(a, Node.const(3)))
        expr = Node(Op.ADD, dtype=dtypes.uint64, src=(inner, Node.const(5)))
        out = rewrite(expr, algebraic)
        assert out.op is Op.ADD
        assert a in out.src
        assert any(n.op is Op.CONST and int(n.arg) == 8 for n in out.src)

    def test_mul_reassociate_folds_consts(self):
        # (X * C1) * C2 -> X * (C1 * C2)
        # Test: (a * 3) * 5 -> a * 15
        a = _dummy()
        inner = Node(Op.MUL, dtype=dtypes.uint64, src=(a, Node.const(3)))
        expr = Node(Op.MUL, dtype=dtypes.uint64, src=(inner, Node.const(5)))
        out = rewrite(expr, algebraic)
        assert out.op is Op.MUL
        assert a in out.src
        assert any(n.op is Op.CONST and int(n.arg) == 15 for n in out.src)

    def test_chained_add_consts(self):
        # Test: ((a + 2) + 3) + 4 -> a + 9 (through repeated application)
        a = _dummy()
        e1 = Node(Op.ADD, dtype=dtypes.uint64, src=(a, Node.const(2)))
        e2 = Node(Op.ADD, dtype=dtypes.uint64, src=(e1, Node.const(3)))
        e3 = Node(Op.ADD, dtype=dtypes.uint64, src=(e2, Node.const(4)))
        out = rewrite(e3, algebraic)
        assert out.op is Op.ADD
        assert a in out.src
        assert any(n.op is Op.CONST and int(n.arg) == 9 for n in out.src)

    def test_const_bubbling_then_fold(self):
        # ((a + 3) + b) + 5 -> (a + b) + 8
        # Requires bubbling: (X + C) + Y -> (X + Y) + C
        # Then folding: (... + 3) + 5 -> ... + 8
        a, b = _dummy(), _dummy()
        inner1 = Node(Op.ADD, dtype=dtypes.uint64, src=(a, Node.const(3)))
        inner2 = Node(Op.ADD, dtype=dtypes.uint64, src=(inner1, b))
        expr = Node(Op.ADD, dtype=dtypes.uint64, src=(inner2, Node.const(5)))
        # The integrated pass must bubble the constant outward before folding it.
        out = rewrite(expr, algebraic)
        assert out.op is Op.ADD
        # Should have folded 3 + 5 = 8
        assert any(n.op is Op.CONST and int(n.arg) == 8 for n in out.src)


class TestDomainTransforms:
    def test_ntt_intt_cancel(self):
        # NTT(INTT(X)) -> X
        x = _dummy()
        intt_x = Node(Op.INTT, dtype=dtypes.uint64, src=(x,))
        ntt_intt = Node(Op.NTT, dtype=dtypes.uint64, src=(intt_x,))
        out = rewrite(ntt_intt, domain)
        assert out is x

    def test_intt_ntt_cancel(self):
        # INTT(NTT(X)) -> X
        x = _dummy()
        ntt_x = Node(Op.NTT, dtype=dtypes.uint64, src=(x,))
        intt_ntt = Node(Op.INTT, dtype=dtypes.uint64, src=(ntt_x,))
        out = rewrite(intt_ntt, domain)
        assert out is x

    @pytest.mark.parametrize("op", [Op.NTT, Op.INTT])
    def test_repeated_conversion(self, op):
        converted = Node(op, dtype=dtypes.uint64, src=(_dummy(),))
        repeated = Node(op, dtype=dtypes.uint64, src=(converted,))

        assert rewrite(repeated, domain) is converted


class TestRnsTransforms:
    @pytest.mark.parametrize("outside_ring", [False, True])
    def test_coefficient_roundtrip_preserves_balanced_encoding(self, outside_ring: bool):
        ring = large_prime_ring(4)[:2]
        values = [ring.q + 1, -ring.q - 1, ring.q // 2 + 1, -ring.q // 2] if outside_ring else [1, -2, 3, -4]
        raw = Poly(values, ring).node.from_rns()

        output = rewrite(raw, algebraic + ring_rules)

        assert output.ring == ring
        assert output.domain is Domain.COEFF
        result = Poly(output).to("python").realize()
        words = result.node.buffer.as_typed_buffer()
        assert len(words) == ring.n * from_rns_limb_count(ring)
        decoded = [int.from_bytes(b"".join(int(w).to_bytes(4, "little") for w in words[i :: ring.n]), "little", signed=True) for i in range(ring.n)]
        assert decoded == [(value + ring.q // 2) % ring.q - ring.q // 2 for value in values]

    @pytest.mark.parametrize("same_ring, domain", [(True, Domain.COEFF), (False, Domain.COEFF), (True, Domain.NTT)])
    def test_residue_roundtrip_requires_same_ring_and_coefficient_domain(self, same_ring: bool, domain: Domain):
        ring = large_prime_ring(4)[:2]
        source = Node.new_buffer("python", ring.storage_size, ring.dtype).as_ring(ring, domain)
        target = ring if same_ring else ring[:1]
        roundtrip = Node(Op.TO_RNS, target.dtype, src=(source.from_rns(),), arg=ToRnsArg(target, from_rns_limb_count(ring)))

        output = rewrite(roundtrip, algebraic + ring_rules)

        assert output is (source if same_ring and domain is Domain.COEFF else roundtrip)


class TestRuleIsolation:
    """Test that rules only match what they're supposed to."""

    def test_algebraic_doesnt_affect_ntt(self):
        # Algebraic rules shouldn't cancel NTT/INTT
        x = _dummy()
        ntt_x = Node(Op.NTT, dtype=dtypes.uint64, src=(x,))
        intt_ntt = Node(Op.INTT, dtype=dtypes.uint64, src=(ntt_x,))
        out = rewrite(intt_ntt, algebraic)
        assert out.op is Op.INTT  # Should be unchanged

    def test_domain_doesnt_affect_algebra(self):
        # Domain rules shouldn't affect algebraic expressions
        x = _dummy()
        expr = Node(Op.ADD, dtype=dtypes.uint64, src=(x, Node.const(0)))
        out = rewrite(expr, domain)
        assert out.op is Op.ADD  # Should be unchanged


class TestCopyRules:
    """Test COPY optimization rules."""

    def test_copy_chain_collapsing(self):
        # COPY(COPY(x, d1), d2) -> COPY(x, d2)
        from ren.engine.rules import copy

        buf = Node.new_buffer("python", 100, dtypes.uint64)
        device1 = Node(Op.DEVICE, arg="cuda")
        device2 = Node(Op.DEVICE, arg="metal")
        copy1 = Node(Op.COPY, dtype=dtypes.uint64, src=(buf, device1))
        copy2 = Node(Op.COPY, dtype=dtypes.uint64, src=(copy1, device2))

        out = rewrite(copy2, copy)
        assert out.op is Op.COPY
        assert out.src[0] is buf  # Inner value, skip intermediate copy
        assert out.src[1] is device2  # Outer device

    def test_same_device_elimination(self):
        # COPY(x, same_device) -> x
        from ren.engine.rules import copy

        buf = Node.new_buffer("python", 100, dtypes.uint64)
        device_same = Node(Op.DEVICE, arg="python")
        copy_same = Node(Op.COPY, dtype=dtypes.uint64, src=(buf, device_same))

        out = rewrite(copy_same, copy)
        assert out is buf

    def test_same_device_elimination_non_contiguous(self):
        # COPY(non_contiguous_x, same_device) -> non_contiguous_x
        from ren.engine.rules import copy

        buf = Node.new_buffer("python", 100, dtypes.uint64)
        non_contiguous = Node(Op.NTT, dtype=dtypes.uint64, src=(buf,))
        device_same = Node(Op.DEVICE, arg="python")
        copy_same = Node(Op.COPY, dtype=dtypes.uint64, src=(non_contiguous, device_same))

        out = rewrite(copy_same, copy)
        assert out is non_contiguous

    def test_different_device_no_change(self):
        # COPY(x, different_device) should NOT be eliminated
        from ren.engine.rules import copy

        buf = Node.new_buffer("python", 100, dtypes.uint64)
        device_diff = Node(Op.DEVICE, arg="cuda")
        copy_diff = Node(Op.COPY, dtype=dtypes.uint64, src=(buf, device_diff))

        out = rewrite(copy_diff, copy)
        assert out.op is Op.COPY  # Should remain a COPY
        assert out.src[0] is buf
        assert out.src[1] is device_diff


class TestPreOrderRewrite:
    """Test pre-order (bottom_up=True) rewrite mode."""

    def test_preorder_processes_parent_before_children(self):
        # Pre-order should see original node structure
        # Create NTT(INTT(x)) and track which nodes are seen
        x = _dummy()
        intt_x = Node(Op.INTT, dtype=dtypes.uint64, src=(x,))
        ntt_intt = Node(Op.NTT, dtype=dtypes.uint64, src=(intt_x,))

        seen_ops = []

        def track_op(x):
            seen_ops.append(x.op)
            return x

        # Rule that just tracks what it sees (returns unchanged)
        rule = (pat.x, track_op)
        rewrite(ntt_intt, rules([rule]), bottom_up=True)

        # In pre-order, NTT is seen first, then INTT, then BUFFER
        assert seen_ops[0] is Op.NTT

    def test_preorder_cancel_before_children(self):
        # NTT(INTT(x)) -> x should work in pre-order
        x = _dummy()
        intt_x = Node(Op.INTT, dtype=dtypes.uint64, src=(x,))
        ntt_intt = Node(Op.NTT, dtype=dtypes.uint64, src=(intt_x,))

        rule = (pat(Op.NTT, pat(Op.INTT, pat.x)), lambda x: x)
        out = rewrite(ntt_intt, rules([rule]), bottom_up=True)
        assert out is x

    def test_subtract_from_zero_rewrites_to_neg_in_both_orders(self):
        x = _dummy()
        zero_sub_x = Node(Op.SUB, dtype=dtypes.uint64, src=(Node.const(0), x))

        rule = (0 - pat.y, lambda y: -y)

        out_post = rewrite(zero_sub_x, rules([rule]), bottom_up=False)
        assert out_post.op is Op.NEG

        out_pre = rewrite(zero_sub_x, rules([rule]), bottom_up=True)
        assert out_pre.op is Op.NEG

    def test_preorder_nested_match(self):
        # Test that pre-order can match nested patterns before children are modified
        # Create: NTT(INTT(ADD(x, 0))) - pre-order should first cancel NTT/INTT
        x = _dummy()
        add_zero = Node(Op.ADD, dtype=dtypes.uint64, src=(x, Node.const(0)))
        intt_add = Node(Op.INTT, dtype=dtypes.uint64, src=(add_zero,))
        ntt_intt = Node(Op.NTT, dtype=dtypes.uint64, src=(intt_add,))

        rewrite_rules = rules([
            (pat(Op.NTT, pat(Op.INTT, pat.x)), lambda x: x),  # Cancel NTT/INTT
            (pat.x + 0, lambda x: x),  # Remove +0
        ])

        # Pre-order: NTT(INTT(ADD(x, 0))) -> ADD(x, 0) -> x
        out_pre = rewrite(ntt_intt, rewrite_rules, bottom_up=True)
        assert out_pre is x

        # Post-order should also work (process ADD first, then NTT/INTT)
        out_post = rewrite(ntt_intt, rewrite_rules, bottom_up=False)
        assert out_post is x


class TestCanonicalization:
    """Tests for arithmetic canonicalization rules."""

    def test_constants_go_right(self):
        """Constants should be moved to the right: 3 + x → x + 3."""
        x = _dummy()
        three_plus_x = Node.const(3) + x

        result = rewrite(three_plus_x, algebraic)
        # Constant should be on the right
        assert result.src[1].const_val == 3
        assert result.src[0] is x

    def test_tuplize_property(self):
        """Test that tuplize creates comparable tuples."""
        a, b = _dummy(), _dummy()

        # Different nodes should have different tuplize
        assert a.tuplize != b.tuplize

        # Tuplize should be comparable for ordering
        assert (a.tuplize < b.tuplize) or (a.tuplize > b.tuplize)

        # Same structure should compare correctly
        a_plus_b = a + b
        # tuplize format: (op.value, _tuplize_arg(arg), dtype, *children.tuplize)
        assert a_plus_b.tuplize[0] == Op.ADD.value
        assert a_plus_b.tuplize[1] is None  # arg is None for ADD
        assert a_plus_b.tuplize[2] == a.dtype
        assert a_plus_b.tuplize[3:] == (a.tuplize, b.tuplize)

        pa = Node.new_buffer("python", size=1, dtype=dtypes.uint64)
        pb = Node.new_buffer("python", size=1, dtype=dtypes.uint64)
        f0 = Node.sink(Node.param(0, pa)).call(pa)
        f1 = Node.sink(Node.param(1, pb)).call(pb)
        assert f0.tuplize != f1.tuplize
        assert (f0.tuplize < f1.tuplize) or (f0.tuplize > f1.tuplize)

    def test_commutative_dedup(self):
        """a + b and b + a should become the same node after canonicalization."""
        Node._cache.clear()
        a, b = _dummy(), _dummy()

        c = a + b
        d = b + a

        # Before rewrite, they're different nodes
        assert c is not d

        c_rewritten = rewrite(c, algebraic)
        d_rewritten = rewrite(d, algebraic)

        # After canonicalization, they should be the same node (cache dedup)
        assert c_rewritten is d_rewritten

    def test_double_mul_neg_one(self):
        """(x * -1) * -1 should simplify to x via general rules."""
        x = _dummy()
        expr = (x * -1) * -1

        result = rewrite(expr, algebraic)

        # Should simplify to just x
        assert result is x

    def test_commutative_orderings_retry_when_action_returns_none(self):
        """Commutative patterns must try alternate bindings when action rejects one."""
        x = _dummy()
        expr = Node.const(7) + x

        out = rewrite(
            expr,
            rules([
                # First ordering binds x=CONST -> returns None.
                # Second ordering binds x=BUFFER -> returns x.
                (pat.x + pat.y, lambda x, y: x if x.op is Op.BUFFER else None),
            ]),
        )
        assert out is x


class TestTermCombination:
    """Tests for term combination rules."""

    def test_x_plus_x(self):
        """x + x should become x * 2."""
        x = _dummy()
        expr = x + x

        result = rewrite(expr, algebraic)
        assert result.op is Op.MUL
        assert x in result.src
        assert any(s.const_val == 2 for s in result.src)

    def test_y_plus_x_plus_x(self):
        """(y + x) + x should become y + x*2."""
        x, y = _dummy(), _dummy()
        expr = (y + x) + x

        result = rewrite(expr, algebraic)
        # Should have y + (x * 2) structure
        assert result in (y + x * 2, x * 2 + y)

    def test_chain_simplification(self):
        """a + b + a + b + a + b should become a*3 + b*3."""
        a, b = _dummy(), _dummy()
        # Build: ((((a+b)+a)+b)+a)+b
        expr = a + b
        expr = expr + a
        expr = expr + b
        expr = expr + a
        expr = expr + b

        result = rewrite(expr, algebraic)

        # Two MULs and one ADD, with both coefficients equal to 3.
        assert result in (a * 3 + b * 3, b * 3 + a * 3)

    def test_chain_with_coefficients(self):
        """a*2 + b + a*3 should combine to a*5 + b."""
        a, b = _dummy(), _dummy()
        expr = (a * 2 + b) + a * 3

        result = rewrite(expr, algebraic)

        # Combine the a terms and preserve b.
        assert result in (a * 5 + b, b + a * 5)

    def test_addition_with_negation(self):
        """a + (-b) + a should simplify using negation canonicalization."""
        a, b = _dummy(), _dummy()
        # a + (-b) becomes a - b via existing rules
        neg_b = -b
        expr = (a + neg_b) + a

        result = rewrite(expr, algebraic)

        # Convert the negation to subtraction without dropping either term.
        assert result in ((a - b) + a, a + (a - b))


def test_const_pattern_uses_cast_value():
    value = Node.const(1 << 32).cast(dtypes.uint32)
    assert rules([(pat.c.const(0), lambda c: (c,))]).match(value) == (value,)
    assert rules([(pat.c.const(1 << 32), lambda c: (c,))]).match(value) is None
