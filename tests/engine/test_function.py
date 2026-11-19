import importlib
from dataclasses import dataclass

import pytest
import ren
from ren.context import Context
from ren.device import Device
from ren.dtype import dtypes
from ren.engine import schedule
from ren.engine.realize import call_op
from ren.ir import Node, Op
from ren.poly import Poly
from ren.ring import Domain, RingSpec
from tests.utils import large_prime_ring


@pytest.mark.parametrize("decorator", [ren.function, ren.jit])
def test_explicit_input_device_does_not_detect_default(monkeypatch, decorator):
    monkeypatch.setattr(type(Device), "_select_device", property(lambda self: pytest.fail("unexpected detection")))

    @decorator
    def double(x):
        return x * 2

    with Context(DEV=""):
        for values in ([1, 2, 3, 4], [4, 3, 2, 1]):
            x = Poly(values, large_prime_ring(4), device="python")
            assert double(x).tolist() == [2 * v for v in values]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("sign", [1, -1])
def test_function_preserves_exact_polynomial_constants(dtype, sign):
    ring = RingSpec(8, (17, 97), dtype=dtype)
    value = sign * ((1 << 200) + 7)

    @ren.jit
    @ren.function
    def evaluate(x):
        return x + value, x.ntt() * value

    for values in ([1, 2, 3, 4, 0, 0, 0, 0], [4, 3, 2, 1, 0, 0, 0, 0]):
        added, scaled = evaluate(Poly(values, ring).realize())
        assert added.tolist() == [(values[0] + value) % ring.q, *values[1:]]
        assert scaled.tolist() == [(v * value) % ring.q for v in values]


@pytest.mark.parametrize("jitted", [False, True])
def test_function_returns_different_rings(jitted: bool):
    ring = large_prime_ring(4)

    @ren.function
    def body(x):
        return x * 2, x.to_quotient_ring(ring[:2]) * 3

    run = ren.jit(body) if jitted else body
    for values in ([1, 2, 3, 4], [4, 3, 2, 1]):
        full, small = run(Poly(values, ring))
        assert full.tolist() == [v * 2 for v in values]
        assert small.tolist() == [v * 3 for v in values]


@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_function_multiplies_projected_product(jitted: bool, offset: int):
    ring = large_prime_ring(8)[:3]
    smaller_ring = ring[offset : offset + 2]

    @ren.function
    def body(x: Poly) -> Poly:
        return (x * x).to_quotient_ring(smaller_ring) * 2

    run = ren.jit(body) if jitted else body
    for a, b in ((1, 2), (2, 1)):
        x = Poly([a, b, 0, 0, 0, 0, 0, 0], ring).ntt().realize()
        result = run(x)
        assert result.ring == smaller_ring
        assert result.tolist() == [2 * a * a, 4 * a * b, 2 * b * b, 0, 0, 0, 0, 0]


def test_function_schedule_caches_are_bounded(schedule_cache):
    scheduler = importlib.import_module("ren.engine.schedule")
    ring = large_prime_ring(4)

    @ren.function
    def scale(x, factor):
        return x * factor

    x = Poly([1, 2, 3, 4], ring).realize()
    for factor in range(2, 302):
        scale(x, factor).realize()
    assert len(schedule_cache) <= 256
    assert len(scheduler._prepare_cache) <= 256


def test_function_uses_fresh_inputs_and_outputs():
    ring = large_prime_ring(4)

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    x1 = Poly([1, 2, 3, 4], ring, device="python")
    x2 = Poly([4, 3, 2, 1], ring, device="python")
    y1, y2 = double(x1), double(x2)

    ren.realize(y1, y2)

    assert double.__name__ == "double"
    assert list(y1) == [2, 4, 6, 8]
    assert list(y2) == [8, 6, 4, 2]
    assert y1.node.base.buffer is not y2.node.base.buffer


@pytest.mark.parametrize("value, shift", [(0, 0), (7, 1)])
def test_function_materializes_simplified_arguments(value, shift):
    ring = large_prime_ring(4)

    @ren.function
    def add(x: Poly, y: Poly) -> Poly:
        return x + y

    x = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT)

    x = x.ntt().realize()
    expected = [1, 2, 3, 4]
    expected[shift] += value
    assert list(add(x, (x * 0 + value).rotate(shift))) == expected


def test_function_rejects_missing_parameter():
    ring = large_prime_ring(4)
    source = Node.new_buffer("python", ring.storage_size, ring.dtype)
    param = Node.param(0, source).as_ring(ring, Domain.COEFF)
    with pytest.raises(RuntimeError, match="missing argument for parameter 0"):
        schedule(Node.sink(source.as_ring(ring, None).after(Node.sink(param + 1).call())))


def test_function_calls_compose():
    ring = large_prime_ring(4)

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    @ren.function
    def quadruple(x: Poly) -> Poly:
        doubled = double(x)
        return doubled + doubled

    @ren.function
    def fivefold(x: Poly) -> Poly:
        return quadruple(x) + x

    first = fivefold(Poly([1, 2, 3, 4], ring, device=Device.DEFAULT))
    second = fivefold(Poly([4, 3, 2, 1], ring, device=Device.DEFAULT))

    ren.realize(first, second)
    assert list(first) == [5, 10, 15, 20]
    assert list(second) == [20, 15, 10, 5]


def test_function_reuses_allocations_and_preserves_rotation_tables(scheduled_roots: list[Node]):
    ring = large_prime_ring(4)

    @ren.function
    def rotate(x: Poly) -> Poly:
        return x.clone().rotate(1)

    inputs = [(values, Poly(values, ring, device=Device.DEFAULT).realize()) for values in ([1, 2, 3, 4], [4, 3, 2, 1])]
    compiled = None
    for values, source in inputs:
        result = rotate(source)
        assert result.tolist() == [ring.q - values[-1], *values[:-1]]
        if compiled is not None:
            assert scheduled_roots == compiled
        compiled = scheduled_roots.copy()


def test_function_reuses_body_after_input_domain_placement(prepared_roots: list[Node], scheduled_roots: list[Node]):
    ring = large_prime_ring(4)

    @ren.function
    def rotate(x: Poly) -> Poly:
        return x.rotate(1)

    first = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT).realize()
    second = Poly([4, 3, 2, 1], ring, device=Device.DEFAULT).realize()
    # Buffer-backed factors keep these as polynomial multiplications requiring NTT.
    two, three = [Poly([c, 0, 0, 0], ring, device=Device.DEFAULT).realize() for c in (2, 3)]
    result = rotate(first * two).realize()
    prepared_roots.clear()
    scheduled_roots.clear()
    other = rotate(second.ntt() * three).realize()

    assert len(prepared_roots) == 1
    assert not scheduled_roots  # Both the caller and function reuse their parameterized schedules.
    assert result.tolist() == [ring.q - 8, 2, 4, 6]
    assert other.tolist() == [ring.q - 3, 12, 9, 6]


def test_function_reuses_lowering_after_simplification(scheduled_roots: list[Node]):
    ring = large_prime_ring(4)

    @ren.function
    def add(x: Poly) -> Poly:
        return x + x

    @ren.function
    def multiply(x: Poly) -> Poly:
        return x * 2

    first = add(Poly([1, 2, 3, 4], ring)).realize()
    scheduled_roots.clear()
    second = multiply(Poly([4, 3, 2, 1], ring)).realize()

    assert not scheduled_roots
    assert first.tolist() == [2, 4, 6, 8]
    assert second.tolist() == [8, 6, 4, 2]


@pytest.mark.parametrize("multiply", [False, True])
def test_function_prepares_body_once(multiply: bool, prepared_roots: list[Node], monkeypatch):
    scheduler = importlib.import_module("ren.engine.schedule")
    prepare = scheduler._prepare_function
    calls = []

    def prepare_function(body, trace):
        calls.append(body)
        return prepare(body, trace)

    monkeypatch.setattr(scheduler, "_prepare_function", prepare_function)
    ring = large_prime_ring(4)

    @ren.function
    def rotate(x: Poly) -> Poly:
        return x.rotate(1).intt()

    result = rotate(Poly([1, 2, 3, 4], ring, device=Device.DEFAULT))
    if multiply:
        result = result * 2
    assert not prepared_roots
    result.realize()
    assert len(prepared_roots) == 2  # The caller and the function body.
    assert len(calls) == 1
    assert result.tolist() == ([ring.q - 8, 2, 4, 6] if multiply else [ring.q - 4, 1, 2, 3])


def test_function_preparation_does_not_schedule(scheduled_roots: list[Node]):
    scheduler = importlib.import_module("ren.engine.schedule")

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    result = double(Poly([1, 2, 3, 4], large_prime_ring(4)))
    prepared = scheduler._prepare_body(Node.sink(result.node), None)

    assert prepared.src[0].domain is Domain.COEFF
    assert not scheduled_roots


@pytest.mark.parametrize("argument", ["conversion", "function"])
def test_function_inputs_do_not_add_copy_kernels(argument: str):
    ring = large_prime_ring(4)

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    x = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT).realize()
    source = x.ntt() if argument == "conversion" else double(x)
    result = double(source)
    linear, _ = schedule(Node.sink(result.node))

    assert not any(call_op(call) in {Op.LOAD, Op.COPY} for call in linear.src)
    assert result.tolist() == ([2, 4, 6, 8] if argument == "conversion" else [4, 8, 12, 16])


@pytest.mark.parametrize("decorate", [lambda fn: fn, ren.function], ids=["eager", "function"])
@pytest.mark.parametrize("producer", [Op.NTT, Op.MOD_REDUCE], ids=["conversion", "arithmetic"])
def test_joint_outputs_materialize_shared_value_once(decorate, producer: Op):
    ring = large_prime_ring(4)

    @decorate
    def outputs(x: Poly) -> tuple[Poly, Poly]:
        shared = x.ntt() if producer is Op.NTT else x + 1
        return shared, shared + 1 if producer is Op.NTT else shared.ntt()

    x = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT).realize()
    first, second = outputs(x)
    linear, _ = schedule(Node.sink(first.node, second.node))

    assert sum(call_op(call) is producer for call in linear.src) == 1
    ren.realize(first, second)
    assert first.tolist() == ([1, 2, 3, 4] if producer is Op.NTT else [2, 2, 3, 4])
    assert second.tolist() == [2, 2, 3, 4]


def test_eager_graph_reuses_scheduled_template(schedule_cache, scheduled_roots: list[Node]):
    ring = large_prime_ring(4)

    x = Poly([1, 2, 3, 4], ring, device="python")
    input_base = x.node.base
    first = (x.clone() * 2).realize()
    compiled = scheduled_roots.copy()
    assert compiled
    linear, output_nodes = next(iter(schedule_cache.values()))
    template = Node.sink(linear, *output_nodes)
    assert input_base not in template.toposort(enter_calls=True)
    assert first.node.base not in template.toposort(enter_calls=True)

    second = (Poly([4, 3, 2, 1], ring, device="python").clone() * 2).realize()
    assert scheduled_roots == compiled
    assert first.node.base.buffer is not second.node.base.buffer
    assert list(first) == [2, 4, 6, 8]
    assert list(second) == [8, 6, 4, 2]


def test_nested_function_keeps_temporary_allocations_separate():
    ring = large_prime_ring(4)

    @ren.function
    def inner(x: Poly) -> Poly:
        temporary = (x + 1).clone()
        return (temporary + 1).clone()

    temporary = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT).clone()
    assert (temporary + inner(temporary)).tolist() == [4, 4, 6, 8]


@pytest.mark.parametrize("lazy", [False, True])
def test_nested_function_reuses_scheduled_template(scheduled_roots: list[Node], lazy: bool):
    ring = large_prime_ring(4)

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    @ren.function
    def triple(x: Poly) -> Poly:
        return double(x) + x

    x = Poly([1, 2, 3, 4], ring, device="python")
    schedule(Node.sink(triple(x * 2 if lazy else x).node))
    compiled = set(scheduled_roots)
    scheduled_roots.clear()
    x = Poly([4, 3, 2, 1], ring, device="python")
    schedule(Node.sink((triple(x * 3 if lazy else x) + x).node))

    assert len(scheduled_roots) == 1
    assert compiled.isdisjoint(scheduled_roots)


def test_function_returns_buffer_view():
    ring = large_prime_ring(4)
    small_ring = ring[:2]

    @ren.function
    def truncate(x: Poly) -> Poly:
        return x.to_quotient_ring(small_ring)

    result = truncate(Poly([1, 2, 3, 4], ring, device=Device.DEFAULT))

    assert result.ring == small_ring
    assert list(result) == [1, 2, 3, 4]


def test_function_buffer_view_output_splices_copy():
    ring = large_prime_ring(4)
    small_ring = ring[:2]

    @ren.function
    def truncate(x: Poly) -> Poly:
        return x.to_quotient_ring(small_ring)

    x = Poly([1, 2, 3, 4], ring, device="python").realize()
    linear, _ = schedule(Node.sink(truncate(x).node))

    assert [call.src[0].op for call in linear.src] == [Op.COPY]


def test_function_walks_nested_args_and_outputs():
    ring = large_prime_ring(4)

    @ren.function
    def add(values: dict[str, Poly], *, bias: Poly) -> dict[str, Poly]:
        return {"sum": values["x"] + bias}

    result = add({"x": Poly([1, 2, 3, 4], ring, device="python")}, bias=Poly([4, 3, 2, 1], ring, device="python"))

    assert list(result["sum"]) == [5, 5, 5, 5]


@pytest.mark.parametrize("jitted", [False, True])
@pytest.mark.parametrize("slotted", [False, True])
def test_function_uses_record_storage_and_preserves_aliases(jitted: bool, slotted: bool):
    ring = large_prime_ring(4)

    class Record:
        if slotted:
            __slots__ = ("value", "label")

        def __init__(self, value: Poly, label: str):
            self.value = value
            self.label = label

    @ren.function
    def double(record: Record):
        result = Record(record.value + record.value, record.label)
        return result, result, result.value

    run = ren.jit(double) if jitted else double
    first = run(Record(Poly([1, 2, 3, 4], ring, device="python"), "metadata"))
    second = run(Record(Poly([4, 3, 2, 1], ring, device="python"), "metadata"))

    for result, expected in [(first, [2, 4, 6, 8]), (second, [8, 6, 4, 2])]:
        assert result[0] is result[1]
        assert result[0].value is result[2]
        assert result[0].label == "metadata"
        assert result[2].tolist() == expected
    assert first[0] is not second[0]


def test_function_composes_with_jit():
    ring = large_prime_ring(4)

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    @ren.jit
    def jitted(x: Poly) -> Poly:
        return double(x)

    assert list(jitted(Poly([1, 2, 3, 4], ring, device="python"))) == [2, 4, 6, 8]
    assert list(jitted(Poly([4, 3, 2, 1], ring, device="python"))) == [8, 6, 4, 2]
    jitted.clear_jit()
    assert list(jitted(Poly([2, 4, 6, 8], ring, device="python"))) == [4, 8, 12, 16]


def test_function_recomputes_metadata_while_reusing_schedule(scheduled_roots: list[Node]):
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring, device="python").realize()
    calls: list[str] = []

    @dataclass
    class Result:
        value: Poly
        label: str

    @ren.function
    def scale(x: Poly, factor: int, label: str) -> Result:
        calls.append(label)
        return Result(x * factor, label)

    results: list[Result] = []
    schedule_counts: list[int] = []
    for factor, label in [(2, "first"), (2, "second"), (3, "third"), (2, "fourth")]:
        result = scale(x, factor, label)
        result.value.realize()
        results.append(result)
        schedule_counts.append(len(scheduled_roots))
        assert result.label == label

    assert calls == ["first", "second", "third", "fourth"]
    assert schedule_counts[0] == schedule_counts[1] < schedule_counts[2] == schedule_counts[3]
    assert [list(result.value) for result in results] == [[2, 4, 6, 8], [2, 4, 6, 8], [3, 6, 9, 12], [2, 4, 6, 8]]


def test_function_specializes_input_domains():
    ring = large_prime_ring(4)

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    x = Poly([1, 2, 3, 4], ring, device="python")

    assert list(double(x)) == [2, 4, 6, 8]
    assert list(double(x.ntt())) == [2, 4, 6, 8]


def test_function_traces_concrete_inputs():
    ring = large_prime_ring(4)

    @ren.function
    def add_by_state(x: Poly) -> Poly:
        return x + (1 if x.is_realized else 2)

    x = Poly([1, 2, 3, 4], ring, device="python").realize()

    assert list(add_by_state(x)) == [2, 2, 3, 4]


def test_function_leaves_domain_placement_to_scheduler():
    ring = large_prime_ring(4)

    @ren.function
    def identity(x: Poly) -> Poly:
        return x

    x = Poly([1, 2, 3, 4], ring, device="python")
    result = identity(x + x.ntt())

    assert result.domain is None
    assert not result.is_realized
    result.realize()
    assert list(result) == [2, 4, 6, 8]
    assert result.domain is Domain.NTT


@pytest.mark.parametrize("reverse", [False, True])
def test_function_combines_resolved_and_unresolved_domains(reverse: bool):
    ring = large_prime_ring(4)

    @ren.function
    def double(x: Poly) -> Poly:
        return x + x

    x = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT)
    resolved, unresolved = double(x), double(x + x.ntt())
    result = unresolved + resolved if reverse else resolved + unresolved

    assert list(result) == [6, 12, 18, 24]


def test_function_places_domains_across_its_boundary():
    ring = large_prime_ring(4)

    @ren.function
    def square(x: Poly) -> Poly:
        return x * x

    @ren.function
    def rescale(x: Poly) -> Poly:
        return x.rescale(ring.moduli[:-1])

    x = Poly([1, 2, 3, 4], ring, device="python")
    squared, rescaled = square(x), rescale(x.ntt())
    squared.realize()
    rescaled.realize()

    assert squared.domain is Domain.NTT
    assert rescaled.domain is Domain.COEFF


def test_function_output_domain_is_placed_for_caller():
    ring = large_prime_ring(4)

    @ren.function
    def square(x: Poly) -> Poly:
        return x * x

    result = square(Poly([1, 2, 3, 4], ring, device="python")).rescale(ring.moduli[:-1])
    result.realize()

    assert result.domain is Domain.COEFF


def test_function_preserves_output_domains_while_tracing():
    ring = large_prime_ring(4)

    @ren.function
    def split_domains(x: Poly) -> tuple[Poly, Poly]:
        return x, x.ntt()

    @ren.function
    def outer(x: Poly) -> tuple[Poly, Poly]:
        coeff, ntt = split_domains(x)
        return coeff if coeff.domain is Domain.COEFF else coeff + 1, ntt if ntt.domain is Domain.NTT else ntt + 1

    coeff, ntt = outer(Poly([1, 2, 3, 4], ring, device=Device.DEFAULT))
    assert (coeff.domain, ntt.domain) == (Domain.COEFF, Domain.NTT)
    assert not coeff.is_realized and not ntt.is_realized
    ren.realize(coeff, ntt)

    assert coeff.domain is Domain.COEFF
    assert ntt.domain is Domain.NTT
    assert coeff.tolist() == ntt.tolist() == [1, 2, 3, 4]


def test_function_preserves_identity_outputs():
    ring = large_prime_ring(4)

    @ren.function
    def duplicate(x: Poly) -> tuple[Poly, Poly]:
        return x, x

    x = Poly([1, 2, 3, 4], ring, device="python")
    left, right = duplicate(x)

    assert left is right
    assert list(left) == [1, 2, 3, 4]


def test_function_distinct_output_polys_keep_shared_storage():
    ring = large_prime_ring(4)

    @ren.function
    def pair(x: Poly) -> tuple[Poly, Poly]:
        result = x + x
        return result, Poly(result.node)

    left, right = pair(Poly([1, 2, 3, 4], ring, device="python"))
    ren.realize(left, right)

    assert left is not right
    assert left.node.base.buffer is right.node.base.buffer
    assert list(left) == list(right) == [2, 4, 6, 8]


def test_function_replays_distinct_outputs_with_same_ring():
    ring = large_prime_ring(4)

    @ren.function
    def neighbors(x: Poly) -> tuple[Poly, Poly]:
        return x + 1, x + 2

    @ren.jit
    def jitted(x: Poly) -> tuple[Poly, Poly]:
        return neighbors(x)

    left, right = jitted(Poly([1, 2, 3, 4], ring, device="python"))
    assert list(left) == [2, 2, 3, 4]
    assert list(right) == [3, 2, 3, 4]

    left, right = jitted(Poly([4, 3, 2, 1], ring, device="python"))
    assert list(left) == [5, 3, 2, 1]
    assert list(right) == [6, 3, 2, 1]


def test_function_materializes_constant_output():
    ring = large_prime_ring(4)

    @ren.function
    def constant(_x: Poly) -> Poly:
        return Poly.const(7, ring, device="python")

    result = constant(Poly([1, 2, 3, 4], ring, device="python")).realize()

    assert list(result) == [7, 0, 0, 0]
    assert result.node.base.op is Op.BUFFER


def test_function_binds_unrealized_random_argument():
    ring = large_prime_ring(4)
    weight = Poly.uniform(ring)

    @ren.function
    def add_weight(x: Poly, weight: Poly) -> Poly:
        return x + weight

    x = Poly([1, 2, 3, 4], ring, device=Device.DEFAULT)
    result = add_weight(x, weight)

    assert list(result) == list(x + weight)


@pytest.mark.parametrize("ntt", [False, True])
def test_function_binds_realized_implicit_polys(ntt: bool):
    ring = large_prime_ring(4)
    weight = Poly([4, 3, 2, 1], ring, device="python").realize()

    @ren.function
    def add_weight(x: Poly) -> Poly:
        return x + weight

    x = Poly([1, 2, 3, 4], ring, device="python")
    y = Poly([4, 3, 2, 1], ring, device="python")
    first = add_weight(x.ntt() if ntt else x)
    second = add_weight(y.ntt() if ntt else y)

    assert list(first) == [5, 5, 5, 5]
    assert list(second) == [8, 6, 4, 2]


def test_function_rejects_randomness_in_implicit_polys():
    ring = large_prime_ring(4)
    weight = Poly.uniform(ring)

    @ren.function
    def add_weight(x: Poly) -> Poly:
        return x + weight

    with pytest.raises(RuntimeError, match=r"^ren\.function: random values must be created outside the function$"):
        add_weight(Poly([1, 2, 3, 4], ring, device=weight.device))


def test_function_rejects_realization_while_tracing():
    ring = large_prime_ring(4)

    @ren.function
    def realize_input(x: Poly) -> Poly:
        return x.realize()

    with pytest.raises(RuntimeError, match="Poly realization is not allowed while tracing"):
        realize_input(Poly([1, 2, 3, 4], ring, device="python"))


@pytest.mark.parametrize("populated", [False, True])
def test_function_rejects_container_realization_while_tracing(populated: bool):
    ring = large_prime_ring(4)

    @dataclass
    class Batch(ren.PolyContainer):
        values: list[Poly]

    @ren.function
    def realize_batch(x: Poly) -> Poly:
        Batch([x] if populated else []).realize()
        return x

    with pytest.raises(RuntimeError, match="Poly realization is not allowed while tracing"):
        realize_batch(Poly([1, 2, 3, 4], ring, device="python"))
