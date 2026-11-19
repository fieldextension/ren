import gc
import weakref
from collections import deque, namedtuple
from dataclasses import dataclass
from functools import partial

import pytest
from attrs import define, field, make_class

from ren import tree
from ren.ir.node import Node
from ren.poly import Poly
from ren.utils import disable_gc
from tests.utils import Leaf


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Pair:
    a: Poly
    b: Poly


@pytest.mark.parametrize("decorate", [lambda cls: cls, lambda cls: dataclass(cls, slots=True)], ids=["dict", "slots"])
def test_runtime_fields_do_not_depend_on_annotations(decorate):
    @decorate
    class Record:
        value: str

    source = object.__new__(Record)
    object.__setattr__(source, "value", Leaf(1))
    leaves, spec = tree.extract(source, leaf=Poly)
    assert leaves == tree.collect(source, leaf=Poly) == [source.value]
    replacement = Poly.const(2, leaves[0].ring)
    for result in (spec.unflatten([replacement]), tree.replace(source, lambda _: replacement, leaf=Poly)):
        assert type(result) is Record and result.value is replacement
    assert source.value is leaves[0]


def test_instance_dictionary_shapes_are_not_cached_per_class():
    class Box:
        pass

    first, second = Box(), Box()
    vars(first)["p"], vars(second)["a.b"] = Leaf(1), Leaf(2)
    for source, name in ((first, "p"), (second, "a.b")):
        leaves, spec = tree.extract(source, leaf=Poly)
        assert leaves == tree.collect(source, leaf=Poly) == [vars(source)[name]]
        replacement = Leaf(3)
        result = spec.unflatten([replacement])
        assert vars(result) == {name: replacement}


def test_traversal_does_not_retain_unbounded_temporary_classes():
    def traverse():
        class Temporary:
            __slots__ = ("value",)

        source = Temporary()
        source.value = 1
        assert tree.collect(source, leaf=int) == [1]
        return weakref.ref(Temporary)

    original = traverse()
    for _ in range(1024):
        traverse()
    gc.collect()
    assert original() is None


def test_stored_slots_bypass_overriding_properties():
    class Base:
        __slots__ = ("p",)

    class Box(Base):
        @property
        def p(self):
            raise AssertionError("property must not run")

    source = Box()
    Base.p.__set__(source, Leaf(1))
    leaves, spec = tree.extract(source, leaf=Poly)
    replacement = Leaf(2)
    assert len(leaves) == 1
    result = spec.unflatten([replacement])
    assert Base.p.__get__(result) is replacement


@pytest.mark.parametrize("base, args", [(deque, ([Leaf(1)],)), (Exception, (Leaf(1),)), (partial, (lambda p: p, Leaf(1)))])
def test_native_storage_is_rejected_even_with_declared_fields(base, args):
    class Native(base):
        label: str

    with pytest.raises(TypeError, match="Native"):
        tree.collect(Native(*args), leaf=Poly)


@pytest.mark.parametrize(
    "wrap", [iter, reversed, lambda p: (x for x in p), lambda p: dict(enumerate(p)).values(), lambda p: dict(enumerate(p)).items(), lambda _: len]
)
def test_native_storage_is_rejected_before_reconstruction(wrap):
    source = wrap([Leaf(1)])
    for operation in (tree.collect, tree.extract, lambda value, **kwargs: tree.replace(value, lambda p: p, **kwargs)):
        with pytest.raises(TypeError, match="not a supported storage type"):
            operation(source, leaf=Poly)


def test_namedtuple_subclasses_cannot_hide_extra_storage():
    class Output(namedtuple("OutputBase", "p")):
        extra: Poly

    source = Output(Leaf(1))
    source.extra = Leaf(2)
    for operation in (tree.collect, tree.extract, lambda value, **kwargs: tree.replace(value, lambda p: p, **kwargs)):
        with pytest.raises(TypeError, match="Output"):
            operation(source, leaf=Poly)


def test_consumed_slot_declarations_are_rejected():
    class Record:
        __slots__ = iter(("p",))
        p: Poly

    source = Record()
    source.p = Leaf(1)
    for operation in (tree.collect, tree.extract, lambda value, **kwargs: tree.replace(value, lambda p: p, **kwargs)):
        with pytest.raises(TypeError, match="__slots__"):
            operation(source, leaf=Poly)


@pytest.mark.parametrize(
    "base, derived",
    [
        (lambda cls: cls, dataclass),
        (dataclass, define),
        (define, dataclass),
        (lambda cls: make_class(cls.__name__, {"p": field()}), dataclass),
    ],
    ids=["annotated-dataclass", "dataclass-attrs", "attrs-dataclass", "unannotated-attrs-dataclass"],
)
def test_inherited_fields_are_traversed_and_rebuilt(base, derived):
    @base
    class Base:
        p: Poly
        tag = "original"

    @derived
    class State(Base):
        q: Poly

        def __init__(self, *args, **kwargs):
            raise AssertionError("constructor must not run")

    source = object.__new__(State)
    source.p, source.q = Leaf(1), Leaf(2)
    leaves, spec = tree.extract(source, leaf=Poly)
    assert leaves == tree.collect(source, leaf=Poly)
    assert {id(p) for p in leaves} == {id(source.p), id(source.q)}
    replacements = {p: Leaf(i + 3) for i, p in enumerate(leaves)}
    for result in (spec.unflatten(map(replacements.__getitem__, leaves)), tree.replace(source, replacements.__getitem__, leaf=Poly)):
        assert type(result) is State and result is not source
        assert result.p is replacements[source.p] and result.q is replacements[source.q]
    assert source.p in leaves and source.q in leaves
    Base.tag = "updated"
    assert result.tag == "updated"


@disable_gc()
def test_namedtuple_reconstruction_bypasses_hooks_and_releases_objects():
    class Output(namedtuple("OutputBase", "left right")):
        __slots__ = ()

        def __new__(cls, left, right):
            raise AssertionError("constructor")

        def __iter__(self):
            raise AssertionError("iterator")

    p, q = Leaf(1), Leaf(2)
    pair = Pair(p, p)
    source = tuple.__new__(Output, [pair, pair])
    originals, spec = tree.extract(source, leaf=Leaf)
    assert originals == [p, p]
    rebuilt_refs: list[weakref.ReferenceType] = [weakref.ref(q)]
    for result in (spec.unflatten([q, q]), tree.replace(source, lambda _, replacement=q: replacement, leaf=Leaf)):
        assert type(result) is Output and type(result.left) is Pair
        assert result.left is result.right and result.left is not pair
        assert result.left.a is result.left.b is q
        rebuilt_refs.append(weakref.ref(result.left))
    assert pair.a is pair.b is p
    source_refs = [weakref.ref(pair), weakref.ref(p)]
    del source, pair, p, originals
    assert all(ref() is None for ref in source_refs)
    del result, q
    assert all(ref() is None for ref in rebuilt_refs)


@disable_gc()
def test_plan_does_not_retain_source_or_prior_replays():
    source = Pair(Leaf(1), Leaf(2))
    refs = [weakref.ref(source), weakref.ref(source.a), weakref.ref(source.b)]
    assert tree.collect(source, leaf=Poly) == [source.a, source.b]
    originals, spec = tree.extract([source, source], leaf=Poly)
    del originals, source
    assert all(ref() is None for ref in refs)
    first, second = spec.unflatten([Leaf(3), Leaf(4)]), spec.unflatten([Leaf(5), Leaf(6)])
    assert first[0] is first[1] and second[0] is second[1]
    assert first[0] is not second[0]
    refs = [weakref.ref(value) for value in (first[0], first[0].a, first[0].b, second[0], second[0].a, second[0].b)]
    del first, second
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("source", [1, [1, 2], {"empty": []}])
def test_unflatten_does_not_mutate_the_replacement_sequence(source):
    originals, spec = tree.extract(source, leaf=int)
    replacements = originals.copy()
    first = spec.unflatten(replacements)
    assert len(replacements) == len(originals) and all(a is b for a, b in zip(replacements, originals))
    replacements[:] = [value + 10 for value in originals]
    assert tree.collect(spec.unflatten(replacements), leaf=int) == replacements
    assert first == source
    with pytest.raises(ValueError, match="incorrect number"):
        spec.unflatten([*originals, 99])


def test_selected_nodes_are_terminal_without_entering_the_ir_graph():
    @dataclass
    class LazyValue:
        node: Node
        label: str = "value"

    node, replacement = (Leaf(i).node for i in (1, 2))
    source = LazyValue(node)
    leaves, spec = tree.extract([source, source], leaf=Node)
    assert leaves == tree.collect([source, source], leaf=Node) == [node]
    for result in (spec.unflatten([replacement]), tree.replace([source, source], lambda _: replacement, leaf=Node)):
        assert result[0] is result[1] and result[0] is not source
        assert result[0].node is replacement and result[0].label == "value"
    assert tree.replace(node, lambda _: replacement, leaf=Node) is replacement


def test_type_selection_has_independent_cached_layouts():
    @dataclass
    class Record:
        count: int
        label: str

    source = Record(7, "label")
    for leaf, original, replacement, expected in (
        (int, 7, 9, Record(9, "label")),
        (str, "label", "new", Record(7, "new")),
        (int, 7, 11, Record(11, "label")),
    ):
        leaves, spec = tree.extract(source, leaf=leaf)
        assert leaves == tree.collect(source, leaf=leaf) == [original]
        result = spec.unflatten([replacement])
        assert result == expected
        assert tree.replace(source, lambda _: replacement, leaf=leaf) == result
