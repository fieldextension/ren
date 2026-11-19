import weakref
from collections import ChainMap, defaultdict, deque
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, make_dataclass
from types import MappingProxyType
from typing import Annotated, Any, ForwardRef, assert_type, cast
from unittest.mock import Mock

from attrs import define, field as attrs_field, make_class
from flint import acb
import numpy as np
import pytest

from ren import realize
import ren.poly as poly_module
from ren.container import PolyContainer
from ren.device import Device
from ren.poly import Poly
from ren.utils import disable_gc
from tests.utils import Leaf, large_prime_ring


@dataclass
class Pair(PolyContainer):
    left: Poly | PolyContainer | str
    right: Poly | PolyContainer | str
    tag: str = "pair"


@dataclass
class Batch(PolyContainer):
    pending: deque[Poly]
    label: str

    def __tree_reduce__(self):
        cls = type(self)

        def rebuild(polys, label):
            return cls(deque(polys), label)

        return rebuild, (tuple(self.pending), self.label)


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


def test_base_container_keeps_arbitrary_root_traversal():
    empty = PolyContainer()
    assert empty.replace(polys=()) is empty
    leaf, second = Leaf(1), Leaf(2)
    assert PolyContainer.polys([leaf], {"p": leaf}) == [leaf, leaf]
    assert PolyContainer.polys() == []
    assert PolyContainer.polys(leaf, [second, leaf], second) == [leaf, second, leaf, second]
    shared = [leaf, leaf]
    assert PolyContainer.polys(leaf, shared, shared, leaf) == [leaf] * 4
    assert PolyContainer.polys(shared, list(shared)) == [leaf] * 4


def test_map_polys_preserves_shared_containers_and_separates_poly_positions():
    leaf, first, second = Leaf(1), Leaf(2), Leaf(3)
    shared = Pair(leaf, leaf)
    original = Pair(shared, shared)
    polys, spec = PolyContainer.flatten(original)
    assert polys == original.polys() == [leaf, leaf]
    replacements = iter((first, second))
    for result in (original.map_polys(lambda _: next(replacements)), spec.unflatten([first, second])):
        assert_type(result, Pair)
        assert result.left is result.right and result.left is not shared
        assert isinstance(result.left, Pair)
        assert result.left.left is first and result.left.right is second
        assert result.tag == original.tag
    assert shared.left is shared.right is leaf
    renamed = result.replace(tag="mapped")
    assert renamed.tag == "mapped" and renamed.left is renamed.right is result.left
    assert original.map_polys(lambda poly: poly) is original


def test_to_gathers_polynomials_from_multiple_devices():
    ring = large_prime_ring(4)
    shared = Pair(Poly.const(1, ring, device="python"), Poly.const(2, ring, device="cuda:1"))
    source = Pair(shared, shared, tag="distributed")
    moved = source.to(" CPU ")
    assert_type(moved, Pair)
    assert moved.left is moved.right and moved.left is not shared
    assert moved.tag == source.tag
    assert [poly.device for poly in moved.polys()] == ["python", "python"]
    assert [poly.device for poly in source.polys()] == ["python", "cuda:1"]
    assert moved.to("python") is moved
    empty = Pair("left", "right")
    assert empty.to("python") is empty


def test_to_preserves_realized_values():
    ring = large_prime_ring(4)
    source = Pair(Poly([1, 2, 3, 4], ring, device="python"), Poly([4, 3, 2, 1], ring, device="python")).realize()
    moved = source.to(Device.DEFAULT).realize()
    assert all(poly.device == Device.DEFAULT for poly in moved.polys())
    restored = moved.to("python").realize()
    assert [poly.tolist() for poly in restored.polys()] == [[1, 2, 3, 4], [4, 3, 2, 1]]


def test_realize_batches_multiple_containers(monkeypatch: pytest.MonkeyPatch):
    ring = large_prime_ring(4)
    left = Poly([1, 2, 3, 4], ring) + 1
    right = Poly([4, 3, 2, 1], ring) + 1
    schedule = Mock(wraps=poly_module.schedule)
    monkeypatch.setattr(poly_module, "schedule", schedule)

    first, second = Pair(left, "opaque"), Pair(right, left)
    assert first.polys(second) == [left, right, left]
    assert realize(left, {"first": first}, [second]) is None
    assert schedule.call_count == 1
    assert left.is_realized and right.is_realized
    assert realize() is None
    assert realize([], first, second) is None
    assert first.realize() is first
    assert left.realize() is left
    assert schedule.call_count == 1
    assert left.tolist() == [2, 2, 3, 4]
    assert right.tolist() == [5, 3, 2, 1]


def test_parent_specialization_preserves_subclass_fields():
    @dataclass
    class Base(PolyContainer):
        a: Leaf

    first, second, replacement = Leaf(1), Leaf(2), Leaf(3)
    Base(first).replace(polys=[replacement])

    @dataclass
    class Derived(Base):
        b: Leaf
        label: str

    value = Derived(first, second, "original")
    assert value.polys() == [first, second]
    result = value.replace(polys=[replacement, first], label="updated")
    assert type(result) is Derived
    assert result.a is replacement and result.b is first
    assert result.label == "updated"
    assert value.label == "original"


def test_regular_polycontainers_support_polynomial_operations():
    class Plain(PolyContainer):
        def __init__(self, p):
            self.p = p

    class Reduced(Plain):
        def __tree_reduce__(self):
            return type(self), (self.p,)

    @define
    class Attrs(PolyContainer):
        p: Poly

    leaf, replacement = Leaf(1), Leaf(2)
    for cls in (Plain, Reduced, Attrs):
        source = cls(leaf)
        polys, spec = source.flatten(source)
        assert polys == source.polys() == [leaf]
        assert source.replace() is source
        for result in (source.map_polys(lambda _: replacement), source.replace(polys=[replacement]), spec.unflatten([replacement])):
            assert type(result) is cls and result.p is replacement
        assert source.p is leaf
    with pytest.raises(TypeError):
        Plain(leaf).replace(p=replacement)


def test_replace_uses_custom_named_field_replacement():
    class Box(PolyContainer):
        def __init__(self, p, label):
            self.p, self.label = p, label

        def __replace__(self, *, label):
            return type(self)(self.p, label.upper())

    leaf, replacement = Leaf(1), Leaf(2)
    source = Box(leaf, "original")
    renamed = source.replace(label="new")
    assert renamed.p is leaf and renamed.label == "NEW"
    result = source.replace(polys=[replacement], label="mapped")
    assert type(result) is Box and result.p is replacement and result.label == "MAPPED"
    assert source.p is leaf and source.label == "original"


def test_container_fields_follow_current_values_for_unknown_types():
    @dataclass
    class Box(PolyContainer):
        value: Leaf | None
        history: list[Leaf] = field(default_factory=list)
        payload: object = None

    leaf = Leaf(1)
    assert PolyContainer.polys(Box(None)) == []
    assert PolyContainer.polys(Box(leaf)) == [leaf]
    value = Box(None, payload=leaf)
    value.history.append(leaf)
    assert PolyContainer.polys(value) == [leaf, leaf]
    value.history.clear()
    assert PolyContainer.polys(value) == [leaf]
    value.payload = None
    assert value.replace(polys=(Leaf(2) for _ in value.polys())) is value


def test_replace_preserves_unchanged_children_while_replay_copies_them():
    @dataclass
    class Box(PolyContainer):
        value: object
        metadata: list

    leaf, replacement = Leaf(1), Leaf(2)
    value = Box({"batch": [(leaf, leaf), None]}, [[]])
    polys, spec = PolyContainer.flatten(value)
    assert polys == value.polys() == [leaf, leaf]
    replaced, replayed = value.replace(polys=[replacement, replacement]), spec.unflatten([replacement, replacement])
    for rebuilt in (replaced, replayed):
        assert rebuilt.value == {"batch": [(replacement, replacement), None]}
        assert rebuilt.metadata == value.metadata
    assert replaced.metadata is value.metadata
    assert replayed.metadata is not value.metadata and replayed.metadata[0] is not value.metadata[0]


def test_container_preserves_declaration_order_across_direct_and_nested_fields():
    @dataclass
    class Bundle(PolyContainer):
        first: Poly
        middle: Pair
        last: Poly
        optional: Poly | None

    first, second, third, last = (Leaf(i) for i in range(4))
    source = Bundle(first, Pair(second, third), last, first)
    polys, spec = PolyContainer.flatten(source)
    assert polys == source.polys() == [first, second, third, last, first]
    replacements = [Leaf(i) for i in range(10, 15)]
    for result in (source.replace(polys=replacements), spec.unflatten(replacements)):
        assert [result.first, result.middle.left, result.middle.right, result.last, result.optional] == replacements


def test_unresolved_annotations_do_not_hide_stored_polys():
    Box = make_dataclass("Box", [("value", ForwardRef("UnresolvedNode"))], bases=(PolyContainer,))
    source = Box([Leaf(1)])
    polys, spec = PolyContainer.flatten(source)
    assert polys == source.value
    assert spec.unflatten(polys).value == source.value


@pytest.mark.parametrize("value", [make_dataclass("Empty", [])(), make_dataclass("Metadata", [("label", str)])("label")])
def test_scalar_only_dataclasses_rebuild_as_records(value):
    polys, spec = PolyContainer.flatten([value])
    result = spec.unflatten([])[0]
    assert polys == [] and result == value and result is not value


@pytest.mark.parametrize("record", [dataclass, define], ids=["dataclass", "attrs"])
@pytest.mark.parametrize("decorate", [True, False], ids=["decorated", "annotated"])
def test_metadata_subclasses_can_add_polynomials(record, decorate):
    @record
    class Metadata:
        label: str

        def __init__(self, label):
            self.label = label

    class WithPoly(Metadata):
        p: Poly

        def __init__(self, label, p):
            super().__init__(label)
            self.p = p

    if decorate:
        WithPoly = record(WithPoly)

    @dataclass
    class Batch(PolyContainer):
        metadata: Metadata

    original, replacement = Leaf(1), Leaf(2)
    source = Batch(WithPoly("nested", original))
    polys, spec = PolyContainer.flatten(source)
    assert polys == source.polys() == [original]
    for result in (source.replace(polys=[replacement]), spec.unflatten([replacement])):
        assert isinstance(result.metadata, WithPoly)
        assert PolyContainer.polys(result.metadata) == [replacement]
        assert result.metadata.label == "nested"


@pytest.mark.parametrize("collection", [set, frozenset])
def test_capture_preserves_distinct_set_positions(collection):
    left, right = Leaf(1), Leaf(2)
    source = (collection((left, right)), left)
    polys, spec = PolyContainer.flatten(source)
    replacements = [Leaf(3), Leaf(4), Leaf(5)]
    assert len(polys) == 3
    result, repeated = spec.unflatten(replacements)
    assert type(result) is collection and result == collection(replacements[:2])
    assert repeated is replacements[2]


@pytest.mark.parametrize("aliased", [False, True])
def test_dataclass_reconstruction_preserves_or_recomputes_derived_fields(aliased):
    calls = []

    @dataclass(frozen=True, slots=True)
    class Base(PolyContainer):
        values: Annotated[list[Leaf], "unrelated metadata"]

    @dataclass(frozen=True, slots=True)
    class Derived(Base):
        extra: Leaf
        metadata: object
        multiplier: int = 1
        total: int = field(init=False)

        def __post_init__(self):
            calls.append(self)
            object.__setattr__(self, "total", self.multiplier * (self.values[0].value + self.extra.value))

    leaf, replacement = Leaf(1), Leaf(2)
    metadata = object()
    original = Derived([leaf], leaf if aliased else Leaf(3), metadata)
    _, spec = PolyContainer.flatten(original)
    for rebuilt in (
        original.replace(polys=[replacement, replacement]),
        original.map_polys(lambda _: replacement),
        spec.unflatten([replacement, replacement]),
    ):
        assert rebuilt is not original
        assert rebuilt.values[0] is rebuilt.extra is replacement
        assert rebuilt.metadata is metadata
        assert rebuilt.total == original.total
    assert calls == [original]

    calls.clear()
    replacements = [Leaf(11), Leaf(12)]
    result = original.replace(polys=iter(replacements), multiplier=2)
    assert calls == [result]
    assert result.values[0] is replacements[0] and result.extra is replacements[1]
    assert result.total == 46
    metadata_only = original.replace(polys=original.polys(), multiplier=3)
    assert metadata_only.values is original.values and metadata_only.extra is original.extra
    assert metadata_only.total == 3 * original.total
    assert original.multiplier == 1
    with pytest.raises(TypeError, match="init=False"):
        original.replace(polys=replacements, total=0)


@disable_gc()
def test_tree_reduce_hook_rebuilds_unsupported_storage():
    first, second = Leaf(1), Leaf(2)
    batch = Batch(deque([first, second]), "batch")
    source = Pair(batch, batch)
    polys, spec = PolyContainer.flatten(source)
    assert polys == source.polys() == [first, second]
    assert source.map_polys(lambda poly: poly) is source
    replacements = [Leaf(3), Leaf(4)]
    for result in (source.replace(polys=replacements), spec.unflatten(replacements)):
        assert result.left is result.right and type(result.left) is Batch
        assert type(result.left.pending) is deque and list(result.left.pending) == replacements
        assert result.left.label == "batch"
    renamed = batch.replace(polys=replacements, label="renamed")
    assert list(renamed.pending) == replacements and renamed.label == "renamed"
    assert list(batch.pending) == [first, second]
    refs = [weakref.ref(batch), weakref.ref(first), weakref.ref(second)]
    del source, batch, first, second, polys, result, renamed
    assert all(ref() is None for ref in refs)


def test_container_operations_reject_cycles():
    value = Pair("left", "right")
    value.right = value
    recursive: list[object] = [Leaf(1)]
    recursive.append(recursive)
    reduced = Batch(deque(), "cyclic")
    reduced.pending.append(cast(Any, reduced))
    for root in (value, recursive, reduced):
        with pytest.raises(ValueError, match="containers cannot contain cycles"):
            PolyContainer.flatten(root)
        with pytest.raises(ValueError, match="containers cannot contain cycles"):
            PolyContainer.polys(root)
    for root in (value, reduced):
        with pytest.raises(ValueError, match="containers cannot contain cycles"):
            root.replace(polys=())


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType])
def test_capture_preserves_mapping_type_and_order(mapping_type):
    first = Leaf(1)
    second = Leaf(2)
    root = mapping_type({"b": second, "a": first})

    polys, spec = PolyContainer.flatten(root)
    replacement_a = Leaf(3)
    replacement_b = Leaf(4)
    rebuilt = spec.unflatten([replacement_a, replacement_b])

    assert polys == [first, second]
    assert type(rebuilt) is type(root)
    assert list(rebuilt) == ["b", "a"]
    assert rebuilt == {"b": replacement_b, "a": replacement_a}


def test_capture_copies_mutable_static_containers():
    root = (Leaf(1), [1], {"value": 1}, Pair("left", "right"))
    polys, spec = PolyContainer.flatten(root)

    first = spec.unflatten(polys)
    first[1].append(2)
    first[2]["value"] = 2
    first[3].tag = "changed"
    second = spec.unflatten(polys)

    assert second[1] == [1]
    assert second[2] == {"value": 1}
    assert second[1] is not first[1]
    assert second[2] is not first[2]
    assert second[3] == root[3]
    assert second[3] is not root[3]


def test_specs_with_matching_structure_keep_independent_metadata_and_polys():
    first = Pair(Leaf(1), Leaf(2), "first")
    second = Pair(Leaf(3), Leaf(4), "second")
    first_polys, first_spec = PolyContainer.flatten(first)
    second_polys, second_spec = PolyContainer.flatten(second)

    first_result = first_spec.unflatten(second_polys)
    second_result = second_spec.unflatten(first_polys)
    assert_type(first_result, Pair)

    assert first_result.tag == "first"
    assert second_result.tag == "second"
    assert (first_result.left, first_result.right) == tuple(second_polys)
    assert (second_result.left, second_result.right) == tuple(first_polys)


@pytest.mark.parametrize("root", [Leaf(1), [Leaf(1)], "opaque"])
def test_spec_requires_exact_poly_count(root):
    polys, spec = PolyContainer.flatten(root)
    replacements = [Leaf(2) for _ in polys]
    result = spec.unflatten(iter(replacements))
    if isinstance(root, Poly):
        assert polys == [root]
        assert result is replacements[0]

    if polys:
        with pytest.raises(ValueError):
            spec.unflatten(polys[:-1])

    with pytest.raises(ValueError):
        spec.unflatten([*polys, Leaf(2)])


@pytest.mark.parametrize("count", [0, 2])
def test_replace_iterable_follows_positions_and_requires_exact_count(count):
    @dataclass
    class Box(PolyContainer):
        items: tuple[Poly, ...]

    polys = tuple(Leaf(i) for i in range(count))
    source = Box(polys * 2)
    assert source.replace(polys=source.polys()) is source
    replacements = tuple(Leaf(i + 10) for i in range(count * 2))
    result = source.replace(polys=iter(replacements))
    assert result.items == replacements
    if not count:
        assert result is source
    else:
        with pytest.raises(ValueError):
            source.replace(polys=iter(replacements[:-1]))
    with pytest.raises(ValueError):
        source.replace(polys=iter((*replacements, Leaf(99))))


def test_replace_combines_polys_and_root_fields():
    @dataclass
    class Record(PolyContainer):
        left: Poly
        right: Poly
        label: str
        optional: int | None
        metadata: object

        def __post_init__(self):
            if not self.label:
                raise ValueError("label must not be empty")

    leaf, replacement, metadata = Leaf(1), Leaf(2), object()
    source = Record(leaf, leaf, "original", None, metadata)

    replacements = [replacement, Leaf(3)]
    result = source.replace(polys=iter(replacements), label="updated", optional=7)
    assert result.polys() == replacements
    assert result.label == "updated"
    assert result.optional == 7
    assert result.metadata is metadata
    assert source.label == "original"
    assert source.optional is None
    assert source.polys() == [leaf, leaf]

    def callback(value: int) -> int:
        return value + 1

    for polys in (None, source.polys()):
        assigned = source.replace(polys=polys, metadata=callback, optional=0)
        assert assigned.metadata is callback and assigned.optional == 0
        assert assigned.left is leaf and assigned.right is leaf
        assert assigned.replace(optional=None).optional is None
        assert assigned.replace() is assigned
    assert source.metadata is metadata

    overridden = source.replace(polys=[replacement, replacement], left=leaf)
    assert overridden.left is leaf
    assert overridden.right is replacement
    with pytest.raises(ValueError, match="label must not be empty"):
        source.replace(polys=source.polys(), label="")
    with pytest.raises(TypeError, match="typo"):
        source.replace(polys=source.polys(), typo="updated")


def test_named_dataclass_replacement_accepts_initvars():
    @dataclass
    class Record(PolyContainer):
        p: Poly
        label: InitVar[str]
        derived: str = field(init=False)

        def __post_init__(self, label):
            self.derived = label

    source = Record(Leaf(1), "original")
    replacement = Leaf(2)
    for polys, expected in ((None, source.p), ([replacement], replacement)):
        result = source.replace(polys=polys, label="updated")
        assert result.p is expected
        assert result.derived == "updated" and source.derived == "original"


@pytest.mark.parametrize("collect", [PolyContainer.polys, lambda value: PolyContainer.flatten(value)[0]], ids=["iterate", "capture"])
def test_traversal_does_not_evaluate_computed_attributes(collect):
    class Computed:
        first: tuple[Poly, ...]
        second: tuple[Poly, ...]
        third: tuple[Poly, ...]
        fourth: tuple[Poly, ...]

        def __getattr__(self, name):
            raise AssertionError("computed attribute must not run")

    source = Computed()
    source.first = (Leaf(1),)
    assert collect(source) == list(source.first)


def test_capture_reads_storage_without_attribute_side_effects():
    root = {}
    first, second, later = Leaf(1), Leaf(2), Leaf(3)

    @dataclass
    class Head(PolyContainer):
        p: Poly

        def __getattribute__(self, name):
            if name == "p":
                root["late"] = later
            return object.__getattribute__(self, name)

    root.update(head=Head(first), tail=second)
    polys, spec = PolyContainer.flatten(root)
    assert list(root) == ["head", "tail"]
    rebuilt = spec.unflatten(polys)
    assert list(rebuilt) == ["head", "tail"]
    assert rebuilt["tail"] is second


@pytest.mark.parametrize("replacement", [lambda leaf: leaf, lambda leaf: Leaf(leaf.value + 1), lambda leaf: -leaf])
@pytest.mark.parametrize("aliased", [False, True])
@disable_gc()
def test_replace_releases_source_without_cyclic_gc(replacement, aliased):
    root = Pair(Leaf(1), Leaf(2))
    if aliased:
        root.right = root.left
    refs = [weakref.ref(root), *map(weakref.ref, root.polys())]
    mapped = root.replace(polys=map(replacement, root.polys()))
    del root, mapped
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize(
    "root",
    [
        ListSubclass(),
        TupleSubclass(),
        defaultdict[str, object](list, {"leaf": Leaf(1)}),
        ChainMap({"leaf": Leaf(1)}, {"other": Leaf(2)}),
    ],
)
def test_capture_rejects_unsupported_container_subclasses(root):
    with pytest.raises(TypeError, match=type(root).__name__):
        PolyContainer.flatten(root)


def test_capture_rejects_non_orderable_dict_keys():
    with pytest.raises(TypeError):
        PolyContainer.flatten({"leaf": Leaf(1), 1: Leaf(2)})


@pytest.mark.parametrize("wrap_key", [lambda p: p, lambda p: (p,)])
def test_mapping_keys_cannot_hide_polys(wrap_key):
    @dataclass
    class Box(PolyContainer):
        items: Mapping[object, int]

    value = Box({wrap_key(Leaf(1)): 0})
    for operation in (PolyContainer.polys, PolyContainer.flatten, lambda v: v.replace(polys=())):
        with pytest.raises(TypeError, match="mapping keys cannot contain Polys"):
            operation(value)


@pytest.mark.parametrize("kind", ["dataclass", "attrs"])
def test_stored_non_init_polys_are_captured_and_restored(kind):
    calls = []

    if kind == "dataclass":

        @dataclass(frozen=True, slots=True)
        class DataclassBox(PolyContainer):
            p: Poly
            derived: Poly = field(init=False)

            def __post_init__(self):
                calls.append(self)
                object.__setattr__(self, "derived", -self.p)

        box = DataclassBox
    else:

        @define(frozen=True)
        class AttrsBox:
            p: Poly
            derived: Poly = attrs_field(init=False)

            def __attrs_post_init__(self):
                calls.append(self)
                object.__setattr__(self, "derived", -self.p)

        box = AttrsBox

    original = Poly.uniform(large_prime_ring(4))
    source = box(original)
    polys, spec = PolyContainer.flatten(source)
    assert polys == [source.p, source.derived]
    replacements = [original.ntt(), source.derived.ntt()]
    result = spec.unflatten(replacements)
    assert result.p is replacements[0]
    assert result.derived is replacements[1]
    if isinstance(source, PolyContainer):
        result = source.replace(polys=replacements)
        assert result.p is replacements[0] and result.derived is replacements[1]
        with pytest.raises(TypeError, match="init=False"):
            source.replace(derived=original)
    assert len(calls) == 1


def test_attrs_traversal_uses_stored_fields_not_annotations_or_aliases():
    Box = make_class("Box", {"_p": attrs_field(type=Poly, alias="payload"), "other": attrs_field()})
    original = Poly.uniform(large_prime_ring(4))
    replacement = -original
    source = Box(original, original)
    polys, spec = PolyContainer.flatten(source)
    assert polys == PolyContainer.polys(source) == [original, original]
    result = spec.unflatten([replacement, original])
    assert result._p is replacement and result.other is original
    assert source._p is source.other is original


@pytest.mark.parametrize("array", [np.array([1.0, 2.0]), np.array([acb(1)], dtype=object), np.array([(acb(1),)], dtype=[("x", object)])])
def test_numpy_metadata_is_preserved(array):
    @dataclass
    class Box:
        p: Poly
        values: np.ndarray

    original = Leaf(1)
    polys, spec = PolyContainer.flatten(Box(original, array))
    assert polys == [original]
    replacement = Leaf(2)
    result = spec.unflatten([replacement])
    assert result.p is replacement and result.values is array


@pytest.mark.parametrize("dtype", [object, [("payload", object)], [("outer", [("payload", object)])]])
@pytest.mark.parametrize("shape", [(), (2, 2)])
def test_object_arrays_cannot_hide_nested_polys(dtype, shape):
    @dataclass
    class Box(PolyContainer):
        values: np.ndarray

    array = np.empty(shape, dtype=dtype)
    payload: np.ndarray = array
    while payload.dtype.names:
        payload = payload[payload.dtype.names[0]]
    payload.flat[-1] = {"nested": [Leaf(1)]}
    for value in (array, Box(array)):
        for operation in (PolyContainer.polys, PolyContainer.flatten):
            with pytest.raises(TypeError, match="only contain scalars"):
                operation(value)
    with pytest.raises(TypeError, match="only contain scalars"):
        Box(array).replace(polys=())


@pytest.mark.parametrize("contents", [lambda array, shared: array, lambda array, shared: [1, 2], lambda array, shared: shared])
def test_object_arrays_reject_non_scalar_contents(contents):
    shared = [Leaf(1)]
    array = np.empty(1, dtype=object)
    array[0] = contents(array, shared)
    for operation in (PolyContainer.polys, PolyContainer.flatten, lambda v: Pair(v, "metadata").replace(polys=[shared[0]])):
        with pytest.raises(TypeError, match="only contain scalars"):
            operation((shared, array))
