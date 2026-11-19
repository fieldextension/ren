"""Fuse independent pointwise dots that share inputs."""

from collections import Counter, defaultdict
from collections.abc import Iterator
from functools import cache

from ren.context import DOT_TILE
from ren.dtype import dtypes
from ren.engine.pattern import RuleSet, pat, rules
from ren.ir import Op
from ren.ir.node import Node
from ren.ring import Domain

from .indexing import gidx_offset
from .kernelize import add_buffer


_load = pat(Op.PARAM).index().load()
_constant = pat(Op.MOD_REDUCE, (pat(Op.CONST),), dtype=dtypes.uints)

_product = rules([
    (_load * _load, lambda: True),
    (_load * _constant, lambda: True),
    (_constant * _constant, lambda: True),
])


# Operations return their operands; supported leaves return an empty tuple.
_operands = rules([
    (pat.op({Op.ADD, Op.MUL}, dtype=dtypes.uints), lambda op: op.src),
    (pat.op(Op.MOD_REDUCE, (pat({Op.ADD, Op.MUL, Op.LOAD, Op.MOD_REDUCE, Op.CAST}),), dtype=dtypes.uints), lambda op: op.src),
    (_load, lambda: ()),
    (_constant, lambda: ()),
    (pat(Op.CONST), lambda: ()),
    (pat(Op.CAST, (pat(Op.CONST),), dtype=dtypes.uints), lambda: ()),
])


def dot_steps(value: Node) -> list[Node] | None:
    """Visit shared expressions once in dependency order, preserving prefixes and reductions."""
    steps = []
    for n in value.toposort(gate=lambda n: n.op not in {Op.INDEX, Op.DEVICE}):
        if (operands := _operands.match(n)) is None:
            return None
        if operands:
            steps.append(n)
    return steps


def _dot_inputs(body: Node, value: Node):
    """Return the common limb offset and input slots of a sum of products."""
    info = body.arg
    if info is None or not info.device.startswith("cuda") or info.domain is not Domain.NTT:
        return None
    if (steps := dot_steps(value)) is None:
        return None
    if sum(_product.match(n) is True for n in steps) < 2:
        return None
    addresses = {load.src[0].src for step in steps for load in step.src if load.op is Op.LOAD}
    offsets = {gidx_offset(idx, info.ring.n) for _, idx in addresses}
    if len(offsets) != 1 or None in offsets:
        return None
    return info, offsets.pop(), frozenset(param.arg.slot for param, _ in addresses)


_dot_body = rules([(pat.body.sink(pat().index(pat(Op.GIDX)).store(pat.value)), _dot_inputs)])


def _dot_tiles(root: Node, tile_size: int) -> Iterator[list[Node]]:
    """Group dots at the same dependency depth that read from common inputs."""
    depths: dict[Node, int] = {}
    buckets = defaultdict(list)
    # Calls can share a body; its parameter slots and offsets do not change.
    match_body = cache(_dot_body.match)
    for n in root.toposort():
        depths[n] = max((depths[s] for s in n.inputs), default=0) + (n.op is Op.CALL)
        if n.op is not Op.CALL or (matched := match_body(n.src[0])) is None:
            continue
        info, offset, slots = matched
        args = n.inputs
        buffers = frozenset(args[slot].base for slot in slots)
        buckets[info, depths[n], offset].append((n, buffers))

    for bucket in buckets.values():
        calls, inputs = zip(*bucket)
        readers = defaultdict(set)
        for i, buffers in enumerate(inputs):
            for buf in buffers:
                readers[buf].add(i)
        remaining = dict(enumerate(inputs))
        for seed in range(len(calls)):
            if (buffers := remaining.pop(seed, None)) is None:
                continue
            tile, buffers = [seed], set(buffers)
            while len(tile) < tile_size:
                scores = Counter(i for buf in buffers for i in readers[buf] if i in remaining)
                if not scores:
                    break
                best = max(scores, key=lambda i: (scores[i], -i))
                tile.append(best)
                buffers.update(remaining.pop(best))
            if len(tile) > 1:
                yield [calls[i] for i in sorted(tile)]


def _merge(tile: list[Node]) -> Node:
    """Combine stores using one parameter per buffer, keeping all call dependencies."""
    bindings: dict[Node, tuple[Node, Node]] = {}
    stores = []
    for call in tile:
        slots = [add_buffer(bindings, dep) for dep in call.inputs]
        (store,) = call.src[0].src
        stores.append(store.map(lambda n: slots[n.arg.slot] if n.op is Op.PARAM else None))
    return tile[0].src[0].replace(src=tuple(stores)).call(*(dep for _, dep in bindings.values()))


def dot_fusion(root: Node) -> RuleSet | None:
    """Replace every CALL in a dot tile with the same fused CALL before buffer binding."""
    if (tile_size := DOT_TILE.value) <= 1:
        return None
    replacements = {}
    for tile in _dot_tiles(root, tile_size):
        replacements.update(dict.fromkeys(tile, _merge(tile)))
    return rules([(pat.n(Op.CALL), lambda n: replacements.get(n))]) if replacements else None
