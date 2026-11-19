from ren.dtype import dtypes
from ren.engine.pattern import Gate, pat, rewrite, rules
from ren.ir import Op
from ren.ir.node import Node
from ren.ir.op import GroupOp

from .simplify import algebraic


def gidx_offset(node: Node, n: int) -> int | None:
    """Constant offset in GIDX + offset, aligned to an RNS limb."""
    if node.dtype is not dtypes.index:
        return None

    offset = 0
    if node.op is Op.ADD and node.src[1].const_val is not None and node.src[1].dtype is dtypes.index:
        node, offset = node.src[0], node.src[1].const_val
    if node.op is not Op.GIDX or node.dtype is not dtypes.index or node.arg not in (None, 0):
        return None
    return offset if offset >= 0 and offset % n == 0 else None


def _is_buffer_source(n: Node) -> bool:
    return n.base.op in GroupOp.Defines or n.base.op is Op.BUFFER_VIEW


def is_conversion_store_value(value: Node) -> bool:
    base = value.base
    # View-only conversions over compute stay fused; real conversion boundaries
    # become standalone kernels.
    return base.op is Op.BUFFER_VIEW or (base.op in GroupOp.Conversion and (base.view_offset is None or base.has_buffer_identity))


def _index_store(target: Node, value: Node) -> Node | None:
    if not _is_buffer_source(target) or is_conversion_store_value(value):
        return None
    idx = Node(Op.GIDX, dtypes.index)
    return target.index(idx).store(_read(value))


def _read(source: Node) -> Node:
    return source.index(Node(Op.GIDX, dtypes.index)).load() if _is_buffer_source(source) else source


def _index_view(view: Node, value: Node) -> Node | None:
    # TODO: Lower quotient projections to explicit index mappings before this pass,
    # shared by computed expressions and storage views.
    offset = view.view_offset
    if offset is None:
        return None
    if offset:
        # Shift this consumer's reads; AFTER and STORE end the traversal.
        shift_reads = rules([
            (pat.source.index(pat.idx).load(), lambda source, idx: Gate(source.index(idx + offset).load())),
            (pat.n({Op.AFTER, Op.STORE}), lambda n: Gate(n)),
        ])
        value = rewrite(value, shift_reads, bottom_up=True)
    assert view.ring is not None and view.domain is not None, f"view semantics missing ring/domain for {view.op}"
    return value.as_ring(view.ring, view.domain)


# Elementwise ops read storage at GIDX; quotient views translate those reads into
# the source's index space. AFTER and buffer views remain storage boundaries.
indexify = rules([
    (pat.n(dtype=dtypes.index), lambda n: algebraic.match(n)),
    # Each elementwise STORE starts in its own output index space. Conversion
    # kernels keep whole-buffer arguments, and dependency stores are indexed separately.
    (pat(Op.STORE, (pat.target, pat.value)), _index_store),
    (pat.view(Op.TO_RING, pat.value), _index_view),
    (pat.node(GroupOp.Elementwise), lambda node: node.replace(src=tuple(_read(source) for source in node.src))),
])
