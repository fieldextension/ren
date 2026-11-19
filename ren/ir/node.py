# Parts of Ren's IR node implementation are adapted from tinygrad's UOp.
from dataclasses import dataclass
from typing import Any, Callable, cast
from functools import cached_property, total_ordering
from math import prod
from collections import Counter
from collections.abc import Collection
import itertools
import operator

from ren.context import DEBUG
from ren.dtype import DType, dtypes
from ren.buffer import Buffer, BufferSpec
from ren.device import Device
from ren.polydist import PolyDist
from ren.random import ChaChaRng
from ren.utils import CacheMeta, find_contiguous_subset
from ren.ring import RingSpec, Domain, Residue
from ren.mixin import ScalarMixin

from .op import Op, GroupOp


python_alu: dict[Op, Callable] = {
    Op.CMPEQ: operator.eq,
    Op.CMPLT: operator.lt,
    Op.MOD: operator.mod,
    Op.AND: operator.and_,
    Op.SHL: operator.lshift,
    Op.SHR: operator.rshift,
    Op.IDIV: operator.floordiv,
    Op.ADD: operator.add,
    Op.SUB: operator.sub,
    Op.MUL: operator.mul,
    Op.NEG: operator.neg,
}


class recursive_property[TNode: Node, TValue]:
    """Cached property evaluated in graph postorder including callable bodies."""

    def __init__(self, fxn: Callable[[TNode], TValue]):
        self.fxn = fxn
        self.__doc__ = fxn.__doc__
        self.attrname: str

    def __set_name__(self, owner, name):
        self.attrname = name

    def __get__(self, node: TNode, _=None) -> TValue:
        if self.attrname in node.__dict__:
            return node.__dict__[self.attrname]

        stack: list[tuple[Node, bool]] = [(node, False)]
        while stack:
            current, visited = stack.pop()
            if self.attrname in current.__dict__:
                continue
            if not visited:
                stack.append((current, True))
                for source in reversed(current.src):
                    stack.append((source, False))
            else:
                setattr(current, self.attrname, self.fxn(cast(TNode, current)))
        return node.__dict__[self.attrname]


@dataclass(frozen=True, slots=True)
class BufferizeArg:
    """Argument for BUFFERIZE op - captures materialization metadata."""

    device: str | None
    ring: RingSpec | None


@dataclass(frozen=True, slots=True, order=True)
class ParamArg:
    """Storage identity, extent, device, and optional backing buffer."""

    slot: int
    size: int
    device: str
    buffer: Buffer | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("storage size must be positive")


@dataclass(frozen=True, slots=True, order=True)
class ToRingArg:
    """Argument for TO_RING op - destination ring plus residue policy."""

    ring: RingSpec
    residue: Residue = Residue.BALANCED


@dataclass(frozen=True, slots=True, order=True)
class ToRnsArg:
    """Argument for TO_RNS op - destination ring plus signed-limb width of the source."""

    ring: RingSpec
    limb_count: int


@total_ordering
@dataclass(frozen=True, slots=True)
class RandomArg:
    """Argument for RANDOM op"""

    dist: PolyDist
    stream_id: int

    def _order_tuple(self):
        return (id(self.dist), self.stream_id)

    def __eq__(self, other):
        if not isinstance(other, RandomArg):
            return NotImplemented
        return self._order_tuple() == other._order_tuple()

    def __lt__(self, other):
        if not isinstance(other, RandomArg):
            return NotImplemented
        return self._order_tuple() < other._order_tuple()


def from_rns_limb_count(ring: RingSpec) -> int:
    """u32 words per coefficient in a FROM_RNS reconstruction of ``ring``."""
    # TODO: derive limb width from ring.dtype once dtype tracks limb representation.
    return (ring.q.bit_length() + 31) // 32


@dataclass(slots=True, eq=False, weakref_slot=True)
class Node(ScalarMixin, metaclass=CacheMeta):
    op: Op
    dtype: DType = dtypes.void
    src: tuple[Node, ...] = ()
    arg: Any = None

    unique_num = itertools.count(0)

    def __reduce__(self):
        args = [self.op, self.dtype, self.src, self.arg]
        return Node, tuple(args)

    @staticmethod
    def __cache_key__(op: Op, dtype=dtypes.void, src: tuple[Node, ...] = (), arg=None) -> tuple:
        return (op, dtype, src, arg)

    @staticmethod
    def const(val: int, dtype: DType = dtypes.weakint) -> Node:
        """Create an exact integer. Passing a dtype casts it to that width.
        A polynomial can keep the exact value until lowering reduces it modulo each prime.
        """
        return Node(Op.CONST, dtypes.weakint, arg=operator.index(val)).cast(dtype)

    @staticmethod
    def random(dist: PolyDist, ring: RingSpec, rng: ChaChaRng) -> Node:
        from ren.engine.realize import capturing

        if capturing:
            raise RuntimeError("Random number generation during jit capture is not supported")
        if dist.n != ring.n:
            raise ValueError(f"Ring degree mismatch: {ring.n} != {dist.n}")
        stream_id = rng.reserve()
        arg = RandomArg(dist=dist, stream_id=stream_id)
        return Node(Op.RANDOM, dtype=ring.dtype, src=(rng._key_node,), arg=arg).as_ring(ring, Domain.COEFF)

    @staticmethod
    def new_buffer(device: str, size: int, dtype: DType) -> Node:
        return Node.from_buffer(Buffer(BufferSpec(dtype, size, device)))

    @staticmethod
    def alloc(device: str, size: int, dtype: DType) -> Node:
        return Node(Op.ALLOC, dtype, arg=ParamArg(next(Node.unique_num), size, device))

    @staticmethod
    def param(slot: int, source: Node) -> Node:
        if slot < 0:
            raise ValueError("parameter slot must be non-negative")
        return Node(Op.PARAM, source.dtype, arg=ParamArg(slot, source._storage_size(), source.device))

    def call(self, *inputs: Node) -> Node:
        """Call a body with supplied input and output storage."""
        return Node(Op.CALL, src=(self, *inputs))

    @staticmethod
    def from_buffer(buf: Buffer) -> Node:
        """Create a fresh BUFFER node owning the supplied Buffer."""
        return Node._uncached(Op.BUFFER, buf.dtype, (), ParamArg(next(Node.unique_num), buf.size, buf.device, buf))

    def const_like(self, val: int) -> Node:
        if (ring := self.ring) is not None:
            # Domain injection resolves ambiguous polynomial representations.
            return Node.const(val).as_ring(ring, self.domain or Domain.COEFF)
        return Node.const(val, self.dtype)

    def as_ring(self, ring: RingSpec, domain: Domain | None) -> Node:
        """Attach polynomial ring/domain semantics and the ring's coefficient dtype."""
        if self.op is Op.AS_RING:
            return self.replace(arg=(ring, domain))
        dtype = ring.dtype if self.dtype is dtypes.weakint else self.dtype
        if self.op is Op.CAST and self.src[0].op is Op.CONST:
            dtype = ring.dtype
        return Node(Op.AS_RING, dtype, src=(self,), arg=(ring, domain))

    @staticmethod
    def sink(*src: Node) -> Node:
        """Create a SINK node aggregating one or more outputs."""
        return Node(Op.SINK, dtypes.void, src=tuple(src))

    @recursive_property
    def _device(self) -> str | None:
        if self.op is Op.DEVICE:
            return self.arg
        if self.op in GroupOp.Defines or self.op is Op.BUFFERIZE:
            return self.arg.device
        if self.op in {Op.SINK, Op.AFTER}:
            return self.src[0]._device
        if self.op is Op.COPY:
            return self.src[1]._device
        for x in self.src:
            if (dev := x._device) is not None:
                return dev
        return None

    @property
    def device(self) -> str:
        dev = self._device
        if dev is None:
            raise RuntimeError(f"Node {self.op} has no device")
        return dev

    @property
    def size(self) -> int:
        if self.op in GroupOp.Defines:
            return self.arg.size

        raise RuntimeError(f"size is only defined for BUFFER, ALLOC and PARAM nodes, not {self.op}")

    def _storage_size(self, ring: RingSpec | None = None) -> int:
        """Element count for this node's materialized buffer."""
        if self.op in GroupOp.Defines:
            return self.size
        if self.op is Op.BUFFER_VIEW:
            return self.arg[0]
        if self.op in {Op.AS_RING, Op.COPY, Op.CONTIGUOUS, Op.AFTER, Op.BUFFERIZE} and self.src:
            return self.src[0]._storage_size(self.arg[0] if self.op is Op.AS_RING else ring)
        ring = self.ring or ring
        if self.op is Op.FROM_RNS:
            assert ring is not None
            return ring.n * from_rns_limb_count(ring)
        if ring is not None:
            return ring.storage_size
        raise RuntimeError(f"cannot infer flat storage size for {self.op}")

    @recursive_property
    def ring(self) -> RingSpec | None:
        if self.op is Op.AS_RING:
            return self.arg[0]
        if self.op is Op.AFTER:
            return self.src[0].ring
        if self.op is Op.BUFFERIZE:
            return self.arg.ring
        if self.op is Op.TO_RING:
            return self.arg.ring
        if self.op in {Op.RESCALE, Op.MOD_REDUCE}:
            return self.arg
        if self.op is Op.TO_RNS:
            return self.arg.ring
        if self.op in {Op.CALL, Op.LINEAR, Op.SINK}:
            return None

        rings = [s.ring for s in self.src if s.ring is not None]
        if not rings:
            return None
        if not all(r == rings[0] for r in rings):
            raise RuntimeError(f"ring mismatch on {self.op}: {rings}")
        return rings[0]

    @property
    def domain(self) -> Domain | None:
        return self._domain

    @recursive_property
    def _domain(self) -> Domain | None:
        """Compute domain by propagation from sources.

        Only explicit domain markers (AS_RING, NTT, INTT) set domain directly.
        All other ops propagate domain from their sources.
        Domain requirements (what domain an op needs its inputs in) are handled
        by domain injection during scheduling, not here.
        """
        # Explicit domain markers
        if self.op is Op.AS_RING:
            return self.arg[1]
        if self.op is Op.AFTER:
            return self.src[0]._domain
        if self.op is Op.NTT:
            return Domain.NTT
        if self.op is Op.INTT:
            return Domain.COEFF
        if self.op is Op.TO_RNS:
            return Domain.COEFF
        if self.op is Op.FROM_RNS:
            return Domain.COEFF
        if self.op in {Op.CALL, Op.LINEAR, Op.SINK}:
            return None

        # Propagate from sources
        doms = [s._domain for s in self.src if s._domain is not None]
        if not doms:
            return None
        if not all(d == doms[0] for d in doms):
            # Ambiguous domain - let domain injection resolve it
            return None
        return doms[0]

    @staticmethod
    def coeffs_to_rns(source: Node, ring: RingSpec, limb_count: int) -> Node:
        if source.ring is not None:
            raise RuntimeError(f"COEFFS_TO_RNS source must be raw coefficient storage, got ring node {source.op}")

        # TODO: also accept uint64 (and other widths) once limb representation is tracked in dtype.
        if source.dtype is not dtypes.uint32:
            raise TypeError(f"COEFFS_TO_RNS source must use uint32 limbs, got {source.dtype}")
        if limb_count <= 0:
            raise ValueError(f"COEFFS_TO_RNS limb_count must be positive, got {limb_count}")
        return Node(Op.TO_RNS, ring.dtype, src=(source,), arg=ToRnsArg(ring=ring, limb_count=limb_count))

    def from_rns(self) -> Node:
        """Reconstruct integer coefficients from RNS limbs as raw uint32 limbs.

        Output is the balanced [-Q/2, Q/2) representative, encoded as signed
        two's-complement packed little-endian into ``limb_count = ceil(bitlen(Q)/32)``
        uint32 words -- the same signed-limb encoding TO_RNS consumes. Encoding
        this result into the same ring preserves coefficient-domain residues;
        reconstructing an arbitrary encoded integer instead selects its balanced
        representative and may change its storage width.
        """
        if self.ring is None:
            raise RuntimeError(f"Cannot reconstruct coefficients from node without ring semantics: {self.op}")
        return Node(Op.FROM_RNS, dtypes.uint32, src=(self,))

    def to_ring(self, ring: RingSpec, *, residue: Residue = Residue.BALANCED) -> Node:
        src_ring = self.ring
        if src_ring is None:
            raise RuntimeError(f"Cannot convert node without ring semantics: {self.op}")
        if ring == src_ring:
            return self
        if ring.n != src_ring.n:
            raise ValueError(f"ring degree mismatch: {src_ring.n} vs {ring.n}")
        if ring.dtype is not src_ring.dtype:
            raise ValueError(f"ring dtype mismatch: {src_ring.dtype} vs {ring.dtype}")
        return Node(Op.TO_RING, dtype=self.dtype, src=(self,), arg=ToRingArg(ring, residue))

    def mod_reduce(self, ring: RingSpec | None) -> Node:
        target = ring or self.ring
        return Node(Op.MOD_REDUCE, target.dtype if target is not None else self.dtype, src=(self,), arg=target)

    def _binop(self, op: Op, other: Node | int, reverse: bool = False) -> Node:
        if not isinstance(other, Node):
            # Algebraic literals inherit the ring; scalar literals take only the dtype.
            other = self.const_like(other) if op in {Op.ADD, Op.SUB, Op.MUL} else Node.const(other, self.dtype)
        lhs, rhs = (other, self) if reverse else (self, other)
        return Node(op, dtype=dtypes.bool if op in {Op.CMPEQ, Op.CMPLT} else lhs.dtype, src=(lhs, rhs))

    def _unary(self, op: Op, *, arg: Any = None) -> Node:
        return Node(op, dtype=self.dtype, src=(self,), arg=arg)

    def cast(self, dtype: DType) -> Node:
        """Integer casts wrap to the target width; bool casts test for nonzero."""
        return self if dtype is self.dtype else Node(Op.CAST, dtype, src=(self,))

    def __bool__(self) -> bool:
        if self.dtype is not dtypes.bool:
            raise TypeError("Only boolean nodes can be converted to bool")
        if (value := self.const_val) is None:
            raise ValueError("Cannot convert a non-constant boolean node to bool")
        return bool(value)

    def logical_not(self) -> Node:
        """Negate a boolean expression."""
        assert self.dtype is dtypes.bool, "logical_not requires a boolean node"
        return self.eq(False)

    def where(self, yes: Node | int, no: Node | int) -> Node:
        """Select yes where this boolean node is true, else no."""
        if not isinstance(yes, Node):
            assert isinstance(no, Node), "where needs at least one typed branch"
            yes = Node.const(yes, no.dtype)
        if not isinstance(no, Node):
            no = Node.const(no, yes.dtype)
        return Node(Op.WHERE, yes.dtype, src=(self, yes, no))

    # Kernel-level ops for lowering
    def index(self, idx: Node) -> Node:
        """Create INDEX node: self[idx]."""
        return Node(Op.INDEX, self.dtype, src=(self, idx))

    def load(self) -> Node:
        """Create LOAD from this INDEX."""
        return Node(Op.LOAD, self.dtype, src=(self,))

    def store(self, value: Node) -> Node:
        """Create STORE: store value at this location."""
        return Node(Op.STORE, value.dtype, src=(self, value))

    @cached_property
    def buffer(self) -> Buffer:
        if self is not self.base:
            return self.src[0].buffer
        if self.op is Op.BUFFER_VIEW:
            return self.src[0].buffer.view(*self.arg)
        assert self.op is Op.BUFFER, f"buffer property only valid on BUFFER nodes, got {self.op}"
        assert self.arg.buffer is not None
        return self.arg.buffer

    @property
    def realized(self) -> Buffer | None:
        """Return the Buffer if allocated, else None."""
        if self.op not in {Op.BUFFER, Op.BUFFER_VIEW}:
            return None
        buf = self.buffer
        return buf if buf.is_allocated() else None

    def is_realized(self) -> bool:
        """Whether the value is ready to read from an allocated buffer."""
        value = self.src[0] if self.op is Op.AS_RING else self
        return value.realized is not None

    @cached_property
    def view_offset(self) -> int | None:
        """Element offset for view ops. None if not a view."""
        # Views require a source buffer - constants have no buffer to view
        if self is not self.base:
            return self.base.view_offset
        if self.op is Op.CONST:
            return None
        if self.op is Op.TO_RING:
            src_ring, dst_ring = self.src[0].ring, self.arg.ring
            if src_ring is None:
                if DEBUG >= 4:
                    print("  [view_offset] TO_RING: src has no ring")
                return None
            start = find_contiguous_subset(src_ring.moduli, dst_ring.moduli)
            if start is None:
                if DEBUG >= 4:
                    # Log why this TO_RING can't be a view
                    src_n = len(src_ring.moduli)
                    dst_n = len(dst_ring.moduli)
                    # Check if moduli are compatible but just not contiguous
                    dst_set = set(dst_ring.moduli)
                    src_set = set(src_ring.moduli)
                    if dst_set <= src_set:
                        reason = "non-contiguous subset"
                        # Show which indices in src contain the dst primes
                        dst_indices = [i for i, p in enumerate(src_ring.moduli) if p in dst_set]
                        print(f"  [view_offset] TO_RING not view: {src_n} → {dst_n} primes ({reason})")
                        print(f"    dst primes at src indices: {dst_indices}")
                    elif not dst_set & src_set:
                        reason = "disjoint moduli"
                        print(f"  [view_offset] TO_RING not view: {src_n} → {dst_n} primes ({reason})")
                    else:
                        reason = "partial overlap"
                        common = len(dst_set & src_set)
                        new_primes = len(dst_set - src_set)
                        print(f"  [view_offset] TO_RING not view: {src_n} → {dst_n} primes ({reason}: {common} common, {new_primes} new)")
                return None
            return start * src_ring.n
        return None

    @cached_property
    def has_buffer_identity(self) -> bool:
        """True for storage references, including allocations and parameters, directly or through views."""
        n = self.base
        while n.op is Op.BUFFER_VIEW or n.view_offset is not None:
            n = n.src[0].base
        return n.op in GroupOp.Defines

    def copy_to_device(self, device: str):
        return Node(Op.COPY, self.dtype, src=(self, Node(Op.DEVICE, arg=device)))

    def empty_like(self, device: str | None = None) -> Node:
        out = Node.alloc(Device.canonicalize(device or self._device), size=self._storage_size(), dtype=self.dtype)
        if self.ring is None:
            return out
        return out.as_ring(self.ring, self.domain)

    def clone(self, device: str | None = None) -> Node:
        ret = self.empty_like(device)
        source = self.src[0] if self.op is Op.CONTIGUOUS else self
        if source._device is not None and source.device != ret.device:
            source = source.copy_to_device(ret.device)
        return ret.after(ret.store(source))

    def replace(self, **kwargs) -> Node:
        """Create a new node with selectively replaced properties.

        Any property not specified in kwargs is preserved from self. Returns self
        when nothing changes, so identity survives for uncached nodes too.
        """
        op, dtype, src, arg = kwargs.get("op", self.op), kwargs.get("dtype", self.dtype), kwargs.get("src", self.src), kwargs.get("arg", self.arg)
        if op is self.op and dtype is self.dtype and src == self.src and (arg is self.arg or arg == self.arg):
            return self
        return Node(op, dtype, src, arg)

    @property
    def inputs(self) -> tuple[Node, ...]:
        """Inputs in this scope, excluding a callable's body."""
        return self.src[1:] if self.op is Op.CALL else self.src

    def toposort(self, *, gate: Callable[[Node], bool] | None = None, enter_calls: bool = False) -> Collection[Node]:
        """Visit dependencies once in this scope; a gate can stop at known subgraphs."""
        order: dict[Node, None] = {}
        stack: list[tuple[Node, bool]] = [(self, False)]

        while stack:
            n, seen = stack.pop()
            if n in order or (gate is not None and not gate(n)):
                continue
            if not seen:
                stack.append((n, True))
                for s in reversed(n.src if enter_calls else n.inputs):
                    stack.append((s, False))
            else:
                order[n] = None

        return order

    @recursive_property
    def tuplize(self) -> tuple:
        # An unresolved domain sorts before either concrete domain.
        arg = (self.arg[0], self.arg[1] or 0) if self.op is Op.AS_RING else self.arg
        return (self.op.value, arg, self.dtype) + tuple(s.tuplize for s in self.src)

    def map(self, mapper: Callable[[Node], Node | None], *, enter_calls: bool = False) -> Node:
        """Map nodes once, skipping replaced subgraphs and opaque callable bodies."""
        replacements: dict[Node, Node] = {}
        stack: list[tuple[Node, bool]] = [(self, False)]
        while stack:
            node, seen = stack.pop()
            if node in replacements:
                continue
            sources = node.src if enter_calls else node.inputs
            if not seen:
                if (replacement := mapper(node)) is not None:
                    replacements[node] = replacement
                    continue
                stack.append((node, True))
                stack.extend((source, False) for source in reversed(sources))
                continue
            src = tuple(replacements[source] for source in sources)
            if not enter_calls and node.op is Op.CALL:
                src = (node.src[0], *src)
            replacements[node] = node if src == node.src else node.replace(src=src)
        return replacements[self]

    def substitute(self, mapping: dict[Node, Node], *, enter_calls: bool = False) -> Node:
        """Replace nodes, leaving callable bodies opaque by default."""
        return self if not mapping else self.map(mapping.get, enter_calls=enter_calls)

    @property
    def base(self) -> Node:
        """Get the underlying storage/computation node, unwrapping view ops."""
        if self.op in GroupOp.Metadata:
            return self.src[0].base
        return self

    @recursive_property
    def const_val(self) -> int | None:
        if self.op is Op.CONST:
            return self.arg
        if self.op is Op.LOAD and self.src[0].op is Op.INDEX:
            table, index = self.src[0].src
            if table.op is Op.CONST_ARRAY and (i := index.const_val) is not None and 0 <= i < len(table.arg):
                return table.arg[i]
        if self.op is Op.CAST and (value := self.src[0].const_val) is not None:
            return self.dtype.const(value)
        if self.op in GroupOp.Metadata:
            return self.src[0].const_val
        if self.op in {Op.COPY, Op.CONTIGUOUS, Op.NTT, Op.INTT, Op.AUTOMORPHISM}:
            return self.src[0].const_val
        if self.op is Op.TO_RING:
            val, src_ring = self.src[0].const_val, self.src[0].ring
            if val is None or src_ring is None:
                return None
            # Projections preserve residues; lifting first selects a source-ring representative.
            if not self.arg.ring.is_quotient_of(src_ring):
                val %= src_ring.q
                if self.arg.residue is Residue.BALANCED and val > src_ring.q // 2:
                    val -= src_ring.q
            return val
        if self.op in python_alu:
            operands = []
            for source in self.src:
                if (value := source.const_val) is None:
                    return None
                operands.append(value)
            ring = self.ring
            if ring is not None and self.op not in {Op.ADD, Op.SUB, Op.MUL, Op.IDIV, Op.NEG}:
                return None
            try:
                value = python_alu[self.op](*operands)
            except ZeroDivisionError:
                if self.op is Op.IDIV and self.dtype is dtypes.weakint:
                    raise
                return None  # A zero divisor can occur in an unused WHERE branch.
            return value if ring is not None else self.dtype.const(value)
        if self.op is Op.MOD_REDUCE:
            ring = self.ring
            if ring is None:
                return None
            val = self.src[0].const_val
            return val % ring.q if val is not None else None
        if self.op is Op.RESCALE:
            val = self.src[0].const_val
            if val is None:
                return None
            src_ring = self.src[0].ring
            dst_ring = self.ring
            if src_ring is None or dst_ring is None:
                return None
            if val == 0:
                return 0
            src_moduli, dst_moduli = set(src_ring.moduli), set(dst_ring.moduli)
            removed_prod = prod(p for p in src_ring.moduli if p not in dst_moduli)
            added_prod = prod(p for p in dst_ring.moduli if p not in src_moduli)
            # Scale by the added basis, then divide by the removed basis with
            # centered rounding, preserving this transition's rounding step.
            scaled = val * added_prod
            centered = scaled % removed_prod
            if centered > removed_prod // 2:
                centered -= removed_prod
            return (scaled - centered) // removed_prod
        return None

    def bufferize(self, device: str | None = None) -> Node:
        """Create a BUFFERIZE node capturing this node's materialization metadata."""
        ring = self.ring
        arg = BufferizeArg(device=device or self._device, ring=ring)
        return Node(Op.BUFFERIZE, self.dtype, src=(self,), arg=arg)

    def contiguous(self) -> Node:
        """Ensure this node has contiguous buffer access."""
        if self._device is None or self.op is Op.CONTIGUOUS or self.has_buffer_identity:
            return self
        return Node(Op.CONTIGUOUS, self.dtype, src=(self,))

    def after(self, *deps: Node) -> Node:
        """Create an AFTER node: use this value after dependencies complete.

        AFTER(value, *deps) passes through `value` but establishes ordering
        dependencies on `deps`. Used for:
        - Buffer dependencies: buffer.after(store) means "use buffer after store"
        - Parallel execution: independent ops without AFTER can run in parallel
        - Correct scheduling ordering on toposort
        """
        if not deps:
            return self
        return Node(Op.AFTER, self.dtype, src=(self, *deps))

    def _format_src_refs(self, node_map: dict[Node, int]) -> str:
        if not self.src:
            return ""
        return "[" + ", ".join(str(node_map[s]) for s in self.src) + "]"

    def _format_arg(self) -> str:
        if self.op is Op.CONST:
            return _format_int(self.arg)
        if self.op in GroupOp.Defines:
            return f"{self.arg.slot}: {self.arg.device}[{self.arg.size}]"
        if self.op in {Op.GIDX, Op.DEVICE}:
            return str(self.arg)
        if self.op is Op.BUFFER_VIEW:
            size, offset = self.arg
            return f"size={size}, offset={offset}"
        if self.op is Op.AS_RING:
            ring, domain = self.arg
            domain_str = "ntt" if domain == Domain.NTT else "coeff"
            return f"{domain_str} {ring}"
        if self.op in {Op.ROTATE, Op.AUTOMORPHISM}:
            return f"k={self.arg}"
        if self.op in {Op.RESCALE, Op.TO_RING}:
            dst_ring = self.arg.ring if self.op is Op.TO_RING else self.arg
            src_ring = self.src[0].ring
            prefix = f"{src_ring} -> {dst_ring}" if src_ring else f"? -> {dst_ring}"
            if self.op is Op.TO_RING and self.arg.residue is not Residue.BALANCED:
                return f"{prefix} residue={self.arg.residue.name.lower()}"
            return prefix
        if self.op is Op.COPY:
            return f"to={self._device}"
        return ""

    def _format_node_line(self, node_map: dict[Node, int]) -> str:
        """Format a single node as a line in the computation trace."""
        refs = self._format_src_refs(node_map)
        arg = self._format_arg()
        parts = [f"{self.op.name:12s}", refs, arg]
        return " ".join(p for p in parts if p)

    def format_trace(self, lines: int | None = None) -> str:
        """
        Return a multi-line computation trace showing the full DAG.
        """
        nodes = list(self.toposort(enter_calls=True))
        node_map = {node: i for i, node in enumerate(nodes)}
        count = len(nodes)
        width = len(str(count))

        def fmt(i):
            return f"{i:>{width}}  {nodes[i]._format_node_line(node_map)}"

        if lines is not None and count > lines:
            half = lines // 2
            skipped = count - lines
            result = [fmt(i) for i in range(half)]
            result.append(f"... ({skipped} lines skipped) ...")
            result.extend(fmt(i) for i in range(count - half, count))
        else:
            result = [fmt(i) for i in range(count)]

        trace = "\n".join(result)
        summary = _format_summary(nodes)
        return f"{trace}\n{summary}"

    def __str__(self) -> str:
        return self.format_trace(lines=None)


def _format_summary(nodes: list[Node]) -> str:
    counts = Counter(n.op for n in nodes)
    header = "\n====== Summary ======"
    rows = [f"{op.name:14s} {count:5d}" for op, count in counts.most_common()]
    return header + "\n" + "\n".join(rows)


def _format_int(val: int) -> str:
    bits = val.bit_length()
    if bits <= 64:
        return str(val)
    return f"<{bits}b>"
