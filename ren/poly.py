from typing import Any, Self, overload
import operator
import weakref

import numpy as np

import ren.montgomery as mont
from ren import tree
from ren.buffer import Buffer
from ren.device import Device
from ren.dtype import DType, dtypes, to_np_dtype
from ren.prof import trace
from ren.random import ChaChaRng, global_rng
from ren.utils import disable_gc
from ren.engine.schedule import schedule
from ren.engine.realize import ExecUnit, capturing, make_exec_units, run_exec_units
from ren.engine.memory import plan_memory
from ren.ir.node import Node, Op
from ren.mixin import OpMixin
from ren.polydist import PolyDist, SparseTernary, Ternary, UniformBounded, UniformModQ
from ren.ring import Domain, Residue, RingSpec
from ren.tables import crt_precompute

# Only unrealized polys can be affected by realization rewrites.
_pending_polys = weakref.WeakSet()
_function_tracing: list[None] = []


def _byteorder_dtype(dtype: DType) -> np.dtype:
    return np.dtype(f"<u{dtype.itemsize}")


def _signed_limb_count(lo: int, hi: int) -> int:
    signed_bits = max(hi, ~lo).bit_length() + 1
    return (signed_bits + 31) // 32


# Coefficient chunks use the uint32 format required by TO_RNS, independently of the ring dtype.
def _pack_signed_u32_limbs(data: list[int] | np.ndarray) -> np.ndarray:
    n = len(data)
    if isinstance(data, list) or data.dtype == object:
        flat = list(map(operator.index, data))
        limb_count = _signed_limb_count(min(flat), max(flat))
        width = 4 * limb_count
        raw = b"".join([x.to_bytes(width, "little", signed=True) for x in flat])
        return np.frombuffer(raw, dtype="<u4").reshape(n, limb_count).T.copy()

    flat = np.ascontiguousarray(data)
    kind = flat.dtype.kind
    if kind == "f":
        if not np.isfinite(flat).all() or (flat != np.rint(flat)).any():
            raise ValueError("coefficients must be finite integers")
    elif kind not in "iu":
        raise TypeError(f"polynomial coefficients must be integers, got {flat.dtype}")

    lo = 0 if kind == "u" else int(flat.min())
    limb_count = _signed_limb_count(lo, int(flat.max()))
    if limb_count <= 2:
        dtype = f"<{'u' if kind == 'u' else 'i'}{4 * limb_count}"
        return flat.astype(dtype, copy=False).view("<u4").reshape(n, limb_count).T.copy()

    if kind == "f":
        # Use at least float64 so every uint32 limb is exactly representable.
        flat = flat.astype(np.promote_types(flat.dtype, np.float64), copy=False)
        limbs = np.empty((limb_count, n), dtype="<u4")
        for limb in limbs:
            quotient = np.floor(np.ldexp(flat, -32))
            # Power-of-two division leaves an exact remainder in [0, 2**32).
            limb[:] = flat - np.ldexp(quotient, 32)
            flat = quotient
        return limbs

    # uint64 values above the signed range need a zero sign limb.
    out = np.zeros((limb_count, n), dtype="<u4")
    out[:2] = flat.astype("<u8", copy=False).view("<u4").reshape(n, 2).T
    return out


def _apply_map_to_polys(becomes_map: dict[Node, Node]) -> None:
    """
    Substitute poly nodes with their realized BUFFER-backed versions.

    Only updates polys whose graphs share computation with what was realized.
    """
    if not becomes_map:
        return

    in_scope: dict[Node, bool] = {}

    def needs_update(root: Node) -> bool:
        stack: list[tuple[Node, bool]] = [(root, False)]
        while stack:
            node, seen = stack.pop()
            if node in in_scope:
                continue
            src = node.inputs
            if not seen:
                stack.append((node, True))
                stack.extend((source, False) for source in reversed(src))
            else:
                in_scope[node] = node in becomes_map or any(in_scope[source] for source in src)
        return in_scope[root]

    pending = list(_pending_polys)
    polys = [poly for poly in pending if needs_update(poly.node)]
    if polys:
        sink = Node.sink(*[poly.node for poly in polys])
        new_sink = sink.substitute(becomes_map)
        for poly, old, new in zip(polys, sink.src, new_sink.src):
            if old is not new:
                poly.node = new

    for poly in pending:
        if poly.node.is_realized():
            _pending_polys.discard(poly)


def _schedule_and_lower(polys: list[Poly]) -> list[ExecUnit]:
    """Schedule polynomial graphs, update their wrappers, and lower executable units."""
    root = Node.sink(*(poly.node for poly in polys))

    with trace("schedule"):
        linear, becomes_map = schedule(root)

    _apply_map_to_polys(becomes_map)

    if not linear.src:
        return []

    for calls in capturing.values():
        calls.extend(linear.src)

    # Drop the rewrite map before memory planning.
    del becomes_map

    # Preserve output buffer identities.
    noopt_buffers = {poly.node.base.buffer.base for poly in polys if poly.node.base.op in {Op.BUFFER, Op.BUFFER_VIEW}}

    assigned = plan_memory(linear, noopt_buffers)
    return make_exec_units(linear, assigned)


def realize(*values: object) -> None:
    """Realize all Polys in supported nested values together, once per Poly identity.

    Device-free values stay virtual. Each placed result runs the dependencies it needs.
    """
    if _function_tracing:
        raise RuntimeError("ren.function: Poly realization is not allowed while tracing")

    polys = tree.collect(*values, leaf=Poly)
    to_realize = [poly for poly in {id(poly): poly for poly in polys}.values() if poly.device is not None and not poly.is_realized]
    if to_realize:
        with disable_gc(), trace("poly.realize"):
            run_exec_units(_schedule_and_lower(to_realize))


class Poly(OpMixin):
    __slots__ = ("node",)

    def realize(self) -> Self:
        """Realize this Poly and return self."""
        realize(self)
        return self

    @overload
    def __init__(self, data: Node, ring: RingSpec | None = None, device: str | None = None): ...

    @overload
    def __init__(self, data: list[int] | np.ndarray, ring: RingSpec, device: str | None = None): ...

    def __init__(self, data: list[int] | np.ndarray | Node, ring: RingSpec | None = None, device: str | None = None):
        """Create a Poly from integer coefficients or a polynomial node.

        NumPy floating-point arrays are accepted when every value is a finite integer.
        Coefficients are imported exactly as represented, without rounding.

        Args:
            data: Integer coefficient list, one-dimensional real NumPy array of length ring.n, or a polynomial node
            ring: Required for coefficients and ringless nodes; otherwise an optional consistency check
            device: Target device; nodes stay on their current device when omitted
        """
        if isinstance(data, Node):
            node_ring = data.ring
            if node_ring is None:
                if ring is None:
                    raise TypeError(f"Poly requires a ring for ringless node {data.op}")
                if data.dtype not in {dtypes.weakint, ring.dtype}:
                    raise ValueError(f"node dtype {data.dtype} does not match ring dtype {ring.dtype}")
                if data.op in {Op.BUFFER, Op.BUFFER_VIEW} and (size := data._storage_size()) != ring.storage_size:
                    raise ValueError(f"buffer size {size} does not match ring storage size {ring.storage_size}")
                data = data.as_ring(ring, data.domain or Domain.COEFF)
            elif ring is not None and ring != node_ring:
                raise ValueError(f"ring does not match node ring: {ring} != {node_ring}")
            self.node = data if device is None else self._to_device(data, device)
        else:
            if ring is None:
                raise TypeError("ring is required for coefficient data")
            n = ring.n
            if isinstance(data, np.ndarray) and data.ndim != 1:
                raise ValueError(f"coefficient data must be one-dimensional, got shape {data.shape}")
            if len(data) != n:
                raise ValueError(f"data length {len(data)} does not match degree {n}")
            limbs = _pack_signed_u32_limbs(data)
            source = self._to_device(self._host_array_buffer(limbs, dtypes.uint32), device)
            self.node = Node.coeffs_to_rns(source, ring, limb_count=limbs.shape[0])

        if not self.node.is_realized():
            _pending_polys.add(self)

    @staticmethod
    def _to_device(node: Node, device: str | None) -> Node:
        device = Device.canonicalize(device)
        src_device = node._device
        if src_device == device:
            return node
        return node.clone(device) if src_device is None else node.copy_to_device(device)

    @classmethod
    def _from_buffer(cls, buffer: Buffer, ring: RingSpec, domain: Domain) -> Poly:
        if buffer.dtype is not ring.dtype or buffer.size != ring.storage_size:
            raise ValueError(f"buffer {buffer.spec} does not match {ring}")
        if not buffer.is_allocated():
            raise ValueError("buffer must be allocated")
        node = Node.from_buffer(buffer)
        result = object.__new__(cls)
        result.node = Node._uncached(Op.AS_RING, node.dtype, (node,), (ring, domain))
        return result

    @staticmethod
    def _host_array_buffer(arr: np.ndarray, dtype: DType) -> Node:
        assert arr.flags.c_contiguous
        node = Node.new_buffer("python", size=arr.size, dtype=dtype)
        node.buffer.allocate(memoryview(arr).cast("B"))
        return node

    @classmethod
    def zero(cls, ring: RingSpec, device: str | None = None) -> Poly:
        return cls.const(0, ring, device)

    @classmethod
    def uniform_on_interval(cls, ring: RingSpec, *, bound: int, rng: ChaChaRng | None = None) -> Poly:
        """Random coefficients uniformly distributed in [-bound, bound]."""
        return cls.rand(UniformBounded(ring.n, bound), ring, rng=rng)

    @classmethod
    def uniform(cls, ring: RingSpec, *, rng: ChaChaRng | None = None) -> Poly:
        """Random coefficients uniformly distributed modulo q."""
        return cls.rand(UniformModQ(ring.n), ring, rng=rng)

    @classmethod
    def ternary(cls, ring: RingSpec, rng: ChaChaRng | None = None) -> Poly:
        """Ternary coefficients {-1, 0, 1} with probabilities (1/4, 1/2, 1/4)."""
        return cls.rand(Ternary(n=ring.n), ring, rng=rng)

    @classmethod
    def sparse_ternary(cls, ring: RingSpec, *, rng: ChaChaRng | None = None, h: int) -> Poly:
        """
        Sparse ternary coefficients {-1, 0, 1} with exactly h +-1 coefficients and n-h 0 coefficients.
        The locations of the nonzero coefficients are chosen as a uniformly random size-h subset of {0, ..., n-1},
        and each coefficient in those locations is independently set to -1 or 1 with probability 1/2.
        """
        return cls.rand(SparseTernary(n=ring.n, h=h), ring, rng=rng)

    @classmethod
    def rand(cls, dist: PolyDist, ring: RingSpec, *, rng: ChaChaRng | None = None) -> Poly:
        """Create a polynomial with coefficients drawn from the given distribution."""
        if rng is None:
            rng = global_rng()
        node = Node.random(rng=rng, dist=dist, ring=ring)
        return cls(node)

    @classmethod
    def const(cls, value: int, ring: RingSpec, device: str | None = None) -> Poly:
        """Create a constant polynomial. Pass a device to give it its own storage."""
        node = Node.const(value).as_ring(ring, Domain.COEFF)
        return cls(node if device is None else node.clone(device))

    @classmethod
    def monomial(cls, c: int, k: int, ring: RingSpec, device: str | None = None) -> Poly:
        """Create a monomial c * X^k."""
        return cls.const(c, ring, device).rotate(k)

    def _check_compatibility(self, other: Poly) -> None:
        """Raise an error if the polynomials' devices or rings do not match."""
        if self.device is not None and other.device is not None and self.device != other.device:
            raise ValueError(f"polynomial device mismatch in binary op: {self.device} != {other.device}")
        if self.ring != other.ring:
            raise ValueError("polynomial ring mismatch in binary op")

    @property
    def device(self) -> str | None:
        return self.node._device

    @property
    def ring(self) -> RingSpec:
        ring = self.node.ring
        assert ring is not None, "polynomial does not have ring semantics"
        return ring

    @property
    def is_realized(self) -> bool:
        return self.node.is_realized()

    @property
    def domain(self) -> Domain | None:
        return self.node.domain

    def ntt(self) -> Poly:
        if self.is_realized and self.domain is Domain.NTT:
            return self
        # Lazy domains can change during scheduling, so retain the explicit request.
        return self._unary(Op.NTT)

    def intt(self) -> Poly:
        if self.is_realized and self.domain is Domain.COEFF:
            return self
        # Lazy domains can change during scheduling, so retain the explicit request.
        return self._unary(Op.INTT)

    def automorphism(self, galois_elt: int) -> Poly:
        return self._unary(Op.AUTOMORPHISM, arg=galois_elt)

    def _binop(self, op: Op, other: Poly | int, reverse: bool = False) -> Poly:
        if isinstance(other, Poly):
            self._check_compatibility(other)
            lhs, rhs = self.node, other.node
        else:
            lhs = self.node
            # Lift scalars to ring constants so downstream lowering can treat
            # constant polynomials uniformly (AS_RING(CONST)).
            rhs = lhs.const_like(other)
        if reverse:
            lhs, rhs = rhs, lhs
        return Poly(Node(op, dtype=lhs.dtype, src=(lhs, rhs)))

    def _unary(self, op: Op, *, arg: Any = None) -> Poly:
        return Poly(self.node._unary(op, arg=arg))

    def _to_ring(self, new_ring: RingSpec, *, residue: Residue = Residue.BALANCED) -> Poly:
        """
        Convert polynomial to a new ring. Backend picks the algorithm based on
        the relationship between source and target rings, while preserving the
        requested residue system when arithmetic lifting is required.
        """
        node = self.node.to_ring(new_ring, residue=residue)
        return self if node is self.node else Poly(node)

    def _to_ring_small_coeffs(self, new_ring: RingSpec) -> Poly:
        # only BALANCED preserves the small coefficient property
        return self._to_ring(new_ring, residue=Residue.BALANCED)

    def to_quotient_ring(self, new_ring: RingSpec) -> Poly:
        if not new_ring.is_quotient_of(self.ring):
            raise ValueError("destination ring is not a quotient of source ring")
        # since we're taking a quotient, the residue mode has no effect
        return self._to_ring(new_ring)

    def lift_to_ring(self, new_ring: RingSpec, residue: Residue = Residue.BALANCED) -> Poly:
        if not self.ring.is_quotient_of(new_ring):
            raise ValueError("source ring is not a quotient of destination ring")
        return self._to_ring(new_ring, residue=residue)

    def scale_by_primes(self, scalars: tuple[int, ...]) -> Poly:
        """
        Multiply each RNS limb by a per-prime scalar.

        Unlike polynomial multiplication, this scales all coefficients in limb i
        by scalars[i]. This is used for embedding masks in keyswitching where
        different RNS primes need different scaling factors.

        Mathematically, this multiplies by the ring element specified in factored
        coordinates of the product ring Z_q ≅ ∏Z_{pi}. The tuple (s0, s1, ..., sk-1)
        represents the unique S ∈ Z_q where S ≡ si (mod pi).
        """
        # CRT lift: reconstruct S from per-prime scalars
        moduli = self.ring.moduli
        partial_prods, inverses = crt_precompute(self.ring.moduli)
        s = 0
        for i, (r, p) in enumerate(zip(scalars, moduli)):
            diff = (r - s % p) % p
            s = s + (inverses[i] * diff % p) * partial_prods[i]
        const_poly = Poly.const(s, self.ring)
        return self * const_poly

    def rescale(self, target_moduli: tuple[int, ...]) -> Poly:
        """
        Rescale into a lower-modulus ring by approximately applying the modulus ratio
        between the source and destination bases.
        """
        moduli = self.ring.moduli
        num_primes = len(moduli)
        target_set = set(target_moduli)
        num_b = sum(1 for modulus in moduli if modulus not in target_set)
        if not (0 < num_b <= num_primes):
            raise ValueError(f"num_b out of bounds (num_b={num_b}, num_primes={num_primes})")
        dst_ring = RingSpec(n=self.ring.n, moduli=target_moduli, dtype=self.ring.dtype)
        return self._unary(Op.RESCALE, arg=dst_ring)

    def rotate(self, k: int) -> Poly:
        """
        Multiplies a polynomial by X^k
        """
        return self._unary(Op.ROTATE, arg=k)

    def to(self, device: str) -> Poly:
        device = Device.canonicalize(device)
        if self.device is None or device == self.device:
            return self

        # copy nodes preserves metadata
        return Poly(self.node.copy_to_device(device))

    def clone(self, device: str | None = None) -> Poly:
        """Return a value-equivalent polynomial backed by fresh storage."""
        return Poly(self.node.clone(device))

    def contiguous(self) -> Poly:
        """Return a contiguous poly.

        This should be called before accessing .buffer when view ops may be present.
        """
        node = self.node.contiguous()
        return self if node is self.node else Poly(node)

    def _materialize_python(self) -> Poly:
        if capturing:
            raise RuntimeError("cannot read polynomial data during JIT capture; the value would be frozen into the replay")
        if self.device is None:
            if (value := self.node.const_val) is not None:
                return self if self.domain is not None else Poly(self.node.const_like(value))
            p = self.clone("python")
        else:
            p = self.to("python")
        return p.contiguous().realize()

    def data(self) -> memoryview:
        """Return a zero-copy memoryview over the raw RNS limb bytes.

        The layout is little-endian coefficient limbs in prime-major order with shape
        ``(len(ring.moduli), ring.n)``.  The returned view is tied to the
        polynomial's buffer — do **not** mutate it.

        Raises :class:`RuntimeError` for constant polynomials; use :meth:`tobytes` instead.
        """
        p = self._materialize_python()
        ring = p.ring
        if p.node.const_val is not None:
            raise RuntimeError("data() is not supported for constant polynomials; use tobytes()")

        base = p.node.base
        if base.op not in {Op.BUFFER, Op.BUFFER_VIEW}:
            raise RuntimeError("polynomial has no realized buffer")

        mv = base.buffer.as_buffer()
        expected = ring.nbytes
        if len(mv) != expected:
            raise RuntimeError(f"unexpected buffer size: got {len(mv)}, expected {expected}")
        return mv

    def tobytes(self) -> bytes:
        """Return raw RNS limb bytes for this polynomial (always copies).

        The layout is little-endian coefficient limbs in prime-major order with shape
        ``(len(ring.moduli), ring.n)``. This exports the polynomial in its
        current domain.
        """
        p = self._materialize_python()
        ring = p.ring

        # Constant nodes have no backing buffer; synthesize canonical raw limbs.
        if (const_value := p.node.const_val) is not None:
            limbs = np.zeros((len(ring.moduli), ring.n), dtype=_byteorder_dtype(ring.dtype))
            for i, modulus in enumerate(ring.moduli):
                reduced = mont.context(ring.dtype, modulus).encode(const_value)
                if p.domain == Domain.NTT:
                    limbs[i].fill(reduced)
                else:
                    limbs[i, 0] = reduced
            return limbs.tobytes()

        return bytes(p.data())

    @classmethod
    def frombytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        ring: RingSpec,
        domain: Domain = Domain.COEFF,
        device: str | None = None,
    ) -> Poly:
        """Build a polynomial from raw RNS limb bytes.

        Accepts any bytes-like object. The underlying buffer is referenced
        without copying, so callers must not mutate or resize mutable inputs
        (for example ``bytearray`` or writable ``memoryview``) while the
        resulting polynomial is still alive.
        """
        # Normalise memoryviews to flat bytes before checking their length.
        if isinstance(data, memoryview):
            data = data.cast("B")
        expected_len = ring.nbytes
        if len(data) != expected_len:
            raise ValueError(f"invalid poly byte length: {len(data)} != {expected_len}")

        limbs = np.frombuffer(data, dtype=_byteorder_dtype(ring.dtype)).astype(to_np_dtype(ring.dtype), copy=False)
        node = cls._host_array_buffer(limbs, ring.dtype).as_ring(ring, domain)
        return cls(cls._to_device(node, device))

    def __reduce__(self):
        """Pickle as a materialized poly: realize and store concrete bytes
        (or a const value) plus ring/domain, never the lazy IR graph."""
        self.realize()
        value = self.node.const_val
        snapshot = self._materialize_python() if self.domain is None else self
        if value is None:
            value = snapshot.tobytes()
        domain = snapshot.domain
        if domain is None:
            raise ValueError("poly has no concrete domain metadata")
        return type(self)._unpickle, (self.ring, domain, value)

    @classmethod
    def _unpickle(cls, ring: RingSpec, domain: Domain, value: int | bytes) -> Poly:
        if isinstance(value, bytes):
            return cls.frombytes(value, ring=ring, domain=domain)
        poly = cls.const(value, ring)
        return poly if domain is Domain.COEFF else poly.ntt()

    def to_centered_list(self) -> list[int]:
        """Reconstruct coefficients as the balanced [-Q/2, Q/2) representative.

        This is the native output of Op.FROM_RNS. Reconstruction runs on the source
        device's backend (the GPU for cuda polys); only the resulting limb buffer is
        copied to host for the final big-int combine. tolist() derives [0, Q) from this.
        """
        if capturing:
            raise RuntimeError("cannot read polynomial data during JIT capture; the value would be frozen into the replay")
        ring = self.ring
        n, q = ring.n, ring.q

        coeff = self.intt()  # ensure COEFF domain (graph op; redundant INTT is elided by rules)

        # CONST polynomials: compute analytically.
        if (c := coeff.node.const_val) is not None:
            return [((c + q // 2) % q) - q // 2] + [0] * (n - 1)

        # Single schedule: the scheduler materializes FROM_RNS's RNS input internally
        # (conversion sources are bufferized), FROM_RNS reconstructs on the source
        # device (the GPU for cuda polys), and only the limb buffer is copied to host.
        source = coeff.clone("python") if coeff.device is None else coeff
        fr = Poly(source.node.from_rns()).to("python").contiguous()
        fr.realize()

        # uint32 limbs, [limb_count, n];
        raw = np.array(fr.node.buffer.as_typed_buffer(), copy=False)
        limb_count = len(raw) // n
        # Column order groups each coefficient's little-endian two's-complement
        # bytes together. int.from_bytes decodes them at C speed
        # (signed=True absorbs the centering).
        b = raw.reshape(limb_count, n).tobytes(order="F")
        stride = limb_count * raw.itemsize
        return [int.from_bytes(b[i * stride : (i + 1) * stride], "little", signed=True) for i in range(n)]

    def tolist(self) -> list[int]:
        """Reconstruct integer coefficients in [0, Q), derived from to_centered_list()."""
        q = self.ring.q
        return [c % q for c in self.to_centered_list()]

    def __iter__(self):
        yield from self.tolist()

    def equals(self, other: Poly) -> bool:
        """
        Check if two polynomials are equal.

        This returns True if and only if the two polynomials have the same RingSpec and
        the same coefficients. (In particular it does not care about which domain the polys are in.)
        """
        if self.ring != other.ring:
            return False
        diff = self - other
        return all(c == 0 for c in diff.to_centered_list())
