"""Matrices stored by diagonals, and their application to ciphertexts.

* :class:`SparseMatrix` -- plain NumPy linear algebra on matrices whose
  nonzero entries occupy a small number of (wrapping) diagonals.
* :class:`LinearTransform` -- a :class:`SparseMatrix` encoded at one level of
  the active context, ready to multiply ciphertexts.  With a direct rotation
  key per step it is double-hoisted (Bossuat et al., EUROCRYPT 2021).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from math import gcd, isfinite, prod
from operator import add
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from ren.bigfloat import BigFloat, FloatLike
from ren.container import PolyContainer
from ren.ring import RingSpec
from ren.ckks._rlwe import RLWECiphertext
from ren.ckks.arithmetic_ops import horner_step, sum_rotated
from ren.ckks.ckks import Ciphertext, HoistedDigits, Plaintext, _decompose_and_modraise, _keyswitch_product, encode
from ren.ckks.context import context
from ren.ckks.keys import SwitchKey
from ren.ckks.utils import base_rotation_amounts, bsgs_index, bsgs_steps, normalize_rotation_amounts, steps_to_galois_elt
from ren.ckks.errors import ParameterMismatchError


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class SparseMatrix:
    """
    An n-by-n matrix whose nonzero entries occupy a small number of diagonals.

    Diagonal ``d`` holds the entries ``M[r, (r + d) mod n]``, so diagonals wrap
    around. For example, the matrix

    [0  1  0  3 ]
    [4  5  6  0 ]
    [0  9  10 11]
    [12 0  14 15]

    is stored as ``offsets = [0, 1, 3]`` with

    data = [[0, 5, 10, 15],
            [1, 6, 11, 12],
            [3, 4, 9,  14]].

    This is the layout a ciphertext product needs (see :class:`LinearTransform`):
    rotate the ciphertext by each offset, multiply each rotation with the
    corresponding diagonal, and sum the results.

    ``offsets`` is sorted and unique in ``[0, n)`` and ``data`` is an owned
    complex128 array of shape ``(len(offsets), n)``; both are read-only. The
    constructor takes that shape directly; :meth:`from_diagonals`,
    :meth:`from_entries` and :meth:`from_dense` build it from other forms.
    """

    n: int
    offsets: np.ndarray
    data: np.ndarray

    def __init__(self, n: int, offsets: ArrayLike, data: Sequence[ArrayLike] | np.ndarray):
        """Matrix whose diagonal ``offsets[i]`` is ``data[i]``, a length-``n``
        row. Offsets that agree mod ``n`` are summed into one diagonal."""
        given = np.asarray(offsets, dtype=np.int64).ravel()
        if len(given) != len(data):
            raise ValueError(f"expected one row of data per offset, got {len(given)} offsets")
        unique, rows = np.unique(given % n, return_inverse=True)
        owned = np.zeros((len(unique), n), dtype=np.complex128)
        for row, diagonal in zip(rows.tolist(), data):
            diagonal = np.asarray(diagonal, dtype=np.complex128)
            if diagonal.shape != (n,):
                raise ValueError(f"diagonal must have shape ({n},), got {diagonal.shape}")
            owned[row] += diagonal
        owned.flags.writeable = False
        unique.flags.writeable = False

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "offsets", unique)
        object.__setattr__(self, "data", owned)

    @classmethod
    def from_diagonals(cls, n: int, diagonals: Mapping[int, ArrayLike]) -> Self:
        """
        Constructs the n-by-n matrix from a mapping of offset to diagonal,
        e.g. ``{0: [0, 5, 10, 15], 1: [1, 6, 11, 12]}``.
        """
        return cls(n, list(diagonals), list(diagonals.values()))

    @classmethod
    def from_entries(cls, n: int, *, rows: ArrayLike, cols: ArrayLike, values: ArrayLike) -> Self:
        """
        Constructs the n-by-n matrix with ``values[i]`` at position
        ``(rows[i], cols[i])``. Repeated positions are summed.
        """
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        values = np.asarray(values, dtype=np.complex128).ravel()
        if not len(rows) == len(cols) == len(values):
            raise ValueError(f"rows, cols, and values must have the same length, got {len(rows)}, {len(cols)}, {len(values)}")
        if len(rows) and (min(rows.min(), cols.min()) < 0 or max(rows.max(), cols.max()) >= n):
            raise ValueError(f"entries lie outside the {n}x{n} matrix")

        offset = (cols - rows) % n
        offsets = np.flatnonzero(np.bincount(offset, minlength=n))
        number = np.empty(n, dtype=np.int64)
        number[offsets] = np.arange(len(offsets))
        flat = number[offset] * n + rows
        size = len(offsets) * n
        summed = np.bincount(flat, weights=values.real, minlength=size) + 1j * np.bincount(flat, weights=values.imag, minlength=size)
        return cls(n, offsets, summed.reshape(len(offsets), n))

    @classmethod
    def from_dense(cls, array: ArrayLike) -> Self:
        """
        Constructs the matrix from a dense square array. Only useful for
        small matrices: the dense form of an n-by-n matrix has n^2 entries.
        """
        arr = np.asarray(array, dtype=np.complex128)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Input must be a square matrix, got shape {arr.shape}")
        rows, cols = np.nonzero(arr)
        return cls.from_entries(arr.shape[0], rows=rows, cols=cols, values=arr[rows, cols])

    @classmethod
    def zero(cls, n: int) -> Self:
        """
        Returns the n-by-n zero matrix.
        """
        return cls(n, [], [])

    @classmethod
    def identity(cls, n: int) -> Self:
        """
        Returns the n-by-n identity matrix.
        """
        return cls(n, [0], np.ones((1, n)))

    def _entries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, cols, values)`` of the nonzero entries."""
        number, rows = np.nonzero(self.data)
        return rows, (rows + self.offsets[number]) % self.n, self.data[number, rows]

    def toarray(self) -> np.ndarray:
        """
        Converts the sparse matrix to a dense array.
        This is the inverse of `from_dense`.
        """
        matrix = np.zeros((self.n, self.n), dtype=np.complex128)
        rows, cols, values = self._entries()
        matrix[rows, cols] = values
        return matrix

    def _aligned(self, other: SparseMatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The union of both offset sets, and each matrix's data laid out on it."""
        if self.n != other.n:
            raise ValueError(f"Matrix dimensions must match: {self.n} != {other.n}")
        offsets = np.union1d(self.offsets, other.offsets)
        left = np.zeros((len(offsets), self.n), dtype=np.complex128)
        right = np.zeros_like(left)
        left[np.searchsorted(offsets, self.offsets)] = self.data
        right[np.searchsorted(offsets, other.offsets)] = other.data
        return offsets, left, right

    def __add__(self, other: SparseMatrix) -> SparseMatrix:
        offsets, left, right = self._aligned(other)
        return SparseMatrix(self.n, offsets, left + right)

    def __sub__(self, other: SparseMatrix) -> SparseMatrix:
        offsets, left, right = self._aligned(other)
        return SparseMatrix(self.n, offsets, left - right)

    def __matmul__(self, other: SparseMatrix) -> SparseMatrix:
        """
        Multiplies two sparse matrices (standard matrix multiplication)
        """
        if self.n != other.n:
            raise ValueError(f"Matrix dimensions must match: {self.n} != {other.n}")
        n = self.n
        offsets, number = np.unique((self.offsets[:, None] + other.offsets[None, :]) % n, return_inverse=True)
        data = np.zeros((len(offsets), n), dtype=np.complex128)
        for d, diagonal, rows in zip(self.offsets.tolist(), self.data, number.reshape(len(self.offsets), len(other.offsets))):
            np.add.at(data, rows, diagonal * np.roll(other.data, -d, axis=1))
        return SparseMatrix(n, offsets, data)

    def __mul__(self, scalar: complex | FloatLike) -> SparseMatrix:
        """
        Multiplies the matrix by a scalar. A real scalar must be representable
        as a nonzero finite float: the entries are complex128.
        """
        if not isinstance(scalar, complex):
            exact = BigFloat(scalar)
            scalar = float(exact)
            if not exact.is_finite() or not isfinite(scalar) or (exact and scalar == 0):
                raise ValueError("scale must be finite and representable as a float")
        return SparseMatrix(self.n, self.offsets, scalar * self.data)

    def __rmul__(self, scalar: complex | FloatLike) -> SparseMatrix:
        return self * scalar


def _extended_ring(ring: RingSpec) -> RingSpec:
    """``P * Q``: ``ring`` extended by the key-switching primes."""
    return RingSpec(n=ring.n, moduli=context().params.ks_primes + ring.moduli, dtype=ring.dtype)


def _rotate_unreduced(ct: RLWECiphertext, hoisted: HoistedDigits, galois_elt: int, key: SwitchKey) -> RLWECiphertext:
    """Apply ``galois_elt`` to a ciphertext held in ``P * Q`` with its message
    scaled by ``prod(P)``; ``hoisted`` are the digits of its ``a`` in ``Q``.
    The key-switch product stays un-rescaled, so the result has the same form."""
    ks = _keyswitch_product(hoisted, key).map_polys(lambda poly: poly.to_quotient_ring(ct.ring))
    return RLWECiphertext(
        a=ks.a.automorphism(galois_elt),
        b=(ks.b + ct.b).automorphism(galois_elt),
        noise_info=replace(ks.noise_info, msg_bound=0) + ct.noise_info,
    )


@dataclass(frozen=True, slots=True)
class LinearTransformGroup(PolyContainer):
    """One giant step: the baby rotations it sums and their plaintexts."""

    giant_step: int
    baby_steps: tuple[int, ...]
    plaintexts: tuple[Plaintext, ...]


@dataclass(frozen=True, slots=True)
class LinearTransform(PolyContainer):
    """A :class:`SparseMatrix` encoded to multiply ciphertexts at ``level``.

    Each offset ``d`` is decomposed as ``d = baby + giant (mod n)`` by
    :func:`bsgs_index`; the diagonals are pre-encoded as NTT-domain plaintexts,
    rotated by ``-giant`` to compensate for the deferred giant-step rotation.

    :meth:`rotation_amounts` takes the key policy ``hoist``. By default the
    baby phase needs only the gcd base key and the giant phase, which runs
    after the rescale, only the Horner fold's
    ``±step`` key; the rotation primitive composes base keys to reach every
    step. With ``hoist`` every nonzero step gets a direct key at the input
    level instead. Whenever all those direct keys are available, :meth:`encode`
    selects double-hoisting: one decompose+mod-raise serves all baby rotations,
    each nonzero giant rotation reduces only the ``a`` it decomposes, and a single full ModDown,
    folded into the final rescale, reduces the sum.  More key memory; faster
    and more precise transforms.

    Encoding is the expensive part: encode once and apply many times. Under
    :func:`ren.jit`, pass the transform as an argument to replay one capture
    with different matrices of the same offsets -- a transform the traced
    function merely closes over is baked into the capture as constants.
    """

    level: int
    groups: tuple[LinearTransformGroup, ...]

    @staticmethod
    def rotation_amounts(matrix: SparseMatrix, *, hoist: bool = False) -> dict[int, set[int]]:
        """Canonical rotation amounts by level relative to the transform's
        input: ``0`` for the baby rotations and ``-1`` for the giant
        rotations, which run after the rescale (all at ``0`` when hoisted),
        for :func:`ren.ckks.keygen`."""
        baby_steps, giant_steps = bsgs_steps(n=matrix.n, diagonals=matrix.offsets.tolist())
        n = matrix.n
        if hoist:
            return {0: normalize_rotation_amounts({*baby_steps, *giant_steps}, n), -1: set()}
        giant_amounts = set(giant_steps) if horner_step(giant_steps) is None else base_rotation_amounts(giant_steps)
        return {0: normalize_rotation_amounts(base_rotation_amounts(baby_steps), n), -1: normalize_rotation_amounts(giant_amounts, n)}

    @classmethod
    def encode(cls, matrix: SparseMatrix, *, level: int) -> Self:
        """Encode a matrix of the active context's slot count at ``level``.
        Double-hoist when every required rotation has a direct key at this level."""
        ctx = context()
        if matrix.n != ctx.params.slot_count:
            raise ValueError(f"matrix dimension {matrix.n} must match the slot count {ctx.params.slot_count}")

        # A double-hoisted transform multiplies un-rescaled rotations, so its
        # plaintexts live in P * Q.
        ring = ctx.params.ring_at(level)
        rotations = cls.rotation_amounts(matrix, hoist=True)[0]
        direct = ctx.keyset._rotation_galois_elts_at(level)
        if rotations and all(steps_to_galois_elt(amount, ring.n) in direct for amount in rotations):
            ring = _extended_ring(ring)

        n = matrix.n
        diagonals = dict(zip(matrix.offsets.tolist(), matrix.data))
        groups: list[LinearTransformGroup] = []
        for giant_step, baby_steps in bsgs_index(n=n, diagonals=diagonals).items():
            rotated = (np.roll(diagonals[(baby_step + giant_step) % n], giant_step % n) for baby_step in baby_steps)
            plaintexts = [encode(diagonal, msg_bound=float(np.max(np.abs(diagonal), initial=0.0)), level=level) for diagonal in rotated]
            groups.append(LinearTransformGroup(giant_step, baby_steps, tuple(replace(pt, m=pt.m.lift_to_ring(ring).ntt()) for pt in plaintexts)))
        return cls(level=level, groups=tuple(groups))

    @property
    def baby_steps(self) -> tuple[int, ...]:
        """The rotations of the input shared by all groups."""
        return tuple(sorted({step for group in self.groups for step in group.baby_steps}))

    def __call__(self, ct: Ciphertext) -> Ciphertext:
        """The product of the encoded matrix with a ciphertext (vector).
        Requires the keys of :meth:`rotation_amounts`."""
        if ct.level != self.level:
            raise ValueError(f"transform was encoded for level {self.level}, got a ciphertext at level {ct.level}")
        if not self.groups:
            return Ciphertext.zero(level=ct.level)
        # The plaintexts' ring says how the transform was encoded: in the
        # ciphertext's ring, or double-hoisted in ``P * Q``.
        ring = self.groups[0].plaintexts[0].ring
        if ring == ct.ring:
            return self._apply_composed(ct)
        if ring == _extended_ring(ct.ring):
            return self._apply_double_hoisted(ct)
        raise ParameterMismatchError("transform ring does not match the ciphertext ring")

    def _apply_composed(self, ct: Ciphertext) -> Ciphertext:
        """Each rotation is key-switched and reduced on its own."""
        # Babies are non-negative multiples of their gcd, so chaining the base
        # key reaches them all within ``largest / gcd`` key-switches.
        baby_steps = self.baby_steps
        baby_hops = baby_steps[-1] // (reduce(gcd, baby_steps) or 1)
        rotated = dict(zip(baby_steps, ct.rotate(list(baby_steps), max_ks=baby_hops)))

        # Giant phase: rotate each group's sum and sum the results;
        # sum_rotated picks the strategy from the giants and the provisioned
        # keys (shared-rescale direct rotations, Horner fold, or per-group
        # rotation).
        group_sums = [(group.giant_step, Ciphertext.dot(group.plaintexts, [rotated[step] for step in group.baby_steps])) for group in self.groups]
        return sum_rotated(group_sums, max_ks=1)

    def _apply_double_hoisted(self, ct: Ciphertext) -> Ciphertext:
        """Nothing leaves ``P * Q`` until the end: the plaintexts multiply the
        un-rescaled baby rotations, a giant rotation reduces only the ``a`` it
        decomposes, and the final rescale is the only full ModDown."""
        if ct.level == 0:
            raise ValueError("cannot multiply at level 0")
        ctx, n = context(), ct.ring.n
        galois_elts = {step: steps_to_galois_elt(step % (n // 2), n) for step in (*self.baby_steps, *(g.giant_step for g in self.groups))}
        keys = {g: ctx.keyset.get_galois_key(g) for g in set(galois_elts.values()) - {1}}

        # prod(P) * ct in the extended ring (zero on the P primes) is the
        # unrotated term, and what each key-switch product is added to.
        p = prod(ctx.params.ks_primes)
        ring_qt = _extended_ring(ct.ring)
        scaled = RLWECiphertext(a=ct.a.lift_to_ring(ring_qt) * p, b=ct.b.lift_to_ring(ring_qt) * p, noise_info=ct.noise_info * p)

        babies = dict.fromkeys(self.baby_steps, scaled)
        if rotated := [step for step in self.baby_steps if galois_elts[step] != 1]:
            # One decomposition of ct.a serves every baby rotation.
            hoisted = _decompose_and_modraise(ct.a, keys[galois_elts[rotated[0]]])
            babies |= {step: _rotate_unreduced(scaled, hoisted, galois_elts[step], keys[galois_elts[step]]) for step in rotated}

        group_sums: list[RLWECiphertext] = []
        for group in self.groups:
            group_sum = reduce(add, (babies[step] * pt for step, pt in zip(group.baby_steps, group.plaintexts, strict=True)))
            # Both halves read the same plaintexts; realizing them together lets one kernel compute both.
            group_sum = replace(group_sum, a=group_sum.a.contiguous(), b=group_sum.b.contiguous())
            g = galois_elts[group.giant_step]
            if g != 1:
                group_sum = _rotate_unreduced(group_sum, _decompose_and_modraise(group_sum.a.rescale(ct.ring.moduli), keys[g]), g, keys[g])
            group_sums.append(group_sum)

        total = reduce(add, group_sums)._rescale(ctx.params.ring_at(ct.level - 1).moduli)
        values = ct._map_values(self._tracked_values, *(pt for group in self.groups for pt in group.plaintexts))
        return replace(Ciphertext._from_rlwe(total), _values=values)

    def _tracked_values(self, values: np.ndarray, *plaintexts: np.ndarray) -> np.ndarray:
        """Apply the transform to tracked ciphertext and diagonal slot values."""
        rotated_by = {step: np.roll(values, -step) for step in self.baby_steps}
        diagonals = iter(plaintexts)
        group_values = (
            np.roll(reduce(add, (next(diagonals) * rotated_by[step] for step in group.baby_steps)), -group.giant_step) for group in self.groups
        )
        return reduce(add, group_values)
