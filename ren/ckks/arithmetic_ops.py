"""Arithmetic operations for CKKS ciphertexts and plaintexts."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from functools import reduce
from operator import add

import numpy as np

from ren.bigfloat import BigFloat
from ren.poly import Poly
from ren.ring import RingSpec
from ren.ckks.errors import ParameterMismatchError

from .ckks import (
    Ciphertext,
    Plaintext,
    _decompose_and_modraise,
    _keyswitch,
    _keyswitch_product,
    encode,
)
from .context import context
from . import encoding
from .encoding import RawSlots, Scalar, slot_array
from .metadata import CKKSNoiseInfo
from .params import RescaleStep
from .utils import count_calls, steps_to_galois_elt


def _sum_polys(polys: Sequence[Poly], ring: RingSpec) -> Poly:
    for i, poly in enumerate(polys):
        if poly.ring != ring:
            raise ParameterMismatchError(f"poly[{i}] ring does not match expected ring")

    if not polys:
        return Poly.zero(ring)

    return reduce(add, polys)


def sum_ciphertexts(cts: Sequence[Ciphertext]) -> Ciphertext:
    """Sum a sequence of ciphertexts, possibly at mixed levels.

    Apply each conversion up to its final rescale. Contributions with the same
    final working ring share that rescale after accumulation. Earlier rounding
    boundaries are preserved. Rounding the sum can differ from summing separately
    rounded inputs, as in other deferred-rescale arithmetic.
    """
    if not cts:
        raise ValueError("Cannot sum empty sequence of ciphertexts")
    if len(cts) == 1:
        return cts[0]

    min_level = min(ct.level for ct in cts)
    base_cts = [ct for ct in cts if ct.level == min_level]
    high_cts = [ct for ct in cts if ct.level > min_level]

    if not high_cts:
        return reduce(add, base_cts)

    params = context().params
    target_ring = params.ring_at(min_level)
    groups: defaultdict[RingSpec, list[tuple[Poly, Poly]]] = defaultdict(list)
    noise_terms: list[CKKSNoiseInfo] = []
    by_level: defaultdict[int, list[Ciphertext]] = defaultdict(list)
    for ct in high_cts:
        by_level[ct.level].append(ct)
    for level, terms in by_level.items():
        transition = params.transition(level, min_level)
        ratio = transition.poly_ratio
        for ct in terms:
            groups[transition.steps[-1].src_ring].append((transition.prepare(ct.a), transition.prepare(ct.b)))
            noise_terms.append(ct.noise_info * ratio)

    a_terms = [ct.a for ct in base_cts]
    b_terms = [ct.b for ct in base_cts]
    for ring, terms in groups.items():
        a_terms.append(_sum_polys([a for a, _ in terms], ring).rescale(target_ring.moduli))
        b_terms.append(_sum_polys([b for _, b in terms], ring).rescale(target_ring.moduli))
    final_a = _sum_polys(a_terms, target_ring)
    final_b = _sum_polys(b_terms, target_ring)
    noise_terms.extend(ct.noise_info for ct in base_cts)

    # Plaintext values: abstract message is preserved across level drops, so the
    # tracked sum is just the elementwise sum across all inputs.
    final_plaintext_values = cts[0]._map_values(lambda *values: reduce(add, values), *cts[1:])
    noise_info = CKKSNoiseInfo.sum(noise_terms)
    return Ciphertext(a=final_a, b=final_b, noise_info=noise_info, _values=final_plaintext_values)


def multiply_ciphertexts(ciphertexts: list[Ciphertext]) -> Ciphertext:
    """
    Compute the product of multiple ciphertexts.

    This function multiplies all ciphertexts together by recursively finding
    the two ciphertexts with the highest levels and combining them. This
    strategy minimizes level consumption.
    """
    if len(ciphertexts) == 0:
        raise ValueError("cannot compute product of empty list")

    if len(ciphertexts) == 1:
        return ciphertexts[0]

    # Work on a copy to preserve the original list
    cts = list(ciphertexts)

    while len(cts) > 1:
        # Sort by level in descending order and take the two highest
        cts.sort(key=lambda ct: ct.level, reverse=True)
        ct1 = cts.pop(0)
        ct2 = cts.pop(0)

        # Multiply and add result back to the list
        cts.append(ct1 * ct2)

    return cts[0]


_Operand = RawSlots | Plaintext | Ciphertext


def _dot_values(left: Sequence[_Operand], right: Sequence[_Operand], slot_count: int, *, prec: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute tracked slot values and the sum of raw products at FLINT precision."""

    def slots(value: _Operand) -> np.ndarray:
        data = value._values if isinstance(value, Plaintext | Ciphertext) else encoding.slots(value, slot_count, prec=prec)
        assert data is not None
        return data[:1] if data.strides == (0,) else data

    tracked = context().track_plaintext_values and all(
        v._values is not None for side in (left, right) for v in side if isinstance(v, Plaintext | Ciphertext)
    )
    values = pure_sum = None
    with encoding.flint_precision(prec):
        for x, y in zip(left, right, strict=True):
            raw = not isinstance(x, Plaintext | Ciphertext) and not isinstance(y, Plaintext | Ciphertext)
            if not tracked and not raw:
                continue
            product = slots(x) * slots(y)
            if tracked:
                values = product if values is None else values + product
            if raw:
                pure_sum = product if pure_sum is None else pure_sum + product
    return (
        None if values is None else np.broadcast_to(values, (slot_count,)),
        None if pure_sum is None else np.broadcast_to(pure_sum, (slot_count,)),
    )


@count_calls
def dot(left: Sequence[_Operand], right: Sequence[_Operand]) -> Ciphertext:
    """Heterogeneous ``sum_i left[i] * right[i]``, at the lowest input level minus one.

    Entries can be ciphertexts, plaintexts, scalars, or per-slot lists/1-d arrays
    of at most ``slot_count`` values. At least one input must be encoded.
    Scalars broadcast and shorter rows are zero-padded.

    Products with a common working ring and scale share a rounded conversion.
    Inputs are aligned first when their product cannot fit or its integer scale
    correction is too coarse. Components are accumulated together across
    plaintext/ciphertext pairs, with one relinearization and one final rescale.
    Pure raw pairs are summed and encoded directly at the output level.

    Invalid lengths/types/shapes raise; ciphertext products require a relin key.
    Slot values are tracked only when tracking is enabled and every encoded
    input has values. Noise metadata propagates each conversion's
    linear factor, excluding new rounding.
    """
    if len(left) != len(right):
        raise ParameterMismatchError(f"dot length mismatch: {len(left)} vs {len(right)}")
    ctx = context()
    params = ctx.params
    encoded = []
    prepared: list[complex | float | int | np.ndarray | Plaintext | Ciphertext] = []
    for v in (*left, *right):
        if isinstance(v, Plaintext | Ciphertext):
            encoded.append(v)
        elif isinstance(v, Scalar):
            if isinstance(v, np.integer):
                v = int(v)
        else:
            # Keep native array dtypes and exact Python values until their numerical boundary.
            v = slot_array(v, params.slot_count, dtype=v.dtype if isinstance(v, np.ndarray) else object)
        prepared.append(v)
    left, right = prepared[: len(left)], prepared[len(left) :]
    if not encoded:
        raise ValueError("dot requires at least one Plaintext or Ciphertext input")
    level = min(v.level for v in encoded)
    if level == 0:
        raise ValueError("cannot multiply at level 0")
    target_ring, target_scale = params.ring_at(level), params.scale(level)
    # Raw encoding rounds at scale target_scale**2 / input_scale. Allow at most
    # one coefficient unit at the less precise working/output scale.
    raw_scale_limit = 2 * target_scale**2 / min(target_scale, params.scale(level - 1))
    # Reuse the smallest working ring chosen by quotient-chain conversions.
    fused_levels: set[int] = set()
    for source in {v.level for v in encoded} - {level}:
        transition = params.transition(source, level)
        if (
            len(transition.steps) == 1
            and transition.steps[0].src_ring == params.ring_at(level + 1)
            and target_ring.is_quotient_of(transition.steps[0].src_ring)
        ):
            fused_levels.add(source)
    values, pure_sum = _dot_values(left, right, params.slot_count, prec=ctx._slot_prec)
    groups: dict[RingSpec, list[tuple[Plaintext | Ciphertext, Plaintext | Ciphertext, int, BigFloat]]] = defaultdict(list)
    plans: dict[tuple[int, int], tuple[int | None, RescaleStep, BigFloat]] = {}
    aligned: dict[int, Plaintext | Ciphertext] = {}
    for x, y in zip(left, right, strict=True):
        if not isinstance(x, Plaintext | Ciphertext):
            if not isinstance(y, Plaintext | Ciphertext):
                continue
            x, y = y, x
        if not isinstance(y, Plaintext | Ciphertext):
            bound = abs(y) if isinstance(y, Scalar) else encoding.max_abs(y, prec=ctx.encoding_precision_bits)
            if x.scale <= raw_scale_limit and target_ring.is_quotient_of(x.ring) and 2 * x.scale * (BigFloat(2) ** params.log_range) < target_ring.q:
                scale = target_scale**2 / x.scale
                poly, noise_info = encoding._encode(y, target_ring, scale=scale, msg_bound=bound, prec=ctx.encoding_precision_bits)
                pt = Plaintext(poly, noise_info=noise_info)
                groups[target_ring].append((x, pt, 1, BigFloat(1)))
                continue
            y = encode(y, msg_bound=bound, level=x.level)
        pair = (x.level, y.level)
        if pair not in plans:
            if pair == (level, level):
                plans[pair] = (None, RescaleStep(target_ring, target_ring, 1), BigFloat(1))
            elif x.level in fused_levels and y.level in fused_levels:
                step = RescaleStep.from_scales(params.ring_at(level + 1), target_ring, x.scale * y.scale, target_scale**2)
                plans[pair] = (None, step, step.poly_ratio)
            else:
                moduli = tuple(q for q in x.ring.moduli if q in y.ring.moduli)
                product_scale = x.scale * y.scale
                ring = RingSpec(n=params.n, moduli=moduli, dtype=params.dtype) if moduli else None
                if (
                    ring is not None
                    and set(moduli) - set(target_ring.moduli)
                    and 2 * max(x.scale, y.scale, product_scale * (BigFloat(2) ** params.log_range)) * (BigFloat(2) ** params.log_range) < ring.q
                ):
                    step = RescaleStep.from_scales(ring, target_ring, product_scale, target_scale**2)
                    ratio = step.poly_ratio
                    # Limit the added scale bias to one unit at the least precise
                    # input/working scale; otherwise convert the inputs first.
                    if step.k > 0 and abs(ratio * product_scale / target_scale**2 - 1) <= 1 / min(x.scale, y.scale, target_scale):
                        plans[pair] = (None, step, ratio)
                if pair not in plans:
                    plans[pair] = (level, RescaleStep(target_ring, target_ring, 1), BigFloat(1))
        align_level, step, ratio = plans[pair]
        if align_level is not None:
            inputs = []
            for value in (x, y):
                if value.level != align_level:
                    key = id(value)
                    if key not in aligned:
                        aligned[key] = value.to_level(align_level)
                    value = aligned[key]
                inputs.append(value)
            x, y = inputs
        groups[step.src_ring].append((x, y, step.k, ratio))

    totals: list[Poly | None] = [None, None, None]
    noise_terms: list[CKKSNoiseInfo] = []
    noise_products: dict[tuple[CKKSNoiseInfo, CKKSNoiseInfo, BigFloat], CKKSNoiseInfo] = {}
    for ring, terms in groups.items():
        sums: list[Poly | None] = [None, None, None]
        for x, y, k, ratio in terms:
            xs = (x.m,) if isinstance(x, Plaintext) else (x.b, x.a)
            ys = (y.m,) if isinstance(y, Plaintext) else (y.b, y.a)
            if x.ring != ring:
                xs = tuple(p.to_quotient_ring(ring) for p in xs)
            if y.ring != ring:
                ys = tuple(p.to_quotient_ring(ring) for p in ys)
            if k != 1:
                xs = tuple(p * k for p in xs)
            for i, a in enumerate(xs):
                for j, b in enumerate(ys):
                    product = a * b
                    previous = sums[i + j]
                    sums[i + j] = product if previous is None else previous + product
            noise_key = (x.noise_info, y.noise_info, ratio)
            noise = noise_products.get(noise_key)
            if noise is None:
                noise = x.noise_info * y.noise_info
                noise_products[noise_key] = noise = noise if ratio == 1 else noise * ratio
            noise_terms.append(noise)
        for i, component in enumerate(sums):
            if component is not None:
                converted = component if ring == target_ring else component.rescale(target_ring.moduli)
                previous = totals[i]
                totals[i] = converted if previous is None else previous + converted
    c0, c1, c2 = totals
    if c0 is None:
        raise ValueError("dot requires at least one encoded product")
    if c1 is None:
        c1 = Poly.zero(target_ring)
    if c2 is not None:
        rlk = context().keyset.rlk
        if rlk is None:
            raise RuntimeError("no relinearization key available")
        switched = _keyswitch(c2, rlk)
        c0 += switched.b
        c1 += switched.a
        noise_terms.append(replace(switched.noise_info, msg_bound=0))
    result = Ciphertext(a=c1, b=c0, noise_info=CKKSNoiseInfo.sum(noise_terms))._rescale(params.ring_at(level - 1).moduli)
    if pure_sum is not None:
        result += encode(pure_sum, msg_bound=encoding.max_abs(pure_sum, prec=ctx._slot_prec), level=level - 1)
    return result.replace(_values=values)


def horner_step(rotations: Iterable[int]) -> int | None:
    """The step :func:`sum_rotated` Horner-folds these rotations by, or
    ``None`` when they do not fold: they must be an arithmetic progression of
    multiples of its step.

    A family that folds needs only the ``+/- step`` rotation keys, which are
    its :func:`base_rotation_amounts`; one that does not needs a direct key
    per rotation.
    """
    values = sorted(set(rotations))
    if len(values) <= 1:
        return None
    step = values[1] - values[0]
    if any(b - a != step for a, b in zip(values, values[1:])) or any(r % step for r in values):
        return None
    return step


def sum_rotated(
    operands: Sequence[tuple[int, Ciphertext]],
    *,
    max_ks: int | None = None,
) -> Ciphertext:
    """Compute ``sum_i rho^{r_i}(x_i)``, picking a strategy from the operands.

    Strategies, in order of preference:

    1. When every nonzero ``r_i`` (mod ``n/2``) has a direct rotation key in
       the active context, key-switch each operand once and share a single
       rescale across the accumulated outputs (:func:`_sum_rotated_direct`).
    2. When the ``r_i`` form an arithmetic progression of multiples of its
       step, Horner-fold each sign side using only the ``+/- step`` keys
       (:func:`_sum_rotated_horner`).
    3. Otherwise rotate each operand independently and sum the results.

    ``max_ks`` bounds the keyswitch hops per rotation where a multi-hop
    rotation can occur: the generic fallback and the single-operand case.
    Strategies 1 and 2 are single-hop per rotation regardless.
    """
    if not operands:
        raise ValueError("sum_rotated: operands must be non-empty")

    ring = operands[0][1].ring
    for i, (_, x) in enumerate(operands):
        if x.ring != ring:
            raise ParameterMismatchError(f"sum_rotated: operand[{i}] ring does not match operand[0]")

    half_n = ring.n // 2
    if len(operands) == 1:
        r, x = operands[0]
        return x if r % half_n == 0 else x.rotate(r, max_ks=max_ks)

    level = operands[0][1].level
    direct = context().keyset._rotation_galois_elts_at(level)
    if all(steps_to_galois_elt(r % half_n, ring.n) in direct for r, _ in operands if r % half_n != 0):
        return _sum_rotated_direct(operands)

    # Preserve the signed schedule: centering rotations can require the opposite key.
    step = horner_step(r for r, _ in operands)
    if step is not None:
        return _sum_rotated_horner(operands, step=step)

    return Ciphertext.sum([x if r % half_n == 0 else x.rotate(r, max_ks=max_ks) for r, x in operands])


def _sum_rotated_horner(
    operands: Sequence[tuple[int, Ciphertext]],
    step: int,
) -> Ciphertext:
    """Compute ``sum_i rho^{r_i}(x_i)`` using only ``+/- step`` rotation keys.

    Every ``r_i`` must be a multiple of ``step``.  Operands are split by
    sign of ``r_i`` and Horner-folded on each side using ``+step``
    (positive side) or ``-step`` (negative side); ``r_i = 0`` is added
    directly.  ``+step`` must be a direct key in the active context when
    any ``r_i > 0``, and ``-step`` when any ``r_i < 0``.
    """
    if step == 0:
        raise ValueError("sum_rotated: step must be non-zero")

    # Coalesce duplicate r_i and validate divisibility.
    by_rot: dict[int, Ciphertext] = {}
    for r, x in operands:
        if r % step != 0:
            raise ValueError(f"sum_rotated: r={r} is not a multiple of step={step}")
        by_rot[r] = (by_rot[r] + x) if r in by_rot else x

    pos_ops = sorted((r, x) for r, x in by_rot.items() if r > 0)
    neg_ops = sorted(((r, x) for r, x in by_rot.items() if r < 0), reverse=True)
    zero_op = by_rot.get(0)

    # Validate +step / -step keys are present in the active context.
    first_ct = operands[0][1]
    base_galois_elts = context().keyset._rotation_galois_elts_at(first_ct.level)
    half_n = first_ct.ring.n // 2
    if pos_ops and steps_to_galois_elt(step % half_n, first_ct.ring.n) not in base_galois_elts:
        raise RuntimeError(f"sum_rotated: rotation key +{step} not in context")
    if neg_ops and steps_to_galois_elt(-step % half_n, first_ct.ring.n) not in base_galois_elts:
        raise RuntimeError(f"sum_rotated: rotation key -{step} not in context")

    def horner_side(ops: list[tuple[int, Ciphertext]], side_step: int) -> Ciphertext | None:
        """Horner-fold ``ops`` using ``side_step``; ``ops`` is sorted so each
        ``ops[i+1][0] - ops[i][0]`` is a positive multiple of ``side_step``."""
        if not ops:
            return None
        acc = ops[-1][1]
        for i in range(len(ops) - 2, -1, -1):
            n_chained = (ops[i + 1][0] - ops[i][0]) // side_step
            for _ in range(n_chained):
                acc = acc.rotate(side_step, max_ks=1)
            acc = ops[i][1] + acc
        for _ in range(ops[0][0] // side_step):
            acc = acc.rotate(side_step, max_ks=1)
        return acc

    parts = [p for p in (horner_side(pos_ops, step), horner_side(neg_ops, -step), zero_op) if p is not None]
    return reduce(add, parts)


def _sum_rotated_direct(operands: Sequence[tuple[int, Ciphertext]]) -> Ciphertext:
    """Compute ``sum_i rho^{r_i}(x_i)`` sharing a single rescale across all keyswitches.

    Every nonzero ``r_i`` (mod ``n/2``) must have a direct rotation key in the
    active context. Each ``x_i.a`` is digit-decomposed and key-switched without
    the trailing rescale, permuted by its own Galois automorphism in the
    extended ring, and accumulated there; the accumulated sum is rescaled once.
    The ``b`` components skip key-switching entirely: they are permuted and
    summed in the base ring and added after the rescale, together with any
    ``r_i = 0`` operands.
    """
    ctx = context()
    ring = operands[0][1].ring
    n = ring.n
    half_n = n // 2

    # Coalesce operands that share a rotation (equal mod n/2).
    by_rot: dict[int, Ciphertext] = {}
    for r, x in operands:
        key = r % half_n
        by_rot[key] = (by_rot[key] + x) if key in by_rot else x

    zero_op = by_rot.pop(0, None)
    if not by_rot:
        assert zero_op is not None
        return zero_op

    level = operands[0][1].level
    direct = ctx.keyset._rotation_galois_elts_at(level)
    for r in by_rot:
        if steps_to_galois_elt(r, n) not in direct:
            raise RuntimeError(f"sum_rotated: rotation key +{r} not in context")

    target_base = tuple(ring.moduli)
    # Extended-ring accumulation: un-rescaled keyswitch outputs, each permuted
    # by its own automorphism, all under the same secret and rescale factor.
    ks_acc = None
    b_acc = None
    b_noise = CKKSNoiseInfo.zero()
    for r, x in by_rot.items():
        galois_elt = steps_to_galois_elt(r, n)
        gk = ctx.keyset.get_galois_key(galois_elt)
        hoisted = _decompose_and_modraise(x.a, gk)
        ks = _keyswitch_product(hoisted, gk)
        ks = ks.map_polys(lambda poly: poly.automorphism(galois_elt))
        ks_acc = ks if ks_acc is None else ks_acc + ks

        b_rot = x.b.automorphism(galois_elt)
        b_acc = b_rot if b_acc is None else b_acc + b_rot
        b_noise = b_noise + x.noise_info

    assert ks_acc is not None and b_acc is not None
    rescaled = ks_acc._rescale(target_moduli=target_base)

    final_a = rescaled.a
    final_b = rescaled.b + b_acc
    final_noise = replace(rescaled.noise_info, msg_bound=0) + b_noise
    if zero_op is not None:
        final_a = final_a + zero_op.a
        final_b = final_b + zero_op.b
        final_noise = final_noise + zero_op.noise_info

    all_parts = ([(0, zero_op)] if zero_op is not None else []) + list(by_rot.items())
    rotations, parts = zip(*all_parts)
    final_plaintext_values = parts[0]._map_values(
        lambda *values: reduce(add, (np.roll(value, -r) for r, value in zip(rotations, values, strict=True))),
        *parts[1:],
    )
    return Ciphertext(a=final_a, b=final_b, noise_info=final_noise, _values=final_plaintext_values)
