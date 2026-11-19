"""CKKS slot preparation and numerical encoding transforms."""

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from functools import cache, lru_cache
from math import inf, isfinite, nextafter, pi, sqrt, ulp
from threading import RLock
from typing import Any

import numpy as np
from flint import acb, arb, ctx as flint_ctx, fmpq

from ren.bigfloat import BigFloat, FloatLike
from ren.ckks.errors import InvalidParameterError
from ren.poly import Poly
from ren.ring import RingSpec

from .metadata import CKKSNoiseInfo


Scalar = complex | float | int | np.integer
type RawSlots = Scalar | arb | acb | Sequence[Scalar | arb | acb] | np.ndarray

FLOAT_PRECISION_BITS = 53
_flint_lock = RLock()


def _normalize_precision(prec: int) -> int:
    """Validate the bit count and use at least 53 bits."""
    if prec <= 0:
        raise InvalidParameterError(f"prec must be positive, got {prec}")
    return max(prec, FLOAT_PRECISION_BITS)


@contextmanager
def flint_precision(precision_bits: int) -> Generator[None]:
    """
    Context manager that sets ``flint.ctx.prec`` to ``precision_bits`` for the
    duration of the ``with`` block and restores the previous value on exit
    (including on exception). Nested calls share the same reentrant lock.
    """
    with _flint_lock:
        old_precision = flint_ctx.prec
        flint_ctx.prec = precision_bits
        try:
            yield
        finally:
            flint_ctx.prec = old_precision


def apply[T](fn: Callable[..., T], *values: Any, prec: int) -> T:
    """Run a numerical operation at explicit precision, serializing FLINT access."""
    with flint_precision(prec):
        return fn(*values)


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def slot_array(values: RawSlots, slot_count: int, *, dtype: type | np.dtype) -> np.ndarray:
    """Normalize slot shape, broadcasting scalars and zero-padding short vectors.

    The caller selects the numerical representation; object arrays retain the
    original input values for subsequent FLINT conversion.
    """
    values = _native(values)
    if isinstance(values, str | bytes) or not isinstance(values, Scalar | arb | acb | Sequence | np.ndarray):
        raise TypeError(f"Unrecognized slot input type: {type(values).__name__}")
    if isinstance(values, np.ndarray) and values.shape == (slot_count,) and values.strides == (0,):
        values = values[:1].reshape(())
    raw = np.asarray(values, dtype=dtype)
    if raw.ndim == 0:
        return np.broadcast_to(raw, (slot_count,))
    if raw.ndim != 1:
        raise InvalidParameterError(f"slot values must be one-dimensional, got shape {raw.shape}")
    if raw.size > slot_count:
        raise InvalidParameterError(f"too many slots for ring dimension, got {raw.size}, max={slot_count}")
    if raw.size == slot_count:
        return raw
    padded = np.zeros(slot_count, dtype=raw.dtype)
    padded[: raw.size] = raw
    return padded


def slots(values: RawSlots, slot_count: int, *, prec: int) -> np.ndarray:
    """Convert slots to FLINT values, keeping scalars as broadcast views."""
    raw = slot_array(values, slot_count, dtype=object)
    with flint_precision(prec):
        if raw.strides == (0,):
            return np.broadcast_to(np.array(acb(_native(raw[0])), dtype=object), (slot_count,))
        # Numeric ndarrays already convert their elements to Python scalars in slot_array.
        elements = raw if isinstance(values, np.ndarray) and values.dtype != object else map(_native, raw)
        return np.fromiter(map(acb, elements), dtype=object, count=slot_count)


@cache
def _slot_perm(n: int) -> np.ndarray:
    """
    Permutation π(i) = (5^i mod 2n - 1) // 2 mapping CKKS slot ``i`` to its
    coefficient index. The generator 5 makes the Galois automorphism σ_5
    correspond to slot rotation by one.
    """
    perm = np.empty(n // 2, dtype=np.int64)
    value, m = 1, 2 * n
    for i in range(n // 2):
        perm[i] = (value - 1) // 2
        value = (5 * value) % m
    return perm


@cache
def _np_half_twists(n: int) -> np.ndarray:
    """``ζ^k`` for ``k < n/2``, with ``ζ = exp(iπ/n)`` the primitive 2n-th root of unity."""
    return np.exp((1j * pi / n) * np.arange(n // 2, dtype=np.float64))


def _slots_to_coeffs_np(slots: np.ndarray) -> np.ndarray:
    """Convert normalized slots to ``n`` real coefficients with a half-size FFT.

    Slot ``i`` is placed at evaluation point ζ^(5^i).

    The coefficients are ``c_k = (2/n) Re(sum_i z_i ζ^(-5^i k))``. Every
    ``5^i`` is ``4m + 1`` for a distinct ``m < n/2``, so the sum is ``ζ^(-k)``
    times a length-``n/2`` DFT of the slots placed at their ``m``. And since
    ``ζ^(n/2) = i``, the DFT value at ``k`` yields both ``c_k`` (its real part)
    and ``c_(k + n/2)`` (its imaginary part): one half-size transform, the
    "special FFT" of Lattigo and OpenFHE.
    """
    n = 2 * len(slots)
    placed = np.empty(n // 2, dtype=np.complex128)
    placed[_slot_perm(n) // 2] = slots
    half = np.fft.fft(placed, out=placed)
    half *= np.conjugate(_np_half_twists(n)) * (2 / n)
    return np.concatenate([half.real, half.imag])


def _coeffs_to_slots_np(coeffs: np.ndarray) -> np.ndarray:
    """Invert ``_slots_to_coeffs_np`` to recover ``n // 2`` slots from real coefficients."""
    n = len(coeffs)
    folded = (coeffs[: n // 2] + 1j * coeffs[n // 2 :]) * _np_half_twists(n)
    return (np.fft.ifft(folded) * (n // 2))[_slot_perm(n) // 2]


@lru_cache(maxsize=4)
def _flint_negacyclic_twists(n: int, precision_bits: int) -> tuple[tuple[acb, ...], tuple[acb, ...]]:
    """Forward and inverse twists at the active FLINT precision.

    ``precision_bits`` is part of the cache key and must match ``flint.ctx.prec``.
    """
    m = n // 2
    roots = [acb(arb(-k) / n).exp_pi_i() for k in range(m)]
    return tuple(root / m for root in roots), tuple(root.conjugate() * m for root in roots)


def _round_coeffs_flint(coeffs: Sequence[arb], *, scale: BigFloat) -> list[int]:
    """Round scaled coefficients at the active FLINT precision, rejecting uncertain results."""
    flint_scale = arb(fmpq(*scale.as_integer_ratio()))
    half = arb("0.5")
    result = []
    for coeff in coeffs:
        scaled = coeff * flint_scale
        if scaled < 0:
            rounded = (scaled - half).ceil()
        else:
            rounded = (scaled + half).floor()
        # All values in the interval must round to the same integer.
        integer = rounded.unique_fmpz()
        if integer is None:
            raise InvalidParameterError("cannot certify half-unit rounding; increase prec or use tighter input bounds")
        result.append(int(integer))
    return result


def _slots_to_coeffs_flint(slots: np.ndarray) -> list[arb]:
    """Convert normalized slots to real coefficients at the active FLINT precision.

    Slot ``i`` is placed at evaluation point ζ^(5^i), as in ``_slots_to_coeffs_np``.
    """
    n = 2 * len(slots)
    forward, _ = _flint_negacyclic_twists(n, flint_ctx.prec)
    placed: list[complex | arb | acb] = [acb(0)] * (n // 2)
    for value, i in zip(slots, _slot_perm(n), strict=True):
        placed[int(i) // 2] = _native(value)
    half = [value * twist for value, twist in zip(acb.dft(placed), forward, strict=True)]
    return [value.real for value in half] + [value.imag for value in half]


def _coeffs_to_slots_flint(coeffs: Sequence[arb]) -> list[acb]:
    """Recover ``n // 2`` FLINT slots from real coefficients at the active precision."""
    n = len(coeffs)
    _, inverse = _flint_negacyclic_twists(n, flint_ctx.prec)
    m = n // 2
    folded = [acb(coeffs[k], coeffs[k + m]) * inverse[k] for k in range(m)]
    evaluated = acb.dft(folded, inverse=True)
    return [evaluated[int(i) // 2] for i in _slot_perm(n)]


def scale_slots(values: np.ndarray, factor: BigFloat, *, prec: int) -> np.ndarray:
    """Scale FLINT slots, keeping scalars as broadcast views."""
    compact = values[:1] if values.strides == (0,) else values
    with flint_precision(prec):
        result = compact * acb(fmpq(*factor.as_integer_ratio()))
    return np.broadcast_to(result, values.shape)


def max_abs(values: RawSlots, *, prec: int) -> float:
    """Bound slot magnitudes, evaluating broadcast constants only once."""
    if isinstance(values, np.ndarray) and values.strides == (0,):
        values = values[:1]
    raw = np.asarray(values, dtype=np.complex128 if prec <= FLOAT_PRECISION_BITS else object)
    if raw.dtype != object:
        return float(np.max(np.abs(raw), initial=0))
    with flint_precision(prec):
        upper = max((acb(_native(value)).abs_upper() for value in raw.flat), default=arb(0))
        bound = float(upper)
        return nextafter(bound, inf) if upper > bound else bound


def _encode(values: RawSlots, ring: RingSpec, *, scale: BigFloat, msg_bound: FloatLike, prec: int) -> tuple[Poly, CKKSNoiseInfo]:
    """Encode slots into a polynomial with bounds on the message and encoding error."""
    prec = _normalize_precision(prec)
    error = BigFloat(0)
    if prec == FLOAT_PRECISION_BITS:
        raw = slot_array(values, ring.n // 2, dtype=np.complex128)
        # Each component rounds to nearest with relative error <= 2^-53.
        # One subnormal unit also covers the two components near zero.
        error = msg_bound * scale * 2**-53 + scale * ulp(0.0)
        if raw.strides == (0,):
            value = complex(raw[0])
            coeffs = [round(value.real * scale), round(value.imag * scale)]
        else:
            coeffs = np.rint(_slots_to_coeffs_np(raw) * float(scale))
            # Budget eight unit roundoffs per FFT stage, plus twist and scaling.
            # The normalized coefficient 2-norm is <= coeff_bound; evaluating
            # its error at a slot costs at most sqrt(n) by Cauchy-Schwarz.
            roundings = 8 * ring.n.bit_length()
            roundoff = roundings * BigFloat(2) ** -53
            coeff_bound = BigFloat(msg_bound) * (1 + BigFloat(2) ** -53) + ulp(0.0)
            float_scale = BigFloat(float(scale))
            scale_error = abs(float_scale - scale) + abs(float_scale) * roundoff / (1 - roundoff)
            error += sqrt(ring.n) * coeff_bound * scale_error + ring.n * (abs(float_scale) * roundings + 1) * ulp(0.0)
    else:
        with flint_precision(prec):
            raw = slot_array(values, ring.n // 2, dtype=object)
            if raw.strides == (0,):
                value = acb(_native(raw[0]))
                coeffs = _round_coeffs_flint([value.real, value.imag], scale=scale)
            else:
                coeffs = _round_coeffs_flint(_slots_to_coeffs_flint(raw), scale=scale)

    if raw.strides == (0,):
        real, imag = coeffs
        poly = Poly.const(real, ring)
        if imag != 0:
            poly += Poly.monomial(imag, ring.n // 2, ring)
        # A tighter public bound for real scalars would reveal that the imaginary part is zero.
        noise_bound = sqrt(0.5)
        noise_heuristic_bound = 0.5
    else:
        poly = Poly(coeffs, ring)
        noise_bound = noise_heuristic_bound = ring.n / pi

    noise_info = CKKSNoiseInfo(
        noise_bound=noise_bound + error,
        noise_heuristic_bound=noise_heuristic_bound + error,
        msg_bound=msg_bound * scale,
    )
    return poly, noise_info


def _decode(poly: Poly, *, scale: BigFloat, prec: int) -> np.ndarray:
    """Decode to ``complex128`` at 53 bits, or FLINT ``acb`` values at higher precision."""
    prec = _normalize_precision(prec)
    centered = poly.to_centered_list()
    if prec > FLOAT_PRECISION_BITS:
        with flint_precision(prec):
            flint_scale = arb(fmpq(*scale.as_integer_ratio()))
            scaled = [arb(int(c)) / flint_scale for c in centered]
            return np.array(_coeffs_to_slots_flint(scaled), dtype=object)
    try:
        float_scale = float(scale)
        if not isfinite(float_scale) or (scale and float_scale == 0):
            raise OverflowError
        scaled = [float(c) / float_scale for c in centered]
    except OverflowError:
        scaled = [float(c / scale) for c in centered]
    if any(not isfinite(c) for c in scaled):
        max_bits = max(centered, key=abs).bit_length()
        raise ValueError(f"Decoding failed: scaled coefficient too large for float ({max_bits} coefficient bits). Plaintext may be corrupted.")
    return _coeffs_to_slots_np(np.asarray(scaled, dtype=np.float64))
