import asyncio
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from math import isqrt

import pytest
import numpy as np
from flint import acb, arb, ctx

from ren.bigfloat import BigFloat
from ren.poly import Poly
from ren.ring import RingSpec
from ren.ckks import encoding
from ren.ckks.errors import InvalidParameterError

from ren.ckks.encoding import (
    _round_coeffs_flint,
    _coeffs_to_slots_flint,
    _slots_to_coeffs_flint,
    flint_precision,
    max_abs,
    scale_slots,
    slot_array,
)

from ren.ckks.utils import (
    bsgs_index,
    bsgs_steps,
    CallCounter,
    count_calls,
    is_prime_u64,
    select_moduli,
)


def test_primality_matches_trial_division():
    for n in range(-5, 4096):
        expected = n >= 2 and all(n % divisor for divisor in range(2, isqrt(n) + 1))
        assert is_prime_u64(n) is expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (2**31 - 1, True),
        (2**61 - 1, True),
        (2**64 - 59, True),
        (561, False),
        (3215031751, False),
        (341550071728321, False),
        (3825123056546413051, False),
        ((2**31 - 1) ** 2, False),
        (2**64 - 1, False),
    ],
)
def test_primality_large_values(n: int, expected: bool):
    assert is_prime_u64(n) is expected


def test_call_counter_accepts_bound_methods():
    class Example:
        def increment(self, value: int) -> int:
            return value + 1

    method = Example().increment
    tracked = count_calls(method)
    with CallCounter() as counter:
        assert tracked(1) == 2
    assert counter.counts == {method.__qualname__: 1}


def test_call_counter_counts_only_outermost_call():
    @count_calls
    def inner():
        pass

    @count_calls
    def outer():
        inner()

    with CallCounter() as counter:
        outer()

    assert counter.counts == {outer.__qualname__: 1}


def test_call_counter_names_inherited_methods_for_receiver():
    class Base:
        @count_calls
        def transform(self):
            pass

    class Child(Base):
        pass

    @count_calls
    def transform(value):
        pass

    child = Child()
    with CallCounter() as counter:
        child.transform()
        Base.transform(self=child)
        transform(child)

    assert counter.counts == {
        f"{Child.__qualname__}.transform": 2,
        transform.__qualname__: 1,
    }


def test_call_counter_restores_guard_after_exception():
    @count_calls
    def fail():
        raise RuntimeError

    @count_calls
    def succeed():
        pass

    with CallCounter() as counter:
        with pytest.raises(RuntimeError):
            fail()
        succeed()

    assert counter.counts == {
        fail.__qualname__: 1,
        succeed.__qualname__: 1,
    }


def test_call_counter_rejects_nesting_without_disturbing_outer_counter():
    @count_calls
    def tracked():
        pass

    with CallCounter() as outer:
        tracked()
        with pytest.raises(RuntimeError, match="only be one CallCounter"):
            with CallCounter():
                pass
        tracked()

    assert outer.counts == {tracked.__qualname__: 2}


def test_call_counter_is_task_local():
    @count_calls
    def tracked():
        pass

    async def count_once(counter: CallCounter):
        with counter:
            await asyncio.sleep(0)
            tracked()

    async def run_counters():
        counters = CallCounter(), CallCounter()
        await asyncio.gather(*(count_once(counter) for counter in counters))
        return counters

    counters = asyncio.run(run_counters())
    assert all(counter.counts == {tracked.__qualname__: 1} for counter in counters)


@pytest.mark.parametrize("bits", [100, 140, 256, 512])
@pytest.mark.parametrize("degree", range(8))
def test_flint_transforms_match_monomials(bits, degree):
    n = 8
    with flint_precision(bits + 80):
        # X^degree evaluated at the CKKS roots, independently of the FFT.
        slots = [acb(arb(degree * pow(5, i, 2 * n)) / n).exp_pi_i() for i in range(n // 2)]
        coeffs = [arb(int(k == degree)) for k in range(n)]
        with flint_precision(bits):
            encoded = _slots_to_coeffs_flint(np.array(slots, dtype=object))
            decoded = _coeffs_to_slots_flint(coeffs)
        threshold = arb(2) ** -(bits - 10)
        assert all(abs(a - b).abs_upper() < threshold for a, b in zip(encoded, coeffs, strict=True))
        assert all(abs(a - b).abs_upper() < threshold for a, b in zip(decoded, slots, strict=True))


@pytest.mark.parametrize("bits", [100, 140, 256, 512])
@pytest.mark.parametrize("count", [0, 1, 3, 32])
def test_flint_transforms_high_precision_roundtrip(bits, count):
    n = 64
    rng = random.Random(42)
    with flint_precision(bits + 80):
        values = [acb(arb(rng.getrandbits(bits - 20)) / 2 ** (bits - 20), arb(rng.getrandbits(bits - 20)) / 2 ** (bits - 20)) for _ in range(count)]
        with flint_precision(bits):
            encoded = _slots_to_coeffs_flint(slot_array(values, n // 2, dtype=object))
            decoded = _coeffs_to_slots_flint(encoded)
        expected = values + [acb(0)] * (n // 2 - count)
        assert all(abs(a - b).abs_upper() < arb(2) ** -(bits - 15) for a, b in zip(decoded, expected, strict=True))


@pytest.mark.parametrize("bits", [140, 256, 512])
def test_flint_encoding_integer_coefficients_match_direct_transform(bits):
    n = 16
    scale = BigFloat(2) ** (bits - 40)
    with flint_precision(bits + 80):
        values = [acb(arb(i + 1) / 17 + arb(2) ** -80, arb(2 - i) / 19) for i in range(n // 2)]
        # Invert the canonical embedding directly, including the conjugate roots.
        reference = []
        for k in range(n):
            total = acb(0)
            for i, value in enumerate(values):
                root = acb(arb(-k * pow(5, i, 2 * n)) / n).exp_pi_i()
                total += value * root + value.conjugate() * root.conjugate()
            reference.append((total / n).real)
        expected = _round_coeffs_flint(reference, scale=scale)
        with flint_precision(bits):
            actual = _slots_to_coeffs_flint(np.array(values, dtype=object))
            assert _round_coeffs_flint(actual, scale=scale) == expected


def test_numpy_rounding_ties_to_even():
    ring = RingSpec(n=8, moduli=(17,))
    rounded = []
    for value in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
        poly, _ = encoding._encode(np.full(ring.n // 2, value), ring, scale=BigFloat(1), msg_bound=3, prec=53)
        rounded.append(poly.to_centered_list()[0])
    assert rounded == [-2, -2, 0, 0, 2, 2]


@pytest.mark.parametrize("scale_offset", [0, 1])
def test_numpy_encoding_noise_covers_fft_roundoff(scale_offset):
    n, bound = 1024, 2**10
    ring = RingSpec(n=n, moduli=tuple(select_moduli(n=n, num_primes=3, bit_length=30)))
    scale = BigFloat(2) ** 50 + scale_offset
    rng = np.random.default_rng(2)
    values = rng.uniform(-1, 1, n // 2) + 1j * rng.uniform(-1, 1, n // 2)
    values *= bound / np.max(np.abs(values))
    poly, noise = encoding._encode(values, ring, scale=scale, msg_bound=bound, prec=53)
    with flint_precision(200):
        actual = encoding._decode(poly, scale=scale, prec=200)
        error = max(abs(a - acb(b)).abs_upper() for a, b in zip(actual, values, strict=True)) * arb(str(scale))
        assert error <= arb(str(noise.noise_bound))


@pytest.mark.parametrize("sign", [-1, 1])
def test_flint_rounding_half_away_from_zero(sign):
    with flint_precision(180):
        delta = arb(2) ** -100
        values = [sign * (arb("0.5") + offset) for offset in (-delta, arb(0), delta)]
        assert _round_coeffs_flint(values, scale=BigFloat(1)) == [0, sign, sign]


def test_flint_scaling_preserves_large_exact_scales():
    scale = BigFloat(2) ** 256
    with flint_precision(320):
        scaled = scale_slots(np.array([acb(1)], dtype=object), scale, prec=320).item()
        assert scaled == acb(2**256)
        assert _round_coeffs_flint([arb(1)], scale=scale) == [2**256]


def test_codec_precision_is_per_call():
    ring = RingSpec(n=16, moduli=tuple(select_moduli(n=16, num_primes=10, bit_length=30)))
    scale = BigFloat(2) ** 240
    with flint_precision(400):
        values = [acb(1 + arb(2) ** -180, arb(2) ** -200)]
    with flint_precision(53):
        encoding._decode(Poly.const(1, ring), scale=scale, prec=60)
        assert ctx.prec == 53
        with pytest.raises(InvalidParameterError, match="cannot certify"):
            encoding._encode(values, ring, scale=scale, msg_bound=2, prec=60)
        assert ctx.prec == 53
        encoded, _ = encoding._encode(values, ring, scale=scale, msg_bound=2, prec=320)
        assert ctx.prec == 53
        decoded = encoding._decode(encoded, scale=scale, prec=320)
        assert ctx.prec == 53
    with flint_precision(400):
        assert abs(decoded[0] - values[0]).abs_upper() < arb(2) ** -230
        assert abs(decoded[0] - 1).abs_lower() > arb(2) ** -181


def test_numerical_operations_keep_precision_across_threads():
    ready = Barrier(2)

    def worker(bits):
        ready.wait(timeout=5)

        def calculate():
            time.sleep(0.005)
            return (arb(1) + arb(2) ** -(bits - 20) - 1).mid()

        return [encoding.apply(calculate, prec=bits) for _ in range(4)]

    with flint_precision(200):
        expected = [arb(2) ** -(bits - 20) for bits in (60, 180)]
    previous = ctx.prec
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, (60, 180)))
    assert all(value == target for values, target in zip(results, expected, strict=True) for value in values)
    assert ctx.prec == previous
    with pytest.raises(ZeroDivisionError):
        encoding.apply(lambda: 1 / 0, prec=180)
    assert ctx.prec == previous


def _covers(n: int, diagonals: list[int], baby_steps: tuple[int, ...], giant_steps: tuple[int, ...]) -> bool:
    reachable = {(b + g) % n for b in baby_steps for g in giant_steps}
    return all(d % n in reachable for d in diagonals)


def test_bsgs_steps_keeps_a_stride_coprime_to_n():
    """127 shares no factor with n, but the diagonals are still 255 multiples of it."""
    n = 32768
    diagonals = [(127 * k) % n for k in range(-127, 128)]

    baby_steps, giant_steps = bsgs_steps(n=n, diagonals=diagonals)

    assert baby_steps == tuple(127 * b for b in range(16))
    assert giant_steps == tuple(127 * 16 * g for g in range(-8, 8))
    assert _covers(n, diagonals, baby_steps, giant_steps)


def test_bsgs_steps_reads_a_one_sided_family_from_zero():
    n = 32768
    diagonals = [128 * i for i in range(256)]

    baby_steps, giant_steps = bsgs_steps(n=n, diagonals=diagonals)

    assert baby_steps == tuple(128 * b for b in range(16))
    assert giant_steps == tuple(2048 * g for g in range(16))


@pytest.mark.parametrize("stride", [1, 3, 512])
@pytest.mark.parametrize("half_width", [1, 3, 7, 31, 100])
def test_bsgs_steps_splits_a_symmetric_family_around_zero(stride: int, half_width: int):
    """Symmetric families (every bootstrap stage) get ``s = ceil(sqrt(2l + 1))``
    babies and giants on multiples of ``s * stride`` aligned to 0."""
    n = 1 << 20  # wide enough that no family wraps around the circle
    diagonals = [(stride * k) % n for k in range(-half_width, half_width + 1)]
    s = math.ceil(math.sqrt(2 * half_width + 1))

    baby_steps, giant_steps = bsgs_steps(n=n, diagonals=diagonals)

    assert baby_steps == tuple(stride * b for b in range(s))
    assert giant_steps == tuple(stride * s * g for g in range(-half_width // s, half_width // s + 1))
    assert _covers(n, diagonals, baby_steps, giant_steps)


def test_bsgs_steps_lists_only_the_steps_it_uses():
    n = 1024
    diagonals = [5, 700, 701]

    baby_steps, giant_steps = bsgs_steps(n=n, diagonals=diagonals)

    assert _covers(n, diagonals, baby_steps, giant_steps)
    assert len(baby_steps) <= len(diagonals) and len(giant_steps) <= len(diagonals)


def test_bsgs_index_splits_each_diagonal_once_across_the_wrap():
    """With t = n/16, -t is 3t + (-4t), although 3t + 12t also reaches it mod n."""
    n = 1024
    t = n // 16
    diagonals = [t * m for m in range(-1, 14)]

    index = bsgs_index(n=n, diagonals=diagonals)

    babies = tuple(t * b for b in range(4))
    assert index == {-4 * t: (3 * t,), 0: babies, 4 * t: babies, 8 * t: babies, 12 * t: (0, t)}


def test_bsgs_steps_of_nothing_is_empty():
    assert bsgs_steps(n=16, diagonals=[]) == ((), ())
    assert bsgs_index(n=16, diagonals=[]) == {}


def test_slot_magnitude_bounds_overlapping_enclosures():
    with flint_precision(180):
        values = [acb(arb("1 +/- 0.1")), acb(arb("1.05 +/- 0.001"))]
        for order in (values, values[::-1]):
            assert max_abs(order, prec=180) >= 1.1


@pytest.mark.parametrize("offset,exponent", [(1, -80), (0, -1100)])
def test_slot_magnitude_bound_rounds_outward(offset, exponent):
    with flint_precision(180):
        value = offset + arb(2) ** exponent
        bound = max_abs([acb(value)], prec=180)
        assert arb(bound) >= value


@pytest.mark.parametrize("broadcast", [False, True])
def test_slot_scaling_preserves_precision_and_enclosures(broadcast):
    with flint_precision(256):
        value = acb(1 + arb(2) ** -80, arb(2) ** -90)
        factor = BigFloat(1) / 3
        exact = value / 3
        source = np.array([value], dtype=object)
        slots = np.broadcast_to(source, (4,)) if broadcast else np.repeat(source, 4)
        scaled = scale_slots(slots, factor, prec=140)
        source[0] = acb(7)
        for result in scaled:
            assert exact.real in result.real
            assert exact.imag in result.imag
            assert abs(result - exact).abs_upper() < arb(2) ** -130
