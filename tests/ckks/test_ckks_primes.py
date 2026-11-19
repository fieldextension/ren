"""Tests for the modulus chains generate_prefix builds."""

from fractions import Fraction
from itertools import combinations
from math import ceil, isqrt, prod

import pytest

from ren.dtype import dtypes
from ren.ckks import BootstrappingParams, CKKSLevel, CKKSParams
from ren.ckks.primes import _closest_product, _primes_near, find_closest_almost_prime
from ren.ckks.utils import is_prime_u64

LOGN = 16
MESSAGE_RATIO = 256.0

DEPTH_9 = BootstrappingParams(cts_depth=2, stc_depth=2, exp_degree=16, squaring_count=0, failure_probability=2**-16)
DEPTH_10 = BootstrappingParams(cts_depth=2, stc_depth=2, exp_degree=31, squaring_count=1, failure_probability=2**-16)
DEPTH_15 = BootstrappingParams(cts_depth=3, stc_depth=3, exp_degree=63, squaring_count=3, failure_probability=2**-64)

# (compute_bits, boot_bits, compute_count, dtype, spec)
PARAM_COMBOS = [
    (50, 61, 10, dtypes.uint32, DEPTH_9),
    (30, 60, 8, dtypes.uint32, DEPTH_9),  # 45-bit uint32 entries drift past sqrt(2) within five levels
    (50, 61, 10, dtypes.uint64, DEPTH_15),
    (40, 60, 10, dtypes.uint64, DEPTH_9),
    (62, 80, 10, dtypes.uint64, DEPTH_10),  # larger precision forcing k > 1
]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("target", [1, 3, 6, 17, 23, 24, 1155, 1156, 2000, 5000, 10000, 30000, 100000])
def test_closest_prime_search_matches_exhaustive_result(k: int, target: int):
    primes = [3, 5, 7, 11, 13, 17, 19, 23]
    expected = min(
        combinations(primes, k),
        key=lambda factors: (max(Fraction(prod(factors), target), Fraction(target, prod(factors))), -prod(factors)),
    )
    assert _closest_product(primes, target, k) == expected


@pytest.mark.parametrize("k", [2, 3, 4])
def test_closest_prime_search_resolves_near_ratio_ties(k: int):
    primes = [2135031809, 2135162881, 2135818241, 2142502913, 2144468993]
    lower = tuple(primes[:k])
    upper = (*primes[: k - 1], primes[k])
    target = isqrt(prod(lower) * prod(upper))
    # These are the two smallest products. The exact ratios favor the lower
    # product, but floating-point ratios round to the same value.
    assert Fraction(target, prod(lower)) < Fraction(prod(upper), target)
    assert _closest_product(primes, target, k) == lower


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("minimum_product", [1, 17, 2000, 1000000])
def test_closest_prime_search_with_minimum_matches_exhaustive_result(k, minimum_product):
    primes = [3, 5, 7, 11, 13, 17, 19, 23]
    eligible = [factors for factors in combinations(primes, k) if prod(factors) >= minimum_product]
    for target in (1, 17, 100, 2000, 10000, 30000, 100000):
        if not eligible:
            with pytest.raises(ValueError):
                _closest_product(primes, target, k, minimum_product)
        else:
            expected = min(eligible, key=lambda factors: (max(Fraction(prod(factors), target), Fraction(target, prod(factors))), -prod(factors)))
            assert _closest_product(primes, target, k, minimum_product) == expected


def test_uint32_minimum_product_does_not_get_stuck_below_bound():
    assert find_closest_almost_prime(target=786434, minimum_product=786434, logn=16, dtype=dtypes.uint32, banned_factors=(), k=1) == (1179649,)


def test_nearest_prime_search_respects_uint64_limit():
    (prime,) = find_closest_almost_prime(target=(1 << 63) + (1 << 30), logn=LOGN, dtype=dtypes.uint64, banned_factors=(), k=1)
    assert prime < 1 << 63


def test_nearest_prime_search_rejects_empty_uint64_range():
    with pytest.raises(ValueError):
        find_closest_almost_prime(target=1 << 60, logn=LOGN, dtype=dtypes.uint64, banned_factors=(), k=1, minimum_factor=1 << 63)


@pytest.mark.parametrize(
    "target, k, banned, expected",
    [
        (14, 1, (), (17,)),
        (100, 2, (), (5, 29)),
        (225, 2, (), (5, 53)),
        (100, 2, (13,), (5, 29)),
        (1120, 3, (), (5, 13, 29)),
    ],
)
def test_nearest_prime_search_reuses_factors_after_rejected_product(target, k, banned, expected):
    factors = find_closest_almost_prime(target=target, minimum_product=target, logn=1, dtype=dtypes.uint64, banned_factors=banned, k=k)
    assert factors == expected
    assert prod(factors) >= target


@pytest.mark.parametrize("m, lower, upper", [(2, 2, 40), (3, 2, 53), (4, 5, 30), (8, 17, 42), (16, 18, 97), (8, 97, 97)])
def test_prime_iterator_matches_exhaustive_result(m: int, lower: int, upper: int):
    primes = [p for p in range(lower, upper) if p % m == 1 and all(p % divisor for divisor in range(2, isqrt(p) + 1))]
    for target in range(lower - m, upper + m + 1):
        assert list(_primes_near(target, m, lower, upper)) == sorted(primes, key=lambda p: (abs(p - target), p))


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
def test_prime_search_preserves_exclusions_and_factor_bounds(dtype):
    banned = [1179649, 2752513]
    original = banned.copy()
    factors = find_closest_almost_prime(
        target=1 << 50, minimum_product=1 << 50, minimum_factor=1 << 20, logn=LOGN, dtype=dtype, banned_factors=banned, k=2
    )
    assert banned == original
    assert not set(factors) & set(banned)
    assert len(factors) == len(set(factors)) == 2
    assert all(p >= 1 << 20 and p % (1 << (LOGN + 1)) == 1 for p in factors)
    assert prod(factors) >= 1 << 50


@pytest.mark.parametrize("k", [1, 2, 3])
def test_uint64_prime_search_rejects_impossible_minimum(k):
    with pytest.raises(ValueError):
        find_closest_almost_prime(
            target=1 << (63 * k),
            minimum_product=1 << (63 * k),
            minimum_factor=(1 << 63) - (1 << 20),
            logn=LOGN,
            dtype=dtypes.uint64,
            banned_factors=(),
            k=k,
        )


@pytest.mark.parametrize(
    "bits, previous",
    [(80, (1099510054913, 1099512938497)), (140, (111703415980033, 111703417159681, 111703418601473))],
)
def test_uint64_product_search_improves_independent_prime_selection(bits, previous):
    target = 1 << bits
    factors = find_closest_almost_prime(target=target, logn=LOGN, dtype=dtypes.uint64, banned_factors=(), k=len(previous))
    assert max(Fraction(prod(factors), target), Fraction(target, prod(factors))) < max(
        Fraction(prod(previous), target), Fraction(target, prod(previous))
    )


def _within_factor_two(product: int, target: int) -> bool:
    return max(product / target, target / product) <= 2.0


def _increments(levels: tuple[CKKSLevel, ...]) -> list[tuple[int, ...]]:
    return [tuple(sorted(set(upper.moduli) - set(lower.moduli))) for lower, upper in zip(levels[:-1], levels[1:], strict=True)]


@pytest.mark.parametrize(
    "compute_bits, boot_bits, compute_count, dtype, spec",
    PARAM_COMBOS,
    ids=[f"c{c}b{b}cc{cc}bc{s.depth}_{d.name[:3]}" for c, b, cc, d, s in PARAM_COMBOS],
)
def test_bootstrap_prefix_chain(compute_bits, boot_bits, compute_count, dtype, spec):
    log_range = boot_bits - compute_bits - 1
    params = CKKSParams.generate_prefix(
        log_n=LOGN,
        depth=compute_count,
        log_precision=compute_bits,
        log_range=log_range,
        dtype=dtype,
        boot_params=spec,
        bootstrap_level_bit_length=boot_bits,
        bootstrap_message_ratio=MESSAGE_RATIO,
        security=None,
    )
    levels = params.levels
    assert params.boot_params is not None
    base_level = params.bootstrap_base_level
    assert base_level is not None

    assert len(levels) == compute_count + spec.depth + 1
    ntt_modulus = 1 << (LOGN + 1)
    all_primes = [*levels[-1].moduli, *params.ks_primes]  # the base and every level are prefixes of the highest level
    assert all(is_prime_u64(p) and p % ntt_modulus == 1 for p in all_primes)
    assert len(set(all_primes)) == len(all_primes)

    # Base q0 tracks the boot target; level 0 is q0 plus a q1 sized for the bootstrap input.
    q0 = prod(base_level.moduli)
    assert _within_factor_two(q0, 1 << boot_bits)
    assert set(base_level.moduli) <= set(levels[0].moduli)
    assert prod(levels[0].moduli) >= ceil(levels[0].scale * (2**log_range) * MESSAGE_RATIO)

    increments = _increments(levels)
    for increment in increments[:compute_count]:
        assert _within_factor_two(prod(increment), 1 << compute_bits)
    for increment in increments[compute_count:]:
        assert _within_factor_two(prod(increment), 1 << boot_bits)
