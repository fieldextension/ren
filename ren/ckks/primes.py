"""
Generate CKKS RNS modulus chains from NTT-friendly primes.

The top-level entry point is :meth:`CKKSParams.generate_prefix`, which builds a
complete CKKS modulus chain consisting of a base modulus, a sequence of
compute-level moduli, and a sequence of boot-level moduli.

Each entry in the chain is a product of ``k`` distinct primes (an
"almost-prime").  Within :meth:`CKKSParams.generate_prefix`, ``k`` is chosen
per section by :func:`_k_for_target` as the smallest value for which
``k * max_prime_bits >= log2(target) + 1``, where ``max_prime_bits`` is
31 for ``uint32`` or 63 for ``uint64``.

Under the hood the chain is assembled from two helper layers:

* :func:`almost_prime_chain` -- produces a sequence of almost-primes whose
  associated scale factors stay within a sqrt(2) factor of a target value.
* :func:`find_closest_almost_prime` -- finds a single ``k``-almost-prime
  close to a given target.

For ``uint32``, all candidate primes are less than 2^31 and congruent to
1 modulo 2^(logn+1).  The full pool is cached and searched for the product
closest to the target by ratio.

For ``uint64``, candidate primes are less than 2^63 and congruent to
1 modulo 2^(logn+1). A bounded pool is drawn around ``target^(1/k)`` and
searched by the same product optimizer. Supports k in {1, 2, 3}.
"""

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from functools import cache
from heapq import merge
from itertools import islice
from math import ceil, isqrt, log2, prod

from ren.bigfloat import BigFloat
from ren.dtype import DType, dtypes

from .utils import is_prime_u64

UINT64_PRIME_CANDIDATES = 256


@cache
def _prime_pool(logn: int) -> tuple[int, ...]:
    """Sorted uint32 NTT-prime pool: p < 2^31 and p ≡ 1 (mod 2^(logn+1)), cached by logn."""
    return tuple(_primes_near(2, 1 << (logn + 1), 2, 1 << 31))


def _primes_near(target: int, m: int, lower: int, upper: int) -> Iterator[int]:
    """Yield primes ≡ 1 (mod *m*) in ``[lower, upper)`` by distance from
    *target*, preferring the smaller prime on ties."""
    base = ((target - 1) // m) * m + 1
    low = range(min(base, ((upper - 2) // m) * m + 1), lower - 1, -m)
    high = range(max(base + m, ((lower + m - 2) // m) * m + 1), upper, m)
    return (p for p in merge(low, high, key=lambda p: abs(p - target)) if is_prime_u64(p))


def find_closest_almost_prime(
    *,
    target: int,
    logn: int,
    dtype: DType,
    banned_factors: Sequence[int],
    k: int | None = None,
    minimum_product: int = 1,
    minimum_factor: int | None = None,
) -> tuple[int, ...]:
    """Find *k* distinct NTT-friendly primes whose product approximates
    *target*.

    For ``uint32`` this performs an exhaustive combinatorial search over a
    cached pool of all primes below 2^31, returning the product closest to
    *target* by ratio.  Supports k in {1, 2, 3, 4}.

    For ``uint64`` the same product search uses up to 128 eligible primes on
    each side of ``target^(1/k)``. The pool expands upward if needed to meet
    ``minimum_product``. This is optimal within that pool, not across all
    uint64 primes. For k=1, just the prime on each side is needed. Supports
    k in {1, 2, 3}.

    Args:
        target: The value to approximate.
        logn: Log-base-2 of the polynomial ring degree.  The NTT
            compatibility condition is ``p ≡ 1 (mod 2^(logn+1))``.
        banned_factors: Primes that must *not* appear in the result.
        dtype: ``dtypes.uint32`` or ``dtypes.uint64``.
        k: Number of distinct prime factors. Defaults to the fewest primes of
            the dtype's width that fit *target* with headroom.
        minimum_product: Optional inclusive lower bound on the selected
            product.
        minimum_factor: Inclusive lower bound on every selected prime.
            Defaults to just above the square root of the largest banned
            factor, so later draws cannot collide with earlier products.

    Returns:
        A sorted tuple of *k* distinct primes whose product approximates
        *target*.

    Raises:
        ValueError: If *k* is out of the supported range for the dtype, or
            fewer than *k* eligible primes can be found.
    """
    if k is None:
        k = _k_for_target(target, dtype)
    if minimum_factor is None:
        minimum_factor = _default_minimum_factor(banned_factors)
    if minimum_product < 1:
        raise ValueError("minimum_product must be positive")
    if minimum_factor < 2:
        raise ValueError("minimum_factor must be at least two")

    banned = set(banned_factors)
    search_target = max(target, minimum_product, minimum_factor**k)
    if dtype is dtypes.uint32:
        if not 1 <= k <= 4:
            raise ValueError(f"k must be between 1 and 4 for uint32, got {k}")
        primes = [p for p in _prime_pool(logn) if p not in banned and p >= minimum_factor]
        if len(primes) < k:
            raise ValueError(f"Need at least {k} eligible primes but only {len(primes)} remain after excluding {len(banned)} banned factors")
    elif dtype is dtypes.uint64:
        if not 1 <= k <= 3:
            raise ValueError(f"k must be between 1 and 3 for uint64, got {k}")
        m = 1 << (logn + 1)
        center = search_target if k == 1 else round(search_target ** (1.0 / k))
        primes = []
        for lower, upper in ((minimum_factor, min(center + 1, 1 << 63)), (max(minimum_factor, center + 1), 1 << 63)):
            primes.extend(
                islice((p for p in _primes_near(center, m, lower, upper) if p not in banned), 1 if k == 1 else UINT64_PRIME_CANDIDATES // 2)
            )
        primes.sort()
        if primes:
            lower = primes[-1] + 1
            higher = (p for p in _primes_near(lower, m, lower, 1 << 63) if p not in banned)
            while prod(primes[-k:]) < minimum_product:
                if (prime := next(higher, None)) is None:
                    break
                primes.append(prime)
    else:
        raise TypeError(f"unsupported dtype for modulus chain generation: {dtype}")

    return _closest_product(primes, target, k, minimum_product)


_SQRT2 = BigFloat(2).sqrt()


def almost_prime_chain(
    *,
    target: int,
    logn: int,
    length: int,
    dtype: DType,
    banned_factors: Sequence[int] = (),
    k: int | None = None,
    minimum_factor: int | None = None,
) -> tuple[list[tuple[int, ...]], list[BigFloat]]:
    """Produce a chain of almost-primes with scale factors tracking *target*.

    The prefix schedule generator uses this for the compute and boot-top
    sections of the CKKS modulus chain.  Each entry is a ``k``-almost-prime;
    ``k`` defaults to the fewest primes of the dtype's width that fit
    *target* (see :func:`_k_for_target`).

    The first entry has an implicit scale factor equal to *target* (exact).
    Each subsequent entry's scale is ``prev_scale**2 / product(prev_factors)``,
    which drifts slightly from *target*.  The chain is ordered so that
    entry 0 carries the exact scale and later entries accumulate drift;
    the generator places entry 0 at the *top* of its section so that
    accuracy degrades downward.

    No prime appears in more than one entry or in *banned_factors*.

    Args:
        target: The target scale value.
        logn: Log-base-2 of the polynomial ring degree.
        length: Number of entries to produce.
        dtype: ``dtypes.uint32`` or ``dtypes.uint64``.
        banned_factors: Primes that must not be used in any entry.
        k: Number of distinct prime factors per entry.
        minimum_factor: Inclusive lower bound for every selected prime,
            fixed for the whole chain. Defaults from *banned_factors* as in
            :func:`find_closest_almost_prime`.

    Returns:
        A pair ``(factors_list, scales)``.  ``factors_list`` is a list of
        prime-factor tuples (one per chain entry).  ``scales`` has
        ``length + 1`` elements: ``scales[0]`` is the scale *before*
        the first entry (= *target*), and ``scales[i]`` is the tracked
        scale after processing entry ``i - 1``.  Entry 0 carries the
        exact target scale; subsequent entries accumulate drift.

    Raises:
        ValueError: If any scale factor drifts more than a sqrt(2) factor
            from *target*.
    """
    if k is None:
        k = _k_for_target(target, dtype)
    if minimum_factor is None:
        minimum_factor = _default_minimum_factor(banned_factors)
    factors_list: list[tuple[int, ...]] = []
    target_scale = BigFloat(target)
    scales: list[BigFloat] = [target_scale]
    banned: list[int] = list(banned_factors)
    scale = target_scale

    for i in range(length):
        ratio = scale / target_scale if scale >= target_scale else target_scale / scale
        if ratio > _SQRT2:
            raise ValueError(f"scale {scale} at position {i} is more than sqrt(2) away from target {target} (ratio={ratio})")

        ap_target = round(scale * scale / target_scale)
        factors = find_closest_almost_prime(
            target=ap_target,
            logn=logn,
            k=k,
            banned_factors=banned,
            dtype=dtype,
            minimum_factor=minimum_factor,
        )

        product = prod(factors)

        factors_list.append(factors)
        banned.extend(factors)

        scale = scale * scale / product
        scales.append(scale)

    return factors_list, scales


def _closer(product: int, than: int, target: int) -> bool:
    """Compare distance by ratio exactly, preferring the larger product on ties."""
    if product >= target:
        return product < than if than >= target else product * than <= target * target
    return product > than if than <= target else product * than > target * target


def _bounds(best: int, target: int) -> tuple[int, int]:
    """Inclusive product range a candidate must fall in to beat ``best``."""
    t = target
    return (t * t // best, best) if best >= t else (best, -(-(t * t) // best))


def _closest_pair(
    primes: list[int], target: int, start: int, prefix: int, factors: tuple[int, ...], best: tuple[int, ...], minimum_product: int
) -> tuple[int, ...]:
    """Best of ``best`` and ``factors`` extended by two primes from ``primes[start:]``
    (``prefix`` is ``prod(factors)``): a two-pointer scan, O(P)."""
    best_product = prod(best)
    lo, hi = start, len(primes) - 1
    left = prefix * primes[lo]
    while lo < hi:
        product = left * primes[hi]
        if _closer(product, best_product, target) and product >= minimum_product:
            best, best_product = (*factors, primes[lo], primes[hi]), product
            if product == target:
                break
        if product < target:
            lo += 1
            left = prefix * primes[lo]
        else:
            hi -= 1
    return best


def _search(
    primes: list[int], target: int, k: int, start: int, prefix: int, factors: tuple[int, ...], best: tuple[int, ...], minimum_product: int
) -> tuple[int, ...]:
    """Best of ``best`` and ``factors`` extended by ``k >= 2`` primes from
    ``primes[start:]``: fix the smallest remaining prime and recurse down to
    :func:`_closest_pair`, O(P^(k-1)). Returns early on an exact hit."""
    if k == 2:
        return _closest_pair(primes, target, start, prefix, factors, best, minimum_product)
    num = len(primes)
    for i in range(start, num - k + 1):
        low, high = _bounds(prod(best), target)
        low = max(low, minimum_product)
        if prefix * prod(primes[i : i + k]) > high:
            break  # smallest completion already overshoots; later i only grow
        if prefix * primes[i] * prod(primes[num - k + 1 :]) < low:
            continue  # largest completion still undershoots
        best = _search(primes, target, k - 1, i + 1, prefix * primes[i], (*factors, primes[i]), best, minimum_product)
        if prod(best) == target:
            break
    return best


def _closest_k4(primes: list[int], target: int, minimum_product: int) -> tuple[int, ...]:
    """Meet in the middle over sorted pair products.

    Enumerate all C(P, 2) pair products, sort them, and for each pair product
    scan the disjoint complements whose total falls in the range that can
    still beat the incumbent.  Designed for the typical CKKS regime where
    logn ~ 16 yields about 1600 eligible primes.
    """
    num = len(primes)
    if target <= prod(primes[:4]):
        return tuple(primes[:4])
    if target >= prod(primes[-4:]):
        return tuple(primes[-4:])
    # Seed near target^(1/4) so the bounds are tight from the start.
    start = max(0, min(bisect_left(primes, round(target**0.25)) - 2, num - 4))
    best = tuple(primes[start : start + 4])
    best_product = prod(best)
    if best_product < minimum_product:
        best = tuple(primes[-4:])
        best_product = prod(best)
    if best_product == target:
        return best

    pairs = [(primes[i] * primes[j], i, j) for i in range(num) for j in range(i + 1, num)]
    pairs.sort()
    pair_prods = [p for p, _, _ in pairs]
    low, high = _bounds(best_product, target)
    low = max(low, minimum_product)
    for prod1, i, j in pairs:
        if prod1 * pair_prods[0] > high:
            break  # pairs are sorted: every later prod1 overshoots too
        if prod1 * pair_prods[-1] < low:
            continue
        for r in range(bisect_left(pair_prods, -(-low // prod1)), len(pairs)):
            prod2, ki, kj = pairs[r]
            product = prod1 * prod2
            if product > high:
                break
            if ki == i or ki == j or kj == i or kj == j or not _closer(product, best_product, target):
                continue
            best, best_product = tuple(sorted((primes[i], primes[j], primes[ki], primes[kj]))), product
            low, high = _bounds(best_product, target)
            low = max(low, minimum_product)
            if product == target:
                return best
    return best


def _default_minimum_factor(banned_factors: Sequence[int]) -> int:
    return isqrt(max(banned_factors)) + 1 if banned_factors else 2


def _closest_product(primes: list[int], target: int, k: int, minimum_product: int = 1) -> tuple[int, ...]:
    """Closest product of ``k`` distinct primes to ``target`` by ratio, at
    least ``minimum_product`` (ties: larger product)."""
    if len(primes) < k or prod(primes[-k:]) < minimum_product:
        raise ValueError(f"No product of {k} eligible primes is at least {minimum_product}")
    target = max(target, minimum_product)
    if k == 1:
        pos = bisect_left(primes, target)
        below, above = primes[max(pos - 1, 0)], primes[min(pos, len(primes) - 1)]
        return (above,) if below < minimum_product or _closer(above, below, target) else (below,)
    if k == 4:
        return _closest_k4(primes, target, minimum_product)
    # Seed with the better extreme product so the bounds prune from the first step.
    smallest, largest = tuple(primes[:k]), tuple(primes[-k:])
    best = smallest if prod(smallest) >= minimum_product and _closer(prod(smallest), prod(largest), target) else largest
    return best if prod(best) == target else _search(primes, target, k, 0, 1, (), best, minimum_product)


def _k_for_target(target: int, dtype: DType) -> int:
    """Return the number of primes needed to approximate the target with the given dtype."""
    if (max_bits := {dtypes.uint32: 31, dtypes.uint64: 63}.get(dtype)) is None:
        raise TypeError(f"unsupported dtype for modulus chain generation: {dtype}")
    return ceil((log2(target) + 1) / max_bits)  # We add one to log scale here
    # because we need wiggle room because future targets in the chain may vary by
    # a factor of sqrt(2), and we also need some freedom in choosing primes.
