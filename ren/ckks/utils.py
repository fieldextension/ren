from annotationlib import Format
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from math import ceil, sqrt, gcd
from functools import cache, reduce, wraps
from inspect import signature
from types import TracebackType
from typing import Callable, overload
import flint


from flint import acb, arb
from ren.ckks.errors import InvalidParameterError
from .encoding import flint_precision


@cache
def is_prime_u64(n: int) -> bool:
    """Return whether ``n`` is prime, using FLINT's deterministic primality test."""
    return bool(flint.fmpz(n).is_prime())


def get_prev_prime(m: int, bound: int, exclude: set[int] | None = None) -> int:
    """
    Computes the largest prime p < bound for which p = 1 mod m.
    Raises ValueError if no such prime exists.
    """
    # Start from the largest number < bound that is 1 mod m
    candidate = ((bound - 2) // m) * m + 1

    count = 0
    while candidate > 1:
        if is_prime_u64(candidate):
            if exclude is None or candidate not in exclude:
                return candidate
        candidate -= m
        count += 1
        if count > 10000:
            raise ValueError(f"Performed {count} attempts to find a prime less than {bound} that is 1 modulo {m}")

    raise ValueError(f"No prime less than {bound} that is 1 modulo {m}")


def get_next_prime(start: int, m: int, exclude: set[int] | None = None) -> int:
    """
    Finds the smallest prime q > start such that q = 1 (mod m).
    """
    # Start from the smallest number > start that is 1 mod m
    candidate = ((start + m - 1) // m) * m + 1
    while not is_prime_u64(candidate) or (exclude is not None and candidate in exclude):
        candidate += m
    return candidate


def _select_moduli_attempt(m: int, depth: int, limit: int, offset: int, exclude: set[int] | None = None) -> list[int] | None:
    """One attempt at ``select_moduli``: ``q_L`` is the first prime above
    ``limit - offset``; each lower prime tracks the running scale
    ``scale_l = scale_{l+1}^2 / q_{l+1}``, alternating below / above the target
    so the drift stays centered. Returns None when any prime reaches ``limit``.
    """
    start = limit - offset
    if start <= 1:
        raise ValueError("Cannot find suitable primes within bit length")

    used_primes = set(exclude) if exclude is not None else set()
    q_L = get_next_prime(start, m, exclude=used_primes)
    if q_L >= limit:
        return None
    qs = {depth: q_L}
    used_primes.add(q_L)

    scale = float(q_L)
    for flip, l in enumerate(range(depth - 1, 0, -1)):
        scale = scale * scale / qs[l + 1]
        target = round(scale)
        if flip % 2 == 0:
            try:
                candidate = get_prev_prime(m, target, exclude=used_primes)
            except ValueError:
                candidate = get_next_prime(target, m, exclude=used_primes)
        else:
            candidate = get_next_prime(target, m, exclude=used_primes)
        if candidate >= limit:
            return None
        qs[l] = candidate
        used_primes.add(candidate)

    return [qs[i] for i in range(1, depth + 1)]


def select_moduli(*, n: int, num_primes: int, bit_length: int, exclude: set[int] | None = None) -> list[int]:
    """
    Algorithm 3: Selecting prime moduli.
    Returns primes [p_1, ..., p_L].
    Attempts to choose the primes so that the sequence scale_i is roughly constant.
    """
    m = 2 * n
    limit = 1 << bit_length

    # Initial offset: start search for q_L slightly below limit.
    # We use a heuristic step size (e.g. m) to back off.
    current_offset = (num_primes + 1) * m  # Start conservatively
    assert current_offset > 0
    while True:
        result = _select_moduli_attempt(m, num_primes, limit, current_offset, exclude=exclude)
        if result is not None:
            return result
        # Increase offset to try smaller primes
        current_offset *= 2


@overload
def flint_mean(values: Sequence[flint.arb]) -> flint.arb: ...
@overload
def flint_mean(values: Sequence[flint.acb]) -> flint.acb: ...


def flint_mean(values: Sequence[flint.arb] | Sequence[flint.acb]) -> flint.acb | flint.arb:
    total = flint.arb(0)
    for v in values:
        total = total + v
    return total / len(values)


def flint_std(values: Sequence[flint.acb] | Sequence[flint.arb]) -> flint.arb:
    mean = flint_mean(values)
    variance_sum = flint.arb(0)
    for v in values:
        diff = v - mean
        variance_sum = variance_sum + abs(diff) ** 2
    return (variance_sum / len(values)).sqrt()


class NoiseStats:
    def __init__(self, errors: Sequence[acb], *, prec: int):
        self.prec = prec
        with flint_precision(prec):
            self.errors = [e.mid() for e in errors]
            self.abs_errors = [abs(e) for e in self.errors]
            self.mean = flint_mean(self.errors)
            self.mean_abs = flint_mean(self.abs_errors)
            self.max_abs = max(self.abs_errors)
            self.min_abs = min(self.abs_errors)
            self.std_abs = flint_std(self.abs_errors)
            self.std = flint_std(self.errors)

    @classmethod
    def format_arb(cls, value: arb) -> str:
        width = 40
        s = str(value)
        return s.ljust(width) + f"({float(-value.log_base(2)):.2f} bits)"

    def __str__(self):
        with flint_precision(self.prec):
            return f"""
------------------ STATS ON COMPLEX VALUES ----------------------
Abs of mean error   : {self.format_arb(abs(self.mean))}
Std error           : {self.format_arb(self.std)}
------------------ STATS ON ABSOLUTE VALUES ---------------------
Mean absolute error : {self.format_arb(self.mean_abs)}
Max absolute error  : {self.format_arb(self.max_abs)}
Min absolute error  : {self.format_arb(self.min_abs)}
Std absolute error  : {self.format_arb(self.std_abs)}
-----------------------------------------------------------------
"""


class CallCounter:
    """Collect outermost calls to functions decorated with :func:`count_calls`.

    If a tracked function calls another tracked function, only the outer one
    increments the counter.
    """

    counts: dict[str, int]
    _inside_call: bool

    def __init__(self) -> None:
        self.counts = {}
        self._inside_call = False

    def __enter__(self) -> CallCounter:
        if _active_call_counter.get() is not None:
            raise RuntimeError("There can only be one CallCounter active at a time")
        _active_call_counter.set(self)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if _active_call_counter.get() is not self:
            raise RuntimeError("CallCounter is not active")
        _active_call_counter.set(None)
        print("--- Call Summary ---")
        if not self.counts:
            print("No tracked functions were called.")
        for func_name, count in sorted(self.counts.items()):
            print(f"{func_name}: {count} calls")


_active_call_counter: ContextVar[CallCounter | None] = ContextVar("active_call_counter", default=None)


def count_calls[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that counts calls to the wrapped function when a CallCounter
    context is active. Only outermost calls are counted: if a tracked function
    calls another tracked function, only the outer one increments the counter.
    This avoids double-counting when, e.g., ``rotate`` internally calls
    ``Ciphertext.__mul__``. Calls returning ``NotImplemented`` are operator
    dispatch fallbacks and are not counted.
    Instance methods (with a leading self parameter) are named for their
    concrete receiver; other functions retain their qualified name.
    """

    instance_method = next(iter(signature(func, annotation_format=Format.STRING).parameters), None) == "self"

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        counter = _active_call_counter.get()
        if counter is None or counter._inside_call:
            return func(*args, **kwargs)

        call_name = name
        if instance_method and (args or "self" in kwargs):
            receiver = args[0] if args else kwargs["self"]
            call_name = f"{type(receiver).__qualname__}.{wrapper.__name__}"

        counter._inside_call = True
        should_count = True
        try:
            result = func(*args, **kwargs)
            should_count = result is not NotImplemented
            return result
        finally:
            counter._inside_call = False
            if should_count:
                counter.counts[call_name] = counter.counts.get(call_name, 0) + 1

    name = wrapper.__qualname__
    return wrapper


def _require_galois_elt(galois_elt: int, n: int) -> int:
    if type(galois_elt) is not int or not 0 < galois_elt < 2 * n or galois_elt % 2 == 0:
        raise InvalidParameterError(f"invalid Galois element: {galois_elt}")
    return galois_elt


def normalize_rotation_amounts(rotation_amounts: Iterable[int], slot_count: int) -> set[int]:
    """Return distinct canonical non-identity rotation amounts."""
    normalized = {rotation_amount % slot_count for rotation_amount in rotation_amounts}
    normalized.discard(0)
    return normalized


def steps_to_galois_elt(steps: int, n: int) -> int:
    """
    Convert canonical rotation steps to a Galois element.

    For CKKS with ring degree n:
    - steps=0 -> galois_elt = 1 (identity on slots)
    - steps>0 -> galois_elt = 5^steps mod 2n (rotate left)

    The generator 5 is standard for cyclotomic rings with m=2n.
    """
    slots = n // 2
    if not 0 <= steps < slots:
        raise ValueError(f"rotation steps must be in range({slots}), got {steps}")
    return pow(5, steps, 2 * n)


def base_rotation_amounts(rotation_amounts: Iterable[int]) -> set[int]:
    """Return the signed gcd base rotations for an additive rotation family."""
    nonzero = [rotation_amount for rotation_amount in rotation_amounts if rotation_amount != 0]
    if not nonzero:
        return set()
    base = reduce(gcd, (abs(v) for v in nonzero))
    required: set[int] = set()
    if any(v > 0 for v in nonzero):
        required.add(base)
    if any(v < 0 for v in nonzero):
        required.add(-base)
    return required


def bsgs_index(*, n: int, diagonals: Iterable[int]) -> dict[int, tuple[int, ...]]:
    """The baby-step giant-step split of the diagonals, as in Lattigo's
    ``BSGSIndex``: each giant step maps to its baby steps, and every diagonal
    is exactly one ``baby + giant (mod n)``.

    The diagonals are read off the circle of residues starting after their
    largest gap, so a family centered on 0 becomes signed while a one-sided
    family stays non-negative.  They are then multiples ``k * m`` of their gcd
    ``k`` with ``m`` spanning a short range, which is split with
    ``s = ceil(sqrt(span))`` babies ``{0, k, ..., (s-1)*k}``.

    Giants are aligned to 0 (multiples of ``s*k``), so the giant grid is
    ``{..., -2 s k, -s k, 0, s k, 2 s k, ...}`` clipped to the range the
    diagonals actually need.  Giants and their babies come out sorted.
    """
    residues = sorted({d % n for d in diagonals})
    if not residues:
        return {}
    # gaps[i] precedes residues[i]; gaps[0] wraps around from the last residue.
    gaps = [residues[0] - (residues[-1] - n)] + [b - a for a, b in zip(residues, residues[1:])]
    start = gaps.index(max(gaps))
    signed = [d - n for d in residues[start:]] + residues[:start] if start else residues

    k = reduce(gcd, (abs(d) for d in signed)) or 1
    s = ceil(sqrt((signed[-1] - signed[0]) // k + 1))
    giant_step_size = s * k
    index: dict[int, list[int]] = {}
    for d in signed:
        giant = (d // giant_step_size) * giant_step_size
        index.setdefault(giant, []).append(d - giant)
    return {giant: tuple(babies) for giant, babies in index.items()}


def bsgs_steps(*, n: int, diagonals: Iterable[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The sorted baby and giant steps of :func:`bsgs_index`: the rotations its split needs."""
    index = bsgs_index(n=n, diagonals=diagonals)
    return tuple(sorted({b for babies in index.values() for b in babies})), tuple(index)
