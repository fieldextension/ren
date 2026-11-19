from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from math import ceil, isfinite, log2, prod
from typing import TYPE_CHECKING, Self

from ren.bigfloat import BigFloat, FloatLike
from ren.dtype import DType, dtypes
from ren.poly import Poly
from ren.polydist import MINIMUM_ENCRYPTION_NOISE_STD_DEV, DiscreteGaussian, KeyDist, PolyDistWithBounds, SparseTernary, Ternary
from ren.ring import RingSpec
from ren.ckks._security import validate_modulus_security
from ren.ckks.encoding import FLOAT_PRECISION_BITS
from ren.ckks.primes import almost_prime_chain, find_closest_almost_prime
from ren.ckks.utils import select_moduli, steps_to_galois_elt
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError
from ren.utils import find_contiguous_subset

if TYPE_CHECKING:
    from .context import CKKSContext
    from .matrix import SparseMatrix

_DEFAULT_SPARSE_MAIN_KEY_WEIGHT = 192


@dataclass(frozen=True, slots=True, init=False)
class CKKSLevel:
    """One scheduled CKKS level.

    Parameters:
        moduli: Active coefficient moduli at this level.
        scale: Nominal plaintext scale at this level.
    """

    moduli: tuple[int, ...]
    scale: BigFloat

    def __init__(self, moduli: Sequence[int], scale: FloatLike) -> None:
        object.__setattr__(self, "moduli", tuple(int(modulus) for modulus in moduli))
        object.__setattr__(self, "scale", BigFloat(scale))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.moduli:
            raise ValueError("scheduled CKKS levels must include at least one modulus")
        if len(self.moduli) != len(set(self.moduli)):
            raise ValueError("scheduled CKKS level moduli must be unique")
        if not self.scale.is_finite() or self.scale <= 0:
            raise ValueError("scheduled CKKS level scale must be finite and positive")


def _default_ks_prime_bits(dtype: DType) -> int:
    if dtype is dtypes.uint32:
        return 30
    if dtype is dtypes.uint64:
        return 60
    raise TypeError(f"unsupported CKKS dtype: {dtype}")


def _build_prefix_levels(
    moduli: Sequence[tuple[int, ...]],
    scales: Sequence[BigFloat],
) -> tuple[CKKSLevel, ...]:
    level_moduli: list[tuple[int, ...]] = []
    active_moduli = []
    for modulus in moduli:
        active_moduli.extend(modulus)
        level_moduli.append(tuple(active_moduli))
    return tuple(CKKSLevel(moduli=moduli, scale=scale) for moduli, scale in zip(level_moduli, scales, strict=True))


@dataclass(slots=True, frozen=True)
class BootstrappingParams:
    """CKKS bootstrapping algorithm, shared by parameter generation and execution.

    The concrete bootstrap-input moduli and scale belong to
    :attr:`CKKSParams.bootstrap_base_level`. Prime sizing is generator policy.

    Parameters:
        cts_depth: Depth of the coeff-to-slots transform.
        stc_depth: Depth of the slots-to-coeffs transform.
        exp_degree: Degree of the exponential approximation.
        squaring_count: Number of repeated squaring steps in eval-mod.
        failure_probability: Target bootstrapping failure probability.
        precision_gains: Conservative precision gain assumed before each
            residual correction iteration.
        input_margin: Multiplicative allowance above the nominal input bound.
        sparse_encapsulation_weight: Number of nonzero coefficients in the
            temporary key used for modulus raising. Its weight determines the
            mod-one range K for the requested failure probability. The main key
            can be dense or sparse. If ``None``, modulus raising uses the main
            key directly. A dense main key needs a wider interval for modular
            reduction. For sparse encapsulation, see https://eprint.iacr.org/2022/024.
    """

    cts_depth: int
    stc_depth: int
    exp_degree: int
    squaring_count: int
    failure_probability: float
    precision_gains: Sequence[int] = ()
    input_margin: float = 1.0
    sparse_encapsulation_weight: int | None = None

    @property
    def sparse_encapsulation(self) -> bool:
        """Whether bootstrap sparse-secret encapsulation is enabled."""
        return self.sparse_encapsulation_weight is not None

    def __post_init__(self) -> None:
        if self.cts_depth < 2:
            raise ValueError("cts_depth is too small")
        if self.stc_depth < 2:
            raise ValueError("stc_depth is too small")
        if self.exp_degree < 16:
            raise ValueError("exp_degree must be at least 16")
        if self.squaring_count < 0:
            raise ValueError("squaring_count must be nonnegative")
        if self.squaring_count > 5:
            raise ValueError("squaring_count must be at most five")
        if self.sparse_encapsulation_weight is not None and self.sparse_encapsulation_weight < 32:
            raise ValueError("sparse_encapsulation_weight must be at least 32")
        if not 0 < self.failure_probability <= 2**-16:
            raise ValueError(f"failure probability must be in (0, 2^-16], got {self.failure_probability!r}")
        precision_gains = tuple(self.precision_gains)
        if any(gain < 1 for gain in precision_gains):
            raise ValueError("precision gains must be positive")
        if not isfinite(self.input_margin) or self.input_margin < 1.0:
            raise ValueError("input_margin must be finite and at least 1.0")
        object.__setattr__(self, "precision_gains", precision_gains)

    @property
    def depth(self) -> int:
        mod_one_depth = self.exp_degree.bit_length() + self.squaring_count
        return self.cts_depth + self.stc_depth + mod_one_depth

    def bounds(self, log_bound: float) -> tuple[float, ...]:
        bounds = [log_bound + log2(self.input_margin)]
        for gain in self.precision_gains:
            bounds.append(bounds[-1] - gain)
        return tuple(bounds)

    def matrix_stages(self, params: CKKSParams) -> tuple[tuple[int, tuple[SparseMatrix, ...]], ...]:
        """The CTS and STC factorizations, each paired with the level its first stage consumes."""
        from .bootstrap import l_inv_matrix_partial_factorization, l_matrix_partial_factorization

        mod_input_level = params.highest_level - self.cts_depth
        stc_start_level = mod_input_level - self.exp_degree.bit_length() - self.squaring_count
        return (
            (params.highest_level, l_matrix_partial_factorization(params.logn, self.cts_depth)),
            (stc_start_level, l_inv_matrix_partial_factorization(params.logn, self.stc_depth)),
        )

    def galois_elts_by_level(self, params: CKKSParams, *, hoist: bool) -> dict[int, set[int]]:
        """Galois elements this circuit uses at each level of ``params``.

        With ``hoist`` the linear transforms get a direct key per baby- and
        giant-step rotation, all at their input level, so they are
        double-hoisted; otherwise they compose base keys.
        """
        from .matrix import LinearTransform

        stages = self.matrix_stages(params)
        galois_elts_by_level: defaultdict[int, set[int]] = defaultdict(set)
        for start_level, matrices in stages:
            for stage, matrix in enumerate(matrices):
                input_level = start_level - stage
                for offset, amounts in LinearTransform.rotation_amounts(matrix, hoist=hoist).items():
                    galois_elts_by_level[input_level + offset].update(steps_to_galois_elt(amount, params.n) for amount in amounts)

        cts_start_level = stages[0][0]
        stc_start_level = stages[1][0]
        mod_input_level = cts_start_level - self.cts_depth
        conjugation_elt = 2 * params.n - 1
        galois_elts_by_level[mod_input_level].add(conjugation_elt)
        galois_elts_by_level[stc_start_level].add(conjugation_elt)
        return dict(galois_elts_by_level)


@dataclass(frozen=True, slots=True)
class RescaleStep:
    """One integer multiplication and rounded basis conversion.

    ``src_ring`` is the working ring. An input in a larger ring is first
    CRT-truncated to it, without changing its coefficient scale.
    """

    src_ring: RingSpec
    dst_ring: RingSpec
    k: int

    def __post_init__(self) -> None:
        if type(self.k) is not int:
            raise InvalidParameterError("rescale multiplier must be an integer")
        if self.src_ring.n != self.dst_ring.n or self.src_ring.dtype != self.dst_ring.dtype:
            raise InvalidParameterError("rescale rings must have the same degree and dtype")
        if self.src_ring != self.dst_ring and not set(self.src_ring.moduli) - set(self.dst_ring.moduli):
            raise InvalidParameterError("rescale must remove at least one source modulus")

    @classmethod
    def from_scales(cls, src_ring: RingSpec, dst_ring: RingSpec, src_scale: BigFloat, dst_scale: BigFloat) -> RescaleStep:
        if not src_scale.is_finite() or src_scale <= 0 or not dst_scale.is_finite() or dst_scale <= 0:
            raise InvalidParameterError("rescale scales must be finite and positive")
        return cls(src_ring, dst_ring, round(src_ring.q * dst_scale / (dst_ring.q * src_scale)))

    def prepare(self, poly: Poly) -> Poly:
        """Truncate and multiply, leaving the final rounding available for batching."""
        return poly.to_quotient_ring(self.src_ring) * self.k

    def apply(self, poly: Poly) -> Poly:
        result = self.prepare(poly)
        return result if self.src_ring == self.dst_ring else result.rescale(self.dst_ring.moduli)

    @property
    def poly_ratio(self) -> BigFloat:
        """Realized coefficient and noise scaling, excluding rounding error."""
        return BigFloat(self.k * self.dst_ring.q) / self.src_ring.q


@dataclass(frozen=True, slots=True)
class CKKSTransition:
    """An executable conversion between CKKS representations.

    Steps preserve their individual rounding boundaries. ``poly_ratio`` scales
    coefficient-domain metadata. ``message_ratio`` also accounts for interpreting
    the result at ``dst_scale`` and can differ slightly from one because each
    integer multiplier is rounded. It equals ``poly_ratio * src_scale / dst_scale``.
    Bootstrap normalization uses this realized message ratio to restore values
    after converting to its separate base scale.
    """

    src_ring: RingSpec
    src_scale: BigFloat
    dst_scale: BigFloat
    steps: tuple[RescaleStep, ...]

    def __post_init__(self) -> None:
        if not self.src_scale.is_finite() or self.src_scale <= 0 or not self.dst_scale.is_finite() or self.dst_scale <= 0:
            raise InvalidParameterError("transition scales must be finite and positive")
        if not self.steps:
            raise InvalidParameterError("transition must contain at least one step")
        ring = self.src_ring
        for step in self.steps:
            if not step.src_ring.is_quotient_of(ring):
                raise InvalidParameterError("transition step cannot consume the preceding ring")
            ring = step.dst_ring

    @property
    def dst_ring(self) -> RingSpec:
        return self.steps[-1].dst_ring

    @property
    def poly_ratio(self) -> BigFloat:
        return prod((step.poly_ratio for step in self.steps), start=BigFloat(1))

    @property
    def message_ratio(self) -> BigFloat:
        return self.poly_ratio * self.src_scale / self.dst_scale

    def prepare(self, poly: Poly) -> Poly:
        """Apply all but the final rounding, for a shared-rescale accumulator."""
        if poly.ring != self.src_ring:
            raise InvalidParameterError("polynomial ring does not match transition source")
        for step in self.steps[:-1]:
            poly = step.apply(poly)
        return self.steps[-1].prepare(poly)

    def apply(self, poly: Poly) -> Poly:
        poly = self.prepare(poly)
        target = self.dst_ring
        return poly if poly.ring == target else poly.rescale(target.moduli)


@dataclass(frozen=True, slots=True)
class KeySwitchParams:
    """Resolved Method II parameters for key-switching to one target Q ring."""

    ring_qt: RingSpec
    digit_ranges: tuple[range, ...]
    p_prod_mod_q: tuple[int, ...]

    @classmethod
    def for_target(
        cls,
        *,
        highest_ring: RingSpec,
        target_ring: RingSpec,
        p_primes: tuple[int, ...],
        digit_sizes: tuple[int, ...],
    ) -> Self:
        projection = _project_key_switch_digits(highest_ring.moduli, digit_sizes, target_ring.moduli)
        if projection is None:
            raise InvalidParameterError("key-switch target must be a contiguous subrange of the highest Q basis")
        if sum(digit_sizes) != len(highest_ring.moduli):
            raise InvalidParameterError("key-switch digit sizes must partition the highest Q basis")
        if not projection:
            raise InvalidParameterError("key-switch target has no Q digits")

        p_prod = prod(p_primes)
        return cls(
            ring_qt=RingSpec(n=target_ring.n, moduli=p_primes + target_ring.moduli, dtype=target_ring.dtype),
            digit_ranges=tuple(digit_range for _, digit_range in projection),
            p_prod_mod_q=tuple(p_prod % q for q in target_ring.moduli),
        )

    @property
    def p_prime_count(self) -> int:
        return len(self.ring_qt.moduli) - len(self.p_prod_mod_q)

    @property
    def target_ring(self) -> RingSpec:
        return self.ring_qt[self.p_prime_count :]

    @property
    def digit_sizes(self) -> tuple[int, ...]:
        return tuple(len(digit_range) for digit_range in self.digit_ranges)

    @property
    def d(self) -> int:
        return len(self.digit_ranges)

    def validate_digits(self, digits: Sequence[Poly]) -> None:
        """Check the digit count and ring against the key-switch parameters."""
        if len(digits) != self.d:
            raise ParameterMismatchError("key-switch digits do not match the expected digit layout")
        if {poly.ring for poly in digits} != {self.ring_qt}:
            raise ParameterMismatchError("key-switch digits do not match the expected ring")

    def mask_scalars(self, digit_range: range) -> tuple[int, ...]:
        """Return prod(P) on this digit's Q moduli and zero elsewhere in P + Q."""
        return (
            (0,) * (self.p_prime_count + digit_range.start)
            + self.p_prod_mod_q[digit_range.start : digit_range.stop]
            + (0,) * (len(self.p_prod_mod_q) - digit_range.stop)
        )


def _project_key_switch_digits(
    source_moduli: tuple[int, ...],
    digit_sizes: tuple[int, ...],
    target_moduli: tuple[int, ...],
) -> tuple[tuple[int, range], ...] | None:
    """Map source key-switch digits onto a contiguous target Q basis."""
    target_offset = find_contiguous_subset(source_moduli, target_moduli)
    if target_offset is None:
        return None

    target_stop = target_offset + len(target_moduli)
    digit_start = 0
    projection: list[tuple[int, range]] = []
    for digit_index, digit_size in enumerate(digit_sizes):
        digit_stop = digit_start + digit_size
        intersection_start = max(digit_start, target_offset)
        intersection_stop = min(digit_stop, target_stop)
        if intersection_start < intersection_stop:
            projection.append((digit_index, range(intersection_start - target_offset, intersection_stop - target_offset)))
        digit_start = digit_stop
    return tuple(projection)


@dataclass(slots=True, frozen=True)
class CKKSParams:
    """
    Parameters for CKKS encryption.

    CKKS uses an explicit computation-level model:

    - `levels[level]` gives the active moduli and nominal scale at that level
    - level 0 is the lowest ordinary computation level
    - `highest_level` is the highest level in the full schedule
    - `bootstrap_output_level` is where bootstrapping returns ciphertexts; the
      levels above it are consumed by the bootstrap circuit
    - `default_level` is where fresh ciphertexts start unless the context
      overrides it: the bootstrap output level, or `highest_level`
    - `bootstrap_base_level` gives the bootstrap-input moduli q0 and scale;
      its moduli are a contiguous subrange of computation level 0, but its
      scale can differ, and it is not included in `levels`

    A prefix schedule is one valid instance of this model.

    Construction split:

    - `from_levels(...)` is the explicit constructor for hand-authored
      schedules, including `u32` schedules. It takes the exact canonical
      key-switch digit sizes.
    - `generate_prefix(...)` builds a generated prefix schedule.
    """

    # Serialized scheme parameters.
    n: int
    dtype: DType
    levels: tuple[CKKSLevel, ...]
    log_range: int
    ks_primes: tuple[int, ...]
    ks_digit_sizes: tuple[int, ...]

    noise_dist: PolyDistWithBounds
    secret_key_dist: KeyDist

    boot_params: BootstrappingParams | None
    bootstrap_base_level: CKKSLevel | None

    # Derived host state.
    _level_rings: tuple[RingSpec, ...] = field(init=False, repr=False, compare=False)
    _level_by_ring: dict[RingSpec, int] = field(init=False, repr=False, compare=False)
    _highest_key_switch_params: KeySwitchParams = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (self.boot_params is None) != (self.bootstrap_base_level is None):
            raise InvalidParameterError("boot_params and bootstrap_base_level must be provided together")
        levels = self.levels
        if not levels:
            raise InvalidParameterError("CKKS parameters must include at least one level")

        ks_primes = self.ks_primes
        if not ks_primes:
            raise InvalidParameterError("ks_primes must be non-empty")
        if len(ks_primes) != len(set(ks_primes)):
            raise InvalidParameterError("ks_primes must be unique")
        ks_digit_sizes = self.ks_digit_sizes
        if not ks_digit_sizes:
            raise InvalidParameterError("ks_digit_sizes must be non-empty")
        if not all(size > 0 for size in ks_digit_sizes):
            raise InvalidParameterError("ks_digit_sizes must be positive")
        if sum(ks_digit_sizes) != len(levels[-1].moduli):
            raise InvalidParameterError("ks_digit_sizes must partition the highest-level Q basis")

        if self.secret_key_dist.n != self.n:
            raise InvalidParameterError("secret_key_dist degree must match n")
        secret_key_1_norm_bound = self.secret_key_dist.coeff_1_norm_bound()
        if not isfinite(secret_key_1_norm_bound) or secret_key_1_norm_bound <= 0:
            raise InvalidParameterError("secret key coefficient 1-norm bound must be finite and positive")
        secret_norm = self.secret_key_dist.expected_squared_l2_norm()
        if not isfinite(secret_norm) or secret_norm <= 0:
            raise InvalidParameterError("secret expected squared L2 norm must be finite and positive")
        if self.noise_dist.n != self.n:
            raise InvalidParameterError("noise_dist degree must match n")
        level0 = levels[0]
        level_rings = tuple(RingSpec(n=self.n, moduli=level.moduli, dtype=self.dtype) for level in levels)
        highest_ring = level_rings[-1]
        level_by_ring: dict[RingSpec, int] = {}
        for level, ring in enumerate(level_rings):
            if ring in level_by_ring:
                raise InvalidParameterError("scheduled computation levels must use distinct rings")
            offset = find_contiguous_subset(highest_ring.moduli, ring.moduli)
            if offset is None:
                raise InvalidParameterError("every scheduled ring must be a contiguous subrange of the highest-level ring")
            level_by_ring[ring] = level

        if set(highest_ring.moduli) & set(ks_primes):
            raise InvalidParameterError("ks_primes must be disjoint from ring.moduli")

        boot_params = self.boot_params
        if boot_params is not None:
            base_level = self.bootstrap_base_level
            assert base_level is not None
            # Bootstrap inputs are normalized by q0 / scale; the mod-one interval only
            # recovers them when the message is small against q0.
            if self.bootstrap_message_ratio <= 2:
                raise InvalidParameterError("base level scale must be less than half of q0")
            if len(levels) - 1 < boot_params.depth:
                raise InvalidParameterError("bootstrapping depth exceeds the available scheduled levels")
            if find_contiguous_subset(level0.moduli, base_level.moduli) is None:
                raise InvalidParameterError("boot base moduli must be a contiguous subrange of computation level 0")
            # This minimum applies to the main key even when a temporary key handles modulus raising.
            if isinstance(self.secret_key_dist, SparseTernary):
                if self.secret_key_dist.h < 128:
                    raise InvalidParameterError("bootstrapping secret key Hamming weight must be at least 128")
            elif not isinstance(self.secret_key_dist, Ternary):
                raise InvalidParameterError("bootstrapping requires a ternary or sparse ternary secret_key_dist")

            q0q1 = prod(level0.moduli)
            minimum_bootstrap_input_modulus = (
                BigFloat(boot_params.input_margin) * level0.scale * (BigFloat(2) ** self.log_range) * self.bootstrap_message_ratio
            )
            if q0q1 < minimum_bootstrap_input_modulus:
                raise InvalidParameterError("q0*q1 is too small for log_range, message_ratio, and the level-0 computation scale")

        else:
            q0 = prod(level0.moduli)
            minimum_base_modulus = 2 * level0.scale * (BigFloat(2) ** self.log_range)
            if q0 < minimum_base_modulus:
                raise InvalidParameterError("base modulus is too small for log_range and the level-0 scale")

        highest_key_switch_params = KeySwitchParams.for_target(
            highest_ring=highest_ring,
            target_ring=highest_ring,
            p_primes=ks_primes,
            digit_sizes=ks_digit_sizes,
        )

        object.__setattr__(self, "_level_rings", level_rings)
        object.__setattr__(self, "_level_by_ring", level_by_ring)
        object.__setattr__(self, "_highest_key_switch_params", highest_key_switch_params)

    def validate_security(self, security: int) -> None:
        """Check the main key's modulus and, when using encapsulation, the temporary key's modulus."""
        validate_modulus_security(self._highest_key_switch_params.ring_qt.q, secret=self.secret_key_dist, error=self.noise_dist, security=security)
        if self.boot_params is not None and (h := self.boot_params.sparse_encapsulation_weight) is not None:
            assert self.bootstrap_base_level is not None
            # The dense-to-sparse switch key uses the bootstrap base and the first P prime.
            # It is the only key material encrypted under the temporary secret.
            validate_modulus_security(
                prod(self.bootstrap_base_level.moduli) * self.ks_primes[0],
                secret=SparseTernary(n=self.n, h=h),
                error=self.noise_dist,
                security=security,
            )

    @property
    def highest_ring(self) -> RingSpec:
        return self._level_rings[-1]

    @property
    def highest_level(self) -> int:
        return len(self.levels) - 1

    @property
    def bootstrap_output_level(self) -> int | None:
        """Level at which bootstrapping returns ciphertexts, or None without bootstrapping."""
        if self.boot_params is None:
            return None
        return self.highest_level - self.boot_params.depth

    @property
    def bootstrap_message_ratio(self) -> BigFloat:
        """How much larger q0 is than the bootstrap-input scale."""
        if self.bootstrap_base_level is None:
            raise InvalidParameterError("bootstrapping is not enabled for this parameter set")
        return BigFloat(prod(self.bootstrap_base_level.moduli)) / self.bootstrap_base_level.scale

    @property
    def default_level(self) -> int:
        """Where fresh ciphertexts start unless the context overrides it: the bootstrap output level, else the highest level."""
        output_level = self.bootstrap_output_level
        return self.highest_level if output_level is None else output_level

    @property
    def logn(self) -> int:
        return self.n.bit_length() - 1

    @property
    def slot_count(self) -> int:
        return self.highest_ring.n // 2

    def _require_level(self, level: int, *, name: str = "level") -> int:
        if not 0 <= level < len(self.levels):
            raise InvalidParameterError(f"{name} must be between 0 and {len(self.levels) - 1}")
        return level

    def scale(self, level: int | RingSpec) -> BigFloat:
        if isinstance(level, RingSpec):
            scheduled_level = self._level_by_ring.get(level)
            if scheduled_level is not None:
                return self.levels[scheduled_level].scale
            base_level = self.bootstrap_base_level
            if base_level is not None and level.n == self.n and level.dtype is self.dtype and level.moduli == base_level.moduli:
                return base_level.scale
            raise InvalidParameterError("ring is not part of this CKKS schedule")
        return self.levels[self._require_level(level)].scale

    @property
    def max_precision(self) -> int:
        scales = tuple(level.scale for level in self.levels)
        if self.bootstrap_base_level is not None:
            scales = (*scales, self.bootstrap_base_level.scale)
        return round(max(scales).log2())

    def ring_at(self, level: int) -> RingSpec:
        return self._level_rings[self._require_level(level)]

    def level_of_ring(self, ring: RingSpec) -> int:
        level = self._level_by_ring.get(ring)
        if level is None:
            raise InvalidParameterError("ring is not part of this CKKS schedule")
        return level

    def transition(
        self, src_level: int, dst_level: int, *, current_scale: FloatLike | None = None, correction_factor: FloatLike = 1
    ) -> CKKSTransition:
        """Resolve conversion to the destination's ring and nominal scale.

        Quotient-only ranges can truncate to the final working ring when it has
        capacity for the source scale and configured message range. Otherwise
        the conversion rescales directly from the source ring. Intermediate
        scheduled levels do not impose additional rounding boundaries.

        ``current_scale`` overrides the source's nominal scale. The final step
        incorporates ``correction_factor`` so the represented message is
        approximately multiplied by that factor, rather than preserved.
        """
        src_level = self._require_level(src_level, name="src_level")
        dst_level = self._require_level(dst_level, name="dst_level")
        if dst_level >= src_level:
            raise InvalidParameterError("transition destination must be a strictly lower level")
        src_ring, dst_ring = self.ring_at(src_level), self.ring_at(dst_level)
        src_scale = self.scale(src_level) if current_scale is None else BigFloat(current_scale)
        dst_scale = self.scale(dst_level)
        work_ring = self.ring_at(dst_level + 1)
        if not (
            all(self.ring_at(upper - 1).is_quotient_of(self.ring_at(upper)) for upper in range(src_level, dst_level, -1))
            and 2 * src_scale * (BigFloat(2) ** self.log_range) < work_ring.q
        ):
            work_ring = src_ring
        step = RescaleStep.from_scales(work_ring, dst_ring, src_scale, dst_scale * correction_factor)
        return CKKSTransition(src_ring, src_scale, dst_scale, (step,))

    def bootstrap_transition(self, *, current_scale: FloatLike | None = None, correction_factor: FloatLike = 1) -> CKKSTransition:
        """Resolve conversion from computation level 0 to the bootstrap base."""
        if self.bootstrap_base_level is None:
            raise InvalidParameterError("bootstrapping is not enabled for this parameter set")
        src_ring = self.ring_at(0)
        dst_ring = RingSpec(n=self.n, moduli=self.bootstrap_base_level.moduli, dtype=self.dtype)
        src_scale = self.scale(0) if current_scale is None else BigFloat(current_scale)
        dst_scale = self.bootstrap_base_level.scale
        step = RescaleStep.from_scales(src_ring, dst_ring, src_scale, dst_scale * correction_factor)
        return CKKSTransition(src_ring, src_scale, dst_scale, (step,))

    @classmethod
    def from_levels(
        cls,
        *,
        n: int,
        dtype: DType,
        levels: Sequence[CKKSLevel],
        ks_primes: Sequence[int],
        log_range: int,
        ks_digit_sizes: Sequence[int],
        boot_params: BootstrappingParams | None = None,
        bootstrap_base_level: CKKSLevel | None = None,
        noise_dist: PolyDistWithBounds | None = None,
        secret_key_dist: KeyDist | None = None,
        security: int | None = 128,
    ) -> CKKSParams:
        """Construct params from an explicit scheduled level list.

        `dtype` is a property of the whole parameter set, not of individual
        levels. Every derived ring in this parameter set uses that one dtype.

        `ks_digit_sizes` is the exact canonical partition of the highest-level
        Q basis. This constructor does not infer a partitioning policy.

        """
        normalized_levels = tuple(levels)
        if not normalized_levels:
            raise ValueError("levels must be non-empty")
        normalized_ks_primes = tuple(ks_primes)
        if secret_key_dist is None:
            if boot_params is not None and not boot_params.sparse_encapsulation:
                secret_key_dist = SparseTernary(n=n, h=_DEFAULT_SPARSE_MAIN_KEY_WEIGHT)
            else:
                secret_key_dist = Ternary(n=n)

        if noise_dist is None:
            noise_dist = DiscreteGaussian(n=n, stdev=MINIMUM_ENCRYPTION_NOISE_STD_DEV)

        params = cls(
            n=n,
            dtype=dtype,
            levels=normalized_levels,
            log_range=log_range,
            boot_params=boot_params,
            bootstrap_base_level=bootstrap_base_level,
            ks_primes=normalized_ks_primes,
            noise_dist=noise_dist,
            secret_key_dist=secret_key_dist,
            ks_digit_sizes=tuple(ks_digit_sizes),
        )
        if security is not None:
            params.validate_security(security)
        return params

    @classmethod
    def generate_prefix(
        cls,
        *,
        log_n: int,
        depth: int,
        log_precision: int,
        log_range: int,
        dtype: DType,
        boot_params: BootstrappingParams | None = None,
        bootstrap_level_bit_length: int | None = None,
        bootstrap_message_ratio: float | None = None,
        p_prime_count: int | None = None,
        p_prime_bit_length: int | None = None,
        ks_digit_size: int | None = None,
        noise_dist: PolyDistWithBounds | None = None,
        security: int | None = 128,
        secret_key_dist: KeyDist | None = None,
        log_extension_factor: int = 0,
    ) -> CKKSParams:
        """
        Build params from a generated prefix schedule.

        With ``boot_params``, ``bootstrap_level_bit_length`` (default 60) is the
        target bit length of q0 and of each bootstrap level, and
        ``bootstrap_message_ratio`` (default 256) is how much larger q0 is made
        than the bootstrap base scale. Both are sizing policy for this
        generator only and are only accepted alongside ``boot_params``. The
        resulting moduli and scale are stored in ``bootstrap_base_level``.

        ``ks_digit_size`` independently controls the maximum number of Q
        primes in each Method-II decomposition digit. It defaults to
        ``p_prime_count``. ``p_prime_count`` defaults to the minimum generated
        P width for ``dtype`` (four primes for uint32, two for uint64).

        ``log_extension_factor`` selects primes that are also NTT-valid for a
        degree-``2**log_extension_factor * n`` extension ring (p ≡ 1 mod the
        extension ring's 2n'). KG+ transmission keys need ``log_extension_factor=1``
        (degree 2n).
        """
        if log_n < 10 or log_n > 17:
            raise ValueError("n must be between 2^10 and 2^17")
        if log_extension_factor < 0:
            raise ValueError("log_extension_factor must be nonnegative")
        n = 1 << log_n
        log_extension_n = log_n + log_extension_factor
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        if log_range < 0:
            raise ValueError("log_range must be nonnegative")
        # Particularly low P prime counts will use an excessive amount of GPU memory
        # since each rotation key will contain many ciphertexts.
        # We force a minimum of roughly 120 bits of P modulus.
        if dtype is dtypes.uint32:
            min_ks_prime_count = 4
        elif dtype is dtypes.uint64:
            min_ks_prime_count = 2
        else:
            raise TypeError(f"unsupported CKKS dtype: {dtype}")
        if p_prime_count is None:
            p_prime_count = min_ks_prime_count
        elif p_prime_count < min_ks_prime_count:
            raise ValueError(f"generated prefix schedule requires at least {min_ks_prime_count} ks primes")
        if p_prime_bit_length is None:
            p_prime_bit_length = _default_ks_prime_bits(dtype)
        elif p_prime_bit_length <= 0:
            raise ValueError("p_prime_bit_length must be positive")
        if ks_digit_size is None:
            ks_digit_size = p_prime_count
        elif ks_digit_size <= 0:
            raise ValueError("ks_digit_size must be positive")

        p_bit_budget = p_prime_count * p_prime_bit_length
        digit_bit_budget = ks_digit_size * _default_ks_prime_bits(dtype)
        if p_bit_budget < digit_bit_budget:
            raise ValueError(f"P bit budget must cover one key-switch digit, but {p_bit_budget} < {digit_bit_budget}")

        # Reserve P first so the Q chain avoids these primes. Every allocation
        # below extends `used`, so no prime repeats anywhere in QP. Primes are
        # drawn at the (possibly extended) selection degree so they stay
        # NTT-valid for the KG+ extension ring.
        ks_primes = select_moduli(n=1 << log_extension_n, num_primes=p_prime_count, bit_length=p_prime_bit_length)
        used = list(ks_primes)
        compute_chain, compute_scales = almost_prime_chain(
            target=1 << log_precision, length=depth, logn=log_extension_n, dtype=dtype, banned_factors=used
        )
        used += [factor for entry in compute_chain for factor in entry]
        scale0 = compute_scales[-1]

        if boot_params is None:
            if bootstrap_level_bit_length is not None or bootstrap_message_ratio is not None:
                raise ValueError("bootstrap_level_bit_length and bootstrap_message_ratio require boot_params")
            # The base is sized so level 0 represents 2**log_range messages at its scale.
            q0 = find_closest_almost_prime(
                target=1 << (log_precision + 1 + log_range),
                minimum_product=ceil(2 * scale0 * BigFloat(2) ** log_range),
                logn=log_extension_n,
                dtype=dtype,
                banned_factors=used,
            )
            # Bottom-up: the base, then the compute entries with the exact target scale on top.
            levels = _build_prefix_levels([q0, *reversed(compute_chain)], list(reversed(compute_scales)))
            bootstrap_base_level = None
        else:
            level_bits = 60 if bootstrap_level_bit_length is None else bootstrap_level_bit_length
            message_ratio = 256.0 if bootstrap_message_ratio is None else bootstrap_message_ratio
            if level_bits <= 0:
                raise ValueError("bootstrap_level_bit_length must be positive")
            if not isfinite(message_ratio) or message_ratio <= 2:
                raise ValueError("bootstrap_message_ratio must be finite and greater than two")
            q0 = find_closest_almost_prime(target=1 << level_bits, logn=log_extension_n, dtype=dtype, banned_factors=used)
            used += q0
            boot_chain, boot_scales = almost_prime_chain(
                target=1 << level_bits, length=boot_params.depth, logn=log_extension_n, dtype=dtype, banned_factors=used
            )
            used += [factor for entry in boot_chain for factor in entry]
            # q1 goes last so it is constrained against every other prime in the chain.
            # An empty q1 means q1 = 1: level 0 and the boot base share a ring.
            minimum_q1 = ceil(scale0 * boot_params.input_margin * BigFloat(2) ** log_range * message_ratio / prod(q0))
            q1 = (
                ()
                if minimum_q1 <= 1
                else find_closest_almost_prime(target=minimum_q1, minimum_product=minimum_q1, logn=log_extension_n, dtype=dtype, banned_factors=used)
            )
            # Bottom-up: base, level 0 (base + q1), the compute entries, then the boot tail.
            # The boot section's bottom scale duplicates the compute top, so it is dropped.
            moduli = [q0, q1, *reversed(compute_chain), *reversed(boot_chain)]
            scales = [BigFloat(prod(q0)) / message_ratio, *reversed(compute_scales), *list(reversed(boot_scales))[1:]]
            bootstrap_base_level, *levels = _build_prefix_levels(moduli, scales)

        q_count = len(levels[-1].moduli)
        ks_digit_sizes = tuple(min(ks_digit_size, q_count - start) for start in range(0, q_count, ks_digit_size))

        return cls.from_levels(
            n=n,
            dtype=dtype,
            levels=levels,
            ks_primes=ks_primes,
            log_range=log_range,
            boot_params=boot_params,
            bootstrap_base_level=bootstrap_base_level,
            noise_dist=noise_dist,
            security=security,
            ks_digit_sizes=ks_digit_sizes,
            secret_key_dist=secret_key_dist,
        )

    @classmethod
    @cache
    def default(
        cls,
        *,
        log_n: int = 16,
        depth: int = 10,
        log_precision: int = 50,
        log_range: int = 10,
        dtype: DType = dtypes.uint64,
        boot_params: BootstrappingParams | None = BootstrappingParams(
            cts_depth=3, stc_depth=3, exp_degree=63, squaring_count=3, failure_probability=2**-64, sparse_encapsulation_weight=32
        ),
        bootstrap_level_bit_length: int | None = None,
        bootstrap_message_ratio: float | None = None,
        p_prime_count: int | None = 3,
        p_prime_bit_length: int | None = None,
        ks_digit_size: int | None = None,
        noise_dist: PolyDistWithBounds | None = None,
        security: int | None = 128,
        secret_key_dist: KeyDist | None = None,
        log_extension_factor: int = 0,
    ) -> CKKSParams:
        """Build CKKS parameters for single-party bootstrapping by default.

        Bootstrapping uses a dense main key and a temporary key with 32 nonzero
        coefficients unless overridden. Pass ``boot_params=None`` to disable it.
        The default uses three P primes. Pass ``p_prime_count=None`` for the
        dtype-specific minimum. See :meth:`generate_prefix` for the other parameters.
        """
        return cls.generate_prefix(
            log_n=log_n,
            depth=depth,
            log_precision=log_precision,
            log_range=log_range,
            dtype=dtype,
            boot_params=boot_params,
            bootstrap_level_bit_length=bootstrap_level_bit_length,
            bootstrap_message_ratio=bootstrap_message_ratio,
            p_prime_count=p_prime_count,
            p_prime_bit_length=p_prime_bit_length,
            ks_digit_size=ks_digit_size,
            noise_dist=noise_dist,
            security=security,
            secret_key_dist=secret_key_dist,
            log_extension_factor=log_extension_factor,
        )

    def context(
        self,
        *,
        default_level: int | None = None,
        track_plaintext_values: bool = False,
        strict_noise_bounds: bool = False,
        encoding_precision_bits: int = FLOAT_PRECISION_BITS,
    ) -> CKKSContext:
        from .keys import CKKSKeySet

        return CKKSKeySet(params=self).context(
            default_level=default_level,
            track_plaintext_values=track_plaintext_values,
            strict_noise_bounds=strict_noise_bounds,
            encoding_precision_bits=encoding_precision_bits,
        )
