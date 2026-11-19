from collections import defaultdict
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from math import sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, TypedDict, Unpack

from ren.device import Device
from ren.poly import Poly
from ren.container import PolyContainer
from ren.polydist import PolyDistWithBounds, SparseTernary
from ren.random import ChaChaRng
from ren.ring import RingSpec
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError
from ren.utils import find_contiguous_subset

from ._rlwe import RLWECiphertext
from .encoding import FLOAT_PRECISION_BITS
from .metadata import CKKSNoiseInfo
from .params import CKKSParams, KeySwitchParams, _project_key_switch_digits
from .utils import _require_galois_elt, normalize_rotation_amounts, steps_to_galois_elt

if TYPE_CHECKING:
    from .multiparty import MultipartyParams
    from .context import CKKSContext


class GaloisKeyRequest(TypedDict, total=False):
    """Galois keys to generate alongside the mandatory scheme keys.

    ``rotations``: a flat collection targets ``params.default_level``; a mapping
    selects rotations per level. ``conjugation``: ``True`` targets
    ``params.default_level``, an int selects its level. ``hoist_bootstrap_rotations``: generate a
    direct key for every bootstrap baby- and giant-step rotation so the linear
    transforms are double-hoisted (one decompose+mod-raise shared by the baby
    rotations and a single full ModDown per transform), trading a ~3-4x increase in rotation-key
    memory for a faster and more precise bootstrap. Bootstrapping's own key
    requirements are always included.
    """

    rotations: Collection[int] | Mapping[int, Collection[int]]
    conjugation: bool | int
    hoist_bootstrap_rotations: bool


def generate_secret(params: CKKSParams, *, level: int | None = None, device: str | None = None, rng: ChaChaRng | None = None) -> Poly:
    """Generate a secret using the configured distribution, at the highest ring by default."""
    ring = params.highest_ring if level is None else params.ring_at(level)
    if rng is None:
        device = Device.canonicalize(device)
        rng = Device[device].default_rng()
    elif device is not None and rng.device != Device.canonicalize(device):
        raise ParameterMismatchError(f"secret generation requested {Device.canonicalize(device)}, got RNG on {rng.device}")
    return Poly.rand(params.secret_key_dist, ring, rng=rng)


def keygen(
    params: CKKSParams,
    *,
    secret: Poly | None = None,
    device: str | None = None,
    rng: ChaChaRng | None = None,
    **galois: Unpack[GaloisKeyRequest],
) -> CKKSKeySet:
    """Generate a complete local key set, optionally from an existing secret key."""
    return CKKSKeySet.generate(params, secret=secret, device=device, rng=rng, **galois)


def _galois_elts_by_level(params: CKKSParams, request: GaloisKeyRequest) -> dict[int, frozenset[int]]:
    """Expand a key request into the complete Galois elements needed at each level."""
    rotations = request.get("rotations", ())
    conjugation = request.get("conjugation", False)
    default_level = params.default_level
    rotations_by_level = rotations if isinstance(rotations, Mapping) else {default_level: rotations}
    galois_elts_by_level: defaultdict[int, set[int]] = defaultdict(set)
    for level, amounts in rotations_by_level.items():
        level = params._require_level(level, name="Galois key level")
        normalized_amounts = normalize_rotation_amounts(amounts, params.slot_count)
        galois_elts_by_level[level].update(steps_to_galois_elt(amount, params.n) for amount in normalized_amounts)

    if type(conjugation) is bool:
        conjugation_level = default_level if conjugation else None
    else:
        conjugation_level = params._require_level(conjugation, name="Galois key level")
    if conjugation_level is not None:
        galois_elts_by_level[conjugation_level].add(2 * params.n - 1)

    hoist = request.get("hoist_bootstrap_rotations", False)
    if params.boot_params is not None:
        for level, galois_elts in params.boot_params.galois_elts_by_level(params, hoist=hoist).items():
            galois_elts_by_level[level].update(galois_elts)
    elif hoist:
        raise InvalidParameterError("hoist_bootstrap_rotations requires bootstrapping parameters")

    return {level: frozenset(galois_elts) for level, galois_elts in sorted(galois_elts_by_level.items()) if galois_elts}


def _galois_key_params(
    params: CKKSParams,
    galois_elts_by_level: Mapping[int, Collection[int]],
) -> dict[int, KeySwitchParams]:
    levels_by_galois_elt: defaultdict[int, set[int]] = defaultdict(set)
    for level, galois_elts in galois_elts_by_level.items():
        for galois_elt in galois_elts:
            levels_by_galois_elt[galois_elt].add(level)

    if len(levels_by_galois_elt) > 200:
        raise InvalidParameterError("Too many Galois keys; maximum allowed is 200 for memory reasons")

    galois_key_params = {}
    highest_moduli = params.highest_ring.moduli
    for galois_elt, levels in levels_by_galois_elt.items():
        windows = []
        for level in levels:
            target_moduli = params.ring_at(level).moduli
            start = find_contiguous_subset(highest_moduli, target_moduli)
            assert start is not None
            windows.append(range(start, start + len(target_moduli)))
        start = min(window.start for window in windows)
        stop = max(window.stop for window in windows)
        galois_key_params[galois_elt] = KeySwitchParams.for_target(
            highest_ring=params.highest_ring,
            target_ring=params.highest_ring[start:stop],
            p_primes=params.ks_primes,
            digit_sizes=params.ks_digit_sizes,
        )

    return dict(sorted(galois_key_params.items()))


@dataclass(slots=True, frozen=True)
class CKKSKeySet(PolyContainer):
    """CKKS parameters and device-resident key material."""

    params: CKKSParams
    sk: Poly | None = None
    pk: PublicKey | None = None
    rlk: SwitchKey | None = None
    galois_keys: Mapping[int, SwitchKey] = field(default_factory=dict)  # galois_elt -> key
    # Sparse-secret encapsulation switch keys (bootstrap only): dense_to_sparse
    # re-encrypts a boot-base ciphertext under the ephemeral sparse secret;
    # sparse_to_dense brings the mod-raised ciphertext back under the scheme secret.
    dense_to_sparse_key: SwitchKey | None = None
    sparse_to_dense_key: SwitchKey | None = None
    multiparty: MultipartyParams | None = None
    _rotation_galois_elts_by_level: tuple[frozenset[int], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        params = self.params
        highest_ring = params.highest_ring
        object.__setattr__(self, "galois_keys", MappingProxyType(dict(self.galois_keys)))
        secret_ring = params.ring_at(0) if self.multiparty is not None and self.multiparty.threshold is not None else highest_ring
        if self.sk is not None and self.sk.ring != secret_ring:
            raise ParameterMismatchError("secret key ring does not match CKKSParams ring")
        if self.pk is not None and self.pk.ct.ring != highest_ring:
            raise ParameterMismatchError("public key ring does not match CKKSParams ring")

        if self.rlk is not None:
            _require_switch_key_layout(self.rlk, p_primes=params.ks_primes, allowed_ring=highest_ring)

        rotation_galois_elts_by_level = [set() for _ in params.levels]
        for galois_elt, key in self.galois_keys.items():
            _require_galois_elt(galois_elt, params.n)
            _require_switch_key_layout(key, p_primes=params.ks_primes, allowed_ring=highest_ring)
            if galois_elt % 4 == 1:
                for level, level_params in enumerate(params.levels):
                    if find_contiguous_subset(key.target_ring.moduli, level_params.moduli) is not None:
                        rotation_galois_elts_by_level[level].add(galois_elt)

        if (self.dense_to_sparse_key is None) != (self.sparse_to_dense_key is None):
            raise ParameterMismatchError("dense-to-sparse and sparse-to-dense keys must be provided together")
        if self.dense_to_sparse_key is not None and self.sparse_to_dense_key is not None:
            boot_params = params.boot_params
            if boot_params is None or not boot_params.sparse_encapsulation:
                raise ParameterMismatchError("encapsulation keys require sparse encapsulation in CKKSParams")
            assert params.bootstrap_base_level is not None
            bootstrap_base_ring = RingSpec(n=params.n, moduli=params.bootstrap_base_level.moduli, dtype=params.dtype)
            _require_switch_key_layout(self.dense_to_sparse_key, p_primes=(params.ks_primes[0],), allowed_ring=bootstrap_base_ring)
            _require_switch_key_layout(self.sparse_to_dense_key, p_primes=params.ks_primes, allowed_ring=highest_ring)

        object.__setattr__(self, "_rotation_galois_elts_by_level", tuple(frozenset(galois_elts) for galois_elts in rotation_galois_elts_by_level))
        devices = {poly.device for poly in self.polys() if poly.device is not None}
        if len(devices) > 1:
            raise ParameterMismatchError(f"CKKS key set polynomial device mismatch: got {', '.join(sorted(devices))}")

    @classmethod
    def generate(
        cls,
        params: CKKSParams,
        *,
        secret: Poly | None = None,
        device: str | None = None,
        rng: ChaChaRng | None = None,
        **galois: Unpack[GaloisKeyRequest],
    ) -> Self:
        """Generate a complete local key set in NTT form, optionally from an existing secret key.

        When ``secret`` is provided, ``params.secret_key_dist`` describes
        the distribution from which it was sampled. See :class:`GaloisKeyRequest`
        for the Galois key keywords.
        """
        galois_elts_by_level = _galois_elts_by_level(params, galois)
        galois_key_params = _galois_key_params(params, galois_elts_by_level)
        if secret is None:
            secret = generate_secret(params, device=device, rng=rng)
        key_device = secret.device
        if device is not None:
            device = Device.canonicalize(device)
            if key_device not in (None, device):
                raise ParameterMismatchError(f"keygen requested {device}, got secret key on {key_device}")

        if secret.ring != params.highest_ring:
            raise ParameterMismatchError("secret key ring does not match CKKSParams ring")
        rng_device = rng.device if rng is not None else None
        device = Device.canonicalize(key_device or device or rng_device)
        if rng is None:
            rng = Device[device].default_rng()
        elif rng.device != device:
            raise ParameterMismatchError(f"CKKS key set is on {device}, got RNG on {rng.device}")

        # Generate a public key from the secret key.
        zero = Poly.zero(params.highest_ring)
        ct = RLWECiphertext._encrypt(zero, secret=secret, error_dist=params.noise_dist, rng=rng)
        pk = PublicKey(ct=ct)

        # Generate the relinearization key for the highest key-switch layout.
        top_ks_params = params._highest_key_switch_params
        # Align the secret to this Q level, then square it for the target secret.
        s_qt = secret._to_ring_small_coeffs(top_ks_params.ring_qt)
        rlk = SwitchKey.generate(top_ks_params, target_secret_qt=s_qt * s_qt, encrypting_secret=secret, error_dist=params.noise_dist, rng=rng)

        # Generate Galois keys with the requested Method-II parameters.
        # Permute-last: each key encrypts s (masked by prod(P) per segment)
        # under sigma_{g^-1}(s). Rotation applies sigma_g to the full key-switch
        # output, mapping the encrypting secret back to s, so the external
        # product runs on un-permuted, hoistable digits.
        galois_keys = {}
        for galois_elt, ks_params in galois_key_params.items():
            s_qt = secret._to_ring_small_coeffs(ks_params.ring_qt)
            g_inv = pow(galois_elt, -1, 2 * params.n)
            enc_secret = secret.automorphism(g_inv)
            galois_keys[galois_elt] = SwitchKey.generate(
                ks_params,
                target_secret_qt=s_qt,
                encrypting_secret=enc_secret,
                error_dist=params.noise_dist,
                rng=rng,
            )

        dense_to_sparse_key = sparse_to_dense_key = None
        boot_params = params.boot_params
        if boot_params is not None and boot_params.sparse_encapsulation_weight is not None:
            # Sample an ephemeral sparse secret s' of the requested weight.
            # Dense->sparse encrypts s under s' at the boot-base modulus (the
            # only material s' protects). Sparse->dense encrypts s' under s
            # over the full bootstrap modulus. Only the switch keys persist.
            assert params.bootstrap_base_level is not None
            bootstrap_base_ring = RingSpec(n=params.n, moduli=params.bootstrap_base_level.moduli, dtype=params.dtype)
            enc_ks_params = KeySwitchParams.for_target(
                highest_ring=bootstrap_base_ring,
                target_ring=bootstrap_base_ring,
                p_primes=(params.ks_primes[0],),
                digit_sizes=(1,) * len(bootstrap_base_ring.moduli),
            )
            sparse_dist = SparseTernary(n=params.n, h=boot_params.sparse_encapsulation_weight)
            s_sparse = Poly.rand(sparse_dist, params.highest_ring, rng=rng)

            target_secret_qt = secret._to_ring_small_coeffs(enc_ks_params.ring_qt)
            dense_to_sparse_key = SwitchKey.generate(
                enc_ks_params, target_secret_qt=target_secret_qt, encrypting_secret=s_sparse, error_dist=params.noise_dist, rng=rng
            )

            target_secret_qt = s_sparse._to_ring_small_coeffs(top_ks_params.ring_qt)
            sparse_to_dense_key = SwitchKey.generate(
                top_ks_params, target_secret_qt=target_secret_qt, encrypting_secret=secret, error_dist=params.noise_dist, rng=rng
            )
        return cls(
            params=params,
            sk=secret,
            pk=pk,
            rlk=rlk,
            galois_keys=galois_keys,
            dense_to_sparse_key=dense_to_sparse_key,
            sparse_to_dense_key=sparse_to_dense_key,
        ).map_polys(Poly.ntt)

    @property
    def num_parties(self) -> int:
        """Every party contributes an independent secret, regardless of the decryption threshold."""
        return 1 if self.multiparty is None else self.multiparty.num_parties

    @property
    def secret_l1_norm_bound(self) -> float:
        return self.num_parties * self.params.secret_key_dist.coeff_1_norm_bound()

    @property
    def secret_expected_squared_l2_norm(self) -> float:
        return self.num_parties * self.params.secret_key_dist.expected_squared_l2_norm()

    @property
    def device(self) -> str:
        return next((device for poly in self.polys() if (device := poly.device) is not None), None) or Device.DEFAULT

    def context(
        self,
        *,
        default_level: int | None = None,
        track_plaintext_values: bool = False,
        strict_noise_bounds: bool = False,
        encoding_precision_bits: int = FLOAT_PRECISION_BITS,
    ) -> CKKSContext:
        """Create the runtime context; ``default_level`` defaults to ``params.default_level``."""
        from .context import CKKSContext

        default_level = self.params.default_level if default_level is None else default_level

        return CKKSContext(
            keyset=self,
            default_level=default_level,
            track_plaintext_values=track_plaintext_values,
            strict_noise_bounds=strict_noise_bounds,
            encoding_precision_bits=encoding_precision_bits,
        )

    def public(self) -> Self:
        """Return this key set with the secret key field cleared."""
        return self if self.sk is None else self.replace(sk=None)

    def get_galois_key(self, galois_elt: int) -> SwitchKey:
        """Return the Galois key for ``galois_elt``."""
        galois_elt = galois_elt % (2 * self.params.n)
        if galois_elt not in self.galois_keys:
            raise RuntimeError(f"No Galois key for galois_elt={galois_elt}")
        return self.galois_keys[galois_elt]

    def _rotation_galois_elts_at(self, level: int) -> frozenset[int]:
        return self._rotation_galois_elts_by_level[self.params._require_level(level)]

    def __reduce_ex__(self, _protocol):
        """Realize all key material together before pickle reduces each poly.

        This keeps every ``Poly.__reduce__`` on the already-realized fast path
        instead of scheduling a separate realization per key polynomial. The
        explicit constructor reducer also leaves runtime device placement out
        of the payload, so polynomials load onto the active device.
        """
        self.realize()
        return (
            type(self),
            (
                self.params,
                self.sk,
                self.pk,
                self.rlk,
                dict(self.galois_keys),
                self.dense_to_sparse_key,
                self.sparse_to_dense_key,
                self.multiparty,
            ),
        )


@dataclass(slots=True, frozen=True)
class PublicKey(PolyContainer):
    ct: RLWECiphertext

    def encrypt(self, m: Poly, params: CKKSParams, *, noise_info: CKKSNoiseInfo, rng: ChaChaRng | None = None) -> RLWECiphertext:
        """Encrypt a polynomial without an active CKKS context.

        Project the key to the message ring and add encryption noise to
        ``noise_info``. Parameters supply the ephemeral-secret and error distributions.
        """
        if rng is not None and self.ct.device is not None and rng.device != self.ct.device:
            raise ParameterMismatchError(f"public key is on {self.ct.device}, got RNG on {rng.device}")
        if not m.ring.is_quotient_of(self.ct.ring):
            raise ParameterMismatchError("ring is not a quotient of the public key's ring")
        pk = self.ct.map_polys(lambda poly: poly.to_quotient_ring(m.ring))
        if rng is None:
            rng = Device[Device.canonicalize(self.ct.device)].default_rng()
        ring = m.ring
        n = ring.n
        # The multiplicative factor uses the same distribution as the secret key.
        r = Poly.rand(params.secret_key_dist, ring, rng=rng)
        e0 = Poly.rand(params.noise_dist, ring, rng=rng)
        e1 = Poly.rand(params.noise_dist, ring, rng=rng)

        # Multiply the public key by r.
        masked_pk = pk._multiply_small_polynomial_coeffs(r, norm_bound=1)
        # Add error polynomials to each coordinate.
        noise_bound = params.noise_dist.coeff_inf_norm_bound()
        # TODO handle coeffs vs slots noise
        error_noise = CKKSNoiseInfo(noise_bound=n * noise_bound, noise_heuristic_bound=sqrt(2 * n / 3) * noise_bound, msg_bound=0)
        # TODO use heuristic noise
        error_ct = RLWECiphertext(a=e0, b=e1, noise_info=error_noise)
        # Add the payload polynomial.
        payload = RLWECiphertext(a=Poly.zero(ring), b=m, noise_info=noise_info)
        return masked_pk + error_ct + payload


@dataclass(slots=True, frozen=True)
class SwitchKey(PolyContainer):
    """
    Method-II key-switching key.

    Stores the key over extended base Q̃ = Q ∪ P, with one digit per D segment.
    The target secret is embedded via target_secret·prod(P) on each segment's
    Q primes.
    """

    digits: tuple[RLWECiphertext, ...]
    digit_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.digits:
            raise ParameterMismatchError("switch key must contain at least one digit")
        if len(self.digit_sizes) != len(self.digits) or any(size <= 0 for size in self.digit_sizes):
            raise ParameterMismatchError("switch key digit sizes must match its digits")
        if self.p_prime_count <= 0:
            raise ParameterMismatchError("switch key must include a P basis")
        if not all(digit.ring == self.ring_qt for digit in self.digits):
            raise ParameterMismatchError("ciphertext ring mismatch between switch key digits")

    @classmethod
    def generate(
        cls,
        ks_params: KeySwitchParams,
        *,
        target_secret_qt: Poly,
        encrypting_secret: Poly,
        error_dist: PolyDistWithBounds,
        noise_info: CKKSNoiseInfo | None = None,
        rng: ChaChaRng,
    ) -> Self:
        """
        Generate a Method-II switch key.

        All switch keys use the same digit structure:
          b_i = -a_i·s' + e_i + target_secret·P  (masked on segment i)

        where the encrypting secret s' is ``encrypting_secret``.

        For relinearization, target_secret = s² and encrypting_secret = s. For
        Galois rotation (permute-last), target_secret = s and encrypting_secret =
        σ_{g⁻¹}(s). Bootstrap encapsulation supplies the dense and sparse
        secrets in the required direction.

        D-segments are contiguous ranges of the target Q basis. ``error_dist``
        is the params' noise distribution for keys over the base ring; KG+ keys
        over the degree-2n extension ring pass the matching extension-ring
        distribution. ``noise_info`` stamps every digit. It describes the
        digit's fresh error only, since the masked target cancels in a key
        switch, and defaults to unbounded, as the scheme's own keys are tracked.
        """
        if noise_info is None:
            noise_info = CKKSNoiseInfo.unbounded()
        digits: list[RLWECiphertext] = []

        for mask_term in _switch_key_mask_terms(target_secret_qt, ks_params):
            ct = RLWECiphertext._encrypt(mask_term, secret=encrypting_secret, error_dist=error_dist, output_noise_info=noise_info, rng=rng)
            digits.append(ct)

        return cls(digits=tuple(digits), digit_sizes=ks_params.digit_sizes)

    @property
    def ring_qt(self) -> RingSpec:
        return self.digits[0].ring

    @property
    def p_prime_count(self) -> int:
        return len(self.ring_qt.moduli) - sum(self.digit_sizes)

    @property
    def p_primes(self) -> tuple[int, ...]:
        return self.ring_qt.moduli[: self.p_prime_count]

    @property
    def target_ring(self) -> RingSpec:
        return self.ring_qt[self.p_prime_count :]

    def _project_digits(self, target_ring: RingSpec) -> tuple[tuple[int, range], ...]:
        """Map the key digits that overlap a target Q ring to their target ranges."""
        if self.ring_qt.n != target_ring.n or self.ring_qt.dtype is not target_ring.dtype:
            raise ParameterMismatchError("switch key and target use different rings")
        projection = _project_key_switch_digits(self.target_ring.moduli, self.digit_sizes, target_ring.moduli)
        if projection is None:
            raise ParameterMismatchError("switch key does not cover the target Q ring")
        return projection


def _require_switch_key_layout(
    key: SwitchKey,
    *,
    p_primes: tuple[int, ...],
    allowed_ring: RingSpec,
) -> None:
    if set(key.p_primes) != set(p_primes):
        raise ParameterMismatchError("switch-key P basis does not match CKKSParams")

    target_ring = key.target_ring
    if target_ring.n != allowed_ring.n or target_ring.dtype is not allowed_ring.dtype:
        raise ParameterMismatchError("switch-key ring does not match CKKSParams")
    if find_contiguous_subset(allowed_ring.moduli, target_ring.moduli) is None:
        raise ParameterMismatchError("switch-key Q basis is outside the allowed CKKSParams ring")


def _switch_key_mask_terms(target_secret_qt: Poly, ks_params: KeySwitchParams) -> Iterator[Poly]:
    """Embed a switch-key target secret into each decomposition digit."""
    for digit_range in ks_params.digit_ranges:
        yield target_secret_qt.scale_by_primes(ks_params.mask_scalars(digit_range))
