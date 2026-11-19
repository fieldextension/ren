import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from functools import cache
from types import MappingProxyType
from typing import NamedTuple, Self, Unpack

from ren.container import PolyContainer
from ren.device import Device
from ren.poly import Poly
from ren.polydist import DiscreteGaussian, Ternary
from ren.random import ChaChaRng
from ren.ring import RingSpec, _max_modulus_for_dtype
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError
from ren.utils import find_contiguous_subset

from ._rlwe import RLWECiphertext, _fresh_encryption_noise
from ._security import max_log2_modulus
from .ckks import _keyswitch
from .keys import CKKSKeySet, GaloisKeyRequest, PublicKey, SwitchKey, _galois_elts_by_level, _galois_key_params
from .metadata import CKKSNoiseInfo
from .params import (
    CKKSParams,
    KeySwitchParams,
    _default_ks_prime_bits,
    _project_key_switch_digits,
)
from .utils import _require_galois_elt, select_moduli, steps_to_galois_elt


@dataclass(frozen=True, slots=True)
class TransmissionKeys(PolyContainer):
    """Public KG+ material uploaded by a client and expanded by a server.

    Section 4.1 of the KG+ paper describes this artifact as an extension-ring
    encryption key, an evaluation-key transmission ciphertext, master rotation
    keys, and a homing key.
    """

    params: CKKSParams
    extension_encryption_key: RLWECiphertext
    evaluation_transmission_key: RLWECiphertext
    master_galois_keys: Mapping[int, SwitchKey]
    homing_key: SwitchKey
    galois_key_rings: Mapping[int, RingSpec]

    def __reduce_ex__(self, _protocol):
        """Realize all material together, then rebuild through the constructor.

        The read-only mapping proxies ``__post_init__`` installs do not pickle;
        the constructor reducer hands plain dicts over, as :class:`CKKSKeySet`
        does, and leaves runtime device placement out of the payload.
        """
        self.realize()
        return (
            type(self),
            (
                self.params,
                self.extension_encryption_key,
                self.evaluation_transmission_key,
                dict(self.master_galois_keys),
                self.homing_key,
                dict(self.galois_key_rings),
            ),
        )

    def __post_init__(self) -> None:
        master_galois_keys = MappingProxyType(dict(self.master_galois_keys))
        galois_key_rings = MappingProxyType(dict(self.galois_key_rings))
        object.__setattr__(self, "master_galois_keys", master_galois_keys)
        object.__setattr__(self, "galois_key_rings", galois_key_rings)

        params = self.params
        key_ring = params._highest_key_switch_params.ring_qt
        extension_key_ring = _extension_ring(key_ring)
        if self.extension_encryption_key.ring != extension_key_ring:
            raise ParameterMismatchError("KG+ extension encryption key ring does not match the highest P + Q basis")

        transmission_ring = self.evaluation_transmission_key.ring
        key_moduli = key_ring.moduli
        if (
            transmission_ring.n != extension_key_ring.n
            or transmission_ring.dtype is not extension_key_ring.dtype
            or len(transmission_ring.moduli) <= len(key_moduli)
            or transmission_ring.moduli[-len(key_moduli) :] != key_moduli
        ):
            raise ParameterMismatchError("KG+ evaluation transmission key ring must be a nonempty auxiliary basis followed by P + Q")

        if self.homing_key.target_ring != key_ring:
            raise ParameterMismatchError("KG+ homing key target does not match the highest P + Q basis")
        if any(key.target_ring != extension_key_ring for key in master_galois_keys.values()):
            raise ParameterMismatchError("KG+ master Galois key target does not match the extension P + Q basis")

        # The server derives every rotation key by composing master rotations,
        # so each element must be a slot rotation (or the conjugation), and
        # each window must be what keygen resolves for some set of levels: the
        # hull of the level rings it contains. Any other window derives a key
        # that a level inside it cannot use, or that no level can use when
        # levels cross (Q[2:3] between Q[0:3] and Q[2:5]), which only shows at
        # the first rotation on the server.
        master_rotation_elts = _rotation_steps_by_galois_elt(params.slot_count, 2 * extension_key_ring.n)
        for galois_elt in master_galois_keys:
            _require_galois_elt(galois_elt, extension_key_ring.n)
            if galois_elt not in master_rotation_elts:
                raise InvalidParameterError(f"KG+ master Galois element {galois_elt} is not a slot rotation over the extension ring")
        rotation_elts = _rotation_steps_by_galois_elt(params.slot_count, 2 * params.n)
        conjugation_elt = 2 * params.n - 1
        highest_moduli = params.highest_ring.moduli
        level_windows = [
            (start, start + len(level.moduli))
            for level in params.levels
            if (start := find_contiguous_subset(highest_moduli, level.moduli)) is not None
        ]
        for galois_elt, ring in galois_key_rings.items():
            _require_galois_elt(galois_elt, params.n)
            if galois_elt != conjugation_elt and galois_elt not in rotation_elts:
                raise InvalidParameterError(f"KG+ Galois element {galois_elt} is neither a slot rotation nor the conjugation")
            if ring.n != params.n or ring.dtype is not params.dtype:
                raise ParameterMismatchError("KG+ Galois key target uses a different ring")
            start = find_contiguous_subset(highest_moduli, ring.moduli)
            if start is None:
                raise ParameterMismatchError("KG+ Galois key target is not a contiguous subrange of the highest Q basis")
            stop = start + len(ring.moduli)
            inside = [(level_start, level_stop) for level_start, level_stop in level_windows if start <= level_start and level_stop <= stop]
            if not inside or min(level_start for level_start, _ in inside) != start or max(level_stop for _, level_stop in inside) != stop:
                raise ParameterMismatchError(
                    f"KG+ Galois key window for element {galois_elt} does not span whole levels: it must be the hull of the levels it contains"
                )

        devices = {poly.device for poly in self.polys() if poly.device is not None}
        if len(devices) > 1:
            raise ParameterMismatchError(f"KG+ transmission key device mismatch: got {', '.join(sorted(devices))}")

    @property
    def device(self) -> str:
        return next((device for poly in self.polys() if (device := poly.device) is not None), None) or Device.DEFAULT

    @classmethod
    def generate(
        cls,
        params: CKKSParams,
        *,
        secret: Poly,
        max_master_keys: int = 3,
        security: int | None = 128,
        rng: ChaChaRng | None = None,
        **galois: Unpack[GaloisKeyRequest],
    ) -> Self:
        """Generate the client artifact for the requested ordinary key layout.

        Sample ``secret`` from ``params.secret_key_dist``. The server uses that
        distribution to compute the secret's expected squared L2 norm.

        ``galois`` is the ordinary keygen's :class:`GaloisKeyRequest`:
        ``rotations`` (a flat collection targets the default level, a mapping
        selects steps per level), ``conjugation`` (``True`` targets the default
        level, an int selects its level) and ``hoist_bootstrap_rotations``.
        Bootstrapping's own Galois keys are always included, and under KG+ the
        hoisted ones add no upload, only derivation work.

        ``max_master_keys`` caps how many master rotation keys are uploaded (the
        paper's per-level master count m, not its hierarchy depth k): the
        requested steps are covered by a power-radix basis of at most this many
        masters, trading a larger upload for fewer noise-adding key-switch
        compositions per derived key.

        The modulus check uses degree n for the homing key and degree 2n for
        the transmission and master keys. Pass ``security=None`` to skip the check.
        For checked bootstrapping, use a dense ``Ternary`` main key,
        ``sparse_encapsulation_weight=None``, and ``log_extension_factor=1`` when generating parameters.
        """
        if max_master_keys < 1:
            raise InvalidParameterError(f"max_master_keys must be positive, got {max_master_keys}")
        if params.boot_params is not None and params.boot_params.sparse_encapsulation:
            raise InvalidParameterError("KG+ cannot derive bootstrap sparse-encapsulation keys server-side; use params without sparse encapsulation")
        if secret.ring != params.highest_ring:
            raise ParameterMismatchError("secret key ring does not match CKKSParams ring")

        rng_device = rng.device if rng is not None else None
        device = Device.canonicalize(secret.device or rng_device)
        if rng is not None and rng.device != device:
            raise ParameterMismatchError(f"secret key is on {device}, got RNG on {rng.device}")
        if not isinstance(params.noise_dist, DiscreteGaussian):
            raise InvalidParameterError(f"KG+ requires a DiscreteGaussian noise distribution, got {type(params.noise_dist).__name__}")
        extension_noise = DiscreteGaussian(n=2 * params.noise_dist.n, stdev=params.noise_dist.stdev)
        key_ring = params._highest_key_switch_params.ring_qt
        extension_key_ring = _extension_ring(key_ring)
        key_moduli = key_ring.moduli

        galois_key_rings, rotation_steps = _resolve_galois_key_rings(params, galois)
        master_steps = _master_rotation_steps(rotation_steps, max_master_keys)
        aux_primes = _select_kg_plus_aux_primes(params, need_master=bool(master_steps), security=security)

        if rng is None:
            rng = Device[device].default_rng()
        # KG+ samples an independent s₁ (from the params' secret distribution,
        # which describes ``secret`` too) and forms s̃ = s + Y·s₁ over the
        # extension ring. The homing key later switches only s₁ back to s.
        aux_secret = Poly.rand(params.secret_key_dist, secret.ring, rng=rng)
        extension_secret = _embed_pair(secret, aux_secret, extension_key_ring)

        homing_key = _generate_homing_key(
            params,
            secret=secret,
            aux_secret=aux_secret,
            homing_primes=aux_primes.homing,
            rng=rng,
        )

        # enck': Enc_s̃(0) over the extension of P ∪ Q. Homing its even part
        # yields the base public key; adding per-digit gadget masks to its `a`
        # part yields the identity rotation key that seeds master derivation.
        extension_encryption_key = RLWECiphertext._encrypt(
            Poly.zero(extension_key_ring), secret=extension_secret, error_dist=extension_noise, rng=rng
        )

        # TrmEvk: Enc_s̃(P'·(s² + Y·σ₋₁(s))) over the extension of P' ∪ P ∪ Q.
        # The server multiplies by per-digit mask constants (introducing the
        # P·G_i factor) and rescales P' away, then homes even/odd parts into
        # the relinearization and conjugation keys.
        transmission_ring = RingSpec(n=2 * params.n, moduli=aux_primes.transmission + key_moduli, dtype=params.dtype)
        s = secret
        conjugation_elt = 2 * params.n - 1
        conjugated = s.automorphism(conjugation_elt) if conjugation_elt in galois_key_rings else Poly.zero(s.ring)
        transmission_prod = math.prod(aux_primes.transmission)
        message = _embed_pair(s * s, conjugated, transmission_ring).scale_by_primes(
            (0,) * len(aux_primes.transmission) + tuple(transmission_prod % prime for prime in key_moduli)
        )
        evaluation_transmission_key = RLWECiphertext._encrypt(
            message,
            secret=extension_secret,
            error_dist=extension_noise,
            output_noise_info=CKKSNoiseInfo.unbounded(),
            rng=rng,
        )

        # Master rotation keys: permute-first Method-II keys over the extension
        # ring at the P^(m) ∪ P ∪ Q modulus, one digit each (dnum = 1).
        master_galois_keys: dict[int, SwitchKey] = {}
        if master_steps:
            master_ks_params = KeySwitchParams.for_target(
                highest_ring=extension_key_ring,
                target_ring=extension_key_ring,
                p_primes=aux_primes.master,
                digit_sizes=(len(extension_key_ring.moduli),),
            )
            for step in sorted(master_steps):
                galois_elt = steps_to_galois_elt(step, 2 * params.n)
                master_galois_keys[galois_elt] = SwitchKey.generate(
                    master_ks_params,
                    target_secret_qt=extension_secret.automorphism(galois_elt).lift_to_ring(master_ks_params.ring_qt),
                    encrypting_secret=extension_secret,
                    error_dist=extension_noise,
                    rng=rng,
                )

        return cls(
            params=params,
            extension_encryption_key=extension_encryption_key,
            evaluation_transmission_key=evaluation_transmission_key,
            master_galois_keys=master_galois_keys,
            homing_key=homing_key,
            galois_key_rings=galois_key_rings,
        )

    def derive_keyset(self) -> CKKSKeySet:
        """Derive an ordinary public key set from this artifact on the server.

        The relinearization key uses the params' highest key-switch layout;
        each derived Galois key is projected onto its recorded Q window,
        matching the per-level layouts of direct keygen.
        Derived Galois keys are converted to this codebase's permute-last
        convention by applying the inverse automorphism to each homed digit.
        """
        params = self.params
        ks_params = params._highest_key_switch_params
        conjugation_elt = 2 * params.n - 1

        # pk: home enck' and drop the P basis (paper Cor 3.2). The homing key
        # carries its fresh noise, so the homed ciphertext's bound follows from
        # enck's; the two rescales (P^(h) inside homing, then P) round under
        # the scheme secret, which the noise model otherwise leaves out.
        even, odd = _split_extension_ciphertext(self.extension_encryption_key)
        pk_ct = _home_extension_ciphertext(even, odd.a.rotate(1), self.homing_key)._rescale(params.highest_ring.moduli)
        rounding = _rescale_rounding_noise(params)
        pk = PublicKey(ct=replace(pk_ct, noise_info=CKKSNoiseInfo.sum((pk_ct.noise_info, rounding, rounding))))

        # rlk / cjk: per-digit mask extraction from TrmEvk (Thm 3.3), then homing.
        digit_cts = self._transmission_digit_ciphertexts()
        homed = (_home_extension_ciphertext(even, odd.a.rotate(1), self.homing_key) for even, odd in map(_split_extension_ciphertext, digit_cts))
        rlk = SwitchKey(digits=tuple(homed), digit_sizes=ks_params.digit_sizes)

        rotation_rings = dict(self.galois_key_rings)
        conjugation_ring = rotation_rings.pop(conjugation_elt, None)
        galois_keys: dict[int, SwitchKey] = {}
        if conjugation_ring is not None:
            digits, digit_sizes, window_ring = _project_extension_digits(digit_cts, params=params, target_ring=conjugation_ring)
            homed = (_home_extension_ciphertext(odd, even.a, self.homing_key) for even, odd in map(_split_extension_ciphertext, digits))
            digits = tuple(ct.map_polys(lambda poly: poly.to_quotient_ring(window_ring).automorphism(conjugation_elt)) for ct in homed)
            galois_keys[conjugation_elt] = SwitchKey(digits=digits, digit_sizes=digit_sizes)

        # Rotation keys: compose master keys onto the identity key over R'
        # (Sec 2.4 / Thm 3.4), home, and convert to permute-last.
        if rotation_rings:
            masters = self._masters_by_step()
            identity_digits = self._identity_rotation_digits()
            for galois_elt, target_ring in rotation_rings.items():
                digits, digit_sizes, window_ring = _project_extension_digits(identity_digits, params=params, target_ring=target_ring)
                step = _rotation_step_from_galois_elt(galois_elt, slot_count=params.slot_count, modulus=2 * params.n)
                digits = _derive_intermediate_rotation_digits(digits, masters, step)
                homed = (_home_extension_ciphertext(even, odd.a.rotate(1), self.homing_key) for even, odd in map(_split_extension_ciphertext, digits))
                inverse_elt = pow(galois_elt, -1, 2 * params.n)
                digits = tuple(ct.map_polys(lambda poly: poly.to_quotient_ring(window_ring).automorphism(inverse_elt)) for ct in homed)
                galois_keys[galois_elt] = SwitchKey(digits=digits, digit_sizes=digit_sizes)

        # The host-side parity split leaves every derived polynomial in the
        # coefficient domain, whereas direct keygen stores keys as NTT values
        # and consumers key-switch in the NTT domain (a ``ren.jit`` plan would
        # pin its own conversion of every key it touches). Convert once, and
        # realize: homing, the window reduction, the automorphism and the NTT
        # are still lazy, and a plan traced over lazy keys would replay that
        # tail on every call.
        return CKKSKeySet(params=params, pk=pk, rlk=rlk, galois_keys=galois_keys).map_polys(Poly.ntt).realize()

    def _transmission_digit_ciphertexts(self) -> tuple[RLWECiphertext, ...]:
        """Recover the intermediate evaluation-key digits from TrmEvk (Thm 3.3).

        TrmEvk encrypts P'·(s² + Y·σ₋₁(s)). Multiplying by the balanced lift of
        each digit's P·G_i mask pattern (taken over P ∪ Q only, magnitude at
        most floor(PQ/2) <= P') and rescaling P' away yields Enc_s̃(P·G_i·m) with a
        small extra error of roughly mask/P' times the fresh noise.
        """
        ks_params = self.params._highest_key_switch_params
        key_moduli = ks_params.ring_qt.moduli
        transmission = self.evaluation_transmission_key

        digits = []
        for digit_range in ks_params.digit_ranges:
            constant = _balanced_crt_lift(ks_params.mask_scalars(digit_range), key_moduli)
            const_poly = Poly.const(constant, transmission.ring)
            digits.append(transmission._multiply_small_polynomial_coeffs(const_poly, norm_bound=abs(constant))._rescale(key_moduli))
        return tuple(digits)

    def _identity_rotation_digits(self) -> tuple[RLWECiphertext, ...]:
        """The index-0 rotation key over R', built from enck' by mask addition.

        Adding the P·G_i mask constant to the `a` part of Enc_s̃(0) adds
        P·G_i·s̃ to the plaintext, which is exactly the i-th digit of a
        permute-first rotation key of index 0.
        """
        ks_params = self.params._highest_key_switch_params
        key_moduli = ks_params.ring_qt.moduli
        enck = self.extension_encryption_key

        digits = []
        for digit_range in ks_params.digit_ranges:
            constant = _balanced_crt_lift(ks_params.mask_scalars(digit_range), key_moduli)
            mask = Poly.const(constant, enck.ring)
            digits.append(replace(enck, a=enck.a + mask, noise_info=CKKSNoiseInfo.unbounded()))
        return tuple(digits)

    def _masters_by_step(self) -> tuple[tuple[int, int, SwitchKey], ...]:
        """Master keys as (step, extension galois elt, key), largest step first."""
        params = self.params
        resolved = []
        for galois_elt, key in self.master_galois_keys.items():
            step = _rotation_step_from_galois_elt(galois_elt, slot_count=params.slot_count, modulus=4 * params.n)
            resolved.append((step, galois_elt, key))
        return tuple(sorted(resolved, reverse=True))


class KgPlusAuxPrimes(NamedTuple):
    """Fresh special moduli selected per artifact, disjoint from P ∪ Q."""

    transmission: tuple[int, ...]  # P' for TrmEvk; covers the mask-constant magnitude ~ PQ
    homing: tuple[int, ...]  # P^(h) for the homing key (base ring)
    master: tuple[int, ...]  # P^(m) for master rotation keys (extension ring, dnum = 1)


# Noise margin of the master modulus over P ∪ Q. A dnum = 1 master key switch
# adds key noise of about σ·√(2n/12)·PQ/P^(m) to a key whose window reaches the
# top of Q (measured at n = 2^16: std 340 per composition at zero margin, against
# 6.5 of rounding under a sparse secret); 12 bits brings it to about a hundredth
# of the rounding term at the cost of at most one more master prime.
_MASTER_MARGIN_BITS = 12


def _kg_plus_modulus_limits(params: CKKSParams, security: int) -> dict[int, int]:
    secret, error = params.secret_key_dist, params.noise_dist
    # Interleaving two independent dense keys preserves the coefficient distribution.
    if not isinstance(secret, Ternary):
        # TODO: Estimate the sparse extension key with exactly h even-indexed and h odd-indexed nonzero coefficients.
        raise InvalidParameterError(f"no {security}-bit security estimate for interleaved {secret} keys")
    limits = {}
    for label, n in (("homing", params.n), ("transmission", 2 * params.n)):
        try:
            limits[n] = max_log2_modulus(replace(secret, n=n), replace(error, n=n), security)
        except ValueError as exc:
            raise InvalidParameterError(f"KG+ {label} keys at degree {n}: {exc}") from exc
    return limits


def _select_kg_plus_aux_primes(params: CKKSParams, *, need_master: bool, security: int | None) -> KgPlusAuxPrimes:
    limits = _kg_plus_modulus_limits(params, security) if security is not None else None
    bit_length = _default_ks_prime_bits(params.dtype)
    key_ring = params._highest_key_switch_params.ring_qt
    key_moduli = key_ring.moduli
    exclude = set(key_moduli)

    def pick(count: int, prime_bits: int = bit_length) -> tuple[int, ...]:
        primes = tuple(select_moduli(n=2 * params.n, num_primes=count, bit_length=prime_bits, exclude=exclude))
        exclude.update(primes)
        return primes

    key_modulus_bits = key_ring.q.bit_length()
    transmission = pick(math.ceil(key_modulus_bits / bit_length))
    while math.prod(transmission) < key_ring.q // 2:
        transmission += pick(1)

    # P^(h) shares the degree-n modulus budget with P and Q (large dnum^(h) in the paper).
    if limits is None:
        homing = pick(len(params.ks_primes))
    else:
        headroom_bits = limits[params.n] - (key_ring.q - 1).bit_length()
        homing = pick(*_homing_prime_layout(params, key_moduli, headroom_bits=headroom_bits))
        if math.prod(homing) <= max(key_moduli):
            # Only reachable when the headroom allows a single prime of the
            # widest key prime's own width, drawn below the key primes; one
            # more bit of headroom lets the layout draw a wider one.
            raise InvalidParameterError(
                "KG+ homing modulus does not cover the widest key-switch prime; params need one more bit of degree-n headroom"
            )

    master: tuple[int, ...] = ()
    if need_master:
        # One digit spans all of P ∪ Q, so P^(m) must clear it by the noise
        # margin: size by bits like P', then top up while the drawn primes fall
        # short (they sit below 2^bits, and uint32's 30-bit primes ≡ 1 mod 4n
        # run well short of it at n = 2^16).
        master = pick(math.ceil((key_modulus_bits + _MASTER_MARGIN_BITS) / bit_length))
        while math.prod(master) < key_ring.q << _MASTER_MARGIN_BITS:
            master += pick(1)
    primes = KgPlusAuxPrimes(transmission=transmission, homing=homing, master=master)
    if limits is not None:
        # Check the actual primes after selecting them by bit length.
        for label, degree, basis in (
            ("homing", params.n, primes.homing),
            ("transmission", 2 * params.n, primes.transmission),
            ("master", 2 * params.n, primes.master),
        ):
            if not basis:
                continue
            headroom_bits = limits[degree] - (math.prod(basis) * key_ring.q - 1).bit_length()
            if headroom_bits < 0:
                raise InvalidParameterError(
                    f"KG+ {label} keys need {-headroom_bits} more bits than the {security}-bit budget allows at the degree-{degree} ring: "
                    f"P ∪ Q is {key_modulus_bits} bits and the {label} special modulus adds "
                    f"{sum(prime.bit_length() for prime in basis)}; reduce the params' modulus or depth"
                )
    return primes


def _resolve_galois_key_rings(params: CKKSParams, request: GaloisKeyRequest) -> tuple[dict[int, RingSpec], set[int]]:
    """Resolve a Galois key request to target Q windows and the rotation steps it needs.

    Uses the ordinary keygen's :func:`_galois_elts_by_level` and window resolution,
    so KG+ derives exactly the keys direct keygen would generate for the same
    request, bootstrapping's included.
    """
    galois_key_params = _galois_key_params(params, _galois_elts_by_level(params, request))
    galois_key_rings = {galois_elt: ks_params.target_ring for galois_elt, ks_params in galois_key_params.items()}
    conjugation_elt = 2 * params.n - 1
    rotation_steps = {
        _rotation_step_from_galois_elt(galois_elt, slot_count=params.slot_count, modulus=2 * params.n)
        for galois_elt in galois_key_rings
        if galois_elt != conjugation_elt
    }
    return galois_key_rings, rotation_steps


def _homing_prime_layout(params: CKKSParams, key_moduli: tuple[int, ...], *, headroom_bits: int) -> tuple[int, int]:
    """Return the number of homing primes and the bit length of each.

    Their product must exceed the largest key prime. A larger product can
    fit more key primes per digit and reduce the homing key upload. Once all
    key primes fit in one digit, a larger modulus buys nothing.

    One homing prime suffices if it is a bit wider than the largest key
    prime. The dtype and CUDA sampler limit how wide a prime can be.
    The lower width target leaves enough primes congruent to 1 modulo 4n.
    """
    widest_key_bits = max(key_moduli).bit_length()
    if headroom_bits < widest_key_bits:
        raise InvalidParameterError(
            f"params leave only {headroom_bits} bits of degree-{params.n} modulus headroom above P ∪ Q; "
            f"the KG+ homing modulus must cover the widest key-switch prime ({widest_key_bits} bits)"
        )
    total_bits = min(headroom_bits, math.prod(key_moduli).bit_length() + 1)
    # RingSpec takes moduli below 2^31 (uint32) and 2^63 (uint64);
    # the CUDA uniform sampler needs them below 2^62.
    max_prime_bits = min(max(_default_ks_prime_bits(params.dtype), widest_key_bits) + 1, _max_modulus_for_dtype(params.dtype).bit_length() - 1, 62)
    min_prime_bits = params.logn + 10
    count = math.ceil(total_bits / max_prime_bits)
    while count > 1 and total_bits // count < min_prime_bits:
        count -= 1
    return count, min(total_bits // count, max_prime_bits)


def _master_rotation_steps(target_steps: Collection[int], max_master_keys: int) -> set[int]:
    """Power-radix master step basis covering every target step.

    Any step up to ``max(target_steps)`` decomposes greedily over the returned
    basis {B⁰, B¹, ...} with B = 2^radix_bits, costing at most B-1 key-switch
    compositions per basis element below the top one. Basis elements above the
    largest target step are dropped: the greedy decomposition never reaches for
    them, so they would only cost upload, and one at or above the slot count
    has no rotation step to derive from.
    """
    max_step = max(target_steps, default=0)
    if not max_step:
        return set()
    step_bits = max_step.bit_length()
    count = min(max_master_keys, step_bits)
    radix_bits = math.ceil(step_bits / count)
    return {step for step in (1 << (radix_bits * i) for i in range(count)) if step <= max_step}


def _generate_homing_key(
    params: CKKSParams,
    *,
    secret: Poly,
    aux_secret: Poly,
    homing_primes: tuple[int, ...],
    rng: ChaChaRng,
) -> SwitchKey:
    """Base-ring switch key from the auxiliary secret s₁ to the scheme secret s.

    Its decomposition basis is the full key modulus P ∪ Q (the modulus homed
    ciphertexts live at), masked by the fresh homing primes P^(h). Digits are
    sized so each digit's product is at most P^(h), giving the
    paper's gadget rank dnum^(h) = ceil(log PQ / log P^(h)).
    """
    base_key_ring = params._highest_key_switch_params.ring_qt
    homing_ks_params = _homing_key_params(base_key_ring, homing_primes)
    # Stamped with its fresh noise (the scheme's own switch keys are unbounded)
    # so that homing enck' yields a bounded public key, and with it bounded
    # fresh encryptions on the server.
    noise_info = _fresh_encryption_noise(params.n, params.noise_dist)
    target_secret_qt = aux_secret.lift_to_ring(homing_ks_params.ring_qt)
    return SwitchKey.generate(
        homing_ks_params, target_secret_qt=target_secret_qt, encrypting_secret=secret, error_dist=params.noise_dist, noise_info=noise_info, rng=rng
    )


def _rescale_rounding_noise(params: CKKSParams) -> CKKSNoiseInfo:
    """Noise added by dropping a basis from a ciphertext under the scheme secret.

    Both halves round by at most 1/2 per coefficient, and the a-half's rounding
    multiplies s, so the error's coefficient bound is (1 + ‖s‖₁)/2, carried in
    the convention of :func:`_fresh_encryption_noise`.
    """
    bound = (1 + params.secret_key_dist.coeff_1_norm_bound()) / 2
    return CKKSNoiseInfo(noise_bound=2 * params.n * bound / math.pi, noise_heuristic_bound=math.sqrt(2 * params.n / 3) * bound, msg_bound=0)


def _homing_key_params(key_ring: RingSpec, homing_primes: tuple[int, ...]) -> KeySwitchParams:
    """Build the exact Method-II layout covered by the homing modulus."""
    homing_product = math.prod(homing_primes)
    digit_sizes = []
    digit_size = 0
    digit_product = 1
    for modulus in key_ring.moduli:
        if modulus > homing_product:
            raise InvalidParameterError("KG+ homing modulus must cover every key-switch modulus")
        next_product = digit_product * modulus
        if digit_size and next_product > homing_product:
            digit_sizes.append(digit_size)
            digit_size = 1
            digit_product = modulus
        else:
            digit_size += 1
            digit_product = next_product
    digit_sizes.append(digit_size)
    return KeySwitchParams.for_target(highest_ring=key_ring, target_ring=key_ring, p_primes=homing_primes, digit_sizes=tuple(digit_sizes))


def _balanced_crt_lift(residues: Collection[int], moduli: Collection[int]) -> int:
    """The centered representative of the CRT pattern ``residues`` over ``moduli``."""
    modulus = 1
    value = 0
    for residue, prime in zip(residues, moduli, strict=True):
        difference = (residue - value) % prime
        value += modulus * (difference * pow(modulus, -1, prime) % prime)
        modulus *= prime
    if value > modulus // 2:
        value -= modulus
    return value


@cache
def _rotation_steps_by_galois_elt(slot_count: int, modulus: int) -> Mapping[int, int]:
    """Map 5^step (mod ``modulus``) to ``step`` for every slot rotation step."""
    steps: dict[int, int] = {}
    current = 1
    for step in range(slot_count):
        steps.setdefault(current, step)
        current = current * 5 % modulus
    return MappingProxyType(steps)


def _rotation_step_from_galois_elt(galois_elt: int, *, slot_count: int, modulus: int) -> int:
    """Recover the rotation step with 5^step = galois_elt (mod ``modulus``)."""
    step = _rotation_steps_by_galois_elt(slot_count, modulus).get(galois_elt % modulus)
    if step is None:
        raise InvalidParameterError(f"no rotation step maps to galois_elt={galois_elt} (mod {modulus})")
    return step


def _project_extension_digits(
    digits: tuple[RLWECiphertext, ...],
    *,
    params: CKKSParams,
    target_ring: RingSpec,
) -> tuple[tuple[RLWECiphertext, ...], tuple[int, ...], RingSpec]:
    """Restrict full-layout extension digits to the digits a P + target-window key needs.

    Digits whose Q segment falls entirely outside the window are dropped. The
    rest are reduced to P + Q[0:stop], the prefix of Q ending with the window,
    not to P + window: the master and homing key switches locate their input
    as a contiguous run of P + Q, which a window that does not start at Q[0]
    is not. Returns those digits, the window's digit sizes and the base ring
    P + window that each homed digit is reduced to.
    """
    highest_moduli = params.highest_ring.moduli
    projection = _project_key_switch_digits(highest_moduli, params.ks_digit_sizes, target_ring.moduli)
    start = find_contiguous_subset(highest_moduli, target_ring.moduli)
    if projection is None or start is None:
        raise ParameterMismatchError("galois key ring is not a contiguous subrange of the highest Q basis")
    prefix_moduli = highest_moduli[: start + len(target_ring.moduli)]
    compute_ring = RingSpec(n=2 * params.n, moduli=params.ks_primes + prefix_moduli, dtype=params.dtype)
    window_ring = RingSpec(n=params.n, moduli=params.ks_primes + target_ring.moduli, dtype=params.dtype)
    projected = tuple(digits[digit_index].map_polys(lambda poly: poly.to_quotient_ring(compute_ring)) for digit_index, _ in projection)
    return projected, tuple(len(digit_range) for _, digit_range in projection), window_ring


def _apply_master_to_digits(
    digits: tuple[RLWECiphertext, ...],
    master_elt: int,
    master_key: SwitchKey,
) -> tuple[RLWECiphertext, ...]:
    """Advance a permute-first rotation key over R' by one master rotation.

    Each digit Enc_s̃(P·G_i·σ_g(s̃)) maps to Enc_s̃(P·G_i·σ_{g_m·g}(s̃)):
    apply σ_{g_m} to both halves, then key-switch the permuted `a` part back
    under s̃ with the master key.
    """
    advanced = []
    for digit in digits:
        rotated_a = digit.a.automorphism(master_elt)
        rotated_b = digit.b.automorphism(master_elt)
        switched = _keyswitch(rotated_a, master_key)
        # The rotated b-half still carries the digit's own noise (an
        # automorphism only permutes it), on top of the key switch's.
        advanced.append(replace(switched, b=switched.b + rotated_b, noise_info=switched.noise_info + digit.noise_info))
    return tuple(advanced)


def _derive_intermediate_rotation_digits(
    identity_digits: tuple[RLWECiphertext, ...],
    masters: tuple[tuple[int, int, SwitchKey], ...],
    target_step: int,
) -> tuple[RLWECiphertext, ...]:
    """Compose master rotations greedily to reach ``target_step`` from index 0."""
    digits = identity_digits
    remaining = target_step
    for step, master_elt, master_key in masters:
        count, remaining = divmod(remaining, step)
        for _ in range(count):
            digits = _apply_master_to_digits(digits, master_elt, master_key)
    if remaining:
        raise InvalidParameterError(f"KG+ master rotation keys cannot derive step {target_step}")
    return digits


def _split_extension_ciphertext(ct: RLWECiphertext) -> tuple[RLWECiphertext, RLWECiphertext]:
    """Split an extension ciphertext into its even and odd base-ring parts.

    For (b, a) = Enc_s̃(m) over R' with s̃ = s + Y·s₁ and m = m₀(Y²) + Y·m₁(Y²),
    splitting every polynomial into even/odd halves gives

        even: b₀ + a₀·s + X·a₁·s₁ = m₀ + e₀
        odd:  b₁ + a₁·s + a₀·s₁ = m₁ + e₁

    Homing key-switches X·a₁ for the even part or a₀ for the odd part from s₁
    to s, completing the base-ring ciphertext (paper Thm 3.1 / Cor 3.2).
    """
    a_even, a_odd = _split_extension_poly(ct.a)
    b_even, b_odd = _split_extension_poly(ct.b)
    return (RLWECiphertext(a=a_even, b=b_even, noise_info=ct.noise_info), RLWECiphertext(a=a_odd, b=b_odd, noise_info=ct.noise_info))


def _home_extension_ciphertext(base: RLWECiphertext, cross_term: Poly, homing_key: SwitchKey) -> RLWECiphertext:
    """Key-switch the cross term to complete a ciphertext under the base secret."""
    return base + _keyswitch(cross_term, homing_key)


def _extension_ring(ring: RingSpec) -> RingSpec:
    """The degree-doubled ring R' = Z[Y]/(Y^{2n}+1) over the same moduli.

    R embeds into R' via X -> Y². Ring construction enforces the KG+ prime
    condition (every modulus 1 mod 4n, so R' is NTT-capable); translate the
    failure into a KG+-flavored error.
    """
    try:
        return RingSpec(n=2 * ring.n, moduli=ring.moduli, dtype=ring.dtype)
    except ValueError as exc:
        raise InvalidParameterError(
            f"moduli are incompatible with the degree-{2 * ring.n} extension ring; KG+ requires every Q and key-switch prime "
            f"to be 1 mod {4 * ring.n}; generate params with CKKSParams.generate_prefix(..., log_extension_factor=1)"
        ) from exc


# TODO: Replace the host round trips in these helpers once Poly/IR exposes
# device-native coefficient interleave/split operations.
def _embed_pair(even: Poly, odd: Poly, extension_ring: RingSpec) -> Poly:
    """Embed two degree-n polynomials as even(Y²) + Y·odd(Y²) in the extension ring.

    Inputs may live at a different modulus basis than ``extension_ring``; they
    are aligned via a balanced (centered) lift, which is exact whenever the
    centered coefficients fit the target basis.
    """
    if even.ring != odd.ring:
        raise ParameterMismatchError("KG+ embedding inputs use different rings")
    if even.device is not None and odd.device is not None and even.device != odd.device:
        raise ParameterMismatchError("KG+ embedding inputs are on different devices")
    if extension_ring.n != 2 * even.ring.n:
        raise ParameterMismatchError("extension ring degree must be twice the base ring degree")

    base_ring = RingSpec(n=even.ring.n, moduli=extension_ring.moduli, dtype=extension_ring.dtype)
    coeffs = [0] * extension_ring.n
    coeffs[0::2] = even._to_ring_small_coeffs(base_ring).to_centered_list()
    coeffs[1::2] = odd._to_ring_small_coeffs(base_ring).to_centered_list()
    return Poly(coeffs, extension_ring, device=even.device or odd.device)


def _split_extension_poly(poly: Poly) -> tuple[Poly, Poly]:
    """Split an extension-ring polynomial into its even and odd degree-n halves.

    Inverse of :func:`_embed_pair` at the same modulus basis: for
    f(Y) = f₀(Y²) + Y·f₁(Y²), returns (f₀, f₁) over the degree-halved ring.
    """
    ring = poly.ring
    if ring.n < 2:
        raise ParameterMismatchError("cannot split a degree-1 ring")
    base_ring = RingSpec(n=ring.n // 2, moduli=ring.moduli, dtype=ring.dtype)
    coeffs = poly.to_centered_list()
    even = Poly(coeffs[0::2], base_ring, device=poly.device)
    odd = Poly(coeffs[1::2], base_ring, device=poly.device)
    return even, odd
