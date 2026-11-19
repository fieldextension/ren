from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import ceil, isfinite, sqrt
from operator import add
from types import MappingProxyType
from typing import Unpack

from ren.container import PolyContainer
from ren.device import Device
from ren.poly import Poly
from ren.polydist import MINIMUM_ENCRYPTION_NOISE_STD_DEV, DiscreteGaussian
from ren.random import ChaChaRng, rng_from_int_seed
from ren.ckks._rlwe import RLWECiphertext, _fresh_encryption_noise
from ren.ckks.ckks import Ciphertext
from ren.ckks.context import context
from ren.ckks.keys import (
    CKKSKeySet,
    GaloisKeyRequest,
    PublicKey,
    generate_secret,
    SwitchKey,
    _galois_elts_by_level,
    _galois_key_params,
    _switch_key_mask_terms,
)
from ren.ckks.metadata import Bound, CKKSNoiseInfo
from ren.ckks.params import CKKSParams, KeySwitchParams
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError


@dataclass(frozen=True, slots=True, kw_only=True)
class MultipartyParams:
    """Choose how the parties share the secret key.

    Omit ``threshold`` to use additive shares. All ``num_parties`` must then help
    decrypt. Set ``threshold`` to use Shamir shares, where any group of that many
    parties can decrypt together.
    """

    num_parties: int
    threshold: int | None = None

    def __post_init__(self) -> None:
        if self.num_parties <= 0:
            raise ValueError("number of parties must be positive")
        if self.threshold is not None and not 1 <= self.threshold <= self.num_parties:
            raise ValueError("threshold must be between 1 and num_parties inclusive")


@dataclass(frozen=True, slots=True)
class PartialDecryption(PolyContainer):
    """A masked decryption contribution with bounds on its flooding noise.

    ``noise_info`` bounds only the added noise in slot units before division
    by the ciphertext scale. Its zero message bound does not bound ``d`` as
    a plaintext.
    """

    d: Poly
    noise_info: CKKSNoiseInfo


def flooding_std_dev(ct_noise: Bound, params: CKKSParams, *, statistical_security: int | None, num_decryptions: int) -> float:
    """Calibrate per-share flooding from a coefficient-space noise estimate.

    Use the ciphertext's noise estimate after rerandomization.
    ``statistical_security=None`` selects the minimum flooding policy.
    """
    # TODO: Provide flooding calibration from CKKS noise metadata with explicit coefficient-space bounds.
    if statistical_security is not None and statistical_security < 0:
        raise ValueError("statistical security must be non-negative")
    if num_decryptions <= 0:
        raise ValueError("number of decryptions must be positive")
    if statistical_security is None:
        flooding_std = float(MINIMUM_ENCRYPTION_NOISE_STD_DEV)
    else:
        if ct_noise > (1 << 500):
            raise ValueError("Noise bound is too large for flooding")
        # TODO explain where this formula comes from (2024/681)
        flooding_std = sqrt(12 * num_decryptions * (1 << statistical_security)) * float(ct_noise)
    return max(flooding_std, params.noise_dist.coeff_inf_norm_bound())


def partial_decrypt(
    ct: Ciphertext,
    *,
    flooding_std_dev: float,
    quorum: Collection[int] | None = None,
    party: int | None = None,
    rng: ChaChaRng | None = None,
) -> PartialDecryption:
    """Compute a partial decryption using the active multiparty keyset.

    Rerandomize the ciphertext before distributing it. The caller binds each
    contribution to that ciphertext and an authenticated party.
    For Shamir shares, ``quorum`` contains the selected one-based party IDs,
    including ``party``. The secret is weighted before adding flooding noise.
    ``flooding_std_dev`` is in coefficient space and rounded up for sampling;
    no additional minimum is imposed.
    """
    ctx = context()
    if (multiparty := ctx.multiparty) is None:
        raise RuntimeError("partial_decrypt requires a multiparty keyset")
    if (secret := ctx.keyset.sk) is None:
        raise RuntimeError("partial_decrypt requires a secret key share")
    if not isfinite(flooding_std_dev) or flooding_std_dev <= 0:
        raise ValueError("flooding standard deviation must be finite and positive")
    device = ct.device or secret.device
    if rng is not None and device is not None and rng.device != device:
        raise ParameterMismatchError(f"decryption is on {device}, got RNG on {rng.device}")
    if multiparty.threshold is None:
        if quorum is not None or party is not None:
            raise ParameterMismatchError("additive shares do not accept threshold-decryption arguments")
    else:
        if quorum is None or party is None:
            raise ParameterMismatchError("quorum and party must be provided for Shamir shares")
        if any(not 1 <= member <= multiparty.num_parties for member in quorum):
            raise ParameterMismatchError("quorum party IDs must be from 1 to num_parties")
        if len(set(quorum)) != len(quorum):
            raise ParameterMismatchError("duplicate party in decryption quorum")
        if len(quorum) < multiparty.threshold:
            raise ParameterMismatchError(f"at least {multiparty.threshold} parties are required")
        if party not in quorum:
            raise ParameterMismatchError("party must be in the decryption quorum")

    try:
        s = secret.to_quotient_ring(ct.ring)
    except ValueError as exc:
        raise ParameterMismatchError("ciphertext ring is not a quotient of secret share ring") from exc
    if quorum is not None:
        assert party is not None
        # Lagrange coefficient: prod_{j != i} j / (j - i) modulo q.
        q = ct.ring.q
        num = den = 1
        for member in quorum:
            if member != party:
                num = num * member % q
                den = den * (member - party) % q
        s = s * (num * pow(den, -1, q) % q)

    noise_dist = DiscreteGaussian(ct.ring.n, Fraction(ceil(flooding_std_dev)))
    if rng is None:
        rng = Device[Device.canonicalize(device)].default_rng()
    d = ct.a * s + Poly.rand(noise_dist, ct.ring, rng=rng)
    return PartialDecryption(d=d, noise_info=_fresh_encryption_noise(ct.ring.n, noise_dist))


@dataclass(frozen=True, slots=True)
class DkgRound1State(PolyContainer):
    """Private material retained by a participant between rounds."""

    secret: Poly
    relinearization_secret: Poly
    transport_secret: Poly | None

    def __post_init__(self) -> None:
        if self.secret.ring != self.relinearization_secret.ring:
            raise ParameterMismatchError("round 1 state secret rings must match")


@dataclass(frozen=True, slots=True)
class DkgRound2Input(PolyContainer):
    """Relinearization terms and transport public keys for round 2."""

    relinearization_h0: tuple[Poly, ...]
    relinearization_h1: tuple[Poly, ...]
    transport_public_keys: tuple[PublicKey, ...]

    def __post_init__(self) -> None:
        h0, h1 = self.relinearization_h0, self.relinearization_h1
        if not h0 or len(h0) != len(h1):
            raise ParameterMismatchError("relinearization digit layout must be nonempty with matching lengths")
        if len({poly.ring for poly in PolyContainer.polys(h0, h1)}) != 1:
            raise ParameterMismatchError("relinearization digit rings must match")
        if len({pk.ct.ring for pk in self.transport_public_keys}) > 1:
            raise ParameterMismatchError("transport public key rings must match")


@dataclass(frozen=True, slots=True)
class DkgRound1Share(PolyContainer):
    public_key_body: Poly
    galois_keys: Mapping[int, tuple[Poly, ...]]
    round2_input: DkgRound2Input

    def __post_init__(self) -> None:
        for digits in self.galois_keys.values():
            if not digits:
                raise ParameterMismatchError("Galois key digits must be nonempty")
            if len({poly.ring for poly in digits}) != 1:
                raise ParameterMismatchError("Galois key digit rings must match")
        object.__setattr__(self, "galois_keys", MappingProxyType(dict(self.galois_keys)))

    def __reduce__(self):
        return type(self), (self.public_key_body, dict(self.galois_keys), self.round2_input)


@dataclass(frozen=True, slots=True)
class DkgRound2Share(PolyContainer):
    relinearization_h_prime: tuple[Poly, ...]
    encrypted_secret_shares: tuple[RLWECiphertext, ...]

    def __post_init__(self) -> None:
        if not self.relinearization_h_prime:
            raise ParameterMismatchError("relinearization digit layout must be nonempty")
        if len({poly.ring for poly in self.relinearization_h_prime}) != 1:
            raise ParameterMismatchError("relinearization digit rings must match")
        if len({ct.ring for ct in self.encrypted_secret_shares}) > 1:
            raise ParameterMismatchError("encrypted secret share rings must match")


@dataclass(frozen=True, slots=True)
class _GaloisKeyDkgMaterial:
    key_switch_params: KeySwitchParams
    masks: tuple[Poly, ...]


@dataclass(frozen=True, slots=True, init=False)
class Dkg:
    """Public parameters and common randomness for two-round DKG.

    The caller retains private state and supplies contributions in the same
    participant order in both rounds. Position i uses Shamir point i + 1.
    Identity, authentication, routing, and duplicate detection belong to the caller.

    The RLWE check applies to the secret each party samples. For bootstrapping,
    set ``sparse_encapsulation_weight=None`` and choose a modular reduction
    polynomial for the combined secret. For shared decryption, choose the
    flooding noise for the circuit and the number of decryptions.
    """

    params: CKKSParams
    multiparty: MultipartyParams
    seed: int
    galois_elts_by_level: Mapping[int, frozenset[int]]
    pk_mask: Poly
    rlk_masks: tuple[Poly, ...]
    galois_key_material: Mapping[int, _GaloisKeyDkgMaterial]

    def __init__(
        self,
        params: CKKSParams,
        common_random_seed: int,
        *,
        multiparty: MultipartyParams,
        device: str | None = None,
        **galois: Unpack[GaloisKeyRequest],
    ):
        self._initialize(params, common_random_seed, multiparty, _galois_elts_by_level(params, galois), device=device)

    def _initialize(
        self,
        params: CKKSParams,
        seed: int,
        multiparty: MultipartyParams,
        galois_elts_by_level: Mapping[int, frozenset[int]],
        *,
        device: str | None,
    ) -> None:
        if params.boot_params is not None and params.boot_params.sparse_encapsulation:
            # TODO: Generate both bootstrap switch keys collectively without reconstructing either secret.
            raise InvalidParameterError("DKG cannot generate sparse encapsulation switch keys; use params without sparse encapsulation")
        if multiparty.threshold is not None:
            if len(params.levels) < 2:
                raise InvalidParameterError("threshold DKG requires a transport level")
            if not params.ring_at(0).is_quotient_of(params.ring_at(1)):
                raise InvalidParameterError("threshold DKG level-0 ring must be a quotient of level 1")

        galois_key_params = _galois_key_params(params, galois_elts_by_level)
        rng = rng_from_int_seed(seed, device=device)
        # Public-key mask does not use key-switch P primes. The rest do.
        pk_mask = Poly.uniform(params.highest_ring, rng=rng.derive(["dkg", "public-key"]))
        highest_ks_params = params._highest_key_switch_params
        rlk_rng = rng.derive(["dkg", "relinearization"])
        rlk_masks = tuple(Poly.uniform(highest_ks_params.ring_qt, rng=rlk_rng) for _ in range(highest_ks_params.d))
        galois_key_material = {}
        for galois_elt, ks_params in galois_key_params.items():
            galois_rng = rng.derive(["dkg", "galois", str(galois_elt)])
            masks = tuple(Poly.uniform(ks_params.ring_qt, rng=galois_rng) for _ in range(ks_params.d))
            galois_key_material[galois_elt] = _GaloisKeyDkgMaterial(key_switch_params=ks_params, masks=masks)

        object.__setattr__(self, "params", params)
        object.__setattr__(self, "multiparty", multiparty)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "galois_elts_by_level", MappingProxyType(dict(galois_elts_by_level)))
        object.__setattr__(self, "pk_mask", pk_mask)
        object.__setattr__(self, "rlk_masks", rlk_masks)
        object.__setattr__(self, "galois_key_material", MappingProxyType(galois_key_material))

    @property
    def device(self) -> str:
        device = self.pk_mask.device
        assert device is not None
        return device

    def __reduce__(self):
        # Public masks are derived from the seed on the active device.
        return type(self)._unpickle, (self.params, self.seed, self.multiparty, dict(self.galois_elts_by_level))

    @classmethod
    def _unpickle(cls, params: CKKSParams, seed: int, multiparty: MultipartyParams, galois_elts_by_level: dict[int, frozenset[int]]) -> Dkg:
        dkg = cls.__new__(cls)
        dkg._initialize(params, seed, multiparty, galois_elts_by_level, device=None)
        return dkg

    def round1(self, secret: Poly | None = None, *, rng: ChaChaRng | None = None) -> tuple[DkgRound1State, DkgRound1Share]:
        """Return private round state and a public contribution.

        ``secret`` is sampled from ``params.secret_key_dist`` when omitted;
        a supplied secret must follow that distribution. Keep the private state
        for ``round2`` and ``finish``.
        """
        params = self.params
        if rng is None:
            rng = Device[self.device].default_rng()
        s = generate_secret(params, device=self.device, rng=rng).ntt() if secret is None else secret
        if s.ring != params.highest_ring:
            raise ParameterMismatchError("secret key share does not match the DKG ring")
        zero = Poly.zero(params.highest_ring)
        public_key_body = self._encrypt(zero, s, self.pk_mask, rng).b

        galois_keys = {}
        for galois_elt, material in self.galois_key_material.items():
            # Permute-last: encrypt s under sigma_{g^-1}(s). Rotation applies
            # sigma_g to the key-switch output; see CKKSKeySet.generate.
            g_inv = pow(galois_elt, -1, 2 * params.n)
            galois_keys[galois_elt] = self._switch_key_digits(s, s.automorphism(g_inv), material.masks, material.key_switch_params, rng)

        # h_{0,i} = -u_i*a_3 + s_i*w + e_{0,i}; h_{1,i} = s_i*a_3 + e_{1,i}.
        # a_3 is the common mask, w the decomposition basis, and u_i a fresh secret.
        u = generate_secret(params, rng=rng)
        h0 = self._switch_key_digits(s, u, self.rlk_masks, params._highest_key_switch_params, rng)
        zero = Poly.zero(params._highest_key_switch_params.ring_qt)
        h1 = tuple(self._encrypt(zero, -s, mask, rng).b for mask in self.rlk_masks)

        v = None
        transport_keys = ()
        if self.multiparty.threshold is not None:
            v = generate_secret(params, level=1, rng=rng)
            a = Poly.uniform(params.ring_at(1), rng=rng)
            transport_keys = (PublicKey(self._encrypt(Poly.zero(a.ring), v, a, rng)),)
        return DkgRound1State(s, u, v), DkgRound1Share(public_key_body, galois_keys, DkgRound2Input(h0, h1, transport_keys))

    def aggregate_round1(self, shares: Sequence[DkgRound1Share]) -> tuple[CKKSKeySet, DkgRound2Input]:
        """Aggregate round-1 contributions in participant order."""
        if len(shares) != self.multiparty.num_parties:
            raise ParameterMismatchError(f"Expected {self.multiparty.num_parties} round 1 shares, got {len(shares)}")
        for share in shares:
            self._validate_round1_share(share)

        params = self.params
        n = self.multiparty.num_parties
        sum_n_errors = CKKSNoiseInfo(
            noise_bound=params.noise_dist.coeff_inf_norm_bound() * n,
            noise_heuristic_bound=params.noise_dist.coeff_inf_norm_bound() * sqrt(n),
            msg_bound=0,
        )
        # TODO actually use heuristic noise
        pk_b = sum((share.public_key_body for share in shares), start=Poly.zero(params.highest_ring))
        pk = PublicKey(RLWECiphertext(a=self.pk_mask, b=pk_b, noise_info=sum_n_errors)).map_polys(Poly.ntt)

        galois_keys = {}
        for galois_elt, material in self.galois_key_material.items():
            # TODO correct the noise tracking
            galois_keys[galois_elt] = self._aggregate_switch_key(
                material.masks, [share.galois_keys[galois_elt] for share in shares], material.key_switch_params
            )

        zero = Poly.zero(params._highest_key_switch_params.ring_qt)
        h0 = tuple(sum((share.round2_input.relinearization_h0[i] for share in shares), start=zero) for i in range(len(self.rlk_masks)))
        h1 = tuple(sum((share.round2_input.relinearization_h1[i] for share in shares), start=zero) for i in range(len(self.rlk_masks)))
        transport_keys = tuple(pk for share in shares for pk in share.round2_input.transport_public_keys)
        keys = CKKSKeySet(params=params, multiparty=self.multiparty, pk=pk, galois_keys=galois_keys)
        return keys, DkgRound2Input(h0, h1, transport_keys)

    def round2(self, state: DkgRound1State, round2_input: DkgRound2Input, *, rng: ChaChaRng | None = None) -> DkgRound2Share:
        self._validate_state(state)
        self._validate_round2_input(round2_input, self.multiparty.num_parties)
        if rng is None:
            rng = Device[self.device].default_rng()
        s = state.secret
        u = state.relinearization_secret
        # h'_{0,i} = s_i*h_0 + e_{2,i}; h'_{1,i} = (u_i - s_i)*h_1 + e_{3,i}.
        # Only their sum h'_i is needed. Encryption negates the secret argument.
        zero = Poly.zero(self.params._highest_key_switch_params.ring_qt)
        h0_prime = tuple(self._encrypt(zero, -s, mask, rng).b for mask in round2_input.relinearization_h0)
        h1_prime = tuple(self._encrypt(zero, -(u - s), mask, rng).b for mask in round2_input.relinearization_h1)
        h_prime = tuple(a + b for a, b in zip(h0_prime, h1_prime, strict=True))

        encrypted_shares = []
        threshold = self.multiparty.threshold
        if threshold is not None:
            # Shamir secret sharing is over Z_{q_0}; position i uses point i + 1.
            ring = self.params.ring_at(0)
            coeffs = [s.to_quotient_ring(ring)] + [Poly.uniform(ring, rng=rng) for _ in range(threshold - 1)]
            scaling_factor = self.params.ring_at(1).q // ring.q
            for point, pk in enumerate(round2_input.transport_public_keys, start=1):
                share = coeffs[-1]
                for coeff in reversed(coeffs[:-1]):
                    share = share * point + coeff
                # Multiplication by the modulus ratio makes the choice of lift immaterial.
                scaled_share = share.lift_to_ring(pk.ct.ring) * scaling_factor
                encrypted_shares.append(pk.encrypt(scaled_share, self.params, noise_info=CKKSNoiseInfo.zero(), rng=rng))
        return DkgRound2Share(h_prime, tuple(encrypted_shares))

    def aggregate_round2(self, round2_input: DkgRound2Input, shares: Sequence[DkgRound2Share]) -> tuple[SwitchKey, tuple[RLWECiphertext, ...]]:
        """Return the relinearization key and encrypted shares in participant order."""
        if len(shares) != self.multiparty.num_parties:
            raise ParameterMismatchError(f"Expected {self.multiparty.num_parties} round 2 shares, got {len(shares)}")
        self._validate_round2_input(round2_input, self.multiparty.num_parties)
        params = self.params
        ks_params = params._highest_key_switch_params
        expected = self.multiparty.num_parties if self.multiparty.threshold is not None else 0
        for share in shares:
            ks_params.validate_digits(share.relinearization_h_prime)
            if len(share.encrypted_secret_shares) != expected:
                raise ParameterMismatchError(f"Expected {expected} encrypted secret shares")
            if expected and share.encrypted_secret_shares[0].ring != params.ring_at(1):
                raise ParameterMismatchError("encrypted secret shares do not match the DKG transport ring")

        # h_0 ~= -u*a_3 + s*w, h_1 ~= s*a_3, h' ~= s*h_0 + (u-s)*h_1.
        # Thus s*h_1 + h' ~= s*h_0 + u*h_1 ~= s^2*w: (h_1, h') encrypts s^2.
        relin_key = self._aggregate_switch_key(round2_input.relinearization_h1, [share.relinearization_h_prime for share in shares], ks_params)
        by_recipient = zip(*(share.encrypted_secret_shares for share in shares), strict=True)
        encrypted_shares = tuple(reduce(add, contributions).replace(noise_info=CKKSNoiseInfo.zero()) for contributions in by_recipient)
        return relin_key, encrypted_shares

    def finish(self, state: DkgRound1State, encrypted_secret_share: RLWECiphertext | None = None) -> CKKSKeySet:
        """Return the participant's keyset without public or evaluation keys.

        For threshold DKG, pass this participant's encrypted share from
        ``aggregate_round2``.
        """
        self._validate_state(state)
        if self.multiparty.threshold is None:
            if encrypted_secret_share is not None:
                raise InvalidParameterError("n-of-n DKG does not accept an encrypted secret share")
            secret = state.secret
        else:
            if encrypted_secret_share is None:
                raise InvalidParameterError("threshold DKG requires an encrypted secret share")
            ct = encrypted_secret_share
            if ct.ring != self.params.ring_at(1):
                raise ParameterMismatchError("encrypted secret share does not match the DKG ring")
            assert state.transport_secret is not None
            secret = ct._decrypt(state.transport_secret).rescale(self.params.ring_at(0).moduli)

        return CKKSKeySet(params=self.params, multiparty=self.multiparty, sk=secret)

    def _encrypt(self, m: Poly, s: Poly, mask: Poly, rng: ChaChaRng) -> RLWECiphertext:
        """Encrypt with DKG's error distribution and explicit shared mask."""
        return RLWECiphertext._encrypt(m, secret=s, mask=mask, error_dist=self.params.noise_dist, output_noise_info=CKKSNoiseInfo.zero(), rng=rng)

    def _switch_key_digits(
        self, target_secret: Poly, encrypting_secret: Poly, masks: tuple[Poly, ...], ks_params: KeySwitchParams, rng: ChaChaRng
    ) -> tuple[Poly, ...]:
        """Encrypt target_secret*P on each D segment under encrypting_secret.

        Each body is -a_i*encrypting_secret + e_i + target_secret*P on segment i.
        Used for both relinearization and Galois keys.
        """
        target_qt = target_secret._to_ring_small_coeffs(ks_params.ring_qt)
        mask_terms = _switch_key_mask_terms(target_qt, ks_params)
        return tuple(self._encrypt(term, encrypting_secret, mask, rng).b for mask, term in zip(masks, mask_terms, strict=True))

    def _aggregate_switch_key(self, masks: tuple[Poly, ...], shares: Sequence[tuple[Poly, ...]], ks_params: KeySwitchParams) -> SwitchKey:
        """Sum digit bodies and pair them with the shared masks in NTT form."""
        bodies = (reduce(add, column) for column in zip(*shares, strict=True))
        digits = tuple(RLWECiphertext(a=a, b=b, noise_info=CKKSNoiseInfo.zero()) for a, b in zip(masks, bodies, strict=True))
        return SwitchKey(digits, ks_params.digit_sizes).map_polys(Poly.ntt)

    def _validate_state(self, state: DkgRound1State) -> None:
        if state.secret.ring != self.params.highest_ring:
            raise ParameterMismatchError("round 1 state does not match the DKG ring")
        if self.multiparty.threshold is not None:
            if state.transport_secret is None:
                raise InvalidParameterError("threshold DKG state has no transport secret key")
            if state.transport_secret.ring != self.params.ring_at(1):
                raise ParameterMismatchError("transport secret does not match the DKG transport ring")
        elif state.transport_secret is not None:
            raise InvalidParameterError("n-of-n DKG state does not use a transport secret key")

    def _validate_round1_share(self, share: DkgRound1Share) -> None:
        self._validate_round2_input(share.round2_input, 1)
        if share.public_key_body.ring != self.params.highest_ring:
            raise ParameterMismatchError("public key share does not match the DKG ring")
        if share.galois_keys.keys() != self.galois_key_material.keys():
            raise ParameterMismatchError("Galois key shares do not match the DKG key layout")
        for elt, digits in share.galois_keys.items():
            self.galois_key_material[elt].key_switch_params.validate_digits(digits)

    def _validate_round2_input(self, value: DkgRound2Input, num_contributions: int) -> None:
        self.params._highest_key_switch_params.validate_digits(value.relinearization_h0)
        expected = num_contributions if self.multiparty.threshold is not None else 0
        if len(value.transport_public_keys) != expected:
            raise ParameterMismatchError(f"Expected {expected} transport public keys")
        if expected and value.transport_public_keys[0].ct.ring != self.params.ring_at(1):
            raise ParameterMismatchError("transport public keys do not match the DKG transport ring")
