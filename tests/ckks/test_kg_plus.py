import math
import pickle
from dataclasses import replace
from fractions import Fraction
from functools import cache

import pytest

from ren.device import Device
from ren.context import Context
from ren.dtype import dtypes
from ren.poly import Poly
from ren.polydist import DiscreteGaussian, SparseTernary, Ternary, UniformBounded
from ren.random import rng_from_int_seed
from ren.ring import Domain, RingSpec
from ren.ckks import (
    BootstrappingParams,
    CKKSKeySet,
    CKKSLevel,
    CKKSParams,
    TransmissionKeys,
    decode,
    decrypt,
    encode,
    encrypt,
    generate_secret,
)
from ren.ckks.bootstrap import bootstrap
from ren.ckks import _security, kg_plus
from ren.ckks._security import max_log2_modulus
from ren.ckks._rlwe import _fresh_encryption_noise
from ren.ckks.ckks import _keyswitch
from ren.ckks.keys import SwitchKey, _galois_elts_by_level, _galois_key_params
from ren.ckks.metadata import CKKSNoiseInfo
from ren.ckks.kg_plus import (
    _apply_master_to_digits,
    _embed_pair,
    _extension_ring,
    _homing_key_params,
    _homing_prime_layout,
    _master_rotation_steps,
    _select_kg_plus_aux_primes,
    _split_extension_poly,
)
from ren.ckks.utils import steps_to_galois_elt
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError


def _max_abs_coeff(poly: Poly) -> int:
    return max(abs(coeff) for coeff in poly.to_centered_list())


# Accumulated homing and master-composition noise in a derived key: measured at most 6 bits at
# n = 2^10 and 12 at n = 2^16 (8 for a top-level key).
_DERIVED_KEY_NOISE_BOUND = 1 << 20


def _assert_switch_key_relation(params: CKKSParams, key: SwitchKey, encrypting_secret: Poly, target: Poly, bound: int) -> None:
    """Each digit of a Method-II key over the highest layout satisfies b + a·s_enc = mask·target + e, |e| < bound."""
    ks_params = params._highest_key_switch_params
    s_enc = encrypting_secret.lift_to_ring(key.ring_qt)
    target_qt = target.lift_to_ring(key.ring_qt)
    for digit, digit_range in zip(key.digits, ks_params.digit_ranges, strict=True):
        residual = digit.b + digit.a * s_enc - target_qt.scale_by_primes(ks_params.mask_scalars(digit_range))
        assert _max_abs_coeff(residual) < bound


def _client_keys(params: CKKSParams, secret_key: Poly) -> CKKSKeySet:
    """A decryption-only key set for the client that holds ``secret_key``."""
    return CKKSKeySet(params, sk=secret_key)


@cache
def _kg_plus_test_params() -> CKKSParams:
    """The small and default non-bootstrap parameter recipes with extension-compatible primes.

    Generated directly (rather than substituting primes into the cached base
    params) so the scale schedule matches the moduli exactly.
    """
    if Device.is_cuda(Device.DEFAULT):
        return CKKSParams.generate_prefix(
            log_n=16,
            depth=10,
            log_precision=50,
            log_range=10,
            dtype=dtypes.uint64,
            p_prime_count=3,
            security=None,
            log_extension_factor=1,
        )
    return CKKSParams.generate_prefix(
        log_n=10,
        depth=5,
        log_precision=50,
        log_range=10,
        dtype=dtypes.uint32,
        p_prime_count=4,
        security=None,
        log_extension_factor=1,
    )


@cache
def _kg_plus_incomparable_level_params() -> CKKSParams:
    """Three levels whose windows are not nested: level 0 and level 1 overlap
    the highest basis at incomparable offsets."""
    base = _kg_plus_test_params()
    highest_moduli = base.highest_ring.moduli
    return replace(
        base,
        levels=(
            CKKSLevel(moduli=highest_moduli[:2], scale=1 << 40),
            CKKSLevel(moduli=highest_moduli[3:5], scale=1 << 40),
            CKKSLevel(moduli=highest_moduli, scale=1 << 40),
        ),
        log_range=0,
    )


@cache
def _kg_plus_crossing_level_params() -> CKKSParams:
    """Three levels whose windows cross: Q[0:3] and Q[2:5] overlap without either containing the other."""
    base = _kg_plus_test_params()
    highest_moduli = base.highest_ring.moduli
    return replace(
        base,
        levels=(
            CKKSLevel(moduli=highest_moduli[:3], scale=1 << 40),
            CKKSLevel(moduli=highest_moduli[2:5], scale=1 << 40),
            CKKSLevel(moduli=highest_moduli, scale=1 << 40),
        ),
        log_range=0,
    )


@cache
def _uint64_n16_params(depth: int) -> CKKSParams:
    """Leave 100 bits for homing at depth 17 and 24 bits at depth 18."""
    return CKKSParams.generate_prefix(
        log_n=16,
        depth=depth,
        log_precision=76,
        log_range=32,
        dtype=dtypes.uint64,
        p_prime_count=4,
        p_prime_bit_length=61,
        log_extension_factor=1,
    )


@cache
def _uint64_n14_params() -> CKKSParams:
    return CKKSParams.generate_prefix(
        log_n=14, depth=4, log_precision=35, log_range=10, dtype=dtypes.uint64, p_prime_count=3, security=128, log_extension_factor=1
    )


@cache
def _tight_headroom_params() -> CKKSParams:
    """128-bit params at n = 2^15 whose degree-n headroom (61 bits) is one bit above the widest key prime."""
    return CKKSParams.generate_prefix(
        log_n=15, depth=8, log_precision=55, log_range=9, dtype=dtypes.uint64, p_prime_count=5, security=128, log_extension_factor=1
    )


@cache
def _narrow_uint32_headroom_params() -> CKKSParams:
    """128-bit uint32 params at n = 2^14 with 46 bits of headroom: two primes would fall under the draw floor."""
    return CKKSParams.generate_prefix(
        log_n=14, depth=9, log_precision=25, log_range=12, dtype=dtypes.uint32, p_prime_count=4, security=128, log_extension_factor=1
    )


@cache
def _bootstrap_kg_plus_params() -> CKKSParams:
    """A bootstrap-capable recipe at n = 2^16 with extension-compatible primes."""
    return CKKSParams.generate_prefix(
        log_n=16,
        depth=6,
        log_precision=90,
        log_range=10,
        dtype=dtypes.uint32,
        p_prime_count=4,
        security=None,
        boot_params=BootstrappingParams(
            cts_depth=3, stc_depth=3, exp_degree=63, squaring_count=3, failure_probability=2**-64, precision_gains=(8, 10, 10, 10, 10)
        ),
        log_extension_factor=1,
    )


def _headroom(params: CKKSParams, key_moduli: tuple[int, ...], *, n: int) -> int:
    limit = max_log2_modulus(Ternary(n=n), replace(params.noise_dist, n=n), 128)
    return limit - (math.prod(key_moduli) - 1).bit_length()


def _assert_maximal_homing_packing(key_ring: RingSpec, homing_primes: tuple[int, ...]) -> None:
    """Every homing digit fits under P^(h), and none could also take the first prime of the next one."""
    homing_product = math.prod(homing_primes)
    digit_sizes = _homing_key_params(key_ring, homing_primes).digit_sizes
    assert sum(digit_sizes) == len(key_ring.moduli)
    start = 0
    for index, size in enumerate(digit_sizes):
        digit_product = math.prod(key_ring.moduli[start : start + size])
        assert digit_product <= homing_product
        if index + 1 < len(digit_sizes):
            assert digit_product * key_ring.moduli[start + size] > homing_product
        start += size


def test_aux_prime_selection_packs_two_q_primes_per_homing_digit():
    """The homing modulus must cover a 61-bit key prime. At this depth it
    also fits two 38-bit Q primes per digit."""
    params = _uint64_n16_params(depth=17)
    key_ring = params._highest_key_switch_params.ring_qt
    key_moduli = key_ring.moduli
    aux = _select_kg_plus_aux_primes(params, need_master=True, security=128)

    fresh = aux.transmission + aux.homing + aux.master
    assert len(set(fresh)) == len(fresh)
    assert not set(fresh) & set(key_moduli)
    assert all(prime % (4 * params.n) == 1 for prime in fresh)

    assert math.prod(aux.homing) > max(key_moduli)
    _assert_maximal_homing_packing(key_ring, aux.homing)
    homing_ks_params = _homing_key_params(key_ring, aux.homing)
    assert max(homing_ks_params.digit_sizes) == 2  # two 38-bit Q primes per digit under a 100-bit P^(h)
    assert 0 <= _headroom(params, key_moduli + aux.homing, n=params.n) < max(key_moduli).bit_length()
    assert _headroom(params, key_moduli + aux.transmission, n=2 * params.n) >= 0
    assert _headroom(params, key_moduli + aux.master, n=2 * params.n) >= 0


@pytest.mark.parametrize(
    "field,dist,message",
    [
        ("secret_key_dist", SparseTernary(n=65536, h=192), "interleaved"),
        ("noise_dist", DiscreteGaussian(n=65536, stdev=Fraction(1)), "Gaussian"),
    ],
)
def test_checked_kg_plus_rejects_unassessed_params(field, dist, message):
    params = replace(_uint64_n16_params(depth=4), **{field: dist})
    with pytest.raises(InvalidParameterError, match=message):
        TransmissionKeys.generate(params, secret=generate_secret(params))


def test_aux_prime_selection_rejects_insufficient_homing_headroom():
    params = _uint64_n16_params(depth=18)
    with pytest.raises(InvalidParameterError, match="headroom"):
        _select_kg_plus_aux_primes(params, need_master=False, security=128)


@pytest.mark.parametrize(
    ("dtype", "widest_key_bits", "headroom_bits", "expected"),
    [
        (dtypes.uint64, 60, 60, (1, 60)),
        (dtypes.uint64, 60, 61, (1, 61)),
        (dtypes.uint64, 60, 122, (2, 61)),
        (dtypes.uint64, 60, 123, (3, 41)),
        (dtypes.uint64, 61, 100, (2, 50)),
        (dtypes.uint64, 61, 124, (2, 62)),
        (dtypes.uint32, 30, 30, (1, 30)),
        (dtypes.uint32, 30, 46, (1, 31)),
        (dtypes.uint32, 30, 48, (2, 24)),
        (dtypes.uint32, 30, 62, (2, 31)),
    ],
)
def test_homing_prime_layout_spends_the_headroom(dtype, widest_key_bits, headroom_bits, expected):
    """Only the widths matter here. Synthetic moduli let us vary the widest key prime and headroom independently."""
    params = _narrow_uint32_headroom_params() if dtype is dtypes.uint32 else _uint64_n14_params()
    key_moduli = (1 << (widest_key_bits - 1),) * 20
    assert _homing_prime_layout(params, key_moduli, headroom_bits=headroom_bits) == expected


@pytest.mark.parametrize(
    ("make_params", "headroom", "homing_bits"),
    [(_tight_headroom_params, 61, 61), (_narrow_uint32_headroom_params, 46, 31)],
    ids=["uint64_61_bits", "uint32_46_bits"],
)
def test_homing_layout_covers_the_widest_key_prime_with_one_wider_prime(make_params, headroom, homing_bits):
    """A homing prime one bit wider than the largest key prime always covers it.
    Two 30-bit primes can have a product smaller than a 60-bit key prime."""
    params = make_params()
    key_ring = params._highest_key_switch_params.ring_qt
    assert _headroom(params, key_ring.moduli, n=params.n) == headroom
    assert max(key_ring.moduli).bit_length() == homing_bits - 1
    assert _homing_prime_layout(params, key_ring.moduli, headroom_bits=headroom) == (1, homing_bits)

    aux = _select_kg_plus_aux_primes(params, need_master=False, security=128)
    assert [prime.bit_length() for prime in aux.homing] == [homing_bits]
    assert math.prod(aux.homing) > max(key_ring.moduli)
    assert _headroom(params, key_ring.moduli + aux.homing, n=params.n) >= 0
    _assert_maximal_homing_packing(key_ring, aux.homing)


@pytest.mark.parametrize(
    "make_params",
    [
        pytest.param(_tight_headroom_params, marks=pytest.mark.requires_cuda, id="uint64_61_bits"),
        pytest.param(_narrow_uint32_headroom_params, id="uint32_46_bits"),
    ],
)
def test_derivation_with_a_homing_prime_wider_than_the_key_primes(make_params):
    """The one-wider homing prime runs through keygen, homing and a rotation on the backend: the derived
    rotation key satisfies the key relation, and the rotation lands within 6 bits of direct keygen's
    (measured 4; the uint32 profile is a 7-bit one even under direct keys, so no absolute tolerance)."""
    params = make_params()
    secret_key = generate_secret(params)
    direct_keys = CKKSKeySet.generate(params, secret=secret_key, rotations=[3])
    transmission_keys = TransmissionKeys.generate(params, secret=secret_key, rotations=[3])
    assert [prime.bit_length() for prime in transmission_keys.homing_key.p_primes] == [
        max(params.highest_ring.moduli + params.ks_primes).bit_length() + 1
    ]
    server_keys = transmission_keys.derive_keyset()

    (galois_elt,) = direct_keys.galois_keys
    s = secret_key
    _assert_switch_key_relation(
        params, server_keys.galois_keys[galois_elt], s.automorphism(pow(galois_elt, -1, 2 * params.n)), s, _DERIVED_KEY_NOISE_BOUND
    )

    values = [complex((index % 7) / 8, (index % 5) / 8) for index in range(params.slot_count)]
    expected = values[3:] + values[:3]
    errors = []
    for keys in (replace(direct_keys, sk=None), server_keys):
        with keys.context():
            rotated = encrypt(encode(values, msg_bound=1)).rotate(3)
        with _client_keys(params, secret_key).context():
            errors.append(max(abs(got - want) for got, want in zip(decode(decrypt(rotated)), expected, strict=True)))
    direct_error, derived_error = errors
    assert derived_error <= 64 * direct_error, f"derived rotation error {derived_error:.3e} vs direct {direct_error:.3e}"


def test_aux_prime_selection_validates_the_homing_ring_budget(monkeypatch):
    """P^(h) ∪ P ∪ Q is checked against the degree-n budget outright, not only through the layout's arithmetic."""
    params = _uint64_n16_params(depth=17)
    moduli = params._highest_key_switch_params.ring_qt.moduli
    count, bits = _homing_prime_layout(params, moduli, headroom_bits=_headroom(params, moduli, n=params.n))
    monkeypatch.setattr(kg_plus, "_homing_prime_layout", lambda *args, **kwargs: (count + 2, bits))
    with pytest.raises(InvalidParameterError, match=r"KG\+ homing keys need \d+ more bits .* degree-65536 ring"):
        _select_kg_plus_aux_primes(params, need_master=False, security=128)


def test_aux_prime_selection_names_the_missing_extension_ring_budget():
    """log_n = 17 validates at 128 bits, but its degree-2n extension ring has no tabulated budget."""
    params = CKKSParams.generate_prefix(log_n=17, depth=10, log_precision=50, log_range=10, dtype=dtypes.uint64, security=128, log_extension_factor=1)
    with pytest.raises(InvalidParameterError, match=r"KG\+ transmission keys at degree 262144: no 128-bit security estimate"):
        _select_kg_plus_aux_primes(params, need_master=True, security=128)


def test_extension_embed_split_roundtrip():
    params = _kg_plus_test_params()
    ring = params.highest_ring
    even = Poly.uniform(ring)
    odd = Poly.uniform(ring)

    split_even, split_odd = _split_extension_poly(_embed_pair(even, odd, _extension_ring(ring)))

    assert split_even.equals(even)
    assert split_odd.equals(odd)


def test_extension_embedding_is_multiplicative():
    # X -> Y² is a ring homomorphism: products of embedded polynomials track base
    # products, and Y·f(Y²) · Y·g(Y²) = (X·f·g)(Y²) — the parity algebra that
    # server-side homing relies on.
    params = _kg_plus_test_params()
    ring = params.highest_ring
    ext_ring = _extension_ring(ring)
    zero = Poly.zero(ring)
    f = Poly.uniform(ring)
    g = Poly.uniform(ring)

    even, odd = _split_extension_poly(_embed_pair(f, zero, ext_ring) * _embed_pair(g, zero, ext_ring))
    assert even.equals(f * g)
    assert odd.equals(zero)

    even, odd = _split_extension_poly(_embed_pair(zero, f, ext_ring) * _embed_pair(zero, g, ext_ring))
    assert even.equals((f * g).rotate(1))
    assert odd.equals(zero)

    even, odd = _split_extension_poly(_embed_pair(f, zero, ext_ring) * _embed_pair(zero, g, ext_ring))
    assert even.equals(zero)
    assert odd.equals(f * g)


@pytest.fixture(scope="module")
def direct_keys() -> CKKSKeySet:
    return CKKSKeySet.generate(_kg_plus_test_params(), rotations=[3], conjugation=True)


@pytest.fixture(scope="module")
def transmission_keys(direct_keys: CKKSKeySet) -> TransmissionKeys:
    assert direct_keys.sk is not None
    return TransmissionKeys.generate(direct_keys.params, secret=direct_keys.sk, rotations=[3], conjugation=True, security=None)


@pytest.fixture(scope="module")
def server_keys(transmission_keys: TransmissionKeys) -> CKKSKeySet:
    return transmission_keys.derive_keyset()


def test_transmission_key_device_does_not_detect_a_default(monkeypatch, transmission_keys):
    expected = transmission_keys.device
    monkeypatch.setattr(type(Device), "_select_device", property(lambda self: pytest.fail("unexpected detection")))
    with Context(DEV=""):
        assert transmission_keys.device == expected


def test_generate_transmission_keys_matches_request(direct_keys, transmission_keys):
    params = direct_keys.params

    rotation_elt = steps_to_galois_elt(3, params.n)
    assert transmission_keys.device == direct_keys.device
    assert set(transmission_keys.galois_key_rings) == {rotation_elt, 2 * params.n - 1} == set(direct_keys.galois_keys)
    assert transmission_keys.galois_key_rings[rotation_elt] == params.ring_at(params.default_level)
    assert 1 <= len(transmission_keys.master_galois_keys) <= 3
    for master in transmission_keys.master_galois_keys.values():
        assert master.ring_qt.n == 2 * params.n


def test_transmission_keys_reject_wrong_extension_encryption_key_ring(transmission_keys):
    with pytest.raises(ParameterMismatchError, match="extension encryption key ring"):
        replace(transmission_keys, extension_encryption_key=transmission_keys.evaluation_transmission_key)


def test_transmission_keys_reject_missing_transmission_auxiliary_basis(transmission_keys):
    with pytest.raises(ParameterMismatchError, match="evaluation transmission key ring"):
        replace(transmission_keys, evaluation_transmission_key=transmission_keys.extension_encryption_key)


def test_transmission_keys_reject_galois_target_outside_highest_basis(transmission_keys):
    params = transmission_keys.params
    galois_elt = steps_to_galois_elt(3, params.n)
    foreign_ring = RingSpec(n=params.n, moduli=params.ks_primes, dtype=params.dtype)

    with pytest.raises(ParameterMismatchError, match="not a contiguous subrange"):
        replace(transmission_keys, galois_key_rings={galois_elt: foreign_ring})


def test_transmission_keys_reject_galois_elements_without_a_rotation_step(transmission_keys):
    """The server composes master rotations to derive rotation keys, so an element with no step is a client error."""
    params = transmission_keys.params
    with pytest.raises(InvalidParameterError, match="neither a slot rotation nor the conjugation"):
        replace(transmission_keys, galois_key_rings={3: params.highest_ring})
    master = next(iter(transmission_keys.master_galois_keys.values()))
    with pytest.raises(InvalidParameterError, match="not a slot rotation over the extension ring"):
        replace(transmission_keys, master_galois_keys={3: master})


def test_transmission_keys_reject_galois_windows_off_level_boundaries(transmission_keys):
    """A window narrower than keygen resolves derives a key no level can use; prefix levels all start at Q[0]."""
    params = transmission_keys.params
    galois_elt = steps_to_galois_elt(3, params.n)
    inside = RingSpec(n=params.n, moduli=params.highest_ring.moduli[1:3], dtype=params.dtype)
    with pytest.raises(ParameterMismatchError, match="does not span whole levels"):
        replace(transmission_keys, galois_key_rings={galois_elt: inside})


def test_transmission_keys_reject_galois_windows_between_crossing_levels():
    """Under crossing levels Q[0:3] and Q[2:5], keygen's window for both is their hull Q[0:5], while Q[2:3]
    starts and stops on level boundaries yet holds no level, so its derived key would serve none."""
    params = _kg_plus_crossing_level_params()
    moduli = params.highest_ring.moduli
    tx = TransmissionKeys.generate(params, secret=generate_secret(params), security=None, rotations={0: [3], 1: [3]})
    galois_elt = steps_to_galois_elt(3, params.n)
    assert tx.galois_key_rings[galois_elt].moduli == moduli[:5]
    between = RingSpec(n=params.n, moduli=moduli[2:3], dtype=params.dtype)
    with pytest.raises(ParameterMismatchError, match="does not span whole levels"):
        replace(tx, galois_key_rings={**tx.galois_key_rings, galois_elt: between})


def test_derived_public_key_carries_a_noise_bound(direct_keys, transmission_keys, server_keys):
    """The homing key's digits carry their fresh noise, so the derived pk and every fresh encryption on the
    server are bounded like direct keygen's; the switch keys stay unbounded as the scheme's own are. The bound
    covers the measured noise, which the two rescale roundings dominate."""
    params = direct_keys.params
    assert not any(digit.noise_info.is_unbounded for digit in transmission_keys.homing_key.digits)
    assert all(digit.noise_info.is_unbounded for digit in server_keys.rlk.digits)

    pk_noise = server_keys.pk.ct.noise_info
    assert not pk_noise.is_unbounded
    assert pk_noise.msg_bound == 0
    assert pk_noise.noise_bound >= direct_keys.pk.ct.noise_info.noise_bound
    assert pk_noise.noise_bound > _max_abs_coeff(server_keys.pk.ct.b + server_keys.pk.ct.a * direct_keys.sk)

    with server_keys.context(strict_noise_bounds=True):
        fresh = encrypt(encode([0.5] * 4, msg_bound=1))
    assert not fresh.noise_info.is_unbounded
    assert fresh.noise_info.msg_bound == pytest.approx(float(params.scale(params.default_level)))


def test_transmission_digits_account_for_mask_magnitude(transmission_keys):
    """Finite transmission noise exposes the mask scaling hidden by unbounded key metadata."""
    params = transmission_keys.params
    ks_params = params._highest_key_switch_params
    transmission = transmission_keys.evaluation_transmission_key
    removed_modulus = transmission.ring.q // ks_params.ring_qt.q
    bounded = replace(transmission, noise_info=CKKSNoiseInfo(noise_bound=removed_modulus, noise_heuristic_bound=0, msg_bound=0))
    digits = replace(transmission_keys, evaluation_transmission_key=bounded)._transmission_digit_ciphertexts()

    for digit, digit_range in zip(digits, ks_params.digit_ranges, strict=True):
        mask = Poly.const(1, ks_params.ring_qt, device=transmission_keys.device).scale_by_primes(ks_params.mask_scalars(digit_range))
        mask_magnitude = _max_abs_coeff(mask)
        assert float(digit.noise_info.noise_bound / mask_magnitude) == pytest.approx(transmission.ring.n)


def test_master_composition_carries_the_digit_noise(transmission_keys):
    """Advancing a digit by a master rotation adds the key switch's noise to the digit's own, which the
    rotated b-half keeps. Both are unbounded in the artifact today, so a dropped term would only show
    once either carried a bound."""
    params = transmission_keys.params
    fresh = _fresh_encryption_noise(2 * params.n, params.noise_dist)
    master_elt, master = next(iter(transmission_keys.master_galois_keys.items()))
    bounded_master = SwitchKey(digits=tuple(replace(digit, noise_info=fresh) for digit in master.digits), digit_sizes=master.digit_sizes)
    digit = replace(transmission_keys._identity_rotation_digits()[0], noise_info=fresh)

    (advanced,) = _apply_master_to_digits((digit,), master_elt, bounded_master)
    switched = _keyswitch(digit.a.automorphism(master_elt), bounded_master)
    assert not advanced.noise_info.is_unbounded
    assert advanced.noise_info.noise_bound == (switched.noise_info + digit.noise_info).noise_bound > switched.noise_info.noise_bound


def test_derived_keys_are_stored_as_ntt_values(direct_keys, server_keys):
    """Derived keys come back realized and in direct keygen's representation: a ``ren.jit``
    plan traced over lazy keys captures the derivation's lazy tail and every conversion of
    the keys it touches, and replays them on each call without realizing the keys."""
    assert all(poly.is_realized for poly in server_keys.polys())
    direct_keys.realize()
    assert all(poly.domain is Domain.NTT for poly in direct_keys.polys())
    assert all(poly.domain is Domain.NTT for poly in server_keys.polys())


def test_derived_keys_satisfy_direct_key_relations(direct_keys, server_keys):
    """Every derived switch key encrypts the same masked target as regular keygen.

    Each digit of a Method-II key satisfies b + a·s_enc = mask·target + e. The
    direct keys realize this with fresh noise only; the derived keys carry the
    accumulated homing/derivation noise, bounded far below the modulus.
    """
    params = direct_keys.params
    direct_bound = 1 << 7  # fresh encryption noise only

    s = direct_keys.sk
    s_squared = s * s
    _assert_switch_key_relation(params, direct_keys.rlk, s, s_squared, direct_bound)
    _assert_switch_key_relation(params, server_keys.rlk, s, s_squared, _DERIVED_KEY_NOISE_BOUND)

    assert set(server_keys.galois_keys) == set(direct_keys.galois_keys)
    for galois_elt, derived_key in server_keys.galois_keys.items():
        direct_key = direct_keys.get_galois_key(galois_elt)
        assert derived_key.ring_qt == direct_key.ring_qt
        # Permute-last: both encrypt the plain secret under σ_{g⁻¹}(s).
        encrypting_secret = s.automorphism(pow(galois_elt, -1, 2 * params.n))
        _assert_switch_key_relation(params, direct_key, encrypting_secret, s, direct_bound)
        _assert_switch_key_relation(params, derived_key, encrypting_secret, s, _DERIVED_KEY_NOISE_BOUND)


def test_derived_galois_keys_respect_requested_levels():
    params = _kg_plus_test_params()
    secret_key = generate_secret(params)
    tx = TransmissionKeys.generate(
        params,
        secret=secret_key,
        rotations={params.default_level: [3], 2: [5]},
        security=None,
    )

    elt3 = steps_to_galois_elt(3, params.n)
    elt5 = steps_to_galois_elt(5, params.n)
    assert tx.galois_key_rings[elt5] == params.ring_at(2)
    assert tx.galois_key_rings[elt3] == params.ring_at(params.default_level)

    server_keys = tx.derive_keyset()
    assert server_keys.get_galois_key(elt5).target_ring == params.ring_at(2)
    assert server_keys.get_galois_key(elt3).target_ring == params.ring_at(params.default_level)

    values = [1.0 + 0.5j, -0.5 + 0.25j, 0.25 - 1.0j, -1.0 - 0.25j, 0.5 + 1.0j]
    slot_values = values + [0j] * (params.slot_count - len(values))
    with server_keys.context():
        rotated_at_level_2 = encrypt(encode(values, msg_bound=2)).to_level(2).rotate(5)
        rotated_at_default_level = encrypt(encode(values, msg_bound=2)).rotate(3)
    with _client_keys(params, secret_key).context():
        assert decode(decrypt(rotated_at_level_2)) == pytest.approx(slot_values[5:] + slot_values[:5], abs=1e-3)
        assert decode(decrypt(rotated_at_default_level)) == pytest.approx(slot_values[3:] + slot_values[:3], abs=1e-3)


def test_derived_galois_key_covers_incomparable_requested_levels():
    params = _kg_plus_incomparable_level_params()
    secret_key = generate_secret(params)
    transmission_keys = TransmissionKeys.generate(
        params,
        secret=secret_key,
        rotations={0: [2], 1: [2]},
        security=None,
    )
    server_keys = transmission_keys.derive_keyset()
    values = [complex((index % 17) / 17, (index % 13) / 13) for index in range(params.slot_count)]

    with server_keys.context():
        rotated = [encrypt(encode(values, msg_bound=2, level=level)).rotate(2) for level in (0, 1)]

    with _client_keys(params, secret_key).context():
        for result in rotated:
            assert decode(decrypt(result)) == pytest.approx(values[2:] + values[:2], abs=2e-2)


@pytest.mark.parametrize("request_", [{"rotations": {1: [2]}}, {"conjugation": 1}], ids=["rotation", "conjugation"])
def test_derived_galois_keys_cover_shifted_level_windows(request_):
    """A window that does not start at Q[0] (level 1 here is Q[3:5]) derives like direct keygen serves it:
    the key switches run on the Q prefix ending with the window, then the digits drop below it."""
    params = _kg_plus_incomparable_level_params()
    secret_key = generate_secret(params)
    direct_keys = CKKSKeySet.generate(params, secret=secret_key, **request_)
    (galois_elt,) = direct_keys.galois_keys
    assert direct_keys.galois_keys[galois_elt].target_ring.moduli == params.highest_ring.moduli[3:5]

    transmission_keys = TransmissionKeys.generate(params, secret=secret_key, security=None, **request_)
    server_keys = transmission_keys.derive_keyset()
    derived = server_keys.galois_keys[galois_elt]
    assert derived.target_ring == direct_keys.galois_keys[galois_elt].target_ring
    assert derived.digit_sizes == direct_keys.galois_keys[galois_elt].digit_sizes

    values = [complex((index % 17) / 17, (index % 13) / 13) for index in range(params.slot_count)]
    with server_keys.context():
        ciphertext = encrypt(encode(values, msg_bound=2, level=1))
        result = ciphertext.rotate(2) if "rotations" in request_ else ciphertext.conjugate()
    expected = values[2:] + values[:2] if "rotations" in request_ else [value.conjugate() for value in values]
    with _client_keys(params, secret_key).context():
        assert decode(decrypt(result)) == pytest.approx(expected, abs=2e-2)


@pytest.mark.parametrize("depth", [12, 14], ids=["margin_requires_another_prime", "mixed_prime_widths"])
def test_master_modulus_is_sized_by_bits(depth):
    """P^(m) must cover P ∪ Q with 12 bits to spare for rotation-key noise and fit
    the degree-2n budget. At depth 12, rounding to whole primes alone leaves only 9 spare bits."""
    params = CKKSParams.generate_prefix(
        log_n=15, depth=depth, log_precision=40, log_range=10, dtype=dtypes.uint64, security=128, log_extension_factor=1
    )
    key_ring = params._highest_key_switch_params.ring_qt
    assert len(key_ring.moduli) == depth + 3
    aux = _select_kg_plus_aux_primes(params, need_master=True, security=128)
    assert len(aux.master) == math.ceil((key_ring.q.bit_length() + 12) / 60)
    assert math.prod(aux.master) >= key_ring.q << 12 > math.prod(aux.master[:-1])
    assert _headroom(params, aux.master + key_ring.moduli, n=2 * params.n) >= 0


def test_transmission_modulus_covers_balanced_masks_when_drawn_primes_run_short():
    params = CKKSParams.generate_prefix(
        log_n=16, depth=14, log_precision=85, log_range=10, dtype=dtypes.uint32, p_prime_count=4, security=128, log_extension_factor=1
    )
    key_ring = params._highest_key_switch_params.ring_qt
    aux = _select_kg_plus_aux_primes(params, need_master=True, security=128)
    assert len(aux.transmission) > math.ceil(key_ring.q.bit_length() / 30)
    assert math.prod(aux.transmission) >= key_ring.q // 2 > math.prod(aux.transmission[:-1])
    assert _headroom(params, aux.transmission + key_ring.moduli, n=2 * params.n) >= 0


def test_master_modulus_is_topped_up_when_the_drawn_primes_run_short():
    """uint32's 30-bit primes ≡ 1 mod 4n sit well below 2^30 at n = 2^16, so the bit-length count
    alone leaves P^(m) under P ∪ Q; the selection draws more until the 12-bit margin holds."""
    params = _bootstrap_kg_plus_params()
    key_ring = params._highest_key_switch_params.ring_qt
    aux = _select_kg_plus_aux_primes(params, need_master=True, security=None)
    assert len(aux.master) > math.ceil((key_ring.q.bit_length() + 12) / 30)
    assert math.prod(aux.master) >= key_ring.q << 12 > math.prod(aux.master[:-1])


def test_aux_prime_selection_names_the_extension_ring_budget(monkeypatch):
    """Lower only the degree-2n limit so the transmission key exceeds its budget."""
    params = _uint64_n16_params(17)
    key_bits = params._highest_key_switch_params.ring_qt.q.bit_length()
    monkeypatch.setitem(_security._MAX_LOG2_MODULUS, Ternary(n=2 * params.n), key_bits + 10)
    with pytest.raises(InvalidParameterError, match=r"KG\+ transmission keys need \d+ more bits .* degree-131072"):
        _select_kg_plus_aux_primes(params, need_master=True, security=128)


def test_aux_prime_selection_rejects_only_the_master_key_over_budget(monkeypatch):
    params = CKKSParams.generate_prefix(log_n=15, depth=12, log_precision=40, log_range=10, dtype=dtypes.uint64, security=128, log_extension_factor=1)
    # The transmission key needs 1,312 bits. The master key needs one more 60-bit prime.
    monkeypatch.setitem(_security._MAX_LOG2_MODULUS, Ternary(2 * params.n), 1312)
    _select_kg_plus_aux_primes(params, need_master=False, security=128)
    with pytest.raises(InvalidParameterError, match=r"KG\+ master keys need 60 more bits .* degree-65536 ring"):
        _select_kg_plus_aux_primes(params, need_master=True, security=128)


@cache
def _small_bootstrap_kg_plus_params() -> CKKSParams:
    """The bootstrap test suite's smallest recipe, with extension-compatible primes."""
    return CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=50,
        log_range=7,
        dtype=dtypes.uint32,
        boot_params=BootstrappingParams(cts_depth=2, stc_depth=2, exp_degree=16, squaring_count=0, failure_probability=2**-16),
        bootstrap_level_bit_length=60,
        p_prime_count=4,
        security=None,
        log_extension_factor=1,
    )


def test_generate_accepts_hoisted_bootstrap_rotations():
    """The full Galois request passes through: hoisted bootstrap rotations get direct keys, which under
    KG+ cost derivation work but no upload (the master basis is unchanged)."""
    params = _small_bootstrap_kg_plus_params()
    secret_key = generate_secret(params)
    plain = TransmissionKeys.generate(params, secret=secret_key, security=None)
    hoisted = TransmissionKeys.generate(params, secret=secret_key, security=None, hoist_bootstrap_rotations=True)
    expected = _galois_key_params(params, _galois_elts_by_level(params, {"hoist_bootstrap_rotations": True}))
    assert set(hoisted.galois_key_rings) == set(expected) > set(plain.galois_key_rings)
    assert hoisted.galois_key_rings == {galois_elt: layout.target_ring for galois_elt, layout in expected.items()}
    assert hoisted.master_galois_keys.keys() == plain.master_galois_keys.keys()


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_derived_keyset_bootstraps_with_hoisted_rotations():
    """Double-hoisted bootstrapping under derived keys matches direct keygen's."""
    params = _bootstrap_kg_plus_params()
    secret_key = generate_secret(params)
    direct_keys = CKKSKeySet.generate(params, secret=secret_key, hoist_bootstrap_rotations=True)
    server_keys = TransmissionKeys.generate(params, secret=secret_key, security=None, hoist_bootstrap_rotations=True).derive_keyset()
    assert set(server_keys.galois_keys) == set(direct_keys.galois_keys)

    values = [complex((index % 5) / 8, (index % 3) / 8) for index in range(params.slot_count)]
    errors = []
    for keys in (replace(direct_keys, sk=None), server_keys):
        with keys.context(strict_noise_bounds=False):
            result = bootstrap(encrypt(encode(values, msg_bound=1)).to_level(0), log_bound=0)
        with _client_keys(params, secret_key).context():
            decoded = decode(decrypt(result))
        assert decoded == pytest.approx(values, abs=1e-3)
        errors.append(max(abs(got - want) for got, want in zip(decoded, values, strict=True)))
    direct_error, derived_error = errors
    # The derived keys carry homing and master-composition noise on top of fresh keygen's: within 2 bits.
    assert derived_error <= 4 * direct_error, f"derived bootstrap error {derived_error:.3e} vs direct {direct_error:.3e}"


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_checked_dense_kg_plus_bootstrap():
    log_n = 16
    params = CKKSParams.default(
        log_n=log_n,
        depth=6,
        log_precision=45,
        log_extension_factor=1,
        secret_key_dist=Ternary(1 << log_n),
        boot_params=BootstrappingParams(cts_depth=3, stc_depth=3, exp_degree=191, squaring_count=5, failure_probability=2**-64, precision_gains=(8,)),
        security=128,
    )
    secret = generate_secret(params)
    evaluator = TransmissionKeys.generate(params, secret=secret, security=128, hoist_bootstrap_rotations=True).derive_keyset()
    values = [complex((i % 13 - 6) / 16, (i % 7 - 3) / 16) for i in range(params.slot_count)]
    with evaluator.context():
        ct = encrypt(encode(values, msg_bound=1))
        result = bootstrap((ct * ct).to_level(0), log_bound=0)
    with _client_keys(params, secret).context():
        assert result.level == 6
        assert decode(decrypt(result)) == pytest.approx([v * v for v in values], abs=1e-4)


def test_transmission_keys_pickle_round_trip(direct_keys, transmission_keys):
    params = direct_keys.params
    loaded = pickle.loads(pickle.dumps(transmission_keys))
    assert loaded.galois_key_rings == transmission_keys.galois_key_rings
    assert loaded.master_galois_keys.keys() == transmission_keys.master_galois_keys.keys()
    assert all(mine.equals(theirs) for mine, theirs in zip(loaded.polys(), transmission_keys.polys(), strict=True))

    values = [complex((index % 7) / 8, (index % 5) / 8) for index in range(params.slot_count)]
    with loaded.derive_keyset().context():
        rotated = encrypt(encode(values, msg_bound=1)).rotate(3)
    with _client_keys(params, direct_keys.sk).context():
        assert decode(decrypt(rotated)) == pytest.approx(values[3:] + values[:3], abs=1e-3)


def test_generate_rejects_extension_incompatible_params(test_params: CKKSParams):
    # Ordinary generated params select primes that are only 1 mod 2n; KG+ needs 1 mod 4n.
    assert any(prime % (4 * test_params.n) != 1 for prime in (*test_params.highest_ring.moduli, *test_params.ks_primes))
    secret_key = generate_secret(test_params)
    with pytest.raises(InvalidParameterError, match="extension ring"):
        TransmissionKeys.generate(test_params, secret=secret_key)


def test_generate_rejects_unsupported_noise_before_sampling():
    base = _kg_plus_test_params()
    params = replace(base, noise_dist=UniformBounded(n=base.n, bound=4))
    secret = Poly.const(1, params.highest_ring)
    rng = rng_from_int_seed(12)
    with pytest.raises(InvalidParameterError, match="requires a DiscreteGaussian"):
        TransmissionKeys.generate(params, secret=secret, security=None, rng=rng)
    assert rng.reserve() == 0


def test_generate_accepts_leveled_conjugation():
    params = _kg_plus_test_params()
    secret_key = generate_secret(params)
    transmission_keys = TransmissionKeys.generate(
        params,
        secret=secret_key,
        conjugation=2,
        security=None,
    )
    assert set(transmission_keys.galois_key_rings) == {2 * params.n - 1}
    assert transmission_keys.galois_key_rings[2 * params.n - 1] == params.ring_at(2)
    server_keys = transmission_keys.derive_keyset()
    values = [1.0 + 0.5j, -0.5 + 0.25j, 0.25 - 1.0j, -1.0 - 0.25j, 0.5 + 1.0j]

    with server_keys.context():
        conjugated = encrypt(encode(values, msg_bound=2)).to_level(2).conjugate()

    slot_values = values + [0j] * (params.slot_count - len(values))
    with _client_keys(params, secret_key).context():
        assert decode(decrypt(conjugated)) == pytest.approx([value.conjugate() for value in slot_values], abs=1e-3)


def test_master_basis_stays_within_the_requested_steps():
    # The greedy decomposition never uses a basis element above the largest step, and one
    # at or above the slot count has no rotation step to derive from.
    assert _master_rotation_steps({9000}, 8) == {1, 4, 16, 64, 256, 1024, 4096}
    assert _master_rotation_steps({1}, 3) == {1}
    assert _master_rotation_steps(set(), 3) == set()
    assert _master_rotation_steps({0}, 3) == set()


def test_derivation_with_a_master_basis_trimmed_to_the_request():
    params = _kg_plus_test_params()
    step = 300
    secret_key = generate_secret(params)
    tx = TransmissionKeys.generate(params, secret=secret_key, rotations=[step], max_master_keys=4, security=None)
    assert set(tx.master_galois_keys) == {steps_to_galois_elt(master, 2 * params.n) for master in (1, 8, 64)}

    values = [complex((i % 7) / 8, (i % 5) / 8) for i in range(params.slot_count)]
    with tx.derive_keyset().context():
        rotated = encrypt(encode(values, msg_bound=2)).rotate(step)
    with _client_keys(params, secret_key).context():
        assert decode(decrypt(rotated)) == pytest.approx(values[step:] + values[:step], abs=1e-3)


def test_generate_rejects_nonpositive_max_master_keys():
    params = _kg_plus_test_params()
    secret_key = generate_secret(params)
    with pytest.raises(InvalidParameterError, match="max_master_keys"):
        TransmissionKeys.generate(params, secret=secret_key, max_master_keys=0)


def test_kg_plus_server_evaluates_and_client_decrypts(direct_keys, server_keys):
    params = direct_keys.params
    client_keys = _client_keys(params, direct_keys.sk)

    assert server_keys.sk is None
    values = [1.0 + 0.25j, -0.75 + 0.5j, 0.5 - 0.125j, -0.25 - 0.5j]
    slot_values = values + [0j] * (params.slot_count - len(values))
    with server_keys.context():
        ciphertext = encrypt(encode(values, msg_bound=2))
        product = ciphertext * ciphertext
        rotated = ciphertext.rotate(3)
        conjugated = ciphertext.conjugate()
        with pytest.raises(RuntimeError, match="no secret key"):
            decrypt(product)

    with client_keys.context():
        assert decode(decrypt(product)) == pytest.approx([value * value for value in slot_values], abs=1e-3)
        assert decode(decrypt(rotated)) == pytest.approx(slot_values[3:] + slot_values[:3], abs=1e-3)
        assert decode(decrypt(conjugated)) == pytest.approx([value.conjugate() for value in slot_values], abs=1e-3)
