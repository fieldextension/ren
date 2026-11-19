import math
import operator
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from typing import Any, Callable, cast
import pytest

import numpy as np
from flint import acb, arb
import ren
from ren.bigfloat import BigFloat
from ren.dtype import dtypes
from ren.poly import Poly
from ren.random import ChaChaRng, set_global_rng_seed
from ren.ring import Domain, RingSpec
from ren.ckks import Ciphertext, CKKSKeySet, Plaintext, decode, decrypt, encode, encrypt, CKKSParams, keygen
from ren.ckks import encoding
from ren.ckks._rlwe import RLWECiphertext
from ren.ckks.metadata import Bound, CKKSNoiseInfo
from ren.ckks.params import CKKSLevel
from ren.ckks.encoding import flint_precision, _slots_to_coeffs_np
from ren.ckks.utils import CallCounter, is_prime_u64, select_moduli
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError
from tests.utils import random_complex_with_norm, random_plaintext, random_values_with_norm, random_ciphertext
from tests.ckks.params import high_precision_params, get_test_params, small_params


params = get_test_params()


@pytest.fixture(scope="module")
def keys(test_params: CKKSParams) -> CKKSKeySet:
    set_global_rng_seed(42)
    rotations = {2**i for i in range(test_params.logn - 1)}
    return keygen(test_params, rotations=rotations, conjugation=True).realize()


def _slots_to_coeffs_np_reference(slot_values: Sequence[complex], n: int) -> list[complex]:
    slots = n // 2
    m = 2 * n
    rot_group: list[int] = []
    value = 1
    for _ in range(slots):
        rot_group.append(value)
        value = (5 * value) % m

    permuted = [0j] * n
    for i, slot in enumerate(slot_values):
        index = (rot_group[i] - 1) // 2
        permuted[index] = slot
        permuted[n - 1 - index] = slot.conjugate()

    fft_result = np.fft.fft(permuted)
    return [fft_result[k] * np.exp(-1j * math.pi * k / n) / n for k in range(n)]


@pytest.mark.parametrize("slot_count", [0, 1, 3, 4])
def test_slots_to_coeffs_np_matches_reference(slot_count: int):
    n = 8
    values = [complex(i + 0.25, -0.5 * i) for i in range(slot_count)]
    prepared = encoding.slot_array(values, n // 2, dtype=np.complex128)
    assert list(_slots_to_coeffs_np(prepared)) == pytest.approx(_slots_to_coeffs_np_reference(values, n), abs=1e-12)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.complex64])
@pytest.mark.parametrize("prec", [53, 100])
def test_encoding_numpy_scalar_sequences(keys: CKKSKeySet, dtype, prec):
    values = np.arange(4, dtype=dtype)
    with keys.context(track_plaintext_values=True):
        expected = encode(values.tolist(), msg_bound=4, prec=prec)
        actual = encode(list(values), msg_bound=4, prec=prec)
        assert actual.m.equals(expected.m)
        assert actual._values is not None and expected._values is not None
        assert all(actual._values == expected._values)
        assert_equal(actual.noise_info, expected.noise_info)


def test_replace_polys_preserves_metadata_without_context(test_params: CKKSParams):
    with keygen(test_params).context(track_plaintext_values=True):
        pt = encode([0.125] * test_params.slot_count, msg_bound=1)
        ct = encrypt(pt)

    for source, count in ((pt, 1), (ct, 2)):
        result = source.replace(polys=map(Poly.ntt, source.polys()))
        assert type(result) is type(source)
        assert [poly.domain for poly in result.polys()] == [Domain.NTT] * count
        assert result.noise_info is source.noise_info
        assert result._values is source._values
        assert source._values is not None


def test_encode_decode_numpy_roundtrip(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        values = np.asarray(random_values_with_norm(keys.params.slot_count, norm=10), dtype=np.complex128)
        pt_from_list = encode(values.tolist(), msg_bound=10)
        pt_from_array = encode(values, msg_bound=10)

        assert pt_from_array.m.equals(pt_from_list.m)
        assert pt_from_array.values == pytest.approx(pt_from_list.values)
        assert_equal(pt_from_array.noise_info, pt_from_list.noise_info)
        assert decode(pt_from_array, len(values)) == pytest.approx(values, abs=1e-12)


def test_encoding_preserves_wide_signed_coefficients(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        values = np.full(keys.params.slot_count, -(2**20), dtype=np.complex128)
        pt = encode(values, msg_bound=2**20)

        assert list(pt.m)[0] == round(-(2**20) * float(keys.params.scale(pt.m.ring))) % pt.m.ring.q


@pytest.mark.parametrize("coefficient", [2**63, -(2**63), 2**64])
def test_encode_preserves_integer_boundaries(keys: CKKSKeySet, coefficient: int):
    with keys.context():
        value = coefficient / float(keys.params.scale(keys.params.default_level))
        pt = encode(np.full(keys.params.slot_count, value), msg_bound=abs(value))

        assert list(pt.m) == [coefficient % pt.ring.q] + [0] * (pt.ring.n - 1)


@pytest.fixture
def integer_boundary_params():
    n = 8
    moduli = select_moduli(n=n, num_primes=3, bit_length=30)
    return CKKSParams.from_levels(
        n=n,
        dtype=dtypes.uint32,
        levels=(CKKSLevel(moduli=moduli, scale=BigFloat(2) ** 63),),
        ks_primes=select_moduli(n=n, num_primes=1, bit_length=30, exclude=set(moduli)),
        ks_digit_sizes=(len(moduli),),
        log_range=0,
        security=None,
    )


@pytest.mark.parametrize("dense", [False, True], ids=["scalar", "vector"])
def test_binary64_encoding_tracks_original_input_and_bounds_loss(integer_boundary_params, dense):
    scale = BigFloat(2) ** 20
    params = replace(integer_boundary_params, levels=(replace(integer_boundary_params.levels[0], scale=scale),))
    value = 2**53 + 1
    raw = [value] * params.slot_count if dense else value
    with params.context(track_plaintext_values=True):
        pt = encode(raw, msg_bound=2**54)
        assert pt.noise_stats().max_abs == 1
        assert abs(pt.m.to_centered_list()[0] - value * int(scale)) <= pt.noise_info.noise_bound


@pytest.mark.parametrize("value", [np.nextafter(1.0, 0), 1.0, np.nextafter(1.0, math.inf)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_numpy_encoding_preserves_integer_boundaries(integer_boundary_params, value, sign):
    params = integer_boundary_params
    value = sign * float(value)
    numerator, denominator = value.as_integer_ratio()
    expected = numerator * 2**63 // denominator
    with params.context():
        plaintext = encode(np.full(params.slot_count, value), msg_bound=2)
        assert plaintext.m.to_centered_list() == [expected] + [0] * (params.n - 1)


@pytest.mark.parametrize("coefficient", [-(2**63) - 1, -(2**63), -(2**63) + 1, 2**63 - 1, 2**63, 2**63 + 1])
def test_flint_encoding_preserves_integer_boundaries(integer_boundary_params, coefficient):
    params = integer_boundary_params
    with params.context(), flint_precision(160):
        value = acb(arb(coefficient) / 2**63)
        plaintext = encode([value] * params.slot_count, msg_bound=2, prec=140)
        assert plaintext.m.to_centered_list() == [coefficient] + [0] * (params.n - 1)


@pytest.mark.parametrize("low_prec", [1, np.int64(53)])
def test_encoding_precision_default_and_override(integer_boundary_params, low_prec):
    params = integer_boundary_params
    with flint_precision(160):
        value = acb(1 + arb(2) ** -63)
    with params.context(encoding_precision_bits=np.int64(140)):
        high = encode(value, msg_bound=2)
        low = encode(value, msg_bound=2, prec=low_prec)
        assert high.m.to_centered_list() == [2**63 + 1] + [0] * (params.n - 1)
        assert low.m.to_centered_list() == [2**63] + [0] * (params.n - 1)
    with params.context(encoding_precision_bits=low_prec):
        assert encode(value, msg_bound=2).m.to_centered_list() == low.m.to_centered_list()
        explicit = encode(value, msg_bound=2, prec=140)
        assert explicit.m.to_centered_list() == high.m.to_centered_list()


def test_decode_precision_preserves_cancellation(integer_boundary_params):
    # At the first slot, X² - X⁶ = sqrt(2). The large terms almost cancel.
    magnitude = 2**60
    coefficient = math.isqrt(2 * magnitude**2)
    with integer_boundary_params.context():
        ring = integer_boundary_params.ring_at(0)
        plaintext = Plaintext(Poly([coefficient, 0, -magnitude, 0, 0, 0, magnitude, 0], ring), noise_info=CKKSNoiseInfo.zero())
        with flint_precision(200):
            expected = float((coefficient - magnitude * arb(2).sqrt()) / int(plaintext.scale))
        result = decode(plaintext, 1, prec=140)[0]
    assert isinstance(result, complex)
    assert abs(result - expected) < 2**-120


def test_jit_encoding_uses_context_precision(integer_boundary_params):
    from ren import jit

    params = integer_boundary_params
    keys = params.context().keyset
    with flint_precision(160):
        value = acb(1 + arb(2) ** -63)

    @jit
    def add_constant(x):
        return x + encode(value, msg_bound=2).m

    x = Poly.zero(params.ring_at(0)).realize()
    for prec, expected in ((53, 2**63), (140, 2**63 + 1), (53, 2**63)):
        with keys.context(encoding_precision_bits=prec):
            assert add_constant(x).to_centered_list() == [expected] + [0] * (params.n - 1)


@pytest.mark.parametrize("sign", [-1, 1])
def test_high_precision_scalar_rounds_complex_ties(integer_boundary_params, sign):
    params = integer_boundary_params
    with flint_precision(180):
        value = acb(sign * arb(2) ** -64, sign * arb(2) ** -64)
    with params.context():
        plaintext = encode(value, msg_bound=1, prec=180)
        expected = [0] * params.n
        expected[0] = expected[params.n // 2] = sign
        assert plaintext.m.to_centered_list() == expected


def test_encoding_rejects_precision_that_cannot_preserve_rounding_bound(integer_boundary_params):
    params = replace(integer_boundary_params, levels=(replace(integer_boundary_params.levels[0], scale=BigFloat(2) ** 63 + 1),))
    with flint_precision(220):
        scaled = arb(2) ** 40 + arb(1) / 2 + arb(2) ** -30
        value = acb(scaled / arb(str(params.scale(0))))
    with params.context():
        with pytest.raises(InvalidParameterError, match="rounding"):
            encode(value, msg_bound=1, prec=54)
        plaintext = encode(value, msg_bound=1, prec=100)
        actual = plaintext.m.to_centered_list()[0]
    with flint_precision(220):
        error = abs(actual - scaled).abs_upper()
        assert error <= arb(str(plaintext.noise_info.noise_bound))


def test_flint_decode_encloses_values_at_exact_scale(integer_boundary_params):
    scale = BigFloat(2) ** 63 + BigFloat(2) ** -100
    params = replace(integer_boundary_params, levels=(replace(integer_boundary_params.levels[0], scale=scale),))
    with params.context():
        decoded = encoding._decode(Poly.const(1, params.ring_at(0)), scale=scale, prec=320)
    with flint_precision(360):
        expected = 1 / (arb(2) ** 63 + arb(2) ** -100)
        assert all(expected in value.real and 0 in value.imag for value in decoded)


@pytest.mark.parametrize("prec", [0, -1])
def test_codec_rejects_nonpositive_precision(integer_boundary_params, prec):
    params = integer_boundary_params
    with pytest.raises(InvalidParameterError):
        params.context(encoding_precision_bits=prec)
    with params.context():
        plaintext = Plaintext.one()
        with pytest.raises(InvalidParameterError):
            encode(1, msg_bound=1, prec=prec)
        with pytest.raises(InvalidParameterError):
            decode(plaintext, prec=prec)


def test_tracking_arithmetic_follows_configured_precision(integer_boundary_params):
    with flint_precision(220):
        correction = acb(arb(2) ** -120)
    with integer_boundary_params.context(encoding_precision_bits=180, track_plaintext_values=True):
        result = encode(1, msg_bound=1) + encode(correction, msg_bound=1) - 1
        assert result._values is not None
    with flint_precision(220):
        assert all(abs(value - correction).abs_upper() < arb(2) ** -160 for value in result._values)


def test_encode_respects_disabled_plaintext_tracking(test_params: CKKSParams):
    with test_params.context():
        pt_from_array = encode([1.0, -2.0], msg_bound=2)
        pt_from_scalar = encode(1.0, msg_bound=1)

        assert pt_from_array._values is None
        assert pt_from_scalar._values is None
        assert pt_from_array.to_ciphertext()._values is None
        assert pt_from_scalar.to_ciphertext()._values is None
        with pytest.raises(RuntimeError, match="plaintext value tracking is not available"):
            _ = pt_from_array.values
        with pytest.raises(RuntimeError, match="plaintext value tracking is not available"):
            _ = pt_from_array.to_ciphertext().values


def test_encoding_snapshots_and_pads_input_slots(keys: CKKSKeySet):
    values = np.array([0.25 + 0.125j, -0.5])
    expected = np.zeros(keys.params.slot_count, dtype=complex)
    expected[:2] = values
    with keys.context(track_plaintext_values=True):
        plaintext = encode(values, msg_bound=1)
        values[:] = 7
        assert plaintext.values == pytest.approx(expected)
        assert decode(plaintext) == pytest.approx(expected, abs=1e-8)


def test_flint_encoding_snapshots_full_length_input_slots(keys: CKKSKeySet):
    with flint_precision(200), keys.context(track_plaintext_values=True):
        exact = acb(1 + arb(2) ** -80)
        values = np.full(keys.params.slot_count, acb(0), dtype=object)
        values[0] = exact
        plaintext = encode(values, msg_bound=2, prec=180)
        values[0] = acb(7)
        assert plaintext._values is not None
        assert abs(plaintext._values[0] - exact).abs_upper() < arb(2) ** -90
        assert all(value == acb(0) for value in plaintext._values[1:])


@pytest.mark.parametrize("value", [np.int64(2), np.array(2)])
def test_encoding_broadcasts_scalar_inputs(keys: CKKSKeySet, value):
    with keys.context(track_plaintext_values=True):
        plaintext = encode(value, msg_bound=2)
        assert plaintext.values == pytest.approx(np.full(keys.params.slot_count, 2))
        assert decode(plaintext) == pytest.approx([2] * keys.params.slot_count, abs=1e-8)


@pytest.mark.parametrize(
    "prec,value",
    [
        pytest.param(53, 0.125, id="numpy"),
        pytest.param(100, np.float64(0.125), id="flint-float64"),
        pytest.param(100, np.complex128(0.125 + 0.25j), id="flint-complex128"),
        pytest.param(100, np.int64(1), id="flint-int64"),
    ],
)
def test_encoding_snapshots_broadcast_input(keys: CKKSKeySet, prec, value):
    scalar = np.array([value])
    with keys.context(track_plaintext_values=True):
        plaintext = encode(np.broadcast_to(scalar, (keys.params.slot_count,)), msg_bound=1, prec=prec)
        expected = encode(scalar[0].item(), msg_bound=1, prec=prec)
        scalar[0] = 7
        assert plaintext.m.equals(expected.m)
        assert_equal(plaintext.noise_info, expected.noise_info)
        assert plaintext.values == pytest.approx([value] * keys.params.slot_count)
        assert decode(plaintext) == pytest.approx([value] * keys.params.slot_count, abs=1e-8)


@pytest.mark.parametrize("value", ["1", None, object()])
def test_encoding_rejects_unsupported_slot_inputs(keys: CKKSKeySet, value):
    with keys.context(), pytest.raises(TypeError):
        encode(value, msg_bound=1)


@pytest.mark.parametrize("prec", [53, 140])
def test_encoding_rejects_invalid_slot_shapes(keys: CKKSKeySet, prec):
    with keys.context():
        for values in (np.array([[1, 2]]), np.zeros(keys.params.slot_count + 1)):
            with pytest.raises(InvalidParameterError):
                encode(values, msg_bound=2, prec=prec)


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
@pytest.mark.parametrize("tracking", [False, True])
def test_slot_operations_follow_tracking_context(keys: CKKSKeySet, kind, tracking):
    with keys.context(track_plaintext_values=True):
        value = kind.const(0.125, level=1)
        assert value._values is not None
    with keys.context(track_plaintext_values=tracking):
        rebuilt = value.replace(_values=value._values)
        assert rebuilt._values is not None and all(rebuilt._values == value._values)
        for result, expected in ((-value, -value.values), (value + value, 2 * value.values), (value.rotate(1), value.values)):
            assert (result._values is not None) == tracking
            if tracking:
                assert result.values == pytest.approx(expected)
        converted = (encrypt(value), value.to_ciphertext()) if isinstance(value, Plaintext) else (decrypt(value),)
        for result in (*converted, value.to_level(0)):
            assert result.values == pytest.approx(value.values)
        missing = kind.const(0.25, level=1).replace(_values=None)
        for op in (operator.add, operator.sub, operator.mul):
            assert op(value, missing)._values is None
            for scalar in (2, np.int64(2), 0.5, 0.5 + 0.25j):
                for result, expected in ((op(value, scalar), op(value.values, scalar)), (op(scalar, value), op(scalar, value.values))):
                    assert (result._values is not None) == tracking
                    if tracking:
                        assert result.values == pytest.approx(expected)

    rebuilt = value.replace(_values=value._values)
    assert rebuilt._values is not None and all(rebuilt._values == value._values)
    if isinstance(value, Plaintext):
        assert value.to_ciphertext().values == pytest.approx(value.values)
    assert (-value)._values is None


@pytest.mark.parametrize("scalar", [2**53 + 1, np.int64(2**53 + 1)])
def test_scalar_tracking_preserves_integer_precision_across_contexts(keys: CKKSKeySet, scalar):
    with flint_precision(53), keys.context(track_plaintext_values=True):
        zero = Plaintext.zero(level=2)
        constant = Plaintext.const(scalar, level=2)
        tracked = zero + scalar
    with keys.context():
        untracked_context = zero + scalar
    expected = acb(int(scalar))
    assert untracked_context._values is None
    for result in (constant, tracked):
        assert result._values is not None
        assert all(value == expected for value in result._values)


@pytest.mark.parametrize("tracking", [False, True])
def test_existing_tracking_still_checks_ciphertext_overflow(keys: CKKSKeySet, tracking: bool):
    with keys.context(track_plaintext_values=True):
        ct = Ciphertext.const(1, level=1)
        assert ct._values is not None
        overflow = ct._values * float(BigFloat(ct.ring.q) / ct.scale)
    with keys.context(track_plaintext_values=tracking):
        with pytest.raises(OverflowError, match="Ciphertext overflow is occurring"):
            ct.replace(_values=overflow)


def test_ciphertext_construction_checks_component_rings(keys: CKKSKeySet):
    with keys.context():
        ct = Ciphertext.const(0.125, level=2)
        different_ring = ct.b.to_quotient_ring(keys.params.ring_at(1))
        with pytest.raises(ParameterMismatchError, match="polynomial ring mismatch"):
            Ciphertext(ct.a, different_ring, noise_info=ct.noise_info)


def test_keyset_to_copies_all_key_material(keys: CKKSKeySet):
    target_device = "cuda:3" if keys.device != "cuda:3" else "python"
    moved_keys = keys.to(target_device)
    moved_ctx = moved_keys.context()

    assert moved_keys is not keys
    assert moved_ctx.device == target_device
    assert moved_ctx.params is keys.params
    assert moved_ctx.keyset.sk is not None
    assert moved_ctx.keyset.pk is not None
    assert moved_ctx.keyset.rlk is not None
    assert moved_ctx.keyset.sk.device == target_device
    assert moved_ctx.keyset.pk.ct.device == target_device
    assert all(digit.device == target_device for digit in moved_ctx.keyset.rlk.digits)
    assert all(digit.device == target_device for key in moved_ctx.keyset.galois_keys.values() for digit in key.digits)
    assert keys.to(keys.device) is keys


def test_encode_decode_flint_roundtrip_uses_acb_values(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        expected = random_values_with_norm(keys.params.slot_count, norm=10)
        values = [acb(value.real, value.imag) for value in expected]
        pt = encode(values, msg_bound=10, prec=100)
        decoded = encoding._decode(pt.m, scale=pt.scale, prec=100)[: len(values)]

        assert all(isinstance(slot, acb) for slot in decoded)
        assert pt.values == pytest.approx(expected)
        assert [complex(float(slot.real), float(slot.imag)) for slot in decoded] == pytest.approx(expected, abs=1e-12)


def test_flint_plaintext_roundtrip_beyond_double_precision():
    n = 64
    moduli = select_moduli(n=n, num_primes=5, bit_length=30)
    params = CKKSParams.from_levels(
        n=n,
        dtype=dtypes.uint32,
        levels=(CKKSLevel(moduli=moduli, scale=BigFloat(2) ** 120),),
        ks_primes=select_moduli(n=n, num_primes=1, bit_length=30, exclude=set(moduli)),
        ks_digit_sizes=(len(moduli),),
        log_range=0,
        security=None,
    )
    with flint_precision(200):
        values = [acb(1 + arb(2) ** -80, arb(2) ** -90), acb(-(arb(2) ** -85), -1)]
        with params.context():
            plaintext = encode(values, msg_bound=2, prec=180)
            decoded = encoding._decode(plaintext.m, scale=plaintext.scale, prec=180)
        expected = values + [acb(0)] * (n // 2 - len(values))
        assert all(abs(a - b).abs_upper() < arb(2) ** -110 for a, b in zip(decoded, expected, strict=True))
        assert abs(decoded[0] - acb(1, arb(2) ** -90)).abs_lower() > arb(2) ** -81


HIGH_PRECISION_FLINT_BITS = 140
HIGH_PRECISION_ACCEPTANCE_BITS = 70


@pytest.fixture(scope="module")
def high_precision_keys():
    set_global_rng_seed(42)
    params = high_precision_params()
    return keygen(params)


def _random_flint_values(*, count: int, seed: int, bound: float = 0.01) -> list:
    rng = random.Random(seed)
    return [acb(arb(str(rng.uniform(-bound, bound))), arb(str(rng.uniform(-bound, bound)))) for _ in range(count)]


def _assert_flint_values_close(actual: Sequence | np.ndarray, expected: Sequence, *, bits: int) -> None:
    threshold = arb(2) ** -bits
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert abs(actual_value - expected_value).abs_upper() < threshold


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_high_precision_flint_encrypt_decrypt_roundtrip(high_precision_keys: CKKSKeySet):
    with flint_precision(HIGH_PRECISION_FLINT_BITS), high_precision_keys.context():
        values = _random_flint_values(count=8, seed=20260505)
        pt = encode(values, msg_bound=0.02, prec=HIGH_PRECISION_FLINT_BITS)
        ct = encrypt(pt)
        decrypted = decrypt(ct)
        decoded = encoding._decode(decrypted.m, scale=decrypted.scale, prec=HIGH_PRECISION_FLINT_BITS)[: len(values)]

    _assert_flint_values_close(decoded, values, bits=HIGH_PRECISION_ACCEPTANCE_BITS)


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_high_precision_flint_to_level_zero_preserves_precision(high_precision_keys: CKKSKeySet):
    with flint_precision(HIGH_PRECISION_FLINT_BITS), high_precision_keys.context():
        values = _random_flint_values(count=8, seed=20260506)
        pt = encode(values, msg_bound=0.02, prec=HIGH_PRECISION_FLINT_BITS)
        ct = encrypt(pt).to_level(0)
        decrypted = decrypt(ct)
        decoded = encoding._decode(decrypted.m, scale=decrypted.scale, prec=HIGH_PRECISION_FLINT_BITS)[: len(values)]

    _assert_flint_values_close(decoded, values, bits=HIGH_PRECISION_ACCEPTANCE_BITS)


params_factories = [
    small_params,
    lambda: CKKSParams.default(boot_params=None),
    CKKSParams.default,
    lambda: CKKSParams.generate_prefix(
        log_n=16, depth=100, log_precision=45, log_range=14, p_prime_count=3, security=None, dtype=dtypes.uint64
    ),  # Tests stability of prime-finding algorithm
]


@pytest.mark.parametrize("params_factory", params_factories)
def test_params_moduli(params_factory: Callable[[], CKKSParams]):
    """
    Asserts that all moduli are prime, distinct, and within dtype range.
    """
    params = params_factory()
    moduli = (*params.highest_ring.moduli, *params.ks_primes)
    assert len(moduli) == len(set(moduli)), "Not all primes are distinct"
    for prime in moduli:
        assert is_prime_u64(prime), f"{prime} is not prime"
        assert 0 < prime < 1 << (8 * params.dtype.itemsize)


@pytest.mark.parametrize("params_factory", params_factories)
def test_scale_rejects_out_of_range_level(params_factory: Callable[[], CKKSParams]):
    params = params_factory()
    with pytest.raises(InvalidParameterError):
        params.scale(params.highest_level + 1)


def test_encrypt_decrypt_roundtrip(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        pt = random_plaintext(msg_bound=1000)
    with keys.context():
        ct = encrypt(pt)
        assert ct.values == pytest.approx(pt.values)
    with keys.context(track_plaintext_values=True):
        decoded = decode(decrypt(ct), keys.params.slot_count)
        assert decoded == pytest.approx(pt.values, abs=1e-4)


def test_ciphertext_clone_realizes_clone_without_materializing_source(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=10)
        clone = ct.clone().realize()

        assert clone is not ct
        assert not ct.a.is_realized
        assert not ct.b.is_realized
        assert clone.a.is_realized
        assert clone.b.is_realized
        assert decode(decrypt(clone), keys.params.slot_count) == pytest.approx(ct.values, abs=1e-4)


def test_ciphertext_clone_uses_fresh_storage_for_realized_source(keys: CKKSKeySet):
    @dataclass(frozen=True, slots=True)
    class TaggedCiphertext(Ciphertext):
        tag: str

    with keys.context(track_plaintext_values=True):
        source = random_ciphertext(msg_bound=10).realize()
        ct = TaggedCiphertext(source.a, source.b, noise_info=source.noise_info, _values=source._values, tag="keep")
        clone = ct.clone().realize()

        assert type(clone) is TaggedCiphertext
        assert clone.tag == ct.tag
        assert clone.a.node.base.buffer is not ct.a.node.base.buffer
        assert clone.b.node.base.buffer is not ct.b.node.base.buffer
        assert decode(decrypt(clone), keys.params.slot_count) == pytest.approx(ct.values, abs=1e-4)


def test_public_key_encrypt_at_each_level(keys: CKKSKeySet):
    if keys.pk is None:
        pytest.skip("context has no public key")

    pk_only_ctx = CKKSKeySet(params=keys.params, pk=keys.pk).context()

    with pk_only_ctx:
        pt = encode(1, msg_bound=1)
    expected = [1] * keys.params.slot_count

    for level in range(keys.params.highest_level + 1):
        with pk_only_ctx:
            ct = encrypt(pt.to_level(level))
            assert ct.level == level
            assert ct._values is None

        with keys.context():
            assert decode(decrypt(ct), keys.params.slot_count) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_add_sub(keys: CKKSKeySet, op):
    with keys.context(track_plaintext_values=True):
        a = random_ciphertext(msg_bound=1000)
        b = random_ciphertext(msg_bound=1000)
        expected = op(a.values, b.values)
        result = op(a, b)
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_scalar_add_sub(keys: CKKSKeySet, op):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        scalar = random_complex_with_norm(1000)
        expected = op(ct.values, scalar)
        result = op(ct, scalar)
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-4)


def test_scalar_add_sub_after_realization(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        ct = (random_ciphertext(msg_bound=10) + random_ciphertext(msg_bound=10)).realize()
        assert decode(decrypt(ct + 3)) == pytest.approx(ct.values + 3, abs=1e-4)
        assert decode(decrypt(ct - 3)) == pytest.approx(ct.values - 3, abs=1e-4)


@pytest.mark.parametrize("make_value,tolerance", [(random_plaintext, 2e-5), (random_ciphertext, 1e-4)])
@pytest.mark.parametrize("tracking", [False, True])
def test_negate(keys: CKKSKeySet, make_value, tolerance, tracking):
    with keys.context(track_plaintext_values=True):
        value = make_value(msg_bound=1000)
        expected = -value.values
        if not tracking:
            value = value.replace(_values=None)
        result = -value
        assert result.level == value.level
        if tracking:
            assert result.values == pytest.approx(expected, abs=1e-9)
        else:
            assert result._values is None
        plaintext = decrypt(result) if isinstance(result, Ciphertext) else result
        assert decode(plaintext) == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("samples", [1, pytest.param(50, marks=pytest.mark.slow, id="stress")])
def test_mul(keys: CKKSKeySet, samples: int):
    for _ in range(samples):
        with keys.context(track_plaintext_values=True):
            a = random_ciphertext(msg_bound=100)
            b = random_ciphertext(msg_bound=100)
            expected = a.values * b.values
            result = a * b
            decoded = decode(decrypt(result))
            assert result.values == pytest.approx(expected, abs=1e-9)
            assert decoded == pytest.approx(expected, abs=1e-4)


def test_scalar_mul(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        scalar = random_complex_with_norm(10)
        expected = ct.values * scalar
        result = ct * scalar
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-4)
        # TODO: Check the noise estimate after scalar multiplication.


@pytest.mark.parametrize("make_value", [random_plaintext, random_ciphertext])
@pytest.mark.parametrize(
    "level,msg_bound,tracking,scalar",
    [(None, 1000, False, -3), (None, 1000, True, -3), (0, 0.125, False, -3), (0, 0.125, True, -3), (0, 0.125, True, np.int64(-3))],
)
def test_integer_scalar_mul(keys: CKKSKeySet, make_value, level, msg_bound, tracking, scalar):
    with keys.context(track_plaintext_values=True):
        value = make_value(msg_bound=msg_bound, level=level)
        expected = -3 * value.values
        if not tracking:
            value = value.replace(_values=None)
        for result in (value * scalar, scalar * value):
            # Integer multiplication scales coefficients directly: no rescale, no level drop.
            assert type(result) is type(value)
            assert result.level == value.level
            if tracking:
                assert result.values == pytest.approx(expected, abs=1e-9)
            else:
                assert result._values is None
            plaintext = decrypt(result) if isinstance(result, Ciphertext) else result
            assert decode(plaintext) == pytest.approx(expected, abs=1e-4)
            assert_equal(result.noise_info, value.noise_info * 3)


def test_multi_mul(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        a = random_ciphertext(msg_bound=10)
        b = random_ciphertext(msg_bound=10)
        c = random_ciphertext(msg_bound=10)
        d = random_ciphertext(msg_bound=10)
        expected = a.values * b.values * c.values * d.values
        result = a * b * c * d
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-1)


def test_linear(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        expected = 2 * ct.values + 1
        result = 2 * ct + 1
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("e", range(7))
def test_pow(keys: CKKSKeySet, e: int):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=10)
        expected = ct.values**e
        result = ct**e
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=5e-2)


def test_pow_zero_preserves_level(keys: CKKSKeySet):
    with keys.context():
        ct = random_ciphertext(msg_bound=10, level=0)
        assert (ct**0).level == ct.level


@pytest.mark.parametrize("exponent", [0.5, 2.0])
def test_pow_requires_integer_exponent(keys: CKKSKeySet, exponent):
    with keys.context(), pytest.raises(TypeError):
        _ = Ciphertext.one() ** exponent


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
def test_slot_constant_factories_preserve_type_level_and_bound(keys: CKKSKeySet, kind):
    @dataclass(frozen=True, slots=True)
    class Tagged(kind):
        label: str = "constant"

    with keys.context():
        for cls in (kind, Tagged):
            for value, expected, level, bound in (
                (cls.zero(level=0), 0, 0, 0),
                (cls.one(level=0), 1, 0, 1),
                (cls.const(0, msg_bound=0.0, level=1), 0, 1, 0),
                (cls.const(0.125 + 0.25j, msg_bound=2.0, level=1), 0.125 + 0.25j, 1, 2.0),
                (cls.const(0.125 + 0.25j, level=1), 0.125 + 0.25j, 1, abs(0.125 + 0.25j)),
            ):
                assert type(value) is cls
                if cls is Tagged:
                    assert value.label == "constant"
                assert value.level == level
                assert value.noise_info.msg_bound == bound * value.scale
                plaintext = decrypt(value) if isinstance(value, Ciphertext) else value
                assert decode(plaintext, length=1) == pytest.approx([expected], abs=1e-9)


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext, RLWECiphertext])
def test_rescale_without_slot_context(keys: CKKSKeySet, kind):
    with keys.context(track_plaintext_values=True):
        pt = Plaintext.const(0.125, level=2)
        ct = pt.to_ciphertext()
        value = pt if kind is Plaintext else ct if kind is Ciphertext else RLWECiphertext(ct.a, ct.b, ct.noise_info)
        target = keys.params.ring_at(1)

    result = value._rescale(target.moduli)
    assert type(result) is kind
    assert result.ring == target
    assert result.noise_info == value.noise_info * (BigFloat(target.q) / value.ring.q)
    for original, rescaled in zip(value.polys(), result.polys(), strict=True):
        assert rescaled.equals(original.rescale(target.moduli))
    if isinstance(result, (Plaintext, Ciphertext)):
        assert pt._values is not None
        assert result._values is None


@pytest.mark.parametrize("k", [*range(-10, 11), params.slot_count + 1, -params.slot_count - 1])
def test_rotate(keys: CKKSKeySet, k: int):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        expected = np.roll(ct.values, -k)
        result = ct.rotate(k)
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=2e-5)
    # TODO: Check the noise estimate once tracking is more accurate.


@pytest.mark.parametrize("realize_input", [False, True])
def test_function_rotate_matches_direct_without_plaintext_tracking(keys: CKKSKeySet, realize_input: bool):
    with keys.context():
        values = random_values_with_norm(keys.params.slot_count, norm=10)

        @ren.function
        def rotate_once(ct: Ciphertext) -> Ciphertext:
            return ct.rotate(1)

        direct_input = encrypt(encode(values, msg_bound=10))
        assert direct_input.rotate(0) is direct_input
        function_input = encrypt(encode(values, msg_bound=10))
        if realize_input:
            function_input.realize()

        expected = decode(decrypt(direct_input.rotate(1)))
        actual = decode(decrypt(rotate_once(function_input)))

        assert actual == pytest.approx(expected, abs=2e-5)


@pytest.mark.parametrize("jitted, expected_calls", [(False, 3), (True, 2)], ids=["function", "jit"])
def test_function_rebinds_context_inputs(keys: CKKSKeySet, jitted: bool, expected_calls: int):
    ctx = keys.context()
    set_global_rng_seed(43)
    alternate_ctx = keygen(keys.params, rotations={1}).realize().context()
    calls = 0

    @ren.function
    def rotate_once(ct: Ciphertext) -> Ciphertext:
        nonlocal calls
        calls += 1
        return ct.rotate(1)

    run = ren.jit(rotate_once) if jitted else rotate_once
    values = random_values_with_norm(keys.params.slot_count, norm=10)
    for active in (ctx, alternate_ctx, ctx):
        with active:
            direct_input = encrypt(encode(values, msg_bound=10))
            function_input = encrypt(encode(values, msg_bound=10))
            expected = decode(decrypt(direct_input.rotate(1)))
            actual = decode(decrypt(run(function_input)))

            assert actual == pytest.approx(expected, abs=2e-5)
    assert calls == expected_calls


@pytest.mark.parametrize("tracked", [False, True])
def test_conjugate(keys: CKKSKeySet, tracked: bool):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        expected = ct.values.conjugate()
        if not tracked:
            ct = ct.replace(_values=None)
        result = ct.conjugate()
        decoded = decode(decrypt(result))
        assert decoded == pytest.approx(expected, abs=1e-4)
        if tracked:
            assert result.values == pytest.approx(expected, abs=1e-4)
        else:
            assert result._values is None
    # TODO: Check the noise estimate once tracking is more accurate.


def test_conjugate_preserves_high_precision_references(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True), flint_precision(200):
        value = acb(arb(1) / 8 + arb(2) ** -80, arb(1) / 4 - arb(2) ** -90)
        ciphertext = encrypt(encode([value], msg_bound=1, prec=180))
        with flint_precision(53):
            result = ciphertext.conjugate()
        assert result._values is not None
        actual = result._values[0]
        expected = value.conjugate()
        assert expected.real in actual.real
        assert expected.imag in actual.imag
        assert abs(actual - acb(0.125, -0.25)).abs_lower() > arb(2) ** -81
        assert all(slot == acb(0) for slot in result._values[1:])


def test_rotate_hoisting_identity(keys: CKKSKeySet):
    """Both rotations use the same digits of ``a`` and apply σ_g after the external
    product. Hoisting reuses the decomposition, so both paths produce identical
    ciphertext coefficients.
    """

    with keys.context():
        ct = random_ciphertext(msg_bound=1000)
        normal = ct.rotate(1)
        hoisted = ct.rotate([1, 2])[0]
        assert hoisted.a.equals(normal.a)
        assert hoisted.b.equals(normal.b)


def test_galois_key_permute_last_relation(keys: CKKSKeySet):
    """Each permute-last galois key digit encrypts P·s·u_i under σ_{g⁻¹}(s).

    Verifies the key in isolation, with no rotation path involved: decrypt
    digit i directly under the encrypting secret σ_{g⁻¹}(s) and check it matches
    the masked secret P·s·u_i up to encryption noise. ``u_i`` places prod(P) on
    segment i's Q primes, zero elsewhere -- the same mask keygen builds per digit.
    """
    with keys.context():
        assert keys.sk is not None
        s = keys.sk
        n = keys.params.n
        noise_bound = keys.params.noise_dist.coeff_inf_norm_bound()

        tested = 0
        for galois_elt, gk in keys.galois_keys.items():
            ring_qt = gk.ring_qt
            p_prime_count = gk.p_prime_count
            q_moduli = ring_qt.moduli[p_prime_count:]
            p_prod = math.prod(ring_qt.moduli[:p_prime_count])

            # Encrypting secret σ_{g⁻¹}(s) and the plain secret s, both over Q̃.
            g_inv = pow(galois_elt, -1, 2 * n)
            s_inv = s.automorphism(g_inv)._to_ring_small_coeffs(ring_qt)
            s_qt = s._to_ring_small_coeffs(ring_qt)

            digit_start = 0
            for digit_size, digit in zip(gk.digit_sizes, gk.digits, strict=True):
                decrypted = digit.b + digit.a * s_inv

                # Rebuild u_i: prod(P) on segment i's Q primes inside Q̃.
                digit_stop = digit_start + digit_size
                mask_scalars = (
                    (0,) * (p_prime_count + digit_start)
                    + tuple(p_prod % q for q in q_moduli[digit_start:digit_stop])
                    + (0,) * (len(q_moduli) - digit_stop)
                )
                expected = s_qt.scale_by_primes(mask_scalars)

                residual = (decrypted - expected).to_centered_list()
                assert max(abs(c) for c in residual) <= noise_bound
                tested += 1
                digit_start = digit_stop

        assert tested > 0, "no galois keys to test"


def test_times_i(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        expected = 1j * ct.values
        result = ct.times_i()
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-4)
    # TODO: Reduce the error and check the noise estimate.


@pytest.mark.parametrize("level", [0, 2, 3])
def test_to_level(keys: CKKSKeySet, level: int):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        expected = ct.values
        result = ct.to_level(level)
        assert result.level == level
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("level", range(1, params.highest_level + 1))
def test_multiplication_at_different_levels(keys: CKKSKeySet, level: int):
    with keys.context(track_plaintext_values=True):
        a = random_ciphertext(msg_bound=10).to_level(level)
        b = random_ciphertext(msg_bound=100).to_level(level)
        expected = a.values * b.values
        result = a * b
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize("a_level", range(1, params.highest_level + 1, 2))
@pytest.mark.parametrize("b_level", range(2, params.highest_level + 1, 2))
def test_multiplication_precision(keys: CKKSKeySet, a_level: int, b_level: int):
    with keys.context(track_plaintext_values=True):
        a = random_ciphertext(msg_bound=1, level=a_level)
        b = random_ciphertext(msg_bound=1, level=b_level)
        expected = a.values * b.values
        result = a * b
        decoded = decode(decrypt(result))
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("k", [*range(-10, 11), params.slot_count + 1, -params.slot_count - 1])
def test_plaintext_rotate(keys: CKKSKeySet, k: int):
    with keys.context(track_plaintext_values=True):
        pt = random_plaintext(msg_bound=1000)
        expected = np.roll(pt.values, -k)
        result = pt.rotate(k)
        decoded = decode(result)
        assert result.values == pytest.approx(expected, abs=1e-9)
        assert decoded == pytest.approx(expected, abs=2e-5)


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_plaintext_add_sub(keys: CKKSKeySet, op):
    with keys.context(track_plaintext_values=True):
        a = random_plaintext(msg_bound=1000)
        b = random_plaintext(msg_bound=1000)
        a_values, b_values = a.values, b.values
        result = op(a, b)
        assert result.level == a.level
        assert result.values == pytest.approx(op(a_values, b_values), abs=1e-9)
        assert decode(result) == pytest.approx(result.values, abs=2e-5)


def test_plaintext_mul(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        a = random_plaintext(msg_bound=100)
        b = random_plaintext(msg_bound=100)
        a_values, b_values = a.values, b.values
        result = a * b
        # A plaintext's scale is fixed by its ring, so the product must rescale.
        assert result.level == a.level - 1
        assert result.values == pytest.approx(a_values * b_values, abs=1e-9)
        assert decode(result) == pytest.approx(result.values, abs=1e-4)


@pytest.mark.parametrize("a_level", range(1, params.highest_level + 1, 2))
@pytest.mark.parametrize("b_level", range(2, params.highest_level + 1, 2))
def test_plaintext_mixed_levels(keys: CKKSKeySet, a_level: int, b_level: int):
    with keys.context(track_plaintext_values=True):
        a = random_plaintext(msg_bound=10, level=a_level)
        b = random_plaintext(msg_bound=10, level=b_level)
        a_values, b_values = a.values, b.values
        min_level = min(a_level, b_level)

        added = a + b
        assert added.level == min_level
        assert decode(added) == pytest.approx(a_values + b_values, abs=1e-5)

        subtracted = a - b
        assert subtracted.level == min_level
        assert decode(subtracted) == pytest.approx(a_values - b_values, abs=1e-5)

        multiplied = a * b
        assert multiplied.level == min_level - 1
        assert decode(multiplied) == pytest.approx(a_values * b_values, abs=1e-5)


def test_plaintext_scalar_add_sub(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        pt = random_plaintext(msg_bound=1000)
        scalar = random_complex_with_norm(1000)
        values = pt.values
        assert values is not None

        cases = [
            (pt + scalar, values + scalar),
            (scalar + pt, values + scalar),
            (pt - scalar, values - scalar),
            (scalar - pt, scalar - values),
        ]
        for result, expected in cases:
            assert result.level == pt.level
            assert decode(result) == pytest.approx(expected, abs=1e-4)


def test_plaintext_complex_scalar_mul(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        pt = random_plaintext(msg_bound=1000)
        scalar = random_complex_with_norm(10)
        values = pt.values
        assert values is not None
        for result in (pt * scalar, scalar * pt):
            # A non-integer scalar is encoded, so the multiply rescales.
            assert result.level == pt.level - 1
            assert decode(result) == pytest.approx(values * scalar, abs=1e-4)


def test_plaintext_add_sub_negate_noise(keys: CKKSKeySet):
    with keys.context():
        a = random_plaintext(msg_bound=1000)
        b = random_plaintext(msg_bound=1000)
        expected_noise = a.noise_info + b.noise_info
        assert_equal((a + b).noise_info, expected_noise)
        assert_equal((a - b).noise_info, expected_noise)
        assert_equal((-a).noise_info, a.noise_info)


def test_plaintext_scalar_add_sub_after_ntt_realization(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        pt = random_plaintext(msg_bound=10)
        pt = replace(pt, m=pt.m.ntt())
        pt.m.realize()
        assert decode(pt + 3) == pytest.approx(pt.values + 3, abs=1e-4)
        assert decode(pt - 3) == pytest.approx(pt.values - 3, abs=1e-4)


@pytest.mark.parametrize("pt_level,ct_level", [(1, 2), (2, 1), (2, 2)])
def test_plaintext_ciphertext_mixed_dispatch(keys: CKKSKeySet, pt_level: int, ct_level: int):
    with keys.context(track_plaintext_values=True):
        pt = random_plaintext(msg_bound=10, level=pt_level)
        ct = random_ciphertext(msg_bound=10, level=ct_level)
        pt_values = pt.values
        ct_values = ct.values

        # Plaintext's operators defer to Ciphertext's reflected ones.
        cases = [
            (pt + ct, pt_values + ct_values),
            (ct + pt, ct_values + pt_values),
            (pt - ct, pt_values - ct_values),
            (ct - pt, ct_values - pt_values),
            (pt * ct, pt_values * ct_values),
            (ct * pt, ct_values * pt_values),
        ]
        for result, expected in cases:
            assert isinstance(result, Ciphertext)
            assert result.values == pytest.approx(expected, abs=1e-4)
            assert decode(decrypt(result)) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
def test_unsupported_arithmetic_does_not_align_or_count(keys: CKKSKeySet, kind):
    with keys.context():
        value = kind.const(0.125)

    # Unsupported operands must defer before accessing the CKKS context.
    with CallCounter() as counter:
        for op in (operator.add, operator.sub, operator.mul):
            for left, right in ((value, object()), (object(), value)):
                with pytest.raises(TypeError):
                    cast(Callable[[Any, Any], Any], op)(left, right)
    assert counter.counts == {}


@pytest.mark.parametrize("left_kind", [Plaintext, Ciphertext])
@pytest.mark.parametrize("right_kind", [Plaintext, Ciphertext])
@pytest.mark.parametrize("missing", [0, 1])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
def test_arithmetic_propagates_missing_tracked_values(keys: CKKSKeySet, left_kind, right_kind, missing, op):
    with keys.context(track_plaintext_values=True):
        operands = [left_kind.const(0.125, level=2), right_kind.const(0.25, level=2)]
        operands[missing] = operands[missing].replace(_values=None)
        result = op(*operands)
        assert result._values is None
        plaintext = decrypt(result) if isinstance(result, Ciphertext) else result
        assert decode(plaintext) == pytest.approx([op(0.125, 0.25)] * keys.params.slot_count, abs=1e-4)


def test_arithmetic_counter_names_and_reflected_dispatch(keys: CKKSKeySet):
    with keys.context():
        pt = Plaintext.const(0.125, level=1)
        ct = pt.to_ciphertext()
        with CallCounter() as counter:
            _ = pt + pt, ct + pt, pt - ct, 2 * ct
        assert counter.counts == {
            "Plaintext.__add__": 1,
            "Ciphertext.__add__": 1,
            "Ciphertext.__rsub__": 1,
            "Ciphertext.__rmul__": 1,
        }


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
def test_to_level_rejects_raising_and_preserves_noop(keys: CKKSKeySet, kind):
    with keys.context():
        value = kind.const(0.125, level=1)
        assert value.to_level(1) is value
        with pytest.raises(ValueError):
            value.to_level(2)
        with pytest.raises(ValueError):
            value.to_level(-1)


def test_plaintext_mul_at_level_zero_raises(keys: CKKSKeySet):
    with keys.context():
        a = random_plaintext(msg_bound=1, level=0)
        b = random_plaintext(msg_bound=1, level=0)
        with pytest.raises(InvalidParameterError):
            _ = a * b


@pytest.mark.parametrize("level,key_kind", [(None, "secret"), (0, "public")])
@pytest.mark.parametrize("tracking", [False, True])
def test_rerandomize(keys: CKKSKeySet, level, key_kind, tracking):
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        if level is not None:
            ct = ct.to_level(level)
    encryption_keys = keys if key_kind == "secret" else keys.public()
    with encryption_keys.context(track_plaintext_values=tracking):
        res1 = ct.rerandomize()
        res2 = ct.rerandomize()
        assert res1.values == pytest.approx(ct.values)
        assert res2.values == pytest.approx(ct.values)
        assert res1.level == res2.level == ct.level
        assert res1.ring == res2.ring == ct.ring
    with keys.context(track_plaintext_values=True):
        diff1 = ct - res1
        diff2 = ct - res2
        assert decode(decrypt(diff1)) == pytest.approx([0] * keys.params.slot_count, abs=1e-7)
        assert decode(decrypt(diff2)) == pytest.approx([0] * keys.params.slot_count, abs=1e-7)

        assert not res1.a.equals(res2.a)
        assert not res1.b.equals(res2.b)
        assert not ct.a.equals(res1.a)
        assert not ct.b.equals(res1.b)
        assert not ct.a.equals(res2.a)
        assert not ct.b.equals(res2.b)


def test_rerandomize_accepts_shared_randomness(keys: CKKSKeySet):
    seed = bytes(range(ChaChaRng.seed_size))
    with keys.context(track_plaintext_values=True):
        ct = random_ciphertext(msg_bound=1000)
        first = ct.rerandomize(rng=ChaChaRng(keys.device, seed))
        second = ct.rerandomize(rng=ChaChaRng(keys.device, seed))

        assert first.a.equals(second.a)
        assert first.b.equals(second.b)


def test_jit(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):

        @ren.jit
        def compute(a: Ciphertext, b: Ciphertext, c: Ciphertext, d: Ciphertext) -> Ciphertext:
            return a - b + c - d

        a = random_ciphertext(msg_bound=4)
        b = random_ciphertext(msg_bound=4)
        c = random_ciphertext(msg_bound=4)
        d = random_ciphertext(msg_bound=4)
        result: Ciphertext = compute(a, b, c, d)
        decoded = decode(decrypt(result))
        assert decoded == pytest.approx(result.values, abs=1e-4)

        # Test again with different inputs
        a2 = random_ciphertext(msg_bound=4)
        b2 = random_ciphertext(msg_bound=4)
        c2 = random_ciphertext(msg_bound=4)
        d2 = random_ciphertext(msg_bound=4)
        result2 = compute(a2, b2, c2, d2)
        decoded2 = decode(decrypt(result2), keys.params.slot_count)
        # Compute expected values from inputs (result2.values is stale from first call)
        expected2 = [a - b + c - d for a, b, c, d in zip(a2.values, b2.values, c2.values, d2.values)]
        assert decoded2 == pytest.approx(expected2, abs=1e-4)


def test_plaintext_tracking_enabled(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        values = [complex(i) for i in range(keys.params.slot_count)]
        ct = encrypt(encode(values, msg_bound=1000))
        assert ct.values == pytest.approx(values)


def test_context_runtime_flag_defaults(keys: CKKSKeySet):
    default_ctx = keys.context()
    assert default_ctx.track_plaintext_values is False
    assert default_ctx.strict_noise_bounds is False
    assert default_ctx.default_level == keys.params.default_level
    assert keys.context(strict_noise_bounds=True).strict_noise_bounds is True


def test_context_default_level_sets_fresh_level(keys: CKKSKeySet):
    lowered = keys.context(default_level=1)
    with lowered:
        assert encode(1, msg_bound=1).m.ring == keys.params.ring_at(1)
        assert encode(1, msg_bound=1, level=2).m.ring == keys.params.ring_at(2)
    with pytest.raises(InvalidParameterError, match="default_level"):
        keys.context(default_level=keys.params.highest_level + 1)


def test_key_material_does_not_expose_ckks_scale(keys: CKKSKeySet):
    with keys.context():
        ct = encrypt(encode(1, msg_bound=1))
        assert ct.scale == keys.params.scale(ct.ring)

        assert keys.pk is not None
        with pytest.raises(AttributeError):
            getattr(keys.pk.ct, "scale")

        assert keys.rlk is not None
        with pytest.raises(AttributeError):
            getattr(keys.rlk.digits[0], "scale")


def test_noise_overflow_policy_strict_raises(keys: CKKSKeySet):
    strict_ctx = keys.context(strict_noise_bounds=True)
    with strict_ctx:
        ct = random_ciphertext(msg_bound=1)
        bad_noise = CKKSNoiseInfo(noise_bound=0, noise_heuristic_bound=0, msg_bound=ct.ring.q)
        with pytest.raises(OverflowError, match="message bound exceeds modulus q/2"):
            _ = Ciphertext(a=ct.a, b=ct.b, noise_info=bad_noise, _values=None)


def test_noise_overflow_policy_permissive_saturates_to_unbounded(keys: CKKSKeySet):
    permissive_ctx = keys.context()
    bad_noise = CKKSNoiseInfo(noise_bound=0, noise_heuristic_bound=0, msg_bound=permissive_ctx.params.highest_ring.q)
    with permissive_ctx:
        ct = Ciphertext(
            a=Ciphertext.zero().a,
            b=Ciphertext.zero().b,
            noise_info=bad_noise,
            _values=None,
        )
    assert ct.noise_info.msg_bound.is_infinite()
    assert ct.noise_info.noise_bound.is_infinite()
    assert ct.noise_info.noise_heuristic_bound.is_infinite()


def _assert_equal_bound(a: Bound, b: Bound, *, rel: float = 1e-6, abs_tol: float = 1e-12):
    max_abs = max(abs(a), abs(b))
    if max_abs < abs_tol / 2:
        return
    # A difference can be negative, so subtract in the unconstrained BigFloat domain.
    diff = abs(BigFloat(a) - b)
    if diff < abs_tol:
        return
    if diff / max_abs < rel:
        return
    raise AssertionError(f"Bounds are not equal: {a} != {b}")


def assert_equal(a: CKKSNoiseInfo, b: CKKSNoiseInfo, *, rel: float = 1e-6):
    _assert_equal_bound(a.noise_bound, b.noise_bound, rel=rel)
    _assert_equal_bound(a.noise_heuristic_bound, b.noise_heuristic_bound, rel=rel)
    _assert_equal_bound(a.msg_bound, b.msg_bound, rel=rel)


@pytest.mark.parametrize("flt", [float, BigFloat])
def test_noise_operations(flt: Callable[[float], float | BigFloat]):
    noise1 = CKKSNoiseInfo(noise_bound=flt(1), noise_heuristic_bound=flt(1), msg_bound=3)
    noise2 = CKKSNoiseInfo(noise_bound=flt(2), noise_heuristic_bound=flt(2), msg_bound=2)

    expected_sum = CKKSNoiseInfo(noise_bound=flt(3), noise_heuristic_bound=flt(math.sqrt(5)), msg_bound=5)
    expected_product = CKKSNoiseInfo(noise_bound=flt(10), noise_heuristic_bound=flt(10), msg_bound=6)
    expected_scaled = CKKSNoiseInfo(noise_bound=flt(2), noise_heuristic_bound=flt(2), msg_bound=6)

    assert_equal(noise1 + noise2, expected_sum)
    assert_equal(noise1 * noise2, expected_product)
    assert_equal(noise1 * -2, expected_scaled)
    assert_equal(-2 * noise1, expected_scaled)
    assert_equal(noise1 * 0, CKKSNoiseInfo.zero())
    assert_equal(CKKSNoiseInfo.unbounded() * 0, CKKSNoiseInfo.zero())

    # TODO: test dependent addition


def test_encryption_decryption_noise(keys: CKKSKeySet):
    with keys.context():
        values = random_values_with_norm(keys.params.slot_count, norm=1000)
        pt = encode(values, msg_bound=1000)
        _assert_equal_bound(pt.noise_info.msg_bound, Bound(1000 * keys.params.scale(pt.ring)))

        ct = encrypt(pt)
        expected_encryption_noise = CKKSNoiseInfo(
            noise_bound=2 * keys.params.highest_ring.n * keys.params.noise_dist.coeff_inf_norm_bound() / math.pi,
            noise_heuristic_bound=math.sqrt(2 * keys.params.highest_ring.n / 3) * keys.params.noise_dist.coeff_inf_norm_bound(),
            msg_bound=0,
        )
        expected = expected_encryption_noise + pt.noise_info
        assert_equal(ct.noise_info, expected)

        dec = decrypt(ct)
        assert_equal(dec.noise_info, expected)


@pytest.mark.parametrize("prec", [53, 180])
def test_scalar_encoding_noise(keys: CKKSKeySet, prec: int):
    with keys.context():
        real = encode(0.5, msg_bound=5, prec=prec)
        complex_value = encode(1 + 1j, msg_bound=5, prec=prec)
        assert real.noise_info == complex_value.noise_info
        input_error = 5 * real.scale * 2**-53 if prec == 53 else 0
        assert_equal(
            real.noise_info,
            CKKSNoiseInfo(
                noise_bound=math.sqrt(0.5) + input_error,
                noise_heuristic_bound=0.5 + input_error,
                msg_bound=5 * real.scale,
            ),
        )


def test_ciphertext_has_no_truth_value(keys: CKKSKeySet):
    with keys.context():
        for ciphertext in (Ciphertext.zero(), Ciphertext.one()):
            with pytest.raises(TypeError, match="truth value"):
                bool(ciphertext)


@pytest.mark.parametrize("sign", [-1, 1])
def test_scalar_noise_covers_discarded_imaginary_component(keys: CKKSKeySet, sign):
    with keys.context(), flint_precision(200):
        scale = keys.params.scale(keys.params.default_level)
        value = complex(0.49 / float(scale), sign * 0.49 / float(scale))
        plaintext = encode(value, msg_bound=1)
        actual = encoding._decode(plaintext.m, scale=plaintext.scale, prec=180)[0]
        error = abs((actual - acb(value)) * arb(str(scale)))
        assert error.abs_upper() <= arb(str(plaintext.noise_info.noise_bound))


def test_addition_subtraction_noise(keys: CKKSKeySet):
    """Test noise info is correctly combined for add/sub operations."""
    with keys.context():
        a_values = random_values_with_norm(keys.params.slot_count, norm=1000)
        b_values = random_values_with_norm(keys.params.slot_count, norm=1000)
        a = encrypt(encode(a_values, msg_bound=1000))
        b = encrypt(encode(b_values, msg_bound=1000))

        expected_noise = a.noise_info + b.noise_info
        assert_equal((a + b).noise_info, expected_noise)
        assert_equal((a - b).noise_info, expected_noise)


def test_noise_bounds_are_eager_bigfloats():
    bound = Bound(0.5)
    noise = CKKSNoiseInfo(
        noise_bound=bound,
        noise_heuristic_bound=0.5,
        msg_bound=0.5,
    )
    assert noise.noise_bound is bound
    assert isinstance(noise.noise_heuristic_bound, Bound)
    assert isinstance(noise.msg_bound, Bound)


def test_noise_arithmetic_preserves_huge_finite_bounds():
    huge = BigFloat(2) ** 2000
    big_noise = CKKSNoiseInfo(noise_bound=huge, noise_heuristic_bound=huge, msg_bound=huge)
    bigger_noise = big_noise + big_noise
    expected_bigger_noise = CKKSNoiseInfo(
        noise_bound=huge * 2,
        noise_heuristic_bound=huge * BigFloat(2).sqrt(),
        msg_bound=huge * 2,
    )

    assert isinstance(bigger_noise.noise_bound, Bound)
    assert isinstance(bigger_noise.noise_heuristic_bound, Bound)
    assert isinstance(bigger_noise.msg_bound, Bound)
    assert not bigger_noise.is_unbounded
    assert_equal(bigger_noise, expected_bigger_noise)


@pytest.mark.parametrize(
    ("noise_bound", "noise_heuristic_bound", "msg_bound"),
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
)
def test_noise_bounds_reject_negative_inputs(noise_bound: int, noise_heuristic_bound: int, msg_bound: int):
    with pytest.raises(ValueError, match="bound must be non-negative"):
        CKKSNoiseInfo(noise_bound=noise_bound, noise_heuristic_bound=noise_heuristic_bound, msg_bound=msg_bound)


def test_bound_arithmetic_rejects_negative_results():
    with pytest.raises(ValueError, match="bound must be non-negative"):
        _ = Bound(1) - 2


def test_decode_divides_before_downcasting_scale():
    huge = 10**400
    ring = RingSpec(n=2, moduli=tuple(select_moduli(n=2, num_primes=45, bit_length=30)), dtype=dtypes.uint32)
    assert encoding._decode(Poly.const(huge, ring), scale=BigFloat(huge), prec=53)[0] == pytest.approx(1 + 0j)


def test_decode_divides_before_underflowing_scale():
    poly = Poly.zero(params.ring_at(0))
    assert encoding._decode(poly, scale=BigFloat("1e-400"), prec=53)[0] == 0j


def test_generated_keys_use_ntt_representation(keys: CKKSKeySet):
    assert keys.sk is not None and keys.pk is not None and keys.rlk is not None
    polys = [keys.sk, keys.pk.ct.a, keys.pk.ct.b]
    for key in (keys.rlk, *keys.galois_keys.values()):
        polys.extend(poly for digit in key.digits for poly in (digit.a, digit.b))

    assert all(poly.domain is Domain.NTT for poly in polys)
