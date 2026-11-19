"""Tests for arithmetic operations in ren.ckks.arithmetic_ops."""

import math
from dataclasses import replace

import numpy as np
import pytest
from flint import acb, arb

from ren.bigfloat import BigFloat
from ren.dtype import dtypes
from ren.engine.realize import call_op
from ren.engine.schedule import schedule
from ren.ir import Node, Op
from ren.poly import Poly
from ren.random import set_global_rng_seed
from ren.ckks import (
    CKKSKeySet,
    CKKSLevel,
    CKKSParams,
    Ciphertext,
    Plaintext,
    decode,
    decrypt,
    encode,
    encrypt,
    keygen,
)
from ren.ckks.arithmetic_ops import _sum_rotated_direct, _sum_rotated_horner, horner_step, sum_rotated
from ren.ckks.metadata import CKKSNoiseInfo
from ren.ckks import encoding
from ren.ckks.encoding import flint_precision
from ren.ckks.utils import select_moduli
from ren.ckks.errors import InvalidParameterError
from tests.utils import random_ciphertext, random_complex_with_norm, random_values_with_norm
from tests.ckks.params import non_prefix_params, small_params


@pytest.fixture(scope="module")
def non_prefix_keys():
    params = replace(non_prefix_params(), boot_params=None, bootstrap_base_level=None)
    return keygen(params).realize()


@pytest.mark.parametrize(
    "terms, expected",
    [
        ([], CKKSNoiseInfo.zero()),
        ([CKKSNoiseInfo(5, 3, 1)], CKKSNoiseInfo(5, 3, 1)),
        ([CKKSNoiseInfo(5, 3, 1), CKKSNoiseInfo(6, 4, 2), CKKSNoiseInfo(7, 12, 3)], CKKSNoiseInfo(18, 13, 6)),
        ([CKKSNoiseInfo(5, 3, 1), CKKSNoiseInfo(math.inf, 0, 0)], CKKSNoiseInfo.unbounded()),
        ([CKKSNoiseInfo(5, 3, 1), CKKSNoiseInfo(0, math.inf, 0)], CKKSNoiseInfo.unbounded()),
        ([CKKSNoiseInfo(5, 3, 1), CKKSNoiseInfo(0, 0, math.inf)], CKKSNoiseInfo.unbounded()),
    ],
)
def test_noise_sum(terms: list[CKKSNoiseInfo], expected: CKKSNoiseInfo):
    assert CKKSNoiseInfo.sum(iter(terms)) == expected


@pytest.fixture
def high_precision_params():
    moduli = select_moduli(n=8, num_primes=9, bit_length=30)
    return CKKSParams.from_levels(
        n=8,
        dtype=dtypes.uint32,
        levels=[
            CKKSLevel(moduli=moduli[:5], scale=BigFloat(2) ** 240 / math.prod(moduli[5:])),
            CKKSLevel(moduli=moduli, scale=BigFloat(2) ** 120),
        ],
        ks_primes=select_moduli(n=8, num_primes=1, bit_length=30, exclude=set(moduli)),
        ks_digit_sizes=(len(moduli),),
        log_range=0,
        security=None,
    )


def test_dot_raw_cancellation_is_independent_of_tracking(high_precision_params):
    params = high_precision_params
    with params.context(track_plaintext_values=True):
        zero = Plaintext.zero(level=1)
    outputs = []
    for tracking in (False, True):
        with params.context(track_plaintext_values=tracking), flint_precision(53):
            result = Ciphertext.dot([zero, 2**120, 1, -(2**120)], [1, 1, 1, 1])
            assert (result._values is not None) == tracking
            assert decode(Plaintext(result.b, noise_info=result.noise_info)) == pytest.approx([1] * params.slot_count, abs=1e-8)
            outputs.append(result.b)
    assert outputs[0].equals(outputs[1])


def test_dot_uses_configured_precision_for_raw_and_mixed_products(high_precision_params):
    from flint import arb

    with flint_precision(220):
        correction = arb(2) ** -80
        raw = [acb(1 + correction)] * high_precision_params.slot_count
        expected = acb(arb(1) / 4 + correction * 5 / 4)
    with high_precision_params.context(encoding_precision_bits=180), flint_precision(53):
        value = Plaintext.const(0.25, level=1)
        result = Ciphertext.dot([value, raw, -1], [raw, 1, 1])
        actual = encoding._decode(result.b, scale=result.scale, prec=180)
    with flint_precision(220):
        assert all(abs(slot - expected).abs_upper() < arb(2) ** -110 for slot in actual)


def test_dot_raw_constant_bound_covers_encoded_value(high_precision_params):
    from flint import arb

    with flint_precision(220):
        value = acb(1 + arb(2) ** -80)
    with high_precision_params.context(encoding_precision_bits=180):
        result = Ciphertext.dot([Plaintext.zero(level=1), value], [1, 1])
        coefficient = result.b.to_centered_list()[0]
        assert abs(coefficient) <= result.noise_info.msg_bound + result.noise_info.noise_bound


@pytest.mark.parametrize("vector", [False, True])
def test_dot_compensated_encoding_bounds_original_reference(high_precision_params, vector):
    params = high_precision_params
    moduli = params.highest_ring.moduli
    extra = select_moduli(n=params.n, num_primes=1, bit_length=30, exclude=set(moduli + params.ks_primes))
    params = replace(
        params,
        levels=(
            CKKSLevel(moduli=moduli[:4], scale=params.scale(1) ** 2 / math.prod(moduli[4:])),
            params.levels[1],
            CKKSLevel(moduli=(*moduli, *extra), scale=10 * params.scale(1)),
        ),
        ks_digit_sizes=(len(moduli) + 1,),
    )
    with params.context(track_plaintext_values=True):
        one = encode(1, msg_bound=1, level=2, prec=180)
        raw = [1.5] * params.slot_count if vector else 1.5
        result = Ciphertext.dot([one, Plaintext.zero(level=1)], [raw, 0])
        expected = BigFloat(1.5) * result.scale
        error = abs(BigFloat(result.b.to_centered_list()[0]) - expected)
        assert result.values == pytest.approx([1.5] * params.slot_count)
        assert error <= result.noise_info.noise_bound


@pytest.mark.parametrize("prec", [53, 100])
def test_dot_broadcast_raw_partner_matches_scalar(test_params: CKKSParams, prec):
    with test_params.context(encoding_precision_bits=prec):
        value = Plaintext.const(0.25, level=2)
        expected = Ciphertext.dot([value], [0.5])
        actual = Ciphertext.dot([value], [np.broadcast_to(0.5, (test_params.slot_count,))])
        assert actual.a.equals(expected.a)
        assert actual.b.equals(expected.b)
        assert actual.noise_info == expected.noise_info


def test_dot_raw_vector_cancellation_preserves_reference_precision(test_params: CKKSParams):
    from flint import acb, arb

    with test_params.context(track_plaintext_values=True), flint_precision(120):
        correction = arb(2) ** -80
        zero = Plaintext.zero(level=2)
        raw = [acb(1 + correction), acb(-1 - correction)]
        with flint_precision(53):
            result = Ciphertext.dot([zero, raw, [-1, 1]], [1, 1, 1])
        raw[:] = [acb(7), acb(7)]
        assert result._values is not None
        for actual, expected in zip(result._values[:2], [correction, -correction], strict=True):
            assert abs(actual - expected).abs_upper() < arb(2) ** -90
        assert all(value == acb(0) for value in result._values[2:])


def test_dot_pads_raw_sequences_and_keeps_a_snapshot(test_params: CKKSParams):
    with test_params.context(track_plaintext_values=True):
        value = Plaintext.const(0.5, level=2)
        raw = np.array([0.25, -0.5])
        result = Ciphertext.dot([raw, value], [value, (0.25, -0.5)])
        raw[:] = 7
        expected = [0.25, -0.5] + [0] * (test_params.slot_count - 2)
        assert result.values == pytest.approx(expected)
        assert decode(Plaintext(result.b, noise_info=result.noise_info)) == pytest.approx(expected, abs=1e-8)


def test_dot_and_sum_propagate_missing_tracking(test_params: CKKSParams):
    with test_params.context(track_plaintext_values=True):
        tracked = Ciphertext.const(0.25, level=2)
        missing = Ciphertext.const(0.5, level=2).replace(_values=None)
        for result in (Ciphertext.sum([tracked, missing]), Ciphertext.dot([tracked, missing], [1, 1])):
            assert result._values is None
            assert decode(Plaintext(result.b, noise_info=result.noise_info)) == pytest.approx([0.75] * test_params.slot_count, abs=1e-8)
        assert tracked.values == pytest.approx([0.25] * test_params.slot_count)


@pytest.mark.parametrize("value", ["1", None, object()])
def test_dot_rejects_unsupported_slot_inputs(test_params: CKKSParams, value):
    with test_params.context(), pytest.raises(TypeError):
        Ciphertext.dot([Plaintext.one(level=2)], [value])


def test_dot_rejects_invalid_slot_shapes(test_params: CKKSParams):
    with test_params.context():
        for values in (np.array([[1, 2]]), np.zeros(test_params.slot_count + 1)):
            with pytest.raises(InvalidParameterError):
                Ciphertext.dot([Plaintext.one(level=2)], [values])


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
def test_dot_keeps_noise_from_input_conversion(kind):
    with non_prefix_params().context():
        high = kind.const(0.25, level=3)
        high = replace(high, noise_info=CKKSNoiseInfo(noise_bound=3, noise_heuristic_bound=2, msg_bound=high.noise_info.msg_bound))
        low = Plaintext.const(-0.125, level=2)
        partners = [Plaintext.const(0.5, level=2), Plaintext.const(0.25, level=2)]
        result = Ciphertext.dot([high, low], partners)
        explicitly_aligned = Ciphertext.dot([high.to_level(2), low], partners)

        assert result.noise_info.noise_bound.is_finite()
        assert result.noise_info.noise_heuristic_bound.is_finite()
        assert result.noise_info == explicitly_aligned.noise_info


def test_dot_preserves_precision_in_overlapping_rings():
    moduli = select_moduli(n=1024, num_primes=6, bit_length=60)
    ks_primes = select_moduli(n=1024, num_primes=3, bit_length=59, exclude=set(moduli))
    bases = (moduli[:2], moduli[1:5], moduli[2:6], moduli)
    q0, q1, q2, q3 = map(math.prod, bases)
    scale1 = BigFloat(q1).sqrt() / 4
    scale0 = scale1**2 * q0 / q1
    scale2 = (scale1 * q2 / q1).sqrt()
    scale3 = (scale2 * q3 / q2).sqrt()
    params = CKKSParams.from_levels(
        n=1024,
        dtype=dtypes.uint64,
        levels=[CKKSLevel(moduli=basis, scale=scale) for basis, scale in zip(bases, (scale0, scale1, scale2, scale3))],
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2, 2),
        log_range=0,
        security=None,
    )
    with params.context(), flint_precision(256):
        # Exact encoded constants isolate conversion error from encryption and
        # encoding noise. Level 3 has more precision than level 2, despite both
        # rings having ample room for the unconverted product.
        x_coeff = round(scale3 * (BigFloat(1) / 8 + BigFloat(2) ** -75))
        y_coeff = round(scale2 / 4)
        x = Plaintext(Poly.const(x_coeff, params.ring_at(3)), noise_info=CKKSNoiseInfo(0, 0, x_coeff))
        y = Plaintext(Poly.const(y_coeff, params.ring_at(2)), noise_info=CKKSNoiseInfo(0, 0, y_coeff))
        expected = acb(arb(x_coeff) / arb(str(scale3)) * arb(y_coeff) / arb(str(scale2)))
        zero = Plaintext.zero(level=1)
        result = Ciphertext.dot([x, zero], [y, zero])
        actual = encoding._decode(result.b, scale=result.scale, prec=100)[0]
        assert abs(actual - expected) < arb(2) ** -80


@pytest.mark.parametrize("scale2", [BigFloat(2) ** 20, BigFloat(3) * BigFloat(2) ** 38])
@pytest.mark.parametrize("partner_kind", [Plaintext, float, list, np.ndarray])
def test_dot_preserves_values_when_product_conversion_is_coarse(scale2, partner_kind):
    moduli = select_moduli(n=1024, num_primes=7, bit_length=60)
    ks_primes = select_moduli(n=1024, num_primes=3, bit_length=59, exclude=set(moduli))
    bases = (moduli[1:4], moduli[1:5], moduli[2:6], moduli)
    q0, q1, q2, q3 = map(math.prod, bases)
    scale1 = scale2**2 * q1 / q2
    scale0 = scale1**2 * q0 / q1
    scale3 = (scale2 * q3 / q2).sqrt()
    params = CKKSParams.from_levels(
        n=1024,
        dtype=dtypes.uint64,
        levels=[CKKSLevel(moduli=basis, scale=scale) for basis, scale in zip(bases, (scale0, scale1, scale2, scale3))],
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2, 2, 1),
        log_range=0,
        security=None,
    )
    with params.context():
        x = Plaintext.const(0.125, level=3)
        if partner_kind is Plaintext:
            y = Plaintext.const(0.25, level=2)
        elif partner_kind is float:
            y = 0.25
        elif partner_kind is list:
            y = [0.25 + 0j] * params.slot_count
        else:
            y = np.full(params.slot_count, 0.25)
        zero = Plaintext.zero(level=1)
        result = Ciphertext.dot([x, zero], [y, zero])
        assert decode(Plaintext(result.b, noise_info=result.noise_info), length=1)[0] == pytest.approx(0.03125, abs=2**-30)


def test_raw_dot_preserves_output_precision_without_extra_rescales(non_prefix_keys: CKKSKeySet):
    with non_prefix_keys.context(track_plaintext_values=True):
        values = [0.25 + 0.125j, -0.5]
        ct = encrypt(encode(values, msg_bound=1, level=18)).realize()
        zero = Plaintext.zero(level=4)
        result = Ciphertext.dot([ct, zero], [[0.5 + 0j, -0.25 + 0j], zero])
        linear, _ = schedule(Node.sink(result.a.node, result.b.node))

        assert sum(call_op(call) is Op.RESCALE for call in linear.src) == 2
        assert decode(decrypt(result), len(values)) == pytest.approx([0.125 + 0.0625j, 0.125], abs=2**-40)


def test_sum_non_prefix_ciphertexts_at_mixed_levels(non_prefix_keys: CKKSKeySet):
    with non_prefix_keys.context(track_plaintext_values=True):
        values = [0.125 + 0.0625j, -0.25j, -0.125, 0.25]
        cts = [encrypt(encode(values, msg_bound=0.5, level=level)) for level in (3, 2, 1)]
        result = Ciphertext.sum(cts)
        expected = np.asarray(values) * len(cts)

        assert result.level == 1
        assert decode(decrypt(result), len(values)) == pytest.approx(expected, abs=1e-7)
        assert result.values is not None
        assert result.values[: len(values)] == pytest.approx(expected)


@pytest.mark.parametrize(
    "left_kind,left_level,right_kind,right_level",
    [
        pytest.param(np.ndarray, 1, np.ndarray, 1, id="raw-raw"),
        pytest.param(float, 1, Plaintext, 3, id="scalar-pt"),
        pytest.param(Plaintext, 3, float, 1, id="pt-scalar"),
        pytest.param(np.ndarray, 1, Ciphertext, 3, id="vector-ct"),
        pytest.param(Ciphertext, 3, np.ndarray, 1, id="ct-vector"),
        pytest.param(Plaintext, 1, Plaintext, 1, id="pt-pt-same"),
        pytest.param(Plaintext, 3, Plaintext, 1, id="pt-pt-high-low"),
        pytest.param(Plaintext, 3, Plaintext, 2, id="pt-pt-both-high"),
        pytest.param(Plaintext, 1, Ciphertext, 3, id="pt-ct-low-high"),
        pytest.param(Ciphertext, 3, Plaintext, 1, id="ct-pt-high-low"),
        pytest.param(Plaintext, 3, Ciphertext, 2, id="pt-ct-both-high"),
        pytest.param(Ciphertext, 1, Ciphertext, 1, id="ct-ct-same"),
        pytest.param(Ciphertext, 3, Ciphertext, 1, id="ct-ct-high-low"),
        pytest.param(Ciphertext, 3, Ciphertext, 2, id="ct-ct-both-high"),
    ],
)
def test_dot_non_prefix_at_mixed_levels(non_prefix_keys: CKKSKeySet, left_kind, left_level, right_kind, right_level):
    with non_prefix_keys.context(track_plaintext_values=True):
        values = (
            np.asarray([0.015625 + 0.0078125j, -0.015625j, -0.03125, 0.015625]),
            np.asarray([-0.0078125j, 0.03125, 0.015625j, -0.015625]),
        )
        operands = []
        expected_values = []
        for kind, level, vector in zip((left_kind, right_kind), (left_level, right_level), values):
            if kind is float:
                operand = 0.015625
                expected_values.append(np.full(len(vector), operand))
            elif kind is np.ndarray:
                operand = vector
                expected_values.append(vector)
            else:
                plaintext = encode(vector, msg_bound=0.0625, level=level)
                operand = plaintext if kind is Plaintext else encrypt(plaintext)
                expected_values.append(vector)
            operands.append(operand)
        # Keep the target at level 0, including when both tested inputs are higher.
        result = Ciphertext.dot([operands[0], Plaintext.zero(level=1)], [operands[1], 1.0])
        expected = expected_values[0] * expected_values[1]
        assert result.level == 0
        assert decode(decrypt(result), len(expected)) == pytest.approx(expected, abs=1e-7)
        assert result.values is not None
        assert result.values[: len(expected)] == pytest.approx(expected)


@pytest.mark.parametrize("levels", [(3, 2, 1), (18, 5, 1)])
def test_dot_non_prefix_accumulates_distinct_mixed_terms(non_prefix_keys: CKKSKeySet, levels: tuple[int, int, int]):
    with non_prefix_keys.context(track_plaintext_values=True):
        x = np.asarray([0.015625 + 0.0078125j, -0.015625j, -0.03125, 0.015625])
        y = np.asarray([-0.0078125j, 0.03125, 0.015625j, -0.015625])
        x_high = encode(x, msg_bound=0.0625, level=levels[0])
        y_middle = encode(y, msg_bound=0.0625, level=levels[1])
        x_low = encode(x / 2, msg_bound=0.0625, level=levels[2])
        left = [x, x_high, encrypt(y_middle), encrypt(x_high), encrypt(x_low), encrypt(x_high), encrypt(x_high)]
        right = [encrypt(y_middle), encrypt(y_middle), x_high, encrypt(y_middle), 0.0078125, x_high, encrypt(x_high)]
        result = Ciphertext.dot(left, right)
        expected = 4 * x * y + (x / 2) * 0.0078125 + 2 * x**2
        assert result.level == 0
        assert decode(decrypt(result), len(expected)) == pytest.approx(expected, abs=1e-7)
        assert result.values is not None
        assert result.values[: len(expected)] == pytest.approx(expected)


def test_sum_ciphertexts(keys: CKKSKeySet):
    """Test sum_ciphertexts with ciphertexts at mixed levels."""
    with keys.context(track_plaintext_values=True):
        max_level = keys.params.highest_level
        cts = [random_ciphertext(msg_bound=10).to_level(max_level - (i % 3)) for i in range(10)]

        result = Ciphertext.sum(cts)

        # Verify correctness by comparing with expected sum
        assert cts[0].values is not None, "Test requires plaintext tracking"
        expected_sum = [0j] * len(cts[0].values)
        for ct in cts:
            assert ct.values is not None
            for i, val in enumerate(ct.values):
                expected_sum[i] += val

        decoded = decode(decrypt(result))
        assert decoded == pytest.approx(expected_sum, abs=3e-8)

        # Result lands at the minimum input level (no extra level consumed).
        assert result.level == min(ct.level for ct in cts)


@pytest.mark.parametrize("n_elements", [1, 4, 9])
def test_multiply_ciphertexts(keys: CKKSKeySet, n_elements: int):
    """Test multiply_ciphertexts function with varying numbers of ciphertexts."""
    with keys.context(track_plaintext_values=True):
        cts = [random_ciphertext(msg_bound=2) for _ in range(n_elements)]
        result = Ciphertext.multiply(cts)

        # Verify correctness
        decoded = decode(decrypt(result))
        assert decoded == pytest.approx(result.values, abs=1e-1)

        # Verify output level is ceil(log2(count)) levels below the input level
        input_level = cts[0].level
        expected_levels_consumed = math.ceil(math.log2(n_elements)) if n_elements > 1 else 0
        expected_output_level = input_level - expected_levels_consumed
        assert result.level == expected_output_level


def test_dot(keys: CKKSKeySet):
    """Test Ciphertext.dot across all 64 (left_kind, right_kind) combinations.

    There are 8 element kinds, each at a fixed type / level:

    * kind 0: ``complex`` scalar
    * kind 1: ``list[complex]`` per-slot row
    * kind 2: :class:`Plaintext` at level ``l``
    * kind 3: :class:`Plaintext` at level ``l - 1``
    * kind 4: :class:`Plaintext` at level ``l - 2``
    * kind 5: :class:`Ciphertext` at level ``l``
    * kind 6: :class:`Ciphertext` at level ``l - 1``
    * kind 7: :class:`Ciphertext` at level ``l - 2``

    The test builds a length-64 dot product where the ``(8 * lk + rk)``-th
    pair has ``left`` of kind ``lk`` and ``right`` of kind ``rk``, runs a
    single :meth:`Ciphertext.dot` call, and checks the decrypted result
    against the per-slot sum of products computed in the clear.
    """
    with keys.context(track_plaintext_values=True):
        max_level = keys.params.highest_level
        slot_count = keys.params.slot_count

        def make_element(kind: int) -> tuple[complex | list[complex] | Plaintext | Ciphertext, list[complex]]:
            """Return ``(element, per_slot_values)`` for the given kind."""
            if kind == 0:  # complex scalar
                c = random_complex_with_norm(norm=2)
                return c, [c] * slot_count
            if kind == 1:  # list[complex]
                arr = random_values_with_norm(slot_count, norm=2)
                return arr, arr
            if kind in (2, 3, 4):  # Plaintext at l, l-1, l-2
                level = max_level - (kind - 2)
                arr = random_values_with_norm(slot_count, norm=2)
                return encode(arr, msg_bound=2, level=level), arr
            if kind in (5, 6, 7):  # Ciphertext at l, l-1, l-2
                level = max_level - (kind - 5)
                ct = random_ciphertext(msg_bound=2, level=level)
                assert ct.values is not None
                return ct, list(ct.values)
            raise ValueError(f"unknown kind: {kind}")

        left: list[complex | list[complex] | Plaintext | Ciphertext] = []
        right: list[complex | list[complex] | Plaintext | Ciphertext] = []
        # Per-pair user-facing per-slot values, used to compute the expected
        # dot product in the clear without going through ciphertext internals.
        left_values: list[list[complex]] = []
        right_values: list[list[complex]] = []

        for left_kind in range(8):
            for right_kind in range(8):
                l_elem, l_vals = make_element(left_kind)
                r_elem, r_vals = make_element(right_kind)
                left.append(l_elem)
                right.append(r_elem)
                left_values.append(l_vals)
                right_values.append(r_vals)

        result = Ciphertext.dot(left, right)

        # Expected per-slot sum of pairwise products.
        expected = [0j] * slot_count
        for lvs, rvs in zip(left_values, right_values):
            for j, (lv, rv) in enumerate(zip(lvs, rvs)):
                expected[j] += lv * rv

        decoded = decode(decrypt(result))
        assert decoded == pytest.approx(expected, abs=3e-8)

        # The minimum input level among ciphertexts is ``l - 2``; the dot
        # product consumes one level via its final rescale, so the result
        # lands at ``l - 3``.
        assert result.level == max_level - 3


@pytest.fixture(scope="module")
def rotation_keys() -> CKKSKeySet:
    set_global_rng_seed(42)
    return keygen(small_params(), rotations={-3, 3}).realize()


def rotation_inputs(count, slot_count):
    # Distinct slot patterns make each rotation observable on its own ciphertext.
    values = [[(i * j) % slot_count + 0.1 * i for j in range(slot_count)] for i in range(1, count + 1)]
    return values, [encrypt(encode(v, msg_bound=slot_count + 1)) for v in values]


def sum_rotated_values(values, rotations):
    slot_count = len(values[0])
    # Positive steps rotate slots to the left.
    return [sum(v[(j + r) % slot_count] for v, r in zip(values, rotations, strict=True)) for j in range(slot_count)]


def test_sum_rotated_horner(rotation_keys: CKKSKeySet):
    with rotation_keys.context():
        slot_count = rotation_keys.params.slot_count
        # The offsets form a progression with step 3, and only the +3 and -3 keys are available.
        rotations = [-6, -3, 0, 3, 6]
        values, cts = rotation_inputs(len(rotations), slot_count)
        operands = list(zip(rotations, cts))

        result = sum_rotated(operands)

        expected = sum_rotated_values(values, rotations)
        assert decode(decrypt(result)) == pytest.approx(expected, abs=1e-6)

        for wrapped_step in (slot_count + 3, -slot_count - 3):
            wrapped_rotations = [0, wrapped_step, 2 * wrapped_step]
            wrapped_operands = list(zip(wrapped_rotations, cts[:3]))
            wrapped_result = sum_rotated(wrapped_operands)
            wrapped_expected = sum_rotated_values(values[:3], wrapped_rotations)
            assert decode(decrypt(wrapped_result)) == pytest.approx(wrapped_expected, abs=1e-6)

        with pytest.raises(ValueError, match="not a multiple of step"):
            _sum_rotated_horner(operands, step=5)


@pytest.mark.parametrize("direction", [1, -1])
def test_sum_rotated_with_one_sided_keys(direction: int):
    params = small_params()
    slot_count = params.slot_count
    step = direction * slot_count // 4
    keys = keygen(params, rotations={step})
    with keys.context():
        rotations = [0, step, 2 * step, 3 * step, step]
        values = [j / slot_count for j in range(slot_count)]
        ct = encrypt(encode(values, msg_bound=1))
        result = sum_rotated([(rotation, ct) for rotation in rotations])
        expected = [sum(values[(j + rotation) % slot_count] for rotation in rotations) for j in range(slot_count)]
        assert decode(decrypt(result)) == pytest.approx(expected, abs=1e-6)


def test_sum_rotated_direct():
    keys = keygen(small_params(), rotations={-3, 3, 6})
    with keys.context(track_plaintext_values=True):
        slot_count = keys.params.slot_count
        # slot_count + 3 repeats rotation 3 on a different ciphertext.
        rotations = [-3, 0, 3, 6, slot_count + 3]
        values, cts = rotation_inputs(len(rotations), slot_count)
        operands = list(zip(rotations, cts))

        result = sum_rotated(operands)

        expected = sum_rotated_values(values, rotations)
        assert decode(decrypt(result)) == pytest.approx(expected, abs=1e-6)

        # Result stays at the input level: the shared rescale only strips the
        # keyswitch extension, not a ciphertext modulus.
        assert result.level == cts[0].level

        assert result.values == pytest.approx(expected, abs=1e-6)

        with pytest.raises(RuntimeError, match="rotation key"):
            _sum_rotated_direct([(5, cts[0])])

        zero_only = sum_rotated([(0, cts[0]), (0, cts[1])])
        expected_zero = [values[0][j] + values[1][j] for j in range(slot_count)]
        assert decode(decrypt(zero_only)) == pytest.approx(expected_zero, abs=1e-6)


def test_sum_rotated_fallback(rotation_keys: CKKSKeySet, monkeypatch):
    import ren.ckks.arithmetic_ops as arithmetic_ops

    def unexpected_strategy(*args, **kwargs):
        raise AssertionError("sum_rotated should not dispatch to a strategy helper here")

    monkeypatch.setattr(arithmetic_ops, "_sum_rotated_direct", unexpected_strategy)
    monkeypatch.setattr(arithmetic_ops, "_sum_rotated_horner", unexpected_strategy)
    with rotation_keys.context():
        slot_count = rotation_keys.params.slot_count
        # [0, 3, 9] has no constant step, and rotation 9 requires three +3 hops.
        rotations = [0, 3, 9]
        values, cts = rotation_inputs(len(rotations), slot_count)
        operands = list(zip(rotations, cts))

        result = sum_rotated(operands)

        expected = sum_rotated_values(values, rotations)
        assert decode(decrypt(result)) == pytest.approx(expected, abs=1e-6)

        # max_ks caps the keyswitch hops of the fallback's chained rotations.
        with pytest.raises(RuntimeError, match="max_ks"):
            sum_rotated(operands, max_ks=1)

        single = sum_rotated([(3, cts[0])])
        expected_single = [values[0][(j + 3) % slot_count] for j in range(slot_count)]
        assert decode(decrypt(single)) == pytest.approx(expected_single, abs=1e-6)

        with pytest.raises(ValueError, match="non-empty"):
            sum_rotated([])


@pytest.mark.parametrize(
    "rotations,expected",
    [
        ((0, 4, 8, 12), 4),
        ((-8, -4, 0, 4), 4),
        ((8, 12), 4),
        ((0, 3, 9), None),  # not a progression
        ((1, 3, 5), None),  # a progression, but not of multiples of its step
        ((5,), None),
        ((), None),
    ],
)
def test_horner_step(rotations: tuple[int, ...], expected: int | None):
    assert horner_step(rotations) == expected
