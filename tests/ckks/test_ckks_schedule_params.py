import pickle
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from functools import partial
from math import ceil, prod
from typing import Any, cast

import pytest

from ren.bigfloat import BigFloat
from ren.dtype import dtypes
from ren.engine.realize import call_op
from ren.engine.schedule import schedule
from ren.ir import Node, Op
from ren.poly import Poly
from ren.polydist import DiscreteGaussian, SparseTernary, Ternary
from ren.ring import RingSpec
from ren.ckks import (
    BootstrappingParams,
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
from ren.ckks._security import validate_modulus_security
from ren.ckks.params import CKKSTransition, RescaleStep
from ren.ckks.metadata import CKKSNoiseInfo
from ren.ckks.utils import select_moduli, steps_to_galois_elt
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError
from tests.utils import mixed_prime_ring
from tests.ckks.params import high_precision_params, non_prefix_params, small_params


def _small_boot_params(
    *, squaring_count: int = 0, failure_probability: float = 2**-16, input_margin: float = 1.0, precision_gains: Sequence[int] = ()
) -> BootstrappingParams:
    return BootstrappingParams(
        cts_depth=2,
        stc_depth=2,
        exp_degree=16,
        squaring_count=squaring_count,
        failure_probability=failure_probability,
        input_margin=input_margin,
        precision_gains=precision_gains,
    )


def _small_bootstrap_params() -> CKKSParams:
    return CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )


def _explicit_u32_params() -> tuple[CKKSParams, tuple[int, ...]]:
    ring = mixed_prime_ring(1024, dtype=dtypes.uint32)
    highest_moduli = ring.moduli[:6]
    levels = (
        CKKSLevel(moduli=highest_moduli[2:4], scale=1 << 18),
        CKKSLevel(moduli=highest_moduli[1:4], scale=1 << 21),
        CKKSLevel(moduli=highest_moduli[1:5], scale=1 << 24),
        CKKSLevel(moduli=highest_moduli, scale=1 << 27),
    )
    ks_primes = select_moduli(n=ring.n, num_primes=3, bit_length=29, exclude=set(highest_moduli))
    params = CKKSParams.from_levels(
        n=ring.n,
        dtype=dtypes.uint32,
        levels=levels,
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2, 2),
        security=None,
        log_range=0,
    )
    return params, highest_moduli


def test_key_switch_params_validate_digits():
    params = small_params()
    ks_params = params._highest_key_switch_params
    digits = tuple(Poly.zero(ks_params.ring_qt) for _ in range(ks_params.d))
    ks_params.validate_digits(digits)
    for invalid in ((), digits[:-1], (*digits, digits[0])):
        with pytest.raises(ParameterMismatchError, match="digit layout"):
            ks_params.validate_digits(invalid)
    with pytest.raises(ParameterMismatchError, match="ring"):
        ks_params.validate_digits((*digits[:-1], Poly.zero(params.highest_ring)))


def test_max_precision_handles_scale_outside_float_range():
    n = 1024
    moduli = select_moduli(n=n, num_primes=35, bit_length=60)
    ks_primes = select_moduli(n=n, num_primes=1, bit_length=59, exclude=set(moduli))
    params = CKKSParams.from_levels(
        n=n,
        dtype=dtypes.uint64,
        levels=[CKKSLevel(moduli=moduli, scale=BigFloat(2) ** 2000)],
        ks_primes=ks_primes,
        ks_digit_sizes=(len(moduli),),
        log_range=0,
        security=None,
    )

    assert params.max_precision == 2000


def test_explicit_schedule_params_expose_non_prefix_levels():
    params, highest_moduli = _explicit_u32_params()

    assert params.highest_level == 3
    assert params.highest_ring.moduli == highest_moduli
    assert params.ring_at(2).moduli == highest_moduli[1:5]
    assert params.ring_at(0).moduli == highest_moduli[2:4]
    assert params.level_of_ring(params.ring_at(1)) == 1

    transition = params.transition(3, 2)
    (step,) = transition.steps
    assert step.src_ring == params.ring_at(3)
    assert step.dst_ring == params.ring_at(2)
    removed_prod = highest_moduli[0] * highest_moduli[5]
    assert step.k == round(removed_prod * transition.dst_scale / transition.src_scale)
    assert step.poly_ratio == BigFloat(step.k) / removed_prod
    assert transition.message_ratio == step.poly_ratio * transition.src_scale / transition.dst_scale
    scaled = params.transition(3, 2, current_scale=2.5 * transition.src_scale)
    assert scaled.steps[0].k == round(removed_prod * transition.dst_scale / (2.5 * transition.src_scale))


def test_params_reject_scheduled_ring_outside_highest_ring():
    params, highest_moduli = _explicit_u32_params()
    foreign_modulus = select_moduli(
        n=params.n,
        num_primes=1,
        bit_length=29,
        exclude=set(highest_moduli + params.ks_primes),
    )
    levels = (replace(params.levels[0], moduli=foreign_modulus), *params.levels[1:])

    with pytest.raises(InvalidParameterError, match="contiguous subrange of the highest-level ring"):
        replace(params, levels=levels)


@pytest.mark.parametrize("target_level", [0, 2])
def test_highest_level_rotation_key_supports_non_prefix_target_level(target_level: int):
    params, _ = _explicit_u32_params()
    values = [complex((index % 17) / 17, (index % 13) / 13) for index in range(params.slot_count)]
    ctx = keygen(params, rotations={params.highest_level: {1}}).context()

    with ctx:
        result = decode(decrypt(encrypt(encode(values, msg_bound=2, level=target_level)).rotate(1)))

    assert result == pytest.approx(values[1:] + values[:1], abs=2e-2)


def test_default_conjugation_key_targets_default_level():
    params, _ = _explicit_u32_params()
    keys = keygen(params, conjugation=True)

    assert params.default_level == params.highest_level
    assert keys.get_galois_key(2 * params.n - 1).target_ring == params.ring_at(params.default_level)


def test_rotation_keys_cover_incomparable_levels_and_share_hoist():
    ring = mixed_prime_ring(1024, dtype=dtypes.uint32)
    highest_moduli = ring.moduli[:6]
    levels = (
        CKKSLevel(moduli=highest_moduli[:2], scale=1 << 18),
        CKKSLevel(moduli=highest_moduli[3:5], scale=1 << 18),
        CKKSLevel(moduli=highest_moduli, scale=1 << 21),
    )
    ks_primes = select_moduli(n=ring.n, num_primes=3, bit_length=29, exclude=set(highest_moduli))
    params = CKKSParams.from_levels(
        n=ring.n,
        dtype=dtypes.uint32,
        levels=levels,
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2, 2),
        security=None,
        log_range=0,
    )
    values = [complex((index % 17) / 17, (index % 13) / 13) for index in range(params.slot_count)]
    ctx = keygen(params, rotations={0: {1, 2}, 1: {2}}).context()

    rotation_elt = steps_to_galois_elt(1, params.n)
    assert rotation_elt in ctx.keyset._rotation_galois_elts_at(0)
    assert rotation_elt not in ctx.keyset._rotation_galois_elts_at(1)

    with ctx:
        level0_results = [decode(decrypt(result)) for result in encrypt(encode(values, msg_bound=2, level=0)).rotate([1, 2])]
        level1_ct = encrypt(encode(values, msg_bound=2, level=1))
        level1_result = decode(decrypt(level1_ct.rotate(2)))
        with pytest.raises(RuntimeError, match="Could not reach slot rotations"):
            level1_ct.rotate(1)

    assert level0_results[0] == pytest.approx(values[1:] + values[:1], abs=2e-2)
    assert level0_results[1] == pytest.approx(values[2:] + values[:2], abs=2e-2)
    assert level1_result == pytest.approx(values[2:] + values[:2], abs=2e-2)


def test_rotation_key_carries_its_digit_boundaries():
    params, _ = _explicit_u32_params()
    source_keys = keygen(params, rotations={1})
    target_ctx = replace(
        source_keys,
        params=replace(params, ks_digit_sizes=(1, 3, 2)),
        rlk=None,
    ).context()
    values = [complex((index % 17) / 17, (index % 13) / 13) for index in range(params.slot_count)]

    with target_ctx:
        result = decode(decrypt(encrypt(encode(values, msg_bound=2, level=2)).rotate(1)))

    assert result == pytest.approx(values[1:] + values[:1], abs=2e-2)


def test_prefix_generated_params_expose_schedule_api():
    params = small_params()

    assert params.highest_ring.dtype is dtypes.uint32
    assert params.level_of_ring(params.ring_at(2)) == 2

    transition = params.transition(3, 2)
    (step,) = transition.steps
    removed = tuple(q for q in step.src_ring.moduli if q not in step.dst_ring.moduli)
    assert len(removed) == 2
    assert removed == params.ring_at(3).moduli[len(params.ring_at(2).moduli) :]
    assert step.dst_ring.is_quotient_of(step.src_ring)


def test_bootstrap_base_has_distinct_logical_scale_on_same_ring():
    params = non_prefix_params()
    boot_params = params.boot_params
    assert boot_params is not None
    bootstrap_base = params.bootstrap_base_level
    assert bootstrap_base is not None
    transition = params.bootstrap_transition()
    assert transition.src_ring == transition.dst_ring == params.ring_at(0)
    assert params.scale(params.ring_at(0)) == params.levels[0].scale
    assert params.scale(params.ring_at(0)) != bootstrap_base.scale
    assert BigFloat(prod(bootstrap_base.moduli)) / bootstrap_base.scale == params.bootstrap_message_ratio


def test_non_prefix_const_rescale_folds_each_transition():
    params = non_prefix_params()
    for level in range(params.highest_level, 0, -1):
        src, dst = params.ring_at(level), params.ring_at(level - 1)
        for value in (3**100, -(3**100)):
            result = Poly.const(value, src).rescale(dst.moduli)
            expected = round(Fraction(value * dst.q, src.q)) % dst.q
            assert result.node.const_val is not None
            assert result.node.const_val % dst.q == expected
            linear, becomes_map = schedule(Node.sink(result.node))
            assert linear.op is Op.LINEAR and not linear.src
            assert becomes_map[result.node].base.op is Op.CONST
            assert becomes_map[result.node].ring == dst


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
def test_rescale_noise_ratio_does_not_underflow(kind):
    params = non_prefix_params()
    with params.context():
        value = kind.const(0.25, level=params.highest_level)
        target = params.ring_at(0)
        assert float(BigFloat(target.q) / value.ring.q) == 0
        result = value._rescale(target.moduli)

        assert result.noise_info.noise_bound > 0
        assert result.noise_info.noise_heuristic_bound > 0


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
@pytest.mark.parametrize("bases", [((17,), (17, 41), (17, 41, 73)), ((17,), (41,), (17, 41, 73))])
def test_conversion_uses_endpoint_without_truncating_source_capacity(kind, bases):
    base = small_params()
    assert isinstance(base.noise_dist, DiscreteGaussian)
    assert isinstance(base.secret_key_dist, Ternary)
    params = replace(
        base,
        n=4,
        levels=tuple(CKKSLevel(moduli=basis, scale=BigFloat(prod(basis)) / 4) for basis in bases),
        log_range=0,
        ks_primes=(89,),
        ks_digit_sizes=(3,),
        noise_dist=replace(base.noise_dist, n=4),
        secret_key_dist=replace(base.secret_key_dist, n=4),
    )
    with params.context():
        value = kind.const(4488 / float(params.scale(2)), level=2)
        result = value.to_level(0)
        poly = result.m if isinstance(result, Plaintext) else result.b
        # Scale follows Q/4, so direct conversion divides once by the total ratio.
        assert poly.node.const_val == round(Fraction(4488, 41 * 73)) == 1


def test_executable_transition_preserves_explicit_rounding_boundaries():
    src = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtypes.uint32)
    mid = RingSpec(n=4, moduli=(41,), dtype=dtypes.uint32)
    dst = RingSpec(n=4, moduli=(17,), dtype=dtypes.uint32)
    direct = CKKSTransition(src, BigFloat(src.q) / 4, BigFloat(dst.q) / 4, (RescaleStep(src, dst, 1),))
    chained = replace(direct, steps=(RescaleStep(src, mid, 1), RescaleStep(mid, dst, 1)))
    poly = Poly.const(4488, src)

    assert direct.apply(poly).node.const_val == 1
    assert chained.apply(poly).node.const_val == 2
    assert direct.message_ratio == pytest.approx(1)
    assert chained.message_ratio == pytest.approx(1)
    with pytest.raises(InvalidParameterError, match="preceding ring"):
        replace(chained, steps=tuple(reversed(chained.steps)))
    with pytest.raises(InvalidParameterError, match="at least one step"):
        replace(direct, steps=())
    with pytest.raises(InvalidParameterError, match="source"):
        direct.apply(Poly.const(1, mid))


def test_rescale_step_validates_physical_operation():
    src = RingSpec(n=4, moduli=(17, 41), dtype=dtypes.uint32)
    dst = RingSpec(n=4, moduli=(17,), dtype=dtypes.uint32)
    assert RescaleStep(src, dst, 0).apply(Poly.const(3, src)).node.const_val == 0
    with pytest.raises(InvalidParameterError, match="remove"):
        RescaleStep(dst, src, 1)
    with pytest.raises(InvalidParameterError, match="degree and dtype"):
        RescaleStep(src, replace(dst, n=8), 1)


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
def test_transition_accounts_for_added_moduli(kind):
    params = CKKSParams.from_levels(
        n=4,
        dtype=dtypes.uint32,
        levels=(
            CKKSLevel(moduli=(17, 41), scale=16),
            CKKSLevel(moduli=(41, 73), scale=8),
            CKKSLevel(moduli=(17, 41, 73), scale=32),
        ),
        ks_primes=(89,),
        ks_digit_sizes=(3,),
        security=None,
        log_range=0,
    )
    transition = params.transition(1, 0)
    # Doubling the scale while replacing 73 with 17 requires k = round(146/17).
    assert transition.steps[0].k == 9
    assert transition.poly_ratio == BigFloat(153) / 73
    for value, expected in ((11, 23), (-11, -23)):
        result = transition.apply(Poly.const(value, params.ring_at(1)))
        assert list(result) == [expected % (17 * 41), 0, 0, 0]
    with params.context():
        value = replace(kind.zero(level=1), noise_info=CKKSNoiseInfo.zero())
        noisier = replace(value, noise_info=CKKSNoiseInfo(noise_bound=73, noise_heuristic_bound=146, msg_bound=0))
        baseline = value.to_level(0)
        converted = noisier.to_level(0)
        assert float(converted.noise_info.noise_bound - baseline.noise_info.noise_bound) == pytest.approx(153)
        assert float(converted.noise_info.noise_heuristic_bound - baseline.noise_info.noise_heuristic_bound) == pytest.approx(306)


def test_ciphertext_to_level_supports_same_size_remove_add_transition():
    ring = mixed_prime_ring(1024, dtype=dtypes.uint32)
    highest_moduli = ring.moduli[:5]
    levels = (
        CKKSLevel(moduli=highest_moduli[1:3], scale=1 << 18),
        CKKSLevel(moduli=highest_moduli[0:3], scale=1 << 27),
        CKKSLevel(moduli=highest_moduli[1:4], scale=1 << 24),
        CKKSLevel(moduli=highest_moduli, scale=1 << 27),
    )
    ks_primes = select_moduli(n=ring.n, num_primes=3, bit_length=29, exclude=set(highest_moduli))
    params = CKKSParams.from_levels(
        n=ring.n,
        dtype=dtypes.uint32,
        levels=levels,
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2, 1),
        security=None,
        log_range=10,
    )
    assert len(params.ring_at(2).moduli) == len(params.ring_at(1).moduli)
    assert set(params.transition(2, 1).dst_ring.moduli) - set(params.ring_at(2).moduli) == {highest_moduli[0]}

    ctx = keygen(params).context()
    with ctx:
        rng = random.Random(42)
        values = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(params.slot_count)]
        ct = encrypt(encode(values, msg_bound=2, level=2))
        out = ct.to_level(1)
        decrypted = decode(decrypt(out), params.slot_count)

    assert params.level_of_ring(out.ring) == 1
    assert decrypted == pytest.approx(values, abs=2e-4)


def _endpoint_params() -> CKKSParams:
    highest_ring = mixed_prime_ring(1024, dtype=dtypes.uint32)
    highest_moduli = highest_ring.moduli[:6]
    level_moduli = (highest_moduli[:2], highest_moduli[2:4], highest_moduli)
    levels = tuple(CKKSLevel(moduli=moduli, scale=BigFloat(prod(moduli)) / 4) for moduli in level_moduli)
    ks_primes = select_moduli(n=highest_ring.n, num_primes=3, bit_length=29, exclude=set(highest_moduli))
    return CKKSParams.from_levels(
        n=highest_ring.n,
        dtype=dtypes.uint32,
        levels=levels,
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2, 2),
        security=None,
        log_range=0,
    )


@pytest.fixture(scope="module")
def keys() -> CKKSKeySet:
    return keygen(_endpoint_params()).realize()


@pytest.mark.parametrize(
    "wrap", [pytest.param(lambda pt: pt, id="plaintext"), pytest.param(Plaintext.to_ciphertext, id="trivial"), pytest.param(encrypt, id="encrypted")]
)
@pytest.mark.parametrize("values", [0.25, [0.25, -0.5]], ids=["scalar", "vector"])
@pytest.mark.parametrize("level", [0, 1])
def test_to_level_preserves_decoded_values(keys: CKKSKeySet, wrap, values, level):
    with keys.context(track_plaintext_values=True):
        value = wrap(encode(values, msg_bound=1, level=2))
        lowered = value.to_level(level)
        plaintext = lowered if isinstance(lowered, Plaintext) else decrypt(lowered)
        expected = [values] if isinstance(values, float) else values
        assert lowered.level == level
        assert decode(plaintext, len(expected)) == pytest.approx(expected, abs=2e-3)
        assert lowered.values[: len(expected)] == pytest.approx(expected)


@pytest.mark.parametrize("kind", [Plaintext, Ciphertext])
@pytest.mark.parametrize("level", [0, 1])
def test_to_level_propagates_existing_noise(kind, level):
    with _endpoint_params().context(track_plaintext_values=True):
        value = replace(kind.zero(level=2), noise_info=CKKSNoiseInfo.zero())
        noisier = replace(value, noise_info=CKKSNoiseInfo(noise_bound=3 * value.scale, noise_heuristic_bound=4 * value.scale, msg_bound=0))
        baseline = value.to_level(level)
        result = noisier.to_level(level)
        # This schedule has scale Q/4, so conversion preserves normalized noise.
        # Differences isolate existing noise from any conversion-error allowance.
        assert float((result.noise_info.noise_bound - baseline.noise_info.noise_bound) / result.scale) == pytest.approx(3)
        assert float((result.noise_info.noise_heuristic_bound - baseline.noise_info.noise_heuristic_bound) / result.scale) == pytest.approx(4)


def test_sum_propagates_every_inputs_noise():
    with _endpoint_params().context(track_plaintext_values=True):
        values = [replace(Ciphertext.zero(level=level), noise_info=CKKSNoiseInfo.zero()) for level in (2, 2, 0)]
        noisier = [
            replace(value, noise_info=CKKSNoiseInfo(noise_bound=bound * value.scale, noise_heuristic_bound=bound * value.scale, msg_bound=0))
            for value, bound in zip(values, (3, 4, 12))
        ]
        baseline = Ciphertext.sum(values)
        result = Ciphertext.sum(noisier)
        # The strict contributions add to 19; the existing independent-noise
        # convention combines their squared heuristic contributions to 13^2.
        assert float((result.noise_info.noise_bound - baseline.noise_info.noise_bound) / result.scale) == pytest.approx(19)
        variance = result.noise_info.noise_heuristic_bound**2 - baseline.noise_info.noise_heuristic_bound**2
        assert float(variance / result.scale**2) == pytest.approx(169)


def test_sum_preserves_mixed_level_values(keys: CKKSKeySet):
    with keys.context(track_plaintext_values=True):
        inputs = ([0.25, -0.5], [-0.125, 0.25], [-0.25, 0.125])
        ciphertexts = [encrypt(encode(values, msg_bound=max(map(abs, values)), level=level)) for values, level in zip(inputs, (2, 2, 0))]
        result = Ciphertext.sum(ciphertexts)
        expected = [-0.125, -0.125]
        assert result.level == 0
        assert decode(decrypt(result), len(expected)) == pytest.approx(expected, abs=2e-3)
        assert result.values[: len(expected)] == pytest.approx(expected)


@pytest.mark.parametrize("levels,expected_rescales", [((2, 0), 2), ((2, 1, 0), 6)], ids=["direct", "two-rounding-steps"])
def test_sum_shares_only_the_final_rescale(keys: CKKSKeySet, monkeypatch, levels, expected_rescales):
    params = keys.params
    transition = CKKSTransition(
        params.ring_at(2),
        params.scale(2),
        params.scale(0),
        tuple(RescaleStep(params.ring_at(src), params.ring_at(dst), 1) for src, dst in zip(levels, levels[1:])),
    )

    def explicit_transition(self, src_level, dst_level):
        assert (src_level, dst_level) == (2, 0)
        return transition

    with keys.context():
        plaintext = encode([0.25, -0.5], msg_bound=1, level=2)
        ciphertexts = [encrypt(plaintext).realize(), encrypt(plaintext).realize(), Ciphertext.zero(level=0)]
        monkeypatch.setattr(CKKSParams, "transition", explicit_transition)
        result = Ciphertext.sum(ciphertexts)
        linear, _ = schedule(Node.sink(result.a.node, result.b.node))
        # Two components share the final rounding; earlier steps run per input.
        assert sum(call_op(call) is Op.RESCALE for call in linear.src) == expected_rescales
        assert decode(decrypt(result), 2) == pytest.approx([0.5, -1.0], abs=2e-3)


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("source_level", [1, 3, 10])
def test_prefix_to_level_uses_one_rescale_per_polynomial(dtype, source_level: int):
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=10,
        log_precision=30,
        log_range=0,
        dtype=dtype,
        p_prime_count=4,
        security=None,
    )
    with keygen(params).context(track_plaintext_values=True):
        values = [0.25 + 0.125j, -0.5]
        pt = encode(values, msg_bound=1, level=source_level)
        pt.m.realize()
        ct = encrypt(pt)
        ct.realize()

        lowered_pt = pt.to_level(0)
        pt_linear, _ = schedule(Node.sink(lowered_pt.m.node))
        assert sum(call_op(call) is Op.RESCALE for call in pt_linear.src) == 1
        assert lowered_pt.level == 0
        assert decode(lowered_pt, len(values)) == pytest.approx(values, abs=1e-6)
        assert lowered_pt.values[: len(values)] == pytest.approx(values)

        lowered_ct = ct.to_level(0)
        ct_linear, _ = schedule(Node.sink(lowered_ct.a.node, lowered_ct.b.node))
        assert sum(call_op(call) is Op.RESCALE for call in ct_linear.src) == 2
        assert lowered_ct.level == 0
        assert decode(decrypt(lowered_ct), len(values)) == pytest.approx(values, abs=1e-6)
        assert lowered_ct.values[: len(values)] == pytest.approx(values)


def test_prefix_raw_dot_shares_the_final_rescale():
    params = CKKSParams.generate_prefix(log_n=10, depth=3, log_precision=30, log_range=0, dtype=dtypes.uint64, p_prime_count=4, security=None)
    with keygen(params).context(track_plaintext_values=True):
        values = [0.25 + 0.125j, -0.5]
        ct = encrypt(encode(values, msg_bound=1, level=2)).realize()
        zero = Plaintext.zero(level=1)
        result = Ciphertext.dot([ct, zero], [[0.5 + 0j, -0.25 + 0j], zero])
        linear, _ = schedule(Node.sink(result.a.node, result.b.node))

        assert sum(call_op(call) is Op.RESCALE for call in linear.src) == 2
        assert decode(decrypt(result), len(values)) == pytest.approx([0.125 + 0.0625j, 0.125], abs=1e-6)


def test_explicit_schedule_params_pickle_roundtrip():
    params, _highest_moduli = _explicit_u32_params()

    restored = pickle.loads(pickle.dumps(params))

    assert restored.levels == params.levels
    assert restored.noise_dist == params.noise_dist
    assert restored.scale(3) == params.scale(3)


def test_params_scales_are_always_bigfloat():
    explicit_params, _highest_moduli = _explicit_u32_params()
    generated_params = small_params()

    for params in (explicit_params, generated_params):
        assert all(isinstance(level.scale, BigFloat) for level in params.levels)


def test_params_reject_nonpositive_secret_key_bound():
    params = small_params()
    with pytest.raises(InvalidParameterError, match="secret key coefficient 1-norm bound must be finite and positive"):
        replace(params, secret_key_dist=SparseTernary(n=params.n, h=0))


@pytest.mark.parametrize("norm", [0.0, -1.0, float("inf"), float("nan")])
def test_params_reject_invalid_expected_secret_norm(norm):
    class InvalidNorm(Ternary):
        def expected_squared_l2_norm(self) -> float:
            return norm

    params = small_params()
    with pytest.raises(InvalidParameterError, match="secret expected squared L2 norm"):
        replace(params, secret_key_dist=InvalidNorm(params.n))


@pytest.mark.parametrize("scale", [0, -1, BigFloat("inf")])
def test_ckks_level_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        CKKSLevel(moduli=(3,), scale=scale)


@pytest.mark.parametrize("missing", ["boot_params", "bootstrap_base_level"])
def test_bootstrap_configuration_and_base_must_be_present_together(missing):
    params = _small_bootstrap_params()

    with pytest.raises(InvalidParameterError, match="must be provided together"):
        replace(params, **{missing: None})


@pytest.mark.parametrize("ratio", [1, 2])
def test_ckks_params_reject_bootstrap_scale_too_close_to_q0(ratio: int):
    generated = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )
    assert generated.boot_params is not None
    assert generated.bootstrap_base_level is not None
    moduli = generated.bootstrap_base_level.moduli

    with pytest.raises(InvalidParameterError, match="less than half of q0"):
        replace(generated, bootstrap_base_level=CKKSLevel(moduli=moduli, scale=BigFloat(prod(moduli)) / ratio))


def test_bootstrap_schedule_rejects_depth_beyond_levels():
    generated = CKKSParams.generate_prefix(
        log_n=10,
        depth=0,
        log_precision=20,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )

    assert generated.boot_params is not None
    assert generated.bootstrap_base_level is not None
    deeper = _small_boot_params(squaring_count=1)
    with pytest.raises(InvalidParameterError, match="depth exceeds the available scheduled levels"):
        replace(generated, boot_params=deeper)


def test_bootstrap_schedule_allows_spare_levels_below_the_circuit():
    generated = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(squaring_count=1),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )

    assert generated.boot_params is not None
    assert generated.bootstrap_base_level is not None
    shallower = _small_boot_params()
    spare = replace(generated, boot_params=shallower)
    assert generated.bootstrap_output_level is not None
    assert spare.bootstrap_output_level == generated.bootstrap_output_level + 1
    assert spare.default_level == spare.bootstrap_output_level


def test_explicit_params_can_be_security_validated_after_construction():
    params, _highest_moduli = _explicit_u32_params()

    with pytest.raises(ValueError, match="exceeds the 128-bit security estimate"):
        params.validate_security(128)


@pytest.mark.parametrize("failure_probability", [0, -1, 2**-15, float("inf"), float("nan")])
def test_bootstrap_failure_probability_must_be_in_supported_range(failure_probability: float):
    with pytest.raises(ValueError, match=rf"must be in \(0, 2\^-16\], got {failure_probability!r}"):
        _small_boot_params(failure_probability=failure_probability)


@pytest.mark.parametrize("margin", [0.5, 0.0, -1.0, float("inf"), float("nan")])
def test_bootstrap_input_margin_must_be_finite_and_at_least_one(margin: float):
    with pytest.raises(ValueError, match="finite and at least 1.0"):
        _small_boot_params(input_margin=margin)


def test_bootstrap_precision_gains_define_iteration_bounds():
    assert _small_boot_params().bounds(0) == (0,)
    assert _small_boot_params(input_margin=2).bounds(0) == (1,)
    assert _small_boot_params(precision_gains=(10, 12, 12)).bounds(0) == (0, -10, -22, -34)


def test_bootstrap_precision_gains_do_not_retain_mutable_input():
    gains = [10, 12]
    boot_params = _small_boot_params(precision_gains=gains)

    gains[0] = 1

    assert boot_params.precision_gains == (10, 12)
    hash(boot_params)


@pytest.mark.parametrize("gain", [0, -1])
def test_bootstrap_precision_gains_must_be_positive(gain: int):
    with pytest.raises(ValueError, match="precision gains must be positive"):
        _small_boot_params(precision_gains=(gain,))


@pytest.mark.parametrize(
    ("sizing", "error"),
    [
        ({"bootstrap_level_bit_length": 0}, "positive"),
        ({"bootstrap_message_ratio": 2}, "greater than two"),
        ({"bootstrap_message_ratio": float("inf")}, "finite"),
    ],
)
def test_generate_prefix_validates_bootstrap_sizing(sizing, error):
    with pytest.raises(ValueError, match=error):
        CKKSParams.generate_prefix(
            log_n=10,
            depth=2,
            log_precision=20,
            log_range=0,
            dtype=dtypes.uint32,
            boot_params=_small_boot_params(),
            p_prime_count=4,
            security=None,
            **sizing,
        )


@pytest.mark.parametrize("sizing", [{"bootstrap_level_bit_length": 30}, {"bootstrap_message_ratio": 32.0}])
def test_generate_prefix_rejects_bootstrap_sizing_without_bootstrap(sizing):
    generate_prefix = cast(Any, CKKSParams.generate_prefix)
    with pytest.raises(ValueError, match="require boot_params"):
        generate_prefix(
            log_n=10,
            depth=2,
            log_precision=20,
            log_range=0,
            dtype=dtypes.uint32,
            p_prime_count=4,
            security=None,
            **sizing,
        )


def test_generate_prefix_requires_log_range_for_all_parameter_sets():
    generate_prefix = cast(Any, CKKSParams.generate_prefix)
    with pytest.raises(TypeError, match="log_range"):
        generate_prefix(
            log_n=10,
            depth=2,
            log_precision=20,
            dtype=dtypes.uint32,
            p_prime_count=4,
            security=None,
        )

    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=4,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )
    assert params.log_range == 4


def test_generate_prefix_sizes_base_from_precision_and_range_without_bootstrapping():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=3,
        dtype=dtypes.uint32,
        p_prime_count=4,
        security=None,
    )

    assert params.boot_params is None
    q0 = prod(params.ring_at(0).moduli)
    base_target = 1 << (20 + 1 + 3)
    assert max(q0 / base_target, base_target / q0) <= 2
    assert q0 >= ceil(2 * params.scale(0) * (2**3))


def test_generate_prefix_rejects_negative_log_range():
    with pytest.raises(ValueError, match="log_range must be nonnegative"):
        CKKSParams.generate_prefix(
            log_n=10,
            depth=0,
            log_precision=20,
            log_range=-1,
            dtype=dtypes.uint32,
            p_prime_count=4,
            security=None,
        )


def test_generate_prefix_accepts_explicit_p_prime_bit_length():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=4,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        p_prime_bit_length=31,
        security=None,
    )

    assert [prime.bit_length() for prime in params.ks_primes] == [31] * 4


@pytest.mark.parametrize(("dtype", "log_precision", "expected_count"), [(dtypes.uint32, 20, 4), (dtypes.uint64, 50, 2)])
def test_generate_prefix_defaults_p_prime_count_for_dtype(dtype, log_precision, expected_count):
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=log_precision,
        log_range=4,
        dtype=dtype,
        security=None,
    )

    assert len(params.ks_primes) == expected_count
    q_count = len(params.highest_ring.moduli)
    assert params.ks_digit_sizes == tuple(min(expected_count, q_count - start) for start in range(0, q_count, expected_count))


def test_generate_prefix_rejects_nonpositive_p_prime_bit_length():
    with pytest.raises(ValueError, match="p_prime_bit_length must be positive"):
        CKKSParams.generate_prefix(
            log_n=10,
            depth=2,
            log_precision=20,
            log_range=4,
            dtype=dtypes.uint32,
            p_prime_count=4,
            p_prime_bit_length=0,
            security=None,
        )


def test_generate_prefix_decouples_ks_digit_size_from_p_prime_count():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=5,
        log_precision=20,
        log_range=4,
        dtype=dtypes.uint32,
        p_prime_count=4,
        ks_digit_size=3,
        security=None,
    )

    assert len(params.ks_primes) == 4
    assert params.ks_digit_sizes == (3, 3)


def test_generate_prefix_defaults_ks_digit_size_to_p_prime_count():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=5,
        log_precision=20,
        log_range=4,
        dtype=dtypes.uint32,
        p_prime_count=4,
        security=None,
    )

    assert params.ks_digit_sizes == (4, 2)


def test_generate_prefix_rejects_nonpositive_ks_digit_size():
    with pytest.raises(ValueError, match="ks_digit_size must be positive"):
        CKKSParams.generate_prefix(
            log_n=10,
            depth=2,
            log_precision=20,
            log_range=4,
            dtype=dtypes.uint32,
            p_prime_count=4,
            ks_digit_size=0,
            security=None,
        )


def test_generate_prefix_rejects_ks_digit_bit_budget_larger_than_p():
    with pytest.raises(ValueError, match="P bit budget must cover one key-switch digit, but 84 < 180"):
        CKKSParams.generate_prefix(
            log_n=10,
            depth=5,
            log_precision=45,
            log_range=4,
            dtype=dtypes.uint64,
            p_prime_count=2,
            p_prime_bit_length=42,
            ks_digit_size=3,
            security=None,
        )


def test_decoupled_ks_digit_size_supports_rotation():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=5,
        log_precision=20,
        log_range=4,
        dtype=dtypes.uint32,
        p_prime_count=4,
        ks_digit_size=3,
        security=None,
    )
    values = [complex((index % 17) / 17, (index % 13) / 13) for index in range(params.slot_count)]
    ctx = keygen(params, rotations={1}).context()

    with ctx:
        result = decode(decrypt(encrypt(encode(values, msg_bound=2)).rotate(1)))

    assert result == pytest.approx(values[1:] + values[:1], abs=2e-2)


def test_regular_and_meta_bootstrapping_share_level_geometry():
    boot_params = _small_boot_params()

    def generate(boot_params: BootstrappingParams) -> CKKSParams:
        return CKKSParams.generate_prefix(
            log_n=10,
            depth=2,
            log_precision=20,
            log_range=0,
            dtype=dtypes.uint32,
            boot_params=boot_params,
            bootstrap_level_bit_length=30,
            p_prime_count=4,
            security=None,
        )

    regular = generate(boot_params)
    meta = generate(replace(boot_params, precision_gains=(5,)))

    assert regular.levels == meta.levels
    output_level = regular.bootstrap_output_level
    assert output_level is not None
    assert output_level == meta.bootstrap_output_level == 2
    assert regular.highest_level == meta.highest_level
    assert regular.boot_params is not None and meta.boot_params is not None
    assert regular.bootstrap_base_level == meta.bootstrap_base_level
    assert regular.bootstrap_base_level is not None
    q0 = prod(regular.bootstrap_base_level.moduli)

    compute_target = 1 << 20
    for upper in range(1, output_level + 1):
        increment = regular.ring_at(upper).q // regular.ring_at(upper - 1).q
        assert max(increment / compute_target, compute_target / increment) <= 2

    boot_target = 1 << 30
    assert max(q0 / boot_target, boot_target / q0) <= 2
    for upper in range(output_level + 1, regular.highest_level + 1):
        increment = regular.ring_at(upper).q // regular.ring_at(upper - 1).q
        assert max(increment / boot_target, boot_target / increment) <= 2


def test_bootstrap_q1_is_omitted_when_q0_already_has_enough_capacity():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )

    assert params.boot_params is not None
    assert params.bootstrap_base_level is not None
    assert params.bootstrap_base_level.moduli == params.ring_at(0).moduli
    assert params.level_of_ring(params.ring_at(0)) == 0
    assert params.bootstrap_transition().src_ring == params.bootstrap_transition().dst_ring


def test_scale_rejects_bootstrap_moduli_on_a_different_ring():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=20,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )
    assert params.boot_params is not None
    assert params.bootstrap_base_level is not None
    foreign_ring = RingSpec(n=params.n // 2, moduli=params.bootstrap_base_level.moduli, dtype=params.dtype)

    with pytest.raises(InvalidParameterError, match="ring is not part of this CKKS schedule"):
        params.scale(foreign_ring)


def test_bootstrap_q1_is_generated_when_required_by_precision_and_range():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=40,
        log_range=10,
        dtype=dtypes.uint32,
        boot_params=_small_boot_params(input_margin=1.5),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )

    transition = params.bootstrap_transition()
    assert transition.dst_ring != transition.src_ring
    assert transition.dst_ring.is_quotient_of(transition.src_ring)
    q0q1 = prod(params.levels[0].moduli)
    assert params.boot_params is not None
    assert params.bootstrap_base_level is not None
    minimum = params.boot_params.input_margin * params.scale(0) * (2**params.log_range) * params.bootstrap_message_ratio
    assert q0q1 >= minimum


def test_high_precision_params_pickle_preserves_separate_boot_base():
    params = high_precision_params()
    assert params.boot_params is not None
    assert params.bootstrap_base_level is not None

    restored = pickle.loads(pickle.dumps(params))
    assert restored.bootstrap_base_level is not None

    assert restored.boot_params == params.boot_params
    assert restored.noise_dist == params.noise_dist
    assert restored.levels == params.levels
    assert restored.bootstrap_base_level == params.bootstrap_base_level
    assert BigFloat(prod(restored.bootstrap_base_level.moduli)) / restored.bootstrap_base_level.scale == pytest.approx(256.0)


# For any CKKS rescale step, multiplying two ciphertexts at the upper level
# yields scale `upper_scale ** 2`; dropping `removed_prod` (and adding
# `added_prod`) brings us to the lower level with scale
# `upper_scale ** 2 * added_prod / removed_prod`.  The recorded
# `lower_scale` for the destination level must match this within floating
# point error or homomorphic multiplications will silently drift.
#
# The relationship is enforced *within* each chain section (compute and
# boot-top) but is intentionally discontinuous at the boundary between
# them, since the two sections target different scales (compute_target vs
# boot_target).  The boundary lives at `depth + 1 -> depth`.
# The transition from computation level 0 to the separate bootstrap base is
# intentionally outside this computation-level ladder.


@pytest.mark.parametrize("params_factory", [small_params, CKKSParams.default, high_precision_params], ids=["small", "default", "high-precision"])
def test_scale_chain_matches_moduli(params_factory: Callable[[], CKKSParams]):
    params = params_factory()
    boundary = params.bootstrap_output_level  # transition (boundary + 1) -> boundary is the section break
    for upper in range(params.highest_level, 0, -1):
        lower = upper - 1
        if boundary is not None and lower == boundary and upper == boundary + 1:
            continue  # compute<->boot boundary uses different target scales by design
        transition = params.transition(upper, lower)
        upper_scale = transition.src_scale
        lower_scale = transition.dst_scale
        expected_lower = upper_scale * upper_scale * transition.dst_ring.q / transition.src_ring.q
        relative_error = abs(lower_scale - expected_lower) / expected_lower
        assert relative_error <= 1e-12, (
            f"L{upper}->L{lower}: src_scale={upper_scale}, modulus_ratio={BigFloat(transition.dst_ring.q) / transition.src_ring.q}, "
            f"expected dst_scale={expected_lower}, got {lower_scale}"
        )


def test_weight_192_main_key_checks_the_full_key_modulus():
    boot = CKKSParams.default().boot_params
    assert boot is not None
    boot = replace(boot, sparse_encapsulation_weight=None)
    params = CKKSParams.default(depth=6, boot_params=boot)
    assert params.secret_key_dist == SparseTernary(params.n, h=192)
    assert params.bootstrap_output_level == 6
    with pytest.raises(ValueError, match=r"h=192.*maximum modulus 2\^1475"):
        CKKSParams.default(depth=7, boot_params=boot)


def test_encapsulation_checks_the_ephemeral_key():
    params = CKKSParams.default()
    boot = params.boot_params
    assert boot is not None and boot.sparse_encapsulation_weight == 32
    assert params.default_level == params.bootstrap_output_level == 10
    assert isinstance(params.secret_key_dist, Ternary)
    with pytest.raises(ValueError, match=r"h=32.*maximum modulus 2\^121"):
        CKKSParams.default(bootstrap_level_bit_length=62)
    with pytest.raises(ValueError, match=r"no 128-bit .*estimate.*h=63"):
        CKKSParams.default(boot_params=replace(boot, sparse_encapsulation_weight=63))
    with pytest.raises(ValueError, match=r"h=32.*maximum modulus 2\^121"):
        CKKSParams.default(boot_params=None, secret_key_dist=SparseTernary(n=params.n, h=32))


@pytest.mark.parametrize(
    ("log_n", "weight", "base_bits", "p_bits"), [(15, 32, 50, 40), (15, 64, 60, 60), (16, 64, 60, 60), (17, 32, 60, 60), (17, 64, 60, 60)]
)
def test_encapsulation_modulus_limits(log_n: int, weight: int, base_bits: int, p_bits: int):
    boot = replace(_small_boot_params(), sparse_encapsulation_weight=weight)
    build = partial(
        CKKSParams.default, log_n=log_n, depth=2, log_precision=35, p_prime_count=2, p_prime_bit_length=p_bits, ks_digit_size=1, boot_params=boot
    )
    assert build(bootstrap_level_bit_length=base_bits).bootstrap_output_level == 2
    with pytest.raises(ValueError, match=rf"exceeds the 128-bit security estimate.*h={weight}"):
        build(bootstrap_level_bit_length=base_bits + 2)


def test_default_overrides_keep_security_validation():
    with pytest.raises(ValueError, match="128-bit security"):
        CKKSParams.default(log_n=10, depth=1, boot_params=None)


def test_generate_prefix_log_extension_factor_selects_compatible_primes():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=3,
        log_precision=50,
        log_range=10,
        dtype=dtypes.uint32,
        security=None,
        log_extension_factor=1,
    )
    extension_modulus = 4 * params.n
    for prime in (*params.highest_ring.moduli, *params.ks_primes):
        assert prime % extension_modulus == 1
    # The doubled-degree extension ring is constructible from these moduli.
    RingSpec(n=2 * params.n, moduli=params.highest_ring.moduli, dtype=params.dtype)


def test_generate_prefix_rejects_negative_log_extension_factor():
    with pytest.raises(ValueError, match="log_extension_factor"):
        CKKSParams.generate_prefix(
            log_n=10,
            depth=3,
            log_precision=50,
            log_range=10,
            dtype=dtypes.uint32,
            security=None,
            log_extension_factor=-1,
        )


@pytest.mark.parametrize(("secret", "max_bits"), [(Ternary(n=65536), 1746), (SparseTernary(n=65536, h=32), 121)])
@pytest.mark.parametrize("stdev", [Fraction("3.2"), Fraction(4)])
def test_validate_modulus_security_budget(secret, max_bits, stdev):
    noise = DiscreteGaussian(n=secret.n, stdev=stdev)
    limit = 1 << max_bits
    validate_modulus_security(limit, secret=secret, error=noise, security=128)
    with pytest.raises(ValueError, match="exceeds the 128-bit security estimate"):
        validate_modulus_security(limit + 1, secret=secret, error=noise, security=128)


def test_modulus_security_rejects_unassessed_degree():
    with pytest.raises(ValueError, match="no 128-bit security estimate"):
        validate_modulus_security(1024, secret=Ternary(n=512), error=DiscreteGaussian(n=512, stdev=Fraction("3.2")), security=128)


def test_modulus_security_rejects_narrower_noise():
    with pytest.raises(ValueError, match="no 128-bit security estimate"):
        validate_modulus_security(1024, secret=Ternary(n=65536), error=DiscreteGaussian(n=65536, stdev=Fraction("3.199")), security=128)
