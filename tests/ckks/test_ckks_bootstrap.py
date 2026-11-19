import importlib
from collections import OrderedDict
from dataclasses import replace
from math import prod

import numpy as np
import pytest
import ren
from ren.bigfloat import BigFloat
from ren.dtype import dtypes
from ren.ir import Node, Op
from ren.poly import Poly
from ren.polydist import KeyDist, SparseTernary
from ren.random import set_global_rng_seed
from ren.ring import Domain
from ren.ckks import (
    BootstrappingParams,
    CKKSContext,
    CKKSKeySet,
    CKKSParams,
    Ciphertext,
    MultipartyParams,
    bootstrap,
    decode,
    decrypt,
    encode,
    encrypt,
    keygen,
)
from ren.ckks.bootstrap import coefficients_to_slots, l_matrix, l_matrix_partial_factorization, mod_one, slots_to_coefficients
from ren.ckks.matrix import LinearTransform, SparseMatrix
from ren.ckks.utils import steps_to_galois_elt
from ren.ckks.errors import InvalidParameterError
from ren.ckks.ckks import _switch_secret
from tests.utils import random_ciphertext, random_values_with_norm
from tests.ckks.params import high_precision_params, non_prefix_params, small_params


bootstrap_module = importlib.import_module("ren.ckks.bootstrap")


def bit_reverse(x: int, log_n: int) -> int:
    """Reverse the bits of x using log_n bits."""
    result = 0
    for i in range(log_n):
        if x & (1 << i):
            result |= 1 << (log_n - 1 - i)
    return result


@pytest.fixture(scope="module")
def test_params() -> CKKSParams:
    return CKKSParams.default()


@pytest.fixture(scope="module")
def ctx(keys: CKKSKeySet) -> CKKSContext:
    return keys.context(track_plaintext_values=True)


# Composed transforms rotate giants after the rescale; double-hoisted ones rotate everything at the input level.
@pytest.mark.parametrize("hoist,babies,giants,giant_level", [(False, (1,), (3,), 3), (True, (1, 2), (3, 6), 4)])
def test_bootstrap_keys_support_matrix_execution_levels(monkeypatch, hoist, babies, giants, giant_level):
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=4,
        log_precision=30,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=BootstrappingParams(
            cts_depth=2,
            stc_depth=2,
            exp_degree=16,
            squaring_count=0,
            failure_probability=2**-16,
        ),
        bootstrap_level_bit_length=30,
        p_prime_count=4,
        security=None,
    )
    # Offsets 0..8 split as babies {0, 1, 2} by giants {0, 3, 6}.
    diagonals = {amount: (-0.5) ** (amount + 2) for amount in range(9)}
    matrix = SparseMatrix.from_diagonals(
        params.slot_count,
        {amount: np.full(params.slot_count, weight, dtype=np.complex128) for amount, weight in diagonals.items()},
    )
    # Two stages per phase, at input levels 4/3/2/1: composed giants run one level below their stage.
    monkeypatch.setattr(BootstrappingParams, "matrix_stages", lambda self, params: ((4, (matrix, matrix)), (2, (matrix, matrix))))
    keys = keygen(params, hoist_bootstrap_rotations=hoist)
    for amount in babies:
        assert keys.get_galois_key(steps_to_galois_elt(amount, params.n)).target_ring == params.ring_at(4)
    for amount in giants:
        assert keys.get_galois_key(steps_to_galois_elt(amount, params.n)).target_ring == params.ring_at(giant_level)
    assert keys.get_galois_key(2 * params.n - 1).target_ring == params.ring_at(2)

    with keys.context():
        values = np.asarray([complex((i % 7) / 16, (i % 11) / 32) for i in range(params.slot_count)])
        expected = sum(weight * np.roll(values, -amount) for amount, weight in diagonals.items())
        for level in (4, 3, 2, 1):
            result = LinearTransform.encode(matrix, level=level)(encrypt(encode(values, msg_bound=1, level=level)))
            assert result.level == level - 1
            assert decode(decrypt(result)) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("hoist", [False, True])
def test_non_prefix_bootstrap(hoist: bool):
    params = non_prefix_params()
    with keygen(params, hoist_bootstrap_rotations=hoist).context(default_level=6):
        values = [0.5, -1.0, 0.125 + 0.0625j, -0.25j]
        result = bootstrap(encrypt(encode(values, msg_bound=1)), log_bound=0)

        assert result.level == params.bootstrap_output_level == 5
        assert decode(decrypt(result), len(values)) == pytest.approx(values, abs=1e-4)


@pytest.fixture(scope="module")
def small_boot_ctx() -> CKKSContext:
    boot_params = BootstrappingParams(
        cts_depth=2,
        stc_depth=2,
        exp_degree=63,
        squaring_count=3,
        failure_probability=2**-16,
    )
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=50,
        log_range=7,
        dtype=dtypes.uint32,
        boot_params=boot_params,
        bootstrap_level_bit_length=60,
        p_prime_count=4,
        security=None,
    )
    return keygen(params).context(track_plaintext_values=True)


class _StopBootstrap(Exception):
    pass


def test_mod_raise_bounds_dense_collective_secret():
    params = _small_encapsulation_params()
    assert params.boot_params is not None
    params = replace(params, boot_params=replace(params.boot_params, sparse_encapsulation_weight=None))
    keys = CKKSKeySet(params, multiparty=MultipartyParams(num_parties=3, threshold=2))
    with keys.context():
        ct = Ciphertext.zero(level=0)
        half = ct.ring.q // 2
        both = Poly([half] * params.n, ct.ring)
        ct = ct.replace(a=both, b=both)
        # Three identical ternary shares make all terms in the constant coefficient add.
        secret = Poly([3] + [-3] * (params.n - 1), params.highest_ring)
        raised = ct._mod_raise(secret_l1_norm_bound=keys.secret_l1_norm_bound)
        constant = decrypt(raised, secret=secret).m.to_centered_list()[0]
        # At the midpoint, CUDA's CRT rounding can choose the adjacent representative.
        assert constant in {(3 * params.n + 1) * lift for lift in (half, half - ct.ring.q)}
        assert abs(constant) <= raised.noise_info.msg_bound < raised.ring.q // 2


def _bootstrap_base_ciphertext(ct: Ciphertext, log_bound: int = 0) -> Ciphertext:
    captured: list[Ciphertext] = []

    def capture(ct: Ciphertext, *_args: object, **_kwargs: object) -> Ciphertext:
        captured.append(ct)
        raise _StopBootstrap

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Ciphertext, "_mod_raise", capture)
        monkeypatch.setattr(bootstrap_module, "_switch_secret", capture)
        with pytest.raises(_StopBootstrap):
            bootstrap(ct, log_bound=log_bound)
    assert len(captured) == 1
    return captured[0]


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_coefficients_to_slots(ctx: CKKSContext):
    with ctx:
        n = ctx.params.n
        slots = n // 2
        log_n = n.bit_length() - 1
        boot_params = ctx.params.boot_params
        assert boot_params is not None
        ct = random_ciphertext(msg_bound=2, level=ctx.params.highest_level)
        pt = decrypt(ct)
        coeffs = list(pt.m)

        # Center coefficients
        q = pt.m.ring.q
        half_q = q >> 1
        centered_coeffs = [c - q if c > half_q else c for c in coeffs]

        # Apply coefficients_to_slots
        ct1, ct2 = coefficients_to_slots(ct, levels=boot_params.cts_depth)

        # Decrypt the outputs
        slots1 = decode(decrypt(ct1))
        slots2 = decode(decrypt(ct2))

        # The output should be scaled coefficients in bit-reversed order
        # ct1 contains even-indexed coefficients: m_rev(0), m_rev(2), m_rev(4), ...
        # ct2 contains odd-indexed coefficients: m_rev(1), m_rev(3), m_rev(5), ...
        # Both are scaled by 1/n (which is already applied in coefficients_to_slots)

        scale = pt.scale
        for i in range(slots):
            # Even coefficients go to ct1
            even_idx = 2 * i
            rev_even_idx = bit_reverse(even_idx, log_n)
            expected_even = centered_coeffs[rev_even_idx] / scale
            assert slots1[i].real == pytest.approx(expected_even, abs=1e-10), f"ct1 slot {i} mismatch"

            # Odd coefficients go to ct2
            odd_idx = 2 * i + 1
            rev_odd_idx = bit_reverse(odd_idx, log_n)
            expected_odd = centered_coeffs[rev_odd_idx] / scale
            assert slots2[i].real == pytest.approx(expected_odd, abs=1e-10), f"ct2 slot {i} mismatch"


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_mod_one(ctx: CKKSContext):
    with ctx:
        boot_params = ctx.params.boot_params
        assert boot_params is not None
        range_val = bootstrap_module._bootstrap_mod_one_range(ctx.keyset)
        slots = ctx.params.slot_count

        # Create values of the form k + eps where k is an integer and eps is very small
        # The mod_one function should reduce these to approximately eps
        max_k = int(range_val)
        num_k_values = 2 * max_k + 1
        base_values = []
        expected_results = []
        for i in range(slots):
            k = (i % num_k_values) - max_k
            eps = 1e-6 * (((i % 21) - 10) / 10)  # values around 1e-6
            base_values.append(float(k + eps))
            expected_results.append(eps)

        ct = encrypt(encode(base_values, msg_bound=range_val, level=ctx.params.highest_level))

        result = mod_one(
            ct,
            range=range_val,
            exp_degree=boot_params.exp_degree,
            squaring_count=boot_params.squaring_count,
        )

        decrypted = decode(decrypt(result))

        assert decrypted == pytest.approx([complex(val) for val in expected_results], abs=2e-08)


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_slots_to_coefficients(ctx: CKKSContext):
    with ctx:
        boot_params = ctx.params.boot_params
        assert boot_params is not None

        ct = random_ciphertext(msg_bound=2, level=ctx.params.highest_level)
        original_values = ct.values
        assert original_values is not None

        # Apply coefficients_to_slots then slots_to_coefficients (roundtrip)
        ct1, ct2 = coefficients_to_slots(ct, levels=boot_params.cts_depth)
        ct_roundtrip = slots_to_coefficients(ct1, ct2, levels=boot_params.stc_depth)

        # Verify roundtrip yields identity
        decrypted = decode(decrypt(ct_roundtrip))
        assert decrypted == pytest.approx(original_values, abs=2e-9)


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_bootstrap(ctx):
    with ctx:
        ct = random_ciphertext(msg_bound=10)
        ct_squared = ct * ct

        result = ct_squared.bootstrap(log_bound=7)

        assert result.level == ctx.params.bootstrap_output_level

        decrypted = decode(decrypt(result))
        assert decrypted == pytest.approx(result.values, abs=2e-1)


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_bootstrap_jit_prune(ctx: CKKSContext):
    """Bootstrap under @ren.jit(prune=True) — exercises JIT capture, pruning, and replay."""
    with ctx:

        @ren.jit(prune=True)
        def jit_bootstrap(ct: Ciphertext) -> Ciphertext:
            return bootstrap(ct, log_bound=7)

        for _ in range(2):
            ct = random_ciphertext(msg_bound=10)
            ct_squared = ct * ct
            expected = ct_squared.values
            assert expected is not None

            result = jit_bootstrap(ct_squared)

            assert result.level == ctx.params.bootstrap_output_level
            decrypted = decode(decrypt(result))
            assert decrypted == pytest.approx(expected, abs=2e-1)


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_high_precision_bootstrap_plaintext_matches_decode():
    """Boot-base rescale uses bootstrap_base_level.scale; tracked slot values must follow the transition ratio.

    Bootstrapping is exercised on CUDA (`DEV=cuda`); skipped when `DEV=python` (see conftest).
    """
    params = high_precision_params()
    ctx = keygen(params).realize().context(track_plaintext_values=True)
    with ctx:
        vals = [0.01 + 0j, 0.02 + 0j, 0.03 + 0j, 0.04 + 0j]
        ct = encrypt(encode(vals, msg_bound=0.05))
        ct_boot = bootstrap(ct, log_bound=0)
        dec_boot = decode(decrypt(ct_boot), len(vals))
        assert ct_boot.values is not None
        for a, b in zip(dec_boot, ct_boot.values):
            assert a.real == pytest.approx(b.real, abs=2e-1)
            assert a.imag == pytest.approx(b.imag, abs=2e-1)


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_high_precision_precompute_covers_bootstrap_plans(monkeypatch: pytest.MonkeyPatch):
    params = high_precision_params()
    ctx = keygen(params).realize().context()
    ctx.precompute_bootstrap(log_bound=0)

    def unexpected_encode(*_args: object, **_kwargs: object) -> LinearTransform:
        pytest.fail("bootstrap requested a transform that was not precomputed")

    monkeypatch.setattr(LinearTransform, "encode", classmethod(unexpected_encode))
    with ctx:
        ct = random_ciphertext(msg_bound=1)

        result = bootstrap(ct, log_bound=0)
        assert result.level == params.bootstrap_output_level

        original = decode(decrypt(ct))
        decrypted = decode(decrypt(result))
        assert decrypted == pytest.approx(original, abs=1e-12)


def _small_encapsulation_params(*, h: int = 32, secret_key_dist: KeyDist | None = None) -> CKKSParams:
    """Use a temporary key of weight ``h`` and a dense main key unless overridden.

    The encapsulation method is described in https://eprint.iacr.org/2022/024.
    """
    boot_params = BootstrappingParams(
        cts_depth=2,
        stc_depth=2,
        exp_degree=63,
        squaring_count=2,
        failure_probability=2**-16,
        sparse_encapsulation_weight=h,
    )
    return CKKSParams.generate_prefix(
        log_n=10,
        depth=5,
        log_precision=50,
        log_range=10,
        dtype=dtypes.uint32,
        p_prime_count=4,
        boot_params=boot_params,
        bootstrap_level_bit_length=60,
        security=None,
        secret_key_dist=secret_key_dist,
    )


@pytest.fixture(scope="module")
def encapsulation_keys() -> CKKSKeySet:
    set_global_rng_seed(43)
    return keygen(_small_encapsulation_params()).realize()


def test_encapsulation_keygen():
    keys = keygen(_small_encapsulation_params())
    params = keys.params
    assert params.boot_params is not None
    assert params.bootstrap_base_level is not None
    assert keys.dense_to_sparse_key is not None
    assert keys.sparse_to_dense_key is not None
    dense_to_sparse_key = keys.dense_to_sparse_key
    sparse_to_dense_key = keys.sparse_to_dense_key

    # The dense->sparse key is the only material encrypted under the ephemeral
    # sparse secret; a single ks prime keeps its modulus minimal.
    assert dense_to_sparse_key.p_prime_count == 1
    assert dense_to_sparse_key.ring_qt.moduli[1:] == params.bootstrap_base_level.moduli

    assert sparse_to_dense_key.p_prime_count == len(params.ks_primes)
    assert sparse_to_dense_key.ring_qt.moduli[len(params.ks_primes) :] == params.highest_ring.moduli
    assert sparse_to_dense_key.digit_sizes == params.ks_digit_sizes
    assert all(poly.domain is Domain.NTT for key in (dense_to_sparse_key, sparse_to_dense_key) for digit in key.digits for poly in (digit.a, digit.b))


def test_encapsulation_switch_mod_raise_round_trip(encapsulation_keys: CKKSKeySet):
    """dense->sparse switch, mod-raise, sparse->dense switch preserves the message mod q_0."""
    with encapsulation_keys.context():
        params = encapsulation_keys.params
        assert encapsulation_keys.dense_to_sparse_key is not None and encapsulation_keys.sparse_to_dense_key is not None
        values = [((i % 21) - 10) / 16 for i in range(params.slot_count)]
        ct = _bootstrap_base_ciphertext(encrypt(encode(values, msg_bound=1)))
        coeffs_before = list(decrypt(ct).m)
        q_0 = prod(int(m) for m in ct.ring.moduli)

        ct_sparse = _switch_secret(ct, encapsulation_keys.dense_to_sparse_key)
        assert params.boot_params is not None and params.boot_params.sparse_encapsulation_weight is not None
        ct_up = ct_sparse._mod_raise(secret_l1_norm_bound=params.boot_params.sparse_encapsulation_weight)
        assert ct_up.ring == params.highest_ring
        ct_back = _switch_secret(ct_up, encapsulation_keys.sparse_to_dense_key)
        coeffs_after = list(decrypt(ct_back).m)

        # Error budget from the key-switch spec, not from observed output: each
        # hybrid key-switch adds per-coefficient noise at most about
        # d * n * B_err * max(q_segment)/(2*P) (external product of the digit
        # polynomials with the key digits' noise, rescaled by P) plus O(n)
        # rescale rounding. With n=2^10, B_err = 6*sigma ~ 2^4.3, and
        # q_segment/P <= 2^2 for both keys, each switch stays below 2^18, so
        # 2^20 over-bounds the pair while remaining ~2^-30 relative to the
        # 2^50 scale and far below q_0/2 (no wraparound tolerated).
        max_switch_noise = 2**20

        def centered(x: int, q: int) -> int:
            x %= q
            return x - q if x > q // 2 else x

        for before, after in zip(coeffs_before, coeffs_after, strict=True):
            assert abs(centered(int(after), q_0) - centered(int(before), q_0)) <= max_switch_noise


def test_bootstrap_with_sparse_encapsulation(encapsulation_keys: CKKSKeySet):
    with encapsulation_keys.context(track_plaintext_values=True):
        values = [((i % 21) - 10) / 16 for i in range(encapsulation_keys.params.slot_count)]
        ct = encrypt(encode(values, msg_bound=1)).to_level(0)

        result = bootstrap(ct, log_bound=0)

        assert result.level == encapsulation_keys.params.bootstrap_output_level
        assert result.values == pytest.approx(values)
        decrypted = [v.real for v in decode(decrypt(result))]
        # EvalMod's sine approximation limits these parameters to about 25 bits
        # of precision. The tolerance allows about five bits of headroom.
        assert decrypted == pytest.approx(values, abs=1e-6)


def test_bootstrap_with_sparse_encapsulation_and_sparse_scheme_secret():
    """Encapsulation with a sparse (rather than the default dense) scheme secret."""
    set_global_rng_seed(43)
    params = _small_encapsulation_params(secret_key_dist=SparseTernary(n=1 << 10, h=128))
    sparse_ctx = keygen(params).realize().context()
    with sparse_ctx:
        values = [((i % 21) - 10) / 16 for i in range(params.slot_count)]
        ct = encrypt(encode(values, msg_bound=1)).to_level(0)

        result = bootstrap(ct, log_bound=0)

        assert result.level == params.bootstrap_output_level
        decrypted = [v.real for v in decode(decrypt(result))]
        # Same bound as test_bootstrap_with_sparse_encapsulation: the scheme
        # secret's weight does not enter K (the ephemeral h=32 does), so the
        # EvalMod approximation floor is unchanged.
        assert decrypted == pytest.approx(values, abs=1e-6)


def test_bootstrap_encapsulation_requires_keys():
    keys = keygen(_small_encapsulation_params())
    params = keys.params
    assert keys.sk is not None
    bare_ctx = replace(keys, dense_to_sparse_key=None, sparse_to_dense_key=None).context()
    with bare_ctx:
        ct = encrypt(encode([0.5] * params.slot_count, msg_bound=1)).to_level(0)
        with pytest.raises(RuntimeError, match="encapsulation keys"):
            bootstrap(ct, log_bound=0)


def test_sparse_encapsulation_param_validation():
    # The ephemeral encapsulation weight must be at least 32.
    with pytest.raises(ValueError, match="at least 32"):
        _small_encapsulation_params(h=16)

    params = _small_encapsulation_params()
    assert params.boot_params is not None
    assert params.bootstrap_base_level is not None

    # A sparse scheme secret must have weight at least 128, with or without
    # encapsulation -- the ephemeral weight bounds the mod-raise, not the scheme
    # secret's own exposure.
    non_encap_boot = replace(params.boot_params, sparse_encapsulation_weight=None)
    with pytest.raises(InvalidParameterError, match="at least 128"):
        replace(params, boot_params=non_encap_boot, secret_key_dist=SparseTernary(n=params.n, h=32))
    with pytest.raises(InvalidParameterError, match="at least 128"):
        replace(params, secret_key_dist=SparseTernary(n=params.n, h=32))

    # Changing the main key leaves the mod-one range unchanged when the temporary key has the same weight.
    encapsulated_range = bootstrap_module._bootstrap_mod_one_range(CKKSKeySet(params))
    sparse_scheme = replace(params, secret_key_dist=SparseTernary(n=params.n, h=192))
    assert bootstrap_module._bootstrap_mod_one_range(CKKSKeySet(sparse_scheme)) == encapsulated_range

    # The same polynomial is too coarse for the dense main key used directly.
    direct = replace(params, boot_params=non_encap_boot)
    with pytest.raises(InvalidParameterError, match="mod-one approximation is too coarse"):
        direct.context().precompute_bootstrap(log_bound=0)


@pytest.fixture(scope="module")
def encap_default_ctx():
    set_global_rng_seed(44)
    return keygen(CKKSParams.default()).realize().context()


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_bootstrap_sparse_encapsulation_default_params(encap_default_ctx: CKKSContext):
    with encap_default_ctx:
        values = random_values_with_norm(encap_default_ctx.params.slot_count, norm=10)
        ct = encrypt(encode(values, msg_bound=10))
        ct_squared = ct * ct

        result = bootstrap(ct_squared, log_bound=4)

        assert result.level == encap_default_ctx.params.bootstrap_output_level
        decrypted = decode(decrypt(result))
        # The relative tolerance allows about 40 times the observed relative error.
        # Some slots are near zero after squaring. For those, the absolute tolerance
        # allows about four times the largest observed absolute error.
        assert decrypted == pytest.approx([value * value for value in values], rel=1e-4, abs=1e-3)


def test_high_precision_bootstrap_sparse_encapsulation():
    """The bootstrap base uses a different modulus basis from every scheduled level."""
    params = high_precision_params()
    assert params.boot_params is not None and params.boot_params.sparse_encapsulation_weight == 32
    assert params.default_level == params.bootstrap_output_level == 6
    assert params.bootstrap_base_level is not None
    assert all(level.moduli != params.bootstrap_base_level.moduli for level in params.levels)


@pytest.mark.slow()
@pytest.mark.requires_cuda()
@pytest.mark.parametrize("level_zero", [False, True], ids=["highest-level", "level-zero"])
def test_high_precision_bootstrap_jit_replays_meta_bootstrap_calls(monkeypatch: pytest.MonkeyPatch, level_zero: bool):
    """Meta-bootstrap reuses its function body and accepts fresh inputs on replay."""
    params = high_precision_params()
    ctx = keygen(params).realize().context()
    ctx.precompute_bootstrap(log_bound=0)
    scheduler = importlib.import_module("ren.engine.schedule")
    assert params.boot_params is not None
    iterations = len(params.boot_params.bounds(0))
    assert iterations > 1
    monkeypatch.setattr(scheduler, "_prepare_cache", OrderedDict())
    monkeypatch.setattr(scheduler, "_schedule_cache", OrderedDict())
    bufferize_plan = scheduler.rules.bufferize_plan
    outer_calls = outer_compilations = 0

    def function_calls(root: Node) -> list[Node]:
        return [
            node
            for node in root.toposort()
            if node.op is Op.CALL and (node.src[0].op is Op.LINEAR or (node.src[0].op is Op.SINK and node.src[0].arg is None))
        ]

    def track_compilation(root):
        nonlocal outer_calls, outer_compilations
        # The caller invokes bootstrap bodies; each body invokes mod_reduce.
        if calls := function_calls(root):
            if any(function_calls(call.src[0]) for call in calls):
                outer_calls += len(calls)
            else:
                outer_compilations += 1
        return bufferize_plan(root)

    monkeypatch.setattr(scheduler.rules, "bufferize_plan", track_compilation)

    with ctx:

        @ren.jit(prune=True)
        def jit_bootstrap(ct: Ciphertext) -> Ciphertext:
            return bootstrap(ct, log_bound=0)

        for _ in range(2):
            ct = random_ciphertext(msg_bound=1)
            if level_zero:
                ct = ct.to_level(0)
            expected = decode(decrypt(ct))
            result = jit_bootstrap(ct)

            assert decode(decrypt(result)) == pytest.approx(expected, abs=1e-12)
            assert outer_calls == iterations
            assert outer_compilations == 1


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_high_precision_bootstrap_preserves_plaintext_tracking():
    params = high_precision_params()
    ctx = keygen(params).realize().context(track_plaintext_values=True)
    with ctx:
        ct = random_ciphertext(msg_bound=1)
        assert ct.values is not None

        result = bootstrap(ct, log_bound=0)

        assert result.values is not None
        assert result.values == pytest.approx(ct.values, abs=1e-9)


def test_bootstrap_normalizes_and_tracks_realized_ratio(small_boot_ctx: CKKSContext):
    with small_boot_ctx:
        assert small_boot_ctx.params.boot_params is not None
        assert small_boot_ctx.params.bootstrap_base_level is not None
        values = [4.0, -8.0]
        ct = encrypt(encode(values, msg_bound=8.0))
        ct_level_zero = ct.to_level(0)
        step = small_boot_ctx.params.bootstrap_transition(
            current_scale=ct_level_zero.scale,
            correction_factor=BigFloat(2) ** -3,
        )
        ct_down = _bootstrap_base_ciphertext(ct, log_bound=3)
        assert ct_down.ring.moduli == small_boot_ctx.params.bootstrap_base_level.moduli
        tracked = ct_down.values
        assert tracked is not None
        assert tracked[: len(values)] == pytest.approx([value * float(step.message_ratio) for value in values])
        assert max(abs(value) for value in tracked) <= 1 + 1e-6
        assert decode(decrypt(ct_down), len(values)) == pytest.approx(tracked[: len(values)], abs=1e-3)


def test_bootstrap_accepts_level_zero_and_rejects_unrepresentable_bound(small_boot_ctx: CKKSContext):
    with small_boot_ctx:
        assert small_boot_ctx.params.boot_params is not None
        assert small_boot_ctx.params.bootstrap_base_level is not None
        assert _bootstrap_base_ciphertext(Ciphertext.zero(level=0)).ring.moduli == small_boot_ctx.params.bootstrap_base_level.moduli
        with pytest.raises(InvalidParameterError, match="log_range"):
            bootstrap(Ciphertext.zero(level=0), log_bound=1000)


def test_bootstrap_without_q1_only_multiplies_on_existing_ring():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=50,
        log_range=0,
        dtype=dtypes.uint32,
        boot_params=BootstrappingParams(
            cts_depth=2,
            stc_depth=2,
            exp_degree=63,
            squaring_count=3,
            failure_probability=2**-16,
        ),
        bootstrap_level_bit_length=60,
        p_prime_count=4,
        security=None,
    )
    ctx = keygen(params).context(track_plaintext_values=True)
    with ctx:
        assert params.boot_params is not None
        assert params.bootstrap_base_level is not None
        values = [0.5, -1.0]
        ct = encrypt(encode(values, msg_bound=1.0))
        ct_level_zero = ct.to_level(0)
        ct_down = _bootstrap_base_ciphertext(ct)
        assert params.bootstrap_base_level.moduli == params.ring_at(0).moduli
        assert ct_down.ring == ct_level_zero.ring
        assert ct_down.level == 0
        assert ct_down.values is not None
        assert max(abs(value) for value in ct_down.values) <= 1 + 1e-6


@pytest.mark.slow()
@pytest.mark.requires_cuda()
def test_bootstrap_without_q1_restores_values():
    params = CKKSParams.generate_prefix(
        log_n=16,
        depth=10,
        log_precision=50,
        log_range=0,
        dtype=dtypes.uint64,
        boot_params=BootstrappingParams(
            cts_depth=3,
            stc_depth=3,
            exp_degree=63,
            squaring_count=3,
            failure_probability=2**-64,
        ),
        bootstrap_level_bit_length=60,
        p_prime_count=3,
        security=None,
    )
    ctx = keygen(params).realize().context()
    with ctx:
        values = [0.5, -1.0]
        result = bootstrap(encrypt(encode(values, msg_bound=1.0)), log_bound=0)
        assert decode(decrypt(result), len(values)) == pytest.approx(values, abs=2e-4)


def test_bootstrapping_params_allow_multiplication_to_level_zero(small_boot_ctx: CKKSContext):
    with small_boot_ctx:
        ct = Ciphertext.one(level=1)
        assert (ct * ct).level == 0
        assert (ct * 1.0).level == 0


def test_matrix_stages_are_encoded_once_per_context():
    params = small_params()
    matrices = l_matrix_partial_factorization(params.logn, 1)
    rotations = set().union(*(amounts for matrix in matrices for amounts in LinearTransform.rotation_amounts(matrix).values()))
    with keygen(params, rotations=rotations).context() as ctx:
        ct = encrypt(encode(1, msg_bound=1))
        l_matrix(ct, levels=1)
        first = dict(ctx._bootstrap_transforms)
        l_matrix(ct, levels=1)

        assert len(first) == 1
        assert ctx._bootstrap_transforms == first


def test_bootstrap_approximation_accounts_for_the_collective_key_and_requested_gain():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=1,
        log_precision=50,
        log_range=0,
        dtype=dtypes.uint64,
        boot_params=BootstrappingParams(cts_depth=2, stc_depth=2, exp_degree=63, squaring_count=3, failure_probability=2**-64),
        secret_key_dist=SparseTernary(1024, 192),
        security=None,
    )
    params.context().precompute_bootstrap(log_bound=0)

    assert params.boot_params is not None
    iterative = replace(params, boot_params=replace(params.boot_params, precision_gains=(40,)))
    with pytest.raises(InvalidParameterError, match="mod-one approximation"):
        iterative.context().precompute_bootstrap(log_bound=0)

    collective = CKKSKeySet(params=params, multiparty=MultipartyParams(num_parties=3))
    with collective.context() as ctx:
        with pytest.raises(InvalidParameterError, match="mod-one approximation"):
            ctx.precompute_bootstrap(log_bound=0)
        with pytest.raises(InvalidParameterError, match="mod-one approximation"):
            bootstrap(Ciphertext.zero(level=0), log_bound=0)


def test_bootstrap_precision_accounts_for_sine_error():
    params = CKKSParams.default(log_n=10, depth=1, bootstrap_message_ratio=256, security=None)
    assert params.boot_params is not None
    # A constant slot of 1 comes back as 256*sin(2*pi/256)/(2*pi), about 0.9998996.
    boot = replace(params.boot_params, precision_gains=(13,))
    replace(params, boot_params=boot).context().precompute_bootstrap(log_bound=0)
    with pytest.raises(InvalidParameterError, match="larger bootstrap_message_ratio or a smaller precision gain"):
        replace(params, boot_params=replace(boot, precision_gains=(14,))).context().precompute_bootstrap(log_bound=0)
