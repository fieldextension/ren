import asyncio
import gc
import pickle
import weakref
from contextlib import nullcontext
from dataclasses import replace
from typing import assert_type

import pytest

from ren import jit
from ren.context import Context, DEV
from ren.device import Device
from ren.dtype import dtypes
from ren.poly import Poly
from ren.polydist import SparseTernary
from ren.random import rng_from_int_seed, set_global_rng_seed
from ren.ring import Domain
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError
from ren.ckks import (
    CKKSContext,
    CKKSKeySet,
    CKKSLevel,
    CKKSParams,
    Ciphertext,
    Plaintext,
    PublicKey,
    RLWECiphertext,
    decode,
    decrypt,
    encode,
    encrypt,
    keygen,
    generate_secret,
)
from ren.ckks.context import context
from ren.ckks.ckks import _decompose_and_modraise, _keyswitch_hoisted
from ren.ckks.keys import SwitchKey
from ren.ckks.metadata import CKKSNoiseInfo
from ren.ckks.utils import select_moduli, steps_to_galois_elt
from tests.utils import mixed_prime_ring
from tests.ckks.params import small_params


def test_keygen_rejects_hoisted_bootstrap_rotations_without_bootstrapping():
    with pytest.raises(InvalidParameterError, match="hoist_bootstrap_rotations"):
        keygen(small_params(), hoist_bootstrap_rotations=True)


@pytest.fixture(scope="module")
def keys() -> CKKSKeySet:
    set_global_rng_seed(42)
    params = small_params()
    return keygen(
        params,
        rotations={2**i for i in range(params.logn - 1)},
        conjugation=True,
    ).realize()


def test_keygen_does_not_activate_runtime_context():
    assert context(required=False) is None
    generated = keygen(small_params())

    assert context(required=False) is None
    assert isinstance(generated, CKKSKeySet)
    assert generated.sk is not None
    assert generated.pk is not None
    assert generated.rlk is not None
    assert not generated.galois_keys


def test_keygen_accepts_default_and_per_level_galois_key_requests():
    params = small_params()
    generated = keygen(
        params,
        rotations={params.default_level: {2}, 0: {1}},
        conjugation=1,
    )

    rotation_1_elt = steps_to_galois_elt(1, params.n)
    rotation_2_elt = steps_to_galois_elt(2, params.n)
    conjugation_elt = 2 * params.n - 1
    rotation_1 = generated.get_galois_key(rotation_1_elt)
    rotation_2 = generated.get_galois_key(rotation_2_elt)
    conjugation = generated.get_galois_key(conjugation_elt)
    assert rotation_1.target_ring == params.ring_at(0)
    assert rotation_2.target_ring == params.ring_at(params.default_level)
    assert conjugation.target_ring == params.ring_at(1)

    default_keys = keygen(params, rotations={1})
    default_rotation = default_keys.get_galois_key(steps_to_galois_elt(1, params.n))
    assert default_rotation.target_ring == params.ring_at(params.default_level)


def test_keygen_rejects_invalid_galois_key_requests():
    params = small_params()

    with pytest.raises(InvalidParameterError, match="Galois key level"):
        keygen(params, rotations={params.highest_level + 1: {1}})
    with pytest.raises(InvalidParameterError, match="Too many Galois keys"):
        keygen(params, rotations=range(1, 202))


def test_secret_key_generate():
    params = small_params()
    rng = rng_from_int_seed(42, device="python")
    secret_key = generate_secret(params, rng=rng)

    assert secret_key.ring == params.highest_ring
    assert secret_key.device == "python"
    keys = CKKSKeySet(params=params, sk=secret_key)
    assert keys.sk is secret_key

    with pytest.raises(ParameterMismatchError, match="secret generation requested cuda:3, got RNG on python"):
        generate_secret(params, device="cuda:3", rng=rng)


@pytest.mark.parametrize("level,sparse", [(None, False), (1, True)])
def test_generate_secret_uses_configured_distribution_and_level(level, sparse):
    params = small_params()
    if sparse:
        params = replace(params, secret_key_dist=SparseTernary(params.n, 8))
    secret = generate_secret(params, level=level, rng=rng_from_int_seed(42))
    assert secret.ring == (params.highest_ring if level is None else params.ring_at(level))
    coefficients = secret.to_centered_list()
    assert set(coefficients) <= {-1, 0, 1}
    if sparse:
        assert sum(value != 0 for value in coefficients) == 8
    expected = Poly.rand(params.secret_key_dist, secret.ring, rng=rng_from_int_seed(42))
    assert secret.equals(expected)


@pytest.mark.parametrize("level", [-1, 999])
def test_generate_secret_rejects_invalid_level_before_sampling(level):
    rng = rng_from_int_seed(42)
    with pytest.raises(InvalidParameterError):
        generate_secret(small_params(), level=level, rng=rng)
    assert rng.reserve() == 0


def test_context_is_required_by_default_and_optional_when_requested():
    assert context(required=False) is None
    with pytest.raises(RuntimeError, match="no active CKKSContext"):
        context()

    ctx = keygen(small_params()).context()
    with ctx:
        assert context() is ctx
        assert context(required=False) is ctx


def test_keyset_generate_accepts_existing_secret_key(keys: CKKSKeySet):
    generated = CKKSKeySet.generate(small_params(), rotations={1}, device="python")
    assert isinstance(keys.sk, Poly)
    derived = keygen(keys.params, rotations={1}, secret=keys.sk)

    for candidate in (generated, derived):
        assert candidate.sk is not None
        assert candidate.pk is not None
        assert candidate.rlk is not None
        assert candidate.galois_keys

    assert derived.sk is keys.sk

    target_device = "cuda:3" if keys.device != "cuda:3" else "python"
    with pytest.raises(ParameterMismatchError, match=f"got secret key on {keys.device}"):
        CKKSKeySet.generate(keys.params, secret=keys.sk, device=target_device)


def test_keyset_context_uses_keyset_device():
    keys = keygen(small_params(), rotations={1}, conjugation=True)
    previous_device = DEV.value
    previous_default = Device.DEFAULT
    target_device = "cuda:3" if previous_default != "cuda:3" else "python"
    moved_keys = keys.to(target_device)

    with Context(DEV=previous_device):
        ctx = moved_keys.context()

    assert moved_keys.device == target_device
    assert ctx.device == target_device
    with ctx:
        assert DEV.value == previous_device
        assert Device.DEFAULT == target_device
        assert Plaintext.zero().m.device is None
        assert encode([1, 2], msg_bound=2).m.device == target_device
    assert DEV.value == previous_device
    assert Device.DEFAULT == previous_default
    assert moved_keys.to(target_device) is moved_keys


def test_explicit_keyset_does_not_detect_a_default(monkeypatch):
    monkeypatch.setattr(type(Device), "_select_device", property(lambda self: pytest.fail("unexpected detection")))
    with Context(DEV=""):
        keys = keygen(small_params(), device="python")
        assert keys.device == "python"
        with keys.context():
            assert Device.DEFAULT == "python"
            assert encode([1, 2], msg_bound=2).m.device == "python"


def test_keyset_maps_all_key_material():
    keys = keygen(small_params(), rotations={1}, conjugation=True)

    def material(keyset: CKKSKeySet) -> list[Poly]:
        assert keyset.sk is not None and keyset.pk is not None and keyset.rlk is not None
        return [keyset.sk, keyset.pk.ct.a, keyset.pk.ct.b] + [
            poly for key in (keyset.rlk, *keyset.galois_keys.values()) for digit in key.digits for poly in (digit.a, digit.b)
        ]

    coeff_keys = keys.replace(polys=map(Poly.intt, keys.polys()))
    original_polys, coeff_polys = material(keys), material(coeff_keys)

    assert coeff_keys is not keys
    assert coeff_keys.params is keys.params
    assert type(coeff_keys.galois_keys) is type(keys.galois_keys)
    assert list(coeff_keys.galois_keys) == list(keys.galois_keys)
    assert all(mapped is not original for mapped, original in zip(coeff_polys, original_polys, strict=True))
    assert all(poly.domain == Domain.COEFF for poly in coeff_polys)


def test_jit_replays_replaced_galois_key(keys: CKKSKeySet):
    galois_elt, key = next(iter(keys.galois_keys.items()))
    digit = key.digits[0]
    changed_digit = replace(digit, b=(digit.b + 1).realize())
    changed_key = replace(key, digits=(changed_digit, *key.digits[1:]))
    changed_keys = replace(keys, galois_keys={**keys.galois_keys, galois_elt: changed_key})
    captures = 0

    @jit
    def read_key(keyset: CKKSKeySet):
        nonlocal captures
        captures += 1
        return keyset.galois_keys[galois_elt].digits[0].b + 1

    assert read_key(keys).equals(digit.b + 1)
    assert read_key(changed_keys).equals(changed_digit.b + 1)
    assert captures == 1


def test_context_rejects_nesting_without_disturbing_outer_context():
    keys = keygen(small_params())
    outer = keys.context()
    inner = keys.public().context()

    with outer:
        with pytest.raises(RuntimeError, match="only be one CKKSContext"):
            with inner:
                pass
        assert context() is outer
        assert Device.DEFAULT == outer.device


def test_context_and_device_are_task_local():
    keys = keygen(small_params())

    async def use_context(ctx: CKKSContext) -> str:
        with ctx:
            await asyncio.sleep(0)
            assert context() is ctx
            return Device.DEFAULT

    async def run() -> tuple[str, ...]:
        return tuple(
            await asyncio.gather(
                use_context(keys.to("python").context()),
                use_context(keys.to("cuda:3").context()),
            )
        )

    assert asyncio.run(run()) == ("python", "cuda:3")


def test_context_holds_keyset():
    keys = keygen(small_params())
    ctx = keys.context()

    assert ctx.keyset is keys
    assert ctx.params is keys.params


def test_keygen_device_and_rng_must_match():
    rng = rng_from_int_seed(42, device="python")
    rng_keys = keygen(small_params(), rng=rng)
    keys = keygen(small_params(), device="python", rng=rng)

    assert rng_keys.device == "python"
    assert keys.device == "python"

    with pytest.raises(ParameterMismatchError, match="requested cuda:3, got RNG on python"):
        keygen(small_params(), device="cuda:3", rng=rng)

    assert isinstance(keys.sk, Poly)
    with pytest.raises(ParameterMismatchError, match="CKKS key set is on cuda:3, got RNG on python"):
        keygen(keys.params, secret=keys.sk.to("cuda:3"), rng=rng)


def test_single_party_keyset_uses_secret_distribution_norms():
    keys = CKKSKeySet(small_params())
    secret = keys.params.secret_key_dist
    assert keys.secret_l1_norm_bound == secret.coeff_1_norm_bound()
    assert keys.secret_expected_squared_l2_norm == secret.expected_squared_l2_norm()


def test_keyset_rejects_mixed_key_devices():
    keys = keygen(small_params())
    assert isinstance(keys.sk, Poly)
    assert keys.pk is not None
    target_device = "cuda:3" if keys.device != "cuda:3" else "python"

    with pytest.raises(ParameterMismatchError, match="polynomial device mismatch"):
        replace(keys, pk=keys.pk.to(target_device))


def test_switch_key_only_requires_matching_devices_for_used_digits(keys: CKKSKeySet):
    assert keys.rlk is not None
    assert len(keys.rlk.digits) > 1
    digits = (*keys.rlk.digits[:-1], keys.rlk.digits[-1].to("cuda:999"))

    key = SwitchKey(digits=digits, digit_sizes=keys.rlk.digit_sizes)
    low = Poly.const(7, keys.params.ring_at(0), device=keys.device)
    hoisted = _decompose_and_modraise(low, key)
    actual = _keyswitch_hoisted(hoisted, key)
    expected = _keyswitch_hoisted(hoisted, keys.rlk)
    assert all(a.equals(b) for a, b in zip(actual.polys(), expected.polys(), strict=True))

    high = Poly.const(7, keys.params.highest_ring, device=keys.device)
    with pytest.raises(ValueError, match="device mismatch"):
        _keyswitch_hoisted(_decompose_and_modraise(high, key), key)


def test_keyset_rejects_key_material_from_another_ring():
    keys = keygen(small_params())
    assert isinstance(keys.sk, Poly)
    assert keys.pk is not None
    lower_ring = keys.params.ring_at(0)

    with pytest.raises(ParameterMismatchError, match="secret key ring"):
        replace(keys, sk=keys.sk.to_quotient_ring(lower_ring))

    with pytest.raises(ParameterMismatchError, match="secret key ring"):
        keygen(keys.params, secret=keys.sk.to_quotient_ring(lower_ring))

    with pytest.raises(ParameterMismatchError, match="public key ring"):
        replace(keys, pk=replace(keys.pk, ct=keys.pk.ct.map_polys(lambda poly: poly.to_quotient_ring(lower_ring))))


def test_highest_ring_owns_keys_and_projects_digits_to_non_prefix_level():
    ring = mixed_prime_ring(1024, dtype=dtypes.uint32)
    highest_moduli = ring.moduli[:8]
    levels = (
        CKKSLevel(moduli=highest_moduli[2:4], scale=1 << 22),
        CKKSLevel(moduli=highest_moduli[2:6], scale=1 << 40),
        CKKSLevel(moduli=highest_moduli, scale=1 << 44),
    )
    ks_primes = select_moduli(n=ring.n, num_primes=3, bit_length=29, exclude=set(highest_moduli))
    params = CKKSParams.from_levels(
        n=ring.n,
        dtype=dtypes.uint32,
        levels=levels,
        ks_primes=ks_primes,
        log_range=0,
        ks_digit_sizes=(3, 3, 2),
        security=None,
    )
    generated = keygen(params)

    assert generated.sk is not None and generated.sk.ring == params.highest_ring
    assert generated.pk is not None and generated.pk.ct.ring == params.highest_ring
    assert generated.rlk is not None and generated.rlk.target_ring == params.highest_ring

    values = [complex((i % 7) / 64, -(i % 11) / 128) for i in range(params.slot_count)]
    with generated.context(default_level=1):
        result = decode(decrypt(encrypt(encode(values, msg_bound=1)) ** 2))
    assert result == pytest.approx([value**2 for value in values], abs=2e-3)


def test_keyset_owns_immutable_galois_keys():
    keys = keygen(small_params(), rotations={1}, conjugation=True)
    assert keys.pk is not None
    source = dict(keys.galois_keys)
    owned = replace(keys, galois_keys=source)

    source.clear()
    for candidate in (owned, pickle.loads(pickle.dumps(owned))):
        assert candidate.galois_keys.keys() == keys.galois_keys.keys()
        assert all(a.equals(b) for a, b in zip(candidate.polys(), keys.polys(), strict=True))
        galois_elt, key = next(iter(candidate.galois_keys.items()))
        with pytest.raises(TypeError):
            candidate.galois_keys[galois_elt] = key  # ty: ignore[invalid-assignment]


def test_keyset_rejects_invalid_galois_element():
    keys = keygen(small_params(), rotations={1})
    assert keys.pk is not None
    galois_elt, key = next(iter(keys.galois_keys.items()))

    with pytest.raises(InvalidParameterError, match="invalid Galois element"):
        replace(keys, galois_keys={2: key})
    with pytest.raises(InvalidParameterError, match="invalid Galois element"):
        replace(keys, galois_keys={float(galois_elt): key})  # type: ignore[dict-item]


def test_keyset_rejects_noncanonical_galois_element():
    """A Galois element is its representative in (0, 2n): a congruent integer outside that range is rejected."""
    keys = keygen(small_params(), rotations={1})
    assert keys.pk is not None
    galois_elt, key = next(iter(keys.galois_keys.items()))

    for noncanonical in (galois_elt + 2 * keys.params.n, galois_elt - 2 * keys.params.n):
        with pytest.raises(InvalidParameterError, match="invalid Galois element"):
            replace(keys, galois_keys={noncanonical: key})


def test_keyset_rejects_partial_encapsulation_key_pair():
    keys = keygen(small_params())
    assert keys.pk is not None
    assert keys.rlk is not None

    with pytest.raises(ParameterMismatchError, match="provided together"):
        replace(keys, dense_to_sparse_key=keys.rlk)


def test_keyset_accepts_relinearization_key_with_reordered_p_basis(keys: CKKSKeySet):
    assert isinstance(keys.sk, Poly)
    assert keys.pk is not None
    assert keys.rlk is not None
    params = keys.params
    reordered = replace(
        keys,
        params=replace(params, ks_primes=tuple(reversed(params.ks_primes))),
        galois_keys={},
    )
    values = [0.125] * params.slot_count

    with reordered.context():
        ct = encrypt(encode(values, msg_bound=1))
        result = decode(decrypt(ct * ct))

    assert result == pytest.approx([value**2 for value in values], abs=1e-4)


def test_keyset_rejects_relinearization_key_with_another_p_basis():
    keys = keygen(small_params())
    assert keys.pk is not None
    assert keys.rlk is not None
    params = keys.params
    other_ks_primes = tuple(
        select_moduli(
            n=params.n,
            num_primes=len(params.ks_primes),
            bit_length=params.ks_primes[0].bit_length(),
            exclude=set(params.highest_ring.moduli + params.ks_primes),
        )
    )

    with pytest.raises(ParameterMismatchError, match="P basis"):
        replace(
            keys.public(),
            params=replace(params, ks_primes=other_ks_primes),
            galois_keys={},
        )


def test_keyset_public(keys: CKKSKeySet):
    public_keys = keys.public()

    assert isinstance(keys.sk, Poly)
    assert public_keys.sk is None
    assert public_keys.public() is public_keys


def test_public_keyset_context_encrypts(keys: CKKSKeySet):
    values = [complex(i % 7 - 3, i % 5 - 2) for i in range(keys.params.slot_count)]

    with keys.context(track_plaintext_values=True):
        pt = encode(values, msg_bound=4)
    with keys.public().context():
        ct = encrypt(pt)
        assert ct.values == pytest.approx(values)

    with keys.context():
        restored = decode(decrypt(ct), len(values))

    assert restored == pytest.approx(values, abs=1e-4)


def test_context_is_runtime_only():
    keys = keygen(small_params())
    with pytest.raises(TypeError, match="CKKSContext is runtime-only"):
        pickle.dumps(keys.context())


@pytest.mark.parametrize(
    "params",
    [small_params(), small_params(dtype=dtypes.uint64), CKKSParams.default(boot_params=None)],
    ids=["small-uint32", "small-uint64", "default"],
)
def test_public_evaluation_with_jit(params: CKKSParams):
    keys = keygen(params, rotations={1})
    public_keys = keys.public()
    captures = 0

    @jit
    def evaluate(x):
        nonlocal captures
        captures += 1
        return (x * x).rotate(1) + 0.25

    for values in ([0.125, -0.25, 0.5, -0.375], [-0.5, 0.25, -0.125, 0.375]):
        with keys.context():
            encrypted = encrypt(encode(values, msg_bound=1)).realize()
        with public_keys.context():
            result = evaluate(encrypted)
        with keys.context():
            decoded = decode(decrypt(result), length=3)
        assert decoded == pytest.approx([x * x + 0.25 for x in values[1:]], abs=1e-6)

    assert captures == 1


def test_jit_reuses_contexts_without_retaining_them(keys: CKKSKeySet):
    class WeakContext(CKKSContext):
        __slots__ = ("__weakref__",)

    captures = []

    @jit
    def step(x):
        captures.append(None)
        return x + context().default_level

    x = Poly([1] * keys.params.n, keys.params.ring_at(0)).realize()
    contexts = []
    for _ in range(3):
        ctx = WeakContext(keys, default_level=keys.params.default_level)
        contexts.append(weakref.ref(ctx))
        with ctx:
            assert step(x).tolist()[0] == 1 + keys.params.default_level
    ctx = None
    gc.collect()
    assert all(ref() is None for ref in contexts)
    assert captures == [None]


@pytest.mark.parametrize("option", ["default_level", "track_plaintext_values", "strict_noise_bounds"])
def test_jit_specializes_context_options(keys: CKKSKeySet, option: str):
    captures = []

    @jit
    def step(x):
        captures.append(None)
        return x + int(getattr(context(), option))

    x = Poly([1] * keys.params.n, keys.params.ring_at(0)).realize()
    for value in (0, 1, 0):
        ctx = keys.context(default_level=value) if option == "default_level" else keys.context(**{option: bool(value)})
        with ctx:
            assert step(x).tolist()[0] == 1 + value
    assert len(captures) == 2


@pytest.mark.parametrize("fail", [False, True], ids=["normal", "exception"])
def test_context_restores_outer_jit_scope(keys: CKKSKeySet, fail: bool):
    captures = []

    @jit
    def step(x):
        active = context(required=False)
        captures.append(active)
        return x + (active.default_level if active is not None else 0)

    x = Poly([1] * keys.params.n, keys.params.ring_at(0)).realize()
    with jit.scope("outer"):
        assert step(x).tolist()[0] == 1
        with pytest.raises(ValueError, match="body failed") if fail else nullcontext():
            with keys.context() as ctx:
                assert step(x).tolist()[0] == 1 + keys.params.default_level
                if fail:
                    raise ValueError("body failed")
        assert step(x).tolist()[0] == 1
        assert captures == [None, ctx]
        assert context(required=False) is None
    assert step(x).tolist()[0] == 1
    assert captures == [None, ctx, None]


def test_keygen_normalizes_existing_coefficient_secret_key(keys: CKKSKeySet):
    assert isinstance(keys.sk, Poly)
    coefficient_key = keys.sk.intt()
    derived = keygen(keys.params, secret=coefficient_key).realize()

    assert derived.sk is not None
    assert derived.sk.domain is Domain.NTT
    assert derived.sk.equals(keys.sk)
    with derived.context():
        assert decode(decrypt(encrypt(encode([0.125], msg_bound=1))), length=1) == pytest.approx([0.125], abs=1e-5)


@pytest.mark.parametrize("level", [0, 1])
def test_public_key_encrypt_and_decrypt_without_context(keys: CKKSKeySet, level: int):
    assert keys.pk is not None
    m = Poly.const(1234, keys.params.ring_at(level))
    noise = CKKSNoiseInfo(noise_bound=11, noise_heuristic_bound=7, msg_bound=1234)
    ct = keys.pk.encrypt(m, keys.params, noise_info=noise, rng=rng_from_int_seed(5))
    assert_type(ct, RLWECiphertext)
    assert type(ct) is RLWECiphertext
    assert ct.ring == m.ring
    assert ct.noise_info.msg_bound == noise.msg_bound
    assert ct.noise_info.noise_bound > noise.noise_bound
    assert ct.noise_info.noise_heuristic_bound > noise.noise_heuristic_bound
    assert isinstance(keys.sk, Poly)
    with Device.scope("cuda:999"):
        plaintext = ct._decrypt(keys.sk)
    assert_type(plaintext, Poly)
    assert plaintext.device == ct.device
    error = plaintext - m
    assert 0 < max(abs(c) for c in error.to_centered_list()) <= ct.noise_info.noise_bound


@pytest.mark.parametrize("tracking", [False, True])
def test_secret_key_decrypt_preserves_tracking(keys: CKKSKeySet, tracking: bool):
    assert isinstance(keys.sk, Poly)
    with keys.context(track_plaintext_values=tracking):
        ct = encrypt(encode([0.25, -0.5], msg_bound=1))
        contextual = decrypt(ct)
    direct = decrypt(ct, secret=keys.sk)
    for plaintext in (direct, contextual):
        assert plaintext.noise_info == ct.noise_info
        if tracking:
            assert plaintext.values.tolist() == ct.values.tolist()
        else:
            with pytest.raises(RuntimeError, match="tracking is not available"):
                _ = plaintext.values
    assert direct.m.equals(contextual.m)


def test_explicit_decryption_uses_supplied_secret(keys):
    assert isinstance(keys.sk, Poly)
    with keys.context():
        ct = encrypt(encode([0.25, -0.5], msg_bound=1))
    other = keys.replace(sk=generate_secret(keys.params))
    with other.context():
        plaintext = decrypt(ct, secret=keys.sk)
        assert_type(plaintext, Plaintext)
        assert decode(plaintext, 2) == pytest.approx([0.25, -0.5], abs=1e-6)


@pytest.mark.parametrize("level", [0, 1])
def test_public_key_encrypt_rejects_message_above_key_ring(keys: CKKSKeySet, level: int):
    assert keys.pk is not None
    key = keys.pk.map_polys(lambda poly: poly.to_quotient_ring(keys.params.ring_at(level)))
    m = Poly.zero(keys.params.highest_ring)
    rng = rng_from_int_seed(12)
    with pytest.raises(ParameterMismatchError, match="not a quotient"):
        key.encrypt(m, keys.params, noise_info=CKKSNoiseInfo.zero(), rng=rng)
    assert rng.reserve() == 0


@pytest.mark.parametrize(
    "kind,level,supplied_mask",
    [
        (RLWECiphertext, 0, False),
        (RLWECiphertext, 0, True),
        (RLWECiphertext, 1, False),
        (RLWECiphertext, 1, True),
        (Ciphertext, 1, True),
    ],
)
def test_rlwe_encrypt_with_explicit_noise_metadata(keys, kind, level, supplied_mask):
    assert isinstance(keys.sk, Poly)
    m = Poly.const(1234, keys.params.ring_at(level))
    noise = CKKSNoiseInfo.zero()
    mask = Poly.uniform(m.ring, rng=rng_from_int_seed(9)) if supplied_mask else None
    ciphertexts = [
        kind._encrypt(m, secret=keys.sk, mask=mask, error_dist=keys.params.noise_dist, output_noise_info=noise, rng=rng_from_int_seed(seed))
        for seed in (8, 8, 10)
    ]
    first, repeated, different = ciphertexts
    assert first.a.equals(repeated.a) and first.b.equals(repeated.b)
    assert not first.b.equals(different.b)
    for ct in ciphertexts:
        assert type(ct) is RLWECiphertext
        assert ct.ring == m.ring
        assert ct.noise_info == noise
        if mask is not None:
            assert ct.a.equals(mask)
        error = ct._decrypt(keys.sk) - m
        assert 0 < max(abs(c) for c in error.to_centered_list()) <= keys.params.noise_dist.coeff_inf_norm_bound()


@pytest.mark.parametrize("fault", ["ring", "device"])
def test_rlwe_encrypt_rejects_mismatched_mask(keys, fault):
    assert isinstance(keys.sk, Poly)
    m = Poly.const(1, keys.params.ring_at(0), device=keys.device)
    mask = Poly.zero(keys.params.ring_at(1)) if fault == "ring" else Poly.zero(m.ring, device="cuda:999")
    rng = rng_from_int_seed(12)
    with pytest.raises(ParameterMismatchError, match="mask"):
        RLWECiphertext._encrypt(m, secret=keys.sk, mask=mask, error_dist=keys.params.noise_dist, output_noise_info=CKKSNoiseInfo.zero(), rng=rng)
    assert rng.reserve() == 0


@pytest.mark.parametrize("fault", ["device", "degree", "dtype"])
def test_rlwe_encrypt_rejects_incompatible_secret(keys, fault):
    ring = keys.params.ring_at(0)
    m = Poly.const(1, ring, device=keys.device)
    if fault == "device":
        secret = Poly.const(1, ring, device="cuda:999")
    else:
        secret_ring = replace(ring, n=ring.n // 2) if fault == "degree" else replace(ring, dtype=dtypes.uint64)
        secret = Poly.const(1, secret_ring)
    rng = rng_from_int_seed(12)
    with pytest.raises(ValueError, match=f"{fault} mismatch"):
        RLWECiphertext._encrypt(m, secret=secret, error_dist=keys.params.noise_dist, output_noise_info=CKKSNoiseInfo.zero(), rng=rng)


def test_public_key_encrypt_rejects_incompatible_message_device(keys):
    assert keys.pk is not None
    m = Poly.const(1, keys.params.ring_at(0), device="cuda:999")
    rng = rng_from_int_seed(12)
    with pytest.raises(ValueError, match="device mismatch"):
        keys.pk.encrypt(m, keys.params, noise_info=CKKSNoiseInfo.zero(), rng=rng)


def test_public_key_encrypt_uses_its_own_device(keys):
    key = keys.pk
    assert isinstance(key, PublicKey)
    m = Poly.const(1, keys.params.ring_at(0), device=keys.device)
    with Device.scope("cuda:999"):
        ciphertext = key.encrypt(m, keys.params, noise_info=CKKSNoiseInfo.zero())
    assert ciphertext.device == keys.device
    ciphertext.realize()


def test_rlwe_encrypt_defaults(keys):
    assert isinstance(keys.sk, Poly)
    m = Poly.const(1, keys.params.ring_at(0), device=keys.device)
    with Device.scope("cuda:999"):
        ciphertext = RLWECiphertext._encrypt(m, secret=keys.sk, error_dist=keys.params.noise_dist)
    assert ciphertext.device == keys.device
    assert ciphertext.noise_info.msg_bound == 0
    assert 0 < ciphertext.noise_info.noise_heuristic_bound <= ciphertext.noise_info.noise_bound < float("inf")
    ciphertext.realize()


@pytest.mark.parametrize("operation", ["public", "secret", "rlwe"])
def test_key_operations_reject_wrong_rng_device_before_sampling(keys, operation):
    assert isinstance(keys.sk, Poly)
    assert keys.pk is not None
    secret = keys.sk.to("cuda:999")
    public = keys.pk.to("cuda:999")
    pt = Plaintext(m=Poly.zero(keys.params.highest_ring, device="cuda:999"), noise_info=CKKSNoiseInfo.zero())
    rng = rng_from_int_seed(12, device="python")
    with pytest.raises(ParameterMismatchError, match="RNG on python"):
        if operation == "public":
            public.encrypt(pt.m, keys.params, noise_info=pt.noise_info, rng=rng)
        elif operation == "secret":
            with keys.to("cuda:999").context():
                encrypt(pt, rng=rng)
        else:
            RLWECiphertext._encrypt(pt.m, secret=secret, error_dist=keys.params.noise_dist, output_noise_info=pt.noise_info, rng=rng)
    assert rng.reserve() == 0
