import hashlib
import pickle
from dataclasses import replace
from fractions import Fraction
from math import ceil, sqrt
from typing import Iterable, assert_type

import pytest

from ren import jit, tree
from ren.device import Device
from ren.random import ChaChaRng, rng_from_int_seed
from ren.dtype import dtypes
from ren.poly import Poly
from ren.polydist import DiscreteGaussian, Ternary
from ren.ring import Domain
from ren.ckks import (
    BootstrappingParams,
    Ciphertext,
    CKKSKeySet,
    CKKSLevel,
    Plaintext,
    PublicKey,
    RLWECiphertext,
    bootstrap,
    decode,
    decrypt,
    encode,
    encrypt,
    multiparty,
    partial_decrypt,
)
from ren.ckks.keys import generate_secret
from ren.ckks.metadata import Bound, CKKSNoiseInfo
from ren.ckks.multiparty import Dkg, DkgRound1Share, DkgRound1State, DkgRound2Input, DkgRound2Share, MultipartyParams, PartialDecryption
from ren.ckks.params import CKKSParams
from ren.ckks.utils import select_moduli, steps_to_galois_elt
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError
from tests.utils import mixed_prime_ring, random_ciphertext, random_values_with_norm
from tests.ckks.params import small_params


SEED = 0xDEADBEEF
STATISTICAL_SECURITY = 8
NUM_DECRYPTIONS = 10


@pytest.fixture(scope="module")
def keys() -> CKKSKeySet:
    return CKKSKeySet.generate(small_params()).realize()


@pytest.fixture(scope="module")
def public_keys(keys) -> CKKSKeySet:
    return keys.public().replace(multiparty=MultipartyParams(num_parties=2))


def test_share_keyset_encrypts_under_collective_public_key(keys, public_keys):
    participant_keys = public_keys.replace(sk=generate_secret(keys.params))
    with participant_keys.context():
        ciphertext = encrypt(encode([0.25, -0.5], msg_bound=1))
    with keys.context():
        assert decode(decrypt(ciphertext), 2) == pytest.approx([0.25, -0.5], abs=1e-6)
    with participant_keys.replace(pk=None).context(), pytest.raises(RuntimeError, match="public key or a single-party secret key"):
        encrypt(encode([0.25, -0.5], msg_bound=1))


@pytest.mark.parametrize("threshold", [None, 2, 3])
def test_share_keyset_roundtrips_and_removes_private_material(keys, threshold):
    multiparty = MultipartyParams(num_parties=3, threshold=threshold)
    share = generate_secret(keys.params)
    if threshold is not None:
        share = share.to_quotient_ring(keys.params.ring_at(0))
    participant_keys = keys.public().replace(sk=share, multiparty=multiparty)
    assert participant_keys.public().multiparty == multiparty
    assert participant_keys.to("python").multiparty == multiparty
    restored = pickle.loads(pickle.dumps(participant_keys))
    assert isinstance(restored.sk, Poly)
    assert restored.multiparty == multiparty
    assert restored.sk.equals(share)
    assert restored.public().sk is None
    assert restored.public().multiparty == multiparty


def test_keyset_params_define_secret_role(keys):
    assert isinstance(keys.sk, Poly)
    with keys.context():
        ct = encrypt(encode([0.25], msg_bound=1)).to_level(0).rerandomize()
        assert isinstance(decrypt(ct), Plaintext)
    collective = keys.replace(multiparty=MultipartyParams(num_parties=1))
    assert collective.sk is not None and collective.sk.equals(keys.sk)
    with collective.context():
        partial = partial_decrypt(ct, flooding_std_dev=64)
        assert isinstance(partial, PartialDecryption)
        assert decode(decrypt(ct, partials=[partial]), 1) == pytest.approx([0.25], abs=1e-6)


@pytest.fixture
def prepared(public_keys: CKKSKeySet) -> Ciphertext:
    with public_keys.context():
        ciphertext = encrypt(encode([0.25, -0.5], msg_bound=1))
        return ciphertext.to_level(0).rerandomize()


def test_additive_decryption_after_ciphertext_and_share_pickle(keys, public_keys, prepared):
    assert isinstance(keys.sk, Poly)
    first = generate_secret(keys.params)
    second = keys.sk - first
    partials = []
    for i, share in enumerate((first, second)):
        participant_keys = pickle.loads(pickle.dumps(public_keys.replace(sk=share)))
        received = pickle.loads(pickle.dumps(prepared))
        with participant_keys.context():
            partial = partial_decrypt(received, flooding_std_dev=64, rng=rng_from_int_seed(100 + i))
        assert isinstance(partial, PartialDecryption)
        partials.append(pickle.loads(pickle.dumps(partial)))
    plaintext = decrypt(prepared, partials=list(reversed(partials)))
    with public_keys.context():
        assert decode(plaintext, 2) == pytest.approx([0.25, -0.5], abs=1e-6)


def test_partial_decryption_uses_explicit_rng(public_keys: CKKSKeySet, prepared: Ciphertext):
    share = generate_secret(public_keys.params)
    with public_keys.replace(sk=share).context():
        first = partial_decrypt(prepared, flooding_std_dev=64, rng=rng_from_int_seed(10))
        second = partial_decrypt(prepared, flooding_std_dev=64, rng=rng_from_int_seed(10))
        different = partial_decrypt(prepared, flooding_std_dev=64, rng=rng_from_int_seed(11))
    assert_type(first, PartialDecryption)
    assert first.d.equals(second.d)
    assert not first.d.equals(different.d)


def test_decrypt_rejects_wrong_secret_kind(keys, public_keys, prepared):
    with keys.context(), pytest.raises(RuntimeError, match="requires a multiparty keyset"):
        partial_decrypt(prepared, flooding_std_dev=64)
    with public_keys.context(), pytest.raises(RuntimeError, match="requires a secret key share"):
        partial_decrypt(prepared, flooding_std_dev=64)
    share = generate_secret(keys.params)
    with public_keys.replace(sk=share).context(), pytest.raises(RuntimeError, match="use partial_decrypt"):
        decrypt(prepared)


@pytest.mark.parametrize(
    "fault,error,match",
    [
        ("empty", ParameterMismatchError, "at least one"),
        ("secret", InvalidParameterError, "mutually exclusive"),
        ("other-ring", ValueError, "ring mismatch"),
        ("other-device", ValueError, "device mismatch"),
    ],
)
def test_decrypt_rejects_invalid_partials(public_keys, prepared, fault, error, match):
    share = generate_secret(public_keys.params)
    with public_keys.replace(sk=share).context():
        partials = [partial_decrypt(prepared, flooding_std_dev=64)]
    if fault == "empty":
        partials.clear()
    elif fault == "other-ring":
        partials[0] = replace(partials[0], d=Poly.zero(public_keys.params.ring_at(1)))
    elif fault == "other-device":
        partials[0] = replace(partials[0], d=partials[0].d.to("cuda:3"))
    with pytest.raises(error, match=match):
        if fault == "secret":
            decrypt(prepared, partials=partials, secret=share)  # ty: ignore[no-matching-overload]
        else:
            decrypt(prepared, partials=partials)


def test_keyset_preserves_share_ring_and_device_invariants(keys, public_keys):
    share = generate_secret(keys.params)
    with pytest.raises(ParameterMismatchError, match="secret key ring"):
        public_keys.replace(sk=share.to_quotient_ring(keys.params.ring_at(0)))
    with pytest.raises(ParameterMismatchError, match="secret key ring"):
        public_keys.replace(sk=share, multiparty=MultipartyParams(num_parties=2, threshold=2))
    with pytest.raises(ParameterMismatchError, match="device mismatch"):
        public_keys.replace(sk=share.to("cuda:3"))


def test_explicit_decryption_with_multiparty_context(keys):
    assert isinstance(keys.sk, Poly)
    with keys.context():
        ct = encrypt(encode([0.25, -0.5], msg_bound=1))
    other = keys.replace(sk=generate_secret(keys.params), multiparty=MultipartyParams(num_parties=2))
    with other.context():
        plaintext = decrypt(ct, secret=keys.sk)
        assert_type(plaintext, Plaintext)
        assert decode(plaintext, 2) == pytest.approx([0.25, -0.5], abs=1e-6)


def test_share_decrypt_uses_ciphertext_device(public_keys, prepared):
    share = generate_secret(public_keys.params)
    with public_keys.replace(sk=share).context(), Device.scope("cuda:999"):
        partial = partial_decrypt(prepared, flooding_std_dev=64)
    assert partial.d.device == prepared.device
    partial.realize()


def test_share_decrypt_rejects_wrong_rng_device_before_sampling(keys, prepared):
    participant_keys = keys.replace(multiparty=MultipartyParams(num_parties=2)).to("cuda:999")
    ciphertext = prepared.to("cuda:999")
    rng = rng_from_int_seed(12, device="python")
    with participant_keys.context(), pytest.raises(ParameterMismatchError, match="RNG on python"):
        partial_decrypt(ciphertext, flooding_std_dev=64, rng=rng)
    assert rng.reserve() == 0


@pytest.mark.parametrize("threshold", [None, 2], ids=["additive", "threshold"])
@pytest.mark.parametrize("sigma", [3.25, 128.25])
def test_partial_decryption_uses_supplied_flooding(public_keys, prepared, threshold, sigma):
    share = generate_secret(public_keys.params)
    if threshold is not None:
        share = share.to_quotient_ring(prepared.ring)
    sharing = MultipartyParams(num_parties=2, threshold=threshold)
    quorum = (1, 2) if threshold is not None else None
    party = 1 if threshold is not None else None
    ct = prepared.replace(noise_info=CKKSNoiseInfo.unbounded())
    with public_keys.replace(sk=share, multiparty=sharing).context():
        partial = partial_decrypt(ct, flooding_std_dev=sigma, quorum=quorum, party=party, rng=rng_from_int_seed(7))
    assert_type(partial, PartialDecryption)
    coefficient = 1 if threshold is None else 2  # Lagrange weight at zero for party 1 in {1, 2}.
    error = partial.d - ct.a * share.to_quotient_ring(ct.ring) * coefficient
    noise_dist = DiscreteGaussian(ct.ring.n, Fraction(ceil(sigma)))
    assert error.equals(Poly.rand(noise_dist, ct.ring, rng=rng_from_int_seed(7)))
    assert 0 < max(abs(c) for c in error.to_centered_list()) <= noise_dist.coeff_inf_norm_bound()
    with public_keys.context():
        slot_error = max(abs(value) for value in decode(Plaintext(m=error, noise_info=partial.noise_info)))
        assert slot_error <= float(partial.noise_info.noise_bound / ct.scale)
        assert slot_error <= float(partial.noise_info.noise_heuristic_bound / ct.scale)
    assert partial.noise_info.msg_bound == 0


@pytest.mark.parametrize("sigma", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_partial_decryption_rejects_invalid_flooding_before_sampling(public_keys, prepared, sigma):
    share = generate_secret(public_keys.params)
    rng = rng_from_int_seed(7)
    with public_keys.replace(sk=share).context(), pytest.raises(ValueError, match="finite and positive"):
        partial_decrypt(prepared, flooding_std_dev=sigma, rng=rng)
    assert rng.reserve() == 0


def test_decrypt_uses_each_partials_noise_metadata(keys):
    ring = keys.params.ring_at(0)
    ct = Ciphertext(a=Poly.zero(ring), b=Poly.const(10, ring), noise_info=CKKSNoiseInfo(msg_bound=7, noise_bound=3, noise_heuristic_bound=4))
    partials = [
        PartialDecryption(Poly.const(2, ring), noise_info=CKKSNoiseInfo(msg_bound=0, noise_bound=7, noise_heuristic_bound=8)),
        PartialDecryption(Poly.const(3, ring), noise_info=CKKSNoiseInfo(msg_bound=0, noise_bound=13, noise_heuristic_bound=16)),
    ]
    plaintext = decrypt(ct, partials=partials)
    assert plaintext.m.equals(Poly.const(15, ring))
    assert plaintext.noise_info.msg_bound == 7
    assert plaintext.noise_info.noise_bound == 23
    assert float(plaintext.noise_info.noise_heuristic_bound) == pytest.approx((4**2 + 8**2 + 16**2) ** 0.5)


@pytest.mark.parametrize("threshold", [None, 2], ids=["additive", "threshold"])
def test_dkg_pickle_resumes_rounds(threshold: int | None):
    params = small_params()
    multiparty_params = MultipartyParams(num_parties=3, threshold=threshold)
    dkg = Dkg(params, common_random_seed=SEED, multiparty=multiparty_params, rotations={1})
    dkg = pickle.loads(pickle.dumps(dkg))
    states, shares = zip(*(dkg.round1(generate_secret(params)) for _ in range(3)), strict=True)
    original_states = states
    keys, round2_input = dkg.aggregate_round1(shares)
    # The caller can persist private state separately from public round data.
    dkg, states, keys, round2_input = pickle.loads(pickle.dumps((dkg, states, keys, round2_input)))
    round2 = [dkg.round2(state, round2_input) for state in states]
    dkg, states, round2 = pickle.loads(pickle.dumps((dkg, states, round2)))
    rlk, encrypted_shares = dkg.aggregate_round2(round2_input, round2)
    keys = pickle.loads(pickle.dumps(keys.replace(rlk=rlk)))
    assert keys.sk is None
    assert keys.multiparty == multiparty_params
    assert keys.secret_l1_norm_bound == 3 * params.secret_key_dist.coeff_1_norm_bound()
    assert keys.secret_expected_squared_l2_norm == 3 * params.secret_key_dist.expected_squared_l2_norm()

    with keys.context(track_plaintext_values=True):
        values = [0.25, -0.5, 0.75]
        ct = encrypt(encode(values, msg_bound=1))
        ct = (ct * ct).rotate(1).to_level(0)
    with keys.context():
        prepared = ct.rerandomize()
        sigma = multiparty.flooding_std_dev(
            prepared.noise_info.noise_heuristic_bound, params, statistical_security=STATISTICAL_SECURITY, num_decryptions=NUM_DECRYPTIONS
        )
    parties = (1, 3) if threshold is not None else (1, 2, 3)
    partials = []
    for i in parties:
        encrypted = encrypted_shares[i - 1] if encrypted_shares else None
        participant_keys = dkg.finish(states[i - 1], encrypted)
        original = dkg.finish(original_states[i - 1], encrypted)
        assert participant_keys.sk is not None and original.sk is not None
        assert participant_keys.sk.equals(original.sk)
        assert participant_keys.params == params
        assert participant_keys.multiparty == dkg.multiparty
        assert participant_keys.pk is None and participant_keys.rlk is None
        assert not participant_keys.galois_keys
        with participant_keys.context():
            partials.append(
                partial_decrypt(
                    prepared,
                    flooding_std_dev=sigma,
                    quorum=parties if threshold is not None else None,
                    party=i if threshold is not None else None,
                )
            )
    with keys.context():
        result = decrypt(prepared, partials=partials)
        assert result.values[: len(values)] == pytest.approx([0.25, 0.5625, 0])
        assert decode(result, len(values)) == pytest.approx([0.25, 0.5625, 0], abs=1e-4)


def test_jit_replays_changed_partial_decryption():
    ring = mixed_prime_ring(4)
    captures = []

    @jit
    def add_record(poly, record):
        captures.append(None)
        return poly + record.d

    poly = Poly([1, 2, 3, 4], ring)
    outputs = [add_record(poly, PartialDecryption(Poly([value] * 4, ring), CKKSNoiseInfo.zero())) for value in (10, 20)]
    assert [output.tolist() for output in outputs] == [[11, 12, 13, 14], [21, 22, 23, 24]]
    assert len(captures) == 1


@pytest.mark.parametrize(
    "threshold,quorum,party,error",
    [
        (None, (1, 2), 1, "additive shares do not accept"),
        (2, None, None, "quorum and party must be provided"),
        (2, (1,), 1, "at least 2 parties"),
        (2, (1, 1), 1, "duplicate party"),
        (2, (1, 2), 3, "party must be in"),
        (2, (0, 1), 1, "from 1 to num_parties"),
        (2, (1, 4), 1, "from 1 to num_parties"),
    ],
)
def test_partial_decrypt_rejects_invalid_share_protocol(public_keys, prepared, threshold, quorum, party, error):
    share = generate_secret(public_keys.params)
    if threshold is not None:
        share = share.to_quotient_ring(prepared.ring)
    with (
        public_keys.replace(sk=share, multiparty=MultipartyParams(num_parties=3, threshold=threshold)).context(),
        pytest.raises(ParameterMismatchError, match=error),
    ):
        partial_decrypt(prepared, quorum=quorum, party=party, flooding_std_dev=64)


def test_threshold_context_rejects_ciphertext_above_share_ring(public_keys):
    # Shamir shares live on the level-0 ring and cannot decrypt higher levels.
    share = Poly.const(1, public_keys.params.ring_at(0))
    keys = public_keys.replace(sk=share, multiparty=MultipartyParams(num_parties=2, threshold=2))
    with keys.context():
        ct = Ciphertext.zero(level=1)
        with pytest.raises(ParameterMismatchError, match="not a quotient of secret share ring"):
            partial_decrypt(ct, flooding_std_dev=64, quorum=(1, 2), party=1)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_partial_decrypt_accepts_ciphertext_ring_below_share_ring(public_keys, level):
    # Additive shares live on the highest ring, so every level is a valid quotient.
    params = public_keys.params
    share = Poly.const(1, params.highest_ring)
    ring = params.ring_at(level)
    ct = Ciphertext(a=Poly.const(1, ring), b=Poly.zero(ring), noise_info=CKKSNoiseInfo.zero())
    with public_keys.replace(sk=share, multiparty=MultipartyParams(num_parties=1)).context():
        pd = partial_decrypt(ct, flooding_std_dev=64)
    assert pd.d.ring == ring


def test_dkg_rejects_sparse_encapsulation():
    with pytest.raises(InvalidParameterError, match="encapsulation"):
        Dkg(CKKSParams.default(), SEED, multiparty=MultipartyParams(num_parties=2))


def test_threshold_dkg_requires_transport_level():
    base = small_params()
    params = replace(base, levels=base.levels[-1:])
    with pytest.raises(InvalidParameterError, match="threshold DKG requires a transport level"):
        Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2, threshold=2))


def test_threshold_dkg_requires_share_ring_to_divide_transport_ring():
    base = small_params()
    moduli = base.highest_ring.moduli[:4]
    params = replace(
        base,
        levels=tuple(CKKSLevel(moduli=basis, scale=1) for basis in (moduli[:2], moduli[1:3], moduli)),
        log_range=0,
        ks_digit_sizes=(2, 2),
    )
    with pytest.raises(InvalidParameterError, match="level-0 ring must be a quotient of level 1"):
        Dkg(params, common_random_seed=SEED, multiparty=MultipartyParams(num_parties=3, threshold=2))


def test_dkg_pickle_preserves_normalized_galois_operations():
    params = CKKSParams.generate_prefix(
        log_n=10,
        depth=2,
        log_precision=50,
        log_range=7,
        dtype=dtypes.uint32,
        boot_params=BootstrappingParams(
            cts_depth=2,
            stc_depth=2,
            exp_degree=16,
            squaring_count=0,
            failure_probability=2**-16,
        ),
        bootstrap_level_bit_length=60,
        p_prime_count=4,
        security=None,
    )
    dkg = Dkg(params, common_random_seed=SEED, multiparty=MultipartyParams(num_parties=2), rotations={params.highest_level: {1}}, conjugation=True)

    restored = pickle.loads(pickle.dumps(dkg))
    # The ceremony stores the normalized request plus bootstrapping's own keys.
    assert restored.galois_elts_by_level == dkg.galois_elts_by_level
    assert 2 * params.n - 1 in restored.galois_elts_by_level[params.default_level]
    assert steps_to_galois_elt(1, params.n) in restored.galois_elts_by_level[params.highest_level]


def test_dkg_pickle_uses_active_device():
    params = small_params()
    payload = pickle.dumps(Dkg(params, common_random_seed=SEED, multiparty=MultipartyParams(num_parties=1)))
    with Device.scope("python"):
        dkg = pickle.loads(payload)
        state, share = dkg.round1(generate_secret(params, device="python"))
        keys, round2_input = dkg.aggregate_round1([share])
        rlk, encrypted = dkg.aggregate_round2(round2_input, [dkg.round2(state, round2_input)])
        keys = keys.replace(rlk=rlk)
        participant_keys = dkg.finish(state)
    assert encrypted == ()
    assert participant_keys.device == "python"
    assert all(poly.device == "python" for poly in keys.polys())


@pytest.mark.parametrize(("num_parties", "threshold"), ((0, None), (2, 0), (2, 3)))
def test_dkg_rejects_invalid_topology(num_parties, threshold):
    with pytest.raises(ValueError):
        MultipartyParams(num_parties=num_parties, threshold=threshold)


@pytest.mark.parametrize(
    ("statistical_security", "num_decryptions"),
    ((-1, 1), (1, 0)),
)
def test_flooding_calibration_rejects_invalid_policy(statistical_security, num_decryptions):
    params = small_params()
    with pytest.raises(ValueError):
        multiparty.flooding_std_dev(Bound(0), params, statistical_security=statistical_security, num_decryptions=num_decryptions)


@pytest.mark.parametrize("ct_noise,security,expected", [(100, 2, 1200), (0, 2, None), (100, None, None)])
def test_flooding_calibration_preserves_policy(ct_noise, security, expected):
    params = small_params()
    sigma = multiparty.flooding_std_dev(Bound(ct_noise), params, statistical_security=security, num_decryptions=3)
    assert sigma == (params.noise_dist.coeff_inf_norm_bound() if expected is None else expected)


def test_flooding_calibration_rejects_excessive_noise():
    with pytest.raises(ValueError, match="Noise bound is too large"):
        multiparty.flooding_std_dev(Bound(1 << 501), small_params(), statistical_security=8, num_decryptions=1)


def _dkg_randomness_test_params() -> CKKSParams:
    ring = mixed_prime_ring(64, dtype=dtypes.uint32)
    highest_moduli = ring.moduli[:4]
    ks_primes = select_moduli(n=ring.n, num_primes=2, bit_length=29, exclude=set(highest_moduli))
    return CKKSParams.from_levels(
        n=ring.n,
        dtype=dtypes.uint32,
        levels=(CKKSLevel(moduli=highest_moduli, scale=1 << 20),),
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2),
        log_range=0,
        security=None,
    )


def test_common_dkg_randomness_known_answer():
    common = Dkg(_dkg_randomness_test_params(), SEED, multiparty=MultipartyParams(num_parties=1), rotations={1})
    polys = [
        common.pk_mask,
        *common.rlk_masks,
        *(poly for galois_elt in sorted(common.galois_key_material) for poly in common.galois_key_material[galois_elt].masks),
    ]
    digest = hashlib.sha256()
    for poly in polys:
        width = (poly.ring.q.bit_length() + 7) // 8
        for value in poly:
            digest.update(value.to_bytes(width, "little"))

    assert digest.hexdigest() == "71587151659500c5bb35fea61ce2879462651ff34440153bc55c9c4695649768"


def test_common_dkg_randomness_is_stable_when_adding_an_artifact():
    params = _dkg_randomness_test_params()
    base = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=1), rotations={1})
    extended = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=1), rotations={1, 2})

    assert base.pk_mask.equals(extended.pk_mask)
    assert all(left.equals(right) for left, right in zip(base.rlk_masks, extended.rlk_masks, strict=True))
    for galois_elt, material in base.galois_key_material.items():
        extended_masks = extended.galois_key_material[galois_elt].masks
        assert all(left.equals(right) for left, right in zip(material.masks, extended_masks, strict=True))


@pytest.mark.requires_cuda
def test_common_dkg_randomness_uses_requested_device():
    common = Dkg(_dkg_randomness_test_params(), SEED, multiparty=MultipartyParams(num_parties=1), rotations={1}, device="python")

    assert common.pk_mask.device == "python"
    assert all(mask.device == "python" for mask in common.rlk_masks)
    assert all(mask.device == "python" for material in common.galois_key_material.values() for mask in material.masks)


def test_multiparty_rotation_key_covers_incomparable_requested_levels():
    ring = mixed_prime_ring(1024, dtype=dtypes.uint32)
    highest_moduli = ring.moduli[:6]
    levels = (
        CKKSLevel(moduli=highest_moduli[:2], scale=1 << 21),
        CKKSLevel(moduli=highest_moduli[3:5], scale=1 << 21),
        CKKSLevel(moduli=highest_moduli, scale=1 << 21),
    )
    ks_primes = select_moduli(n=ring.n, num_primes=3, bit_length=29, exclude=set(highest_moduli))
    params = CKKSParams.from_levels(
        n=ring.n,
        dtype=dtypes.uint32,
        levels=levels,
        ks_primes=ks_primes,
        ks_digit_sizes=(2, 2, 2),
        log_range=0,
        security=None,
    )
    rotations = {0: {1}, 1: {1}}
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2), rotations=rotations)
    secrets = [generate_secret(params) for _ in range(2)]
    round_1_messages = [dkg.round1(secret)[1] for secret in secrets]
    keys, _ = dkg.aggregate_round1(round_1_messages)

    group_sk = sum(secrets, start=Poly.zero(params.highest_ring))
    values = [complex((index % 17) / 17, (index % 13) / 13) for index in range(params.slot_count)]

    with keys.context():
        for level in (0, 1):
            rotated = encrypt(encode(values, msg_bound=2, level=level)).rotate(1)
            assert decode(decrypt(rotated, secret=group_sk)) == pytest.approx(values[1:] + values[:1], abs=2e-2)


def test_dkg_rejects_mismatched_galois_key_requests():
    params = small_params()
    definitions = [Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2), rotations=rotations) for rotations in ({1, 2}, {2})]
    shares = [dkg.round1(generate_secret(params))[1] for dkg in definitions]
    with pytest.raises(ParameterMismatchError, match="Galois key shares do not match the DKG key layout"):
        definitions[1].aggregate_round1(shares)


def test_dkg_rejects_incomplete_rounds():
    params = small_params()
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2))
    states, shares = zip(*(dkg.round1(generate_secret(params)) for _ in range(2)))
    with pytest.raises(ParameterMismatchError, match="Expected 2 round 1 shares"):
        dkg.aggregate_round1(shares[:-1])
    keys, round2_input = dkg.aggregate_round1(shares)
    shares2 = [dkg.round2(state, round2_input) for state in states]
    with pytest.raises(ParameterMismatchError, match="Expected 2 round 2 shares"):
        dkg.aggregate_round2(round2_input, shares2[:-1])


@pytest.fixture
def threshold_dkg_rounds():
    params = small_params()
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2, threshold=2), rotations={1})
    states, shares1 = zip(*(dkg.round1(generate_secret(params)) for _ in range(2)))
    keys, round2_input = dkg.aggregate_round1(shares1)
    shares2 = [dkg.round2(state, round2_input) for state in states]
    return dkg, states, shares1, keys, round2_input, shares2


def test_round2_input_requires_consistent_digits():
    p = Poly.zero(mixed_prime_ring(4))
    other = Poly.zero(replace(p.ring, moduli=p.ring.moduli[:1]))
    for h0, h1 in (((), ()), ((p,), (p, p)), ((p, other), (p, p)), ((p,), (other,))):
        with pytest.raises(ParameterMismatchError, match="digit|ring"):
            DkgRound2Input(h0, h1, ())


def test_round1_share_requires_consistent_galois_digits():
    p = Poly.zero(mixed_prime_ring(4))
    other = Poly.zero(replace(p.ring, moduli=p.ring.moduli[:1]))
    inputs = DkgRound2Input((p,), (p,), ())
    for digits in ((), (p, other)):
        with pytest.raises(ParameterMismatchError, match="digit"):
            DkgRound1Share(p, {3: digits}, inputs)


def test_round2_share_requires_consistent_digits():
    p = Poly.zero(mixed_prime_ring(4))
    other = Poly.zero(replace(p.ring, moduli=p.ring.moduli[:1]))
    for digits in ((), (p, other)):
        with pytest.raises(ParameterMismatchError, match="digit"):
            DkgRound2Share(digits, ())


def test_round1_share_owns_galois_mapping_and_pickles():
    body, h0, h1, galois = (Poly.const(value, mixed_prime_ring(4)) for value in range(1, 5))
    source = DkgRound1Share(body, {3: (galois,)}, DkgRound2Input((h0,), (h1,), ()))
    mapping = dict(source.galois_keys)
    message = replace(source, galois_keys=mapping)
    mapping.clear()
    for value in (message, pickle.loads(pickle.dumps(message))):
        assert value.galois_keys.keys() == source.galois_keys.keys()
        assert all(a.equals(b) for a, b in zip(value.polys(), source.polys(), strict=True))
        elt = next(iter(value.galois_keys))
        with pytest.raises(TypeError):
            value.galois_keys[elt] = value.galois_keys[elt]  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("stage", ["round1", "round1_generate", "round2"])
def test_dkg_rejects_wrong_rng_device_before_sampling(threshold_dkg_rounds, stage):
    dkg, states, _, _, round2_input, _ = threshold_dkg_rounds
    dkg = tree.replace(dkg, lambda poly: poly.to("cuda:999"), leaf=Poly)
    state = states[0].to("cuda:999")
    rng = ChaChaRng("python", bytes(32))
    with pytest.raises(ParameterMismatchError, match="RNG on python"):
        if stage == "round1":
            dkg.round1(state.secret, rng=rng)
        elif stage == "round1_generate":
            dkg.round1(rng=rng)
        else:
            dkg.round2(state, round2_input.to("cuda:999"), rng=rng)
    assert rng.reserve() == 0


def test_additive_dkg_without_transport_level():
    params = CKKSParams.generate_prefix(log_n=10, depth=0, log_precision=40, log_range=0, dtype=dtypes.uint32, security=None)
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2))
    states, shares1 = zip(*(dkg.round1(generate_secret(params)) for _ in range(2)))
    keys, round2_input = dkg.aggregate_round1(shares1)
    shares2 = [dkg.round2(state, round2_input) for state in states]
    rlk, encrypted = dkg.aggregate_round2(round2_input, shares2)
    keys = keys.replace(rlk=rlk)
    assert encrypted == ()
    assert keys.multiparty == dkg.multiparty
    for state in states:
        participant_keys = dkg.finish(state)
        assert participant_keys.sk is not None and participant_keys.sk.equals(state.secret)


@pytest.mark.parametrize("num_parties", [1, 3])
def test_threshold_one_aggregates_transport_shares(num_parties):
    params = CKKSParams.generate_prefix(log_n=10, depth=1, log_precision=40, log_range=0, dtype=dtypes.uint32, security=None)
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=num_parties, threshold=1))
    states, shares1 = zip(*(dkg.round1(generate_secret(params)) for _ in range(num_parties)))
    keys, round2_input = dkg.aggregate_round1(shares1)
    shares2 = [dkg.round2(state, round2_input) for state in states]
    _, encrypted_shares = dkg.aggregate_round2(round2_input, shares2)

    assert isinstance(encrypted_shares, tuple)
    assert len(encrypted_shares) == num_parties
    ring = params.ring_at(0)
    secret = sum((state.secret.to_quotient_ring(ring) for state in states), start=Poly.zero(ring))
    for i, (state, encrypted) in enumerate(zip(states, encrypted_shares, strict=True)):
        contributions = [share.encrypted_secret_shares[i] for share in shares2]
        zero = Poly.zero(params.ring_at(1))
        assert encrypted.a.equals(sum((ct.a for ct in contributions), start=zero))
        assert encrypted.b.equals(sum((ct.b for ct in contributions), start=zero))
        assert encrypted.noise_info == CKKSNoiseInfo.zero()
        assert all(ct.noise_info.noise_bound > 0 for ct in contributions)
        # A constant Shamir polynomial gives every recipient the collective secret.
        participant_keys = dkg.finish(state, encrypted)
        assert participant_keys.sk is not None and participant_keys.sk.equals(secret)


def test_round1_state_requires_compatible_rings():
    p = Poly.zero(mixed_prime_ring(4))
    other = Poly.zero(replace(p.ring, moduli=p.ring.moduli[:1]))
    for secret, ephemeral in ((p, other), (other, p)):
        with pytest.raises(ParameterMismatchError, match="secret rings must match"):
            DkgRound1State(secret, ephemeral, None)


@pytest.mark.parametrize("level", [0, None])
def test_dkg_rejects_wrong_transport_secret_ring(threshold_dkg_rounds, level):
    dkg, states, _, _, inputs, shares2 = threshold_dkg_rounds
    state = replace(states[0], transport_secret=generate_secret(dkg.params, level=level))
    with pytest.raises(ParameterMismatchError, match="transport secret does not match"):
        dkg.round2(state, inputs)
    with pytest.raises(ParameterMismatchError, match="transport secret does not match"):
        dkg.finish(state, shares2[0].encrypted_secret_shares[0])


def test_dkg_rejects_state_from_other_ring(threshold_dkg_rounds):
    dkg, states, _, _, round2_input, shares2 = threshold_dkg_rounds
    state = replace(states[0].map_polys(lambda poly: poly.to_quotient_ring(dkg.params.ring_at(0))))
    with pytest.raises(ParameterMismatchError, match="DKG ring"):
        dkg.round2(state, round2_input)
    with pytest.raises(ParameterMismatchError, match="DKG ring"):
        dkg.finish(state, shares2[0].encrypted_secret_shares[0])


@pytest.mark.parametrize("field", ["secret", "relinearization_secret"])
def test_dkg_only_requires_matching_devices_for_consumed_state(threshold_dkg_rounds, field):
    dkg, states, _, _, round2_input, shares2 = threshold_dkg_rounds
    state = states[0]
    bad = getattr(state, field).to("cuda:999")
    state = replace(state, **{field: bad})
    with pytest.raises(ValueError, match="device mismatch"):
        dkg.round2(state, round2_input)
    ct = shares2[0].encrypted_secret_shares[0]
    actual, expected = dkg.finish(state, ct).sk, dkg.finish(states[0], ct).sk
    assert actual is not None and expected is not None
    assert actual.equals(expected)


def test_dkg_round2_does_not_use_transport_secret(threshold_dkg_rounds):
    dkg, states, _, _, round2_input, _ = threshold_dkg_rounds
    state = states[0]
    assert state.transport_secret is not None
    moved = replace(state, transport_secret=state.transport_secret.to("cuda:999"))
    expected = dkg.round2(state, round2_input, rng=ChaChaRng(dkg.device, bytes(32)))
    actual = dkg.round2(moved, round2_input, rng=ChaChaRng(dkg.device, bytes(32)))
    assert all(a.equals(b) for a, b in zip(actual.polys(), expected.polys(), strict=True))


def test_dkg_aggregation_preserves_independent_transport_placement(threshold_dkg_rounds):
    dkg, _, shares1, _, _, shares2 = threshold_dkg_rounds
    first = shares1[0]
    transport_keys = tuple(pk.to("cuda:999") for pk in first.round2_input.transport_public_keys)
    moved = replace(first, round2_input=replace(first.round2_input, transport_public_keys=transport_keys))
    keys, inputs = dkg.aggregate_round1([moved, shares1[1]])
    assert [pk.ct.device for pk in inputs.transport_public_keys] == ["cuda:999", dkg.device]

    inputs = replace(inputs, relinearization_h0=tuple(poly.to("cuda:999") for poly in inputs.relinearization_h0))
    shares = [
        replace(share, encrypted_secret_shares=(share.encrypted_secret_shares[0].to("cuda:999"), share.encrypted_secret_shares[1]))
        for share in shares2
    ]
    rlk, encrypted = dkg.aggregate_round2(inputs, shares)
    keys = keys.replace(rlk=rlk)
    assert all(poly.device == dkg.device for poly in keys.polys())
    assert [ct.device for ct in encrypted] == ["cuda:999", dkg.device]


def test_dkg_finish_does_not_use_common_masks(threshold_dkg_rounds):
    dkg, states, _, _, _, shares2 = threshold_dkg_rounds
    moved = tree.replace(dkg, lambda poly: poly.to("cuda:999"), leaf=Poly)
    ct = shares2[0].encrypted_secret_shares[0]
    actual, expected = moved.finish(states[0], ct).sk, dkg.finish(states[0], ct).sk
    assert actual is not None and expected is not None
    assert actual.equals(expected)


def test_additive_dkg_finish_preserves_share_placement(threshold_dkg_rounds):
    threshold_dkg, states, _, _, _, _ = threshold_dkg_rounds
    dkg = Dkg(threshold_dkg.params, SEED, multiparty=MultipartyParams(num_parties=2))
    share = states[0].secret.to("cuda:999")
    state = replace(states[0], secret=share, transport_secret=None)
    assert dkg.finish(state).device == share.device


@pytest.mark.parametrize(
    "stage,fault,error,match",
    [
        ("round1", "ring", ParameterMismatchError, "ring"),
        ("round2", "ring", ParameterMismatchError, "ring"),
        ("aggregate2", "ring", ParameterMismatchError, "ring"),
        ("finish", "ring", ParameterMismatchError, "ring"),
        ("round2", "device", ParameterMismatchError, "public key is on cuda:999"),
        ("aggregate2", "device", ValueError, "device mismatch"),
        ("finish", "device", ValueError, "device mismatch"),
    ],
)
def test_dkg_rejects_mismatched_transport_ciphertexts(threshold_dkg_rounds, stage, fault, error, match):
    dkg, states, shares1, keys, round2_input, shares2 = threshold_dkg_rounds
    pk = shares1[0].round2_input.transport_public_keys[0]
    ct = pk.ct.map_polys(lambda poly: poly.to_quotient_ring(dkg.params.ring_at(0))) if fault == "ring" else pk.ct.to("cuda:999")
    bad_pk = pk.replace(ct=ct)
    with pytest.raises(error, match=match):
        if stage == "round1":
            inputs = replace(shares1[0].round2_input, transport_public_keys=(bad_pk,))
            dkg.aggregate_round1([replace(shares1[0], round2_input=inputs), shares1[1]])
        elif stage == "round2":
            inputs = replace(round2_input, transport_public_keys=(bad_pk,) * dkg.multiparty.num_parties)
            dkg.round2(states[0], inputs)
        elif stage == "aggregate2":
            share = replace(shares2[0], encrypted_secret_shares=(ct,) * dkg.multiparty.num_parties)
            dkg.aggregate_round2(round2_input, [share, shares2[1]])
        else:
            dkg.finish(states[0], ct)


def test_dkg_messages_require_consistent_transport_rings():
    p = Poly.zero(mixed_prime_ring(4))
    other = Poly.zero(replace(p.ring, moduli=p.ring.moduli[:1]))
    ciphertexts = tuple(RLWECiphertext(a=poly, b=poly, noise_info=CKKSNoiseInfo.zero()) for poly in (p, other))
    with pytest.raises(ParameterMismatchError, match="rings must match"):
        DkgRound2Input((p,), (p,), tuple(PublicKey(ct) for ct in ciphertexts))
    with pytest.raises(ParameterMismatchError, match="rings must match"):
        DkgRound2Share((p,), ciphertexts)


def test_threshold_dkg_requires_transport_material(threshold_dkg_rounds):
    dkg, states, shares1, keys, round2_input, shares2 = threshold_dkg_rounds
    with pytest.raises(InvalidParameterError, match="no transport secret key"):
        dkg.round2(replace(states[0], transport_secret=None), round2_input)
    with pytest.raises(ParameterMismatchError, match="Expected 1 transport public keys"):
        inputs = replace(shares1[0].round2_input, transport_public_keys=())
        dkg.aggregate_round1([replace(shares1[0], round2_input=inputs), shares1[1]])
    with pytest.raises(ParameterMismatchError, match="Expected 2 transport public keys"):
        dkg.round2(states[0], replace(round2_input, transport_public_keys=()))
    with pytest.raises(ParameterMismatchError, match="Expected 2 encrypted secret shares"):
        dkg.aggregate_round2(round2_input, [replace(shares2[0], encrypted_secret_shares=()), shares2[1]])
    with pytest.raises(InvalidParameterError, match="requires an encrypted secret share"):
        dkg.finish(states[0])


def test_dkg_rejects_multiple_transport_keys_per_participant(threshold_dkg_rounds):
    dkg, _, shares1, _, _, _ = threshold_dkg_rounds
    inputs = shares1[0].round2_input
    share = replace(shares1[0], round2_input=replace(inputs, transport_public_keys=inputs.transport_public_keys * 2))
    with pytest.raises(ParameterMismatchError, match="Expected 1 transport public keys"):
        dkg.aggregate_round1([share, shares1[1]])


def test_additive_dkg_rejects_transport_material(threshold_dkg_rounds):
    threshold_dkg, states, shares1, _, round2_input, shares2 = threshold_dkg_rounds
    dkg = Dkg(threshold_dkg.params, SEED, multiparty=MultipartyParams(num_parties=2), rotations={1})
    state = replace(states[0], transport_secret=None)
    with pytest.raises(InvalidParameterError, match="does not use a transport secret key"):
        dkg.round2(states[0], round2_input)
    with pytest.raises(ParameterMismatchError, match="Expected 0 transport public keys"):
        dkg.aggregate_round1(shares1)
    with pytest.raises(ParameterMismatchError, match="Expected 0 transport public keys"):
        dkg.round2(state, round2_input)
    round2_input = replace(round2_input, transport_public_keys=())
    with pytest.raises(ParameterMismatchError, match="Expected 0 encrypted secret shares"):
        dkg.aggregate_round2(round2_input, shares2)
    with pytest.raises(InvalidParameterError, match="does not accept an encrypted secret share"):
        dkg.finish(state, shares2[0].encrypted_secret_shares[0])


@pytest.mark.parametrize("fault", ["count", "ring", "device"])
def test_dkg_rejects_invalid_digit_layout(threshold_dkg_rounds, fault):
    dkg, states, shares1, _, inputs, shares2 = threshold_dkg_rounds
    digits = inputs.relinearization_h0
    if fault == "count":
        invalid = digits[:-1]
    elif fault == "ring":
        invalid = tuple(poly.to_quotient_ring(dkg.params.highest_ring) for poly in digits)
    else:
        invalid = (*digits[:-1], digits[-1].to("cuda:999"))
    error = ValueError if fault == "device" else ParameterMismatchError
    match = "digit layout" if fault == "count" else fault
    h1 = inputs.relinearization_h1 if fault == "device" else invalid
    contribution = replace(shares1[0].round2_input, relinearization_h0=invalid, relinearization_h1=h1)
    with pytest.raises(error, match=match):
        dkg.aggregate_round1([replace(shares1[0], round2_input=contribution), shares1[1]])
    elt = next(iter(shares1[0].galois_keys))
    with pytest.raises(error, match=match):
        dkg.aggregate_round1([replace(shares1[0], galois_keys={elt: invalid}), shares1[1]])
    with pytest.raises(ParameterMismatchError, match=match):
        dkg.round2(states[0], replace(inputs, relinearization_h0=invalid, relinearization_h1=h1))
    with pytest.raises(error, match=match):
        dkg.aggregate_round2(inputs, [replace(shares2[0], relinearization_h_prime=invalid), shares2[1]])


def compute_lagrange_coefficient(i: int, parties: Iterable[int], q: int) -> int:
    num = 1
    den = 1
    for j in parties:
        if j != i:
            num = (num * j) % q
            den = (den * (j - i)) % q
    den_inv = pow(den, -1, q)
    return (num * den_inv) % q


# Reuse one DKG ceremony per class across the protocol checks.


def pytest_generate_tests(metafunc):
    if "extra_parties" in metafunc.fixturenames:
        cls = metafunc.cls
        counts = [0, 1] if cls.threshold is not None and cls.threshold < cls.num_parties else [0]
        metafunc.parametrize("extra_parties", counts)


class BaseTestMultipartyDKG:
    num_parties: int
    threshold: int | None

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def setup(cls, test_params: CKKSParams):
        params = test_params
        rotations = {2**i for i in range(params.logn - 1)}
        dkg = Dkg(
            params, SEED, multiparty=MultipartyParams(num_parties=cls.num_parties, threshold=cls.threshold), rotations=rotations, conjugation=True
        )
        states, round_1_messages = zip(*(dkg.round1() for _ in range(cls.num_parties)))
        secrets = [state.secret for state in states]
        round1_keys, round2_input = dkg.aggregate_round1(round_1_messages)
        round_2_messages = [dkg.round2(state, round2_input) for state in states]
        rlk, aggregated_shares = dkg.aggregate_round2(round2_input, round_2_messages)
        keys = round1_keys.replace(rlk=rlk).realize()

        cls.params = params
        cls.rotations = rotations
        cls.states = states
        cls.secrets = secrets
        cls.dkg = dkg
        cls.keys = keys
        cls.round1_keys = round1_keys
        cls.round2_input = round2_input
        cls.round_1_messages = round_1_messages
        cls.round_2_messages = round_2_messages
        cls.ctx = keys.context()
        cls.aggregated_shares = aggregated_shares
        cls.group_sk = sum(secrets, start=Poly.zero(params.highest_ring)).realize()

        yield

        cls.ctx = None

    def test_transport_keys_and_shares_structure(self):
        expected = 0 if self.threshold is None else self.num_parties
        assert len(self.round2_input.transport_public_keys) == expected
        assert len(self.aggregated_shares) == expected
        for state in self.states:
            if self.threshold is None:
                assert state.transport_secret is None
            else:
                assert isinstance(state.transport_secret, Poly)
                assert state.transport_secret.ring == self.params.ring_at(1)

    def test_threshold_round1_transport_keys_follow_input_order(self):
        if self.threshold is None:
            pytest.skip("transport keys are only used for threshold DKG")
        shares = tuple(reversed(self.round_1_messages))
        _, round2_input = self.dkg.aggregate_round1(shares)
        for got, share in zip(round2_input.transport_public_keys, shares, strict=True):
            expected = share.round2_input.transport_public_keys[0]
            assert all(a.equals(b) for a, b in zip(got.polys(), expected.polys(), strict=True))

    def test_final_keyset_is_public(self):
        assert self.round1_keys.rlk is None
        assert self.keys.multiparty == self.dkg.multiparty
        assert self.keys.context().multiparty == self.dkg.multiparty
        assert self.keys.sk is None
        assert self.keys.public().sk is None
        assert self.keys.pk is not None
        assert self.keys.rlk is not None
        assert self.keys.galois_keys

    def test_shamir_secret_share_reconstruction(self):
        threshold = self.threshold
        if threshold is None:
            pytest.skip("Shamir secret sharing is not used for n-of-n DKG")
        params = self.params
        states = self.states
        round_2_messages = self.round_2_messages

        party_1_msg = round_2_messages[0]
        party_1_shares = party_1_msg.encrypted_secret_shares

        reconstructed_shares = {}

        for j in range(1, threshold + 1):
            enc_share_j = party_1_shares[j - 1]
            secret_share_j = self.dkg.finish(states[j - 1], enc_share_j).sk
            assert secret_share_j is not None
            reconstructed_shares[j] = secret_share_j

        s_1_reconstructed = Poly.zero(params.ring_at(0))
        q = params.highest_ring.q

        for j in range(1, threshold + 1):
            c_j = compute_lagrange_coefficient(j, range(1, threshold + 1), q)
            s_1_reconstructed = s_1_reconstructed + reconstructed_shares[j] * c_j

        diff = s_1_reconstructed - states[0].secret.to_quotient_ring(params.ring_at(0))
        assert max(abs(c) for c in diff.to_centered_list()) == 0, "Reconstructed s_1 does not match original s_1"

    def test_multiparty_decryption(self, extra_parties):
        params = self.params
        assert self.ctx is not None
        parties = self._decryption_parties(extra_parties)
        with self.ctx:
            ct_prod = random_ciphertext(msg_bound=10) * random_ciphertext(msg_bound=100)
            prepared = ct_prod.to_level(0).rerandomize()
            assert prepared.ring.moduli != self.secrets[0].ring.moduli
            expected_values = decode(decrypt(ct_prod, secret=self.group_sk), params.slot_count)

        sigma = self._sigma(prepared)
        partials = [self._partial_decrypt(prepared, party, parties, flooding_std_dev=sigma) for party in parties]
        assert all(partial.d.ring == prepared.ring for partial in partials)

        with self.ctx:
            values = decode(decrypt(prepared, partials=partials), params.slot_count)
        assert values == pytest.approx(expected_values, rel=1e-3)

    def _decryption_parties(self, extra_parties: int = 0) -> set[int]:
        count = self.num_parties if self.threshold is None else min(self.threshold + extra_parties, self.num_parties)
        return set(range(self.num_parties - count + 1, self.num_parties + 1))

    def _participant_keys(self, party: int) -> CKKSKeySet:
        encrypted_share = self.aggregated_shares[party - 1] if self.aggregated_shares else None
        return self.dkg.finish(self.states[party - 1], encrypted_share)

    def _partial_decrypt(self, ct: Ciphertext, party: int, parties: set[int], *, flooding_std_dev: float) -> PartialDecryption:
        keys = self._participant_keys(party)
        with keys.context():
            return partial_decrypt(
                ct,
                flooding_std_dev=flooding_std_dev,
                party=party if self.threshold is not None else None,
                quorum=parties if self.threshold is not None else None,
            )

    def _sigma(self, ct: Ciphertext) -> float:
        return multiparty.flooding_std_dev(
            ct.noise_info.noise_heuristic_bound, self.params, statistical_security=STATISTICAL_SECURITY, num_decryptions=NUM_DECRYPTIONS
        )

    def test_public_key_shares_have_bounded_noise(self):
        params = self.params
        secrets = self.secrets
        round_1_messages = self.round_1_messages

        def test_share(mask: Poly, share: Poly, secret_share: Poly) -> None:
            s_on_mask_ring = secret_share.to_quotient_ring(mask.ring)
            diff = share + mask * s_on_mask_ring
            assert 0 < max(abs(c) for c in diff.to_centered_list()) <= params.noise_dist.coeff_inf_norm_bound()

        pk = self.round1_keys.pk
        assert pk is not None
        pk_mask = pk.ct.a
        for secret, message in zip(secrets, round_1_messages):
            test_share(pk_mask, message.public_key_body, secret)

        pk_decrypted_poly = pk.ct._decrypt(self.group_sk)
        assert max(abs(c) for c in pk_decrypted_poly.to_centered_list()) <= (self.num_parties * params.noise_dist.coeff_inf_norm_bound())

    def test_generated_keys_use_ntt_representation(self):
        assert all(secret.domain is Domain.NTT for secret in self.secrets)
        keys = self.keys
        assert keys.pk is not None and keys.rlk is not None
        for key in (self.round1_keys, keys):
            assert all(poly.domain is Domain.NTT for poly in key.polys())

    def test_galois_keys(self):
        params = self.params
        assert self.ctx is not None

        with self.ctx:
            ct = random_ciphertext(msg_bound=1000)
            ct_values = decode(decrypt(ct, secret=self.group_sk), params.slot_count)
            for step in self.rotations:
                ct_rotated = ct.rotate(step)
                ct_rotated_values = decode(decrypt(ct_rotated, secret=self.group_sk), params.slot_count)
                expected_values = ct_values[step:] + ct_values[:step]
                assert ct_rotated_values == pytest.approx(expected_values, abs=1e-7)

            ct_conjugated = ct.conjugate()
            ct_conjugated_values = decode(decrypt(ct_conjugated, secret=self.group_sk), params.slot_count)
            expected_values = [v.conjugate() for v in ct_values]
            assert ct_conjugated_values == pytest.approx(expected_values, abs=1e-7)

    def test_relinearization_key(self):
        params = self.params
        assert self.ctx is not None

        with self.ctx:
            ct1 = random_ciphertext(msg_bound=1000)
            ct2 = random_ciphertext(msg_bound=1000)
            ct_prod = ct1 * ct2

            ct_prod_values = decode(decrypt(ct_prod, secret=self.group_sk), params.slot_count)
            ct1_values = decode(decrypt(ct1, secret=self.group_sk), params.slot_count)
            ct2_values = decode(decrypt(ct2, secret=self.group_sk), params.slot_count)
            expected_values = [va * vb for va, vb in zip(ct1_values, ct2_values)]
            assert ct_prod_values == pytest.approx(expected_values, abs=1e-7)

    def test_custom_flooding_amount(self):
        assert self.ctx is not None
        parties = self._decryption_parties()
        pt_noise_amt = 0.0001
        with self.ctx:
            ct = random_ciphertext(msg_bound=1000)
            ct_values = decode(decrypt(ct, secret=self.group_sk), self.params.slot_count)
            prepared = ct.to_level(0).rerandomize()
            # Convert the calibrated decoded noise to a coefficient-space estimate.
            sigma = multiparty.flooding_std_dev(
                Bound(pt_noise_amt * prepared.scale / sqrt(self.params.n)),
                self.params,
                statistical_security=STATISTICAL_SECURITY,
                num_decryptions=NUM_DECRYPTIONS,
            )
        partials = [self._partial_decrypt(prepared, party, parties, flooding_std_dev=sigma) for party in parties]
        with self.ctx:
            values = decode(decrypt(prepared, partials=partials), self.params.slot_count)
        errors = [abs(v1 - v2) for v1, v2 in zip(values, ct_values)]
        max_error = max(errors)
        expected_error_std_dev = pt_noise_amt * sqrt(12 * NUM_DECRYPTIONS * 2**STATISTICAL_SECURITY)
        assert expected_error_std_dev / 2 < max_error <= 13 * expected_error_std_dev


def slow_if_cuda(func):
    if Device.DEFAULT != "python":
        return pytest.mark.slow()(func)
    return func


@slow_if_cuda
class TestMultipartyDKG_3(BaseTestMultipartyDKG):
    num_parties = 3
    threshold = None

    @pytest.mark.parametrize("level", [1, 2])
    def test_multiparty_decryption_at_nonzero_level(self, level):
        assert self.ctx is not None
        parties = self._decryption_parties()
        with self.ctx:
            ct_prod = random_ciphertext(msg_bound=10) * random_ciphertext(msg_bound=100)
            prepared = ct_prod.to_level(level).rerandomize()
            assert prepared.ring == self.params.ring_at(level)
            expected_values = decode(decrypt(ct_prod, secret=self.group_sk), self.params.slot_count)

        sigma = self._sigma(prepared)
        partials = []
        for party in sorted(parties):
            partial = self._partial_decrypt(prepared, party, parties, flooding_std_dev=sigma)
            assert partial.d.ring == prepared.ring
            secret = self._participant_keys(party).sk
            assert secret is not None
            s = secret.to_quotient_ring(prepared.ring)
            diff = partial.d - prepared.a * s
            max_abs = max(abs(c) for c in diff.to_centered_list())
            # Flooding noise must not scale with the ciphertext level.
            assert 3 * sigma < max_abs <= 13 * sigma
            partials.append(partial)

        with self.ctx:
            aggregated = decrypt(prepared, partials=partials)
            assert aggregated.m.ring == self.params.ring_at(level)
            values = decode(aggregated, self.params.slot_count)
        assert values == pytest.approx(expected_values, rel=1e-3)


@slow_if_cuda
class TestMultipartyDKG_5(TestMultipartyDKG_3):
    num_parties = 5
    threshold = None


@slow_if_cuda
class TestThresholdDKG_3_2(BaseTestMultipartyDKG):
    num_parties = 3
    threshold = 2


@slow_if_cuda
class TestThresholdDKG_3_3(BaseTestMultipartyDKG):
    num_parties = 3
    threshold = 3


@slow_if_cuda
class TestThresholdDKG_5_3(BaseTestMultipartyDKG):
    num_parties = 5
    threshold = 3


def _run_dkg(dkg: Dkg) -> tuple[CKKSKeySet, dict[int, CKKSKeySet]]:
    states, shares = zip(*(dkg.round1() for _ in range(dkg.multiparty.num_parties)), strict=True)
    keys, round2 = dkg.aggregate_round1(shares)
    rlk, encrypted = dkg.aggregate_round2(round2, [dkg.round2(state, round2) for state in states])
    participants = {i: dkg.finish(state, encrypted[i - 1] if encrypted else None) for i, state in enumerate(states, 1)}
    return keys.replace(rlk=rlk).realize(), participants


def _joint_decode(ct, public_keys, participants, *, flooding_std_dev, quorum=None):
    parties = participants.keys() if quorum is None else quorum
    partials = []
    for party in parties:
        with participants[party].context():
            partials.append(partial_decrypt(ct, flooding_std_dev=flooding_std_dev, quorum=quorum, party=party if quorum is not None else None))
    with public_keys.context():
        return decode(decrypt(ct, partials=partials))


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_checked_threshold_bootstrap():
    log_n = 16
    params = CKKSParams.default(
        log_n=log_n,
        depth=4,
        p_prime_count=2,
        secret_key_dist=Ternary(1 << log_n),
        boot_params=BootstrappingParams(cts_depth=3, stc_depth=3, exp_degree=255, squaring_count=5, failure_probability=2**-64, precision_gains=(8,)),
        security=128,
    )
    keys, participants = _run_dkg(Dkg(params, SEED, multiparty=MultipartyParams(num_parties=3, threshold=2)))
    values = random_values_with_norm(params.slot_count, norm=0.5)
    with keys.context():
        ct = encrypt(encode(values, msg_bound=1))
        result = bootstrap((ct * ct).to_level(0), log_bound=0)
        assert result.level == 4
        prepared = result.to_level(0).rerandomize()

    decoded = _joint_decode(prepared, keys, participants, flooding_std_dev=2**20, quorum=(2, 3))
    assert decoded == pytest.approx([v * v for v in values], abs=1e-4)


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_high_precision_additive_decryption():
    log_n = 16
    boot_params = BootstrappingParams(
        cts_depth=3, stc_depth=3, exp_degree=63, squaring_count=3, failure_probability=2**-64, precision_gains=(8, 10, 10, 10, 10)
    )
    params = CKKSParams.default(
        log_n=log_n,
        depth=6,
        log_precision=90,
        dtype=dtypes.uint32,
        p_prime_count=4,
        secret_key_dist=Ternary(1 << log_n),
        boot_params=boot_params,
        security=128,
    )
    keys, participants = _run_dkg(Dkg(params, SEED, multiparty=MultipartyParams(num_parties=3), rotations={1}, conjugation=True))
    values = random_values_with_norm(params.slot_count, norm=10)
    sigma = 2**76
    with keys.context():
        ct = encrypt(encode(values, msg_bound=10))
        prepared = (ct * ct).rotate(1).conjugate().to_level(2).rerandomize()
        slot_noise = sigma * sqrt(params.n) / float(prepared.scale)

    expected = [(v * v).conjugate() for v in values[1:] + values[:1]]
    assert _joint_decode(prepared, keys, participants, flooding_std_dev=1) == pytest.approx(expected, abs=1e-7)
    flooded = _joint_decode(prepared, keys, participants, flooding_std_dev=sigma)
    max_error = max(abs(a - b) for a, b in zip(flooded, expected, strict=True))
    assert slot_noise / 2 < max_error <= 13 * slot_noise


@pytest.mark.parametrize("threshold", [None, 2])
def test_dkg_aggregation_and_recovery_replay_with_new_private_state(threshold):
    params = CKKSParams.generate_prefix(log_n=10, depth=1, log_precision=40, log_range=0, dtype=dtypes.uint32, security=None)
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2, threshold=threshold), rotations={1})
    captures = []

    @jit
    def aggregate_round1(shares):
        captures.append("aggregate1")
        return dkg.aggregate_round1(shares)

    @jit
    def aggregate_round2(round2_input, shares):
        captures.append("aggregate2")
        return dkg.aggregate_round2(round2_input, shares)

    @jit
    def finish(state, encrypted):
        captures.append("finish")
        return dkg.finish(state, encrypted)

    public_keys = []
    for _ in range(2):
        secrets = [generate_secret(params) for _ in range(2)]
        states, shares1 = zip(*(dkg.round1(secret) for secret in secrets))
        keys, round2_input = aggregate_round1(shares1)
        shares2 = tuple(dkg.round2(state, round2_input) for state in states)
        rlk, encrypted = aggregate_round2(round2_input, shares2)
        keys = keys.replace(rlk=rlk)
        shares = [finish(state, encrypted[i] if encrypted else None).sk for i, state in enumerate(states)]
        assert shares[0] is not None and shares[1] is not None
        ring = params.highest_ring if threshold is None else params.ring_at(0)
        group_secret = sum(secrets, start=Poly.zero(params.highest_ring))
        expected = group_secret.to_quotient_ring(ring)
        reconstructed = shares[0] + shares[1] if threshold is None else shares[0] * 2 - shares[1]
        assert reconstructed.equals(expected)
        with keys.context():
            ct = encrypt(encode([0.25, -0.5, 0.75], msg_bound=1))
            assert decode(decrypt((ct * ct).rotate(1), secret=group_secret), 3) == pytest.approx([0.25, 0.5625, 0], abs=1e-6)
        assert keys.pk is not None
        public_keys.append(keys.pk.ct.b.tolist())
    assert public_keys[0] != public_keys[1]
    assert sorted(captures) == ["aggregate1", "aggregate2", "finish"]


@pytest.mark.parametrize("threshold", [None, 2])
def test_dkg_rounds_use_explicit_randomness(threshold):
    params = CKKSParams.generate_prefix(log_n=10, depth=1, log_precision=40, log_range=0, dtype=dtypes.uint32, security=None)
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2, threshold=threshold))
    secrets = [generate_secret(params) for _ in range(2)]
    (state1, share1), (state2, share2), (_, different) = [dkg.round1(secrets[0], rng=rng_from_int_seed(seed)) for seed in (11, 11, 12)]
    assert all(a.equals(b) for a, b in zip(state1.polys(share1), state2.polys(share2), strict=True))
    assert not share1.public_key_body.equals(different.public_key_body)
    keys, round2_input = dkg.aggregate_round1([share1, dkg.round1(secrets[1])[1]])
    first, second, different = [dkg.round2(state1, round2_input, rng=rng_from_int_seed(seed)) for seed in (11, 11, 12)]
    assert all(a.equals(b) for a, b in zip(first.polys(), second.polys(), strict=True))
    assert any(not a.equals(b) for a, b in zip(first.polys(), different.polys(), strict=True))


@pytest.mark.parametrize("threshold", [None, 2])
def test_dkg_round1_samples_contribution_from_private_rng(threshold):
    params = small_params()
    dkg = Dkg(params, SEED, multiparty=MultipartyParams(num_parties=2, threshold=threshold))
    state, share = dkg.round1(rng=rng_from_int_seed(17))
    repeated_state, repeated_share = dkg.round1(rng=rng_from_int_seed(17))
    assert all(a.equals(b) for a, b in zip(state.polys(share), repeated_state.polys(repeated_share), strict=True))
    other = Dkg(params, SEED + 1, multiparty=dkg.multiparty)
    assert other.round1(rng=rng_from_int_seed(17))[0].secret.equals(state.secret)
    assert not dkg.round1(rng=rng_from_int_seed(18))[0].secret.equals(state.secret)
