import pickle
from dataclasses import replace

import pytest

from ren import realize
from ren.context import Context
from ren.device import Device
from ren.polydist import KeyDist, SparseTernary
from ren.random import set_global_rng_seed
from ren.ckks import CKKSKeySet, CKKSParams, Ciphertext, decode, decrypt, encode, encrypt, keygen
from ren.ckks.metadata import Bound, CKKSNoiseInfo
from tests.ckks.params import small_params


@pytest.fixture(scope="module")
def test_params() -> CKKSParams:
    return small_params()


@pytest.fixture(scope="module")
def keys(test_params: CKKSParams) -> CKKSKeySet:
    set_global_rng_seed(42)
    rotations = {2**i for i in range(test_params.logn - 1)}
    return keygen(test_params, rotations=rotations, conjugation=True).realize()


def _sample_values(slots: int, *, shift: int = 0) -> list[complex]:
    return [complex((i + shift) % 13 - 6, ((2 * i + shift) % 11) - 5) for i in range(slots)]


def test_ciphertext_pickle_preserves_large_bounds(test_params: CKKSParams):
    noise_bound = Bound("1.234567890123456789e150")
    noise_heuristic_bound = Bound("9.876543210987654321e120")
    msg_bound = Bound("7.6543210987654321e90")
    with test_params.context(track_plaintext_values=True):
        # Replacing a frozen ciphertext re-runs its validation, which needs the
        # active context to resolve the ciphertext scale.
        ct = replace(
            Ciphertext.zero(),
            noise_info=CKKSNoiseInfo(
                noise_bound=noise_bound,
                noise_heuristic_bound=noise_heuristic_bound,
                msg_bound=msg_bound,
            ),
        )

    restored = pickle.loads(pickle.dumps(ct))
    assert restored.noise_info == ct.noise_info


def test_public_keyset_pickle_supports_eval(keys: CKKSKeySet):
    restored = pickle.loads(pickle.dumps(keys.public()))
    assert restored.sk is None
    assert restored.pk is not None
    assert restored.params == keys.params
    eval_ctx = restored.context()

    values_a = _sample_values(eval_ctx.params.slot_count, shift=1)
    values_b = _sample_values(eval_ctx.params.slot_count, shift=2)

    with eval_ctx:
        ct_a = encrypt(encode(values_a, msg_bound=16))
        ct_b = encrypt(encode(values_b, msg_bound=16))
        out_ct = (ct_a * ct_b).rotate(3)
        with pytest.raises(RuntimeError, match="secret key"):
            decrypt(out_ct)

    expected_mul = [a * b for a, b in zip(values_a, values_b)]
    expected = expected_mul[3:] + expected_mul[:3]

    with keys.context():
        out = decode(decrypt(out_ct), keys.params.slot_count)
        assert out == pytest.approx(expected, abs=1e-4)


def test_ciphertext_realize_multiple(keys: CKKSKeySet):
    # The inherited method realizes several ciphertexts together while
    # preserving each one's plaintext.
    with keys.context():
        value_lists = [_sample_values(keys.params.slot_count, shift=i) for i in range(3)]
        cts = [encrypt(encode(values, msg_bound=16)) for values in value_lists]

        realize(*cts)
        assert all(ct.a.is_realized and ct.b.is_realized for ct in cts)

        for ct, values in zip(cts, value_lists):
            out = decode(decrypt(ct), keys.params.slot_count)
            assert out == pytest.approx(values, abs=1e-4)


def test_plaintext_pickle_roundtrip(keys: CKKSKeySet):
    values = _sample_values(keys.params.slot_count)
    with keys.context(track_plaintext_values=True):
        pt = encode(values, msg_bound=16)
        restored = pickle.loads(pickle.dumps(pt))
        out = decode(restored, keys.params.slot_count)
    assert restored._values is None
    assert out == pytest.approx(values, abs=1e-4)


def test_ciphertext_pickle_roundtrip(keys: CKKSKeySet):
    values = _sample_values(keys.params.slot_count)
    with keys.context(track_plaintext_values=True):
        ct = encrypt(encode(values, msg_bound=16))
        restored = pickle.loads(pickle.dumps(ct))
        out = decode(decrypt(restored), keys.params.slot_count)
    assert restored._values is None
    assert out == pytest.approx(values, abs=1e-4)


def test_keyset_pickle_roundtrip_supports_eval_and_decrypt(keys: CKKSKeySet):
    restored_keys = pickle.loads(pickle.dumps(keys))
    restored_ctx = restored_keys.context()

    assert restored_ctx.keyset.sk is not None
    assert restored_ctx.keyset.pk is not None
    assert restored_ctx.keyset.rlk is not None
    assert restored_ctx.keyset.galois_keys

    values_a = _sample_values(restored_ctx.params.slot_count, shift=1)
    values_b = _sample_values(restored_ctx.params.slot_count, shift=2)
    with restored_ctx:
        ct = (encrypt(encode(values_a, msg_bound=16)) * encrypt(encode(values_b, msg_bound=16))).rotate(3)
        out = decode(decrypt(ct), restored_ctx.params.slot_count)

    expected_mul = [a * b for a, b in zip(values_a, values_b)]
    expected = expected_mul[3:] + expected_mul[:3]
    assert out == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("device_scope", [lambda device: Context(DEV=device), Device.scope], ids=["DEV", "scope"])
def test_keyset_pickle_loads_onto_active_device(keys: CKKSKeySet, device_scope):
    payload = pickle.dumps(keys)
    target_device = "cuda:3" if keys.device != "cuda:3" else "python"

    with device_scope(target_device):
        restored_keys = pickle.loads(payload)

    assert restored_keys.device == target_device
    assert restored_keys.context().device == target_device
    assert restored_keys.sk is not None
    assert restored_keys.sk.device == target_device


# Both dense and sparse scheme secrets must survive an encapsulation-key round trip.
@pytest.mark.parametrize("secret_key_dist", [None, SparseTernary(n=1 << 10, h=128)], ids=["dense", "sparse"])
def test_keyset_pickle_roundtrip_encapsulation_keys(secret_key_dist: KeyDist | None):
    from ren.dtype import dtypes
    from ren.ckks.params import BootstrappingParams

    boot_params = BootstrappingParams(
        cts_depth=2,
        stc_depth=2,
        exp_degree=63,
        squaring_count=2,
        failure_probability=2**-16,
        sparse_encapsulation_weight=32,
    )
    encap_params = CKKSParams.generate_prefix(
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
    encap_keys = keygen(encap_params)

    restored = pickle.loads(pickle.dumps(encap_keys))
    assert restored.params.boot_params is not None and restored.params.boot_params.sparse_encapsulation
    assert restored.params.secret_key_dist == encap_params.secret_key_dist
    for original, copy in (
        (encap_keys.dense_to_sparse_key, restored.dense_to_sparse_key),
        (encap_keys.sparse_to_dense_key, restored.sparse_to_dense_key),
    ):
        assert original is not None and copy is not None
        assert copy.ring_qt == original.ring_qt
        assert copy.digit_sizes == original.digit_sizes
        assert len(copy.digits) == len(original.digits)
        # Verify serialization fidelity directly: every digit polynomial must
        # round-trip bit-for-bit (raw RNS limbs). Combined with the separate
        # bootstrap-with-encapsulation test, this makes re-bootstrapping the
        # restored context here redundant (and ~2s cheaper).
        for od, cd in zip(original.digits, copy.digits, strict=True):
            od.realize()
            cd.realize()
            assert bytes(cd.a.data()) == bytes(od.a.data())
            assert bytes(cd.b.data()) == bytes(od.b.data())
