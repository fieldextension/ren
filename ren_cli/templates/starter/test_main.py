import pytest
from ren.ckks import CKKSParams, decode, decrypt, encode, encrypt, keygen

from main import circuit


def test_circuit():
    keys = keygen(CKKSParams.default(boot_params=None))

    with keys.context():
        encrypted = encrypt(encode([1.0, 2.0, 3.0], msg_bound=4))

    with keys.public().context():
        result = circuit(encrypted)

    with keys.context():
        decoded = decode(decrypt(result), length=3)

    assert decoded == pytest.approx([3.0, 7.0, 13.0], rel=0, abs=1e-6)
