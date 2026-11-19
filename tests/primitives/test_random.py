import hashlib
from bisect import bisect_right
from fractions import Fraction

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from ren.dtype import dtypes
from ren.backends.python.random import _chacha_blocks, _reduce_uniform_blocks, _sample_one
from ren.poly import Poly
from ren.polydist import DiscreteGaussian, SparseTernary, Ternary, UniformBounded, UniformModQ
from ren.random import ChaChaRng
from ren.tables import DistTables, TreeNodeEncoder
from tests.utils import mixed_prime_ring


def _stream_bytes(rng: ChaChaRng, stream_id: int, n_bytes: int = 128) -> bytes:
    iv = b"\x00" * 4 + stream_id.to_bytes(8, "little") + b"\x00" * 4
    encryptor = Cipher(algorithms.ChaCha20(rng._seed, iv), mode=None).encryptor()
    return encryptor.update(b"\x00" * n_bytes)


def test_chacha_rng_uses_rfc_stream_nonces():
    rng = ChaChaRng("python", bytes(range(32)))

    stream_0 = _stream_bytes(rng, rng.reserve())
    stream_1 = _stream_bytes(rng, rng.reserve())

    assert hashlib.sha256(stream_0).hexdigest() == "3c2ae99432eee70b39f6c2bfbc0904bd729efc8f7079b1c546a6424878c94d39"
    assert hashlib.sha256(stream_1).hexdigest() == "6b6b3fd633e3418fcfd5418a75303f4c95cabab000a77f30ab06c33b9ba017aa"


def test_chacha_blocks_match_rfc_8439_vector():
    nonce_prefix = bytes.fromhex("000000090000004a")
    block = _chacha_blocks(bytes(range(32)), int.from_bytes(nonce_prefix, "little"), 1, 1)

    assert block.astype("<u4", copy=False).tobytes().hex() == (
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"
    )


def test_chacha_rng_replays_a_reserved_stream():
    key = bytes(range(32))
    rng = ChaChaRng("python", key)
    stream_id = rng.reserve()

    first = _stream_bytes(rng, stream_id)
    second = _stream_bytes(rng, stream_id)

    assert first == second


def test_chacha_rng_derivation_is_hierarchical_and_does_not_advance_parent():
    rng = ChaChaRng("python", bytes(range(32)))

    child = rng.derive(["dkg", "round-1"])
    chained_child = rng.derive("dkg").derive("round-1")

    assert child.device == rng.device
    assert child.reserve() == 0
    assert chained_child.reserve() == 0
    assert child._seed.hex() == "57bc06c79b790c7c2ac9ef62f4aceeab6c43a77e1de17344738fcb47f701cb32"
    assert _stream_bytes(child, 0) == _stream_bytes(chained_child, 0)
    assert _stream_bytes(child, 0) != _stream_bytes(rng, 0)
    assert rng.reserve() == 0


def test_chacha_rng_derivation_treats_a_string_as_one_component():
    rng = ChaChaRng("python", bytes(range(32)))

    assert _stream_bytes(rng.derive("artifact"), 0) == _stream_bytes(rng.derive(["artifact"]), 0)
    assert _stream_bytes(rng.derive("a/b"), 0) != _stream_bytes(rng.derive(["a", "b"]), 0)


@pytest.mark.parametrize("path", [[], [""], ["café"], [1]])
def test_chacha_rng_rejects_invalid_derivation_paths(path):
    rng = ChaChaRng("python", bytes(range(32)))

    with pytest.raises((TypeError, ValueError)):
        rng.derive(path)


def test_uniform_bounded_accepts_large_mathematical_bounds():
    assert UniformBounded(64, 1 << 62).bound == 1 << 62


def test_uniform_rejection_discards_only_the_incomplete_high_bucket():
    modulus = 5
    limit = (1 << 128) - (1 << 128) % modulus
    raw_values = (modulus - 1, limit - 1, limit)
    blocks = np.zeros((len(raw_values), 16), dtype=np.uint32)
    for block, value in zip(blocks, raw_values, strict=True):
        block[:4] = np.frombuffer(value.to_bytes(16, "little"), dtype="<u4")

    remainders, accepted = _reduce_uniform_blocks(blocks, modulus)

    assert remainders.tolist() == [modulus - 1, modulus - 1, 0]
    assert accepted.tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("dtype", "dist", "expected"),
    [
        (dtypes.uint32, Ternary(64), "eabf29aba0caf80f902017b069c5f3d65b3427ce804d6c8d3bd173b973e0d2f6"),
        (dtypes.uint32, SparseTernary(64, 16), "3ca9758c9efd347484bd5c9d288745e6247b4e625fe463552a4ff6bb3418a711"),
        (dtypes.uint32, UniformBounded(64, 10), "64399a618ce922d1690747cd8309aafe95a0b5482a1d807798d1325df7006611"),
        (dtypes.uint32, UniformBounded(64, 100), "9ccac11a4102414bd11aa5cc6e05103e569ab9fc76bde8af3aca413863304b8b"),
        (dtypes.uint32, UniformModQ(64), "c4dafacafc3706862a9dc47ba32d10dcac65083c57912598d79cd846cdcabf42"),
        (
            dtypes.uint32,
            DiscreteGaussian(4096, Fraction(16, 5)),
            "6e7aac159a6c9db7e4de61b4aee5560af4efe5b1eb77e69459d7ddb8beb0c688",
        ),
        (
            dtypes.uint32,
            DiscreteGaussian(4096, Fraction(1 << 70)),
            "893cf73c052e0c9c1038ce2ab0b70d2114ed12a9b94245c7d7743c157d1970a3",
        ),
        (dtypes.uint64, Ternary(64), "0ff1eb6af14ae5d8c96a7a539136f9e1bd6cc1d1f69283aca31ab6e16df2415f"),
        (dtypes.uint64, SparseTernary(64, 16), "9957b679a2ced8e338f2719913303228e3d3595fcb0fe7300ac22f088a81ebac"),
        (dtypes.uint64, UniformBounded(64, 10), "9321fc51ac55ff2d167e6cff0fa56fc3ba9f7aac894e2381f9b07ff873b0a7b8"),
        (dtypes.uint64, UniformBounded(64, 100), "22fcdcb8d58e34a4eb44cff3d0f1171773872eb79a89bbe4d6670ceacaaabd10"),
        (dtypes.uint64, UniformModQ(64), "bff25eefc3619611cb8d958aae82a46bcfc19f352253a4ed6bd99a5e6a0d5382"),
        (
            dtypes.uint64,
            DiscreteGaussian(4096, Fraction(16, 5)),
            "71a0248652fa5e7640db6021de0f39c2857ba06078c151505d15a9dc61fa3c3d",
        ),
        (
            dtypes.uint64,
            DiscreteGaussian(4096, Fraction(1 << 70)),
            "374271b9c4a2f9ce2a9f634847139f9c48f227e20658485229e85d26fe23fdfe",
        ),
    ],
)
def test_random_polynomial_known_answer(dtype, dist, expected):
    ring = mixed_prime_ring(dist.n, dtype)
    poly = Poly.rand(dist, ring, rng=ChaChaRng("python", bytes(range(ChaChaRng.seed_size))))
    width = (ring.q.bit_length() + 7) // 8
    encoded = b"".join(value.to_bytes(width, "little") for value in poly)

    assert hashlib.sha256(encoded).hexdigest() == expected


def test_gaussian_cdf_boundaries():
    cdf = DistTables._discrete_gaussian_cdf(DistTables._mp.mpf(16) / 5)
    encoder = TreeNodeEncoder(dtypes.int8)
    tree = memoryview(DistTables._generate_tree(cdf, DistTables.MAX_DEPTH, encoder))
    # At sigma = 3.2 the last magnitude in the sampling range is 45.
    largest_draw = np.full(DistTables.MAX_DEPTH + 1, 0xFFFFFFFF, dtype=np.uint32)
    assert abs(_sample_one(tree, largest_draw, encoder)) == 45

    precision = 32 * DistTables.MAX_DEPTH
    cuts = [int(p * (1 << precision)) for p in cdf[:-1]]
    for cut in cuts:
        for value in (cut - 1, cut):
            words = [(value >> shift) & 0xFFFFFFFF for shift in range(precision - 32, -1, -32)]
            result = _sample_one(tree, np.array([*words, 0], dtype=np.uint32), encoder)
            assert abs(result) == bisect_right(cuts, value)
