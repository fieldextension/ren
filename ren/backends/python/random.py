from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
import numpy as np

from ren.buffer import Buffer
from ren.dtype import to_np_dtype
import ren.montgomery as mont
from ren.polydist import DiscreteGaussian, PolyDist, SparseTernary, Ternary, UniformBounded, UniformModQ
from ren.ring import RingSpec
from ren.rns import _encoded_rns_coeffs
from ren.tables import DistTables, TreeNodeEncoder


_CHACHA_BLOCK_LIMIT = 1 << 32


def rand(dist: PolyDist, ring: RingSpec, *, key: bytes, stream_id: int, tables: DistTables) -> np.ndarray:
    if dist.n != ring.n:
        raise ValueError(f"Ring degree mismatch: {ring.n} != {dist.n}")
    match dist:
        case Ternary(n=n):
            words = _chacha_blocks(key, stream_id, 0, n)[:, 0]
            values = (words & 1).astype(np.int8) - ((words >> 1) & 1).astype(np.int8)
            return _lookup_mont_rns_coeffs(values.astype(np.int64) + 1, np.array([-1, 0, 1]), ring)

        case SparseTernary(n=n, h=h):
            locations, signs = _sparse_ternary(key, stream_id, n, h)
            coeff_dtype = to_np_dtype(ring.dtype)
            rns_coeffs = np.zeros(n * len(ring.moduli), dtype=coeff_dtype)
            for i, modulus in enumerate(ring.moduli):
                start = i * n
                row = rns_coeffs[start : start + n]
                row[locations] = mont.context(ring.dtype, modulus).encode(signs)
            return rns_coeffs

        case UniformBounded(n=n, bound=bound):
            modulus = 2 * bound + 1
            samples = _uniform_samples(key, stream_id, start=0, count=n, stride=n, modulus=modulus)
            if modulus <= n:
                support = np.arange(-bound, bound + 1, dtype=np.int64)
                return _lookup_mont_rns_coeffs(samples.astype(np.int64), support, ring)
            values = samples.astype(object) - bound
            return _encoded_rns_coeffs(values, ring)

        case UniformModQ(n=n):
            rns_coeffs = np.empty(n * len(ring.moduli), dtype=to_np_dtype(ring.dtype))
            sample_count = n * len(ring.moduli)
            for ring_idx, modulus in enumerate(ring.moduli):
                start = ring_idx * n
                rns_coeffs[start : start + n] = _uniform_samples(
                    key,
                    stream_id,
                    start=start,
                    count=n,
                    stride=sample_count,
                    modulus=modulus,
                )
            return rns_coeffs

        case DiscreteGaussian(n=_, stdev=stdev):
            plan = tables.discrete_gaussian(stdev)
            encoder = TreeNodeEncoder(plan.base_sample_dtype)
            match plan.sample_combination_table:
                case None:
                    return _sample(ring, key, stream_id, plan.base_sampler_table, encoder)
                case Buffer():
                    return _large_discrete_gaussian(
                        ring,
                        key,
                        stream_id,
                        plan.base_sampler_table,
                        plan.sample_combination_table,
                        encoder,
                    )

        case _:
            raise ValueError(f"Unsupported distribution: {dist!r}")


def _chacha_blocks(key: bytes, stream_id: int, start: int, count: int) -> np.ndarray:
    if len(key) != 32:
        raise ValueError("key must be 32 bytes")
    if not 0 <= stream_id < 1 << 64:
        raise ValueError("stream_id must fit in 64 bits")
    if start < 0 or count < 0:
        raise ValueError("ChaCha20 block range must be nonnegative")
    if start + count > _CHACHA_BLOCK_LIMIT:
        raise OverflowError("ChaCha20 stream exhausted")
    if count == 0:
        return np.empty((0, 16), dtype=np.uint32)

    iv = start.to_bytes(4, "little") + stream_id.to_bytes(8, "little") + b"\x00" * 4
    encryptor = Cipher(algorithms.ChaCha20(key, iv), mode=None).encryptor()
    output = encryptor.update(b"\x00" * (64 * count))
    return np.frombuffer(output, dtype="<u4").reshape(count, 16)


def _sparse_ternary(key: bytes, stream_id: int, n: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    locations: list[int] = []
    signs: list[int] = []
    seen: set[int] = set()
    block_offset = 0
    while len(locations) < h:
        words = _chacha_blocks(key, stream_id, block_offset, 1)[0]
        block_offset += 1
        for word in words:
            location = int(word) & (n - 1)
            if location in seen:
                continue
            seen.add(location)
            locations.append(location)
            signs.append(-1 if int(word) >> 31 else 1)
            if len(locations) == h:
                break
    return np.asarray(locations, dtype=np.int64), np.asarray(signs, dtype=np.int64)


def _uniform_samples(
    key: bytes,
    stream_id: int,
    *,
    start: int,
    count: int,
    stride: int,
    modulus: int,
) -> np.ndarray:
    if not 0 < modulus < 1 << 64:
        raise ValueError("uniform modulus must fit in 64 bits")

    blocks = _chacha_blocks(key, stream_id, start, count)
    values, accepted = _reduce_uniform_blocks(blocks, modulus)
    pending = np.flatnonzero(~accepted)
    attempt = 1
    while pending.size:
        rejected: list[int] = []
        for sample_idx in pending.tolist():
            block_offset = start + sample_idx + attempt * stride
            retry_values, retry_accepted = _reduce_uniform_blocks(_chacha_blocks(key, stream_id, block_offset, 1), modulus)
            if retry_accepted[0]:
                values[sample_idx] = retry_values[0]
            else:
                rejected.append(sample_idx)
        pending = np.asarray(rejected, dtype=np.int64)
        attempt += 1
    return values


def _reduce_uniform_blocks(blocks: np.ndarray, modulus: int) -> tuple[np.ndarray, np.ndarray]:
    limbs = np.ascontiguousarray(blocks[:, :4]).view("<u8").reshape(-1, 2)
    low = limbs[:, 0]
    high = limbs[:, 1]
    values = high.astype(object) * (1 << 64) + low.astype(object)
    remainders = values % modulus
    accepted = values - remainders + modulus <= 1 << 128
    return np.asarray(remainders, dtype=np.uint64), accepted


def _sample(ring: RingSpec, key: bytes, stream_id: int, buf: Buffer, encoder: TreeNodeEncoder) -> np.ndarray:
    tree = buf.as_buffer()
    blocks = _chacha_blocks(key, stream_id, 0, ring.n)
    values = np.fromiter((_sample_one(tree, words, encoder) for words in blocks), dtype=np.int32, count=ring.n)
    min_value = values.min()
    max_value = values.max()
    support = np.arange(min_value, max_value + 1, dtype=np.int64)
    return _lookup_mont_rns_coeffs(values - min_value, support, ring)


def _sample_one(tree: memoryview, words: np.ndarray, encoder: TreeNodeEncoder, start: int = 0) -> int:
    tree_pos = 0
    word_pos = start
    while True:
        node = encoder.unpack_node(tree, tree_pos)
        bits = int(words[word_pos])
        branch = int(bits > node.split_point) + int(bits >= node.split_point)
        child = int(node.children[branch])
        if child < 0:
            tree_pos += ~child
            word_pos += branch == 1
            continue
        if word_pos + 1 >= len(words):
            raise OverflowError("discrete Gaussian sample exhausted its ChaCha20 block")
        return -child if int(words[word_pos + 1]) & 1 else child


def _large_discrete_gaussian(
    ring: RingSpec,
    key: bytes,
    stream_id: int,
    tree_buf: Buffer,
    z_buf: Buffer,
    encoder: TreeNodeEncoder,
) -> np.ndarray:
    tree = tree_buf.as_buffer()
    z_values = np.frombuffer(z_buf.as_buffer(), dtype=np.uint64)
    num_lanes = 1 << (z_buf.size - 1)
    blocks = _chacha_blocks(key, stream_id, 0, ring.n * num_lanes)
    base_samples = np.empty((len(blocks), 2), dtype=np.int32)
    for block_idx, words in enumerate(blocks):
        base_samples[block_idx, 0] = _sample_one(tree, words, encoder)
        base_samples[block_idx, 1] = _sample_one(tree, words, encoder, 8)
    base_samples = base_samples.reshape(ring.n, 2 * num_lanes)

    coeff_dtype = to_np_dtype(ring.dtype)
    rns_coeffs = np.empty(ring.n * len(ring.moduli), dtype=coeff_dtype)
    lane_indices = np.arange(num_lanes)
    for ring_idx, modulus in enumerate(ring.moduli):
        ctx = mont.context(ring.dtype, modulus)
        z = int(z_values[-1])
        folded = ctx.add(
            ctx.mul(ctx.encode(base_samples[:, 0::2]), ctx.encode(z)),
            ctx.mul(ctx.encode(base_samples[:, 1::2]), ctx.encode(z - 1)),
        )
        for z_idx in range(z_buf.size - 2, -1, -1):
            other = folded[:, lane_indices ^ (1 << z_idx)]
            sub = np.where((lane_indices >> z_idx) & 1, folded, other)
            folded = ctx.sub(ctx.mul(ctx.add(folded, other), ctx.encode(int(z_values[z_idx]))), sub)
        start = ring_idx * ring.n
        rns_coeffs[start : start + ring.n] = folded[:, 0]
    return rns_coeffs


def _lookup_mont_rns_coeffs(indices: np.ndarray, support: np.ndarray, ring: RingSpec) -> np.ndarray:
    n = ring.n
    coeff_dtype = to_np_dtype(ring.dtype)
    support = np.asarray(support, dtype=np.int64)
    rns_coeffs = np.empty(n * len(ring.moduli), dtype=coeff_dtype)
    for i, modulus in enumerate(ring.moduli):
        start = i * n
        encoded_support = np.asarray(mont.context(ring.dtype, modulus).encode(support), dtype=coeff_dtype)
        rns_coeffs[start : start + n] = encoded_support[indices]
    return rns_coeffs
