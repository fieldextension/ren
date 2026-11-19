import random
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
from flint import acb, arb

import ren
from ren.bigfloat import BigFloat
from ren.ring import Domain
from ren.ckks import CKKSContext, CKKSParams, decode, decrypt, encode, encrypt, keygen
from ren.ckks.encoding import flint_precision
from ren.ckks.matrix import LinearTransform, SparseMatrix
from tests.utils import random_values_with_norm
from tests.ckks.params import get_test_params


params = get_test_params()

MATRIX_SIZES = [1, 4, 9, 16]


def random_matrix(n: int, *, norm: float) -> np.ndarray:
    return np.array([random_values_with_norm(n, norm=norm) for _ in range(n)], dtype=complex)


def random_diagonals(n: int, offsets: list[int]) -> SparseMatrix:
    return SparseMatrix.from_diagonals(n, {d: np.asarray(random_values_with_norm(n, norm=10), dtype=np.complex128) for d in offsets})


def context_for(params: CKKSParams, *matrices: SparseMatrix, hoist: bool = False) -> CKKSContext:
    """A context holding exactly the rotation keys the transforms of ``matrices`` require."""
    rotations: dict[int, set[int]] = {}
    for matrix in matrices:
        for offset, amounts in LinearTransform.rotation_amounts(matrix, hoist=hoist).items():
            rotations.setdefault(params.default_level + offset, set()).update(amounts)
    return keygen(params, rotations=rotations).context()


@pytest.mark.parametrize("n", MATRIX_SIZES)
def test_from_dense_toarray(n: int):
    original = random_matrix(n, norm=100)
    assert SparseMatrix.from_dense(original).toarray() == pytest.approx(original, abs=1e-10)


def test_from_diagonals_toarray():
    sparse = SparseMatrix.from_diagonals(4, {0: [1, 2, 3, 4], 1: [5, 6, 7, 8], 3: [9, 10, 11, 12]})

    expected = np.array(
        [
            [1, 5, 0, 9],
            [10, 2, 6, 0],
            [0, 11, 3, 7],
            [8, 0, 12, 4],
        ],
        dtype=complex,
    )

    assert sparse.toarray() == pytest.approx(expected, abs=1e-10)


def test_constructor_sorts_offsets_and_sums_those_that_agree_mod_n():
    matrix = SparseMatrix(4, [3, 1, -1], [[1, 1, 1, 1], [2, 2, 2, 2], [10, 10, 10, 10]])

    assert matrix.offsets.tolist() == [1, 3]
    assert matrix.data.tolist() == [[2, 2, 2, 2], [11, 11, 11, 11]]


def test_constructor_owns_immutable_data():
    data = np.ones((1, 4), dtype=np.complex128)
    matrix = SparseMatrix(4, [0], data)

    data[0, 0] = 2
    assert matrix.data[0, 0] == 1
    with pytest.raises(ValueError, match="read-only"):
        matrix.data[0, 0] = 2
    with pytest.raises(ValueError, match="read-only"):
        matrix.offsets[0] = 2


def test_constructor_rejects_mismatched_offsets_and_data():
    with pytest.raises(ValueError, match="one row of data per offset"):
        SparseMatrix(4, [0, 1], [[1, 1, 1, 1]])
    with pytest.raises(ValueError, match="diagonal must have shape"):
        SparseMatrix(4, [0], [[1, 1, 1]])


def test_from_entries_places_values_and_sums_repeats():
    matrix = SparseMatrix.from_entries(4, rows=[0, 0, 3, 3], cols=[1, 1, 0, 2], values=[1.0, 2.0, 5.0, 7j])

    expected = np.zeros((4, 4), dtype=np.complex128)
    expected[0, 1] = 3.0
    expected[3, 0] = 5.0
    expected[3, 2] = 7j
    assert np.array_equal(matrix.toarray(), expected)
    assert matrix.offsets.tolist() == [1, 3]


def test_from_entries_of_nothing_is_the_zero_matrix():
    matrix = SparseMatrix.from_entries(4, rows=[], cols=[], values=[])

    assert matrix.offsets.tolist() == []
    assert np.array_equal(matrix.toarray(), np.zeros((4, 4)))


def test_from_entries_rejects_positions_outside_the_matrix():
    with pytest.raises(ValueError, match="outside"):
        SparseMatrix.from_entries(4, rows=[0], cols=[4], values=[1.0])
    with pytest.raises(ValueError, match="same length"):
        SparseMatrix.from_entries(4, rows=[0, 1], cols=[0], values=[1.0])


@pytest.mark.parametrize("n", MATRIX_SIZES)
def test_add(n: int):
    a = random_matrix(n, norm=100)
    b = random_matrix(n, norm=100)

    result = (SparseMatrix.from_dense(a) + SparseMatrix.from_dense(b)).toarray()

    assert result == pytest.approx(a + b, abs=1e-10)


@pytest.mark.parametrize("n", MATRIX_SIZES)
def test_sub(n: int):
    a = random_matrix(n, norm=100)
    b = random_matrix(n, norm=100)

    result = (SparseMatrix.from_dense(a) - SparseMatrix.from_dense(b)).toarray()

    assert result == pytest.approx(a - b, abs=1e-10)


@pytest.mark.parametrize("n", MATRIX_SIZES)
def test_scalar_mul(n: int):
    a = random_matrix(n, norm=100)
    scalar = complex(random.uniform(-10, 10), random.uniform(-10, 10))

    assert (scalar * SparseMatrix.from_dense(a)).toarray() == pytest.approx(scalar * a, abs=1e-10)
    assert (SparseMatrix.from_dense(a) * scalar).toarray() == pytest.approx(scalar * a, abs=1e-10)


@pytest.mark.parametrize("n", MATRIX_SIZES)
def test_matmul(n: int):
    a = random_matrix(n, norm=10)
    b = random_matrix(n, norm=10)

    result = (SparseMatrix.from_dense(a) @ SparseMatrix.from_dense(b)).toarray()

    assert result == pytest.approx(a @ b, abs=1e-10)


def test_matmul_with_the_zero_matrix_is_zero():
    zero, identity = SparseMatrix.zero(4), SparseMatrix.identity(4)

    for product in (zero @ identity, identity @ zero, zero @ zero):
        assert product.offsets.tolist() == []
        assert np.array_equal(product.toarray(), np.zeros((4, 4)))


def test_rotation_amounts_default_to_base_keys_and_hoist_to_direct_keys():
    # Offsets 0..15: babies {0..3}, giants {0, 4, 8, 12}.
    matrix = SparseMatrix(16, range(16), np.ones((16, 16)))

    # Default: gcd base key for the babies, and -- after the rescale -- the
    # Horner step key for the giants.
    assert LinearTransform.rotation_amounts(matrix) == {0: {1}, -1: {4}}
    # Hoisted: a direct key per nonzero baby and giant step, all at the input level.
    assert LinearTransform.rotation_amounts(matrix, hoist=True) == {0: {1, 2, 3, 4, 8, 12}, -1: set()}


def test_rotation_amounts_of_a_single_diagonal_need_one_giant_key():
    """A lone diagonal rotates as a giant step: after the rescale, one level lower."""
    matrix = SparseMatrix.from_diagonals(16, {4: np.ones(16)})

    assert LinearTransform.rotation_amounts(matrix) == {0: set(), -1: {4}}
    assert LinearTransform.rotation_amounts(matrix, hoist=True) == {0: {4}, -1: set()}


OFFSETS = [
    pytest.param(lambda n: list(range(9)), id="dense"),  # babies {0, 1, 2}, giants {0, 3, 6}
    pytest.param(lambda n: random.sample(range(n), 10), id="scattered"),
    pytest.param(lambda n: [(7 * k) % n for k in range(-12, 13)], id="centered-stride"),
    pytest.param(lambda n: [(n // 16) * k for k in range(16)], id="full-circle"),
    pytest.param(lambda n: [(n // 16) * k for k in range(-1, 14)], id="wrapped"),
    pytest.param(lambda n: [5], id="single-giant"),
]
# Composed transforms hold base keys; with a direct key per step they are double-hoisted.
HOIST = [pytest.param(False, id="composed"), pytest.param(True, id="double-hoisted")]


@pytest.mark.parametrize("hoist", HOIST)
@pytest.mark.parametrize("offsets", OFFSETS)
def test_transform_by_ciphertext(test_params: CKKSParams, offsets, hoist):
    """Test multiplication of a sparse matrix by a ciphertext vector."""
    n = test_params.slot_count
    matrix = random_diagonals(n, offsets(n))

    ctx = context_for(test_params, matrix, hoist=hoist)
    with replace(ctx, track_plaintext_values=True):
        vector = random_values_with_norm(n, norm=10)
        ct = encrypt(encode(vector, msg_bound=10))
        transform = LinearTransform.encode(matrix, level=ct.level)
        # A double-hoisted transform's plaintexts carry the key-switching primes too.
        extra = len(ctx.params.ks_primes) if hoist else 0
        assert all(len(poly.ring.moduli) == len(ct.ring.moduli) + extra for poly in transform.polys())

    expected = matrix.toarray() @ np.array(vector, dtype=complex)
    for tracking in (False, True):
        with replace(ctx, track_plaintext_values=tracking):
            result_ct = transform(ct)
            assert (result_ct._values is not None) == tracking
            if tracking:
                assert result_ct.values == pytest.approx(expected)
            assert result_ct.level == ctx.default_level - 1
            result = np.array(decode(decrypt(result_ct), n), dtype=complex)
            assert result == pytest.approx(expected, abs=1e-4)


def test_encode_keeps_each_diagonal_on_its_giant_step(test_params: CKKSParams):
    """With t = n/16, offsets {-t, 0, ..., 13t} split as babies {0, t, 2t, 3t}
    by giants {-4t, 0, ..., 12t}: -t is 3t + (-4t), although 3t + 12t also
    reaches it mod n."""
    n = test_params.slot_count
    t = n // 16
    matrix = random_diagonals(n, [t * m for m in range(-1, 14)])

    with context_for(test_params, matrix) as ctx:
        transform = LinearTransform.encode(matrix, level=ctx.default_level)

        assert (transform.groups[0].giant_step, transform.groups[0].baby_steps) == (-4 * t, (3 * t,))


@pytest.mark.parametrize("hoist", HOIST)
def test_transform_weights_are_jit_arguments(test_params: CKKSParams, hoist):
    """One captured plan serves matrices with the same offsets: replaying with
    another transform swaps its plaintexts in rather than baking the first in."""
    n = test_params.slot_count
    offsets = [(n // 16) * k for k in range(16)]
    matrices = [random_diagonals(n, offsets) for _ in range(2)]

    with context_for(test_params, *matrices, hoist=hoist):
        vector = random_values_with_norm(n, norm=10)
        ct = encrypt(encode(vector, msg_bound=10))
        transforms = [LinearTransform.encode(matrix, level=ct.level) for matrix in matrices]
        captures = 0

        @ren.jit
        def apply(ct, transform):
            nonlocal captures
            captures += 1
            return transform(ct)

        for i in (0, 1, 0):
            result = np.array(decode(decrypt(apply(ct, transforms[i])), n), dtype=complex)
            assert result == pytest.approx(matrices[i].toarray() @ np.array(vector, dtype=complex), abs=1e-4)
        assert captures == 1


def test_encode_precomputes_ntt_diagonals(test_params: CKKSParams):
    matrix = SparseMatrix.identity(test_params.slot_count)
    with context_for(test_params, matrix) as ctx:
        polys = LinearTransform.encode(matrix, level=ctx.default_level).polys()

        assert polys
        assert all(poly.domain == Domain.NTT for poly in polys)
        assert all(poly.ring == ctx.params.ring_at(ctx.default_level) for poly in polys)


@pytest.mark.parametrize("n", [params.slot_count // 2, params.slot_count * 2])
def test_encode_rejects_a_matrix_with_the_wrong_dimension(test_params: CKKSParams, n):
    matrix = SparseMatrix.from_diagonals(n, {1: np.ones(n)})
    with keygen(test_params).context() as ctx:
        with pytest.raises(ValueError, match="matrix dimension.*slot count"):
            LinearTransform.encode(matrix, level=ctx.default_level)


def test_encode_selects_hoisting_from_direct_keys_at_the_input_level(test_params: CKKSParams):
    n = test_params.slot_count
    matrix = SparseMatrix.from_diagonals(n, {i: np.ones(n) for i in range(9)})
    level = test_params.default_level
    with keygen(test_params, rotations={level: {1}, level - 1: {1, 2, 3, 6}}).context():
        for input_level in (level, level - 1):
            ct = encrypt(encode([0.125] * n, msg_bound=1, level=input_level))
            transform = LinearTransform.encode(matrix, level=input_level)
            extra = len(test_params.ks_primes) if input_level == level - 1 else 0
            assert all(len(poly.ring.moduli) == len(ct.ring.moduli) + extra for poly in transform.polys())
            assert decode(decrypt(transform(ct))) == pytest.approx([1.125] * n, abs=1e-4)


def test_transform_rejects_a_ciphertext_at_another_level(test_params: CKKSParams):
    matrix = SparseMatrix.identity(test_params.slot_count)
    with context_for(test_params, matrix) as ctx:
        transform = LinearTransform.encode(matrix, level=ctx.default_level - 1)

        with pytest.raises(ValueError, match="encoded for level"):
            transform(encrypt(encode(1, msg_bound=1)))


def test_transform_requires_its_rotation_keys(test_params: CKKSParams):
    n = test_params.slot_count
    matrix = SparseMatrix.from_diagonals(n, {4: np.ones(n, dtype=np.complex128)})

    with keygen(test_params, rotations={2}).context():
        ct = encrypt(encode([complex(i, 0) for i in range(n)], msg_bound=float(n)))

        with pytest.raises(RuntimeError, match="exceeded max_ks"):
            LinearTransform.encode(matrix, level=ct.level)(ct)


def test_double_hoisted_transform_shares_the_decomposition_and_the_moddown(test_params: CKKSParams, monkeypatch):
    """Babies {0, 1, 2} by giants {0, 3, 6}: one decompose+mod-raise serves both
    baby rotations, each nonzero giant adds one, and no ciphertext is reduced
    from the extended ring except the final sum."""
    import ren.ckks.matrix as matrix_module
    from ren.ckks._rlwe import RLWECiphertext

    n = test_params.slot_count
    matrix = random_diagonals(n, list(range(9)))

    decompositions: list[int] = []
    original_decompose = matrix_module._decompose_and_modraise

    def counting_decompose(c, key, **kwargs):
        decompositions.append(len(c.ring.moduli))
        return original_decompose(c, key, **kwargs)

    reductions: list[int] = []
    original_rescale = RLWECiphertext._rescale

    def counting_rescale(self, target_moduli):
        reductions.append(len(self.ring.moduli) - len(target_moduli))
        return original_rescale(self, target_moduli)

    monkeypatch.setattr(matrix_module, "_decompose_and_modraise", counting_decompose)
    monkeypatch.setattr(RLWECiphertext, "_rescale", counting_rescale)

    with context_for(test_params, matrix, hoist=True) as ctx:
        ct = encrypt(encode(random_values_with_norm(n, norm=10), msg_bound=10))
        LinearTransform.encode(matrix, level=ct.level)(ct)

        level_primes = len(ct.ring.moduli) - len(ctx.params.ring_at(ct.level - 1).moduli)
        # One shared baby decomposition, one per nonzero giant (3 and 6).
        assert len(decompositions) == 3
        # Exactly one ciphertext reduction: the final sum drops P and its
        # level's primes in one step.
        assert reductions == [level_primes + len(ctx.params.ks_primes)]


def test_double_hoisted_transform_accepts_reordered_key_switching_primes(test_params: CKKSParams):
    n = test_params.slot_count
    matrix = SparseMatrix.from_diagonals(n, {i: np.ones(n, dtype=np.complex128) for i in range(9)})
    keys = context_for(test_params, matrix, hoist=True).keyset
    reordered = replace(keys, params=replace(test_params, ks_primes=tuple(reversed(test_params.ks_primes))))

    with reordered.context():
        ct = encrypt(encode([0.125] * n, msg_bound=1))
        transform = LinearTransform.encode(matrix, level=ct.level)
        result = decode(decrypt(transform(ct)))

    assert result == pytest.approx([1.125] * n, abs=1e-4)


def test_double_hoisting_does_not_cost_precision(test_params: CKKSParams):
    n = test_params.slot_count
    matrix = random_diagonals(n, [(7 * k) % n for k in range(-12, 13)])
    vector = random_values_with_norm(n, norm=10)
    expected = matrix.toarray() @ np.array(vector, dtype=complex)

    errors = {}
    for hoist in (False, True):
        with context_for(test_params, matrix, hoist=hoist):
            ct = encrypt(encode(vector, msg_bound=10))
            result = np.array(decode(decrypt(LinearTransform.encode(matrix, level=ct.level)(ct)), n), dtype=complex)
            errors[hoist] = float(np.sqrt(np.mean(np.abs(result - expected) ** 2)))

    assert errors[True] <= 2 * errors[False]


@pytest.mark.parametrize("hoist", HOIST)
def test_transform_tracking_preserves_precision_and_requires_all_inputs(test_params: CKKSParams, hoist):
    n = test_params.slot_count
    matrix = SparseMatrix.from_diagonals(n, {i: np.ones(n) for i in range(9)})
    ctx = replace(context_for(test_params, matrix, hoist=hoist), encoding_precision_bits=180)
    with ctx:
        untracked_transform = LinearTransform.encode(matrix, level=ctx.default_level)

    with replace(ctx, track_plaintext_values=True), flint_precision(180):
        correction = arb(2) ** -120
        ct = encrypt(encode([acb(1 + correction), -1], msg_bound=2))
        transform = LinearTransform.encode(matrix, level=ct.level)
        with flint_precision(53):
            result = transform(ct)
        assert result._values is not None
        assert abs(result._values[0] - correction).abs_upper() < arb(2) ** -160

        for plan, operand in ((transform, ct.replace(_values=None)), (untracked_transform, ct)):
            untracked = plan(operand)
            assert untracked._values is None
            assert untracked.a.equals(result.a)
            assert untracked.b.equals(result.b)


def test_zero_matrix_transform_returns_zero_ciphertext(test_params: CKKSParams):
    n = test_params.slot_count
    with keygen(test_params).context():
        ct = encrypt(encode(random_values_with_norm(n, norm=10), msg_bound=10))
        result_ct = LinearTransform.encode(SparseMatrix.zero(n), level=ct.level)(ct)
        result = np.array(decode(decrypt(result_ct), n), dtype=complex)
        assert result == pytest.approx(np.zeros(n, dtype=complex), abs=1e-4)


@pytest.mark.parametrize("scale", [BigFloat("inf"), BigFloat("1e400"), BigFloat("1e-400"), Decimal("1e-400"), 10**400])
def test_scaling_by_an_unrepresentable_scalar_raises(scale):
    with pytest.raises(ValueError, match="representable as a float"):
        _ = SparseMatrix.identity(4) * scale
