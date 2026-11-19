from collections.abc import Callable
import math
import random
import weakref

import numpy as np
import pytest
import ren

from ren.ckks.complexpoly import ComplexPoly, Monomial, Chebyshev, PolyBasis, _cheb_to_mono_matrix, _mono_to_cheb_matrix
from ren.ckks import CKKSKeySet, decrypt, decode, encode, encrypt, Ciphertext
from ren.utils import disable_gc


TEST_POINTS = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.3, -0.7]

TOL = 1e-10


def make_poly(coeffs: list[complex], basis: PolyBasis) -> ComplexPoly:
    """
    Create a polynomial in the given basis.
    For non-Monomial bases, converts from monomial coefficients.
    """
    return ComplexPoly(coeffs, basis=Monomial()).to_basis(basis)


def assert_polys_equal(p: ComplexPoly, q: ComplexPoly, points: list[float] = TEST_POINTS) -> None:
    """Assert two polynomials evaluate to the same values at test points."""
    for x in points:
        p_val = p.evaluate(x)
        q_val = q.evaluate(x)
        assert abs(p_val - q_val) < TOL, f"Mismatch at x={x}: {p_val} != {q_val}"


def assert_poly_evaluates_to(p: ComplexPoly, expected_fn: Callable[[float], complex], points: list[float] = TEST_POINTS) -> None:
    """Assert polynomial evaluates to expected function values at test points."""
    for x in points:
        p_val = p.evaluate(x)
        expected = expected_fn(x)
        assert abs(p_val - expected) < TOL, f"Mismatch at x={x}: {p_val} != {expected}"


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_add(basis: PolyBasis):
    p = make_poly([1, 2, 3], basis)  # 1 + 2x + 3x^2
    q = make_poly([4, 5], basis)  # 4 + 5x
    zero = make_poly([], basis)

    result = p + q
    assert_poly_evaluates_to(result, lambda x: (1 + 2 * x + 3 * x**2) + (4 + 5 * x))

    result = p + 5
    assert_poly_evaluates_to(result, lambda x: (1 + 2 * x + 3 * x**2) + 5)

    result = 5 + p
    assert_poly_evaluates_to(result, lambda x: 5 + (1 + 2 * x + 3 * x**2))

    result = p + zero
    assert_polys_equal(result, p)

    r = make_poly([4, 5, 6, 7], basis)
    assert_polys_equal(p + r, r + p)

    assert_polys_equal(5 + p, p + 5)


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_sub(basis: PolyBasis):
    p = make_poly([1, 2, 3], basis)  # 1 + 2x + 3x^2
    q = make_poly([4, 5], basis)  # 4 + 5x

    result = p - q
    assert_poly_evaluates_to(result, lambda x: (1 + 2 * x + 3 * x**2) - (4 + 5 * x))

    result = p - 5
    assert_poly_evaluates_to(result, lambda x: (1 + 2 * x + 3 * x**2) - 5)

    result = 5 - p
    assert_poly_evaluates_to(result, lambda x: 5 - (1 + 2 * x + 3 * x**2))

    result = p - p
    assert result.is_zero

    assert_polys_equal(5 - p, -(p - 5))


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_neg(basis: PolyBasis):
    p = make_poly([1, 2, 3], basis)

    result = -p
    assert_poly_evaluates_to(result, lambda x: -(1 + 2 * x + 3 * x**2))

    assert_polys_equal(-(-p), p)


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_mul(basis: PolyBasis):
    p = make_poly([1, 2], basis)  # 1 + 2x
    q = make_poly([3, 4], basis)  # 3 + 4x
    zero = make_poly([], basis)

    result = p * q
    assert_poly_evaluates_to(result, lambda x: (1 + 2 * x) * (3 + 4 * x))

    p2 = make_poly([1, 2, 3], basis)  # 1 + 2x + 3x^2
    q2 = make_poly([4, 5, 6], basis)  # 4 + 5x + 6x^2
    result = p2 * q2
    assert_poly_evaluates_to(result, lambda x: (1 + 2 * x + 3 * x**2) * (4 + 5 * x + 6 * x**2))

    result = p * zero
    assert result.is_zero

    result = zero * p
    assert result.is_zero

    assert_polys_equal(p * q, q * p)

    r = make_poly([5, 6], basis)
    assert_polys_equal((p * q) * r, p * (q * r))

    result = (2 + 3j) * p2
    assert_poly_evaluates_to(result, lambda x: (2 + 3j) * (1 + 2 * x + 3 * x**2))

    result = 0 * p2
    assert result.is_zero

    result = 1 * p2
    assert_polys_equal(result, p2)

    result = p2 * (2 + 3j)
    assert_poly_evaluates_to(result, lambda x: (1 + 2 * x + 3 * x**2) * (2 + 3j))

    result = p2 * 0
    assert result.is_zero

    result = p2 * 1
    assert_polys_equal(result, p2)


def test_change_basis():
    coeffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # Monomial to Chebyshev preserves evaluation
    p_mono = ComplexPoly(coeffs, basis=Monomial())
    p_cheb = p_mono.to_basis(Chebyshev())
    assert isinstance(p_cheb.basis, Chebyshev)
    assert_polys_equal(p_mono, p_cheb)

    # Chebyshev to Monomial preserves evaluation
    p_cheb = ComplexPoly(coeffs, basis=Chebyshev())
    p_mono = p_cheb.to_basis(Monomial())
    assert isinstance(p_mono.basis, Monomial)
    assert_polys_equal(p_cheb, p_mono)


@pytest.mark.parametrize("ncoeffs", [0, 1, 5])
def test_chebyshev_vandermonde(ncoeffs: int):
    xs = np.array([-1, 0, 1], dtype=np.float32)
    expected = np.array([[1, -1, 1, -1, 1], [1, 0, -1, 0, 1], [1, 1, 1, 1, 1]])
    matrix = Chebyshev().vandermonde(xs, ncoeffs)
    np.testing.assert_array_equal(matrix, expected[:, :ncoeffs])
    assert matrix.dtype == np.float64
    assert Chebyshev().vandermonde(xs[:0], ncoeffs).shape == (0, ncoeffs)


def test_lagrange_interpolant():
    """lagrange_interpolant should exactly reproduce degree-n polynomials through n+1 nodes."""

    def f(t):
        return t**2 - 3 * t + 1

    poly = ComplexPoly.lagrange_interpolant([(t, complex(f(t))) for t in (0.0, 1.0, 2.0)])

    for t in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        assert abs(poly.evaluate(t) - f(t)) < TOL, f"Mismatch at t={t}"


def test_chebyshev_interpolant():
    """chebyshev_interpolant with degree 63 should approximate sgn(t) on [-1, 1].
    sgn(t) = -1 for t < 0, +1 for t > 0.
    We only check points outside [-0.25, 0.25] to avoid the Gibbs phenomenon near the discontinuity.
    """
    APPROX_TOL = 0.05

    def sgn(t):
        if t > 0:
            return 1.0
        elif t < 0:
            return -1.0
        else:
            return 0.0

    poly = ComplexPoly.chebyshev_interpolant(sgn, degree=63)

    # Only check points outside [-0.25, 0.25] to avoid the Gibbs phenomenon
    well_behaved_points = [-1.0, -0.9, -0.75, -0.5, -0.3, 0.3, 0.5, 0.75, 0.9, 1.0]
    print("\nChebyshev degree-63 approximation of sgn(t) on [-1, 1]:")
    for t in well_behaved_points:
        approx = poly.evaluate(t).real
        expected = sgn(t)
        error = abs(approx - expected)
        print(f"  t={t:+.2f}: approx={approx:+.6f}, expected={expected:+.1f}, error={error:.6f}")
        assert error < APPROX_TOL, f"Poor approximation at t={t}: got {approx}, expected {expected}"


def test_chebyshev_interpolant_interval():
    """chebyshev_interpolant with a custom interval should approximate f on [a, b]."""

    def f(t):
        return t**2 - 3 * t + 1

    poly = ComplexPoly.chebyshev_interpolant(f, degree=2, interval=(1.0, 3.0))

    for t in [1.0, 1.5, 2.0, 2.5, 3.0]:
        assert abs(poly.evaluate(t) - f(t)) < TOL, f"Mismatch at t={t}"


@pytest.mark.parametrize(
    "basis",
    [Monomial(), Chebyshev()],
)
def test_evaluate(basis: PolyBasis):
    # Small degree-2 polynomial: 1 + 2x + 3x^2.
    p = make_poly([1, 2, 3], basis)

    assert abs(p.evaluate(0.0) - 1) < TOL
    assert abs(p.evaluate(1.0) - 6) < TOL
    assert abs(p.evaluate(-1.0) - 2) < TOL
    assert abs(p.evaluate(0.5) - (1 + 1 + 0.75)) < TOL

    z = 0.5 + 0.5j
    expected = 1 + 2 * z + 3 * z**2
    assert abs(p.evaluate(z) - expected) < TOL

    # Larger degree-15 polynomial with coefficients 1, 2, ..., 16.
    big_coeffs: list[complex] = [complex(c) for c in range(1, 17)]
    q = make_poly(big_coeffs, basis)

    def big_f(x: complex) -> complex:
        return sum(c * x**i for i, c in enumerate(big_coeffs))

    for x in [-0.7, -0.3, 0.0, 0.4, 0.9]:
        assert abs(q.evaluate(x) - big_f(x)) < TOL, f"Mismatch at x={x}"

    z2 = 0.3 - 0.2j
    assert abs(q.evaluate(z2) - big_f(z2)) < TOL


@pytest.mark.parametrize(
    "basis",
    [Monomial(), Chebyshev()],
)
@pytest.mark.parametrize("degree", range(1, 21))
def test_evaluate_depth(keys: CKKSKeySet, degree: int, basis: PolyBasis):
    """These nonconstant polynomials consume floor(log2(degree)) + 1 levels."""
    with keys.context():
        coeffs = [6.28] * (degree + 1)
        poly = ComplexPoly(coeffs, basis=basis)
        ct = Ciphertext.zero()
        result = poly.evaluate(ct)
        assert isinstance(result, type(ct))
        assert ct.level - result.level == math.floor(math.log2(degree)) + 1
        assert poly.depth == ct.level - result.level


@pytest.mark.parametrize(
    "basis",
    [Monomial(), Chebyshev()],
)
@pytest.mark.parametrize(
    "name, f, interval, degree",
    [
        ("sigmoid", lambda x: 1 / (1 + math.exp(-x)), (-4.0, 4.0), 3),
        ("sigmoid", lambda x: 1 / (1 + math.exp(-x)), (-4.0, 4.0), 7),
        ("exp", lambda x: math.exp(x), (-1.0, 1.0), 3),
        ("exp", lambda x: math.exp(x), (-1.0, 1.0), 7),
        ("tanh", lambda x: math.tanh(x), (-2.0, 2.0), 3),
        ("tanh", lambda x: math.tanh(x), (-2.0, 2.0), 7),
    ],
)
def test_remez_approximant(name: str, f: Callable[[float], float], interval: tuple[float, float], degree: int, basis: PolyBasis):
    """Remez fits in either basis should match or beat the Chebyshev interpolant."""
    import numpy as np

    a, b = interval

    remez_poly, remez_err = ComplexPoly.remez_approximant(f, degree=degree, interval=interval, basis=basis)

    assert remez_poly.basis == basis

    cheb_poly = ComplexPoly.chebyshev_interpolant(f, degree=degree, interval=interval)
    xs = np.linspace(a, b, 10000)
    cheb_err = float(np.max(np.abs(np.array([cheb_poly.evaluate(x).real for x in xs]) - np.array([f(x) for x in xs]))))

    assert remez_err <= cheb_err * (1 + 1e-6), (
        f"Remez error ({remez_err:.6e}) worse than Chebyshev ({cheb_err:.6e}) for {name} degree {degree} on {interval} in {type(basis).__name__} basis"
    )


@pytest.mark.parametrize(
    "basis",
    [Monomial(), Chebyshev()],
)
def test_evaluate_ciphertext(keys: CKKSKeySet, basis: PolyBasis):
    """Ciphertext evaluation should produce the same per-slot values as
    evaluating the polynomial classically on each plaintext slot.
    """
    with keys.context():
        random.seed(42)
        coeffs = [random.uniform(-1, 1) + random.uniform(-1, 1) * 1j for _ in range(15)]
        poly = ComplexPoly(coeffs, basis=basis)

        slot_count = keys.params.slot_count
        values = [2 * i / (slot_count - 1) - 1 for i in range(slot_count)]
        ct = encrypt(encode(values, msg_bound=1.0))

        result_ct = poly.evaluate(ct)

        decrypted = decode(decrypt(result_ct), slot_count)
        expected = [poly.evaluate(v) for v in values]

        for i in range(slot_count):
            assert decrypted[i] == pytest.approx(expected[i], abs=1e-2), f"Mismatch at slot {i}"


def test_evaluate_releases_input_wrappers(keys: CKKSKeySet):
    with keys.context(), disable_gc():
        ct = encrypt(encode([0.2] * keys.params.slot_count, msg_bound=1.0))
        inputs = (weakref.ref(ct.a), weakref.ref(ct.b))
        poly = ComplexPoly([0.1] * 8, basis=Monomial())
        result = poly.evaluate(ct)
        del ct
        assert all(ref() is None for ref in inputs)
        assert decode(decrypt(result), keys.params.slot_count) == pytest.approx([poly.evaluate(0.2)] * keys.params.slot_count, abs=1e-2)


def test_jit_evaluate_replays_with_cached_schedules(keys: CKKSKeySet):
    with keys.context():
        values = (0.1, 0.5)
        inputs = [encrypt(encode([value] * keys.params.slot_count, msg_bound=1.0)) for value in values]

        for degree in (3, 5, 7):
            poly = ComplexPoly([0.1] * (degree + 1), basis=Chebyshev())
            captures = []

            @ren.jit
            def evaluate(c):
                captures.append(None)
                return poly.evaluate(c)

            for value, ct in zip(values, inputs):
                result = evaluate(ct)
                decoded = decode(decrypt(result), keys.params.slot_count)
                assert decoded == pytest.approx([poly.evaluate(value)] * keys.params.slot_count, abs=1e-2)
            assert len(captures) == 1


def _stack_fixture(basis: PolyBasis) -> list[ComplexPoly]:
    """Three polynomials of differing degree, sharing a basis."""
    return [
        make_poly([1, 2, 3], basis),
        make_poly([4, 5], basis),
        make_poly([-1, 0, 0, 2], basis),
    ]


@pytest.mark.parametrize(
    "basis",
    [Monomial(), Chebyshev()],
)
def test_stack_matches_per_slot_evaluation(basis: PolyBasis):
    """A stacked polynomial evaluates to one value per slot, each matching the
    polynomial that was stacked into that slot."""
    polys = _stack_fixture(basis)
    stacked = ComplexPoly.stack(polys)

    assert stacked.coeffs.shape == (stacked.degree + 1, len(polys))
    for x in TEST_POINTS:
        values = stacked.evaluate_slots(x)
        assert isinstance(values, np.ndarray)
        assert values.shape == (len(polys),)
        for i, p in enumerate(polys):
            assert values[i] == pytest.approx(p.evaluate(x), abs=TOL)


def test_stack_zero_pads_and_strips():
    """Shorter polynomials are zero-padded; an all-zero stack keeps its width."""
    polys = [ComplexPoly([1, 2, 3]), ComplexPoly([4])]
    stacked = ComplexPoly.stack(polys)
    assert stacked.degree == 2
    assert stacked.coeffs.shape == (3, 2)
    assert stacked.coeffs[1][1] == 0 and stacked.coeffs[2][1] == 0

    empty = ComplexPoly.stack([ComplexPoly.zero(), ComplexPoly.zero()])
    assert empty.is_zero
    assert empty.coeffs.shape == (0, 2)
    # The slot axis survives, so evaluation still returns one value per slot.
    assert np.array_equal(empty.evaluate_slots(0.5), np.zeros(2))


def test_stack_validation():
    with pytest.raises(ValueError):
        ComplexPoly.stack([])
    with pytest.raises(ValueError):
        ComplexPoly.stack([ComplexPoly([1, 2], basis=Monomial()), ComplexPoly([1, 2], basis=Chebyshev())])
    with pytest.raises(TypeError):
        ComplexPoly.stack([ComplexPoly.stack([ComplexPoly([1, 2])])])


def test_scalar_evaluation_requires_matching_shape():
    """``evaluate`` and ``evaluate_slots`` each reject the other's shape.

    ``evaluate(complex)`` promises a single ``complex``, so a stacked
    polynomial has to go through ``evaluate_slots``; the reverse holds for a
    plain one, which has no slot axis to report.
    """
    plain = ComplexPoly([1, 2, 3])
    stacked = ComplexPoly.stack([plain, ComplexPoly([4, 5])])

    with pytest.raises(TypeError):
        stacked.evaluate(0.5)
    with pytest.raises(TypeError):
        plain.evaluate_slots(0.5)

    # Each still works on the shape it is meant for.
    assert isinstance(plain.evaluate(0.5), complex)
    assert stacked.evaluate_slots(0.5) == pytest.approx([plain.evaluate(0.5), 4 + 5 * 0.5], abs=TOL)


def test_mul_by_per_slot_array_stacks():
    """Multiplying by a per-slot array must stack, not multiply coefficient-wise.

    The array length here deliberately equals the coefficient count: numpy's
    trailing-axis broadcasting would silently produce a plain ``(3,)`` result
    instead of a ``(3, 3)`` stacked one if the multiplier were not forced onto
    the batch axis.
    """
    poly = ComplexPoly([1, 2, 3])
    mask = np.array([1.0, 0.0, 2.0])
    assert len(mask) == poly.degree + 1

    for stacked in (poly * mask, mask * poly, poly * [1.0, 0.0, 2.0]):
        assert stacked.coeffs.shape == (3, 3)
        for x in TEST_POINTS:
            expected = poly.evaluate(x)
            assert stacked.evaluate_slots(x) == pytest.approx([expected, 0.0, 2 * expected], abs=TOL)


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_stacked_arithmetic(basis: PolyBasis):
    """Mixing plain and stacked operands broadcasts over slots, in either order."""
    polys = _stack_fixture(basis)
    stacked = ComplexPoly.stack(polys)
    # Longer than the stack, and shorter than it, to exercise both padding sides.
    for plain in (make_poly([7, -1, 2, 0, 5], basis), make_poly([2], basis)):
        for x in TEST_POINTS:
            px = plain.evaluate(x)
            assert (stacked + plain).evaluate_slots(x) == pytest.approx([p.evaluate(x) + px for p in polys], abs=TOL)
            assert (plain + stacked).evaluate_slots(x) == pytest.approx([px + p.evaluate(x) for p in polys], abs=TOL)
            assert (stacked - plain).evaluate_slots(x) == pytest.approx([p.evaluate(x) - px for p in polys], abs=TOL)
            assert (plain - stacked).evaluate_slots(x) == pytest.approx([px - p.evaluate(x) for p in polys], abs=TOL)
            assert (stacked * plain).evaluate_slots(x) == pytest.approx([p.evaluate(x) * px for p in polys], abs=TOL)

    for x in TEST_POINTS:
        assert (stacked + stacked).evaluate_slots(x) == pytest.approx([2 * p.evaluate(x) for p in polys], abs=TOL)
        assert (stacked * stacked).evaluate_slots(x) == pytest.approx([p.evaluate(x) ** 2 for p in polys], abs=TOL)
        assert (stacked + 3).evaluate_slots(x) == pytest.approx([p.evaluate(x) + 3 for p in polys], abs=TOL)
        assert (3 - stacked).evaluate_slots(x) == pytest.approx([3 - p.evaluate(x) for p in polys], abs=TOL)
        assert (-stacked).evaluate_slots(x) == pytest.approx([-p.evaluate(x) for p in polys], abs=TOL)


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_zero_multiplication_preserves_slots(basis: PolyBasis):
    plain = ComplexPoly([1, 2], basis=basis)
    stacked = plain * np.ones(3)
    zero = ComplexPoly.zero(basis=basis)
    stacked_zero = zero * np.ones(3)
    singleton_zero = zero * np.ones(1)

    for left, right in [
        (zero, stacked),
        (stacked_zero, plain),
        (stacked_zero, stacked),
        (zero, stacked_zero),
        (singleton_zero, stacked),
        (singleton_zero, stacked_zero),
    ]:
        for result in (left * right, right * left):
            assert result.basis == basis
            assert result.coeffs.shape == (0, 3)
            assert np.array_equal(result.evaluate_slots(0.5), np.zeros(3))


@pytest.mark.parametrize("coeffs", [[], [1]])
def test_zero_multiplication_requires_same_basis(coeffs: list[complex]):
    zero = ComplexPoly.zero(basis=Monomial())
    other = ComplexPoly(coeffs, basis=Chebyshev())
    for left, right in [(zero, other), (other, zero)]:
        with pytest.raises(ValueError, match="different bases"):
            left * right


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_zero_multiplication_requires_compatible_slots(basis: PolyBasis):
    zero = ComplexPoly.zero(basis=basis) * np.ones(2)
    stacked = ComplexPoly([1, 2], basis=basis) * np.ones(3)
    for other in (stacked, stacked * 0):
        for left, right in [(zero, other), (other, zero)]:
            with pytest.raises(ValueError):
                left * right


@pytest.mark.parametrize("source, target", [(Monomial(), Chebyshev()), (Chebyshev(), Monomial())])
def test_stacked_to_basis(source: PolyBasis, target: PolyBasis):
    """Converting a stacked polynomial converts every slot independently."""
    polys = [ComplexPoly([1, 2, 3, 4, 5, 6], basis=source), ComplexPoly([-2, 1, 0, 3], basis=source), ComplexPoly([0.5], basis=source)]
    stacked = ComplexPoly.stack(polys)

    converted = stacked.to_basis(target)
    expected = np.zeros_like(converted.coeffs)
    for i, p in enumerate(polys):
        column = p.to_basis(target).coeffs
        expected[: len(column), i] = column
    assert converted.coeffs == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("source, target", [(Monomial(), Chebyshev()), (Chebyshev(), Monomial())])
@pytest.mark.parametrize("shape", [(0,), (0, 1), (0, 3)])
def test_zero_to_basis_preserves_shape(source: PolyBasis, target: PolyBasis, shape: tuple[int, ...]):
    poly = ComplexPoly(np.empty(shape, dtype=np.complex128), basis=source)
    converted = poly.to_basis(target)
    assert converted.basis == target
    assert converted.coeffs.shape == shape


def test_mono_to_cheb_matrix_inverts_cheb_to_mono():
    """The two cached basis-change matrices are inverses of one another."""
    for n in (0, 1, 2, 5, 13):
        product = _mono_to_cheb_matrix(n) @ _cheb_to_mono_matrix(n).T
        assert product == pytest.approx(np.eye(n), abs=1e-10)


def test_to_basis_roundtrip_high_degree():
    """Monomial -> Chebyshev -> Monomial at a degree above test_change_basis."""
    random.seed(7)
    poly = ComplexPoly([random.uniform(-1, 1) for _ in range(31)], basis=Monomial())
    assert_polys_equal(poly, poly.to_basis(Chebyshev()).to_basis(Monomial()))


@pytest.mark.parametrize("degree", [30, 31])
def test_to_basis_roundtrip_monomial_coefficients(degree: int):
    coeffs = np.zeros(degree + 1, dtype=np.complex128)
    coeffs[-1] = 1
    for values in (coeffs, coeffs[:, None] * [1, 1j, -0.5 + 0.25j]):
        poly = ComplexPoly(values, Monomial())
        converted = poly.to_basis(Chebyshev())
        assert np.count_nonzero(converted.coeffs[1 - degree % 2 :: 2]) == 0
        restored = converted.to_basis(Monomial())
        np.testing.assert_allclose(restored.coeffs, values, rtol=0, atol=TOL)


def test_to_basis_with_underflowing_coefficients():
    # For x**1080, coefficients at the highest Chebyshev degrees underflow in float64.
    coeffs = np.zeros(1081)
    coeffs[-1] = 1
    converted = ComplexPoly(coeffs).to_basis(Chebyshev())
    assert np.all(np.isfinite(converted.coeffs))
    assert_poly_evaluates_to(converted, lambda x: x**1080, points=[-1, -0.999, 0, 0.999, 1])


def test_constructor_accepts_arrays_and_copies():
    source = np.array([1, 2, 3], dtype=np.complex128)
    poly = ComplexPoly(source)
    assert np.array_equal(poly.coeffs, ComplexPoly([1, 2, 3]).coeffs)

    # Polynomials are immutable and own their coefficients.
    source[0] = 99
    assert poly.coeffs[0] == 1
    assert poly.coeffs.flags.writeable is False

    with pytest.raises(ValueError):
        ComplexPoly(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        ComplexPoly(np.array(1.0))


def test_evaluate_stacked_ciphertext(keys: CKKSKeySet):
    """A stacked polynomial applied to a ciphertext acts per slot -- the path
    used to fuse a mask into a polynomial evaluation."""
    with keys.context():
        random.seed(11)
        slot_count = keys.params.slot_count
        base = ComplexPoly([random.uniform(-1, 1) for _ in range(8)], basis=Chebyshev())
        # Alternate between the polynomial and zero across slots.
        mask = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(slot_count)])
        poly = base * mask

        values = [2 * i / (slot_count - 1) - 1 for i in range(slot_count)]
        ct = encrypt(encode(values, msg_bound=1.0))
        decrypted = decode(decrypt(poly.evaluate(ct)), slot_count)

        for i in range(slot_count):
            expected = base.evaluate(values[i]) * mask[i]
            assert decrypted[i] == pytest.approx(expected, abs=1e-2), f"Mismatch at slot {i}"


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
def test_evaluate_stacked_constant_ciphertext(keys: CKKSKeySet, basis: PolyBasis):
    """A degree-0 stacked polynomial encodes its coefficients as a plaintext."""
    with keys.context():
        slot_count = keys.params.slot_count
        row = np.array([0.25 + 0.5 * (i % 3) for i in range(slot_count)])
        poly = ComplexPoly(row[None, :], basis=basis)
        assert poly.degree == 0

        ct = encrypt(encode([0.5] * slot_count, msg_bound=1.0))
        decrypted = decode(decrypt(poly.evaluate(ct)), slot_count)

        for i in range(slot_count):
            assert decrypted[i] == pytest.approx(row[i], abs=1e-2), f"Mismatch at slot {i}"


@pytest.mark.parametrize("lower, upper", [(6, 8), (14, 16)])
def test_addition_preserves_values_across_degree_boundary(keys: CKKSKeySet, lower: int, upper: int):
    left = ComplexPoly([0] * lower + [1], Chebyshev())
    right = ComplexPoly([0] * upper + [1], Chebyshev())
    poly = left + right
    expected_coeffs = np.zeros(upper + 1)
    expected_coeffs[[lower, upper]] = 1
    for x in TEST_POINTS:
        expected = np.polynomial.chebyshev.chebval(x, expected_coeffs)
        assert poly.evaluate(x) == pytest.approx(expected, abs=TOL)

    with keys.context():
        values = np.resize(TEST_POINTS, keys.params.slot_count)
        ct = encrypt(encode(values, msg_bound=1.0))
        decoded = decode(decrypt(poly.evaluate(ct)), keys.params.slot_count)
        assert decoded == pytest.approx(np.polynomial.chebyshev.chebval(values, expected_coeffs), abs=1e-2)
        assert np.array_equal(poly.coeffs, expected_coeffs)


@pytest.mark.parametrize("lower, upper", [(6, 8), (14, 16)])
def test_stacking_preserves_values_across_degree_boundary(keys: CKKSKeySet, lower: int, upper: int):
    left = ComplexPoly([0] * lower + [1], Chebyshev())
    right = ComplexPoly([0] * upper + [1], Chebyshev())
    stacked = ComplexPoly.stack([left, right])
    for x in TEST_POINTS:
        expected = [np.polynomial.chebyshev.chebval(x, [0] * degree + [1]) for degree in (lower, upper)]
        assert stacked.evaluate_slots(x) == pytest.approx(expected, abs=TOL)

    with keys.context():
        slot_count = keys.params.slot_count
        values = np.resize(TEST_POINTS, slot_count)
        poly = ComplexPoly.stack([left if i % 2 == 0 else right for i in range(slot_count)])
        coefficients = poly.coeffs.copy()
        ct = encrypt(encode(values, msg_bound=1.0))
        decoded = decode(decrypt(poly.evaluate(ct)), slot_count)
        expected = [np.polynomial.chebyshev.chebval(x, [0] * (lower if i % 2 == 0 else upper) + [1]) for i, x in enumerate(values)]
        assert decoded == pytest.approx(expected, abs=1e-2)
        assert np.array_equal(poly.coeffs, coefficients)


@pytest.mark.parametrize("basis", [Monomial(), Chebyshev()])
@pytest.mark.parametrize("coeffs, expected", [([], 0), ([0.25 + 0.5j], 0.25 + 0.5j)])
def test_evaluate_zero_and_constant_ciphertext(keys: CKKSKeySet, basis: PolyBasis, coeffs: list[complex], expected: complex):
    with keys.context():
        poly = ComplexPoly(coeffs, basis)
        ct = encrypt(encode([0.5] * keys.params.slot_count, msg_bound=1.0))
        result = poly.evaluate(ct)
        assert result.level == ct.level
        assert poly.depth == 0
        assert decode(decrypt(result), keys.params.slot_count) == pytest.approx([expected] * keys.params.slot_count, abs=1e-2)
