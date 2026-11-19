import math
import operator
import pickle
from decimal import Decimal
from numbers import Real

import mpmath
import pytest

from ren.bigfloat import BigFloat, FloatLike


def test_bigfloat_constructor_normalizes_supported_inputs():
    value = BigFloat("4.5")
    inputs: tuple[FloatLike, ...] = (1, 1.0, Decimal("3.5"), value)

    assert [BigFloat(value) for value in inputs] == [
        BigFloat(1),
        BigFloat(1),
        BigFloat("3.5"),
        BigFloat("4.5"),
    ]
    assert BigFloat("1.125") == BigFloat(Decimal("1.125"))
    assert BigFloat(value) is value


def test_abs_reuses_nonnegative_immutable_value():
    value = BigFloat(1)

    assert abs(value) is value


@pytest.mark.parametrize("value", [True, 1 + 2j, mpmath.mpf("1.125")])
def test_bigfloat_rejects_unsupported_numeric_inputs(value):
    with pytest.raises(TypeError):
        BigFloat(value)


@pytest.mark.parametrize("value", ["not a number", "NaN", float("nan"), Decimal("NaN")])
def test_bigfloat_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        BigFloat(value)


def test_bigfloat_is_immutable():
    value = BigFloat(1)

    with pytest.raises(AttributeError, match="immutable"):
        value._value = mpmath.mpf(2)


def test_bigfloat_arithmetic_preserves_precision_and_type():
    value = BigFloat("2.5")

    assert value + 2 == BigFloat("4.5")
    assert 2 + value == BigFloat("4.5")
    assert value - 2 == BigFloat("0.5")
    assert 3 - value == BigFloat("0.5")
    assert value * 2 == BigFloat(5)
    assert 2 * value == BigFloat(5)
    assert value / 2 == BigFloat("1.25")
    assert 5 / value == BigFloat(2)
    assert value**2 == BigFloat("6.25")
    assert 2 ** BigFloat(3) == BigFloat(8)
    assert BigFloat(0) + Decimal("0.10000000000000000001") == BigFloat("0.10000000000000000001")


def test_bigfloat_arithmetic_preserves_subclass_invariants():
    class NonNegative(BigFloat):
        def _validate(self) -> None:
            if self < 0:
                raise ValueError("value must be non-negative")

    value = NonNegative(2)

    assert isinstance(value + 1, NonNegative)
    assert isinstance(value.sqrt(), NonNegative)
    with pytest.raises(ValueError, match="non-negative"):
        _ = value - 3


def test_bigfloat_comparison_does_not_round_other_operands():
    value = BigFloat(2) ** 200

    assert value == 2**200
    assert value != 2**200 + 1
    assert value < 2**200 + 1
    assert value >= BigFloat(2) ** 200
    assert hash(value) == hash(2**200)


def test_bigfloat_is_recognized_as_a_real_number():
    value = BigFloat("1.00000000000001")

    assert isinstance(value, Real)
    assert 1.0 == pytest.approx(value, abs=1e-10)


def test_bigfloat_comparison_does_not_round_decimal_operands():
    value = BigFloat("0.1")

    assert value != Decimal("0.1")
    with pytest.raises(TypeError):
        _ = value < Decimal("0.2")


def test_bigfloat_reports_finiteness_without_float_downcast():
    huge = BigFloat(10) ** 400

    assert huge.is_finite()
    assert not huge.is_infinite()
    assert BigFloat("inf").is_infinite()
    assert not BigFloat("inf").is_finite()


def test_bigfloat_log2_preserves_large_finite_values():
    assert (BigFloat(2) ** 2000).log2() == 2000
    assert BigFloat("0.5").log2() == -1
    with pytest.raises(ValueError, match="logarithm"):
        BigFloat(0).log2()
    with pytest.raises(ValueError, match="logarithm"):
        BigFloat(-1).log2()


def test_bigfloat_conversions_and_rounding():
    value = BigFloat("2.5")

    assert float(value) == 2.5
    assert int(value) == 2
    assert value.as_integer_ratio() == (5, 2)
    assert (BigFloat(2) ** 256).as_integer_ratio() == (2**256, 1)
    assert (-(BigFloat(2) ** -256)).as_integer_ratio() == (-1, 2**256)
    assert round(value) == 2
    assert round(BigFloat("3.5")) == 4
    assert round(BigFloat("-2.5")) == -2
    assert round(BigFloat("2.49")) == 2
    assert math.ceil(BigFloat("2.01")) == 3
    assert round(BigFloat(2) ** 100 + BigFloat("0.75")) == 2**100 + 1
    with pytest.raises(TypeError, match="integer rounding"):
        round(value, 1)


def test_bigfloat_str_repr_and_pickle_round_trip_full_precision():
    value = BigFloat(2**199 + 1) / 3

    assert BigFloat(str(value)) == value
    assert repr(value) == f"BigFloat({str(value)!r})"
    assert pickle.loads(pickle.dumps(value)) == value
    assert f"{value}" == str(value)
    with pytest.raises(TypeError):
        format(value, ".4f")


def test_bigfloat_uses_fixed_binary_precision():
    assert BigFloat(2**199 + 1) != BigFloat(2**199)
    assert BigFloat(2**200 + 1) == BigFloat(2**200)


def test_bigfloat_mixed_arithmetic_rounds_only_the_result():
    boundary = 2**200

    assert BigFloat(1) + (boundary + 1) == BigFloat(boundary + 2)


def test_bigfloat_precision_is_independent_from_global_mpmath_context(monkeypatch):
    monkeypatch.setattr(mpmath.mp, "prec", 20)

    value = BigFloat(1) + BigFloat(2) ** -199

    assert value != 1
    assert BigFloat(str(value)) == value


def test_bigfloat_rejects_non_real_results():
    with pytest.raises(ValueError, match="square root"):
        BigFloat(-1).sqrt()
    with pytest.raises(ValueError, match="non-real"):
        _ = BigFloat(-1) ** 0.5
    with pytest.raises(ValueError, match="non-real"):
        _ = (-1) ** BigFloat("0.5")
    with pytest.raises(ValueError, match="NaN"):
        _ = BigFloat(0) * BigFloat("inf")


def test_bigfloat_defers_unknown_mixed_operations():
    class ReflectedAdd:
        def __radd__(self, other: object) -> str:
            assert isinstance(other, BigFloat)
            return "reflected"

    class ReflectedGreater:
        def __gt__(self, other: object) -> bool:
            assert isinstance(other, BigFloat)
            return True

    assert operator.add(BigFloat(1), ReflectedAdd()) == "reflected"
    assert operator.lt(BigFloat(1), ReflectedGreater()) is True
