import operator
from collections.abc import Callable
from decimal import Decimal
from numbers import Real
from typing import Any, Self, cast

from mpmath import mp

DEFAULT_BIGFLOAT_PRECISION_BITS = 200

# Keep Ren's precision independent from mpmath's process-wide context.
_CONTEXT = mp.clone()
_CONTEXT.prec = DEFAULT_BIGFLOAT_PRECISION_BITS
# mpmath uses three guard digits for exact decimal round-trips.
_DECIMAL_DIGITS = _CONTEXT.dps + 3
# Each mpmath context creates a distinct mpf class.
_MPF_TYPE = type(_CONTEXT.mpf(0))

__all__ = ["BigFloat", "FloatLike"]


class BigFloat:
    __slots__ = ("_value",)

    _value: Any

    def __new__(cls, value: FloatLike | str = 0) -> Self:
        if type(value) is cls:
            return cast(Self, value)
        return cls._from_mpf(_coerce(value))

    @classmethod
    def _from_mpf(cls, value: Any) -> Self:
        if not isinstance(value, _MPF_TYPE):
            raise ValueError("operation produced a non-real result")
        if _CONTEXT.isnan(value):
            raise ValueError("operation produced NaN")
        result = object.__new__(cls)
        object.__setattr__(result, "_value", value)
        result._validate()
        return result

    def _validate(self) -> None:
        """Validate invariants added by specialized numeric subclasses."""

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("BigFloat is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BigFloat is immutable")

    def sqrt(self) -> Self:
        if self._value < 0:
            raise ValueError("square root is undefined for negative BigFloat values")
        return type(self)._from_mpf(_CONTEXT.sqrt(self._value))

    def log2(self) -> Self:
        if self._value <= 0:
            raise ValueError("logarithm is undefined for non-positive BigFloat values")
        return type(self)._from_mpf(_CONTEXT.log(self._value, 2))

    def is_finite(self) -> bool:
        return not self.is_infinite()

    def is_infinite(self) -> bool:
        return bool(_CONTEXT.isinf(self._value))

    def _binop(
        self,
        operation: Callable[[Any, Any], Any],
        other: object,
        reverse: bool = False,
    ) -> Self:
        try:
            other_value = _numeric_value(other)
        except TypeError:
            return NotImplemented
        left, right = (other_value, self._value) if reverse else (self._value, other_value)
        try:
            value = operation(left, right)
        except ZeroDivisionError as exc:
            raise ZeroDivisionError("BigFloat division by zero") from exc
        return type(self)._from_mpf(value)

    def _compare(
        self,
        operation: Callable[[Any, Any], bool],
        other: object,
    ) -> bool:
        try:
            other_value = _comparison_value(other)
        except TypeError:
            return NotImplemented
        return operation(self._value, other_value)

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def as_integer_ratio(self) -> tuple[int, int]:
        numerator, denominator = self._value.as_integer_ratio()
        return int(numerator), int(denominator)

    def __round__(self, ndigits: int | None = None) -> int:
        if ndigits is not None:
            raise TypeError("BigFloat only supports integer rounding")
        return int(_CONTEXT.nint(self._value))

    def __ceil__(self) -> int:
        return int(_CONTEXT.ceil(self._value))

    def __bool__(self) -> bool:
        return bool(self._value)

    def __str__(self) -> str:
        if not self._value:
            return "0"
        if self.is_infinite():
            return "-inf" if self._value < 0 else "inf"
        return _CONTEXT.nstr(self._value, n=_DECIMAL_DIGITS)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"

    def __reduce__(self) -> tuple[type[Self], tuple[str]]:
        return type(self), (str(self),)

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        return self._compare(operator.eq, other)

    def __lt__(self, other: object) -> bool:
        return self._compare(operator.lt, other)

    def __le__(self, other: object) -> bool:
        return self._compare(operator.le, other)

    def __gt__(self, other: object) -> bool:
        return self._compare(operator.gt, other)

    def __ge__(self, other: object) -> bool:
        return self._compare(operator.ge, other)

    def __add__(self, other: object) -> Self:
        return self._binop(operator.add, other)

    def __radd__(self, other: object) -> Self:
        return self._binop(operator.add, other, reverse=True)

    def __sub__(self, other: object) -> Self:
        return self._binop(operator.sub, other)

    def __rsub__(self, other: object) -> Self:
        return self._binop(operator.sub, other, reverse=True)

    def __mul__(self, other: object) -> Self:
        return self._binop(operator.mul, other)

    def __rmul__(self, other: object) -> Self:
        return self._binop(operator.mul, other, reverse=True)

    def __truediv__(self, other: object) -> Self:
        return self._binop(operator.truediv, other)

    def __rtruediv__(self, other: object) -> Self:
        return self._binop(operator.truediv, other, reverse=True)

    def __pow__(self, other: object) -> Self:
        return self._binop(operator.pow, other)

    def __rpow__(self, other: object) -> Self:
        return self._binop(operator.pow, other, reverse=True)

    def __neg__(self) -> Self:
        return type(self)._from_mpf(-self._value)

    def __pos__(self) -> Self:
        return self

    def __abs__(self) -> Self:
        return self if self._value >= 0 else type(self)._from_mpf(abs(self._value))


Real.register(BigFloat)


type FloatLike = BigFloat | int | float | Decimal


def _coerce(value: object) -> Any:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid BigFloat input")
    if isinstance(value, BigFloat):
        return value._value
    if isinstance(value, Decimal):
        if value.is_nan():
            raise ValueError("NaN is not a valid BigFloat input")
        source: int | float | str = str(value)
    elif isinstance(value, int | float | str):
        source = value
    else:
        raise TypeError(f"expected BigFloat, int, float, Decimal, or str, got {type(value).__name__}")
    try:
        result = _CONTEXT.mpf(source)
    except (TypeError, ValueError) as exc:
        if isinstance(value, str):
            raise ValueError(f"invalid decimal string: {value!r}") from exc
        raise TypeError(f"expected BigFloat, int, float, Decimal, or str, got {type(value).__name__}") from exc
    if _CONTEXT.isnan(result):
        raise ValueError("NaN is not a valid BigFloat input")
    return result


def _numeric_value(value: object) -> Any:
    if isinstance(value, BigFloat):
        return value._value
    if isinstance(value, bool):
        raise TypeError
    if isinstance(value, int | float):
        return value
    if isinstance(value, Decimal):
        return _coerce(value)
    raise TypeError


def _comparison_value(value: object) -> Any:
    if isinstance(value, bool | Decimal):
        raise TypeError
    if isinstance(value, BigFloat):
        return value._value
    if isinstance(value, int | float):
        return value
    raise TypeError
