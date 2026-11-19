"""Algebraic and scalar operator mixins."""

from typing import Any, Self

from ren.ir.op import Op


class OpMixin:
    """Algebraic operations shared by Node, Pat, and Poly."""

    # required to implement
    def _unary(self, op: Op, *, arg: Any = None) -> Self:
        raise NotImplementedError

    def _binop(self, op: Op, other: Any, reverse: bool = False) -> Self:
        raise NotImplementedError

    def __neg__(self) -> Self:
        return self._unary(Op.NEG)

    def __add__(self, other: Self | int) -> Self:
        return self._binop(Op.ADD, other)

    def __radd__(self, other: int) -> Self:
        return self._binop(Op.ADD, other, reverse=True)

    def __sub__(self, other: Self | int) -> Self:
        return self._binop(Op.SUB, other)

    def __rsub__(self, other: int) -> Self:
        return self._binop(Op.SUB, other, reverse=True)

    def __mul__(self, other: Self | int) -> Self:
        return self._binop(Op.MUL, other)

    def __rmul__(self, other: int) -> Self:
        return self._binop(Op.MUL, other, reverse=True)


class ScalarMixin(OpMixin):
    """Scalar operations shared by Node and Pat."""

    def __lt__(self, other: Self | int) -> Self:
        return self._binop(Op.CMPLT, other)

    def __gt__(self, other: Self | int) -> Self:
        return self._binop(Op.CMPLT, other, reverse=True)

    def __le__(self, other: Self | int) -> Self:
        return (self > other).logical_not()

    def __ge__(self, other: Self | int) -> Self:
        return (self < other).logical_not()

    # Python equality compares node identity and pattern structure.
    def eq(self, other: Self | int) -> Self:
        return self._binop(Op.CMPEQ, other)

    def ne(self, other: Self | int) -> Self:
        return self.eq(other).logical_not()

    def logical_not(self) -> Self:
        """Negate a boolean expression."""
        return self.eq(False)

    def __lshift__(self, bits: Self | int) -> Self:
        return self._binop(Op.SHL, bits)

    def __rshift__(self, bits: Self | int) -> Self:
        return self._binop(Op.SHR, bits)

    def __mod__(self, other: Self | int) -> Self:
        return self._binop(Op.MOD, other)

    def __rmod__(self, other: int) -> Self:
        return self._binop(Op.MOD, other, reverse=True)

    def __floordiv__(self, other: Self | int) -> Self:
        return self._binop(Op.IDIV, other)

    def __rfloordiv__(self, other: int) -> Self:
        return self._binop(Op.IDIV, other, reverse=True)

    def __and__(self, other: Self | int) -> Self:
        return self._binop(Op.AND, other)

    def __rand__(self, other: int) -> Self:
        return self._binop(Op.AND, other, reverse=True)
