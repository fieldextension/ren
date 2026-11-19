from dataclasses import dataclass
import builtins
from typing import (
    ClassVar,
    Final,
    Generic,
    Literal,
    Union,
    TypeIs,
    TypeVar,
)

from ren.utils import CacheMeta


BoolFmt = Literal["?"]
IntFmt = Literal["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"]
FloatFmt = Literal["e", "f", "d"]
NumericFmt = Union[IntFmt, FloatFmt]
FmtStr = Union[BoolFmt, NumericFmt]

F = TypeVar("F", bound=FmtStr | None, covariant=True)


@dataclass(frozen=True, eq=False)
class DType(Generic[F], metaclass=CacheMeta):
    itemsize: int
    name: str
    fmt: F

    @staticmethod
    def __cache_key__(itemsize: int, name: str, fmt: FmtStr | None = None) -> tuple[int, str, FmtStr | None]:
        return (itemsize, name, fmt)

    def __reduce__(self):
        # Reconstruct by *calling* the class so unpickling re-enters CacheMeta and
        # returns the canonical interned instance. The default pickle path uses
        # ``__new__`` + state restore, which bypasses the cache and yields a stray
        # copy that fails identity-based checks (e.g. ``dtype in dtypes.uints``).
        return DType, (self.itemsize, self.name, self.fmt)

    def const(self, value: int) -> int:
        """Apply this dtype's integer conversion to a constant."""
        if self in {dtypes.index, dtypes.weakint}:
            return value
        if self is dtypes.bool:
            return int(bool(value))
        modulus = 1 << (self.itemsize * 8)
        value %= modulus
        if self in {dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64} and value >= modulus // 2:
            return value - modulus
        return value

    def __tree_reduce__(self):
        return self.__reduce__()

    @staticmethod
    def is_int_fmt(fmt: FmtStr | None) -> TypeIs[IntFmt]:
        return fmt in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")

    @staticmethod
    def is_float_fmt(fmt: FmtStr | None) -> TypeIs[FloatFmt]:
        return fmt in ("e", "f", "d")


class dtypes:
    # Scalar/sequence inference adapted from tinygrad's from_py.
    @staticmethod
    def from_py(x) -> DType:
        if x.__class__ is float:
            return dtypes.default_float
        if x.__class__ is int:
            return dtypes.default_int
        if x.__class__ is bool:
            return dtypes.bool
        if x.__class__ is list or x.__class__ is tuple:
            if not x:
                return dtypes.default_float

            dtype = dtypes.from_py(x[0])
            if any(dtypes.from_py(xi) != dtype for xi in x[1:]):
                raise RuntimeError(f"Different dtypes in sequence {x}")
            return dtype
        raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")

    void: Final = DType(0, "void", None)
    index: Final = DType(100, "index", None)
    weakint: Final = DType(0, "weakint", None)
    bool: Final = DType(1, "bool", "?")

    uint8: Final = DType(1, "unsigned char", "B")
    uint32: Final = DType(4, "unsigned int", "I")
    int8: Final = DType(1, "signed char", "b")
    int16: Final = DType(2, "short", "h")
    int32: Final = DType(4, "int", "i")
    int64: Final = DType(8, "long long", "q")
    uint64: Final = DType(8, "unsigned long long", "Q")
    # Double-word scalar arithmetic; lowered to uint64 pairs when the target requires it.
    uint128: Final = DType(16, "uint128", None)
    float32: Final = DType(4, "float", "f")
    float64: Final = DType(8, "double", "d")

    # Default scalar types used throughout the library.
    # Ring-backed polynomial storage is controlled by RingSpec.dtype.
    default_float: ClassVar[DType] = float32
    default_int: ClassVar[DType] = uint64

    # Use sets?
    ints: ClassVar[tuple[DType, ...]] = (int32, uint32, int64, uint64)
    uints: ClassVar[tuple[DType, ...]] = (uint32, uint64)
    floats: ClassVar[tuple[DType, ...]] = (float32, float64)

    @staticmethod
    def is_bool(dt: DType) -> builtins.bool:
        return dt is dtypes.bool

    @staticmethod
    def is_index(dt: DType) -> builtins.bool:
        return dt is dtypes.index

    @staticmethod
    def is_int(dt: DType) -> builtins.bool:
        return dt in dtypes.ints

    @staticmethod
    def is_unsigned(dt: DType) -> builtins.bool:
        return dt in dtypes.uints

    @staticmethod
    def is_float(dt: DType) -> builtins.bool:
        return dt in dtypes.floats


def to_np_dtype(dtype: DType):
    import numpy as np

    if dtype is dtypes.index:
        return np.dtype(np.int64)
    if dtype.fmt is None:
        raise TypeError(f"unsupported numpy dtype conversion for {dtype}")
    return np.dtype(dtype.fmt)


def dtype_word_bits(dtype: DType) -> int:
    return dtype.itemsize * 8


_NP_NAME_TO_DTYPE = {
    to_np_dtype(dtype).name: dtype
    for dtype in (
        dtypes.bool,
        dtypes.uint8,
        dtypes.uint32,
        dtypes.int32,
        dtypes.int64,
        dtypes.uint64,
        dtypes.float32,
        dtypes.float64,
    )
}


def from_np_dtype(npdtype) -> DType:
    import numpy as np

    try:
        return _NP_NAME_TO_DTYPE[np.dtype(npdtype).name]
    except KeyError as exc:
        raise TypeError(f"unsupported Ren dtype for numpy dtype {npdtype!r}") from exc
