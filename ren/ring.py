from dataclasses import dataclass, field
from enum import IntEnum, auto
from math import gcd, prod

from ren.dtype import DType, dtypes
from ren.utils import CacheMeta


def _dtype_tag(dtype: DType) -> str:
    return f"u{dtype.itemsize * 8}"


def _max_modulus_for_dtype(dtype: DType) -> int:
    if dtype is dtypes.uint32:
        return 1 << 31
    if dtype is dtypes.uint64:
        return 1 << 63
    raise TypeError(f"unsupported ring dtype: {dtype}")


@dataclass(slots=True, frozen=True, eq=False, weakref_slot=True)
class RingSpec(metaclass=CacheMeta):
    n: int
    moduli: tuple[int, ...]
    dtype: DType = dtypes.uint64
    q: int = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if self.n <= 0 or (self.n & (self.n - 1)) != 0:
            raise ValueError(f"RingSpec.n must be a positive power of two, got n={self.n}")
        if self.dtype not in dtypes.uints:
            raise ValueError(f"RingSpec.dtype must be an unsigned integer dtype, got {self.dtype}")
        if not self.moduli:
            raise ValueError("moduli is empty")
        if not all(m > 1 for m in self.moduli):
            raise ValueError("All moduli must be positive integers greater than one")
        max_modulus = _max_modulus_for_dtype(self.dtype)
        if not all(m < max_modulus for m in self.moduli):
            raise ValueError(f"All moduli must be less than {max_modulus} for dtype={_dtype_tag(self.dtype)}")
        for i in range(len(self.moduli)):
            for j in range(i + 1, len(self.moduli)):
                if gcd(self.moduli[i], self.moduli[j]) != 1:
                    raise ValueError("All moduli must be pairwise coprime (gcd == 1)")
                    # in fact all the moduli should be prime, but checking this is expensive
        for m in self.moduli:
            if m % (2 * self.n) != 1:
                raise ValueError("Each modulus must be congruent to 1 mod 2n")
        if min(self.moduli) ** 2 <= max(self.moduli):
            raise ValueError("The largest modulus must be smaller than the square of the smallest modulus")
        object.__setattr__(self, "q", prod(self.moduli))

    @staticmethod
    def __cache_key__(n: int, moduli: tuple[int, ...], dtype: DType = dtypes.uint64):
        if not isinstance(moduli, tuple):
            raise TypeError("RingSpec.moduli must be a tuple[int, ...]")
        return (n, moduli, dtype)

    def __reduce__(self):
        # Reconstruct by *calling* the class so unpickling re-enters CacheMeta and
        # returns the canonical interned instance (and so ``dtype`` re-interns via
        # DType.__reduce__). The default ``__new__``-based pickle path skips the
        # cache and the ``__post_init__`` validation, yielding a stray copy.
        return RingSpec, (self.n, self.moduli, self.dtype)

    @property
    def key(self) -> tuple[int, tuple[int, ...], DType]:
        return (self.n, self.moduli, self.dtype)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RingSpec):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __lt__(self, other) -> bool:
        if not isinstance(other, RingSpec):
            return NotImplemented
        return self.key < other.key

    @property
    def storage_size(self) -> int:
        """Element count of a flat buffer holding this ring's RNS coefficients."""
        return self.n * len(self.moduli)

    @property
    def nbytes(self) -> int:
        """Byte size of a flat buffer holding this ring's RNS coefficients."""
        return self.storage_size * self.dtype.itemsize

    def is_quotient_of(self, other: RingSpec) -> bool:
        """
        Check if this ring is a quotient of another ring, or equivalently
        whether this ring's modulus divides the other ring's modulus.
        """
        return self.dtype is other.dtype and self.n == other.n and all(modulus in other.moduli for modulus in self.moduli)

    def __getitem__(self, key: slice) -> RingSpec:
        new_moduli = self.moduli[key]
        return RingSpec(n=self.n, moduli=new_moduli, dtype=self.dtype)

    def __str__(self) -> str:
        return f"(n={self.n} q=<{self.q.bit_length()}b> moduli={len(self.moduli)} dtype={_dtype_tag(self.dtype)})"


class Domain(IntEnum):
    COEFF = auto()
    NTT = auto()


class Residue(IntEnum):
    BALANCED = auto()
    NONNEGATIVE = auto()
