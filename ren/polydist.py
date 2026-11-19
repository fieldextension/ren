from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction


# A value sampled from a standard Gaussian has absolute value greater than 13.109 times the standard deviation
# with probability less than 2**-128
# To replicate:
# from scipy.stats import norm
# p = 2**-128
# print(norm.isf(p / 2))
STD_DEVS_FOR_128_BIT_SECURITY = 13.109

# The generally accepted minimum standard deviation for encryption noise is 3.2 (or 3.19)
MINIMUM_ENCRYPTION_NOISE_STD_DEV = Fraction(32, 10)


@dataclass(frozen=True, slots=True)
class PolyDist(ABC):
    """
    A probability distribution that can be used to generate random polynomials.

    Any class that inherits from PolyDist should have a concrete implementation in
    the cuda and python backends.
    """

    n: int


class PolyDistWithBounds(PolyDist):
    """
    PolyDist with bound information.
    """

    @abstractmethod
    def coeff_1_norm_bound(self) -> float:
        """
        Returns an upper bound on the coefficient 1-norm of a polynomial drawn from this distribution.
        """
        ...

    @abstractmethod
    def coeff_inf_norm_bound(self) -> float:
        """
        Returns an upper bound on the coefficient infinity norm of a polynomial drawn from this distribution.
        """
        ...

    def __add__(self, other: PolyDistWithBounds) -> PolyDistWithBounds:
        return UnknownDist(
            n=self.n,
            _coeff_1_norm_bound=self.coeff_1_norm_bound() + other.coeff_1_norm_bound(),
            _coeff_inf_norm_bound=self.coeff_inf_norm_bound() + other.coeff_inf_norm_bound(),
        )


class KeyDist(PolyDistWithBounds):
    @abstractmethod
    def expected_squared_l2_norm(self) -> float:
        """Return the expected squared L2 norm of a polynomial drawn from this distribution."""
        ...


@dataclass(slots=True, frozen=True)
class Ternary(KeyDist):
    def expected_squared_l2_norm(self) -> float:
        return self.n / 2

    def coeff_1_norm_bound(self) -> float:
        return self.n

    def coeff_inf_norm_bound(self) -> float:
        return 1


@dataclass(slots=True, frozen=True)
class SparseTernary(KeyDist):
    """
    Sparse ternary distribution with h +-1 coefficients and n-h 0 coefficients.
    """

    h: int

    def expected_squared_l2_norm(self) -> float:
        return float(self.h)

    def __post_init__(self) -> None:
        if not 0 <= self.h <= self.n:
            raise ValueError("Sparse ternary weight must be between zero and n")

    def coeff_1_norm_bound(self) -> float:
        return self.h

    def coeff_inf_norm_bound(self) -> float:
        return 1


@dataclass(slots=True, frozen=True)
class UniformBounded(PolyDistWithBounds):
    bound: int

    def __post_init__(self) -> None:
        if self.bound < 0:
            raise ValueError("Uniform bound must be nonnegative")

    def coeff_1_norm_bound(self) -> float:
        return self.n * self.bound

    def coeff_inf_norm_bound(self) -> float:
        return self.bound


@dataclass(slots=True, frozen=True)
class UniformModQ(PolyDist):
    pass


@dataclass(slots=True, frozen=True)
class DiscreteGaussian(PolyDistWithBounds):
    stdev: Fraction

    def __post_init__(self) -> None:
        if self.stdev <= 0:
            raise ValueError("Gaussian standard deviation must be positive")

    def coeff_1_norm_bound(self) -> int:
        from ren.tables import DistTables

        return self.n * DistTables.gaussian_max_coefficient(self.stdev)

    def coeff_inf_norm_bound(self) -> float:
        return STD_DEVS_FOR_128_BIT_SECURITY * float(self.stdev)


@dataclass(slots=True, frozen=True)
class UnknownDist(PolyDistWithBounds):
    """A distribution described by bounds on its coefficient norms."""

    _coeff_1_norm_bound: float
    _coeff_inf_norm_bound: float

    def coeff_1_norm_bound(self) -> float:
        return self._coeff_1_norm_bound

    def coeff_inf_norm_bound(self) -> float:
        return self._coeff_inf_norm_bound
