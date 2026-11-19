from collections.abc import Iterable
from dataclasses import dataclass
from math import inf, prod
from typing import final

from ren.bigfloat import BigFloat, FloatLike


@final
class Bound(BigFloat):
    __slots__ = ()

    def _validate(self) -> None:
        if self._value < 0:
            raise ValueError("bound must be non-negative")


_ZERO_BOUND = Bound(0)
_INFINITE_BOUND = Bound(inf)


@dataclass(slots=True, frozen=True, kw_only=True, init=False)
class CKKSNoiseInfo:
    """
    Information about the noise in a CKKS plaintext.

    We define the norm of a polynomial P in Z_q[x]/(x^n + 1), denoted |P|,
    to be either the coefficient infinity-norm, i.e. the maximum absolute value
    of any coefficient, or the evaluation infinity-norm, i.e. the maximum
    absolute value of P(zeta) for any root zeta of x^n + 1.

    The message of a plaintext is defined as the "true" plaintext, i.e. the plaintext
    as it would have been if it was computed in the clear instead of in FHE.
    The noise of a plaintext is defined as the difference between the plaintext
    polynomial and the message.

    The message and noise of a ciphertext are defined as the message and noise
    of the plaintext resulting from decrypting the ciphertext.

    A CKKSNoiseInfo object holds information about noise and messages as follows.
    For a plaintext or ciphertext with message m and noise e, it holds the
    following fields:

    noise_bound: An upper bound on |e|.
    noise_heuristic_bound: A heuristic upper bound on |e|.
    msg_bound: An upper bound on |m|.
    """

    noise_bound: Bound
    noise_heuristic_bound: Bound
    msg_bound: Bound

    def __init__(self, noise_bound: FloatLike, noise_heuristic_bound: FloatLike, msg_bound: FloatLike) -> None:
        object.__setattr__(self, "noise_bound", Bound(noise_bound))
        object.__setattr__(self, "noise_heuristic_bound", Bound(noise_heuristic_bound))
        object.__setattr__(self, "msg_bound", Bound(msg_bound))

    @classmethod
    def zero(cls) -> CKKSNoiseInfo:
        return _ZERO_NOISE_INFO

    @classmethod
    def unbounded(cls) -> CKKSNoiseInfo:
        return _UNBOUNDED_NOISE_INFO

    @classmethod
    def sum(cls, terms: Iterable[CKKSNoiseInfo]) -> CKKSNoiseInfo:
        """Sum bounds and combine heuristic errors in quadrature."""
        noise_bound, variance, msg_bound = BigFloat(0), BigFloat(0), BigFloat(0)
        for term in terms:
            if term.is_unbounded:
                return cls.unbounded()
            noise_bound += term.noise_bound
            variance += term.noise_heuristic_bound**2
            msg_bound += term.msg_bound
        return cls(noise_bound=noise_bound, noise_heuristic_bound=variance.sqrt(), msg_bound=msg_bound)

    @property
    def is_unbounded(self) -> bool:
        return self.msg_bound.is_infinite() or self.noise_bound.is_infinite() or self.noise_heuristic_bound.is_infinite()

    def __add__(self, other: CKKSNoiseInfo) -> CKKSNoiseInfo:
        """Add two noise information objects."""
        if self.is_unbounded or other.is_unbounded:
            return CKKSNoiseInfo.unbounded()
        # TODO: handle dependent addition
        noise_heuristic_bound = (self.noise_heuristic_bound**2 + other.noise_heuristic_bound**2).sqrt()
        return CKKSNoiseInfo(
            noise_bound=self.noise_bound + other.noise_bound,
            noise_heuristic_bound=noise_heuristic_bound,
            msg_bound=self.msg_bound + other.msg_bound,
        )

    def __rmul__(self, scale: FloatLike) -> CKKSNoiseInfo:
        """Scale the noise of a noise information object by a scalar."""
        return self * scale

    def _rescale(self, src_moduli: tuple[int, ...], dst_moduli: tuple[int, ...]) -> CKKSNoiseInfo:
        """Scale bounds by the basis-conversion ratio, excluding rounding error."""
        src_set, dst_set = set(src_moduli), set(dst_moduli)
        removed_prod = prod(modulus for modulus in src_moduli if modulus not in dst_set)
        added_prod = prod(modulus for modulus in dst_moduli if modulus not in src_set)
        return self * (BigFloat(added_prod) / removed_prod)

    def __mul__(self, other: FloatLike | CKKSNoiseInfo) -> CKKSNoiseInfo:
        # TODO: handle dependent multiplication
        if not isinstance(other, CKKSNoiseInfo):
            magnitude = abs(BigFloat(other))
            if magnitude == 0:
                return CKKSNoiseInfo.zero()
            if magnitude == 1:
                return self
            if self.is_unbounded:
                return CKKSNoiseInfo.unbounded()
            return CKKSNoiseInfo(
                noise_bound=self.noise_bound * magnitude,
                noise_heuristic_bound=self.noise_heuristic_bound * magnitude,
                msg_bound=self.msg_bound * magnitude,
            )

        if self.is_unbounded or other.is_unbounded:
            return CKKSNoiseInfo.unbounded()
        other_total_bound = other.noise_bound + other.msg_bound
        noise_bound = self.noise_bound * other_total_bound + self.msg_bound * other.noise_bound
        noise_heuristic_bound = self.noise_heuristic_bound * other_total_bound + self.msg_bound * other.noise_heuristic_bound
        return CKKSNoiseInfo(noise_bound=noise_bound, noise_heuristic_bound=noise_heuristic_bound, msg_bound=self.msg_bound * other.msg_bound)


_ZERO_NOISE_INFO = CKKSNoiseInfo(
    noise_bound=_ZERO_BOUND,
    noise_heuristic_bound=_ZERO_BOUND,
    msg_bound=_ZERO_BOUND,
)
_UNBOUNDED_NOISE_INFO = CKKSNoiseInfo(
    noise_bound=_INFINITE_BOUND,
    noise_heuristic_bound=_INFINITE_BOUND,
    msg_bound=_INFINITE_BOUND,
)
