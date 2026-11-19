"""Polynomial approximation and evaluation for CKKS."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
import operator
from typing import overload, override

import numpy as np

from ren.ckks.ckks import Ciphertext, encode


def _format_scalar_coeff(c: complex) -> str:
    """Format a single scalar coefficient for :meth:`ComplexPoly.__str__`."""
    if c.imag == 0:
        return f"{c.real:.3f}"
    return f"({c.real:.3f}{c.imag:+.3f}j)"


def _format_coeff(value: np.ndarray) -> str:
    """Format one coefficient of a :class:`ComplexPoly` for :meth:`__str__`.

    A plain coefficient is a NumPy scalar; a per-slot coefficient is a 1-d row,
    abbreviated as ``[c0, c1, ... (n more)]``.
    """
    if np.ndim(value) == 0:
        return _format_scalar_coeff(complex(value))
    n = value.shape[0]
    head = ", ".join(_format_scalar_coeff(complex(v)) for v in value[:2])
    if n > 2:
        return f"[{head}, ... ({n - 2} more)]"
    return f"[{head}]"


# Operand types accepted as a scalar or per-slot multiplier/addend. Anything
# else returns NotImplemented so Python can try the reflected operation.
_SCALAR_LIKE: tuple[type, ...] = (int, float, complex, list, tuple, np.ndarray, np.generic)


def _align_batch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align plain and stacked coefficients before broadcasting.

    Degree is axis 0. Adding a slot axis to the plain array prevents NumPy
    from aligning its degree axis with the stacked array's slot axis.
    """
    if a.ndim < b.ndim:
        a = a[:, None]
    elif b.ndim < a.ndim:
        b = b[:, None]
    return a, b


def _pad_degree(arr: np.ndarray, n: int, batch: tuple[int, ...]) -> np.ndarray:
    """Zero-pad ``arr`` up to ``n`` coefficients, broadcast over ``batch``.

    ``arr`` must already carry the aligned batch rank (see :func:`_align_batch`);
    otherwise the assignment below would broadcast along the wrong axis.
    """
    out = np.zeros((n, *batch), dtype=np.complex128)
    out[: arr.shape[0]] = arr
    return out


class PolyBasis(ABC):
    """Bases must satisfy B_0(x) = 1 and B_1(x) = x.

    Each B_j is independent of the number of coefficients, so zero-padding
    a coefficient array preserves the polynomial.
    """

    @abstractmethod
    def vandermonde(self, xs: np.ndarray, ncoeffs: int) -> np.ndarray:
        """Build the Vandermonde-like matrix ``V[i, j] = B_j(xs[i])``.

        Returns a real array of shape ``(len(xs), ncoeffs)`` for real ``xs``.
        """
        ...

    @abstractmethod
    def evaluate(self, coeffs: np.ndarray, x: complex) -> np.ndarray:
        """Evaluate complex128 coefficients, with degree on axis 0, at a scalar.

        Return an array of shape ``coeffs.shape[1:]``: a 0-d value for plain
        coefficients or a per-slot row for stacked ones. An empty degree axis
        represents zero and preserves the slot shape.
        """
        ...

    @abstractmethod
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply complex128 coefficient arrays with degree on axis 0.

        Inputs must have the same rank, with compatible slot axes. Slot axes
        broadcast; trailing zero coefficients are not trimmed. If either
        input has an empty degree axis, the result does too.
        """
        ...

    @abstractmethod
    def format_term(self, degree: int) -> str:
        """Format B_degree(X), including "1" for degree zero."""
        ...


@dataclass(frozen=True)
class Monomial(PolyBasis):
    """The monomial basis [1, x, x^2, x^3, ...]."""

    @override
    def vandermonde(self, xs: np.ndarray, ncoeffs: int) -> np.ndarray:
        return np.vander(xs, ncoeffs, increasing=True)

    @override
    def evaluate(self, coeffs: np.ndarray, x: complex) -> np.ndarray:
        """Evaluate ``sum_k c_k x^k`` using Horner's method.

        The backward recurrence is
        ``r_k = c_k + x * r_{k+1}`` for ``k = d-1..0`` starting from ``r_d = c_d``.
        """
        if len(coeffs) == 0:
            return np.zeros(coeffs.shape[1:], dtype=np.complex128)
        result = np.asarray(coeffs[-1])
        for coeff in reversed(coeffs[:-1]):
            result = result * x + coeff
        return np.asarray(result)

    @override
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply monomial coefficients by schoolbook multiplication."""
        n, m = a.shape[0], b.shape[0]
        batch = np.broadcast_shapes(a.shape[1:], b.shape[1:])
        if n == 0 or m == 0:
            return np.empty((0, *batch), dtype=np.complex128)
        out = np.zeros((n + m - 1, *batch), dtype=np.complex128)
        # Add a_i * b to the degree window [i, i+m).
        for i in range(n):
            out[i : i + m] += a[i] * b
        return out

    @override
    def format_term(self, degree: int) -> str:
        if degree == 0:
            return "1"
        if degree == 1:
            return "X"
        return f"X^{degree}"


@dataclass(frozen=True)
class Chebyshev(PolyBasis):
    """The Chebyshev basis:

    T_0(x) = 1
    T_1(x) = x
    T_2(x) = 2x^2 - 1
    T_3(x) = 4x^3 - 3x
    T_4(x) = 8x^4 - 8x^2 + 1
    ...

    For j >= 2, T_j(x) = 2x*T_{j-1}(x) - T_{j-2}(x).
    """

    @override
    def vandermonde(self, xs: np.ndarray, ncoeffs: int) -> np.ndarray:
        if ncoeffs == 0:
            return np.empty((len(xs), 0))
        return np.polynomial.chebyshev.chebvander(np.asarray(xs, dtype=np.float64), ncoeffs - 1)

    @override
    def evaluate(self, coeffs: np.ndarray, x: complex) -> np.ndarray:
        """Evaluate ``sum_k c_k T_k(x)`` using Clenshaw's recurrence.

        The backward recurrence is
        ``b_k = c_k + 2x * b_{k+1} - b_{k+2}`` for ``k = d..1`` and returns
        ``c_0 + x * b_1 - b_2``.
        """
        if len(coeffs) == 0:
            return np.zeros(coeffs.shape[1:], dtype=np.complex128)
        b_kp1 = b_kp2 = np.zeros(coeffs.shape[1:], dtype=np.complex128)
        for coeff in reversed(coeffs[1:]):
            b_kp1, b_kp2 = coeff + 2 * x * b_kp1 - b_kp2, b_kp1
        return np.asarray(coeffs[0] + x * b_kp1 - b_kp2)

    @override
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply using T_m * T_n = 1/2 * (T_{m+n} + T_{|m-n|})."""
        n, m = a.shape[0], b.shape[0]
        batch = np.broadcast_shapes(a.shape[1:], b.shape[1:])
        if n == 0 or m == 0:
            return np.empty((0, *batch), dtype=np.complex128)
        out = np.zeros((n + m - 1, *batch), dtype=np.complex128)
        for i in range(n):
            half = 0.5 * (a[i] * b)
            out[i : i + m] += half  # T_{i+j}
            # For T_{|i-j|}, degrees descend for j <= i and ascend for j > i.
            split = min(i + 1, m)
            out[i - split + 1 : i + 1] += half[:split][::-1]
            out[1 : m - split + 1] += half[split:]
        return out

    @override
    def format_term(self, degree: int) -> str:
        return "1" if degree == 0 else f"T_{degree}(X)"


@cache
def _mono_to_cheb_matrix(n: int) -> np.ndarray:
    """Matrix converting ``n`` monomial coefficients to Chebyshev coefficients.

    Column ``i`` contains the Chebyshev coefficients of ``x**i``. Multiplying
    by ``x`` builds each column from the previous one, using ``x*T_0 = T_1``
    and ``x*T_k = (T_{k-1} + T_{k+1})/2`` for ``k > 0``.

    For ``i >= k`` of the same parity, the exact entry in row ``k``, column
    ``i`` is ``2**(1-i) * binom(i, (i-k)/2)``, halved for row zero. Other
    entries are zero. The matrix is stored as float64.
    """
    matrix = np.zeros((n, n))
    if n:
        matrix[0, 0] = 1.0
    for i in range(1, n):
        coeffs = np.polynomial.chebyshev.chebmulx(matrix[:i, i - 1])
        matrix[: len(coeffs), i] = coeffs
    matrix.setflags(write=False)
    return matrix


@cache
def _cheb_to_mono_matrix(n: int) -> np.ndarray:
    """Matrix whose row ``k`` holds ``T_k`` in monomial coefficients.

    Built by the three-term recurrence ``T_k = 2x*T_{k-1} - T_{k-2}``, where
    multiplying by ``x`` is a shift along the coefficient axis. Since
    ``sum_k c_k T_k = sum_i (sum_k c_k L[k, i]) x**i``, converting a coefficient
    vector applies the *transpose*.

    The exact entries are integers. The matrix is stored as float64 and
    promotes to complex128 when applied to complex coefficients.
    """
    matrix = np.zeros((n, n))
    if n >= 1:
        matrix[0, 0] = 1.0
    if n >= 2:
        matrix[1, 1] = 1.0
    for k in range(2, n):
        matrix[k, 1:] = 2.0 * matrix[k - 1, :-1]
        matrix[k] -= matrix[k - 2]
    matrix.setflags(write=False)
    return matrix


def _optimal_cps_terms(j: int, s: int, logd: int) -> list[int]:
    """Return degrees ``[t_0, t_1, ...]`` such that
    ``CPS_j(x) = T_{t_0}(x) * T_{t_1}(x) * ...``.

    The decomposition strips leading powers of two from ``j`` (largest
    first, up to ``2 ** (logd - 1)``), stopping when the remainder is
    smaller than the next power; then splits whatever is left into a
    multiple-of-``s`` part and a residue ``< s``. An empty list means
    ``CPS_j(x) = 1`` (only the case ``j == 0``).
    """
    terms: list[int] = []
    remaining = j
    for i in range(logd - 1, -1, -1):
        if remaining >= (1 << i):
            terms.append(1 << i)
            remaining -= 1 << i
        else:
            break
    n_val = s * (remaining // s)
    if n_val > 0:
        terms.append(n_val)
    m_val = remaining % s
    if m_val > 0:
        terms.append(m_val)
    return terms


def remez_coefficients(
    f: Callable[[float], float],
    *,
    degree: int,
    interval: tuple[float, float],
    basis: PolyBasis = Monomial(),
    maxiter: int = 50,
    grid_size: int = 10000,
) -> tuple[np.ndarray, float]:
    """Approximate real-valued ``f`` on ``[a, b]`` using Remez exchange.

    Remez seeks a minimax polynomial by solving the equioscillation system

        ``sum_j c_j * B_j(refs[i]) + (-1)^i E = f(refs[i])``

    where ``B_j`` are the basis polynomials from :meth:`PolyBasis.vandermonde`.
    The coefficients use the requested basis directly on ``[a, b]``, without
    a change of variable. Error extrema are found on a finite grid, so the
    returned error does not bound the error over the whole interval.

    Args:
        f: The real-valued function to approximate.
        degree: Degree of the approximating polynomial.
        interval: Approximation interval ``[a, b]``.
        basis: Basis instance in which to express the result.
        maxiter: Maximum Remez iterations (default 50).
        grid_size: Number of points in the fine grid used to locate
            error extrema (default 10000).

    Returns:
        ``(coeffs, error)`` where ``coeffs`` is the float64 array of
        basis coefficients ``[c_0, c_1, ..., c_degree]`` (lowest-degree
        first) in the requested ``basis`` and ``error`` is the achieved L∞
        approximation error on the fine grid.
    """
    a, b = interval
    ncoeffs = degree + 1
    nrefs = degree + 2  # equioscillation theorem: degree+2 reference points

    # Start at Chebyshev-Gauss nodes mapped from [-1, 1] to [a, b].
    refs_unit = np.array([np.cos(np.pi * (2 * i + 1) / (2 * nrefs)) for i in range(nrefs)])
    refs = np.sort(0.5 * (a + b) + 0.5 * (b - a) * refs_unit)

    grid = np.linspace(a, b, grid_size)
    f_grid = np.array([f(float(x)) for x in grid])
    V_grid = basis.vandermonde(grid, ncoeffs)
    signs = np.array([(-1.0) ** i for i in range(nrefs)])

    coeffs = np.zeros(ncoeffs)
    best_coeffs = np.zeros(ncoeffs)
    max_err = float("inf")
    best_err = float("inf")

    for _ in range(maxiter):
        # Solve the equioscillation system in the chosen basis.
        V = basis.vandermonde(refs, ncoeffs)
        A = np.column_stack([V, signs])
        rhs = np.array([f(float(t)) for t in refs])
        sol = np.linalg.solve(A, rhs)
        coeffs = sol[:ncoeffs]
        E = sol[ncoeffs]

        p_grid = V_grid @ coeffs
        errors = f_grid - p_grid
        abs_errors = np.abs(errors)
        max_err = float(np.max(abs_errors))

        if max_err < best_err:
            best_err = max_err
            best_coeffs = coeffs.copy()

        # Stop when the maximum grid error matches the reference-point error.
        if abs(max_err - abs(E)) < 1e-12 * max(abs(E), 1.0):
            break

        # Exchange reference points: split the grid at sign changes and pick
        # the largest |error| in each segment to form an alternating set.
        sign_changes: list[int] = []
        for i in range(len(errors) - 1):
            if errors[i] * errors[i + 1] < 0:
                sign_changes.append(i + 1)

        boundaries = [0] + sign_changes + [len(errors)]
        alt: list[int] = []
        for seg_start, seg_end in zip(boundaries, boundaries[1:]):
            best = seg_start + int(np.argmax(abs_errors[seg_start:seg_end]))
            alt.append(best)

        # Remove boundary extrema to preserve alternation when trimming to nrefs.
        while len(alt) > nrefs:
            if abs(errors[alt[0]]) <= abs(errors[alt[-1]]):
                alt.pop(0)
            else:
                alt.pop(-1)

        if len(alt) < nrefs:
            continue  # not enough extrema, skip exchange this iteration

        refs = np.sort(grid[alt])

    # Return the smallest grid error found, even if the last iteration was worse.
    if best_err < max_err:
        coeffs = best_coeffs
        max_err = best_err

    return coeffs, max_err


class ComplexPoly:
    """Polynomials in C[X] for approximation and ciphertext evaluation.

    Coefficients are stored in a complex128 array, from lowest to highest
    degree along axis 0. The shape is ``(ncoeffs,)`` for a plain polynomial
    or ``(ncoeffs, nslots)`` for stacked polynomials, one per CKKS slot.
    Use :meth:`stack` or multiply by a per-slot array to create a stack.
    Arithmetic broadcasts over slots; basis conversion acts on the degree axis.

    These differ from the REN engine's polynomials in Z/qZ[X]/(X^N+1).
    """

    coeffs: np.ndarray
    basis: PolyBasis

    # Defer ``ndarray * poly`` to :meth:`__rmul__` instead of letting NumPy
    # build an object array out of the polynomial.
    __array_ufunc__ = None

    def __init__(self, coeffs: Sequence[complex] | np.ndarray, basis: PolyBasis = Monomial()):
        """
        Initialize a polynomial from a sequence of coefficients, lowest degree
        first, or from a degree-major array. A 2-d array is read as per-slot
        coefficients, one column per slot. Trailing coefficients that are zero
        across every slot are stripped.

        The input is always copied: polynomials are treated as immutable (the
        stored array is marked read-only), and several callers cache them.
        """
        arr = np.array(coeffs, dtype=np.complex128)
        if arr.ndim == 0 or arr.ndim > 2:
            raise ValueError(f"ComplexPoly coefficients must be 1-d (plain) or 2-d (per-slot), got ndim={arr.ndim}")
        if arr.shape[0]:
            nonzero = (arr != 0).reshape(arr.shape[0], -1).any(axis=1)
            idx = np.flatnonzero(nonzero)
            # ``arr[:0]`` rather than a plain empty array so an all-zero stacked
            # polynomial keeps its slot axis.
            arr = arr[: int(idx[-1]) + 1] if idx.size else arr[:0]
        arr.setflags(write=False)
        self.coeffs = arr
        self.basis = basis

    @classmethod
    def stack(cls, polys: Sequence[ComplexPoly]) -> ComplexPoly:
        """
        Bundle a sequence of polynomials, all sharing the same basis, into a
        single :class:`ComplexPoly` whose coefficients are per-slot rows.

        The returned polynomial has ``coeffs[k, i] == polys[i].coeffs[k]``, with
        shorter polynomials zero-padded to the maximum degree.

        Requires ``len(polys) >= 1``. All input polynomials must share the same
        basis and must have plain (non-stacked) coefficients.
        """
        if len(polys) == 0:
            raise ValueError("stack requires at least one polynomial")

        basis = polys[0].basis
        for i, p in enumerate(polys[1:], start=1):
            if p.basis != basis:
                raise ValueError(f"All polynomials must share the same basis; polys[0] has {basis} but polys[{i}] has {p.basis}")
        for i, p in enumerate(polys):
            if p.coeffs.ndim != 1:
                raise TypeError(f"polys[{i}] has per-slot coefficient rows; stack expects polynomials with plain coefficients")

        # Fill poly-major and transpose once: writing columns straight into a
        # degree-major buffer would be one scattered store per coefficient.
        rows = np.zeros((len(polys), max(p.coeffs.shape[0] for p in polys)), dtype=np.complex128)
        for i, p in enumerate(polys):
            rows[i, : p.coeffs.shape[0]] = p.coeffs
        return cls(np.ascontiguousarray(rows.T), basis=basis)

    @classmethod
    def zero(cls, basis: PolyBasis = Monomial()):
        """
        Returns the zero polynomial P(x) = 0.
        """
        return ComplexPoly([], basis=basis)

    @classmethod
    def one(cls, basis: PolyBasis = Monomial()):
        """
        Returns the constant polynomial P(x) = 1.
        """
        return ComplexPoly([1], basis=basis)

    @classmethod
    def id(cls, basis: PolyBasis = Monomial()):
        """
        Returns the identity polynomial P(x) = x.
        """
        return ComplexPoly([0, 1], basis=basis)

    @classmethod
    def lagrange_interpolant(cls, points: Sequence[tuple[complex, complex]]) -> ComplexPoly:
        """
        Compute the Lagrange interpolating polynomial through the given
        ``(node, value)`` pairs. Returns a polynomial in Monomial basis such
        that ``p(node) == value`` for every pair.
        """
        result = ComplexPoly.zero(basis=Monomial())
        for i, (node, value) in enumerate(points):
            # Build the i-th Lagrange basis polynomial L_i(x) = prod_{j != i} (x - node_j) / (node_i - node_j)
            L = ComplexPoly.one(basis=Monomial())
            denom = 1.0 + 0j
            for j, (other, _) in enumerate(points):
                if j != i:
                    L *= ComplexPoly([-other, 1], basis=Monomial())
                    denom *= node - other
            result = result + (value / denom) * L

        return result

    @classmethod
    def chebyshev_interpolant(cls, f: Callable[[float], complex], *, degree: int, interval: tuple[float, float] | None = None) -> ComplexPoly:
        """
        Compute the Chebyshev interpolant of f on [a, b] (default [-1, 1]).
        Uses the Fast Chebyshev Transform (FCT) to find the Chebyshev coefficients at the
        Chebyshev nodes on [-1, 1]. For a custom interval the nodes are mapped to [a, b]
        and Lagrange interpolation is used to return a polynomial in Monomial basis such
        that p.evaluate(t) ≈ f(t) for t in [a, b].
        """
        if degree < 0:
            return ComplexPoly.zero(basis=Chebyshev())

        a, b = (-1, 1) if interval is None else interval
        n = degree + 1
        k = np.arange(n)
        # Chebyshev nodes on [-1, 1], then mapped to [a, b]
        nodes_std = np.cos(np.pi * (2 * k + 1) / (2 * n))
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
        samples = np.array([f(x) for x in nodes], dtype=complex)

        if interval is None:
            coeffs = dct_ii(samples) * (2 / n)
            coeffs[0] /= 2
            return ComplexPoly(coeffs, basis=Chebyshev())
        return cls.lagrange_interpolant(list(zip(nodes, samples)))

    @classmethod
    def remez_approximant(
        cls,
        f: Callable[[float], complex],
        *,
        degree: int,
        interval: tuple[float, float],
        basis: PolyBasis = Monomial(),
        maxiter: int = 50,
        grid_size: int = 10000,
    ) -> tuple[ComplexPoly, float]:
        """Approximate ``f`` on ``[a, b]`` using Remez exchange.

        Fits the real and imaginary parts independently, then combines their
        coefficients. This need not give the complex minimax polynomial.
        The returned error is ``sqrt(err_re**2 + err_im**2)``, an upper bound
        on ``|f(x) - p(x)|`` at the sampled grid points, not over the whole
        interval.

        Args:
            f: The function to approximate. Accepts real- or
                complex-valued returns.
            degree: Degree of the approximating polynomial.
            interval: Approximation interval ``[a, b]``.
            basis: Output basis instance, such as :class:`Monomial` or :class:`Chebyshev`.
            maxiter: Maximum Remez iterations (default 50).
            grid_size: Number of points in the fine grid used to locate
                error extrema (default 10000).

        Returns:
            ``(poly, error)`` where ``poly`` is a :class:`ComplexPoly` in
            the requested basis approximating ``f`` on ``[a, b]`` and
            ``error`` bounds the maximum absolute error on the sampled grid.
        """

        # Remez needs real-valued errors to track sign changes. Converting to
        # complex lets f return either real or complex values.
        def f_real(x: float) -> float:
            return complex(f(x)).real

        def f_imag(x: float) -> float:
            return complex(f(x)).imag

        coeffs_re, err_re = remez_coefficients(f_real, degree=degree, interval=interval, basis=basis, maxiter=maxiter, grid_size=grid_size)
        coeffs_im, err_im = remez_coefficients(f_imag, degree=degree, interval=interval, basis=basis, maxiter=maxiter, grid_size=grid_size)
        coeffs = coeffs_re + 1j * coeffs_im
        err = float(np.hypot(err_re, err_im))
        return ComplexPoly(coeffs, basis=basis), err

    def to_basis(self, basis: PolyBasis) -> ComplexPoly:
        """Convert this polynomial to a different basis.

        The conversion matrix acts on the degree axis and handles every slot
        in a stacked polynomial at once.
        """
        if self.basis == basis:
            return self

        n = len(self.coeffs)
        match self.basis, basis:
            case Monomial(), Chebyshev():
                matrix = _mono_to_cheb_matrix(n)
            case Chebyshev(), Monomial():
                matrix = _cheb_to_mono_matrix(n).T
            case _:
                raise NotImplementedError(f"Basis conversion from {self.basis} to {basis} is not implemented yet.")
        return ComplexPoly(matrix @ self.coeffs, basis=basis)

    @property
    def degree(self) -> int:
        """
        Returns the degree of the polynomial (-1 for zero polynomial).

        For a stacked polynomial this is the largest degree over all slots.
        """
        return self.coeffs.shape[0] - 1

    @property
    def is_zero(self) -> bool:
        """
        Returns true if the polynomial is zero, false otherwise.
        """
        return self.coeffs.shape[0] == 0

    def _elementwise(
        self,
        other: ComplexPoly | complex | Sequence[complex] | np.ndarray,
        op: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> ComplexPoly:
        """Combine with a polynomial, scalar, or per-slot array coefficient-wise.

        Scalars and per-slot arrays are converted to constant polynomials.
        Both operands are aligned and zero-padded to the longer degree, so
        ``op`` sees two arrays of identical shape.
        """
        if isinstance(other, _SCALAR_LIKE):
            const = np.asarray(other, dtype=np.complex128)
            if const.ndim > 1:
                raise ValueError(f"Expected a scalar or a per-slot 1-d array, got ndim={const.ndim}")
            # Add the degree-0 axis so per-slot values stay on the batch axis,
            # even when their count matches the number of coefficients.
            other = ComplexPoly(const[None, ...], basis=self.basis)
        if not isinstance(other, ComplexPoly):
            return NotImplemented
        if self.basis != other.basis:
            raise ValueError(f"Cannot combine polynomials with different bases: {self.basis} vs {other.basis}")
        a, b = _align_batch(self.coeffs, other.coeffs)
        n = max(a.shape[0], b.shape[0])
        batch = np.broadcast_shapes(a.shape[1:], b.shape[1:])
        return ComplexPoly(op(_pad_degree(a, n, batch), _pad_degree(b, n, batch)), basis=self.basis)

    def __add__(self, other: ComplexPoly | complex | Sequence[complex] | np.ndarray) -> ComplexPoly:
        return self._elementwise(other, operator.add)

    def __sub__(self, other: ComplexPoly | complex | Sequence[complex] | np.ndarray) -> ComplexPoly:
        return self._elementwise(other, operator.sub)

    def __radd__(self, other: complex | Sequence[complex] | np.ndarray) -> ComplexPoly:
        return self + other

    def __rsub__(self, other: complex | Sequence[complex] | np.ndarray) -> ComplexPoly:
        return -self + other

    def __neg__(self) -> ComplexPoly:
        return ComplexPoly(-self.coeffs, basis=self.basis)

    def __mul__(self, other: ComplexPoly | complex | Sequence[complex] | np.ndarray) -> ComplexPoly:
        """Multiply by a polynomial, scalar, or per-slot array."""
        if isinstance(other, _SCALAR_LIKE):
            scale = np.asarray(other, dtype=np.complex128)
            if scale.ndim > 1:
                raise ValueError(f"Scalar multiplier must be a scalar or a per-slot 1-d array, got ndim={scale.ndim}")
            # ``scale`` always lives on the batch axis, never the degree axis, so
            # multiplying a plain polynomial by a per-slot array stacks it -- even
            # when the array length equals the coefficient count.
            coeffs = self.coeffs
            if coeffs.ndim == 1 and scale.ndim == 1:
                coeffs = coeffs[:, None]
            return ComplexPoly(coeffs * scale, basis=self.basis)

        if not isinstance(other, ComplexPoly):
            return NotImplemented

        if self.basis != other.basis:
            raise ValueError(f"Cannot multiply polynomials with different bases: {self.basis} vs {other.basis}")

        a, b = _align_batch(self.coeffs, other.coeffs)
        return ComplexPoly(self.basis.multiply(a, b), basis=self.basis)

    def __rmul__(self, scalar: complex | Sequence[complex] | np.ndarray) -> ComplexPoly:
        return self * scalar

    @overload
    def evaluate(self, x: Ciphertext) -> Ciphertext: ...
    @overload
    def evaluate(self, x: complex) -> complex: ...

    def evaluate(self, x: complex | Ciphertext) -> complex | Ciphertext:
        """Evaluate on a ciphertext or complex scalar.

        Ciphertexts use Paterson-Stockmeyer evaluation with internal CPS
        conversion for Chebyshev coefficients. Scalars use Horner's method
        for :class:`Monomial` and Clenshaw's recurrence for :class:`Chebyshev`.

        Scalar evaluation requires a plain polynomial; use :meth:`evaluate_slots`
        for a stacked one. Ciphertext evaluation supports both shapes.
        """
        if isinstance(x, Ciphertext):
            return self._evaluate_paterson_stockmeyer(x)
        if self.coeffs.ndim != 1:
            raise TypeError("evaluate on a complex scalar requires a plain polynomial; use evaluate_slots for a stacked one")
        return complex(self.basis.evaluate(self.coeffs, x))

    def evaluate_slots(self, x: complex) -> np.ndarray:
        """Evaluate a stacked polynomial at a complex scalar, one value per slot.

        Returns an array of shape ``coeffs.shape[1:]``. Plain polynomials
        require :meth:`evaluate` instead.
        """
        if self.coeffs.ndim == 1:
            raise TypeError("evaluate_slots requires a stacked polynomial; use evaluate for a plain one")
        return self.basis.evaluate(self.coeffs, x)

    @property
    def depth(self) -> int:
        """Ciphertext levels consumed by :meth:`evaluate`.

        Returns ``floor(log2(degree)) + 1``, or zero for constants and the zero polynomial.
        """
        if not isinstance(self.basis, Monomial | Chebyshev):
            raise NotImplementedError(f"ciphertext evaluation is not defined for {self.basis}")
        return 0 if self.degree < 1 else self.degree.bit_length()

    def _evaluate_paterson_stockmeyer(self, x: Ciphertext) -> Ciphertext:
        """Paterson-Stockmeyer evaluation on a ciphertext.

        Precomputes baby steps ``B[0..s]`` and giant steps
        ``G[0..num_giants]`` using a power-of-two split ``s`` (near
        ``sqrt(d+1)``). Partitions ``[0, d+1)`` into coefficient
        blocks, saves their pending combinations, then evaluates the
        remaining tail and combines the saved blocks in reverse order.
        Folding the highest-degree segment separately uses the smallest
        available power, keeping the multiplicative depth at
        ``floor(log2(d)) + 1``.

        Chebyshev coefficients are converted to CPS coefficients. The baby and
        giant steps use ``B[m+n] = 2 * B[m] * B[n] - B[|m-n|]``;
        :class:`Monomial` uses ``B[m+n] = B[m] * B[n]``.
        """
        logd = self.depth
        d = self.degree
        if d < 0:
            return Ciphertext.zero(level=x.level)
        if d == 0:
            # Constants need no multiplications. Scalars become constants directly;
            # per-slot coefficients are encoded as plaintext, then wrapped as a
            # trivial ciphertext.
            c = self.coeffs[0]
            if np.ndim(c) == 0:
                return Ciphertext.const(complex(c), level=x.level)
            return encode(c, msg_bound=float(np.max(np.abs(c), initial=0.0)), level=x.level).to_ciphertext()

        chebyshev = isinstance(self.basis, Chebyshev)
        s = 1 << (logd // 2)
        num_giants = (1 << logd) // (2 * s)

        # Baby steps B[i] (degree i) and giant steps G[k] (degree k*s),
        # both built by depth-balanced binary splitting.
        B: list = [1, x]
        for i in range(2, s + 1):
            m = 1 << ((i - 1).bit_length() - 1)
            n = i - m
            B.append(2 * B[m] * B[n] - B[m - n] if chebyshev else B[m] * B[n])

        G: list = [1]
        if num_giants >= 1:
            G.append(B[s])
        for i in range(2, num_giants + 1):
            m = 1 << ((i - 1).bit_length() - 1)
            n = i - m
            G.append(2 * G[m] * G[n] - G[m - n] if chebyshev else G[m] * G[n])

        # Each interval evaluates sum_{i in [lower, upper)} c_i * B[i - lower].
        # A one-term interval can remain a raw coefficient; Ciphertext.dot
        # converts it to a trivial ciphertext. Coefficients are scalars for
        # plain polynomials and per-slot rows for stacked ones.
        coeffs = _cps_coefficients(self.coeffs, s) if chebyshev else self.coeffs
        coeff_values = list(coeffs)

        def _blocks(lower: int, upper: int):
            """``(baby, giant)`` pairs summing ``c[lower:upper]`` in blocks of ``s``:
            the first block pairs raw coefficients with baby steps, later
            blocks pair their dot product (or lone coefficient) with a giant step."""
            for start in range(lower, upper, s):
                end = min(upper, start + s)
                if start == lower:
                    yield from zip(coeff_values[start:end], B[: end - start])
                elif end - start == 1:
                    yield coeff_values[start], G[(start - lower) // s]
                else:
                    yield Ciphertext.dot(coeff_values[start:end], B[: end - start]), G[(start - lower) // s]

        folds = []
        lower, upper = 0, d + 1
        while upper - lower > 2:
            mid = (lower + (1 << logd)) // 2
            if upper <= mid:
                break
            # Fold the topmost block separately so its giant-step
            # factor uses the smallest available power.
            width = mid - lower
            folds.append((list(_blocks(lower, mid)), G[width // s] if width >= s else B[width]))
            lower = mid

        result = coeff_values[lower]
        if upper - lower > 1:
            babys, giants = zip(*_blocks(lower, upper))
            result = Ciphertext.dot(list(babys), list(giants))
        for pairs, giant in reversed(folds):
            pairs.append((result, giant))
            babys, giants = zip(*pairs)
            result = Ciphertext.dot(list(babys), list(giants))
        # The ``d == 0`` early exit above guarantees the top-level
        # interval has size >= 2, so evaluation returns a Ciphertext
        # rather than a size-1 leaf.
        assert isinstance(result, Ciphertext)
        return result

    @override
    def __str__(self) -> str:
        """Format the polynomial for debugging, abbreviating per-slot coefficients as
        ``[c0, c1, ... (n more)]``.
        """
        terms: list[str] = []
        for degree, coeff in enumerate(self.coeffs):
            if not np.any(coeff):
                continue
            term = _format_coeff(coeff)
            if degree > 0:
                term += self.basis.format_term(degree)
            terms.append(term)
        return " + ".join(terms) or "0"


def dct_ii_real(x: np.ndarray) -> np.ndarray:
    """Type-II Discrete Cosine Transform for a real input array.

    Implemented as an ``n``-point FFT on a reordered version of ``x`` followed
    by a twiddle multiplication. Returns a real-valued array of the same
    length as ``x``.
    """
    n = len(x)
    y = np.empty(n, dtype=x.dtype)
    y[: (n + 1) // 2] = x[::2]
    y[(n + 1) // 2 :] = x[-1 - (n % 2) :: -2]
    y_fft = np.fft.fft(y)
    j = np.arange(n)
    twiddle = np.exp(-1j * np.pi * j / (2 * n))
    return np.real(y_fft * twiddle)


def dct_ii(x: np.ndarray) -> np.ndarray:
    """Type-II Discrete Cosine Transform for a complex input array.

    Applies :func:`dct_ii_real` to the real and imaginary parts separately.
    Returns a complex-valued array of the same length as ``x``.
    """
    return dct_ii_real(np.real(x)) + 1j * dct_ii_real(np.imag(x))


def _cps_coefficients(coeffs: np.ndarray, s: int) -> np.ndarray:
    """Convert Chebyshev coefficients to Chebyshev Paterson-Stockmeyer (CPS) coefficients.

    The split ``s`` and trimmed degree ``len(coeffs) - 1`` must match the
    evaluation schedule.
    """
    # For nonempty terms = [t_0, ..., t_{n-1}] with sum k:
    #
    #   T_k(x) = 2^{n-1} * (T_{t_0} * T_{t_1} * ... * T_{t_{n-1}})(x)
    #          - sum T_{|signed sum|}(x)
    #
    # The correction sum fixes the last sign positive and excludes the all-plus choice.
    # The product is CPS_k(x), so c_k contributes 2^{n-1} * c_k to cps_coeffs[k].
    # Process degrees k from highest to lowest, subtracting c_k from working at
    # the absolute value of each signed sum.
    # The loop also visits the all-plus choice to clear working[k].
    working = np.array(coeffs)  # writeable copy
    cps_coeffs = np.zeros_like(coeffs)
    logd = (len(coeffs) - 1).bit_length()
    for k in range(len(coeffs) - 1, -1, -1):
        # Copy the row: the all-plus sum below clears working[k].
        c_k = np.array(working[k])
        terms = _optimal_cps_terms(k, s, logd)
        if len(terms) == 0:
            # ``CPS_0 = 1``, only reachable when ``k == 0``.
            cps_coeffs[0] += c_k
        else:
            factor = 1 << (len(terms) - 1)
            cps_coeffs[k] += factor * c_k
            for i in range(factor):
                signed = terms[-1]
                for j in range(len(terms) - 1):
                    signed += terms[j] * (1 if (i & (1 << j)) else -1)
                working[abs(signed)] -= c_k
    return cps_coeffs
