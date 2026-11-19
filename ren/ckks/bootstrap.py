# This file implements the CKKS bootstrapping algorithm.


from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from math import ceil, cos, expm1, log, log1p, pi, prod, sin, sqrt

import numpy as np

from ren.bigfloat import BigFloat, FloatLike
from ren.function import function
from ren.poly import Poly
from ren.ckks.ckks import Ciphertext, _switch_secret
from ren.ckks.complexpoly import ComplexPoly, Chebyshev
from ren.ckks.context import context
from ren.ckks.encoding import scale_slots
from ren.ckks.keys import CKKSKeySet
from ren.ckks.matrix import LinearTransform, SparseMatrix
from ren.ckks.params import CKKSParams, CKKSTransition
from ren.ckks.utils import count_calls
from ren.ckks.errors import InvalidParameterError

TAU = 2 * pi


def _stage_scales(scale: FloatLike, count: int) -> tuple[BigFloat, ...]:
    if count < 1:
        raise ValueError("expected at least one matrix stage")
    per_stage_scale = BigFloat(scale) ** (BigFloat(1) / count)
    return (per_stage_scale,) * count


@cache
def _mod_r_poly(*, exp_degree: int, squaring_count: int, r: float) -> tuple[ComplexPoly, float]:
    exponent = 1 << squaring_count
    amplitude = 1 / (4 * pi) ** (1 / exponent)

    def expi(x: float) -> complex:
        return cos(x) + 1j * sin(x)

    def f(x: float) -> complex:
        return amplitude * expi(2 * pi * x / (r * exponent))

    remez_poly, grid_error = ComplexPoly.remez_approximant(
        f,
        degree=exp_degree,
        interval=(-1.0, 1.0),
        basis=Chebyshev(),
    )
    # If |p - f| <= e on the grid and |f| = a, the powering error is at most
    # (a + e)**exponent - a**exponent. Mod-one takes twice its imaginary part.
    mod_one_error = expm1(exponent * log1p(grid_error / amplitude)) / (2 * pi)
    return remez_poly, mod_one_error


def root_of_unity(k, n: int):
    """``exp(2*pi*i*k/n)``; accepts a scalar int or an int ndarray of exponents."""
    k = np.mod(k, n)
    angle = (TAU / n) * k
    return np.cos(angle) + 1j * np.sin(angle)


@cache
def _base_exponents(n: int) -> np.ndarray:
    """``5**i mod 2n`` for ``i in range(n // 2)``, via a running product (computed once per n)."""
    n2 = 2 * n
    out = np.empty(n // 2, dtype=np.int64)
    v = 1
    for i in range(n // 2):
        out[i] = v
        v = (v * 5) % n2
    return out


def coefficients_to_slots(ct: Ciphertext, *, levels, scale: FloatLike = 1.0) -> tuple[Ciphertext, Ciphertext]:
    """
    The goal of this function is to take an encrypted polynomial in R[X]/(X^N+1) and
    put its N coefficients into the 2*(N/2) slots of two polynomials homomorphically.

    Specifically, let the encrypted polynomial be m(X) = m_0 + m_1*X + m_2*X^2 + ... + m_(N-1)*X^(N-1)
    in R[X]/(X^N+1). Then its slots are [m(zeta), m(zeta^5), m(zeta^25), ..., m(zeta^(5^(N/2-1)))] in
    C^(N/2), where zeta satisfies X^N+1=0. The goal is to turn m(X) into a polynomial whose
    slots are [m_rev(0), m_rev(1), ..., m_rev(n/2-1)] and [m_rev(n/2), m_rev(n/2+1), ..., m_rev(n-1)],
    where rev(x) denotes the bit-reversal of x as a log_2(n)-digit binary number.
    This is done via the identities

    [1              1                  1                   ... 1                         ] [m(zeta)            ]
    [zeta^-rev(2)   zeta^(-rev(2)*5)   zeta^(-rev(2)*25)   ... zeta^(-rev(2)*5^(N/2-1))  ] [m(zeta^5)          ]
    [zeta^-rev(4)   zeta^(-rev(4)*5)   zeta^(-rev(4)*25)   ... zeta^(-rev(4)*5^(N/2-1))  ] [m(zeta^25)         ]
    [...            ...                ...                 ... ...                       ] [...                ]
    [zeta^-rev(N-2) zeta^(-rev(N-2)*5) zeta^(-rev(N-2)*25) ... zeta^(-rev(N-2)*5^(N/2-1))] [m(zeta^(5^(N/2-1)))]
    +
    [1             1                 1                  ... 1                        ] [m(zeta^-1)          ]
    [zeta^rev(2)   zeta^(rev(2)*5)   zeta^(rev(2)*25)   ... zeta^(rev(2)*5^(N/2-1))  ] [m(zeta^-5)          ]
    [zeta^rev(4)   zeta^(rev(4)*5)   zeta^(rev(4)*25)   ... zeta^(rev(4)*5^(N/2-1))  ] [m(zeta^-25)         ]
    [...           ...               ...                ... ...                      ] [...                 ]
    [zeta^rev(N-2) zeta^(rev(N-2)*5) zeta^(rev(N-2)*25) ... zeta^(rev(N-2)*5^(N/2-1))] [m(zeta^(-5^(N/2-1)))]
    =
        [m_rev(0)  ]
        [m_rev(2)  ]
    N * [m_rev(4)  ]
        [...       ]
        [m_rev(N-2)]

    and

    [zeta^(-rev(1))   zeta^(-rev(1)*5)   zeta^(-rev(1)*25)   ... zeta^(-rev(1)*5^(N/2-1))  ] [m(zeta)            ]
    [zeta^(-rev(3))   zeta^(-rev(3)*5)   zeta^(-rev(3)*25)   ... zeta^(-rev(3)*5^(N/2-1))  ] [m(zeta^5)          ]
    [zeta^(-rev(5))   zeta^(-rev(5)*5)   zeta^(-rev(5)*25)   ... zeta^(-rev(5)*5^(N/2-1))  ] [m(zeta^25)         ]
    [...              ...                ...                 ... ...                       ] [...                ]
    [zeta^(-rev(N-1)) zeta^(-rev(N-1)*5) zeta^(-rev(N-1)*25) ... zeta^(-rev(N-1)*5^(N/2-1))] [m(zeta^(5^(N/2-1)))]
    +
    [zeta^rev(1)   zeta^(rev(1)*5)   zeta^(rev(1)*25)   ... zeta^(rev(1)*5^(N/2-1))  ] [m(zeta^-1)          ]
    [zeta^rev(3)   zeta^(rev(3)*5)   zeta^(rev(3)*25)   ... zeta^(rev(3)*5^(N/2-1))  ] [m(zeta^-5)          ]
    [zeta^rev(5)   zeta^(rev(5)*5)   zeta^(rev(5)*25)   ... zeta^(rev(5)*5^(N/2-1))  ] [m(zeta^-25)         ]
    [...           ...               ...                ... ...                      ] [...                 ]
    [zeta^rev(N-1) zeta^(rev(N-1)*5) zeta^(rev(N-1)*25) ... zeta^(rev(N-1)*5^(N/2-1))] [m(zeta^(-5^(N/2-1)))]
    =
        [m_rev(1)  ]
        [m_rev(3)  ]
    N * [m_rev(5)  ]
        [...       ]
        [m_rev(N-1)]

    Notice that the second matrix is the conjugate of the first matrix, the fourth matrix
    is the conjugate of the third matrix, and the third matrix is the first matrix multiplied
    by -i. Hence, to efficiently compute the product, it suffices to factor the first matrix
    into a product of sparse matrices (here, sparse means "small number of occupied diagonals").
    This is what the l_matrix() function does.

    The optional 'scale' parameter multiplies the result by 'scale' without increasing depth, distributing
    the scale factor over the matrices.
    """
    n = ct.ring.n
    l = l_matrix(ct, levels=levels, scale=scale / n)
    l_conj = l.conjugate()
    first = l_conj + l
    second = (l_conj - l).times_i()
    return (first, second)


def _stage_transforms(matrices: Sequence[SparseMatrix], *, scales: tuple[BigFloat, ...], start_level: int) -> tuple[LinearTransform, ...]:
    """The encoded stages of a factorization, stage ``i`` scaled by
    ``scales[i]`` and consuming level ``start_level - i``.

    Encoded once per context and stage matrix (the factorizations are cached,
    so a stage is always the same matrix object).  The scale is folded into
    the matrix entries, which is exactly what the encoder sees either way, so
    every stage's plaintext magnitude stays near ``scale``. The encoder
    selects double-hoisting when all direct rotation keys are present.
    """
    ctx = context()
    transforms = ctx._bootstrap_transforms
    stages: list[LinearTransform] = []
    for i, (matrix, scale) in enumerate(zip(matrices, scales, strict=True)):
        level = start_level - i
        key = (matrix, level, float(scale))
        if key not in transforms:
            transforms[key] = LinearTransform.encode(matrix * scale, level=level)
        stages.append(transforms[key])
    return tuple(stages)


def _apply_matrix_stages(ct: Ciphertext, matrices: Sequence[SparseMatrix], *, scales: tuple[BigFloat, ...]) -> Ciphertext:
    """Apply a chain of sparse matrix transforms, one per stage scale."""
    for transform in _stage_transforms(matrices, scales=scales, start_level=ct.level):
        ct = transform(ct)
    return ct


def l_matrix(ct: Ciphertext, *, levels: int, scale: FloatLike = 1.0):
    """
    This multiplies a ciphertext by the matrix whose rows are

    [1              1                  1                   ... 1                         ]
    [zeta^-rev(2)   zeta^(-rev(2)*5)   zeta^(-rev(2)*25)   ... zeta^(-rev(2)*5^(N/2-1))  ]
    [zeta^-rev(4)   zeta^(-rev(4)*5)   zeta^(-rev(4)*25)   ... zeta^(-rev(4)*5^(N/2-1))  ]
    [...            ...                ...                 ... ...                       ]
    [zeta^-rev(N-2) zeta^(-rev(N-2)*5) zeta^(-rev(N-2)*25) ... zeta^(-rev(N-2)*5^(N/2-1))]

    using a factorization into sparse matrices. The point of the 'levels' parameter
    is to make sure that applying this function to a ciphertext reduces the ciphertext
    by this number of levels. This is done by combining some of the terms in the
    factorization given by the full l_matrix factorization into "levels" number of
    matrices. The tradeoff here is that the fewer levels that are consumed, the more
    rotations, multiplications, and keys are needed.

    The optional "scale" parameter scales the resulting ciphertext by the given value,
    distributing it evenly over the matrix stages.
    """
    matrices = l_matrix_partial_factorization(ct.ring.n.bit_length() - 1, levels)
    return _apply_matrix_stages(ct, matrices, scales=_stage_scales(scale, len(matrices)))


def _combine_factorization(matrices: Sequence[SparseMatrix], count: int, *, reverse: bool = False) -> tuple[SparseMatrix, ...]:
    """Combine sparse matrix factors into ``count`` balanced groups."""
    if not matrices:
        return ()

    factor_count = len(matrices)
    group_sizes = tuple(factor_count * (i + 1) // count - factor_count * i // count for i in range(count))
    if reverse:
        group_sizes = group_sizes[::-1]

    n = matrices[0].n
    combined: list[SparseMatrix] = []
    start = 0
    for size in group_sizes:
        group = SparseMatrix.identity(n)
        end = start + size
        for matrix in matrices[start:end]:
            group = matrix @ group
        combined.append(group)
        start = end
    return tuple(combined)


@cache
def l_matrix_partial_factorization(logn: int, count: int) -> tuple[SparseMatrix, ...]:
    """
    Combines the full factorization of the l_matrix (which has logn-1 terms)
    into a factorization involving 'count' number of matrices.

    Cached: every call returns the same SparseMatrix objects, which keeps
    context transform-plan lookups (keyed by matrix identity) hitting.
    """
    return _combine_factorization(l_matrix_factorization(logn), count, reverse=True)


def l_matrix_factorization(logn: int) -> tuple[SparseMatrix, ...]:
    """
    Returns a sparse factorization of the matrix

    [1              1                  1                   ... 1                         ]
    [zeta^-rev(2)   zeta^(-rev(2)*5)   zeta^(-rev(2)*25)   ... zeta^(-rev(2)*5^(N/2-1))  ]
    [zeta^-rev(4)   zeta^(-rev(4)*5)   zeta^(-rev(4)*25)   ... zeta^(-rev(4)*5^(N/2-1))  ]
    [...            ...                ...                 ... ...                       ]
    [zeta^-rev(N-2) zeta^(-rev(N-2)*5) zeta^(-rev(N-2)*25) ... zeta^(-rev(N-2)*5^(N/2-1))]

    but with rows permuted by bit-reversal. The matrices are returned
    in right-to-left order; that is, function evaluation order.

    For example, for N=16 this factorization is

    [1    1     1     1     1     1     1     1    ]
    [z^-4 z^-20 z^-4  z^-20 z^-4  z^-20 z^-4  z^-20]
    [z^-2 z^-10 z^-18 z^-26 z^-2  z^-10 z^-18 z^-26]
    [z^-6 z^-30 z^-22 z^-14 z^-6  z^-30 z^-22 z^-14]
    [z^-1 z^-5  z^-25 z^-29 z^-17 z^-21 z^-9  z^-13]
    [z^-5 z^-25 z^-29 z^-17 z^-21 z^-9  z^-13 z^-1 ]
    [z^-3 z^-15 z^-11 z^-23 z^-19 z^-31 z^-27 z^-7 ]
    [z^-7 z^-3  z^-15 z^-11 z^-23 z^-19 z^-31 z^-27]

    =

    [1    1                                     ]
    [z^-4 z^-20                                 ]
    [           1    1                          ]
    [           z^-4 z^-20                      ]
    [                      1    1               ]
    [                      z^-4 z^-20           ]
    [                                 1   1     ]
    [                                 z^-4 z^-20]

    *

    [1          1                                 ]
    [     1           1                           ]
    [z^-2       z^-18                             ]
    [     z^-10       z^-26                       ]
    [                       1          1          ]
    [                            1           1    ]
    [                       z^-2       z^-18      ]
    [                            z^-10       z^-26]

    *

    [1                     1                     ]
    [     1                      1               ]
    [          1                       1         ]
    [                1                      1    ]
    [z^-1                  z^-17                 ]
    [     z^-5                   z^-21           ]
    [          z^-25                   z^-9      ]
    [                z^-29                  z^-13]

    .

    Note that the matrix is a 2^(logn-1) by 2^(logn-1) matrix.
    """
    n = 1 << logn
    base = _base_exponents(n)
    idx = np.arange(n // 2)
    matrices: list[SparseMatrix] = []
    for l in range(logn - 1):
        r = 1 << (logn - 2 - l)
        roots = root_of_unity(-(base << l), 2 * n)
        mask = (idx & (1 << (logn - 2 - l))) == 0
        left = np.where(mask, roots, 0.0)
        middle = np.where(mask, 1.0, roots)
        right = np.where(mask, 0.0, 1.0)
        matrices.append(SparseMatrix.from_diagonals(n // 2, {-r: np.roll(left, -r), 0: middle, r: np.roll(right, r)}))
    return tuple(matrices)


def slots_to_coefficients(ct1: Ciphertext, ct2: Ciphertext, *, levels: int, scale: FloatLike = 1.0) -> Ciphertext:
    """
    This is the inverse of the coefficients_to_slots method. Specifically, it takes two
    ciphertexts encrypting (via slots) the plaintexts

           [m_rev(0)  ]          [m_rev(1)  ]
           [m_rev(2)  ]          [m_rev(3)  ]
    pt_1 = [m_rev(4)  ] , pt_2 = [m_rev(5)  ]
           [...       ]          [...       ]
           [m_rev(N-2)]          [m_rev(N-1)]

    and produces a ciphertext encrypting the polynomial

    pt_out = m_0 + m_1*X + m_2*X^2 + ... + m_(N-1)*X^(N-1)

    in the ring Z/qZ[X]/(X^N+1). It does this by first computing

                      [m_rev(0) +   i * m_rev(1)  ]
                      [m_rev(2) +   i * m_rev(3)  ]
    pt_1 + i * pt_2 = [m_rev(4) +   i * m_rev(5)  ]
                      [...                        ]
                      [m_rev(N-2) + i * m_rev(N-1)]

    then multiplying this by the matrix

    [1   z^rev(2)             z^rev(4)             z^rev(6)             ... z^rev(N-2)            ]
    [1   z^(rev(2)*5)         z^(rev(4)*5)         z^(rev(6)*5)         ... z^(rev(N-2)*5)        ]
    [1   z^(rev(2)*25)        z^(rev(4)*25)        z^(rev(6)*25)        ... z^(rev(N-2)*25)       ]
    [... ...                  ...                  ...                  ... ...                   ]
    [1   z^(rev(2)*5^(N/2-1)) z^(rev(4)*5^(N/2-1)) z^(rev(6)*5^(N/2-1)) ... z^(rev(N-2)*5^(N/2-1))]

    to get

    [m_0 + m_1 * z             + m_2 * z^2             + ... + m_(N-1) * z^(N-1)            ] = [f(z)            ]
    [m_0 + m_1 * z^5           + m_2 * z^10            + ... + m_(N-1) * z^((N-1)*5)        ] = [f(z^5)          ]
    [m_0 + m_1 * z^25          + m_2 * z^50            + ... + m_(N-1) * z^((N-1)*25)       ] = [f(z^25)         ]
    [...                                                                                    ] = [...             ]
    [m_0 + m_1 * z^(5^(N/2-1)) + m_2 * z^(2*5^(N/2-1)) + ... + m_(N-1) * z^((N-1)*5^(N/2-1))] = [f(z^(5^(N/2-1)))]

    where f(X) = m_0 + m_1*X + m_2*X^2 + ... + m_(N-1)*X^(N-1) and z = exp(2*i*pi/(2*N)).

    """
    return l_inv_matrix(ct1 + ct2.times_i(), levels=levels, scale=scale)


def l_inv_matrix(ct: Ciphertext, *, levels: int, scale: FloatLike = 1.0) -> Ciphertext:
    """
    This function multiplies a ciphertext by the matrix

    [1   z^rev(2)             z^rev(4)             z^rev(6)             ... z^rev(N-2)            ]
    [1   z^(rev(2)*5)         z^(rev(4)*5)         z^(rev(6)*5)         ... z^(rev(N-2)*5)        ]
    [1   z^(rev(2)*25)        z^(rev(4)*25)        z^(rev(6)*25)        ... z^(rev(N-2)*25)       ] .
    [... ...                  ...                  ...                  ... ...                   ]
    [1   z^(rev(2)*5^(N/2-1)) z^(rev(4)*5^(N/2-1)) z^(rev(6)*5^(N/2-1)) ... z^(rev(N-2)*5^(N/2-1))]

    This is done efficiently via a sparse factorization. Like the l_matrix function,
    the point of the 'levels' parameter is to make sure that applying this function
    to a ciphertext reduces the ciphertext by this number of levels.

    The optional "scale" parameter scales the resulting ciphertext by the given value,
    distributing it evenly over the matrix stages.
    """
    matrices = l_inv_matrix_partial_factorization(ct.ring.n.bit_length() - 1, levels)
    return _apply_matrix_stages(ct, matrices, scales=_stage_scales(scale, len(matrices)))


@cache
def l_inv_matrix_partial_factorization(logn: int, count: int) -> tuple[SparseMatrix, ...]:
    """
    Combines the full factorization of the l_inv_matrix (which has logn-1 terms)
    into a factorization involving 'count' number of matrices.

    Cached: every call returns the same SparseMatrix objects, which keeps
    context transform-plan lookups (keyed by matrix identity) hitting.
    """
    return _combine_factorization(l_inv_matrix_factorization(logn), count, reverse=False)


def l_inv_matrix_factorization(logn: int) -> tuple[SparseMatrix, ...]:
    """
    Returns a sparse factorization of the matrix

    [1   z^rev(2)             z^rev(4)             z^rev(6)             ... z^rev(N-2)            ]
    [1   z^(rev(2)*5)         z^(rev(4)*5)         z^(rev(6)*5)         ... z^(rev(N-2)*5)        ]
    [1   z^(rev(2)*25)        z^(rev(4)*25)        z^(rev(6)*25)        ... z^(rev(N-2)*25)       ]
    [... ...                  ...                  ...                  ... ...                   ]
    [1   z^(rev(2)*5^(N/2-1)) z^(rev(4)*5^(N/2-1)) z^(rev(6)*5^(N/2-1)) ... z^(rev(N-2)*5^(N/2-1))]

    The matrices are returned in right-to-left order; that is, function evaluation order.

    For example, for N=16 this factorization is

    [1 z^4  z^2  z^6  z    z^5  z^3  z^7 ]
    [1 z^20 z^10 z^30 z^5  z^25 z^15 z^3 ]
    [1 z^4  z^18 z^22 z^25 z^29 z^11 z^15]
    [1 z^20 z^26 z^14 z^29 z^17 z^23 z^11]
    [1 z^4  z^2  z^6  z^17 z^21 z^19 z^23]
    [1 z^20 z^10 z^30 z^21 z^9  z^31 z^19]
    [1 z^4  z^18 z^22 z^9  z^13 z^27 z^31]
    [1 z^20 z^26 z^14 z^13 z    z^7  z^27]

    =

    [1           z                  ]
    [   1             z^5           ]
    [      1               z^25     ]
    [         1                 z^29]
    [1           z^17               ]
    [   1             z^21          ]
    [      1               z^9      ]
    [         1                 z^13]

    *

    [1    z^2                      ]
    [   1      z^10                ]
    [1    z^18                     ]
    [   1      z^26                ]
    [                1    z^2      ]
    [                   1      z^10]
    [                1    z^18     ]
    [                   1      z^26]

    *

    [1  z^4                            ]
    [1  z^20                           ]
    [         1  z^4                   ]
    [         1  z^20                  ]
    [                  1  z^4          ]
    [                  1  z^20         ]
    [                           1  z^4 ]
    [                           1  z^20]

    .

    Note that the matrix is a 2^(logn-1) by 2^(logn-1) matrix.
    """
    n = 1 << logn
    base = _base_exponents(n)
    idx = np.arange(n // 2)
    matrices: list[SparseMatrix] = []
    for l in range(logn - 1):
        r = 1 << l
        roots = root_of_unity(base << (logn - 2 - l), 2 * n)
        mask = (idx & (1 << l)) == 0
        left = np.where(mask, 0.0, 1.0)
        middle = np.where(mask, 1.0, roots)
        right = np.where(mask, roots, 0.0)
        matrices.append(SparseMatrix.from_diagonals(n // 2, {-r: left, 0: middle, r: right}))
    return tuple(matrices)


def mod_r_div_r(x: Ciphertext, *, r: float, exp_degree: int, squaring_count: int) -> Ciphertext:
    """
    Reduces x modulo r then divides by r, where r is a real number between 0 and 1.
    The input x must be in the range [-1, 1] and close to a multiple of r for this to work.
    This is because this function basically evaluates

    mod_r_div_r(x) = Im[1/2pi * e^(2*pi*i*x/r)] = 1/2pi * sin(2*pi*x/r).

    WARNING: this function does not check if the ciphertext actually encrypts
    real numbers (as opposed to complex non-real numbers.)
    """

    assert 0 < r < 1, f"{r} is not between 0 and 1"
    poly, _ = _mod_r_poly(exp_degree=exp_degree, squaring_count=squaring_count, r=r)
    base = poly.evaluate(x)
    # The ideal scaled exponential has magnitude at most one. The bootstrap plan accounts for the approximation error.
    if base.noise_info.msg_bound > base.scale:
        base = base.replace(noise_info=replace(base.noise_info, msg_bound=base.scale))
    z = base ** (1 << squaring_count)
    return (z.conjugate() - z).times_i()


def mod_one(x: Ciphertext, *, range: float, exp_degree: int, squaring_count: int) -> Ciphertext:
    """
    Reduces k + eps to eps for -range < k < range using the approximation

    x mod 1 = i * (conj(z) - z), z = (1/(4pi)^{1/8} * e^{pi*i*x/4})^8.

    WARNING: this function does not check if the ciphertext actually encrypts
    real numbers (as opposed to complex non-real numbers.)
    """
    return mod_r_div_r(x / range, r=1 / range, exp_degree=exp_degree, squaring_count=squaring_count)


def _cts_scale(params: CKKSParams, *, mod_one_range: int) -> BigFloat:
    """CTS scale factor that aligns the mod-raise t-terms for exact cancellation."""
    if params.bootstrap_base_level is None:
        raise InvalidParameterError("bootstrapping is not enabled for this parameter set")
    q_0 = prod(params.bootstrap_base_level.moduli)
    q_0_correction = params.scale(params.highest_level) / q_0
    return q_0_correction / mod_one_range


def _stc_scale(params: CKKSParams) -> BigFloat:
    """STC scale factor that restores the CTS correction."""
    output_level = params.bootstrap_output_level
    bootstrap_base = params.bootstrap_base_level
    if output_level is None or bootstrap_base is None:
        raise InvalidParameterError("bootstrapping is not enabled for this parameter set")
    q_0 = prod(bootstrap_base.moduli)
    tail_scale = params.scale(params.highest_level)
    for upper in range(params.highest_level, output_level, -1):
        tail_scale = tail_scale * tail_scale * params.ring_at(upper - 1).q / params.ring_at(upper).q
    return params.scale(output_level) * q_0 / (bootstrap_base.scale * tail_scale)


@dataclass(frozen=True, slots=True)
class _BootstrapIteration:
    input_transition: CKKSTransition
    stc_scales: tuple[BigFloat, ...]
    output_factor: int


def _bootstrap_mod_one_range(keys: CKKSKeySet) -> int:
    """
    Return the mod-one range required for bootstrapping to fail with
    at most the configured probability. Under sparse encapsulation, ModRaise
    uses the ephemeral secret; otherwise it uses the aggregate secret behind
    the keyset.

    ModRaise modifies every coefficient by some multiple of q=q_0. The goal is
    to find a bound K on this multiple such that

    ```text {.asciimath}
    P(max_i |c_i| > K*q) < p
    ```

    Let h be the expected squared L2 norm of the secret. Each coefficient of
    the raised polynomial is an inner product of the secret and values sampled
    uniformly from [-q/2, q/2]. Approximating this sum by a Gaussian gives

    ```text {.asciimath}
    P(N(0, h*q^2/12) > K*q) < p/(2N)
    P(N(0, 1) > K*sqrt(12/h)) < p/(2N)
    ```

    Using the Mills Ratio P(Z > z) approx e^(-z^2/2) / (z*sqrt(2pi)) gives

    ```text {.asciimath}
    P(Z > z) ~~ e^(-z^2/2)/(z*sqrt(2pi))
    e^(-6K^2/h) * sqrt(h/(24pi)) / K = p/(2N)
    ```

    and the goal is to solve for K. Equivalently,

    ```text {.asciimath}
    -6K^2/h + 0.5*ln(h/(24pi)) - ln(K) = ln(p/(2N))
    6K^2/h + ln(K) = ln((2N)/p) + 0.5*ln(h/(24pi))
    ```

    A reasonable first approximation for K is

    ```text {.asciimath}
    K ~~ sqrt(h/6 * (ln((2N)/p) + 0.5*ln(h/(24pi))))
    ```

    and a better approximation is

    ```text {.asciimath}
    K ~~ sqrt(h/6 * (ln((2N)/p) + 0.5*ln(h/(24pi)) - 0.5*ln(h/6 * (ln((2N)/p) + 0.5*ln(h/(24pi))))))
    ```
    """
    params = keys.params
    boot_params = params.boot_params
    if boot_params is None:
        raise RuntimeError("Bootstrapping not allowed with current parameter set")

    secret_expected_squared_l2_norm = boot_params.sparse_encapsulation_weight
    if secret_expected_squared_l2_norm is None:
        secret_expected_squared_l2_norm = keys.secret_expected_squared_l2_norm
    term1 = log(2 * params.n / boot_params.failure_probability) + 0.5 * log(secret_expected_squared_l2_norm / (24 * pi))
    first_approx = (secret_expected_squared_l2_norm / 6) * term1
    k_squared = (secret_expected_squared_l2_norm / 6) * (term1 - 0.5 * log(first_approx))
    return ceil(sqrt(k_squared))


def _sine_error(radius: BigFloat) -> BigFloat:
    """Decoded slot error from replacing a centered residue x by sin(2*pi*x)/(2*pi).

    If every normalized slot has magnitude at most radius, Parseval gives ||coeffs||₂ <= radius.
    For e(x) = x - sin(2*pi*x)/(2*pi), e(x)/x² increases on (0, 1/2]. Hence the sum of
    the absolute coefficient errors is at most e(radius), attained by a constant polynomial.
    """
    if radius >= 0.5:
        raise InvalidParameterError("normalized bootstrap input reaches half the modulus; increase bootstrap_message_ratio")
    angle = TAU * radius
    if angle < 1e-3:
        return radius * angle**2 / 6  # Cubic upper bound avoids cancellation near zero.
    return radius - sin(float(angle)) / TAU


def _bootstrap_plan(keys: CKKSKeySet, log_bound: int) -> tuple[int, tuple[_BootstrapIteration, ...]]:
    params = keys.params
    boot_params = params.boot_params
    if boot_params is None:
        raise RuntimeError("Bootstrapping not allowed with current parameter set")
    if log_bound > params.log_range:
        raise InvalidParameterError(f"log_bound must not exceed the parameter-set log_range ({params.log_range})")

    mod_one_range = _bootstrap_mod_one_range(keys)
    _, mod_one_error = _mod_r_poly(exp_degree=boot_params.exp_degree, squaring_count=boot_params.squaring_count, r=1 / mod_one_range)
    # The polynomial fit can contribute an error in every coefficient.
    poly_error = params.n * mod_one_error
    bounds = boot_params.bounds(log_bound)
    shared_restore_factor = BigFloat(2) ** bounds[-1]
    stc_scale = _stc_scale(params)
    stc_stage_count = len(l_inv_matrix_partial_factorization(params.logn, boot_params.stc_depth))
    base_stc_scales = _stage_scales(stc_scale * shared_restore_factor, stc_stage_count)
    iterations = []
    # The final pass needs only to resolve its own input, so its required gain is zero.
    for bound, gain in zip(bounds, (*boot_params.precision_gains, 0), strict=True):
        input_transition = params.bootstrap_transition(correction_factor=BigFloat(2) ** -bound)
        restore_factor = 1 / input_transition.message_ratio
        # Use the rounded transition to express the input bound in units of the bootstrap base modulus.
        radius = BigFloat(2) ** bound / (params.bootstrap_message_ratio * restore_factor)
        sine_error = _sine_error(radius)
        budget = radius * BigFloat(2) ** -gain
        if poly_error + sine_error >= budget:
            hint = "Try a larger bootstrap_message_ratio" if sine_error >= poly_error else "Try a higher exp_degree or squaring_count"
            if gain > 0:
                hint += " or a smaller precision gain"
            raise InvalidParameterError(
                f"mod-one approximation is too coarse for bootstrap range K={mod_one_range} and gain {gain} "
                f"(degree {boot_params.exp_degree}, {boot_params.squaring_count} squarings): "
                f"polynomial error {poly_error:.3g} + sine error {float(sine_error):.3g} >= budget {float(budget):.3g}. {hint}."
            )
        output_factor = 1 << round((restore_factor / shared_restore_factor).log2())
        # Keep the shared correction in every STC stage except the last,
        # which also absorbs integer-multiplier rounding for this iteration.
        last_stc_scale = base_stc_scales[-1] * restore_factor / (shared_restore_factor * output_factor)
        iterations.append(
            _BootstrapIteration(
                input_transition=input_transition,
                stc_scales=(*base_stc_scales[:-1], last_stc_scale),
                output_factor=output_factor,
            )
        )
    return mod_one_range, tuple(iterations)


# Count each pass separately because meta-bootstrapping runs multiple passes.
@count_calls
def _bootstrap_iteration(ct_in: Ciphertext, iteration: _BootstrapIteration, *, keys: CKKSKeySet, mod_one_range: int) -> Ciphertext:
    """
    One CKKS bootstrap iteration, with input normalization and output restoration:
    0. Modulus raise the ciphertext to the highest level. This takes
       a ciphertext encrypting a polynomial P_0 in (Z/q_0Z)[X]/(X^N+1) in
       a lower modulus q_0 and sets the modulus to q_L without changing
       the contents of the polynomials making up the ciphertext. The
       resulting ciphertext now encrypts a message P_L = m + q_0*t for some
       t in (Z/q_LZ)[X]/(X^N+1) with small (close to zero) coefficients.
    1. Apply coefficients_to_slots on the raised ciphertext. This takes
       the N coefficients of P_L and puts them in the N/2 slots of two
       ciphertexts C_1 and C_2. (there are 2*N/2 = N total slots among
       the two ciphertexts.)
    2. Apply mod_reduce on both C_1 and C_2, which evaluates a homomorphic
       centered mod function modulo q. Specifically, the slots of C_1 and
       C_2 hold the coefficients of P_L, which if taken modulo q, recover
       the coefficients of P_0.
    3. Apply slots_to_coefficients on C_1 and C_2. This moves the N/2
       coefficients from each of C_1 and C_2 into the coefficients of a
       new ciphertext, which encrypts P_0 at the higher modulus Q_{L-k},
       where k is the number of levels consumed during this bootstrapping
       process.

    The input is normalized to the bootstrap-base ring and restored at the end.
    """
    params = keys.params
    boot_params = params.boot_params
    assert boot_params is not None

    ct_level_zero = ct_in if ct_in.level == 0 else ct_in.to_level(0)
    input_transition = iteration.input_transition
    prec = context()._slot_prec
    # The computation level and bootstrap base can share a physical ring.
    # In that case the transition only applies the normalization multiplier.
    ct_down = ct_level_zero.map_polys(input_transition.apply).replace(
        noise_info=ct_level_zero.noise_info * input_transition.poly_ratio,
        _values=ct_level_zero._map_values(lambda values: scale_slots(values, input_transition.message_ratio, prec=prec)),
    )
    restore_factor = BigFloat(1) / input_transition.message_ratio
    plaintext_values = ct_down._values
    h = boot_params.sparse_encapsulation_weight
    if h is not None:
        dense_to_sparse_key = keys.dense_to_sparse_key
        sparse_to_dense_key = keys.sparse_to_dense_key
        if dense_to_sparse_key is None or sparse_to_dense_key is None:
            raise RuntimeError(
                "sparse_encapsulation is enabled but the context has no encapsulation keys; "
                "generate them on the key set before creating the context "
                "(multiparty key generation does not support them yet)"
            )
        # Sparse-secret encapsulation (https://eprint.iacr.org/2022/024):
        # re-encrypt under the ephemeral sparse secret so the mod-raise
        # overflow term q_0*t is bounded by the small encapsulation weight (which
        # sizes the mod-one interval) while the scheme secret stays dense.
        ct_down = _switch_secret(ct_down, dense_to_sparse_key)
        ct_up = ct_down._mod_raise(secret_l1_norm_bound=h).replace(_values=None)
        ct_up = _switch_secret(ct_up, sparse_to_dense_key)
    else:
        ct_up = ct_down._mod_raise(secret_l1_norm_bound=keys.secret_l1_norm_bound).replace(_values=None)

    ct_out = _bootstrap_once(ct_up, iteration.stc_scales, keys=keys, mod_one_range=mod_one_range)
    ct_out *= iteration.output_factor
    # STC and the output factor apply the restore factor; tracked slot values must follow it.
    if plaintext_values is not None:
        ct_out = ct_out.replace(_values=scale_slots(plaintext_values, restore_factor, prec=prec))
    return ct_out


@function
def _bootstrap_once(ct_up: Ciphertext, stc_scales: tuple[BigFloat, ...], *, keys: CKKSKeySet, mod_one_range: int) -> Ciphertext:
    """
    Bootstrap a raised ciphertext through CTS, modular reduction, and STC.

    Iteration-specific input normalization and output restoration stay outside
    this function so compatible passes can reuse its schedule.
    """
    params = keys.params
    boot_params = params.boot_params
    assert boot_params is not None
    # After mod-raise the ciphertext encrypts m + q_0*t.  The CTS puts the
    # coefficients (divided by scales[highest]) into slots, so the t-terms
    # appear as t * q_0 / scales[highest].  For the mod-one to remove them
    # cleanly as integer multiples of 1/K we need that ratio to be *exactly*
    # 1.  When q_0 (product of bootstrap-base moduli) doesn't perfectly match
    # scales[highest] — which happens for uint32 "almost-prime" moduli — the
    # residual error overwhelms the tiny signal.  Correcting the CTS scale
    # by scales[highest]/q_0 aligns the t-terms; the inverse factor is
    # absorbed into the STC scale so the overall product is unchanged.
    cts_scale = _cts_scale(params, mod_one_range=mod_one_range)
    ct1, ct2 = coefficients_to_slots(ct_up, levels=boot_params.cts_depth, scale=cts_scale)

    @function
    def mod_reduce(ct: Ciphertext, keys: CKKSKeySet) -> Ciphertext:
        boot_params = keys.params.boot_params
        assert boot_params is not None
        return mod_r_div_r(ct, r=1 / mod_one_range, exp_degree=boot_params.exp_degree, squaring_count=boot_params.squaring_count)

    # Matching input domains let the second modular reduction reuse the first schedule.
    ct1 = mod_reduce(ct1.map_polys(Poly.ntt), keys)
    ct2 = mod_reduce(ct2.map_polys(Poly.ntt), keys)
    stc_matrices = l_inv_matrix_partial_factorization(params.logn, boot_params.stc_depth)
    return _apply_matrix_stages(ct1 + ct2.times_i(), stc_matrices, scales=stc_scales)


def bootstrap(ct_in: Ciphertext, log_bound: int) -> Ciphertext:
    """
    Bootstrap a scheduled ciphertext with nominal message-magnitude bound
    ``2**log_bound``, allowing the configured bootstrap input margin above it.
    """
    ctx = context()
    mod_one_range, iterations = _bootstrap_plan(ctx.keyset, log_bound)
    ct_boot = _bootstrap_iteration(ct_in, iterations[0], keys=ctx.keyset, mod_one_range=mod_one_range)
    if len(iterations) == 1:
        return ct_boot

    plaintext_values = ct_boot._values
    for iteration in iterations[1:]:
        ct_err = ct_in - ct_boot
        # Plaintext tracking records intended values, not the encrypted residual.
        ct_boot += _bootstrap_iteration(ct_err.replace(_values=None), iteration, keys=ctx.keyset, mod_one_range=mod_one_range)
    return ct_boot.replace(_values=plaintext_values)


def _precompute_bootstrap_polys(log_bound: int) -> list[Poly]:
    """Warm the caches for bootstrap transform plans and return all Polys for realize()."""
    ctx = context()
    params = ctx.params
    boot_params = params.boot_params
    if boot_params is None:
        raise RuntimeError("Bootstrapping not allowed with current parameter set")
    mod_one_range, iterations = _bootstrap_plan(ctx.keyset, log_bound)

    cts_scale = _cts_scale(params, mod_one_range=mod_one_range)
    (cts_start_level, cts_matrices), (stc_start_level, stc_matrices) = boot_params.matrix_stages(params)
    transforms = list(_stage_transforms(cts_matrices, scales=_stage_scales(cts_scale / params.n, len(cts_matrices)), start_level=cts_start_level))

    for iteration in iterations:
        transforms += _stage_transforms(stc_matrices, scales=iteration.stc_scales, start_level=stc_start_level)

    polys = [poly for transform in transforms for poly in transform.polys()]
    return list({poly.node: poly for poly in polys}.values())
