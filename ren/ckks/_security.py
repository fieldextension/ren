"""Modulus limits for individual RLWE keys at 128 bits of estimated classical security."""

from math import inf

from ren.polydist import MINIMUM_ENCRYPTION_NOISE_STD_DEV, DiscreteGaussian, KeyDist, PolyDistWithBounds, SparseTernary, Ternary

_MAX_LOG2_MODULUS: dict[KeyDist, int] = {
    # Ren's dense secret is the difference of two binary polynomials. Starting with
    # a binary secret s, subtract a*r from each sample b = a*s + e using the same
    # independent binary polynomial r. The new secret s-r has Ren's distribution
    # and the error stays the same, so the binary-secret limits apply.
    #
    # Degrees 2^12 through 2^17 use the published binary column for Gaussian
    # error with sigma=3.2 and unlimited samples. At degrees 2^10 and 2^11,
    # the same seven classical MATZOV/GSA models gave minimum costs of 129.84
    # and 128.06 bits at the respective caps (rounded down).
    # Security Guidelines for Implementing Homomorphic Encryption, §4.2: https://cic.iacr.org/p/1/4/26
    # https://github.com/jdumezy/rlwe-params/blob/51ac1c2b0e1130d42eccd3de5ef491312f347c15/tables/128bits_security.md
    # https://github.com/malb/lattice-estimator/tree/7d59f866655362ea8eab803dbc833710c2bc362a
    Ternary(1024): 25,
    Ternary(2048): 52,
    Ternary(4096): 105,
    Ternary(8192): 213,
    Ternary(16384): 430,
    Ternary(32768): 867,
    Ternary(65536): 1746,
    Ternary(131072): 3521,
    # Ren chooses h nonzero positions and samples each sign independently
    # (SparseBinomial(1, h)). The inline estimates give the lowest finite cost
    # in bits at each cap across the standard estimator and Ogilvie's four
    # ring-aware primal models, using classical MATZOV/GSA and unlimited samples.
    # For dual models that require fixed sign counts, we checked 0 through h positives.
    # https://doi.org/10.1007/978-3-032-35377-1_15
    # https://github.com/TabOg/mlwe-hybrids/tree/a92f335665eef69927d2d8f49832bd595a6dac32
    SparseTernary(n=32768, h=32): 91,  # 143.5 bits
    SparseTernary(n=32768, h=64): 121,  # 214.4 bits
    SparseTernary(n=65536, h=32): 121,  # 157.7 bits
    SparseTernary(n=65536, h=64): 121,  # 260.8 bits
    SparseTernary(n=65536, h=192): 1475,  # 128.1 bits
    SparseTernary(n=131072, h=32): 121,  # 180.5 bits
    SparseTernary(n=131072, h=64): 121,  # 307.1 bits
}


def max_log2_modulus(secret: KeyDist, error: PolyDistWithBounds, security: int) -> int:
    """Return the upper bound on log2(q) for this secret and error distribution."""
    if security != 128:
        raise ValueError("only 128-bit security is currently supported")
    # The table uses the narrowest supported Gaussian, the easiest case for the modeled attacks.
    if isinstance(error, DiscreteGaussian) and secret.n == error.n and MINIMUM_ENCRYPTION_NOISE_STD_DEV <= error.stdev < inf:
        if (limit := _MAX_LOG2_MODULUS.get(secret)) is not None:
            return limit
    raise ValueError(f"no {security}-bit security estimate for {secret} with noise {error}")


def validate_modulus_security(modulus: int, *, secret: KeyDist, error: PolyDistWithBounds, security: int) -> None:
    limit = max_log2_modulus(secret, error, security)
    if modulus > 1 << limit:
        raise ValueError(f"key modulus exceeds the {security}-bit security estimate for {secret} (maximum modulus 2^{limit})")
