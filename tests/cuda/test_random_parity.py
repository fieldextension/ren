from fractions import Fraction

import pytest

pytest.importorskip("cuda.bindings")
pytest.importorskip("cryptography")

from ren.dtype import dtypes
from ren.device import Device
from ren.poly import Poly
from ren.polydist import DiscreteGaussian, SparseTernary, Ternary, UniformBounded, UniformModQ
from ren.random import ChaChaRng
from ren.ckks import Ciphertext, CKKSKeySet, CKKSLevel, CKKSParams
from ren.ckks.utils import select_moduli
from tests.utils import mixed_prime_ring


@pytest.mark.requires_cuda
@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize(
    "dist",
    [
        Ternary(64),
        SparseTernary(64, 16),
        UniformBounded(64, 10),
        UniformBounded(64, 100),
        UniformModQ(64),
        DiscreteGaussian(64, Fraction(16, 5)),
        DiscreteGaussian(64, Fraction(1 << 70)),
        DiscreteGaussian(64, Fraction(1 << 121)),
    ],
    ids=[
        "ternary",
        "sparse-ternary",
        "uniform-bounded-lookup",
        "uniform-bounded-encoded",
        "uniform-mod-q",
        "gaussian",
        "large-gaussian",
        "gaussian-limit",
    ],
)
def test_random_polynomial_matches_python(cuda_device, dtype, dist):
    ring = mixed_prime_ring(dist.n, dtype)
    seed = bytes(range(ChaChaRng.seed_size))

    expected = Poly.rand(dist, ring, rng=ChaChaRng("python", seed).derive("parity"))
    actual = Poly.rand(dist, ring, rng=ChaChaRng(cuda_device.device, seed).derive("parity"))

    assert list(actual) == list(expected)


@pytest.mark.requires_cuda
def test_default_rng_must_match_backend_device(cuda_device):
    seed = bytes(range(ChaChaRng.seed_size))

    with pytest.raises(ValueError, match="RNG device mismatch"):
        cuda_device.set_default_rng(ChaChaRng("python", seed))
    with pytest.raises(ValueError, match="RNG device mismatch"):
        Device["python"].set_default_rng(ChaChaRng(cuda_device.device, seed))


@pytest.mark.requires_cuda
def test_uniform_bounded_rejects_values_that_overflow_signed_cuda_arithmetic(cuda_device):
    ring = mixed_prime_ring(64, dtype=dtypes.uint32)
    poly = Poly.rand(UniformBounded(ring.n, 1 << 62), ring, rng=ChaChaRng(cuda_device.device, bytes(range(ChaChaRng.seed_size))))

    with pytest.raises(ValueError, match="CUDA uniform bound must be less than 2\\^62"):
        poly.realize()


@pytest.mark.requires_cuda
def test_discrete_gaussian_rejects_coefficients_outside_signed_int128():
    ring = mixed_prime_ring(64)
    dist = DiscreteGaussian(ring.n, Fraction(1 << 122))
    with pytest.raises(ValueError, match="Gaussian.*128-bit"):
        Poly.rand(dist, ring).realize()


@pytest.mark.requires_cuda
def test_public_key_and_rerandomization_match_python(cuda_device):
    ring = mixed_prime_ring(64, dtype=dtypes.uint32)
    highest_moduli = ring.moduli[:4]
    params = CKKSParams.from_levels(
        n=ring.n,
        dtype=ring.dtype,
        levels=(CKKSLevel(moduli=highest_moduli, scale=1 << 20),),
        ks_primes=select_moduli(n=ring.n, num_primes=2, bit_length=29, exclude=set(highest_moduli)),
        ks_digit_sizes=(2, 2),
        log_range=0,
        security=None,
    )
    seed = bytes(range(ChaChaRng.seed_size))
    python_keys = CKKSKeySet.generate(params, rng=ChaChaRng("python", seed)).realize()
    cuda_keys = CKKSKeySet.generate(params, rng=ChaChaRng(cuda_device.device, seed)).realize()

    assert python_keys.pk is not None
    assert cuda_keys.pk is not None
    assert list(python_keys.pk.ct.a) == list(cuda_keys.pk.ct.a)
    assert list(python_keys.pk.ct.b) == list(cuda_keys.pk.ct.b)

    rerandomization_seed = bytes(reversed(range(ChaChaRng.seed_size)))
    with python_keys.context():
        ct = Ciphertext.zero(level=0)
        python_ct = ct.rerandomize(rng=ChaChaRng("python", rerandomization_seed))
    with cuda_keys.context():
        cuda_ct = ct.to(cuda_device.device).rerandomize(rng=ChaChaRng(cuda_device.device, rerandomization_seed))

    assert list(python_ct.a) == list(cuda_ct.a)
    assert list(python_ct.b) == list(cuda_ct.b)
