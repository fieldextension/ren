from functools import cache
from math import prod

from ren.bigfloat import BigFloat
from ren.device import Device
from ren.dtype import DType, dtypes
from ren.ckks import BootstrappingParams, CKKSLevel, CKKSParams
from ren.ckks.utils import select_moduli


def high_precision_params() -> CKKSParams:
    """Six computation levels with a 90-bit scale and iterative bootstrapping."""
    bootstrap = BootstrappingParams(
        cts_depth=3,
        stc_depth=3,
        exp_degree=63,
        squaring_count=3,
        failure_probability=2**-64,
        sparse_encapsulation_weight=32,
        precision_gains=(10, 12, 12, 12, 12),
    )
    return CKKSParams.default(depth=6, log_precision=90, dtype=dtypes.uint32, p_prime_count=4, boot_params=bootstrap)


@cache
def small_params(*, dtype: DType = dtypes.uint32) -> CKKSParams:
    """Small, insecure CKKS parameters for tests."""
    return CKKSParams.generate_prefix(log_n=10, depth=5, log_precision=50, log_range=10, dtype=dtype, security=None)


def get_test_params() -> CKKSParams:
    if Device.is_cuda(Device.DEFAULT):
        return CKKSParams.default(boot_params=None)
    return small_params()


@cache
def non_prefix_params() -> CKKSParams:
    """Small, insecure parameters with overlapping bases and unequal bootstrap depths."""
    n = 1024
    boot_params = BootstrappingParams(
        cts_depth=3, stc_depth=2, exp_degree=24, squaring_count=3, failure_probability=2**-16, sparse_encapsulation_weight=32
    )
    level_count = 6 + boot_params.depth
    moduli = select_moduli(n=n, num_primes=2 * level_count, bit_length=30)
    # Levels 1 and 3 drop the first prime and add one at the other end.
    bases = [moduli[1 : 2 * level + 3] if level in (1, 3) else moduli[: 2 * level + 2] for level in range(level_count)]
    # The first bootstrap stages drop primes from the left of the highest basis.
    bases[-3:] = [moduli[4:], moduli[2:], moduli]
    scales = [BigFloat(2) ** 40]
    for lower, upper in zip(bases, bases[1:]):
        scales.append((scales[-1] * prod(upper) / prod(lower)).sqrt())
    ks_primes = select_moduli(n=n, num_primes=4, bit_length=31, exclude=set(moduli))
    return CKKSParams.from_levels(
        n=n,
        dtype=dtypes.uint32,
        levels=[CKKSLevel(moduli=basis, scale=scale) for basis, scale in zip(bases, scales)],
        ks_primes=ks_primes,
        ks_digit_sizes=tuple(min(4, len(moduli) - start) for start in range(0, len(moduli), 4)),
        log_range=0,
        boot_params=boot_params,
        bootstrap_base_level=CKKSLevel(moduli=bases[0], scale=BigFloat(prod(bases[0])) / 32),
        security=None,
    )
