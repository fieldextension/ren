import numpy as np

import ren.montgomery as mont
from ren.dtype import to_np_dtype
from ren.ring import RingSpec


def _encoded_rns_coeffs(data: list[int] | np.ndarray, ring: RingSpec) -> np.ndarray:
    n = ring.n
    rns_coeffs = np.empty(n * len(ring.moduli), dtype=to_np_dtype(ring.dtype))
    for i, modulus in enumerate(ring.moduli):
        start = i * n
        rns_coeffs[start : start + n] = mont.context(ring.dtype, modulus).encode(data)
    return rns_coeffs
