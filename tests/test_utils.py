from contextlib import nullcontext
import gc
import random

import pytest

from ren.utils import disable_gc
from tests.utils import neg_convolve, neg_convolve_naive


@pytest.mark.parametrize("n", [4, 16, 32])
@pytest.mark.parametrize("q", [17, 17 * 97, (1 << 127) - 1])
def test_neg_convolve_matches_naive(n, q):
    rng = random.Random(42)
    for left, right in (
        ([], []),
        ([0] * n, [1] * n),
        ([0] * (n - 1) + [1], [0, 1]),
        ([q + 1, -q - 2], [q - 1, -1]),
        ([rng.randrange(-q, q) for _ in range(n)], [rng.randrange(-q, q) for _ in range(n)]),
    ):
        assert neg_convolve(left, right, n=n, q=q) == neg_convolve_naive(left, right, n=n, q=q)


@pytest.mark.parametrize("left,right", [([1] * 5, [1]), ([1], [1] * 5)])
def test_neg_convolve_rejects_inputs_longer_than_ring_degree(left, right):
    with pytest.raises(AssertionError, match="inputs must have at most n coefficients"):
        neg_convolve(left, right, n=4, q=17)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_disable_gc_recursive_decorator(enabled, fail):
    @disable_gc()
    def nested(depth):
        assert not gc.isenabled()
        if depth:
            nested(depth - 1)
            assert not gc.isenabled()
        elif fail:
            raise RuntimeError("nested failure")

    was_enabled = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        with pytest.raises(RuntimeError, match="nested failure") if fail else nullcontext():
            nested(2)
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if was_enabled else gc.disable)()
