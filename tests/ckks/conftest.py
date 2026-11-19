import pytest

from ren.random import set_global_rng_seed
from ren.ckks import CKKSKeySet, CKKSParams, keygen
from tests.ckks.params import get_test_params


@pytest.fixture(scope="module")
def test_params() -> CKKSParams:
    return get_test_params()


@pytest.fixture(scope="module")
def keys(test_params: CKKSParams) -> CKKSKeySet:
    set_global_rng_seed(42)
    return keygen(test_params).realize()
