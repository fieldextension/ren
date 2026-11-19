import pytest
import random
from ren.random import set_global_rng_seed
from ren.device import Device


def _cuda_device_count() -> int:
    """Get number of available CUDA devices."""
    try:
        from ren.backends.cuda import api as cuda

        cuda.init(0)
        return cuda.device_get_count()
    except ImportError, OSError, RuntimeError:
        return 0


_CUDA_SELECTED = Device.is_cuda(Device.DEFAULT)
_CUDA_DEVICE_COUNT = _cuda_device_count() if _CUDA_SELECTED else 0


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and hardware availability."""
    skip_dev = pytest.mark.skip(reason="requires the CUDA backend")
    skip_cuda = pytest.mark.skip(reason="CUDA device not available")
    skip_multi_gpu = pytest.mark.skip(reason="requires at least 2 GPUs")

    for item in items:
        requires_cuda = item.get_closest_marker("requires_cuda") is not None
        requires_multi_gpu = item.get_closest_marker("requires_multi_gpu") is not None
        if requires_cuda or requires_multi_gpu:
            if not _CUDA_SELECTED:
                item.add_marker(skip_dev)
            elif not _CUDA_DEVICE_COUNT:
                item.add_marker(skip_cuda)
            elif requires_multi_gpu and _CUDA_DEVICE_COUNT < 2:
                item.add_marker(skip_multi_gpu)


@pytest.fixture(scope="session")
def cuda_device():
    """Return the selected CUDA backend, skipping if unavailable."""
    if not _CUDA_DEVICE_COUNT:
        pytest.skip("CUDA device not available")
    return Device[Device.DEFAULT]


@pytest.fixture(scope="session")
def multi_gpu():
    """Fixture that provides all available CUDA device names, skipping if <2 GPUs.

    Use this when you need actual device names for multi-GPU tests.
    For tests that just need 2+ GPUs available, use @pytest.mark.requires_multi_gpu instead.
    """
    if _CUDA_DEVICE_COUNT < 2:
        pytest.skip("requires at least 2 GPUs")
    return tuple(f"cuda:{i}" for i in range(_CUDA_DEVICE_COUNT))


@pytest.fixture(autouse=True)
def set_random_seed():
    seed_number = 42
    random.seed(seed_number)
    set_global_rng_seed(seed_number)
