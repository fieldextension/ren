from unittest.mock import Mock
from typing import assert_type

import pytest

pytest.importorskip("cuda.bindings")

from ren.allocator import BufferSpec
from ren.backends.cuda import api as cuda
from ren.backends.cuda.allocator import CUDAAllocator
from ren.backends.cuda.backend import CUDABackend
from ren.dtype import dtypes


@pytest.mark.requires_cuda
def test_backend_allocator(cuda_device: CUDABackend):
    allocator = assert_type(cuda_device.allocator, CUDAAllocator)
    assert allocator.backend is cuda_device


@pytest.fixture
def allocator() -> CUDAAllocator:
    allocator = CUDAAllocator.__new__(CUDAAllocator)
    allocator.backend = CUDABackend.__new__(CUDABackend)
    allocator.backend.context = object()
    allocator._cache = {}
    return allocator


def test_cache_hit_does_not_set_cuda_context(monkeypatch, allocator):
    cached = 64
    spec = BufferSpec(dtype=dtypes.uint8, size=64, device="cuda:0")
    allocator.free(cached, spec)
    set_current = Mock()
    monkeypatch.setattr(cuda, "ctx_set_current", set_current)

    assert allocator.alloc(spec) == cached
    set_current.assert_not_called()


def test_cache_reuse_order_is_independent_of_release_order(allocator):
    spec = BufferSpec(dtype=dtypes.uint8, size=64, device="cuda:0")
    for address in (64, 128, 192):
        allocator.free(address, spec)
    first = [allocator.alloc(spec) for _ in range(3)]
    for address in first:
        allocator.free(address, spec)
    second = [allocator.alloc(spec) for _ in range(3)]
    assert second == first
    for address in reversed(second):
        allocator.free(address, spec)
    assert [allocator.alloc(spec) for _ in range(3)] == first


def test_cache_miss_allocates_in_cuda_context(monkeypatch, allocator):
    allocated = object()
    spec = BufferSpec(dtype=dtypes.uint8, size=64, device="cuda:0")
    set_current = Mock()
    mem_alloc = Mock(return_value=allocated)
    monkeypatch.setattr(cuda, "ctx_set_current", set_current)
    monkeypatch.setattr(cuda, "mem_alloc", mem_alloc)

    assert allocator.alloc(spec) is allocated
    set_current.assert_called_once_with(allocator.backend.context)
    mem_alloc.assert_called_once_with(64)


def test_nolru_allocation_bypasses_cache(monkeypatch, allocator):
    cached = 64
    allocated = object()
    spec = BufferSpec(dtype=dtypes.uint8, size=64, device="cuda:0", nolru=True)
    cached_spec = BufferSpec(dtype=dtypes.uint8, size=64, device="cuda:0")
    allocator.free(cached, cached_spec)
    set_current = Mock()
    mem_alloc = Mock(return_value=allocated)
    monkeypatch.setattr(cuda, "ctx_set_current", set_current)
    monkeypatch.setattr(cuda, "mem_alloc", mem_alloc)

    assert allocator.alloc(spec) is allocated
    assert allocator.alloc(cached_spec) == cached
    set_current.assert_called_once_with(allocator.backend.context)
    mem_alloc.assert_called_once_with(64)


def test_allocation_failure_frees_cache_and_retries(monkeypatch, allocator):
    cached = 32
    allocated = object()
    spec = BufferSpec(dtype=dtypes.uint8, size=64, device="cuda:0")
    allocator.free(cached, BufferSpec(dtype=dtypes.uint8, size=32, device="cuda:0"))
    mem_alloc = Mock(side_effect=[MemoryError, allocated])
    mem_free = Mock()
    monkeypatch.setattr(cuda, "ctx_set_current", Mock())
    monkeypatch.setattr(cuda, "mem_alloc", mem_alloc)
    monkeypatch.setattr(cuda, "mem_free", mem_free)

    assert allocator.alloc(spec) is allocated
    assert mem_alloc.call_count == 2
    mem_free.assert_called_once_with(cached)
