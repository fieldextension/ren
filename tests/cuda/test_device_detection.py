from unittest.mock import Mock

import pytest

pytest.importorskip("cuda.bindings")

from ren.backends.cuda import api as cuda
from ren.backends.cuda.backend import CUDABackend
from ren.backends.cuda.compiler import CUDACompiler
from ren.context import Context
from ren.device import _DeviceRegistry


@pytest.fixture(autouse=True)
def clear_nvrtc_version_cache():
    cuda.nvrtc_version.cache_clear()
    yield
    cuda.nvrtc_version.cache_clear()


@pytest.mark.parametrize("count, expected", [(0, False), (1, True), (4, True)])
def test_available_devices(monkeypatch, count, expected):
    monkeypatch.setattr(cuda._drv, "cuInit", lambda flags: (cuda._drv.CUresult.CUDA_SUCCESS,))
    monkeypatch.setattr(cuda, "device_get_count", lambda: count)
    version = Mock(return_value=(cuda._nvrtc.nvrtcResult.NVRTC_SUCCESS, 12, 9))
    monkeypatch.setattr(cuda._nvrtc, "nvrtcVersion", version)
    assert cuda.is_available() is expected
    if expected:
        CUDACompiler("sm_80")
    assert version.call_count == int(expected)


def test_missing_nvrtc(monkeypatch, capsys):
    monkeypatch.setattr(cuda, "init", lambda: None)
    monkeypatch.setattr(cuda, "device_get_count", lambda: 1)
    version = Mock(side_effect=cuda.DynamicLibNotFoundError("missing NVRTC"))
    monkeypatch.setattr(cuda._nvrtc, "nvrtcVersion", version)
    with Context(DEBUG=1):
        assert not cuda.is_available()
        with pytest.raises(cuda.DynamicLibNotFoundError, match="missing NVRTC"):
            CUDACompiler("sm_80")
        version.side_effect = None
        version.return_value = (cuda._nvrtc.nvrtcResult.NVRTC_SUCCESS, 12, 9)
        assert cuda.is_available()
    assert capsys.readouterr() == ("", "")


def test_explicit_cuda_requires_nvrtc_before_creating_context(monkeypatch):
    monkeypatch.setattr(cuda, "init", lambda flags=0: None)
    monkeypatch.setattr(cuda, "device_get", lambda index: index)
    monkeypatch.setattr(cuda, "get_arch", lambda index: "sm_80")
    context = Mock(side_effect=AssertionError("created a context before checking NVRTC"))
    monkeypatch.setattr(cuda, "ctx_create", context)
    monkeypatch.setattr(cuda._nvrtc, "nvrtcVersion", Mock(side_effect=cuda.DynamicLibNotFoundError("missing NVRTC")))
    devices = _DeviceRegistry()
    opened = CUDABackend._all_backends.copy()
    with Context(DEV="cuda"), pytest.raises(cuda.DynamicLibNotFoundError, match="missing NVRTC"):
        devices.default
    context.assert_not_called()
    assert devices._backends == {} and CUDABackend._all_backends == opened


def test_nvrtc_error_propagates(monkeypatch):
    monkeypatch.setattr(cuda, "init", lambda: None)
    monkeypatch.setattr(cuda, "device_get_count", lambda: 1)
    monkeypatch.setattr(cuda._nvrtc, "nvrtcVersion", lambda: (cuda._nvrtc.nvrtcResult.NVRTC_ERROR_INTERNAL_ERROR, 0, 0))
    monkeypatch.setattr(cuda._nvrtc, "nvrtcGetErrorString", lambda code: b"internal error")
    with pytest.raises(RuntimeError, match="NVRTC error.*internal error"):
        cuda.is_available()
    with pytest.raises(RuntimeError, match="NVRTC error.*internal error"):
        CUDACompiler("sm_80")


@pytest.mark.parametrize("code", ["CUDA_ERROR_NO_DEVICE", "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"])
def test_driver_status(monkeypatch, code):
    status = getattr(cuda._drv.CUresult, code)
    monkeypatch.setattr(cuda._drv, "cuInit", lambda flags: (status,))
    monkeypatch.setattr(cuda, "_err_string", lambda code: code.name)
    if code == "CUDA_ERROR_NO_DEVICE":
        assert not cuda.is_available()
    else:
        with pytest.raises(cuda.CUDAError, match=code):
            cuda.is_available()


def test_missing_driver(monkeypatch):
    monkeypatch.setattr(cuda._drv, "cuInit", Mock(side_effect=cuda.DynamicLibNotFoundError("missing driver")))
    assert not cuda.is_available()


def test_unexpected_driver_error(monkeypatch):
    monkeypatch.setattr(cuda._drv, "cuInit", Mock(side_effect=RuntimeError("unexpected failure")))
    with pytest.raises(RuntimeError, match="unexpected failure"):
        cuda.is_available()
