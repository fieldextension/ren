import asyncio
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import assert_type
from unittest.mock import Mock

import pytest

from ren.allocator import Allocator
from ren.backend import Backend
from ren.backends.python import PythonAllocator, PythonBackend
import ren.backends.cuda
from ren.context import Context, DEV
from ren.device import CUDADevice, Device, _DeviceRegistry


@pytest.mark.parametrize("available, expected", [(True, "cuda:0"), (False, "python")])
@pytest.mark.parametrize("debug", [0, 1])
def test_default_selection(monkeypatch, capsys, available, expected, debug):
    probe = Mock(return_value=available)
    monkeypatch.setattr(ren.backends.cuda, "api", SimpleNamespace(is_available=probe), raising=False)
    devices = _DeviceRegistry()
    with Context(DEV="", DEBUG=debug):
        assert devices.canonicalize("cuda:3") == "cuda:3"
        with devices.scope("python"):
            assert devices.DEFAULT == "python"
        for override in ("python", "cuda:3"):
            with Context(DEV=override):
                assert devices.DEFAULT == override
        with Context(DEV="cudaa"), pytest.raises(ValueError, match="Invalid device"):
            devices.DEFAULT
        probe.assert_not_called()
        assert capsys.readouterr() == ("", "")
        assert devices.DEFAULT == expected
        with Context(DEV="python"):
            assert devices.DEFAULT == "python"
        assert devices.DEFAULT == expected
    probe.assert_called_once_with()
    assert capsys.readouterr() == ("CUDA unavailable; using python\n" if debug and not available else "", "")


def test_detection_without_cuda_bindings():
    code = "import sys; sys.modules['cuda.bindings'] = None; from ren.device import Device; assert Device.DEFAULT == 'python'"
    result = subprocess.run(
        [sys.executable, "-c", code], env=os.environ | {"DEV": "", "DEBUG": "1"}, capture_output=True, text=True, check=True, timeout=30
    )
    assert result.stdout == "CUDA unavailable; using python\n"
    assert result.stderr == ""


def test_detection_propagates_missing_dependency(tmp_path):
    (tmp_path / "cuda.py").write_text("import missing_cuda_dependency\n")
    result = subprocess.run(
        [sys.executable, "-c", "from ren.device import Device; Device.DEFAULT"],
        cwd=tmp_path,
        env=os.environ | {"DEV": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError: No module named 'missing_cuda_dependency'" in result.stderr


def test_detection_propagates_probe_errors(monkeypatch):
    probe = Mock(side_effect=RuntimeError("broken driver"))
    monkeypatch.setattr(ren.backends.cuda, "api", SimpleNamespace(is_available=probe), raising=False)
    with Context(DEV=""), pytest.raises(RuntimeError, match="broken driver"):
        _DeviceRegistry().DEFAULT


@pytest.mark.parametrize("override", ["", "cuda:3"])
def test_cuda_initialization_errors_do_not_fall_back(monkeypatch, override):
    monkeypatch.setattr(ren.backends.cuda, "api", SimpleNamespace(is_available=lambda: True), raising=False)
    backend = SimpleNamespace(CUDABackend=Mock(side_effect=RuntimeError("initialization failed")))
    monkeypatch.setitem(sys.modules, "ren.backends.cuda.backend", backend)
    with Context(DEV=override), pytest.raises(RuntimeError, match="initialization failed"):
        _DeviceRegistry().default


@pytest.mark.parametrize("name", ["metal", "cudaa", "cuda:-1", "cuda:abc", "cuda:1:2", "python:0", "cuda: 1", " "])
def test_invalid_device_names(name):
    with pytest.raises(ValueError, match="Invalid device"):
        Device.canonicalize(name)
    assert not Device.is_cuda(name)


@pytest.mark.parametrize(
    "name, expected", [("cuda", "cuda:0"), ("cuda:", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:12", "cuda:12"), (" CUDA:003 ", "cuda:3")]
)
def test_cuda_device_names(name: str, expected: str):
    assert Device.is_cuda(name)
    assert_type(name, CUDADevice)
    assert Device.canonicalize(name) == expected


@pytest.mark.parametrize("name", ["cpu", "python", " PYTHON ", "metal:0", "unknown"])
def test_other_device_names(name: str):
    assert not Device.is_cuda(name)


def test_device_scope_overrides_and_restores_default():
    backend = assert_type(Device["cpu"], Backend)
    assert_type(backend.allocator, Allocator)
    assert backend is Device["python"]
    assert isinstance(backend, PythonBackend)
    assert_type(backend.allocator, PythonAllocator)
    with Context(DEV="python"):
        assert Device.DEFAULT == "python"
        with Device.scope("CUDA:3"):
            assert Device.DEFAULT == "cuda:3"
            assert Device.canonicalize(None) == "cuda:3"
            assert Device.canonicalize("python") == "python"
            with Device.scope("cpu"):
                assert Device.DEFAULT == "python"
            assert Device.DEFAULT == "cuda:3"
        assert Device.DEFAULT == "python"


def test_device_scope_restores_default_after_exception():
    with Context(DEV="python"):
        with pytest.raises(RuntimeError, match="boom"):
            with Device.scope("cuda:3"):
                raise RuntimeError("boom")
        assert Device.DEFAULT == "python"


def test_device_scope_is_task_local():
    async def scoped(device: str) -> str:
        with Device.scope(device):
            await asyncio.sleep(0)
            return Device.DEFAULT

    async def run() -> tuple[str, ...]:
        return tuple(await asyncio.gather(scoped("python"), scoped("cuda:3")))

    assert asyncio.run(run()) == ("python", "cuda:3")


def test_device_scope_is_thread_local():
    with Context(DEV="python"), Device.scope("cuda:3"), ThreadPoolExecutor(max_workers=1) as executor:
        assert Device.DEFAULT == "cuda:3"
        assert executor.submit(lambda: Device.DEFAULT).result() == "python"
        assert DEV.value == "python"
