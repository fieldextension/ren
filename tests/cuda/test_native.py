from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.requires_cuda
KERNELS = Path(__file__).resolve().parents[2] / "ren/backends/cuda/kernels"


@pytest.fixture(scope="module")
def nvcc() -> list[str]:
    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.fail("Install the CUDA Toolkit and put nvcc on PATH to run the native CUDA tests.", pytrace=False)
    return [
        compiler,
        "-std=c++17",
        "-x",
        "cu",
        f"-I{KERNELS}",
        "-Xcompiler=-Wall,-Wextra,-Wshadow,-Wconversion",
        "-diag-error=549",
        "-diag-error=940",
    ]


def test_compile_kernels(nvcc, tmp_path):
    subprocess.run([*nvcc, "-c", "-DNUM_SRC=1", "-DNUM_DST=1", *sorted(KERNELS.glob("*.cu"))], cwd=tmp_path, check=True)


@pytest.mark.parametrize("flags", [("-g", "-O0", "-G"), ("-O3", "-lineinfo")], ids=["debug", "release"])
def test_common(nvcc, flags, tmp_path):
    binary = tmp_path / "test_common"
    subprocess.run([*nvcc, *flags, Path(__file__).with_name("test_common.cpp"), "-o", binary], cwd=tmp_path, check=True)
    subprocess.run([binary], cwd=tmp_path, check=True)
