import ctypes
import numpy as np
import pytest

pytest.importorskip("cuda.bindings")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from ren.backends.cuda import api as cuda_api
from ren.backends.cuda.compiler import CUDACompiler
from ren.backends.cuda.kernels import get_headers, load_kernel
from tests.utils import rand_int_arr


def _run_ren_random_chacha(key: np.ndarray, stream_id: int, n_blocks: int) -> bytes:
    compiler = CUDACompiler(cuda_api.get_arch(0), headers=get_headers())
    cubin, lowered_name = compiler.compile(load_kernel("rand.cu"), "ren_random_chacha")
    module = cuda_api.module_load_data(cubin)
    func = cuda_api.module_get_function(module, lowered_name)

    output_nbytes = n_blocks * 64
    output_dev = cuda_api.mem_alloc(output_nbytes)
    key_dev = cuda_api.mem_alloc(32)
    stream = cuda_api.stream_create()
    try:
        cuda_api.memcpy_htod_async(key_dev, key.astype("<u4").tobytes(), 32, stream)

        block = 64
        grid = ((n_blocks + block - 1) // block,)
        args = [
            ctypes.c_void_p(int(output_dev)),
            ctypes.c_void_p(int(key_dev)),
            ctypes.c_ulonglong(stream_id),
            ctypes.c_ulonglong(n_blocks),
        ]
        cuda_api.launch_kernel(func, grid, (block,), args, stream)

        host_buf = bytearray(output_nbytes)
        cuda_api.memcpy_dtoh_async(memoryview(host_buf), output_dev, output_nbytes, stream)
        cuda_api.stream_synchronize(stream)
    finally:
        cuda_api.stream_destroy(stream)
        cuda_api.mem_free(output_dev)
        cuda_api.mem_free(key_dev)

    return bytes(host_buf)


def _cryptography_keystream(key: np.ndarray, stream_id: int, n_blocks: int) -> bytes:
    nonce = stream_id.to_bytes(8, "little") + b"\x00" * 4
    iv = b"\x00" * 4 + nonce
    return Cipher(algorithms.ChaCha20(key.astype("<u4").tobytes(), iv), mode=None).encryptor().update(b"\x00" * (n_blocks * 64))


@pytest.mark.requires_cuda
def test_ren_random_chacha_matches_cryptography(cuda_device):
    cuda_api.ctx_set_current(cuda_device.context)

    key = np.array(rand_int_arr(size=8, bound=1 << 32), dtype=np.uint32)
    stream_id = rand_int_arr(size=1, bound=1 << 64)[0]

    n_blocks = 64
    got = _run_ren_random_chacha(key, stream_id, n_blocks)
    expected = _cryptography_keystream(key, stream_id, n_blocks)
    assert got == expected


@pytest.mark.requires_cuda
def test_ren_random_chacha_uses_all_stream_id_bits(cuda_device):
    cuda_api.ctx_set_current(cuda_device.context)

    key = np.array(rand_int_arr(size=8, bound=1 << 32), dtype=np.uint32)
    stream_id = 0xFFFFFFFFFFFFFFFF

    n_blocks = 64
    got = _run_ren_random_chacha(key, stream_id, n_blocks)
    expected = _cryptography_keystream(key, stream_id, n_blocks)
    assert got == expected
