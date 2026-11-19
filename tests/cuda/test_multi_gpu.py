import pytest

from ren import realize
from ren.poly import Poly
from ren.ir import Op
from ren.ckks import LinearTransform, Plaintext, SparseMatrix, encrypt, keygen
from tests.utils import large_prime_ring, rand_int_arr, neg_convolve
from tests.ckks.params import small_params


@pytest.mark.requires_cuda
class TestPolyTo:
    def test_to_same_device(self):
        """Copying to the same device returns the original Poly."""
        ring = large_prime_ring(256)
        p = Poly([1, 2, 3, 4] + [0] * 252, ring, device="cuda:0")
        p_same = p.to("cuda:0")
        assert p_same is p

    def test_to_different_device_ir(self, multi_gpu):
        ring = large_prime_ring(256)
        p = Poly([1, 2, 3, 4] + [0] * 252, ring, device=multi_gpu[0])
        p2 = p.to(multi_gpu[1])

        assert p.device == multi_gpu[0]
        assert p2.device == multi_gpu[1]

        n = p2.node
        assert n.op is Op.COPY

    def test_to_preserves_values(self, multi_gpu):
        ring = large_prime_ring(256)
        coeffs = rand_int_arr(size=256, bound=ring.q)
        p = Poly(coeffs, ring, device=multi_gpu[0])
        p2 = p.to(multi_gpu[1])
        realize(p, p2)

        p_python = p.to("python")
        p2_python = p2.to("python")

        assert list(p_python) == coeffs
        assert list(p2_python) == coeffs

    def test_python_to_cuda(self):
        ring = large_prime_ring(256)
        coeffs = [1, 2, 3, 4] + [0] * 252
        p = Poly(coeffs, ring, device="python")
        p_cuda = p.to("cuda:0")

        assert p.device == "python"
        assert p_cuda.device == "cuda:0"
        assert list(p_cuda) == coeffs

    def test_cuda_to_python(self):
        ring = large_prime_ring(256)
        coeffs = [1, 2, 3, 4] + [0] * 252
        p = Poly(coeffs, ring, device="cuda:0")
        p_python = p.to("python")

        assert p.device == "cuda:0"
        assert p_python.device == "python"
        assert list(p_python) == coeffs


@pytest.mark.requires_multi_gpu
def test_compile_loads_the_module_into_its_own_context(multi_gpu):
    """Compile with the other context current, then launch in the backend's own context."""
    from ren.backends.cuda import api as cuda
    from ren.backends.cuda.backend import CUDABackend
    from ren.backends.cuda.renderer import RenderedKernel
    from ren.device import Device

    backend, other = Device[multi_gpu[0]], Device[multi_gpu[1]]
    assert isinstance(backend, CUDABackend) and isinstance(other, CUDABackend)
    kernel = RenderedKernel("__attribute__((global)) void ren_test_noop() {}", "&ren_test_noop")

    cuda.ctx_set_current(other.context)
    func = backend._compile(kernel)

    cuda.ctx_set_current(backend.context)
    cuda.launch_kernel(func, (1,), (1,), [], backend.compute_queue.stream)
    backend.synchronize()


@pytest.mark.requires_multi_gpu
class TestMultiGPUExecution:
    def test_independent_ops_different_devices(self, multi_gpu):
        ring = large_prime_ring(256)
        coeffs1 = rand_int_arr(size=256, bound=ring.q)
        coeffs2 = rand_int_arr(size=256, bound=ring.q)

        p1 = Poly(coeffs1, ring, device=multi_gpu[0])
        p2 = Poly(coeffs2, ring, device=multi_gpu[1])

        r1 = p1 + p1
        r2 = p2 + p2
        realize(r1, r2)

        assert r1.device == multi_gpu[0]
        assert r2.device == multi_gpu[1]

        expected1 = [(2 * c) % ring.q for c in coeffs1]
        expected2 = [(2 * c) % ring.q for c in coeffs2]

        assert list(r1) == expected1
        assert list(r2) == expected2

    def test_ckks_keyset_copy_runs_on_target_device(self, multi_gpu):
        keys = keygen(small_params(), device=multi_gpu[0]).realize()
        ctx = keys.context()

        moved_keys = keys.to(multi_gpu[1]).realize()
        moved_ctx = moved_keys.context()
        assert moved_ctx.keyset.sk is not None
        assert ctx.keyset.sk is not None
        assert list(moved_ctx.keyset.sk.to("python")) == list(ctx.keyset.sk.to("python"))

        matrix = SparseMatrix.identity(ctx.params.slot_count)
        with ctx:
            source_transform = LinearTransform.encode(matrix, level=ctx.default_level)
        with moved_ctx:
            moved_transform = LinearTransform.encode(matrix, level=moved_ctx.default_level)
        realize(source_transform, moved_transform)

        assert {poly.device for poly in source_transform.polys()} == {multi_gpu[0]}
        assert {poly.device for poly in moved_transform.polys()} == {multi_gpu[1]}

        with moved_ctx:
            ct = encrypt(Plaintext.zero()).realize()

        assert ct.device == multi_gpu[1]

    def test_cross_device_dependency(self, multi_gpu):
        ring = large_prime_ring(256)
        coeffs = rand_int_arr(size=256, bound=ring.q)

        p = Poly(coeffs, ring, device=multi_gpu[0])
        p_dev1 = p.to(multi_gpu[1])
        result = p_dev1 + p_dev1
        result.realize()

        assert result.device == multi_gpu[1]
        expected = [(2 * c) % ring.q for c in coeffs]
        assert list(result) == expected

    def test_gather_back_to_single_device(self, multi_gpu):
        ring = large_prime_ring(256)
        coeffs1 = rand_int_arr(size=256, bound=ring.q)
        coeffs2 = rand_int_arr(size=256, bound=ring.q)

        p1 = Poly(coeffs1, ring, device=multi_gpu[0])
        p2 = Poly(coeffs2, ring, device=multi_gpu[1])

        r1 = p1.ntt()
        r2 = p2.ntt()
        r2_dev0 = r2.to(multi_gpu[0])
        combined = r1 * r2_dev0
        combined.realize()

        assert combined.device == multi_gpu[0]
        assert list(combined) == neg_convolve(coeffs1, coeffs2, n=ring.n, q=ring.q)

    def test_ntt_on_different_devices(self, multi_gpu):
        ring = large_prime_ring(256)
        coeffs = rand_int_arr(size=256, bound=ring.q)

        ntt0 = Poly(coeffs, ring, device=multi_gpu[0]).ntt()
        ntt1 = Poly(coeffs, ring, device=multi_gpu[1]).ntt()
        # Realize before INTT so the compiler can't cancel the two transforms.
        realize(ntt0, ntt1)
        p0, p1 = ntt0.intt(), ntt1.intt()
        realize(p0, p1)

        assert list(p0) == coeffs
        assert list(p1) == coeffs

    def test_multiplication_different_devices(self, multi_gpu):
        ring = large_prime_ring(256)
        coeffs1 = rand_int_arr(size=256, bound=ring.q)
        coeffs2 = rand_int_arr(size=256, bound=ring.q)

        p1_d0 = Poly(coeffs1, ring, device=multi_gpu[0])
        p2_d0 = Poly(coeffs2, ring, device=multi_gpu[0])
        r_d0 = p1_d0 * p2_d0

        p1_d1 = Poly(coeffs1, ring, device=multi_gpu[1])
        p2_d1 = Poly(coeffs2, ring, device=multi_gpu[1])
        r_d1 = p1_d1 * p2_d1

        realize(r_d0, r_d1)

        expected = neg_convolve(coeffs1, coeffs2, n=ring.n, q=ring.q)
        assert list(r_d0) == expected
        assert list(r_d1) == expected

    def test_chain_of_copies(self, multi_gpu):
        ring = large_prime_ring(256)
        coeffs = [1, 2, 3, 4] + [0] * 252

        p = Poly(coeffs, ring, device=multi_gpu[0])
        # Realize each hop so the copy chain can't collapse to one transfer.
        p1 = p.to(multi_gpu[1]).realize()
        p2 = p1.to(multi_gpu[0]).realize()
        p3 = p2.to(multi_gpu[1]).realize()

        assert list(p3) == coeffs
