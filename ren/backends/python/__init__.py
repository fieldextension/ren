import math
import os
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np

from ren.ir.node import RandomArg, from_rns_limb_count, python_alu
import ren.montgomery as mont
from ren.allocator import Allocator
from ren.backend import Backend
from ren.buffer import Buffer, BufferSpec
from ren.dtype import DType, dtypes, to_np_dtype
from ren.engine import codegen
from ren.engine.pattern import pat, rules
from ren.ir import Node, Op, ToRingArg
from ren.kernel import KernelSpec
from ren.backends.python.random import rand
from ren.random import ChaChaRng
from ren.ring import Residue, RingSpec
from ren.tables import Tables


SUPPORTED_DTYPES = frozenset({dtypes.bool, dtypes.index, dtypes.uint32, dtypes.uint64})

# Batch at most 64K coefficients, or one row for larger degrees.
_NTT_BATCH_COEFFS = 1 << 16


_handlers = rules([
    (pat.n(python_alu), lambda n: (n.src, lambda k, src, bufs, fn=python_alu[n.op]: fn(*src))),
    (pat.n(Op.WHERE), lambda n: (n.src, lambda k, src, bufs: np.where(*src))),
    (pat.n(Op.AFTER), lambda n: (n.src, lambda k, src, bufs: src[0])),
    (pat.n(Op.PARAM), lambda n: ((), lambda k, src, bufs: np.frombuffer(bufs[n.arg.slot].as_typed_buffer(), dtype=to_np_dtype(n.dtype)))),
    (pat(Op.GIDX), lambda: ((), lambda k, src, bufs: np.arange(k.ring.storage_size, dtype=to_np_dtype(dtypes.index)))),
    (pat.n(Op.CAST, pat(Op.CONST)), lambda n: ((), lambda k, src, bufs, value=np.asarray(n.const_val, dtype=to_np_dtype(n.dtype)): value)),
    (pat.n(Op.CAST), lambda n: (n.src, lambda k, src, bufs: src[0])),
    (pat.n(Op.CONST_ARRAY), lambda n: ((), lambda k, src, bufs, value=np.asarray(n.arg, dtype=to_np_dtype(n.dtype)): value)),
    (pat.n(Op.CONST), lambda n: ((), lambda k, src, bufs: n.arg)),
    (pat.n(Op.INDEX), lambda n: (n.src, lambda k, src, bufs: (src[0], src[1]))),
    (pat.n(Op.LOAD), lambda n: (n.src, lambda k, src, bufs: operator.getitem(*src[0]))),
    (pat.n(Op.STORE, (pat(Op.INDEX), pat())), lambda n: (n.src, lambda k, src, bufs: operator.setitem(*src[0], src[1]))),
    (pat.n(Op.STORE, (pat(Op.PARAM), pat())), lambda n: (n.src, lambda k, src, bufs: operator.setitem(src[0], slice(None), src[1]))),
    (pat.n(Op.BITREVERSE), lambda n: (n.src, lambda k, src, bufs: _bit_reverse(src[0], n.arg))),
    (pat.n(Op.NTT), lambda n: (n.src, lambda k, src, bufs: k.backend._handle_ntt(k.ring, src[0]))),
    (pat.n(Op.INTT), lambda n: (n.src, lambda k, src, bufs: k.backend._handle_intt(k.ring, src[0]))),
    (pat.n(Op.TO_RING), lambda n: (n.src, lambda k, src, bufs: k.backend._handle_to_ring(k.ring, n.arg, src[0]))),
    (pat.n(Op.TO_RNS), lambda n: (n.src, lambda k, src, bufs: k.backend._handle_to_rns(k.ring, src[0]))),
    (pat.n(Op.FROM_RNS), lambda n: (n.src, lambda k, src, bufs: k.backend._handle_from_rns(k.ring, src[0]))),
    (pat.n(Op.RESCALE), lambda n: (n.src, lambda k, src, bufs: k.backend._handle_rescale(k.ring, n.arg, src[0]))),
    (pat.n(Op.RANDOM), lambda n: ((), lambda k, src, bufs: k.backend._handle_random(k.ring, n.arg, bufs[n.src[0].arg.slot]))),
])


def _cast(val: Any, dtype: DType) -> np.ndarray:
    dt_np = to_np_dtype(dtype)
    if isinstance(val, np.ndarray) and val.dtype == dt_np:
        return val
    return np.asarray(val, dtype=dt_np)


class PythonProgram:
    def __init__(self, backend: PythonBackend, spec: KernelSpec):
        self.backend, self.ring = backend, spec.ring
        self.program = []
        needed = set(spec.stores)
        for node in reversed(spec.program):
            if node not in needed:
                continue
            if (prepared := _handlers.match(node)) is None:
                raise NotImplementedError(f"python backend: unsupported op {node.op}")
            src, handler = prepared
            # The first consumer found backwards is the last use during execution.
            release = tuple(s for s in dict.fromkeys(src) if s not in needed)
            needed.update(src)
            self.program.append((node, src, handler, release))
        self.program.reverse()

    def __call__(self, bufs: list[Buffer]) -> None:
        values: dict[Node, Any] = {}
        for node, operands, handler, release in self.program:
            value = handler(self, [values[s] for s in operands], bufs)
            if node.op is not Op.STORE:
                values[node] = value if node.op is Op.INDEX or node.dtype is dtypes.weakint else _cast(value, node.dtype)
            for source in release:
                del values[source]


class PythonAllocator(Allocator):
    def _alloc(self, spec: BufferSpec):
        return memoryview(bytearray(spec.size * spec.dtype.itemsize))

    def _free(self, buf: Any, spec: BufferSpec) -> None:
        pass

    def as_buffer(self, buf: Any, nbytes: int) -> memoryview:
        mv = buf if isinstance(buf, memoryview) else memoryview(buf)
        if len(mv) != nbytes:
            mv = mv[:nbytes]
        return mv if mv.format == "B" else mv.cast("B")

    def _copyin(self, buf, src: memoryview):
        buf[:] = src

    def _copyout(self, dest: memoryview, buf):
        dest[:] = buf

    def _offset(self, buf, size: int, offset: int):
        return buf[offset : offset + size]

    def transfer(self, dest, src, size: int, src_dev, dest_dev) -> None:
        dest[:size] = src[:size]


class PythonBackend(Backend):
    def __init__(self) -> None:
        self._allocator = PythonAllocator()
        self.tables = Tables("python")
        self._make_kernel_cache: dict[Node, PythonProgram] = {}
        # We can't allocate buffers in the constructor, so set default rng lazily
        self._default_rng: ChaChaRng | None = None

    @property
    def allocator(self) -> PythonAllocator:
        return self._allocator

    def synchronize(self) -> None:
        return None

    def finalize(self) -> None:
        return None

    def make_kernel(self, ast: Node) -> PythonProgram:
        if (program := self._make_kernel_cache.get(ast)) is None:
            program = self._make_kernel_cache[ast] = PythonProgram(self, codegen.lower_kernel(ast, SUPPORTED_DTYPES))
        return program

    def set_default_rng(self, rng: ChaChaRng):
        if not isinstance(rng, ChaChaRng):
            raise TypeError("rng must be a ChaChaRng")
        if rng.device != self.tables.device:
            raise ValueError(f"RNG device mismatch: {rng.device} != {self.tables.device}")
        self._default_rng = rng

    def default_rng(self) -> ChaChaRng:
        if self._default_rng is None:
            self.set_default_rng(ChaChaRng(self.tables.device, os.urandom(ChaChaRng.seed_size)))
        assert self._default_rng is not None
        return self._default_rng

    def _rns_rows(self, data: Any, ring: RingSpec) -> np.ndarray:
        return np.asarray(data, dtype=to_np_dtype(ring.dtype)).reshape(len(ring.moduli), ring.n)

    def _handle_ntt(self, ring: RingSpec, operand: Any) -> np.ndarray:
        rt = self.tables.ring(ring)
        rows = self._rns_rows(operand, ring).copy()
        primes = np.asarray(ring.moduli, dtype=np.uint64)
        p_inv = np.asarray(rt.mont_p_inv.as_typed_buffer(), dtype=np.uint64)
        psis = self._rns_rows(rt.ntt_psis.as_typed_buffer(), ring)
        batch_size = max(1, _NTT_BATCH_COEFFS // ring.n)
        for start in range(0, len(primes), batch_size):
            batch = slice(start, start + batch_size)
            rows[batch] %= primes[batch, None]
            _ntt_forward(primes[batch], p_inv[batch], psis[batch], rows[batch])
        return rows.reshape(-1)

    def _handle_intt(self, ring: RingSpec, operand: Any) -> np.ndarray:
        rt = self.tables.ring(ring)
        rows = self._rns_rows(operand, ring).copy()
        primes = np.asarray(ring.moduli, dtype=np.uint64)
        p_inv = np.asarray(rt.mont_p_inv.as_typed_buffer(), dtype=np.uint64)
        psis_inv = self._rns_rows(rt.ntt_psis_inv.as_typed_buffer(), ring)
        n_inv = np.asarray(rt.n_inv.as_typed_buffer(), dtype=np.uint64)
        batch_size = max(1, _NTT_BATCH_COEFFS // ring.n)
        for start in range(0, len(primes), batch_size):
            batch = slice(start, start + batch_size)
            rows[batch] %= primes[batch, None]
            _ntt_backward(primes[batch], p_inv[batch], psis_inv[batch], n_inv[batch], rows[batch])
        return rows.reshape(-1)

    def _handle_random(self, ring: RingSpec, arg: RandomArg, key: Buffer) -> np.ndarray:
        return rand(arg.dist, ring, key=bytes(key.as_buffer()), stream_id=arg.stream_id, tables=self.tables.dists)

    def _handle_rescale(self, src_ring: RingSpec, dst_ring: RingSpec, operand: Any) -> np.ndarray:
        if dst_ring.dtype is not src_ring.dtype:
            raise ValueError(f"ring dtype mismatch: {src_ring.dtype} != {dst_ring.dtype}")

        primes = src_ring.moduli
        num_q = len(dst_ring.moduli)
        num_b = len(primes) - num_q
        if dst_ring.moduli == primes[num_b:] or dst_ring.moduli == primes[:num_q]:
            return self._handle_quotient_rescale(src_ring, dst_ring, operand)

        coeff_np_dtype = to_np_dtype(src_ring.dtype)
        src_rows = self._rns_rows(operand, src_ring)
        rescale_rt = self.tables.rescale(src_ring, dst_ring)
        removed_src_indices = tuple(int(idx) for idx in rescale_rt.removed_src_indices.as_typed_buffer())
        src_only = tuple(src_ring.moduli[idx] for idx in removed_src_indices)
        missing = np.iinfo(np.uint32).max

        dst_rows = np.zeros((len(dst_ring.moduli), src_ring.n), dtype=coeff_np_dtype)
        dst_src_indices = rescale_rt.dst_src_indices.as_typed_buffer()
        scale_added = rescale_rt.added_prod != 1
        added_prod_dst_mont = rescale_rt.added_prod_dst_mont.as_typed_buffer()
        for dst_idx, src_idx in enumerate(dst_src_indices):
            src_idx = int(src_idx)
            if src_idx == missing:
                continue
            q = dst_ring.moduli[dst_idx]
            mp = mont.context(src_ring.dtype, q)
            shared = src_rows[src_idx] % q
            if scale_added:
                shared = mp.mul(shared, int(added_prod_dst_mont[dst_idx]))
            dst_rows[dst_idx] = shared

        removed_rows = np.empty((len(src_only), src_ring.n), dtype=coeff_np_dtype)
        added_prod_removed_mont = rescale_rt.added_prod_removed_mont.as_typed_buffer()
        for b_idx, src_idx in enumerate(removed_src_indices):
            p = src_only[b_idx]
            mp = mont.context(src_ring.dtype, p)
            removed = src_rows[src_idx] % p
            if scale_added:
                removed = mp.mul(removed, int(added_prod_removed_mont[b_idx]))
            removed_rows[b_idx] = removed

        b_ring = RingSpec(n=src_ring.n, moduli=src_only, dtype=src_ring.dtype)
        bc = self.tables.base_change(b_ring, dst_ring)
        removed_in_dst_rows = self._lift_limbs_mont(
            removed_rows,
            src_only,
            dst_ring.moduli,
            bc.inv_src.as_typed_buffer(),
            bc.mi_r2_scaled.as_typed_buffer(),
            bc.prod_r2_scaled.as_typed_buffer(),
            src_ring.dtype,
        )

        result_rows = np.empty_like(dst_rows)
        inv_prod_mont = rescale_rt.inv_prod_mont.as_typed_buffer()
        for dst_idx, modulus in enumerate(dst_ring.moduli):
            mp = mont.context(src_ring.dtype, modulus)
            diff = mp.sub(dst_rows[dst_idx], removed_in_dst_rows[dst_idx])
            result_rows[dst_idx] = mp.mul(diff, int(inv_prod_mont[dst_idx]))
        return result_rows.reshape(-1)

    def _handle_quotient_rescale(self, src_ring: RingSpec, dst_ring: RingSpec, operand: Any) -> np.ndarray:
        primes = src_ring.moduli
        num_q = len(dst_ring.moduli)
        num_b = len(primes) - num_q
        a_rows = self._rns_rows(operand, src_ring)
        rescale_rt = self.tables.rescale(src_ring, dst_ring)

        if dst_ring.moduli == primes[num_b:]:
            q_primes = primes[num_b:]
            b_ring = src_ring[:num_b]
            q_rows = a_rows[num_b:].copy()
            b_rows = a_rows[:num_b]
        else:
            q_primes = primes[:num_q]
            b_ring = src_ring[num_q:]
            q_rows = a_rows[:num_q].copy()
            b_rows = a_rows[num_q:]

        inv_prod_mont = rescale_rt.inv_prod_mont.as_typed_buffer()
        b_primes = b_ring.moduli

        if num_b == 1:
            # Match CUDA's exact centering for a single removed prime.
            b = b_primes[0]
            remainder = mont.context(src_ring.dtype, b).decode(b_rows[0] % b)
            correction = remainder > b // 2
            b_in_q_rows = np.empty_like(q_rows)
            for qi_idx, qi in enumerate(q_primes):
                mp = mont.context(src_ring.dtype, qi)
                b_in_q_rows[qi_idx] = mp.sub(mp.encode(remainder), np.where(correction, mp.encode(b), 0))
        else:
            b_in_q_rows = self._lift_limbs_mont(
                b_rows,
                b_primes,
                q_primes,
                rescale_rt.inv_src.as_typed_buffer(),
                rescale_rt.mi_r2_scaled.as_typed_buffer(),
                rescale_rt.prod_r2_scaled.as_typed_buffer(),
                src_ring.dtype,
            )

        for qi_idx in range(num_q):
            qi = q_primes[qi_idx]
            qi_mp = mont.context(src_ring.dtype, qi)
            q_limb = q_rows[qi_idx] % qi
            diff = qi_mp.sub(q_limb, b_in_q_rows[qi_idx])
            q_rows[qi_idx] = qi_mp.mul(diff, int(inv_prod_mont[qi_idx]))
        return q_rows.reshape(-1)

    def _handle_to_ring(self, src_ring: RingSpec, arg: ToRingArg, operand: Any) -> np.ndarray:
        dst_ring = arg.ring
        if dst_ring.dtype is not src_ring.dtype:
            raise ValueError(f"ring dtype mismatch: {src_ring.dtype} != {dst_ring.dtype}")

        bc = self.tables.base_change(src_ring, dst_ring)
        coeff_np_dtype = to_np_dtype(src_ring.dtype)
        src_rows = self._rns_rows(operand, src_ring)
        dst_src_indices = bc.dst_src_indices.as_typed_buffer()
        missing = np.iinfo(np.uint32).max

        dst_rows = np.empty((len(dst_ring.moduli), src_ring.n), dtype=coeff_np_dtype)
        added_dst_indices: list[int] = []
        for dst_idx, src_idx in enumerate(dst_src_indices):
            src_idx = int(src_idx)
            if src_idx == missing:
                added_dst_indices.append(dst_idx)
            else:
                dst_rows[dst_idx] = src_rows[src_idx]

        if added_dst_indices:
            coeffs = self._reconstruct_coeffs_exact(
                src_rows,
                src_ring,
                residue=arg.residue,
            )
            for dst_idx in added_dst_indices:
                q = dst_ring.moduli[dst_idx]
                dst_rows[dst_idx] = mont.context(src_ring.dtype, q).encode(coeffs)

        return dst_rows.reshape(-1)

    def _handle_to_rns(self, ring: RingSpec, operand: Any) -> np.ndarray:
        words = np.asarray(operand, dtype=np.uint32).reshape(-1)
        if len(words) % ring.n != 0:
            raise ValueError(f"TO_RNS expects a multiple of {ring.n} coefficient words, got {len(words)}")
        limb_count = len(words) // ring.n
        limbs = words.reshape(limb_count, ring.n).astype(object)

        coeffs = np.zeros(ring.n, dtype=object)
        for limb_idx in range(limb_count - 1, -1, -1):
            coeffs = coeffs * (1 << 32) + limbs[limb_idx]
        negative = (limbs[-1] & (1 << 31)) != 0
        coeffs[negative] -= 1 << (32 * limb_count)

        coeff_np_dtype = to_np_dtype(ring.dtype)
        rows = np.empty((len(ring.moduli), ring.n), dtype=coeff_np_dtype)
        for p_idx, modulus in enumerate(ring.moduli):
            rows[p_idx] = mont.context(ring.dtype, modulus).encode(coeffs)
        return rows.reshape(-1)

    def _handle_from_rns(self, ring: RingSpec, operand: Any) -> np.ndarray:
        """Inverse of TO_RNS: RNS Montgomery limbs -> raw signed two's-complement uint32 limbs.

        Output is the balanced [-Q/2, Q/2) representative, packed little-endian -- the same
        signed-limb encoding TO_RNS consumes, so the two ops are exact structural inverses.
        """
        limbs_mont = np.asarray(operand).reshape(len(ring.moduli), ring.n)
        coeffs = self._reconstruct_coeffs_exact(limbs_mont, ring, residue=Residue.BALANCED)

        mask = (1 << 32) - 1
        limb_count = from_rns_limb_count(ring)
        words = np.empty((limb_count, ring.n), dtype=np.uint32)
        for w in range(limb_count):
            words[w] = (coeffs >> (32 * w)) & mask  # little-endian, signed two's-complement
        return words.reshape(-1)

    def _reconstruct_coeffs_exact(
        self,
        limbs_mont: np.ndarray,
        ring: RingSpec,
        *,
        residue: Residue,
    ) -> np.ndarray:
        moduli = ring.moduli
        decoded_rows = [np.asarray(mont.context(ring.dtype, prime).decode(limbs_mont[p_idx]), dtype=object) for p_idx, prime in enumerate(moduli)]
        partial_prods, inverses = self.tables.ring(ring).crt_precompute
        coeffs = np.zeros(limbs_mont.shape[1], dtype=object)
        for i, (limb, prime) in enumerate(zip(decoded_rows, moduli)):
            diff = (limb - coeffs % prime) % prime
            coeffs += (inverses[i] * diff % prime) * partial_prods[i]

        if residue is Residue.BALANCED:
            q = math.prod(moduli)
            coeffs = ((coeffs + q // 2) % q) - q // 2
        return coeffs

    def _lift_limbs_mont(
        self,
        limbs: np.ndarray,
        primes_old: Sequence[int],
        primes_new: Sequence[int],
        inv_src: Sequence[int],
        mi_r2_scaled: Sequence[int],
        prod_r2_scaled: Sequence[int],
        dtype: DType,
    ) -> np.ndarray:
        coeff_dtype = to_np_dtype(dtype)
        n = limbs.shape[1] if limbs.size else 0
        num_old = len(primes_old)
        num_new = len(primes_new)
        dst = np.empty((num_new, n), dtype=coeff_dtype)
        t_rows = np.empty((num_old, n), dtype=coeff_dtype)
        old_mps = [mont.context(dtype, p) for p in primes_old]
        new_mps = [mont.context(dtype, q) for q in primes_new]

        acc_float = np.zeros(n, dtype=np.float64)
        for p_idx, p in enumerate(primes_old):
            t_rows[p_idx] = old_mps[p_idx].mul(limbs[p_idx] % p, int(inv_src[p_idx]))
            acc_float += np.asarray(t_rows[p_idx], dtype=np.float64) / float(p)
        correction = np.floor(acc_float + 0.5).astype(np.int64, copy=False)

        for dst_idx, q in enumerate(primes_new):
            mp = new_mps[dst_idx]
            mi_r2_scaled_row = mi_r2_scaled[dst_idx * num_old : (dst_idx + 1) * num_old]
            acc = np.zeros(n, dtype=coeff_dtype)
            for p_idx in range(num_old):
                acc = mp.add(acc, mp.mul(t_rows[p_idx] % q, int(mi_r2_scaled_row[p_idx])))
            dst[dst_idx] = mp.sub(acc, mp.mul(correction, int(prod_r2_scaled[dst_idx])))
        return dst


def _bit_reverse(x: np.ndarray, bits: int) -> np.ndarray | np.integer:
    r = np.zeros_like(x)
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x = x >> 1
    return r


def _ntt_forward(primes: np.ndarray, p_inv: np.ndarray, psis: np.ndarray, coeffs: np.ndarray) -> None:
    batch_size, n = coeffs.shape
    word_bits = coeffs.dtype.itemsize * 8
    q, q_inv = primes[:, None, None], p_inv[:, None, None]
    t = n
    m = 1
    while m < n:
        t //= 2
        blocks = coeffs.reshape(batch_size, m, 2, t)
        u = blocks[:, :, 0, :].astype(np.uint64)
        v = mont.mul(
            blocks[:, :, 1, :].astype(np.uint64, copy=False),
            psis[:, m : 2 * m, None].astype(np.uint64, copy=False),
            q,
            q_inv,
            word_bits,
        )
        blocks[:, :, 0, :] = mont.add(u, v, q)
        blocks[:, :, 1, :] = mont.sub(u, v, q)
        m *= 2


def _ntt_backward(primes: np.ndarray, p_inv: np.ndarray, psis_inv: np.ndarray, n_inv_mont: np.ndarray, coeffs: np.ndarray) -> None:
    batch_size, n = coeffs.shape
    word_bits = coeffs.dtype.itemsize * 8
    q, q_inv = primes[:, None, None], p_inv[:, None, None]
    t = 1
    m = n
    while m > 1:
        h = m // 2
        blocks = coeffs.reshape(batch_size, h, 2, t)
        top = blocks[:, :, 0, :].astype(np.uint64, copy=False)
        bottom = blocks[:, :, 1, :].astype(np.uint64, copy=False)
        summed = mont.add(top, bottom, q)
        diff = mont.sub(top, bottom, q)
        blocks[:, :, 0, :] = summed
        blocks[:, :, 1, :] = mont.mul(diff, psis_inv[:, h : 2 * h, None].astype(np.uint64, copy=False), q, q_inv, word_bits)
        t *= 2
        m //= 2
    coeffs[:] = mont.mul(coeffs.astype(np.uint64, copy=False), n_inv_mont[:, None], primes[:, None], p_inv[:, None], word_bits)
