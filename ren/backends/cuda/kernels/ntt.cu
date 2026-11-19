// Generic merge kernels include portions adapted from GPU-NTT.
// (C) Ulvetanna Inc.
// Developer: Alişah Özcan
// These portions are licensed under Apache-2.0; see THIRD_PARTY.md.
// Modified for Ren's interfaces, Montgomery arithmetic and kernel scheduling.
// Paper: https://eprint.iacr.org/2023/1410

#include "common.cuh"

// Lazy changes only phase-internal representatives; phase input/output and
// twiddles stay canonical. With q<R/4, NTT/INTT live values stay below 4q/2q.
template <typename T, bool Lazy>
__device__ __forceinline__ void ren_ntt_butterfly(T &a, T &b, T w, T q,
                                                  T pinv) {
  if constexpr (Lazy) {
#if DEBUG >= 1
    assert(q < (T(1) << (RenWord<T>::bits - 2)));
    assert(a < 4 * q && b < 4 * q && w < q);
#endif
    const T two_q = q + q;
    const T t = ren_mont_mul_lazy(b, w, q, pinv);
    const T u = ren_mod_reduce(a, two_q);
    a = u + t;
    b = u + two_q - t;
  } else {
    const T u = a, t = ren_mont_mul(b, w, q, pinv);
    a = ren_mod_add(u, t, q);
    b = ren_mod_sub(u, t, q);
  }
}

template <typename T, bool Lazy>
__device__ __forceinline__ void ren_intt_butterfly(T &a, T &b, T w, T q,
                                                   T pinv) {
  if constexpr (Lazy) {
#if DEBUG >= 1
    assert(q < (T(1) << (RenWord<T>::bits - 2)));
    assert(a < 2 * q && b < 2 * q && w < q);
#endif
    const T two_q = q + q;
    const T sum = a + b, diff = a + two_q - b; // Both below 4q<R.
    a = ren_mod_reduce(sum, two_q);
    b = ren_mont_mul_lazy(diff, w, q, pinv);
  } else {
    const T left = a, right = b;
    a = ren_mod_add(left, right, q);
    b = ren_mont_mul(ren_mod_sub(left, right, q), w, q, pinv);
  }
}

/**
 * @brief Merge-style forward NTT kernel based on GPU-NTT's blocked schedule.
 *        Paper: https://eprint.iacr.org/2023/1410
 * @note This kernel computes a partial forward NTT. Multiple launches may be
 *       required to finish a full length-n transform.
 *
 * Example diagram of a partial NTT:
 *   [................................................................]
 *   [|--------block group 1 --------||----------block group 2-------|] < block group is block_grp
 *      ^^ two independent NTT operations that are > 1024 entries ^^
 *   [__--==>>__--==>>    (blank)     ] each symbol is a different block - 4 blocks / group < block is block_idx
 *    ^ tg 1  ^ tg 2      ^ blank as each thread operates on two elements
 *
 * Inputs, twiddles, and outputs are all stored in canonical Montgomery form.
 * That means the kernel body is still the usual Cooley-Tukey schedule, but the
 * twiddle multiply is now a same-modulus Montgomery multiply instead of a
 * Barrett multiply.
 *
 * @param src input polynomial in coefficient domain. src[i * n + j] is
 *            coefficient j modulo primes[i], already in Montgomery form.
 * @param dst output of a partial NTT on src. The scheduler decides how many
 *            launches are needed and which stages each launch covers.
 * @param roots powers of the primitive 2n-th root of unity for each prime,
 *              stored in bit-reversed order and in Montgomery form.
 * @param primes the RNS moduli, stored in the same order as the layout of src.
 * @param p_inv Montgomery reduction constants, one per prime.
 * @param logn log2(n), where n is the number of coefficients in the polynomial.
 * @param logm log2 of the initial m, where m is the number of independent NTT
 *             regions in the current outer iteration.
 * @param outer_iteration_count number of outer-loop iterations to run in this
 *                              launch.
 * @param shared caller-provided shared memory, sized for 2 butterflies per
 *               thread.
 */
template <typename T>
__global__ void
ren_ntt_forward_merge(const T *__restrict__ src, T *__restrict__ dst,
                      const T *__restrict__ roots, const T *__restrict__ primes,
                      const T *__restrict__ p_inv, int logn, int logm,
                      int outer_iteration_count) {
  extern __shared__ T shared[];

#if DEBUG >= 1
  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 &&
      blockIdx.y == 0 && blockIdx.z == 0) {
    assert(1ULL << (outer_iteration_count - 1) == blockDim.y &&
           "numThreadGrps should be 1/2 of 2**(oic)");
    assert(1ULL << logm == gridDim.y &&
           "m should be equal to the number of block groups");
    assert(1ULL << logn ==
               (gridDim.y * gridDim.x * blockDim.y * blockDim.x * 2) &&
           "n should be # threads * 2");
  }
#endif

  const int thread_idx = threadIdx.x;
  const int thread_grp = threadIdx.y;
  const int block_idx = blockIdx.x;
  const int block_grp = blockIdx.y;
  const int p_idx = blockIdx.z;

  const T q = __ldg(&primes[p_idx]);
  const T pinv = __ldg(&p_inv[p_idx]);

  const int log_offset_g = logn - logm - 1;
  const unsigned long long offset_g = 1ULL << log_offset_g;
  const unsigned long long thread_grp_stride =
      offset_g >> (outer_iteration_count - 1);
  const unsigned long long base = (unsigned long long)p_idx << logn;

  const unsigned long long global_addr =
      thread_idx + (unsigned long long)block_idx * blockDim.x +
      (unsigned long long)thread_grp * thread_grp_stride +
      2ULL * (unsigned long long)block_grp * offset_g + base;

  const unsigned long long flattened_thread_idx =
      thread_idx + (unsigned long long)block_idx * blockDim.x +
      (unsigned long long)thread_grp * thread_grp_stride +
      (unsigned long long)block_grp * offset_g;

  const int shared_addr = thread_idx + thread_grp * blockDim.x;
  const int offset_s = blockDim.x * blockDim.y;

#if DEBUG >= 1
  assert(src[global_addr] < q && src[global_addr + offset_g] < q);
#endif

  T U = src[global_addr];
  T V = src[global_addr + offset_g];
  shared[shared_addr] = U;
  shared[shared_addr + offset_s] = V;

  int log_ntt_region_size = log_offset_g + 1;
  int logt = ren_bit_width_u64(offset_s) - 1;
  unsigned long long m = 1ULL << logm;
  int t = 1 << logt;

  for (int lp = 0; lp < outer_iteration_count; ++lp) {
    int u = ((shared_addr >> logt) << logt) + shared_addr;
    const unsigned long long i =
        flattened_thread_idx >> (log_ntt_region_size - 1);
    const unsigned long long root_idx = m + i;
    const T w = __ldg(&roots[base + root_idx]);

    __syncthreads();
    U = shared[u];
    V = shared[u + t];

    V = ren_mont_mul(V, w, q, pinv);
    shared[u] = ren_mod_add(U, V, q);
    shared[u + t] = ren_mod_sub(U, V, q);

    t >>= 1;
    log_ntt_region_size -= 1;
    logt -= 1;
    m <<= 1;
  }

  __syncthreads();
  dst[global_addr] = shared[shared_addr];
  dst[global_addr + offset_g] = shared[shared_addr + offset_s];
}

// Twiddle pairs and groups are naturally aligned. Keep uint32's vector loads
// and stores; uint64 groups of four use two 16-byte accesses.
template <typename T>
__device__ __forceinline__ void ren_ntt_load2(const T *src, T &a, T &b) {
  if constexpr (sizeof(T) == 4) {
    const uint2 v = *reinterpret_cast<const uint2 *>(src);
    a = v.x;
    b = v.y;
  } else {
    const ulonglong2 v = *reinterpret_cast<const ulonglong2 *>(src);
    a = v.x;
    b = v.y;
  }
}

template <typename T>
__device__ __forceinline__ void ren_ntt_load4(const T *src, T &a, T &b, T &c,
                                              T &d) {
  if constexpr (sizeof(T) == 4) {
    const uint4 v = *reinterpret_cast<const uint4 *>(src);
    a = v.x;
    b = v.y;
    c = v.z;
    d = v.w;
  } else {
    ren_ntt_load2(src, a, b);
    ren_ntt_load2(src + 2, c, d);
  }
}

template <typename T>
__device__ __forceinline__ void ren_ntt_store4(T *dst, T a, T b, T c, T d) {
  if constexpr (sizeof(T) == 4) {
    *reinterpret_cast<uint4 *>(dst) = make_uint4(a, b, c, d);
  } else {
    ulonglong2 *out = reinterpret_cast<ulonglong2 *>(dst);
    out[0] = make_ulonglong2(a, b);
    out[1] = make_ulonglong2(c, d);
  }
}

// Three forward stages on eight register values; first is the twiddle index
// at the first of these stages. Subsequent stages use 2*first and 4*first.
template <typename T, bool Lazy = false>
__device__ __forceinline__ void ren_ntt_radix8(T (&local)[8], const T *roots,
                                               int first, T q, T pinv) {
  T w[7];
  w[0] = __ldg(roots + first);
  ren_ntt_load2(roots + 2 * first, w[1], w[2]);
  ren_ntt_load4(roots + 4 * first, w[3], w[4], w[5], w[6]);
#pragma unroll
  for (int stage = 0; stage < 3; ++stage) {
    const int distance = 4 >> stage;
#pragma unroll
    for (int group = 0; group < (1 << stage); ++group) {
#pragma unroll
      for (int j = 0; j < distance; ++j) {
        const int idx = 2 * distance * group + j;
        ren_ntt_butterfly<T, Lazy>(local[idx], local[idx + distance],
                                   w[(1 << stage) - 1 + group], q, pinv);
      }
    }
  }
}

// N=65536 forward phase 0: view each prime's coefficients as 128 rows
// by 512 columns, src[512*row+column]. Each block tiles 64 bytes per row:
// 16 uint32 or 8 uint64 columns, with eight register values per thread. The seven stages are
// grouped 3+3+1; two shared exchanges change ownership within each column.
// Shared layout is [row][column]. A thread overwrites only addresses it has
// finished reading, with a barrier before each change of ownership.
// grid=(512/Columns,1,K), block=(16*Columns,1), shared=8192 bytes. The global layout, canonical
// Montgomery residues, and in-place/out-of-place ABI match the merge kernel.
template <typename T, bool Lazy = false>
__global__ void ren_ntt_forward_radix8_phase0(
    const T *__restrict__ src, T *__restrict__ dst, const T *__restrict__ roots,
    const T *__restrict__ primes, const T *__restrict__ p_inv, int logn,
    int logm, int outer_iteration_count) {
  constexpr int Columns = 64 / sizeof(T);
  extern __shared__ __align__(16) unsigned char phase0_shared_raw[];
  T *phase0_shared = reinterpret_cast<T *>(phase0_shared_raw);
  const int c = threadIdx.x % Columns;
  const int row_thread = threadIdx.x / Columns;
  const int column = blockIdx.x * Columns + c;
  const int prime = blockIdx.z;
  const T q = __ldg(primes + prime);
  const T pinv = __ldg(p_inv + prime);
  const unsigned long long base = ((unsigned long long)prime << 16) + column;
  roots += (unsigned long long)prime << 16;
#if DEBUG >= 1
  assert(logn == 16 && logm == 0 && outer_iteration_count == 7);
  assert(blockDim.x == 16 * Columns && blockDim.y == 1 && blockDim.z == 1);
  assert(gridDim.x == 512 / Columns && gridDim.y == 1);
#endif
  T local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = src[base + (row_thread + 16 * j) * 512];
#if DEBUG >= 1
    assert(local[j] < q);
#endif
  }
  ren_ntt_radix8<T, Lazy>(local, roots, 1, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    phase0_shared[(row_thread + 16 * j) * Columns + c] = local[j];
  __syncthreads();
  const int middle = (row_thread / 2) * 16 + row_thread % 2;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = phase0_shared[(middle + 2 * j) * Columns + c];
  ren_ntt_radix8<T, Lazy>(local, roots, 8 + row_thread / 2, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    phase0_shared[(middle + 2 * j) * Columns + c] = local[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = phase0_shared[(8 * row_thread + j) * Columns + c];
  T w[4];
  ren_ntt_load4(roots + 64 + 4 * row_thread, w[0], w[1], w[2], w[3]);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    ren_ntt_butterfly<T, Lazy>(local[2 * pair], local[2 * pair + 1], w[pair], q,
                               pinv);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if constexpr (Lazy)
      local[j] = ren_mod_reduce(local[j], q, 4);
#if DEBUG >= 1
    assert(local[j] < q);
#endif
    dst[base + (8 * row_thread + j) * 512] = local[j];
  }
}

// N=65536 forward phase 1: one block owns 512 consecutive coefficients.
// Each of 64 threads keeps eight values in registers across three stages.
// Two shared-memory exchanges change ownership for the next group of stages:
//   tid + 64*j -> 64*(tid/8) + tid%8 + 8*j -> 8*tid + j.
// At each exchange these address sets partition all 512 values, so each thread
// overwrites only the shared addresses it has finished reading. A barrier after
// the stores is sufficient before the next ownership change.
// Phase input/output are canonical. With Lazy, internal values are below 4q.
// ABI and table layout match the merge kernel; grid=(1,128,K), block=(64,1),
// dynamic shared memory=512*sizeof(T). Both in-place and out-of-place
// launches are supported.
template <typename T, bool Lazy = false>
__global__ void ren_ntt_forward_radix8_phase1(
    const T *__restrict__ src, T *__restrict__ dst, const T *__restrict__ roots,
    const T *__restrict__ primes, const T *__restrict__ p_inv, int logn,
    int logm, int outer_iteration_count) {
  extern __shared__ __align__(16) unsigned char radix8_shared_raw[];
  T *radix8_shared = reinterpret_cast<T *>(radix8_shared_raw);
  const int tid = threadIdx.x;
  const int tile = blockIdx.y;
  const int prime_idx = blockIdx.z;
  const T q = __ldg(primes + prime_idx);
  const T pinv = __ldg(p_inv + prime_idx);
#if DEBUG >= 1
  assert(logn == 16 && logm == 7 && outer_iteration_count == 9);
  assert(blockDim.x == 64 && blockDim.y == 1 && blockDim.z == 1);
  assert(gridDim.x == 1 && gridDim.y == 128);
#endif
  const unsigned long long base =
      ((unsigned long long)prime_idx << 16) + tile * 512;
  roots += (unsigned long long)prime_idx << 16;
  T local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = src[base + tid + j * 64];
#if DEBUG >= 1
    assert(local[j] < q);
#endif
  }
  // First twiddle indices at logm=7, 10, and 13 follow this phase's tile layout.
  ren_ntt_radix8<T, Lazy>(local, roots, 128 + tile + tid / 64, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    radix8_shared[tid + j * 64] = local[j];
  __syncthreads();
  const int middle = (tid / 8) * 64 + tid % 8;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = radix8_shared[middle + j * 8];
  ren_ntt_radix8<T, Lazy>(local, roots, 1024 + (tile << 3) + tid / 8, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    radix8_shared[middle + j * 8] = local[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = radix8_shared[tid * 8 + j];
  ren_ntt_radix8<T, Lazy>(local, roots, 8192 + (tile << 6) + tid, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    if constexpr (Lazy)
      local[j] = ren_mod_reduce(local[j], q, 4);
#if DEBUG >= 1
#pragma unroll
  for (int j = 0; j < 8; ++j)
    assert(local[j] < q);
#endif
  ren_ntt_store4(dst + base + tid * 8, local[0], local[1], local[2], local[3]);
  ren_ntt_store4(dst + base + tid * 8 + 4, local[4], local[5], local[6],
                 local[7]);
}

/**
 * @brief Merge-style inverse NTT kernel using the same blocked schedule as the
 *        forward kernel.
 * @note This kernel computes a partial inverse NTT. Multiple launches may be
 *       required to finish a full length-n inverse transform.
 *
 * This kernel mirrors the forward kernel in reverse launch order. The main
 * asymmetry is that a Gentleman-Sande butterfly introduces a factor of 2 at
 * each stage, so the final launch must multiply by n^{-1}. Because n_inv is
 * stored in Montgomery form, that final scaling still leaves the output in
 * canonical Montgomery form.
 *
 * @param src input polynomial in NTT domain, stored in Montgomery form.
 * @param dst output of a partial inverse NTT on src.
 * @param roots_inv inverse twiddle table in bit-reversed order and Montgomery
 *                  form.
 * @param primes the RNS moduli, stored in the same order as the layout of src.
 * @param p_inv Montgomery reduction constants, one per prime.
 * @param n_inv n^{-1} mod p_i for each RNS modulus, stored in Montgomery form.
 * @param logn log2(n), where n is the number of coefficients in the polynomial.
 * @param logm_final log2 of the m used in the final iteration executed by this
 *                   kernel launch.
 * @param outer_iteration_count number of outer-loop iterations to run in this
 *                              launch.
 * @param shared caller-provided shared memory, sized for 2 butterflies per
 *               thread.
 */
template <typename T>
__global__ void
ren_ntt_inverse_merge(const T *__restrict__ src, T *__restrict__ dst,
                      const T *__restrict__ roots_inv,
                      const T *__restrict__ primes, const T *__restrict__ p_inv,
                      const T *__restrict__ n_inv, int logn, int logm_final,
                      int outer_iteration_count) {
  extern __shared__ T shared[];

  const int logm = logm_final - 1 + outer_iteration_count;

#if DEBUG >= 1
  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 &&
      blockIdx.y == 0 && blockIdx.z == 0) {
    assert(1ULL << (outer_iteration_count - 1) == blockDim.y &&
           "numThreadGrps should be 1/2 of 2**(oic)");
    assert(1ULL << logm_final == gridDim.y &&
           "m should be equal to the number of block groups");
    assert(1ULL << logn ==
               (gridDim.y * gridDim.x * blockDim.y * blockDim.x * 2) &&
           "n should be # threads * 2");
  }
#endif

  const int thread_idx = threadIdx.x;
  const int thread_grp = threadIdx.y;
  const int block_idx = blockIdx.x;
  const int block_grp = blockIdx.y;
  const int p_idx = blockIdx.z;

  const T q = __ldg(&primes[p_idx]);
  const T pinv = __ldg(&p_inv[p_idx]);

  const int log_offset_g = logn - logm_final - 1;
  const unsigned long long offset_g = 1ULL << log_offset_g;
  const unsigned long long thread_grp_stride =
      offset_g >> (outer_iteration_count - 1);
  const unsigned long long base = (unsigned long long)p_idx << logn;

  const unsigned long long global_addr =
      thread_idx + (unsigned long long)block_idx * blockDim.x +
      (unsigned long long)thread_grp * thread_grp_stride +
      2ULL * (unsigned long long)block_grp * offset_g + base;

  const unsigned long long flattened_thread_idx =
      thread_idx + (unsigned long long)block_idx * blockDim.x +
      (unsigned long long)thread_grp * thread_grp_stride +
      (unsigned long long)block_grp * offset_g;

  const int shared_addr = thread_idx + thread_grp * blockDim.x;
  const int offset_s = blockDim.x * blockDim.y;

#if DEBUG >= 1
  assert(src[global_addr] < q && src[global_addr + offset_g] < q);
#endif

  T U = src[global_addr];
  T V = src[global_addr + offset_g];
  shared[shared_addr] = U;
  shared[shared_addr + offset_s] = V;

  int log_ntt_region_size = logn - logm;
  int logt = ren_bit_width_u64(offset_s) - outer_iteration_count;
  unsigned long long m = 1ULL << logm;
  int t = 1 << logt;
  T S = 0;
  T D = 0;

  for (int lp = 0; lp < outer_iteration_count; ++lp) {
    int u = ((shared_addr >> logt) << logt) + shared_addr;
    const unsigned long long i =
        flattened_thread_idx >> (log_ntt_region_size - 1);
    const unsigned long long root_idx = m + i;
    const T w = __ldg(&roots_inv[base + root_idx]);

    __syncthreads();
    U = shared[u];
    V = shared[u + t];

    S = ren_mod_add(U, V, q);
    D = ren_mont_mul(ren_mod_sub(U, V, q), w, q, pinv);

    shared[u] = S;
    shared[u + t] = D;

    t <<= 1;
    log_ntt_region_size += 1;
    logt += 1;
    m >>= 1;
  }

  T a0 = S;
  T a1 = D;
  if (m == 0) {
    const T ninv = __ldg(&n_inv[p_idx]);
    a0 = ren_mont_mul(a0, ninv, q, pinv);
    a1 = ren_mont_mul(a1, ninv, q, pinv);
  }

  dst[global_addr] = a0;
  dst[global_addr + offset_g] = a1;
}

// Three inverse stages on eight register values. first is the twiddle index
// at the last of these stages; the earlier stages use 4*first and 2*first.
// Each caller supplies the index for its phase-specific data layout.
template <typename T, bool Lazy = false>
__device__ __forceinline__ void ren_intt_radix8(T (&local)[8], const T *roots,
                                                int first, T q, T pinv) {
  T w[7];
  w[0] = __ldg(roots + first);
  ren_ntt_load2(roots + 2 * first, w[1], w[2]);
  ren_ntt_load4(roots + 4 * first, w[3], w[4], w[5], w[6]);
#pragma unroll
  for (int stage = 0; stage < 3; ++stage) {
    const int distance = 1 << stage;
#pragma unroll
    for (int group = 0; group < (4 >> stage); ++group) {
#pragma unroll
      for (int j = 0; j < distance; ++j) {
        const int index = 2 * distance * group + j;
        ren_intt_butterfly<T, Lazy>(local[index], local[index + distance],
                                    w[(4 >> stage) - 1 + group], q, pinv);
      }
    }
  }
}

// N=65536 INTT first launch (phase 0), grid=(1,128,K), block=(64,1), shared=512*sizeof(T) bytes.
// Nine Gentleman-Sande stages, grouped as (1,2,4), (8,16,32), (64,128,256).
// The original inverse table is retained. With Lazy, internal values are
// below 2q. N^{-1} scaling remains in the final INTT phase.
// Each shared-memory reader overwrites only its own previously read addresses;
// the two barriers separate changes of ownership. Both aliasing modes work.
template <typename T, bool Lazy = false>
__global__ void ren_ntt_inverse_radix8_phase0(
    const T *__restrict__ src, T *__restrict__ dst, const T *__restrict__ roots,
    const T *__restrict__ primes, const T *__restrict__ p_inv,
    const T *__restrict__ n_inv, int logn, int logm_final,
    int outer_iteration_count) {
  extern __shared__ __align__(16) unsigned char inverse_radix8_shared_raw[];
  T *inverse_radix8_shared = reinterpret_cast<T *>(inverse_radix8_shared_raw);
  const int tid = threadIdx.x;
  const int tile = blockIdx.y;
  const int prime_idx = blockIdx.z;
  const T q = __ldg(primes + prime_idx);
  const T pinv = __ldg(p_inv + prime_idx);
#if DEBUG >= 1
  assert(logn == 16 && logm_final == 7 && outer_iteration_count == 9);
  assert(blockDim.x == 64 && blockDim.y == 1 && blockDim.z == 1);
  assert(gridDim.x == 1 && gridDim.y == 128);
#endif
  const unsigned long long base =
      ((unsigned long long)prime_idx << 16) + tile * 512;
  roots += (unsigned long long)prime_idx << 16;
  T local[8];
  ren_ntt_load4(src + base + tid * 8, local[0], local[1], local[2], local[3]);
  ren_ntt_load4(src + base + tid * 8 + 4, local[4], local[5], local[6],
                local[7]);
#if DEBUG >= 1
#pragma unroll
  for (int j = 0; j < 8; ++j)
    assert(local[j] < q);
#endif
  ren_intt_radix8<T, Lazy>(local, roots, 8192 + (tile << 6) + tid, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    inverse_radix8_shared[tid * 8 + j] = local[j];
  __syncthreads();
  const int middle = (tid / 8) * 64 + tid % 8;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = inverse_radix8_shared[middle + j * 8];
  ren_intt_radix8<T, Lazy>(local, roots, 1024 + (tile << 3) + tid / 8, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    inverse_radix8_shared[middle + j * 8] = local[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = inverse_radix8_shared[tid + j * 64];
  ren_intt_radix8<T, Lazy>(local, roots, 128 + tile + tid / 64, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if constexpr (Lazy)
      local[j] = ren_mod_reduce(local[j], q);
#if DEBUG >= 1
    assert(local[j] < q);
#endif
    dst[base + tid + j * 64] = local[j];
  }
}

// Final INTT launch, N=65536. Each column of src[512*row+column] is a
// 128-value inverse transform. Execute seven stages as 1+3+3, then scale by
// N^{-1}. The two shared exchanges reverse forward phase 0's ownership:
// 8*r+j -> 16*(r/2)+r%2+2*j -> r+16*j.
// A writer overwrites only addresses it has finished reading; the barriers
// separate changes of ownership. Blocks own disjoint sets of columns.
// Tile 64 bytes per row, as in forward phase 0.
// grid=(512/Columns,1,K), block=(16*Columns,1), shared=8192 bytes.
template <typename T, bool Lazy = false>
__global__ void ren_ntt_inverse_radix8_phase1(
    const T *__restrict__ src, T *__restrict__ dst, const T *__restrict__ roots,
    const T *__restrict__ primes, const T *__restrict__ p_inv,
    const T *__restrict__ n_inv, int logn, int logm_final,
    int outer_iteration_count) {
  constexpr int Columns = 64 / sizeof(T);
  extern __shared__ __align__(16) unsigned char phase1_shared_raw[];
  T *phase1_shared = reinterpret_cast<T *>(phase1_shared_raw);
  const int c = threadIdx.x % Columns;
  const int r = threadIdx.x / Columns;
  const int column = blockIdx.x * Columns + c;
  const int prime = blockIdx.z;
  const T q = __ldg(primes + prime);
  const T pinv = __ldg(p_inv + prime);
  const T ninv = __ldg(n_inv + prime);
  const unsigned long long base = ((unsigned long long)prime << 16) + column;
  roots += (unsigned long long)prime << 16;
#if DEBUG >= 1
  assert(logn == 16 && logm_final == 0 && outer_iteration_count == 7);
  assert(blockDim.x == 16 * Columns && blockDim.y == 1 && blockDim.z == 1);
  assert(gridDim.x == 512 / Columns && gridDim.y == 1);
#endif
  T local[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    local[j] = src[base + (8 * r + j) * 512];
#if DEBUG >= 1
    assert(local[j] < q);
#endif
  }
  T w[4];
  ren_ntt_load4(roots + 64 + 4 * r, w[0], w[1], w[2], w[3]);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    ren_intt_butterfly<T, Lazy>(local[2 * pair], local[2 * pair + 1], w[pair],
                                q, pinv);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j)
    phase1_shared[(8 * r + j) * Columns + c] = local[j];
  __syncthreads();
  const int middle = (r / 2) * 16 + r % 2;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = phase1_shared[(middle + 2 * j) * Columns + c];
  ren_intt_radix8<T, Lazy>(local, roots, 8 + r / 2, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    phase1_shared[(middle + 2 * j) * Columns + c] = local[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j)
    local[j] = phase1_shared[(r + 16 * j) * Columns + c];
  ren_intt_radix8<T, Lazy>(local, roots, 1, q, pinv);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const T result = ren_mont_mul(local[j], ninv, q, pinv);
#if DEBUG >= 1
    assert(result < q);
#endif
    dst[base + (r + 16 * j) * 512] = result;
  }
}
