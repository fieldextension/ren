// ChaCha code based on the reference implementation by D. J. Bernstein,
// which is in the public domain.
// Reference version 20080118.

#include "common.cuh"

__device__ __forceinline__ unsigned int ROTATE(unsigned int v, unsigned int c) {
  return __funnelshift_lc(v, v, c);
}
__device__ __forceinline__ unsigned int XOR(unsigned int v, unsigned int w) {
  return v ^ w;
}
__device__ __forceinline__ unsigned int PLUS(unsigned int v, unsigned int w) {
  return v + w;
}

const __constant__ unsigned int sigma[4] = {0x61707865, 0x3320646e, 0x79622d32,
                                            0x6b206574};

#define QUARTERROUND(a, b, c, d)                                               \
  x[a] = PLUS(x[a], x[b]);                                                     \
  x[d] = ROTATE(XOR(x[d], x[a]), 16);                                          \
  x[c] = PLUS(x[c], x[d]);                                                     \
  x[b] = ROTATE(XOR(x[b], x[c]), 12);                                          \
  x[a] = PLUS(x[a], x[b]);                                                     \
  x[d] = ROTATE(XOR(x[d], x[a]), 8);                                           \
  x[c] = PLUS(x[c], x[d]);                                                     \
  x[b] = ROTATE(XOR(x[b], x[c]), 7);

template <int ROUNDS> __device__ void chacha_core(unsigned int (&x)[16]) {
  for (int i = ROUNDS; i > 0; i -= 2) {
    QUARTERROUND(0, 4, 8, 12)
    QUARTERROUND(1, 5, 9, 13)
    QUARTERROUND(2, 6, 10, 14)
    QUARTERROUND(3, 7, 11, 15)
    QUARTERROUND(0, 5, 10, 15)
    QUARTERROUND(1, 6, 11, 12)
    QUARTERROUND(2, 7, 8, 13)
    QUARTERROUND(3, 4, 9, 14)
  }
}

__device__ void chacha_initialize(unsigned int (&data)[16],
                                  const unsigned int *__restrict__ key,
                                  unsigned long long stream_id,
                                  unsigned int block_counter) {
  for (int i = 0; i < 4; i++) {
    data[i] = sigma[i];
  }
  for (int i = 0; i < 8; i++) {
    data[i + 4] = __ldg(&key[i]);
  }
  data[12] = block_counter;
  data[13] = (unsigned int)stream_id;
  data[14] = (unsigned int)(stream_id >> 32);
  data[15] = 0;
}

__device__ void chacha_finalize(unsigned int (&data)[16],
                                const unsigned int *__restrict__ key,
                                unsigned long long stream_id,
                                unsigned int block_counter) {
  for (int i = 0; i < 4; i++) {
    data[i] += sigma[i];
  }
  for (int i = 0; i < 8; i++) {
    data[i + 4] += key[i];
  }
  data[12] += block_counter;
  data[13] += (unsigned int)stream_id;
  data[14] += (unsigned int)(stream_id >> 32);
}

__device__ void chacha_generate(unsigned int (&data)[16],
                                const unsigned int *__restrict__ key,
                                unsigned long long stream_id,
                                unsigned long long block_offset) {
  unsigned int block_counter = static_cast<unsigned int>(block_offset);
  chacha_initialize(data, key, stream_id, block_counter);
  chacha_core<20>(data);
  chacha_finalize(data, key, stream_id, block_counter);
}

template <typename T> struct tree_node {
  unsigned int split_point;
  T children[3];
};

template <typename T>
__device__ T sample(const tree_node<T> *__restrict__ tree,
                    const unsigned int *__restrict__ bits) {
  for (;;) {
    unsigned int split = tree->split_point;
    unsigned int rand_bits = *bits;
    int idx = (rand_bits > split) + (rand_bits >= split);
    T next = tree->children[idx];
    if (next < 0) {
      tree += ~next;
      bits += (idx == 1);
    } else {
      if (bits[1] & 1) {
        return -next;
      } else {
        return next;
      }
    }
  }
}

__device__ unsigned long long thread_index() {
  return threadIdx.x + blockDim.x * blockIdx.x;
}

/**
 * @brief Generates random bits using ChaCha20.
 *
 * This kernel is used for testing, to make sure that our implementation of
 * ChaCha is consistent with a more established implementation.
 *
 * @param key the ChaCha key
 * @param stream_id the ChaCha nonce stream identifier
 * @param output the output array
 * @param size the number of values to generate
 */
extern "C" __global__ void
ren_random_chacha(unsigned int *__restrict__ output,
                  const unsigned int *__restrict__ key,
                  unsigned long long stream_id, unsigned long long size) {
  unsigned long long gid = thread_index();
  if (gid >= size) {
    return;
  }
  unsigned int bits[16];
  chacha_generate(bits, key, stream_id, gid);
  for (unsigned int i = 0; i < 16; i++) {
    output[16 * gid + i] = bits[i];
  }
}

template <typename T>
__device__ typename RenWord<T>::signed_t signed_difference(T v1, T v2) {
  return static_cast<typename RenWord<T>::signed_t>(v1 - v2);
}

template <typename T>
__device__ T signed_to_mont(typename RenWord<T>::signed_t diff, T prime,
                            T p_inv, T r2) {
  while (diff < 0) {
    diff += prime;
  }
  return ren_mont_mul(static_cast<T>(diff), r2, prime, p_inv);
}

template <typename T>
__device__ T signed_to_mont_wide(long long diff, T prime, T p_inv, T r2) {
  long long p = static_cast<long long>(prime);
  long long reduced = diff % p;
  if (reduced < 0) {
    reduced += p;
  }
  return ren_mont_mul(static_cast<T>(reduced), r2, prime, p_inv);
}

/**
 * @brief Generates an array of ternary distributed random values.
 *
 * The distribution assigns a probability of 1/2 to 0, and 1/4 each
 * to -1 and 1.
 *
 * @param output the output array
 * @param key the ChaCha key
 * @param primes the RNS moduli
 * @param p_inv Montgomery reduction constants
 * @param r2 Montgomery encoding constants
 * @param stream_id the ChaCha nonce stream identifier
 * @param n the number of values to generate
 * @param num_rings the number of RNS moduli
 */
template <typename T>
__global__ void
ren_random_ternary(T *__restrict__ output, const unsigned int *__restrict__ key,
                   const T *__restrict__ primes, const T *__restrict__ p_inv,
                   const T *__restrict__ r2, unsigned long long stream_id,
                   unsigned long long n, unsigned long long num_rings) {
  unsigned long long coeff_idx = thread_index();
  if (coeff_idx < n) {
    unsigned int data[16];
    chacha_generate(data, key, stream_id, coeff_idx);
    auto value = signed_difference(data[0] & 1, (data[0] >> 1) & 1);
    for (unsigned long long ring_idx = 0; ring_idx < num_rings; ring_idx++) {
      const unsigned long long idx = coeff_idx + ring_idx * n;
      output[idx] =
          signed_to_mont(static_cast<typename RenWord<T>::signed_t>(value),
                         __ldg(&primes[ring_idx]), __ldg(&p_inv[ring_idx]),
                         __ldg(&r2[ring_idx]));
    }
  }
}

/**
 * @brief Generates an array of sparse ternary distributed random values.
 *
 * @param output the output array
 * @param key the ChaCha key
 * @param primes the RNS moduli
 * @param p_inv Montgomery reduction constants
 * @param r2 Montgomery encoding constants
 * @param stream_id the ChaCha nonce stream identifier
 * @param n the number of values to generate
 * @param num_rings the number of RNS moduli
 * @param h the number of nonzero values
 */
template <typename T>
__global__ void ren_random_sparse_ternary(
    T *__restrict__ output, const unsigned int *__restrict__ key,
    const T *__restrict__ primes, const T *__restrict__ p_inv,
    const T *__restrict__ r2, unsigned long long stream_id,
    unsigned long long n, unsigned long long num_rings, unsigned long long h) {
  // Currently there isn't a convenient way to specify that a node should have
  // its output buffer zeroed before the kernel runs, so for now we just do that
  // here.
  for (long long idx = n * num_rings - threadIdx.x - 1; idx >= 0;
       idx -= blockDim.x) {
    output[idx] = 0;
  }
  __syncthreads();
  for (unsigned long long ring_idx = threadIdx.x; ring_idx < num_rings;
       ring_idx += blockDim.x) {
    T *ring_output = output + n * ring_idx;
    unsigned long long remaining = h;
    unsigned long long block_offset = 0;
    while (remaining > 0) {
      if (block_offset >= (1ULL << 32)) {
        asm("trap;");
      }
      unsigned int data[16];
      chacha_generate(data, key, stream_id, block_offset++);
      for (int i = 0; i < 16 && remaining > 0; i++) {
        unsigned long long offset = data[i] & (n - 1);
        if (ring_output[offset] == 0) {
          auto value = signed_difference(1U, (data[i] >> 30) & 2);
          ring_output[offset] =
              signed_to_mont(static_cast<typename RenWord<T>::signed_t>(value),
                             __ldg(&primes[ring_idx]), __ldg(&p_inv[ring_idx]),
                             __ldg(&r2[ring_idx]));
          remaining--;
        }
      }
    }
  }
}

__device__ unsigned long long uniform_u64(const unsigned int *__restrict__ key,
                                          unsigned long long stream_id,
                                          unsigned long long sample_idx,
                                          unsigned long long sample_count,
                                          unsigned long long modulus) {
  for (unsigned long long attempt = 0;; attempt++) {
    unsigned long long block_offset = sample_idx + attempt * sample_count;
    if (block_offset >= (1ULL << 32)) {
      asm("trap;");
    }
    unsigned int data[16];
    chacha_generate(data, key, stream_id, block_offset);
    unsigned __int128 value = static_cast<unsigned __int128>(data[0]) |
                              (static_cast<unsigned __int128>(data[1]) << 32) |
                              (static_cast<unsigned __int128>(data[2]) << 64) |
                              (static_cast<unsigned __int128>(data[3]) << 96);
    unsigned long long remainder =
        static_cast<unsigned long long>(value % modulus);
    if (value - remainder + modulus - 1 >= value) {
      return remainder;
    }
  }
}

/**
 * @brief Generates an array of uniform random values.
 *
 * @param output the output array
 * @param key the ChaCha key
 * @param primes the RNS moduli
 * @param p_inv Montgomery reduction constants
 * @param r2 Montgomery encoding constants
 * @param stream_id the ChaCha nonce stream identifier
 * @param bound the bound; output is in the range [-bound,bound]
 * @param n the number of values to generate
 * @param num_rings the number of RNS moduli
 */
template <typename T>
__global__ void ren_random_uniform_bounded(
    T *__restrict__ output, const unsigned int *__restrict__ key,
    const T *__restrict__ primes, const T *__restrict__ p_inv,
    const T *__restrict__ r2, unsigned long long stream_id,
    unsigned long long bound, unsigned long long n,
    unsigned long long num_rings) {
  unsigned long long coeff_idx = thread_index();
  if (coeff_idx < n) {
    long long value = static_cast<long long>(uniform_u64(
                          key, stream_id, coeff_idx, n, 2 * bound + 1)) -
                      static_cast<long long>(bound);
    for (unsigned long long ring_idx = 0; ring_idx < num_rings; ring_idx++) {
      const unsigned long long idx = coeff_idx + ring_idx * n;
      output[idx] =
          signed_to_mont_wide(value, __ldg(&primes[ring_idx]),
                              __ldg(&p_inv[ring_idx]), __ldg(&r2[ring_idx]));
    }
  }
}

/**
 * @brief Generates an array of uniform random values.
 *
 * @param output the output array
 * @param key the ChaCha key
 * @param primes the RNS moduli
 * @param stream_id the ChaCha nonce stream identifier
 * @param n the number of values to generate
 * @param num_rings the number of RNS moduli
 */
template <typename T>
__global__ void ren_random_uniform_mod_q(T *__restrict__ output,
                                         const unsigned int *__restrict__ key,
                                         const T *__restrict__ primes,
                                         unsigned long long stream_id,
                                         unsigned long long n,
                                         unsigned long long num_rings) {
  unsigned long long idx = thread_index();
  unsigned long long ring_idx = idx / n;
  if (ring_idx < num_rings) {
    const unsigned long long modulus = __ldg(&primes[ring_idx]);
    output[idx] = static_cast<T>(
        uniform_u64(key, stream_id, idx, n * num_rings, modulus));
  }
}

/**
 * @brief Generates an array of random values using Knuth-Yao sampling.
 * @note See tables.py for tree generation.
 *
 * @param output the output array
 * @param key the ChaCha key
 * @param primes the RNS moduli
 * @param p_inv Montgomery reduction constants
 * @param r2 Montgomery encoding constants
 * @param stream_id the ChaCha nonce stream identifier
 * @param tree the tree describing the random number distribution
 * @param n the number of values to generate
 * @param num_rings the number of RNS moduli
 */
template <typename T, typename U>
__global__ void
ren_random_sample(T *__restrict__ output, const unsigned int *__restrict__ key,
                  const T *__restrict__ primes, const T *__restrict__ p_inv,
                  const T *__restrict__ r2, unsigned long long stream_id,
                  const tree_node<U> *__restrict__ tree, unsigned long long n,
                  unsigned long long num_rings) {
  unsigned long long coeff_idx = thread_index();
  if (coeff_idx < n) {
    unsigned int data[16];
    chacha_generate(data, key, stream_id, coeff_idx);
    auto value = static_cast<typename RenWord<T>::signed_t>(sample(tree, data));
    for (unsigned long long ring_idx = 0; ring_idx < num_rings; ring_idx++) {
      const unsigned long long idx = coeff_idx + ring_idx * n;
      output[idx] =
          signed_to_mont(value, __ldg(&primes[ring_idx]),
                         __ldg(&p_inv[ring_idx]), __ldg(&r2[ring_idx]));
    }
  }
}

/**
 * @brief Warp shuffle for __int128 values.
 *
 * Implemented by doing two 64-bit warp shuffles.
 */
__device__ __forceinline__ __int128
shfl_xor_sync_int128(unsigned int mask, __int128 value, int laneMask) {
  const unsigned __int128 value_unsigned =
      static_cast<unsigned __int128>(value);
  const unsigned long long lo_send =
      static_cast<unsigned long long>(value_unsigned);
  const unsigned long long hi_send =
      static_cast<unsigned long long>(value_unsigned >> 64);
  const unsigned long long lo_recv = __shfl_xor_sync(mask, lo_send, laneMask);
  const unsigned long long hi_recv = __shfl_xor_sync(mask, hi_send, laneMask);
  const unsigned __int128 ret_unsigned =
      (static_cast<unsigned __int128>(hi_recv) << 64) | lo_recv;
  return static_cast<__int128>(ret_unsigned);
}

/**
 * @brief Generates an array of discrete Gaussian distributed values.
 *
 * This function handles large standard deviations using the method described
 * in section 5.1 of https://eprint.iacr.org/2017/259.pdf.
 * It first samples some discrete Gaussians with smaller standard deviation
 * using Knuth-Yao sampling and then takes a linear combination.
 *
 * The caller checks that every possible sample fits in a signed 128-bit integer.
 *
 * @param output the output array
 * @param key the ChaCha key
 * @param primes the RNS moduli
 * @param p_inv Montgomery reduction constants
 * @param r2 Montgomery encoding constants
 * @param stream_id the ChaCha nonce stream identifier
 * @param tree the tree describing the random number distribution
 * @param z_values the values denoted z_i in the above mentioned paper
 * @param n the number of values to generate
 * @param num_rings the number of RNS moduli
 * @param num_z_values the number of z_i
 */
template <typename T, typename U>
__global__ void ren_random_large_discrete_gaussian(
    T *__restrict__ output, const unsigned int *__restrict__ key,
    const T *__restrict__ primes, const T *__restrict__ p_inv,
    const T *__restrict__ r2, unsigned long long stream_id,
    const tree_node<U> *__restrict__ tree,
    const unsigned long long *__restrict__ z_values, unsigned long long n,
    unsigned long long num_rings, unsigned long long num_z_values) {
  const unsigned long long coeff_idx = threadIdx.y + blockDim.y * blockIdx.x;
  bool active = coeff_idx < n;
  unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, active);
  if (active) {
    unsigned int data[16];
    chacha_generate(data, key, stream_id, coeff_idx * blockDim.x + threadIdx.x);
    // Compute two base samples per thread and combine them.
    const int value0 =
        static_cast<typename RenWord<T>::signed_t>(sample(tree, data));
    const int value1 =
        static_cast<typename RenWord<T>::signed_t>(sample(tree, data + 8));
    const __int128 z = __ldg(&z_values[num_z_values - 1]);
    // In this form of z*x + (z-1)*y, the intermediate product also fits.
    __int128 value = (z - 1) * (value0 + value1) + value0;
    // Combine base samples across threads to make the large sample.
    for (int z_idx = num_z_values - 2; z_idx >= 0; z_idx--) {
      const __int128 other = shfl_xor_sync_int128(warp_mask, value, 1 << z_idx);
      const __int128 x = (threadIdx.x >> z_idx) & 1 ? other : value;
      const unsigned long long z = __ldg(&z_values[z_idx]);
      value = (z - 1) * (value + other) + x;
    }
    // Convert the large sample to RNS form.
    for (unsigned long long ring_idx = threadIdx.x; ring_idx < num_rings;
         ring_idx += blockDim.x) {
      const unsigned long long idx = coeff_idx + ring_idx * n;
      const T q = __ldg(&primes[ring_idx]);
      auto value_reduced =
          static_cast<typename RenWord<T>::signed_t>(value % q);
      output[idx] = signed_to_mont(value_reduced, q, __ldg(&p_inv[ring_idx]),
                                   __ldg(&r2[ring_idx]));
    }
  }
}
