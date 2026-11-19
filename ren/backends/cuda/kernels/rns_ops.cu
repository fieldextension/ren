#include "common.cuh"

/**
 * @brief Materialize signed two's-complement coefficient limbs into canonical
 *        RNS Montgomery storage.
 *
 * The source uses u32 limb-major layout [limb][coeff], little-endian by limb.
 * The destination uses Ren's standard RNS layout [prime][coeff], with each
 * limb stored as (coeff mod p_i) * R mod p_i.
 */
template <typename T, unsigned int LimbCount>
__global__ void
ren_to_rns_limbs(const unsigned int *__restrict__ src, T *__restrict__ dst,
                 unsigned long long n, unsigned long long num_primes,
                 const T *__restrict__ primes, const T *__restrict__ p_inv,
                 const T *__restrict__ r2, const T *__restrict__ radix32) {
  static_assert(LimbCount > 0, "LimbCount must be positive");

  unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total = n * num_primes;
  if (gid >= total)
    return;

  unsigned long long p_idx = gid / n;
  unsigned long long coeff_idx = gid - p_idx * n;
  const T q = __ldg(&primes[p_idx]);
  const RenMontCtx<T> mont(q, __ldg(&p_inv[p_idx]));
  const T r2_p = __ldg(&r2[p_idx]);
  const T radix = __ldg(&radix32[p_idx]);
  // Sign-extend the coefficient before folding its two's-complement limbs.
  const bool is_negative =
      (src[(unsigned long long)(LimbCount - 1) * n + coeff_idx] &
       0x80000000U) != 0;
  const T sign = is_negative ? mont.neg(mont.mul((T)1, r2_p)) : (T)0;
  T acc = sign;
  for (int w = (int)LimbCount - 1; w >= 0; --w)
    acc =
        mont.add(mont.mul(acc, radix),
                 mont.mul((T)src[(unsigned long long)w * n + coeff_idx], r2_p));
  dst[gid] = acc;
}

// FROM_RNS stores words in little-endian order. Reduce the occupied words
// using successive powers of 2^32.
template <typename T, unsigned int LimbCount>
__device__ __forceinline__ T bignum_to_mont(const unsigned int *limbs,
                                            unsigned int limb_count,
                                            const RenMontCtx<T> &mont, T r2,
                                            T radix) {
  T sum = 0, power = r2; // power = 2^(32*w) * R^2 mod p
#pragma unroll
  for (unsigned int w = 0; w < LimbCount; ++w) {
    if (w >= limb_count)
      break;
    sum = mont.add(sum, mont.mul((T)limbs[w], power));
    power = mont.mul(power, radix);
  }
  return sum;
}

// Update the occupied words, then write the carry into the new words.
// Carry beyond LimbCount is zero because the sum stays below Q.
template <typename T, unsigned int LimbCount>
__device__ __forceinline__ void
bignum_add_scaled(unsigned int *acc, const unsigned int *__restrict__ pp, T t,
                  unsigned int limb_count) {
  using Wide = typename RenWord<T>::wide_t;
  constexpr unsigned int Words = RenWord<T>::bits / 32;
  T carry = 0;
#pragma unroll
  for (unsigned int w = 0; w < LimbCount; ++w) {
    if (w < limb_count) {
      const Wide cur = (Wide)t * __ldg(&pp[w]) + acc[w] + carry;
      acc[w] = (unsigned int)cur;
      carry = (T)(cur >> 32);
    } else {
      if (w >= limb_count + Words)
        break;
      acc[w] = (unsigned int)carry;
      carry = (T)((unsigned long long)carry >> 32);
    }
  }
}

// Strict bignum compare a > b (both LimbCount words, little-endian).
template <unsigned int LimbCount>
__device__ __forceinline__ bool bignum_gt(const unsigned int *a,
                                          const unsigned int *__restrict__ b) {
#pragma unroll
  for (int w = (int)LimbCount - 1; w >= 0; --w) {
    const unsigned int bw = __ldg(&b[w]);
    if (a[w] != bw)
      return a[w] > bw;
  }
  return false; // equal -> not strictly greater
}

// acc -= q (mod 2^(32*LimbCount)); for acc < q the result is the two's-complement
// encoding of the negative value acc - q.
template <unsigned int LimbCount>
__device__ __forceinline__ void bignum_sub(unsigned int *acc,
                                           const unsigned int *__restrict__ q) {
  long long borrow = 0;
#pragma unroll
  for (unsigned int w = 0; w < LimbCount; ++w) {
    const long long d = (long long)acc[w] - (long long)__ldg(&q[w]) - borrow;
    acc[w] = (unsigned int)((unsigned long long)d & 0xFFFFFFFFULL);
    borrow = (d < 0) ? 1 : 0;
  }
}

/**
 * @brief Reconstruct integer coefficients from RNS Montgomery limbs (inverse of
 *        ren_to_rns_limbs).
 *
 * One thread per coefficient runs Garner's mixed-radix CRT into a
 * LimbCount-word bignum giving the [0, Q) value, then
 * centers to the balanced representative [-Q/2, Q/2) and writes it as signed
 * two's-complement u32 limbs in [limb][coeff] layout -- the same encoding
 * ren_to_rns_limbs consumes.
 */
template <typename T, unsigned int K, unsigned int LimbCount>
__global__ void
ren_from_rns_limbs(const T *__restrict__ src, unsigned int *__restrict__ dst,
                   unsigned long long n, const T *__restrict__ primes,
                   const T *__restrict__ p_inv, const T *__restrict__ r2,
                   const T *__restrict__ radix32,
                   const T *__restrict__ crt_inverses,
                   const unsigned int *__restrict__ crt_partial_prods,
                   const unsigned int *__restrict__ crt_q,
                   const unsigned int *__restrict__ crt_q_half) {
  static_assert(K > 0, "K must be positive");
  static_assert(LimbCount > 0, "LimbCount must be positive");

  const unsigned long long coeff = blockIdx.x * blockDim.x + threadIdx.x;
  if (coeff >= n)
    return;

  constexpr unsigned int Words = RenWord<T>::bits / 32;
  unsigned int acc[LimbCount];
  const T q0 = __ldg(&primes[0]);
  const RenMontCtx<T> mont0(q0, __ldg(&p_inv[0]));
  const T first = mont0.mul(__ldg(&src[coeff]), (T)1);
  acc[0] = (unsigned int)first;
  if constexpr (Words > 1 && LimbCount > 1)
    acc[1] = (unsigned int)((unsigned long long)first >> 32);

  // Each prime fits in Words u32 words, so acc and partial_prods[i] fit
  // in i * Words words; the remaining partial-product words are zero.
  // Garner mixed-radix CRT: acc = sum_i t_i * partial_prods[i], t_i in [0, p_i).
  for (unsigned int i = 1; i < K; ++i) {
    const unsigned int used = i * Words < LimbCount ? i * Words : LimbCount;
    const T q = __ldg(&primes[i]);
    const RenMontCtx<T> mont(q, __ldg(&p_inv[i]));
    const T r = bignum_to_mont<T, LimbCount>(acc, used, mont, __ldg(&r2[i]),
                                             __ldg(&radix32[i]));
    const T diff = mont.sub(__ldg(&src[(unsigned long long)i * n + coeff]), r);
    // diff is Montgomery-encoded; the normal-form inverse leaves t unencoded.
    const T t = mont.mul(diff, __ldg(&crt_inverses[i]));
    bignum_add_scaled<T, LimbCount>(
        acc, &crt_partial_prods[(unsigned long long)i * LimbCount], t, used);
  }

  // Center to [-Q/2, Q/2): subtract Q iff acc > floor(Q/2).
  if (bignum_gt<LimbCount>(acc, crt_q_half))
    bignum_sub<LimbCount>(acc, crt_q);

#pragma unroll
  for (unsigned int w = 0; w < LimbCount; ++w)
    dst[(unsigned long long)w * n + coeff] = acc[w];
}

/**
 * @brief Extends a coefficient-domain polynomial by adding new primes.
 *
 * Existing limbs are copied through unchanged because storage is already
 * canonical Montgomery under the old basis. Newly added limbs are computed
 * with the same residue-aware lift as the general base-change kernel,
 * specialized to the prefix case where the old primes are the prefix of the
 * new basis.
 *
 * Let m be the product of the old primes and m_i = m / primes_all[i] for
 * i < NumOld. NumOld and NumNew are compile-time basis sizes, and Mode selects
 * the residue convention at compile time.
 *
 * @param src input polynomial. src[i * n + j] is coefficient j modulo
 *            primes_all[i] for i < NumOld, stored in Montgomery form.
 * @param dst output polynomial. dst[i * n + j] is coefficient j modulo
 *            primes_all[i], also stored in Montgomery form.
 * @param n number of coefficients in the polynomial.
 * @param primes_all all output primes, with the old basis as a prefix.
 * @param p_inv_all Montgomery reduction constants for primes_all.
 * @param inv_src m_i^{-1} mod p_i in normal form for the old basis.
 * @param mi_r2_scaled (m_i mod q_j) * R^2 mod q_j, laid out as [new_prime][old_idx].
 * @param prod_r2_scaled (m mod q_j) * R^2 mod q_j for the residue correction term.
 */
template <typename T, unsigned long long NumOld, unsigned long long NumNew,
          RenResidueMode Mode>
__global__ void ren_rns_prefix_extend(const T *__restrict__ src,
                                      T *__restrict__ dst, unsigned long long n,
                                      const T *__restrict__ primes_all,
                                      const T *__restrict__ p_inv_all,
                                      const T *__restrict__ inv_src,
                                      const T *__restrict__ mi_r2_scaled,
                                      const T *__restrict__ prod_r2_scaled) {
  static_assert(NumOld > 0, "NumOld must be positive");
  static_assert(NumNew > NumOld,
                "NumNew must exceed NumOld for prefix extension");

  unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total = n * NumNew;
  if (gid >= total)
    return;

  unsigned long long p_idx = gid / n;
  unsigned long long coeff_idx = gid - p_idx * n;
  const T q = __ldg(&primes_all[p_idx]);

  if (p_idx < NumOld) {
    const T value = __ldg(&src[p_idx * n + coeff_idx]);
#if DEBUG >= 1
    assert((unsigned long long)value < (unsigned long long)q);
#endif
    dst[gid] = value;
    return;
  }

  const T pinv_q = __ldg(&p_inv_all[p_idx]);

  T acc = 0;
  double acc_float = 0.0;
  for (unsigned long long old_idx = 0; old_idx < NumOld; ++old_idx) {
    const T p_old = __ldg(&primes_all[old_idx]);
    const T pinv_old = __ldg(&p_inv_all[old_idx]);
    const T a_i = __ldg(&src[old_idx * n + coeff_idx]);

#if DEBUG >= 1
    assert((unsigned long long)a_i < (unsigned long long)p_old);
#endif

    const T t_i = ren_mont_mul(a_i, __ldg(&inv_src[old_idx]), p_old, pinv_old);
    const T term = ren_mont_mul(
        t_i, __ldg(&mi_r2_scaled[p_idx * NumOld + old_idx]), q, pinv_q);
    acc = ren_mod_add(acc, term, q);
    acc_float += (double)t_i / (double)p_old;
  }

  // When this kernel is used to perform the mod-raise part of a bootstrap, errors introduced by floating point are insignificant.
  // If mod-eval is designed to be accurate on [-K, K], then it will be accurate on a very slightly larger interval.
  const long long correction = ren_residue_correction<Mode>(acc_float);
  const T correction_mod =
      ren_mont_mul(__ldg(&prod_r2_scaled[p_idx]),
                   (T)((unsigned long long)correction), q, pinv_q);
  dst[gid] = ren_mod_sub(acc, correction_mod, q);
}
