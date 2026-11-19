#include "common.cuh"

// Per-coefficient base change: each thread loads all source limbs for one
// coefficient, computes every destination limb for that coefficient, then
// writes them back. That keeps the source coefficient hot in registers instead
// of re-reading the same source limbs once per destination prime.
//
// NumSrc and NumDst are template-provided compile-time basis sizes for this
// kernel instance, and Mode selects the residue convention at compile time.
/**
 * @brief Performs coefficient-domain ring transport between RNS bases.
 *
 * Storage stays canonical Montgomery at the boundaries:
 * - src[i * n + j] is coefficient j modulo primes_src[i], in Montgomery form
 * - dst[i * n + j] is coefficient j modulo primes_dst[i], also in Montgomery form
 *
 * Let m be the product of the source primes and m_i = m / primes_src[i]. This
 * kernel keeps the standard exact-base-change algorithm, but updates the table
 * conventions to match canonical Montgomery storage:
 * - inv_src[i] stores m_i^{-1} mod p_i in normal form, so
 *   mont_mul(src_i_mont, inv_src[i]) produces the normal-form partial t_i
 * - mi_r2_scaled[d * NumSrc + i] stores (m_i mod q_d) * R^2 mod q_d, so
 *   mont_mul(t_i, mi_r2_scaled[...]) re-enters the destination modulus in
 *   Montgomery form
 * - prod_r2_scaled[d] stores (m mod q_d) * R^2 mod q_d for the correction term
 *   chosen by the requested residue system
 *
 * The residue system is selected at compile time:
 * - balanced residues use round(sum_i t_i / p_i)
 * - nonnegative residues use floor(sum_i t_i / p_i)
 *
 * Each normal-form partial fits in T. Multiplication by a canonical table
 * weight satisfies the Montgomery reduction bound without first reducing the
 * partial modulo the destination prime.
 *
 * @param src source polynomial in coefficient domain.
 * @param dst destination polynomial in coefficient domain.
 * @param n number of coefficients in the polynomial.
 * @param primes_src source primes.
 * @param primes_dst destination primes.
 * @param p_inv_src Montgomery reduction constants for the source primes.
 * @param p_inv_dst Montgomery reduction constants for the destination primes.
 * @param dst_src_indices source-basis index for each destination prime, or
 *                        UINT32_MAX when that destination prime is newly added.
 * @param inv_src m_i^{-1} mod p_i in normal form.
 * @param mi_r2_scaled (m_i mod q_d) * R^2 mod q_d, laid out as [dst_idx][src_idx].
 * @param prod_r2_scaled (m mod q_d) * R^2 mod q_d for the residue correction term.
 */
template <typename T, unsigned long long NumSrc, unsigned long long NumDst,
          RenResidueMode Mode>
__global__ void ren_base_change_coeff(
    const T *__restrict__ src, T *__restrict__ dst, unsigned long long n,
    const T *__restrict__ primes_src, const T *__restrict__ primes_dst,
    const T *__restrict__ p_inv_src, const T *__restrict__ p_inv_dst,
    const unsigned int *__restrict__ dst_src_indices,
    const T *__restrict__ inv_src, const T *__restrict__ mi_r2_scaled,
    const T *__restrict__ prod_r2_scaled) {
  static_assert(NumSrc > 0, "NumSrc must be positive");
  static_assert(NumDst > 0, "NumDst must be positive");

  unsigned long long coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (coeff_idx >= n)
    return;

  unsigned long long partials[NumSrc];
  double acc_float = 0.0;

  for (unsigned long long src_idx = 0; src_idx < NumSrc; ++src_idx) {
    const T p_src = __ldg(&primes_src[src_idx]);
    const T pinv_src = __ldg(&p_inv_src[src_idx]);
    const T a_i = __ldg(&src[src_idx * n + coeff_idx]);

#if DEBUG >= 1
    assert((unsigned long long)a_i < (unsigned long long)p_src);
#endif

    const T t_i = ren_mont_mul(a_i, __ldg(&inv_src[src_idx]), p_src, pinv_src);
    partials[src_idx] = (unsigned long long)t_i;
    acc_float += (double)t_i / (double)p_src;
  }

  // When this kernel is used to perform the mod-raise part of a keyswitch,
  // errors introduced by floating point are insignificant. The rounding error
  // can cause entries of dst to be off by the product of the primes in
  // primes_src. The keyswitch later multiplies the result by the key-switching
  // key and then does a rescale that divides by a number that is larger than
  // this product.  The net result is that the error for the full keyswitch is
  // at most the noise term in the key-switching key.
  //
  // When this kernel is used to raise the modulus of a secret key, and the
  // residue mode is Balanced, the floating point approximation will never
  // cause an error, because the secret key is never close to the rounding
  // boundary.
  //
  // There are some places in the code that lift a polynomial from modulus Q to
  // PQ and then multiply by P.  Multiplication by P kills the correction term,
  // so rounding errors have no effect.
  const long long correction = ren_residue_correction<Mode>(acc_float);

  for (unsigned long long dst_idx = 0; dst_idx < NumDst; ++dst_idx) {
    const unsigned int src_idx = __ldg(&dst_src_indices[dst_idx]);
    if (src_idx != 0xFFFFFFFFU) {
      dst[dst_idx * n + coeff_idx] =
          __ldg(&src[(unsigned long long)src_idx * n + coeff_idx]);
      continue;
    }

    const T q = __ldg(&primes_dst[dst_idx]);
    const T pinv_q = __ldg(&p_inv_dst[dst_idx]);
    dst[dst_idx * n + coeff_idx] = ren_lift_partials_exact<T, NumSrc>(
        partials, correction, q, pinv_q, &mi_r2_scaled[dst_idx * NumSrc],
        __ldg(&prod_r2_scaled[dst_idx]));
  }
}
