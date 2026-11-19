#include "common.cuh"

/**
 * @brief Rounded rescale when the destination ring is a quotient of the source.
 *
 * Each thread computes the removed-basis lift once for one coefficient, then
 * reuses it across all destination limbs.
 */
template <typename T, unsigned long long NumRemoved, unsigned long long NumDst>
__global__ void ren_rescale_quotient(
    const T *__restrict__ src, T *__restrict__ dst, unsigned long long n,
    const T *__restrict__ src_primes, const T *__restrict__ dst_primes,
    const T *__restrict__ src_p_inv, const T *__restrict__ dst_p_inv,
    const unsigned int *__restrict__ removed_src_indices,
    const unsigned int *__restrict__ dst_src_indices,
    const T *__restrict__ inv_src, const T *__restrict__ mi_r2_scaled,
    const T *__restrict__ prod_r2_scaled, const T *__restrict__ inv_prod_mont) {
  static_assert(NumRemoved > 0, "NumRemoved must be positive");
  static_assert(NumDst > 0, "NumDst must be positive");

  const unsigned long long coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (coeff_idx >= n)
    return;

  unsigned long long partials[NumRemoved];
  double acc_float = 0.0;
  for (unsigned long long removed_idx = 0; removed_idx < NumRemoved;
       ++removed_idx) {
    const unsigned int src_idx = __ldg(&removed_src_indices[removed_idx]);
    const T b = __ldg(&src_primes[src_idx]);
    const T pinv_b = __ldg(&src_p_inv[src_idx]);
    const T a_b = __ldg(&src[(unsigned long long)src_idx * n + coeff_idx]);

#if DEBUG >= 1
    assert((unsigned long long)a_b < (unsigned long long)b);
#endif

    const T t = ren_mont_mul(a_b, __ldg(&inv_src[removed_idx]), b, pinv_b);
    partials[removed_idx] = (unsigned long long)t;
    if constexpr (NumRemoved != 1) {
      acc_float += (double)t / (double)b;
    }
  }

  long long correction;
  if constexpr (NumRemoved == 1) {
    const unsigned int src_idx = __ldg(&removed_src_indices[0]);
    correction =
        partials[0] > (unsigned long long)__ldg(&src_primes[src_idx]) / 2;
  } else {
    // Errors introduced by using floating point are insignificant.
    // The kernel divides coefficients by a constant and rounds them to the nearest integer.
    // Floating point rounding error would cause a coefficient near the boundary to be rounded the wrong way.
    correction = llround(acc_float);
  }

  for (unsigned long long dst_idx = 0; dst_idx < NumDst; ++dst_idx) {
    const T q = __ldg(&dst_primes[dst_idx]);
    const T pinv_q = __ldg(&dst_p_inv[dst_idx]);
    const unsigned int src_idx = __ldg(&dst_src_indices[dst_idx]);

#if DEBUG >= 1
    assert(src_idx != 0xFFFFFFFFU);
#endif

    const T shared_term =
        __ldg(&src[(unsigned long long)src_idx * n + coeff_idx]);

#if DEBUG >= 1
    assert((unsigned long long)shared_term < (unsigned long long)q);
#endif

    const T removed_lift = ren_lift_partials_exact<T, NumRemoved>(
        partials, correction, q, pinv_q, &mi_r2_scaled[dst_idx * NumRemoved],
        __ldg(&prod_r2_scaled[dst_idx]));
    const T diff = ren_mod_sub(shared_term, removed_lift, q);
    dst[dst_idx * n + coeff_idx] =
        ren_mont_mul(diff, __ldg(&inv_prod_mont[dst_idx]), q, pinv_q);
  }
}

/**
 * @brief Rounded rescale for a scheduled mixed-basis transition.
 *
 * This handles the case where the destination basis drops some source primes
 * and adds new ones. The kernel first scales both the shared and removed limbs
 * by the product of destination-only primes, then performs the removed-basis
 * lift with the CRT rounding correction, subtracts it from the shared
 * destination contribution, and finally divides by the removed prime product.
 *
 * @param src input polynomial in coefficient domain over the source basis.
 * @param dst output polynomial in coefficient domain over the destination basis.
 * @param n number of coefficients in the polynomial.
 * @param src_primes source primes in source-basis order.
 * @param dst_primes destination primes in destination-basis order.
 * @param src_p_inv Montgomery reduction constants for the source primes.
 * @param dst_p_inv Montgomery reduction constants for the destination primes.
 * @param removed_src_indices source-basis indices for the removed primes.
 * @param dst_src_indices source-basis indices for each destination prime, or
 *                        UINT32_MAX when the destination prime is newly added.
 * @param inv_src y_i^{-1} mod removed_i in normal form.
 * @param mi_r2_scaled (y_i mod dst_j) * R^2 mod dst_j, laid out as [dst_idx][removed_idx].
 * @param prod_r2_scaled (prod removed_i mod dst_j) * R^2 mod dst_j for the CRT
 *                    rounding correction.
 * @param inv_prod_mont (prod removed_i)^{-1} mod dst_j in Montgomery form.
 * @param added_prod_removed_mont added_prod encoded in Montgomery form over the
 *                                removed basis.
 * @param added_prod_dst_mont added_prod encoded in Montgomery form over the
 *                            destination basis.
 */
template <typename T, unsigned long long NumRemoved, unsigned long long NumDst>
__global__ void ren_rescale_mixed(
    const T *__restrict__ src, T *__restrict__ dst, unsigned long long n,
    const T *__restrict__ src_primes, const T *__restrict__ dst_primes,
    const T *__restrict__ src_p_inv, const T *__restrict__ dst_p_inv,
    const unsigned int *__restrict__ removed_src_indices,
    const unsigned int *__restrict__ dst_src_indices,
    const T *__restrict__ inv_src, const T *__restrict__ mi_r2_scaled,
    const T *__restrict__ prod_r2_scaled, const T *__restrict__ inv_prod_mont,
    const T *__restrict__ added_prod_removed_mont,
    const T *__restrict__ added_prod_dst_mont) {
  static_assert(NumRemoved > 0, "NumRemoved must be positive");
  static_assert(NumDst > 0, "NumDst must be positive");

  const unsigned long long coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (coeff_idx >= n)
    return;

  unsigned long long partials[NumRemoved];
  double acc_float = 0.0;
  for (unsigned long long removed_idx = 0; removed_idx < NumRemoved;
       ++removed_idx) {
    const unsigned int src_idx = __ldg(&removed_src_indices[removed_idx]);
    const T b = __ldg(&src_primes[src_idx]);
    const T pinv_b = __ldg(&src_p_inv[src_idx]);
    const T a_b = __ldg(&src[(unsigned long long)src_idx * n + coeff_idx]);

#if DEBUG >= 1
    assert((unsigned long long)a_b < (unsigned long long)b);
#endif

    const T scaled_b = ren_mont_mul(
        a_b, __ldg(&added_prod_removed_mont[removed_idx]), b, pinv_b);
    const T t = ren_mont_mul(scaled_b, __ldg(&inv_src[removed_idx]), b, pinv_b);
    partials[removed_idx] = (unsigned long long)t;
    acc_float += (double)t / (double)b;
  }

  // Errors introduced by using floating point are insignificant for the same
  // reason as in ren_rescale_quotient above.
  const long long correction = llround(acc_float);

  for (unsigned long long dst_idx = 0; dst_idx < NumDst; ++dst_idx) {
    const T q = __ldg(&dst_primes[dst_idx]);
    const T pinv_q = __ldg(&dst_p_inv[dst_idx]);
    const unsigned int src_idx = __ldg(&dst_src_indices[dst_idx]);

    T shared_term = 0;
    if (src_idx != 0xFFFFFFFFU) {
      const T a_q = __ldg(&src[(unsigned long long)src_idx * n + coeff_idx]);
#if DEBUG >= 1
      assert((unsigned long long)a_q < (unsigned long long)q);
#endif
      shared_term =
          ren_mont_mul(a_q, __ldg(&added_prod_dst_mont[dst_idx]), q, pinv_q);
    }

    const T removed_lift = ren_lift_partials_exact<T, NumRemoved>(
        partials, correction, q, pinv_q, &mi_r2_scaled[dst_idx * NumRemoved],
        __ldg(&prod_r2_scaled[dst_idx]));
    const T diff = ren_mod_sub(shared_term, removed_lift, q);
    dst[dst_idx * n + coeff_idx] =
        ren_mont_mul(diff, __ldg(&inv_prod_mont[dst_idx]), q, pinv_q);
  }
}
