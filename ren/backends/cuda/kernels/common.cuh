#pragma once

#ifndef __CUDACC_RTC__
#include <cassert>
#endif

#include "montgomery.cuh"

// Shared modular arithmetic helpers for RNS-aware CUDA kernels.
// Layout convention: [prime][coeff] = [p0[0..n-1], p1[0..n-1], ...]

/**
 * @returns the bit width of `p`
 */
__device__ __host__ __forceinline__ unsigned int
ren_bit_width_u64(unsigned long long p) {
  if (p == 0)
    return 0;
#if defined(__CUDA_ARCH__)
  return 64U - (unsigned int)__clzll(p);
#else
  return 64U - (unsigned int)__builtin_clzll(p);
#endif
}

enum class RenResidueMode {
  Balanced,
  Nonnegative,
};

// Nonnegative mode: acc_float is the sum of t_i/p_i and is always >= 0 for the
// lift paths that use this helper, so the plain (long long) cast is equivalent
// to floor() and keeps the correction from overshooting into negatives.
template <RenResidueMode Mode>
__device__ __host__ __forceinline__ long long
ren_residue_correction(double acc_float) {
  if constexpr (Mode == RenResidueMode::Nonnegative)
    return (long long)acc_float;
  return llround(acc_float);
}

// Basic same-modulus arithmetic on already-reduced residues.
template <typename T>
__device__ __host__ __forceinline__ T ren_mod_add(T a, T b, T p) {
  T s = a + b;
  return (s >= p) ? (s - p) : s;
}

template <typename T>
__device__ __host__ __forceinline__ T ren_mod_sub(T a, T b, T p) {
  return (a >= b) ? (a - b) : (a + p - b);
}

template <typename T>
__device__ __host__ __forceinline__ T ren_mod_neg(T a, T p) {
  return (a == 0) ? 0 : (p - a);
}

// Reduce a in [0, factor*p) to [0, p). factor >= 1, p > 0, and factor*p
// must fit in T.
template <typename T>
__device__ __host__ __forceinline__ T ren_mod_reduce(T a, T p,
                                                     unsigned int factor = 2) {
#if DEBUG >= 1
  using Wide = typename RenWord<T>::wide_t;
  const Wide bound = (Wide)p * factor;
  assert(factor > 0 && p > 0 && bound < ((Wide)1 << RenWord<T>::bits) &&
         (Wide)a < bound);
#endif
  while (factor > 1) {
    factor -= factor / 2; // Halve the bound, rounding up.
    const T multiple = p * factor;
    a = a >= multiple ? a - multiple : a;
  }
  return a;
}

template <typename T>
__device__ __host__ __forceinline__ T ren_reduce(unsigned long long a, T p) {
  const unsigned long long p_wide = (unsigned long long)p;
  while (a >= p_wide)
    a -= p_wide;
  return (T)a;
}

// Montgomery multiply: a*b must be below p*R, with p < R/2. A word-sized
// operand and a canonical operand satisfy this bound. The lazy result is in
// [0, 2p). The caller chooses operand/table forms to preserve the intended domain.
template <typename T>
__device__ __host__ __forceinline__ T ren_mont_mul_lazy(T a, T b, T p,
                                                        T p_inv) {
  using Wide = typename RenWord<T>::wide_t;
  Wide z = (Wide)a * (Wide)b;
  T t = (T)z * p_inv;
  Wide u = (Wide)t * (Wide)p;
  T r = (T)((z + u) >> RenWord<T>::bits);
#if DEBUG >= 1
  assert(r < p + p);
#endif
  return r;
}

// Canonical Montgomery result in [0, p), sharing the lazy multiply body.
template <typename T>
__device__ __host__ __forceinline__ T ren_mont_mul(T a, T b, T p, T p_inv) {
  return ren_mod_reduce(ren_mont_mul_lazy(a, b, p, p_inv), p);
}

template <typename T, unsigned long long NUM_PARTIALS>
__device__ __forceinline__ T
ren_lift_partials_approx(const unsigned long long (&partials)[NUM_PARTIALS],
                         T q, T pinv_q, const T *mi_r2_scaled_row) {
  using Wide = typename RenWord<T>::wide_t;
  constexpr unsigned int bits = RenWord<T>::bits;
  // Keep the sum of products below q*R so one Montgomery reduction finishes the lift.
  Wide z = 0;
  for (unsigned long long src_idx = 0; src_idx < NUM_PARTIALS; ++src_idx) {
    z += (Wide)(T)partials[src_idx] * (Wide)__ldg(&mi_r2_scaled_row[src_idx]);
    if ((T)(z >> bits) >= q)
      z -= (Wide)q << bits;
  }
  const T t = (T)z * pinv_q;
  return ren_mod_reduce((T)((z + (Wide)t * (Wide)q) >> bits), q);
}

template <typename T, unsigned long long NUM_PARTIALS>
__device__ __forceinline__ T ren_lift_partials_exact(
    const unsigned long long (&partials)[NUM_PARTIALS], long long correction,
    T q, T pinv_q, const T *mi_r2_scaled_row, T prod_r2_scaled) {
  const T acc = ren_lift_partials_approx<T, NUM_PARTIALS>(partials, q, pinv_q,
                                                          mi_r2_scaled_row);
  const T correction_mod = ren_mont_mul(
      prod_r2_scaled, (T)((unsigned long long)correction), q, pinv_q);
  return ren_mod_sub(acc, correction_mod, q);
}
