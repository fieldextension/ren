#pragma once

template <typename T> struct RenWord;

template <> struct RenWord<unsigned int> {
  using signed_t = int;
  using wide_t = unsigned long long;
  using signed_wide_t = long long;
  static constexpr unsigned int bits = 32;
};

template <> struct RenWord<unsigned long long> {
  using signed_t = long long;
  using wide_t = unsigned __int128;
  using signed_wide_t = __int128;
  static constexpr unsigned int bits = 64;
};

// Same-modulus Montgomery arithmetic context. Operations return canonical
// residues in [0, p).
template <typename T> struct RenMontCtx {
  T p;
  T p_inv;

  __device__ __host__ __forceinline__ RenMontCtx(T modulus, T modulus_inv)
      : p(modulus), p_inv(modulus_inv) {}

  __device__ __host__ __forceinline__ bool valid(T a) const { return a < p; }

  __device__ __host__ __forceinline__ T
  reduce(typename RenWord<T>::wide_t a) const {
    using Wide = typename RenWord<T>::wide_t;
    const Wide p_wide = (Wide)p;
    while (a >= p_wide)
      a -= p_wide;
    return (T)a;
  }

  __device__ __host__ __forceinline__ T add(T a, T b) const {
    T s = a + b;
    return (s >= p) ? (s - p) : s;
  }

  template <typename... Rest>
  __device__ __host__ __forceinline__ T add(T a, T b, Rest... rest) const {
    using Wide = typename RenWord<T>::wide_t;
    Wide acc = (Wide)a + (Wide)b;
    ((acc += (Wide)rest), ...);
    return reduce(acc);
  }

  __device__ __host__ __forceinline__ T sub(T a, T b) const {
    return (a >= b) ? (a - b) : (a + p - b);
  }

  __device__ __host__ __forceinline__ T neg(T a) const {
    return (a == 0) ? 0 : (p - a);
  }

  __device__ __host__ __forceinline__ T
  redc(typename RenWord<T>::wide_t z) const {
    using Wide = typename RenWord<T>::wide_t;
    T t = (T)z * p_inv;
    Wide u = (Wide)t * (Wide)p;
    T r = (T)((z + u) >> RenWord<T>::bits);
    return (r >= p) ? (r - p) : r;
  }

  __device__ __host__ __forceinline__ T mul(T a, T b) const {
    using Wide = typename RenWord<T>::wide_t;
    return redc((Wide)a * (Wide)b);
  }

  __device__ __forceinline__ void ntt_forward(T &a, T &b, T w) const {
    const T u = a;
    const T v = mul(b, w);
    a = add(u, v);
    b = sub(u, v);
  }

  __device__ __forceinline__ void ntt_inverse(T &a, T &b, T w) const {
    const T u = a;
    const T v = b;
    a = add(u, v);
    b = mul(sub(u, v), w);
  }
};
