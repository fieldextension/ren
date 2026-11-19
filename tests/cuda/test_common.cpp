#include "common.cuh"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

template <typename T> T test_mont_p_inv(T p) {
  T inv = 1;
  for (unsigned int bits = 1; bits < RenWord<T>::bits; bits <<= 1) {
    inv *= (T)(2 - p * inv);
  }
  return (T)(0 - inv);
}

template <typename T> T test_to_mont(T a, T p) {
  const unsigned __int128 r_mod_p =
      (((unsigned __int128)1) << RenWord<T>::bits) % p;
  return (T)(((unsigned __int128)a * r_mod_p) % p);
}

template <typename T> T test_mul_mod(T a, T b, T p) {
  return (T)(((unsigned __int128)a * (unsigned __int128)b) % p);
}

TEST_SUITE("common.cuh modules") {

  TEST_CASE("ren_bit_width_u64") {
    SUBCASE("a == 0") { CHECK(ren_bit_width_u64(0) == 0); }

    SUBCASE("a is a power of 2") { CHECK(ren_bit_width_u64(1ULL << 42) == 43); }

    SUBCASE("a has several bits set") {
      CHECK(ren_bit_width_u64(0b100101001010001) == 15);
    }
  }

  TEST_CASE("ren_mod_add") {
    SUBCASE("u32") {
      unsigned int max_p = 1U << 31;
      CHECK(ren_mod_add<unsigned int>(4U, 7U, 16U) == 11U);
      CHECK(ren_mod_add<unsigned int>(15U, 4U, 19U) == 0U);
      CHECK(ren_mod_add<unsigned int>(23U, 53U, 61U) == 15U);
      CHECK(ren_mod_add<unsigned int>(max_p - 2, max_p - 3, max_p) ==
            max_p - 5);
    }

    SUBCASE("u64") {
      unsigned long long max_p = 1ULL << 63;
      CHECK(ren_mod_add<unsigned long long>(4ULL, 7ULL, 16ULL) == 11ULL);
      CHECK(ren_mod_add<unsigned long long>(15ULL, 4ULL, 19ULL) == 0ULL);
      CHECK(ren_mod_add<unsigned long long>(23ULL, 53ULL, 61ULL) == 15ULL);
      CHECK(ren_mod_add<unsigned long long>(max_p - 2, max_p - 3, max_p) ==
            max_p - 5);
    }
  }

  TEST_CASE("ren_mod_sub") {
    SUBCASE("u32") {
      unsigned int max_p = 1U << 31;
      CHECK(ren_mod_sub<unsigned int>(15U, 7U, 17U) == 8U);
      CHECK(ren_mod_sub<unsigned int>(58U, 58U, 90U) == 0U);
      CHECK(ren_mod_sub<unsigned int>(64U, 248U, 503U) == 319U);
      CHECK(ren_mod_sub<unsigned int>(max_p - 2, max_p - 1, max_p) ==
            max_p - 1);
    }

    SUBCASE("u64") {
      unsigned long long max_p = 1ULL << 63;
      CHECK(ren_mod_sub<unsigned long long>(15ULL, 7ULL, 17ULL) == 8ULL);
      CHECK(ren_mod_sub<unsigned long long>(58ULL, 58ULL, 90ULL) == 0ULL);
      CHECK(ren_mod_sub<unsigned long long>(64ULL, 248ULL, 503ULL) == 319ULL);
      CHECK(ren_mod_sub<unsigned long long>(max_p - 2, max_p - 1, max_p) ==
            max_p - 1);
    }
  }

  TEST_CASE("ren_mod_neg") {
    SUBCASE("u32") {
      CHECK(ren_mod_neg<unsigned int>(15U, 18U) == 3U);
      CHECK(ren_mod_neg<unsigned int>(0U, 65U) == 0U);
    }

    SUBCASE("u64") {
      CHECK(ren_mod_neg<unsigned long long>(15ULL, 18ULL) == 3ULL);
      CHECK(ren_mod_neg<unsigned long long>(0ULL, 65ULL) == 0ULL);
    }
  }

  TEST_CASE_TEMPLATE("ren_mod_reduce", T, unsigned int, unsigned long long) {
    const unsigned int factors[] = {1, 2, 3, 4, 5, 8};
    for (unsigned int factor : factors) {
      const T moduli[] = {(T)17, (T)65537, (T)((T)-1 / factor)};
      for (T p : moduli) {
        // Test both sides of every modulus boundary, including factor*p - 1.
        for (unsigned int k = 0; k < factor; ++k) {
          const T values[] = {(T)(k * p), (T)(k * p + 1), (T)(k * p + p - 1)};
          for (T a : values) {
            const T reduced = ren_mod_reduce(a, p, factor);
            CHECK(reduced < p);
            CHECK(reduced == a % p);
            if (factor == 2)
              CHECK(ren_mod_reduce(a, p) == reduced);
          }
        }
      }
    }
  }

  TEST_CASE_TEMPLATE("ren_mod_reduce preserves lazy NTT ranges", T,
                     unsigned int, unsigned long long) {
    const T moduli[] = {(T)17, (T)(((T)1 << (RenWord<T>::bits - 2)) - 1)};
    const unsigned int multiples[] = {1, 2};
    for (T q : moduli) {
      for (unsigned int multiple : multiples) {
        const T p = multiple * q;
        const T values[] = {(T)0, (T)(p - 1), p, (T)(p + 1), (T)(2 * p - 1)};
        for (T a : values) {
          const T reduced = ren_mod_reduce(a, p);
          CHECK(reduced < p);
          CHECK(reduced == a % p);
          CHECK(reduced % q == a % q);
        }
      }
    }
  }

  TEST_CASE_TEMPLATE("ren_mont_mul accepts a word-sized operand", T,
                     unsigned int, unsigned long long) {
    const T large_prime =
        sizeof(T) == 4 ? (T)2013265921ULL : (T)2305843009213693951ULL;
    const T near_word_limit_prime =
        sizeof(T) == 4 ? (T)2147483647ULL : (T)9223372036854775783ULL;
    const T primes[] = {(T)65537, large_prime, near_word_limit_prime};
    for (T p : primes) {
      const T p_inv = test_mont_p_inv(p);
      const T operands[] = {(T)0,       (T)1,           (T)(p - 1),     p,
                            (T)(p + 1), (T)((T)-1 / 2), (T)((T)-1 - 1), (T)-1};
      const T residues[] = {(T)0, (T)1, (T)(p / 2), (T)(p - 1)};
      bool saw_noncanonical = false;
      for (T a : operands)
        for (T b : residues) {
          const T b_mont = test_to_mont(b, p);
          const T expected = test_mul_mod((T)(a % p), b, p);
          const T raw = ren_mont_mul_lazy(a, b_mont, p, p_inv);
          const T reduced = ren_mont_mul(a, b_mont, p, p_inv);
          CHECK(raw < 2 * p);
          CHECK(raw % p == expected);
          CHECK(reduced == expected);
          saw_noncanonical |= raw >= p;
        }
      CHECK(saw_noncanonical);
    }
  }

  TEST_CASE("ren_mont_mul") {
    SUBCASE("u32 montgomery x normal -> normal") {
      const unsigned int p = 2013265921U;
      const unsigned int p_inv = test_mont_p_inv(p);
      const unsigned int x = 1234567U;
      const unsigned int y = 7654321U;
      const unsigned int x_mont = test_to_mont(x, p);

      CHECK(ren_mont_mul<unsigned int>(x_mont, 1U, p, p_inv) == x);
      CHECK(ren_mont_mul<unsigned int>(x_mont, y, p, p_inv) ==
            test_mul_mod(x, y, p));
    }

    SUBCASE("u32 montgomery x montgomery -> montgomery") {
      const unsigned int p = 2013265921U;
      const unsigned int p_inv = test_mont_p_inv(p);
      const unsigned int x = 1234567U;
      const unsigned int y = 7654321U;
      const unsigned int x_mont = test_to_mont(x, p);
      const unsigned int y_mont = test_to_mont(y, p);
      const unsigned int xy_mont = test_to_mont(test_mul_mod(x, y, p), p);

      CHECK(ren_mont_mul<unsigned int>(x_mont, y_mont, p, p_inv) == xy_mont);
    }

    SUBCASE("u64 montgomery x normal -> normal") {
      const unsigned long long p = 2305843009213693951ULL;
      const unsigned long long p_inv = test_mont_p_inv(p);
      const unsigned long long x = 12345678901234567ULL;
      const unsigned long long y = 98765432109876543ULL;
      const unsigned long long x_mont = test_to_mont(x, p);

      CHECK(ren_mont_mul<unsigned long long>(x_mont, 1ULL, p, p_inv) == x);
      CHECK(ren_mont_mul<unsigned long long>(x_mont, y, p, p_inv) ==
            test_mul_mod(x, y, p));
    }

    SUBCASE("u64 montgomery x montgomery -> montgomery") {
      const unsigned long long p = 2305843009213693951ULL;
      const unsigned long long p_inv = test_mont_p_inv(p);
      const unsigned long long x = 12345678901234567ULL;
      const unsigned long long y = 98765432109876543ULL;
      const unsigned long long x_mont = test_to_mont(x, p);
      const unsigned long long y_mont = test_to_mont(y, p);
      const unsigned long long xy_mont = test_to_mont(test_mul_mod(x, y, p), p);

      CHECK(ren_mont_mul<unsigned long long>(x_mont, y_mont, p, p_inv) ==
            xy_mont);
    }
  }
}
