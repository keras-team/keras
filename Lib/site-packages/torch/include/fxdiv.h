#pragma once
#ifndef FXDIV_H
#define FXDIV_H

#if defined(__cplusplus) && (__cplusplus >= 201103L)
	#include <cstddef>
	#include <cstdint>
	#include <climits>
#elif !defined(__OPENCL_VERSION__)
	#include <stddef.h>
	#include <stdint.h>
	#include <limits.h>
#endif

#if defined(_MSC_VER)
	#include <intrin.h>
	#if defined(_M_IX86) || defined(_M_X64)
		#include <immintrin.h>
	#endif
#endif

#ifndef FXDIV_USE_INLINE_ASSEMBLY
	#define FXDIV_USE_INLINE_ASSEMBLY 0
#endif

static inline uint64_t fxdiv_mulext_uint32_t(uint32_t a, uint32_t b) {
#if defined(_MSC_VER) && defined(_M_IX86)
	return (uint64_t) __emulu((unsigned int) a, (unsigned int) b);
#else
	return (uint64_t) a * (uint64_t) b;
#endif
}

static inline uint32_t fxdiv_mulhi_uint32_t(uint32_t a, uint32_t b) {
#if defined(__OPENCL_VERSION__)
	return mul_hi(a, b);
#elif defined(__CUDA_ARCH__)
	return (uint32_t) __umulhi((unsigned int) a, (unsigned int) b);
#elif defined(_MSC_VER) && defined(_M_IX86)
	return (uint32_t) (__emulu((unsigned int) a, (unsigned int) b) >> 32);
#elif defined(_MSC_VER) && defined(_M_ARM)
	return (uint32_t) _MulUnsignedHigh((unsigned long) a, (unsigned long) b);
#else
	return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32);
#endif
}

static inline uint64_t fxdiv_mulhi_uint64_t(uint64_t a, uint64_t b) {
#if defined(__OPENCL_VERSION__)
	return mul_hi(a, b);
#elif defined(__CUDA_ARCH__)
	return (uint64_t) __umul64hi((unsigned long long) a, (unsigned long long) b);
#elif defined(_MSC_VER) && defined(_M_X64)
	return (uint64_t) __umulh((unsigned __int64) a, (unsigned __int64) b);
#elif defined(__GNUC__) && defined(__SIZEOF_INT128__)
	return (uint64_t) (((((unsigned __int128) a) * ((unsigned __int128) b))) >> 64);
#else
	const uint32_t a_lo = (uint32_t) a;
	const uint32_t a_hi = (uint32_t) (a >> 32);
	const uint32_t b_lo = (uint32_t) b;
	const uint32_t b_hi = (uint32_t) (b >> 32);

	const uint64_t t = fxdiv_mulext_uint32_t(a_hi, b_lo) +
		(uint64_t) fxdiv_mulhi_uint32_t(a_lo, b_lo);
	return fxdiv_mulext_uint32_t(a_hi, b_hi) + (t >> 32) +
		((fxdiv_mulext_uint32_t(a_lo, b_hi) + (uint64_t) (uint32_t) t) >> 32);
#endif
}

static inline size_t fxdiv_mulhi_size_t(size_t a, size_t b) {
#if SIZE_MAX == UINT32_MAX
	return (size_t) fxdiv_mulhi_uint32_t((uint32_t) a, (uint32_t) b);
#elif SIZE_MAX == UINT64_MAX
	return (size_t) fxdiv_mulhi_uint64_t((uint64_t) a, (uint64_t) b);
#else
	#error Unsupported platform
#endif
}

struct fxdiv_divisor_uint32_t {
	uint32_t value;
	uint32_t m;
	uint8_t s1;
	uint8_t s2;
};

struct fxdiv_result_uint32_t {
	uint32_t quotient;
	uint32_t remainder;
};

struct fxdiv_divisor_uint64_t {
	uint64_t value;
	uint64_t m;
	uint8_t s1;
	uint8_t s2;
};

struct fxdiv_result_uint64_t {
	uint64_t quotient;
	uint64_t remainder;
};

struct fxdiv_divisor_size_t {
	size_t value;
	size_t m;
	uint8_t s1;
	uint8_t s2;
};

struct fxdiv_result_size_t {
	size_t quotient;
	size_t remainder;
};

static inline struct fxdiv_divisor_uint32_t fxdiv_init_uint32_t(uint32_t d) {
	struct fxdiv_divisor_uint32_t result = { d };
	if (d == 1) {
		result.m = UINT32_C(1);
		result.s1 = 0;
		result.s2 = 0;
	} else {
		#if defined(__OPENCL_VERSION__)
			const uint32_t l_minus_1 = 31 - clz(d - 1);
		#elif defined(__CUDA_ARCH__)
			const uint32_t l_minus_1 = 31 - __clz((int) (d - 1));
		#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64) || defined(_M_ARM) || defined(_M_ARM64))
			unsigned long l_minus_1;
			_BitScanReverse(&l_minus_1, (unsigned long) (d - 1));
		#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)) && FXDIV_USE_INLINE_ASSEMBLY
			uint32_t l_minus_1;
			__asm__("BSRL %[d_minus_1], %[l_minus_1]"
				: [l_minus_1] "=r" (l_minus_1)
				: [d_minus_1] "r" (d - 1)
				: "cc");
		#elif defined(__GNUC__)
			const uint32_t l_minus_1 = 31 - __builtin_clz(d - 1);
		#else
			/* Based on Algorithm 2 from Hacker's delight */

			uint32_t l_minus_1 = 0;
			uint32_t x = d - 1;
			uint32_t y = x >> 16;
			if (y != 0) {
				l_minus_1 += 16;
				x = y;
			}
			y = x >> 8;
			if (y != 0) {
				l_minus_1 += 8;
				x = y;
			}
			y = x >> 4;
			if (y != 0) {
				l_minus_1 += 4;
				x = y;
			}
			y = x >> 2;
			if (y != 0) {
				l_minus_1 += 2;
				x = y;
			}
			if ((x & 2) != 0) {
				l_minus_1 += 1;
			}
		#endif
		uint32_t u_hi = (UINT32_C(2) << (uint32_t) l_minus_1) - d;

		/* Division of 64-bit number u_hi:UINT32_C(0) by 32-bit number d, 32-bit quotient output q */
		#if defined(__GNUC__) && defined(__i386__) && FXDIV_USE_INLINE_ASSEMBLY
			uint32_t q;
			__asm__("DIVL %[d]"
				: "=a" (q), "+d" (u_hi)
				: [d] "r" (d), "a" (0)
				: "cc");
		#elif (defined(_MSC_VER) && _MSC_VER >= 1920) && !defined(__clang__) && !defined(__INTEL_COMPILER) && (defined(_M_IX86) || defined(_M_X64))
			unsigned int remainder;
			const uint32_t q = (uint32_t) _udiv64((unsigned __int64) ((uint64_t) u_hi << 32), (unsigned int) d, &remainder);
		#else
			const uint32_t q = ((uint64_t) u_hi << 32) / d;
		#endif

		result.m = q + UINT32_C(1);
		result.s1 = 1;
		result.s2 = (uint8_t) l_minus_1;
	}
	return result;
}

static inline struct fxdiv_divisor_uint64_t fxdiv_init_uint64_t(uint64_t d) {
	struct fxdiv_divisor_uint64_t result = { d };
	if (d == 1) {
		result.m = UINT64_C(1);
		result.s1 = 0;
		result.s2 = 0;
	} else {
		#if defined(__OPENCL_VERSION__)
			const uint32_t nlz_d = clz(d);
			const uint32_t l_minus_1 = 63 - clz(d - 1);
		#elif defined(__CUDA_ARCH__)
			const uint32_t nlz_d = __clzll((long long) d);
			const uint32_t l_minus_1 = 63 - __clzll((long long) (d - 1));
		#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
			unsigned long l_minus_1;
			_BitScanReverse64(&l_minus_1, (unsigned __int64) (d - 1));
			unsigned long bsr_d;
			_BitScanReverse64(&bsr_d, (unsigned __int64) d);
			const uint32_t nlz_d = bsr_d ^ 0x3F;
		#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_ARM))
			const uint64_t d_minus_1 = d - 1;
			const uint8_t d_is_power_of_2 = (d & d_minus_1) == 0;
			unsigned long l_minus_1;
			if ((uint32_t) (d_minus_1 >> 32) == 0) {
				_BitScanReverse(&l_minus_1, (unsigned long) d_minus_1);
			} else {
				_BitScanReverse(&l_minus_1, (unsigned long) (uint32_t) (d_minus_1 >> 32));
				l_minus_1 += 32;
			}
			const uint32_t nlz_d = ((uint8_t) l_minus_1 ^ UINT8_C(0x3F)) - d_is_power_of_2;
		#elif defined(__GNUC__) && defined(__x86_64__) && FXDIV_USE_INLINE_ASSEMBLY
			uint64_t l_minus_1;
			__asm__("BSRQ %[d_minus_1], %[l_minus_1]"
				: [l_minus_1] "=r" (l_minus_1)
				: [d_minus_1] "r" (d - 1)
				: "cc");
		#elif defined(__GNUC__)
			const uint32_t l_minus_1 = 63 - __builtin_clzll(d - 1);
			const uint32_t nlz_d = __builtin_clzll(d);
		#else
			/* Based on Algorithm 2 from Hacker's delight */
			const uint64_t d_minus_1 = d - 1;
			const uint32_t d_is_power_of_2 = (d & d_minus_1) == 0;
			uint32_t l_minus_1 = 0;
			uint32_t x = (uint32_t) d_minus_1;
			uint32_t y = d_minus_1 >> 32;
			if (y != 0) {
				l_minus_1 += 32;
				x = y;
			}
			y = x >> 16;
			if (y != 0) {
				l_minus_1 += 16;
				x = y;
			}
			y = x >> 8;
			if (y != 0) {
				l_minus_1 += 8;
				x = y;
			}
			y = x >> 4;
			if (y != 0) {
				l_minus_1 += 4;
				x = y;
			}
			y = x >> 2;
			if (y != 0) {
				l_minus_1 += 2;
				x = y;
			}
			if ((x & 2) != 0) {
				l_minus_1 += 1;
			}
			const uint32_t nlz_d = (l_minus_1 ^ UINT32_C(0x3F)) - d_is_power_of_2;
		#endif
		uint64_t u_hi = (UINT64_C(2) << (uint32_t) l_minus_1) - d;

		/* Division of 128-bit number u_hi:UINT64_C(0) by 64-bit number d, 64-bit quotient output q */
		#if defined(__GNUC__) && defined(__x86_64__) && FXDIV_USE_INLINE_ASSEMBLY
			uint64_t q;
			__asm__("DIVQ %[d]"
				: "=a" (q), "+d" (u_hi)
				: [d] "r" (d), "a" (UINT64_C(0))
				: "cc");
		#elif 0 && defined(__GNUC__) && defined(__SIZEOF_INT128__)
			/* GCC, Clang, and Intel Compiler fail to inline optimized implementation and call into support library for 128-bit division */
			const uint64_t q = (uint64_t) (((unsigned __int128) u_hi << 64) / ((unsigned __int128) d));
		#elif (defined(_MSC_VER) && _MSC_VER >= 1920) && !defined(__clang__) && !defined(__INTEL_COMPILER) && defined(_M_X64)
			unsigned __int64 remainder;
			const uint64_t q = (uint64_t) _udiv128((unsigned __int64) u_hi, 0, (unsigned __int64) d, &remainder);
		#else
			/* Implementation based on code from Hacker's delight */

			/* Normalize divisor and shift divident left */
			d <<= nlz_d;
			u_hi <<= nlz_d;
			/* Break divisor up into two 32-bit digits */
			const uint64_t d_hi = (uint32_t) (d >> 32);
			const uint32_t d_lo = (uint32_t) d;

			/* Compute the first quotient digit, q1 */
			uint64_t q1 = u_hi / d_hi;
			uint64_t r1 = u_hi - q1 * d_hi;

			while ((q1 >> 32) != 0 || fxdiv_mulext_uint32_t((uint32_t) q1, d_lo) > (r1 << 32)) {
				q1 -= 1;
				r1 += d_hi;
				if ((r1 >> 32) != 0) {
					break;
				}
			}

			/* Multiply and subtract. */
			u_hi = (u_hi << 32) - q1 * d;

			/* Compute the second quotient digit, q0 */
			uint64_t q0 = u_hi / d_hi;
			uint64_t r0 = u_hi - q0 * d_hi;

			while ((q0 >> 32) != 0 || fxdiv_mulext_uint32_t((uint32_t) q0, d_lo) > (r0 << 32)) {
				q0 -= 1;
				r0 += d_hi;
				if ((r0 >> 32) != 0) {
					break;
				}
			}
			const uint64_t q = (q1 << 32) | (uint32_t) q0;
		#endif
		result.m = q + UINT64_C(1);
		result.s1 = 1;
		result.s2 = (uint8_t) l_minus_1;
	}
	return result;
}

static inline struct fxdiv_divisor_size_t fxdiv_init_size_t(size_t d) {
#if SIZE_MAX == UINT32_MAX
	const struct fxdiv_divisor_uint32_t uint_result = fxdiv_init_uint32_t((uint32_t) d);
#elif SIZE_MAX == UINT64_MAX
	const struct fxdiv_divisor_uint64_t uint_result = fxdiv_init_uint64_t((uint64_t) d);
#else
	#error Unsupported platform
#endif
	struct fxdiv_divisor_size_t size_result = {
		(size_t) uint_result.value,
		(size_t) uint_result.m,
		uint_result.s1,
		uint_result.s2
	};
	return size_result;
}

static inline uint32_t fxdiv_quotient_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) {
	const uint32_t t = fxdiv_mulhi_uint32_t(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}

static inline uint64_t fxdiv_quotient_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) {
	const uint64_t t = fxdiv_mulhi_uint64_t(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}

static inline size_t fxdiv_quotient_size_t(size_t n, const struct fxdiv_divisor_size_t divisor) {
#if SIZE_MAX == UINT32_MAX
	const struct fxdiv_divisor_uint32_t uint32_divisor = {
		(uint32_t) divisor.value,
		(uint32_t) divisor.m,
		divisor.s1,
		divisor.s2
	};
	return fxdiv_quotient_uint32_t((uint32_t) n, uint32_divisor);
#elif SIZE_MAX == UINT64_MAX
	const struct fxdiv_divisor_uint64_t uint64_divisor = {
		(uint64_t) divisor.value,
		(uint64_t) divisor.m,
		divisor.s1,
		divisor.s2
	};
	return fxdiv_quotient_uint64_t((uint64_t) n, uint64_divisor);
#else
	#error Unsupported platform
#endif
}

static inline uint32_t fxdiv_remainder_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) {
	const uint32_t quotient = fxdiv_quotient_uint32_t(n, divisor);
	return n - quotient * divisor.value;
}

static inline uint64_t fxdiv_remainder_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) {
	const uint64_t quotient = fxdiv_quotient_uint64_t(n, divisor);
	return n - quotient * divisor.value;
}

static inline size_t fxdiv_remainder_size_t(size_t n, const struct fxdiv_divisor_size_t divisor) {
	const size_t quotient = fxdiv_quotient_size_t(n, divisor);
	return n - quotient * divisor.value;
}

static inline uint32_t fxdiv_round_down_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t granularity) {
	const uint32_t quotient = fxdiv_quotient_uint32_t(n, granularity);
	return quotient * granularity.value;
}

static inline uint64_t fxdiv_round_down_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t granularity) {
	const uint64_t quotient = fxdiv_quotient_uint64_t(n, granularity);
	return quotient * granularity.value;
}

static inline size_t fxdiv_round_down_size_t(size_t n, const struct fxdiv_divisor_size_t granularity) {
	const size_t quotient = fxdiv_quotient_size_t(n, granularity);
	return quotient * granularity.value;
}

static inline struct fxdiv_result_uint32_t fxdiv_divide_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) {
	const uint32_t quotient = fxdiv_quotient_uint32_t(n, divisor);
	const uint32_t remainder = n - quotient * divisor.value;
	struct fxdiv_result_uint32_t result = { quotient, remainder };
	return result;
}

static inline struct fxdiv_result_uint64_t fxdiv_divide_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) {
	const uint64_t quotient = fxdiv_quotient_uint64_t(n, divisor);
	const uint64_t remainder = n - quotient * divisor.value;
	struct fxdiv_result_uint64_t result = { quotient, remainder };
	return result;
}

static inline struct fxdiv_result_size_t fxdiv_divide_size_t(size_t n, const struct fxdiv_divisor_size_t divisor) {
	const size_t quotient = fxdiv_quotient_size_t(n, divisor);
	const size_t remainder = n - quotient * divisor.value;
	struct fxdiv_result_size_t result = { quotient, remainder };
	return result;
}

#endif /* FXDIV_H */
