/* Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
 * All rights reserved.
 *
 * This package is an SSL implementation written
 * by Eric Young (eay@cryptsoft.com).
 * The implementation was written so as to conform with Netscapes SSL.
 *
 * This library is free for commercial and non-commercial use as long as
 * the following conditions are aheared to.  The following conditions
 * apply to all code found in this distribution, be it the RC4, RSA,
 * lhash, DES, etc., code; not just the SSL code.  The SSL documentation
 * included with this distribution is covered by the same copyright terms
 * except that the holder is Tim Hudson (tjh@cryptsoft.com).
 *
 * Copyright remains Eric Young's, and as such any Copyright notices in
 * the code are not to be removed.
 * If this package is used in a product, Eric Young should be given attribution
 * as the author of the parts of the library used.
 * This can be in the form of a textual message at program startup or
 * in documentation (online or textual) provided with the package.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    "This product includes cryptographic software written by
 *     Eric Young (eay@cryptsoft.com)"
 *    The word 'cryptographic' can be left out if the rouines from the library
 *    being used are not cryptographic related :-).
 * 4. If you include any Windows specific code (or a derivative thereof) from
 *    the apps directory (application code) you must include an acknowledgement:
 *    "This product includes software written by Tim Hudson (tjh@cryptsoft.com)"
 *
 * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * The licence and distribution terms for any publically available version or
 * derivative of this code cannot be changed.  i.e. this code cannot simply be
 * copied and put under another distribution licence
 * [including the GNU Public Licence.]
 */
/* ====================================================================
 * Copyright (c) 1998-2001 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com). */

#ifndef OPENSSL_HEADER_CRYPTO_INTERNAL_H
#define OPENSSL_HEADER_CRYPTO_INTERNAL_H

#include <openssl/crypto.h>
#include <openssl/ex_data.h>
#include <openssl/stack.h>
#include <openssl/thread.h>

#include <assert.h>
#include <string.h>

#if defined(BORINGSSL_CONSTANT_TIME_VALIDATION)
#include <valgrind/memcheck.h>
#endif

#if defined(BORINGSSL_FIPS_BREAK_TESTS)
#include <stdlib.h>
#endif

#if !defined(__cplusplus)
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdalign.h>
#elif defined(_MSC_VER) && !defined(__clang__)
#define alignas(x) __declspec(align(x))
#define alignof __alignof
#else
// With the exception of MSVC, we require C11 to build the library. C11 is a
// prerequisite for improved refcounting performance. All our supported C
// compilers have long implemented C11 and made it default. The most likely
// cause of pre-C11 modes is stale -std=c99 or -std=gnu99 flags in build
// configuration. Such flags can be removed.
//
// TODO(davidben): In MSVC 2019 16.8 or higher (_MSC_VER >= 1928),
// |__STDC_VERSION__| will be 201112 when passed /std:c11 and unset otherwise.
// C11 alignas and alignof are only implemented in C11 mode. Can we mandate C11
// mode for those versions?
#error "BoringSSL must be built in C11 mode or higher."
#endif
#endif

#if defined(OPENSSL_THREADS) && \
    (!defined(OPENSSL_WINDOWS) || defined(__MINGW32__))
#include <pthread.h>
#define OPENSSL_PTHREADS
#endif

#if defined(OPENSSL_THREADS) && !defined(OPENSSL_PTHREADS) && \
    defined(OPENSSL_WINDOWS)
#define OPENSSL_WINDOWS_THREADS
OPENSSL_MSVC_PRAGMA(warning(push, 3))
#include <windows.h>
OPENSSL_MSVC_PRAGMA(warning(pop))
#endif

#if defined(__cplusplus)
extern "C" {
#endif


#if defined(OPENSSL_X86) || defined(OPENSSL_X86_64) || defined(OPENSSL_ARM) || \
    defined(OPENSSL_AARCH64)
// OPENSSL_cpuid_setup initializes the platform-specific feature cache.
void OPENSSL_cpuid_setup(void);
#endif

#if (defined(OPENSSL_ARM) || defined(OPENSSL_AARCH64)) && \
    !defined(OPENSSL_STATIC_ARMCAP)
// OPENSSL_get_armcap_pointer_for_test returns a pointer to |OPENSSL_armcap_P|
// for unit tests. Any modifications to the value must be made after
// |CRYPTO_library_init| but before any other function call in BoringSSL.
OPENSSL_EXPORT uint32_t *OPENSSL_get_armcap_pointer_for_test(void);
#endif


#if (!defined(_MSC_VER) || defined(__clang__)) && defined(OPENSSL_64_BIT)
#define BORINGSSL_HAS_UINT128
typedef __int128_t int128_t;
typedef __uint128_t uint128_t;

// clang-cl supports __uint128_t but modulus and division don't work.
// https://crbug.com/787617.
#if !defined(_MSC_VER) || !defined(__clang__)
#define BORINGSSL_CAN_DIVIDE_UINT128
#endif
#endif

#define OPENSSL_ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))

// Have a generic fall-through for different versions of C/C++.
#if defined(__cplusplus) && __cplusplus >= 201703L
#define OPENSSL_FALLTHROUGH [[fallthrough]]
#elif defined(__cplusplus) && __cplusplus >= 201103L && defined(__clang__)
#define OPENSSL_FALLTHROUGH [[clang::fallthrough]]
#elif defined(__cplusplus) && __cplusplus >= 201103L && defined(__GNUC__) && \
    __GNUC__ >= 7
#define OPENSSL_FALLTHROUGH [[gnu::fallthrough]]
#elif defined(__GNUC__) && __GNUC__ >= 7 // gcc 7
#define OPENSSL_FALLTHROUGH __attribute__ ((fallthrough))
#elif defined(__clang__)
#if __has_attribute(fallthrough) && __clang_major__ >= 5
// Clang 3.5, at least, complains about "error: declaration does not declare
// anything", possibily because we put a semicolon after this macro in
// practice. Thus limit it to >= Clang 5, which does work.
#define OPENSSL_FALLTHROUGH __attribute__ ((fallthrough))
#else // clang versions that do not support fallthrough.
#define OPENSSL_FALLTHROUGH
#endif
#else // C++11 on gcc 6, and all other cases
#define OPENSSL_FALLTHROUGH
#endif

// For convenience in testing 64-bit generic code, we allow disabling SSE2
// intrinsics via |OPENSSL_NO_SSE2_FOR_TESTING|. x86_64 always has SSE2
// available, so we would otherwise need to test such code on a non-x86_64
// platform.
#if defined(__SSE2__) && !defined(OPENSSL_NO_SSE2_FOR_TESTING)
#define OPENSSL_SSE2
#endif

#if defined(BORINGSSL_MALLOC_FAILURE_TESTING)
// OPENSSL_reset_malloc_counter_for_testing, when malloc testing is enabled,
// resets the internal malloc counter, to simulate further malloc failures. This
// should be called in between independent tests, at a point where failure from
// a previous test will not impact subsequent ones.
OPENSSL_EXPORT void OPENSSL_reset_malloc_counter_for_testing(void);
#else
OPENSSL_INLINE void OPENSSL_reset_malloc_counter_for_testing(void) {}
#endif


// Pointer utility functions.

// buffers_alias returns one if |a| and |b| alias and zero otherwise.
static inline int buffers_alias(const uint8_t *a, size_t a_len,
                                const uint8_t *b, size_t b_len) {
  // Cast |a| and |b| to integers. In C, pointer comparisons between unrelated
  // objects are undefined whereas pointer to integer conversions are merely
  // implementation-defined. We assume the implementation defined it in a sane
  // way.
  uintptr_t a_u = (uintptr_t)a;
  uintptr_t b_u = (uintptr_t)b;
  return a_u + a_len > b_u && b_u + b_len > a_u;
}

// align_pointer returns |ptr|, advanced to |alignment|. |alignment| must be a
// power of two, and |ptr| must have at least |alignment - 1| bytes of scratch
// space.
static inline void *align_pointer(void *ptr, size_t alignment) {
  // |alignment| must be a power of two.
  assert(alignment != 0 && (alignment & (alignment - 1)) == 0);
  // Instead of aligning |ptr| as a |uintptr_t| and casting back, compute the
  // offset and advance in pointer space. C guarantees that casting from pointer
  // to |uintptr_t| and back gives the same pointer, but general
  // integer-to-pointer conversions are implementation-defined. GCC does define
  // it in the useful way, but this makes fewer assumptions.
  uintptr_t offset = (0u - (uintptr_t)ptr) & (alignment - 1);
  ptr = (char *)ptr + offset;
  assert(((uintptr_t)ptr & (alignment - 1)) == 0);
  return ptr;
}


// Constant-time utility functions.
//
// The following methods return a bitmask of all ones (0xff...f) for true and 0
// for false. This is useful for choosing a value based on the result of a
// conditional in constant time. For example,
//
// if (a < b) {
//   c = a;
// } else {
//   c = b;
// }
//
// can be written as
//
// crypto_word_t lt = constant_time_lt_w(a, b);
// c = constant_time_select_w(lt, a, b);

// crypto_word_t is the type that most constant-time functions use. Ideally we
// would like it to be |size_t|, but NaCl builds in 64-bit mode with 32-bit
// pointers, which means that |size_t| can be 32 bits when |BN_ULONG| is 64
// bits. Since we want to be able to do constant-time operations on a
// |BN_ULONG|, |crypto_word_t| is defined as an unsigned value with the native
// word length.
#if defined(OPENSSL_64_BIT)
typedef uint64_t crypto_word_t;
#elif defined(OPENSSL_32_BIT)
typedef uint32_t crypto_word_t;
#else
#error "Must define either OPENSSL_32_BIT or OPENSSL_64_BIT"
#endif

#define CONSTTIME_TRUE_W ~((crypto_word_t)0)
#define CONSTTIME_FALSE_W ((crypto_word_t)0)
#define CONSTTIME_TRUE_8 ((uint8_t)0xff)
#define CONSTTIME_FALSE_8 ((uint8_t)0)

// value_barrier_w returns |a|, but prevents GCC and Clang from reasoning about
// the returned value. This is used to mitigate compilers undoing constant-time
// code, until we can express our requirements directly in the language.
//
// Note the compiler is aware that |value_barrier_w| has no side effects and
// always has the same output for a given input. This allows it to eliminate
// dead code, move computations across loops, and vectorize.
static inline crypto_word_t value_barrier_w(crypto_word_t a) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__("" : "+r"(a) : /* no inputs */);
#endif
  return a;
}

// value_barrier_u32 behaves like |value_barrier_w| but takes a |uint32_t|.
static inline uint32_t value_barrier_u32(uint32_t a) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__("" : "+r"(a) : /* no inputs */);
#endif
  return a;
}

// value_barrier_u64 behaves like |value_barrier_w| but takes a |uint64_t|.
static inline uint64_t value_barrier_u64(uint64_t a) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__("" : "+r"(a) : /* no inputs */);
#endif
  return a;
}

// constant_time_msb_w returns the given value with the MSB copied to all the
// other bits.
static inline crypto_word_t constant_time_msb_w(crypto_word_t a) {
  return 0u - (a >> (sizeof(a) * 8 - 1));
}

// constant_time_lt_w returns 0xff..f if a < b and 0 otherwise.
static inline crypto_word_t constant_time_lt_w(crypto_word_t a,
                                               crypto_word_t b) {
  // Consider the two cases of the problem:
  //   msb(a) == msb(b): a < b iff the MSB of a - b is set.
  //   msb(a) != msb(b): a < b iff the MSB of b is set.
  //
  // If msb(a) == msb(b) then the following evaluates as:
  //   msb(a^((a^b)|((a-b)^a))) ==
  //   msb(a^((a-b) ^ a))       ==   (because msb(a^b) == 0)
  //   msb(a^a^(a-b))           ==   (rearranging)
  //   msb(a-b)                      (because ∀x. x^x == 0)
  //
  // Else, if msb(a) != msb(b) then the following evaluates as:
  //   msb(a^((a^b)|((a-b)^a))) ==
  //   msb(a^(𝟙 | ((a-b)^a)))   ==   (because msb(a^b) == 1 and 𝟙
  //                                  represents a value s.t. msb(𝟙) = 1)
  //   msb(a^𝟙)                 ==   (because ORing with 1 results in 1)
  //   msb(b)
  //
  //
  // Here is an SMT-LIB verification of this formula:
  //
  // (define-fun lt ((a (_ BitVec 32)) (b (_ BitVec 32))) (_ BitVec 32)
  //   (bvxor a (bvor (bvxor a b) (bvxor (bvsub a b) a)))
  // )
  //
  // (declare-fun a () (_ BitVec 32))
  // (declare-fun b () (_ BitVec 32))
  //
  // (assert (not (= (= #x00000001 (bvlshr (lt a b) #x0000001f)) (bvult a b))))
  // (check-sat)
  // (get-model)
  return constant_time_msb_w(a^((a^b)|((a-b)^a)));
}

// constant_time_lt_8 acts like |constant_time_lt_w| but returns an 8-bit
// mask.
static inline uint8_t constant_time_lt_8(crypto_word_t a, crypto_word_t b) {
  return (uint8_t)(constant_time_lt_w(a, b));
}

// constant_time_ge_w returns 0xff..f if a >= b and 0 otherwise.
static inline crypto_word_t constant_time_ge_w(crypto_word_t a,
                                               crypto_word_t b) {
  return ~constant_time_lt_w(a, b);
}

// constant_time_ge_8 acts like |constant_time_ge_w| but returns an 8-bit
// mask.
static inline uint8_t constant_time_ge_8(crypto_word_t a, crypto_word_t b) {
  return (uint8_t)(constant_time_ge_w(a, b));
}

// constant_time_is_zero returns 0xff..f if a == 0 and 0 otherwise.
static inline crypto_word_t constant_time_is_zero_w(crypto_word_t a) {
  // Here is an SMT-LIB verification of this formula:
  //
  // (define-fun is_zero ((a (_ BitVec 32))) (_ BitVec 32)
  //   (bvand (bvnot a) (bvsub a #x00000001))
  // )
  //
  // (declare-fun a () (_ BitVec 32))
  //
  // (assert (not (= (= #x00000001 (bvlshr (is_zero a) #x0000001f)) (= a #x00000000))))
  // (check-sat)
  // (get-model)
  return constant_time_msb_w(~a & (a - 1));
}

// constant_time_is_zero_8 acts like |constant_time_is_zero_w| but returns an
// 8-bit mask.
static inline uint8_t constant_time_is_zero_8(crypto_word_t a) {
  return (uint8_t)(constant_time_is_zero_w(a));
}

// constant_time_eq_w returns 0xff..f if a == b and 0 otherwise.
static inline crypto_word_t constant_time_eq_w(crypto_word_t a,
                                               crypto_word_t b) {
  return constant_time_is_zero_w(a ^ b);
}

// constant_time_eq_8 acts like |constant_time_eq_w| but returns an 8-bit
// mask.
static inline uint8_t constant_time_eq_8(crypto_word_t a, crypto_word_t b) {
  return (uint8_t)(constant_time_eq_w(a, b));
}

// constant_time_eq_int acts like |constant_time_eq_w| but works on int
// values.
static inline crypto_word_t constant_time_eq_int(int a, int b) {
  return constant_time_eq_w((crypto_word_t)(a), (crypto_word_t)(b));
}

// constant_time_eq_int_8 acts like |constant_time_eq_int| but returns an 8-bit
// mask.
static inline uint8_t constant_time_eq_int_8(int a, int b) {
  return constant_time_eq_8((crypto_word_t)(a), (crypto_word_t)(b));
}

// constant_time_select_w returns (mask & a) | (~mask & b). When |mask| is all
// 1s or all 0s (as returned by the methods above), the select methods return
// either |a| (if |mask| is nonzero) or |b| (if |mask| is zero).
static inline crypto_word_t constant_time_select_w(crypto_word_t mask,
                                                   crypto_word_t a,
                                                   crypto_word_t b) {
  // Clang recognizes this pattern as a select. While it usually transforms it
  // to a cmov, it sometimes further transforms it into a branch, which we do
  // not want.
  //
  // Adding barriers to both |mask| and |~mask| breaks the relationship between
  // the two, which makes the compiler stick with bitmasks.
  return (value_barrier_w(mask) & a) | (value_barrier_w(~mask) & b);
}

// constant_time_select_8 acts like |constant_time_select| but operates on
// 8-bit values.
static inline uint8_t constant_time_select_8(uint8_t mask, uint8_t a,
                                             uint8_t b) {
  return (uint8_t)(constant_time_select_w(mask, a, b));
}

// constant_time_select_int acts like |constant_time_select| but operates on
// ints.
static inline int constant_time_select_int(crypto_word_t mask, int a, int b) {
  return (int)(constant_time_select_w(mask, (crypto_word_t)(a),
                                      (crypto_word_t)(b)));
}

#if defined(BORINGSSL_CONSTANT_TIME_VALIDATION)

// CONSTTIME_SECRET takes a pointer and a number of bytes and marks that region
// of memory as secret. Secret data is tracked as it flows to registers and
// other parts of a memory. If secret data is used as a condition for a branch,
// or as a memory index, it will trigger warnings in valgrind.
#define CONSTTIME_SECRET(ptr, len) VALGRIND_MAKE_MEM_UNDEFINED(ptr, len)

// CONSTTIME_DECLASSIFY takes a pointer and a number of bytes and marks that
// region of memory as public. Public data is not subject to constant-time
// rules.
#define CONSTTIME_DECLASSIFY(ptr, len) VALGRIND_MAKE_MEM_DEFINED(ptr, len)

#else

#define CONSTTIME_SECRET(ptr, len)
#define CONSTTIME_DECLASSIFY(ptr, len)

#endif  // BORINGSSL_CONSTANT_TIME_VALIDATION

static inline crypto_word_t constant_time_declassify_w(crypto_word_t v) {
  // Return |v| through a value barrier to be safe. Valgrind-based constant-time
  // validation is partly to check the compiler has not undone any constant-time
  // work. Any place |BORINGSSL_CONSTANT_TIME_VALIDATION| influences
  // optimizations, this validation is inaccurate.
  //
  // However, by sending pointers through valgrind, we likely inhibit escape
  // analysis. On local variables, particularly booleans, we likely
  // significantly impact optimizations.
  //
  // Thus, to be safe, stick a value barrier, in hopes of comparably inhibiting
  // compiler analysis.
  CONSTTIME_DECLASSIFY(&v, sizeof(v));
  return value_barrier_w(v);
}

static inline int constant_time_declassify_int(int v) {
  static_assert(sizeof(uint32_t) == sizeof(int),
                "int is not the same size as uint32_t");
  // See comment above.
  CONSTTIME_DECLASSIFY(&v, sizeof(v));
  return value_barrier_u32(v);
}


// Thread-safe initialisation.

#if !defined(OPENSSL_THREADS)
typedef uint32_t CRYPTO_once_t;
#define CRYPTO_ONCE_INIT 0
#elif defined(OPENSSL_WINDOWS_THREADS)
typedef INIT_ONCE CRYPTO_once_t;
#define CRYPTO_ONCE_INIT INIT_ONCE_STATIC_INIT
#elif defined(OPENSSL_PTHREADS)
typedef pthread_once_t CRYPTO_once_t;
#define CRYPTO_ONCE_INIT PTHREAD_ONCE_INIT
#else
#error "Unknown threading library"
#endif

// CRYPTO_once calls |init| exactly once per process. This is thread-safe: if
// concurrent threads call |CRYPTO_once| with the same |CRYPTO_once_t| argument
// then they will block until |init| completes, but |init| will have only been
// called once.
//
// The |once| argument must be a |CRYPTO_once_t| that has been initialised with
// the value |CRYPTO_ONCE_INIT|.
OPENSSL_EXPORT void CRYPTO_once(CRYPTO_once_t *once, void (*init)(void));


// Reference counting.

// Automatically enable C11 atomics if implemented.
#if !defined(OPENSSL_C11_ATOMIC) && defined(OPENSSL_THREADS) &&   \
    !defined(__STDC_NO_ATOMICS__) && defined(__STDC_VERSION__) && \
    __STDC_VERSION__ >= 201112L
#define OPENSSL_C11_ATOMIC
#endif

// CRYPTO_REFCOUNT_MAX is the value at which the reference count saturates.
#define CRYPTO_REFCOUNT_MAX 0xffffffff

// CRYPTO_refcount_inc atomically increments the value at |*count| unless the
// value would overflow. It's safe for multiple threads to concurrently call
// this or |CRYPTO_refcount_dec_and_test_zero| on the same
// |CRYPTO_refcount_t|.
OPENSSL_EXPORT void CRYPTO_refcount_inc(CRYPTO_refcount_t *count);

// CRYPTO_refcount_dec_and_test_zero tests the value at |*count|:
//   if it's zero, it crashes the address space.
//   if it's the maximum value, it returns zero.
//   otherwise, it atomically decrements it and returns one iff the resulting
//       value is zero.
//
// It's safe for multiple threads to concurrently call this or
// |CRYPTO_refcount_inc| on the same |CRYPTO_refcount_t|.
OPENSSL_EXPORT int CRYPTO_refcount_dec_and_test_zero(CRYPTO_refcount_t *count);


// Locks.
//
// Two types of locks are defined: |CRYPTO_MUTEX|, which can be used in
// structures as normal, and |struct CRYPTO_STATIC_MUTEX|, which can be used as
// a global lock. A global lock must be initialised to the value
// |CRYPTO_STATIC_MUTEX_INIT|.
//
// |CRYPTO_MUTEX| can appear in public structures and so is defined in
// thread.h as a structure large enough to fit the real type. The global lock is
// a different type so it may be initialized with platform initializer macros.

#if !defined(OPENSSL_THREADS)
struct CRYPTO_STATIC_MUTEX {
  char padding;  // Empty structs have different sizes in C and C++.
};
#define CRYPTO_STATIC_MUTEX_INIT { 0 }
#elif defined(OPENSSL_WINDOWS_THREADS)
struct CRYPTO_STATIC_MUTEX {
  SRWLOCK lock;
};
#define CRYPTO_STATIC_MUTEX_INIT { SRWLOCK_INIT }
#elif defined(OPENSSL_PTHREADS)
struct CRYPTO_STATIC_MUTEX {
  pthread_rwlock_t lock;
};
#define CRYPTO_STATIC_MUTEX_INIT { PTHREAD_RWLOCK_INITIALIZER }
#else
#error "Unknown threading library"
#endif

// CRYPTO_MUTEX_init initialises |lock|. If |lock| is a static variable, use a
// |CRYPTO_STATIC_MUTEX|.
OPENSSL_EXPORT void CRYPTO_MUTEX_init(CRYPTO_MUTEX *lock);

// CRYPTO_MUTEX_lock_read locks |lock| such that other threads may also have a
// read lock, but none may have a write lock.
OPENSSL_EXPORT void CRYPTO_MUTEX_lock_read(CRYPTO_MUTEX *lock);

// CRYPTO_MUTEX_lock_write locks |lock| such that no other thread has any type
// of lock on it.
OPENSSL_EXPORT void CRYPTO_MUTEX_lock_write(CRYPTO_MUTEX *lock);

// CRYPTO_MUTEX_unlock_read unlocks |lock| for reading.
OPENSSL_EXPORT void CRYPTO_MUTEX_unlock_read(CRYPTO_MUTEX *lock);

// CRYPTO_MUTEX_unlock_write unlocks |lock| for writing.
OPENSSL_EXPORT void CRYPTO_MUTEX_unlock_write(CRYPTO_MUTEX *lock);

// CRYPTO_MUTEX_cleanup releases all resources held by |lock|.
OPENSSL_EXPORT void CRYPTO_MUTEX_cleanup(CRYPTO_MUTEX *lock);

// CRYPTO_STATIC_MUTEX_lock_read locks |lock| such that other threads may also
// have a read lock, but none may have a write lock. The |lock| variable does
// not need to be initialised by any function, but must have been statically
// initialised with |CRYPTO_STATIC_MUTEX_INIT|.
OPENSSL_EXPORT void CRYPTO_STATIC_MUTEX_lock_read(
    struct CRYPTO_STATIC_MUTEX *lock);

// CRYPTO_STATIC_MUTEX_lock_write locks |lock| such that no other thread has
// any type of lock on it.  The |lock| variable does not need to be initialised
// by any function, but must have been statically initialised with
// |CRYPTO_STATIC_MUTEX_INIT|.
OPENSSL_EXPORT void CRYPTO_STATIC_MUTEX_lock_write(
    struct CRYPTO_STATIC_MUTEX *lock);

// CRYPTO_STATIC_MUTEX_unlock_read unlocks |lock| for reading.
OPENSSL_EXPORT void CRYPTO_STATIC_MUTEX_unlock_read(
    struct CRYPTO_STATIC_MUTEX *lock);

// CRYPTO_STATIC_MUTEX_unlock_write unlocks |lock| for writing.
OPENSSL_EXPORT void CRYPTO_STATIC_MUTEX_unlock_write(
    struct CRYPTO_STATIC_MUTEX *lock);

#if defined(__cplusplus)
extern "C++" {

BSSL_NAMESPACE_BEGIN

namespace internal {

// MutexLockBase is a RAII helper for CRYPTO_MUTEX locking.
template <void (*LockFunc)(CRYPTO_MUTEX *), void (*ReleaseFunc)(CRYPTO_MUTEX *)>
class MutexLockBase {
 public:
  explicit MutexLockBase(CRYPTO_MUTEX *mu) : mu_(mu) {
    assert(mu_ != nullptr);
    LockFunc(mu_);
  }
  ~MutexLockBase() { ReleaseFunc(mu_); }
  MutexLockBase(const MutexLockBase<LockFunc, ReleaseFunc> &) = delete;
  MutexLockBase &operator=(const MutexLockBase<LockFunc, ReleaseFunc> &) =
      delete;

 private:
  CRYPTO_MUTEX *const mu_;
};

}  // namespace internal

using MutexWriteLock =
    internal::MutexLockBase<CRYPTO_MUTEX_lock_write, CRYPTO_MUTEX_unlock_write>;
using MutexReadLock =
    internal::MutexLockBase<CRYPTO_MUTEX_lock_read, CRYPTO_MUTEX_unlock_read>;

BSSL_NAMESPACE_END

}  // extern "C++"
#endif  // defined(__cplusplus)


// Thread local storage.

// thread_local_data_t enumerates the types of thread-local data that can be
// stored.
typedef enum {
  OPENSSL_THREAD_LOCAL_ERR = 0,
  OPENSSL_THREAD_LOCAL_RAND,
  OPENSSL_THREAD_LOCAL_FIPS_COUNTERS,
  OPENSSL_THREAD_LOCAL_FIPS_SERVICE_INDICATOR_STATE,
  OPENSSL_THREAD_LOCAL_TEST,
  NUM_OPENSSL_THREAD_LOCALS,
} thread_local_data_t;

// thread_local_destructor_t is the type of a destructor function that will be
// called when a thread exits and its thread-local storage needs to be freed.
typedef void (*thread_local_destructor_t)(void *);

// CRYPTO_get_thread_local gets the pointer value that is stored for the
// current thread for the given index, or NULL if none has been set.
OPENSSL_EXPORT void *CRYPTO_get_thread_local(thread_local_data_t value);

// CRYPTO_set_thread_local sets a pointer value for the current thread at the
// given index. This function should only be called once per thread for a given
// |index|: rather than update the pointer value itself, update the data that
// is pointed to.
//
// The destructor function will be called when a thread exits to free this
// thread-local data. All calls to |CRYPTO_set_thread_local| with the same
// |index| should have the same |destructor| argument. The destructor may be
// called with a NULL argument if a thread that never set a thread-local
// pointer for |index|, exits. The destructor may be called concurrently with
// different arguments.
//
// This function returns one on success or zero on error. If it returns zero
// then |destructor| has been called with |value| already.
OPENSSL_EXPORT int CRYPTO_set_thread_local(
    thread_local_data_t index, void *value,
    thread_local_destructor_t destructor);


// ex_data

typedef struct crypto_ex_data_func_st CRYPTO_EX_DATA_FUNCS;

DECLARE_STACK_OF(CRYPTO_EX_DATA_FUNCS)

// CRYPTO_EX_DATA_CLASS tracks the ex_indices registered for a type which
// supports ex_data. It should defined as a static global within the module
// which defines that type.
typedef struct {
  struct CRYPTO_STATIC_MUTEX lock;
  STACK_OF(CRYPTO_EX_DATA_FUNCS) *meth;
  // num_reserved is one if the ex_data index zero is reserved for legacy
  // |TYPE_get_app_data| functions.
  uint8_t num_reserved;
} CRYPTO_EX_DATA_CLASS;

#define CRYPTO_EX_DATA_CLASS_INIT {CRYPTO_STATIC_MUTEX_INIT, NULL, 0}
#define CRYPTO_EX_DATA_CLASS_INIT_WITH_APP_DATA \
    {CRYPTO_STATIC_MUTEX_INIT, NULL, 1}

// CRYPTO_get_ex_new_index allocates a new index for |ex_data_class| and writes
// it to |*out_index|. Each class of object should provide a wrapper function
// that uses the correct |CRYPTO_EX_DATA_CLASS|. It returns one on success and
// zero otherwise.
OPENSSL_EXPORT int CRYPTO_get_ex_new_index(CRYPTO_EX_DATA_CLASS *ex_data_class,
                                           int *out_index, long argl,
                                           void *argp,
                                           CRYPTO_EX_free *free_func);

// CRYPTO_set_ex_data sets an extra data pointer on a given object. Each class
// of object should provide a wrapper function.
OPENSSL_EXPORT int CRYPTO_set_ex_data(CRYPTO_EX_DATA *ad, int index, void *val);

// CRYPTO_get_ex_data returns an extra data pointer for a given object, or NULL
// if no such index exists. Each class of object should provide a wrapper
// function.
OPENSSL_EXPORT void *CRYPTO_get_ex_data(const CRYPTO_EX_DATA *ad, int index);

// CRYPTO_new_ex_data initialises a newly allocated |CRYPTO_EX_DATA|.
OPENSSL_EXPORT void CRYPTO_new_ex_data(CRYPTO_EX_DATA *ad);

// CRYPTO_free_ex_data frees |ad|, which is embedded inside |obj|, which is an
// object of the given class.
OPENSSL_EXPORT void CRYPTO_free_ex_data(CRYPTO_EX_DATA_CLASS *ex_data_class,
                                        void *obj, CRYPTO_EX_DATA *ad);


// Endianness conversions.

#if defined(__GNUC__) && __GNUC__ >= 2
static inline uint16_t CRYPTO_bswap2(uint16_t x) {
  return __builtin_bswap16(x);
}

static inline uint32_t CRYPTO_bswap4(uint32_t x) {
  return __builtin_bswap32(x);
}

static inline uint64_t CRYPTO_bswap8(uint64_t x) {
  return __builtin_bswap64(x);
}
#elif defined(_MSC_VER)
OPENSSL_MSVC_PRAGMA(warning(push, 3))
#include <stdlib.h>
OPENSSL_MSVC_PRAGMA(warning(pop))
#pragma intrinsic(_byteswap_uint64, _byteswap_ulong, _byteswap_ushort)
static inline uint16_t CRYPTO_bswap2(uint16_t x) {
  return _byteswap_ushort(x);
}

static inline uint32_t CRYPTO_bswap4(uint32_t x) {
  return _byteswap_ulong(x);
}

static inline uint64_t CRYPTO_bswap8(uint64_t x) {
  return _byteswap_uint64(x);
}
#else
static inline uint16_t CRYPTO_bswap2(uint16_t x) {
  return (x >> 8) | (x << 8);
}

static inline uint32_t CRYPTO_bswap4(uint32_t x) {
  x = (x >> 16) | (x << 16);
  x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
  return x;
}

static inline uint64_t CRYPTO_bswap8(uint64_t x) {
  return CRYPTO_bswap4(x >> 32) | (((uint64_t)CRYPTO_bswap4(x)) << 32);
}
#endif


// Language bug workarounds.
//
// Most C standard library functions are undefined if passed NULL, even when the
// corresponding length is zero. This gives them (and, in turn, all functions
// which call them) surprising behavior on empty arrays. Some compilers will
// miscompile code due to this rule. See also
// https://www.imperialviolet.org/2016/06/26/nonnull.html
//
// These wrapper functions behave the same as the corresponding C standard
// functions, but behave as expected when passed NULL if the length is zero.
//
// Note |OPENSSL_memcmp| is a different function from |CRYPTO_memcmp|.

// C++ defines |memchr| as a const-correct overload.
#if defined(__cplusplus)
extern "C++" {

static inline const void *OPENSSL_memchr(const void *s, int c, size_t n) {
  if (n == 0) {
    return NULL;
  }

  return memchr(s, c, n);
}

static inline void *OPENSSL_memchr(void *s, int c, size_t n) {
  if (n == 0) {
    return NULL;
  }

  return memchr(s, c, n);
}

}  // extern "C++"
#else  // __cplusplus

static inline void *OPENSSL_memchr(const void *s, int c, size_t n) {
  if (n == 0) {
    return NULL;
  }

  return memchr(s, c, n);
}

#endif  // __cplusplus

static inline int OPENSSL_memcmp(const void *s1, const void *s2, size_t n) {
  if (n == 0) {
    return 0;
  }

  return memcmp(s1, s2, n);
}

static inline void *OPENSSL_memcpy(void *dst, const void *src, size_t n) {
  if (n == 0) {
    return dst;
  }

  return memcpy(dst, src, n);
}

static inline void *OPENSSL_memmove(void *dst, const void *src, size_t n) {
  if (n == 0) {
    return dst;
  }

  return memmove(dst, src, n);
}

static inline void *OPENSSL_memset(void *dst, int c, size_t n) {
  if (n == 0) {
    return dst;
  }

  return memset(dst, c, n);
}


// Loads and stores.
//
// The following functions load and store sized integers with the specified
// endianness. They use |memcpy|, and so avoid alignment or strict aliasing
// requirements on the input and output pointers.

static inline uint32_t CRYPTO_load_u32_le(const void *in) {
  uint32_t v;
  OPENSSL_memcpy(&v, in, sizeof(v));
  return v;
}

static inline void CRYPTO_store_u32_le(void *out, uint32_t v) {
  OPENSSL_memcpy(out, &v, sizeof(v));
}

static inline uint32_t CRYPTO_load_u32_be(const void *in) {
  uint32_t v;
  OPENSSL_memcpy(&v, in, sizeof(v));
  return CRYPTO_bswap4(v);
}

static inline void CRYPTO_store_u32_be(void *out, uint32_t v) {
  v = CRYPTO_bswap4(v);
  OPENSSL_memcpy(out, &v, sizeof(v));
}

static inline uint64_t CRYPTO_load_u64_le(const void *in) {
  uint64_t v;
  OPENSSL_memcpy(&v, in, sizeof(v));
  return v;
}

static inline void CRYPTO_store_u64_le(void *out, uint64_t v) {
  OPENSSL_memcpy(out, &v, sizeof(v));
}

static inline uint64_t CRYPTO_load_u64_be(const void *ptr) {
  uint64_t ret;
  OPENSSL_memcpy(&ret, ptr, sizeof(ret));
  return CRYPTO_bswap8(ret);
}

static inline void CRYPTO_store_u64_be(void *out, uint64_t v) {
  v = CRYPTO_bswap8(v);
  OPENSSL_memcpy(out, &v, sizeof(v));
}

static inline crypto_word_t CRYPTO_load_word_le(const void *in) {
  crypto_word_t v;
  OPENSSL_memcpy(&v, in, sizeof(v));
  return v;
}

static inline void CRYPTO_store_word_le(void *out, crypto_word_t v) {
  OPENSSL_memcpy(out, &v, sizeof(v));
}

static inline crypto_word_t CRYPTO_load_word_be(const void *in) {
  crypto_word_t v;
  OPENSSL_memcpy(&v, in, sizeof(v));
#if defined(OPENSSL_64_BIT)
  static_assert(sizeof(v) == 8, "crypto_word_t has unexpected size");
  return CRYPTO_bswap8(v);
#else
  static_assert(sizeof(v) == 4, "crypto_word_t has unexpected size");
  return CRYPTO_bswap4(v);
#endif
}


// Bit rotation functions.
//
// Note these functions use |(-shift) & 31|, etc., because shifting by the bit
// width is undefined. Both Clang and GCC recognize this pattern as a rotation,
// but MSVC does not. Instead, we call MSVC's built-in functions.

static inline uint32_t CRYPTO_rotl_u32(uint32_t value, int shift) {
#if defined(_MSC_VER)
  return _rotl(value, shift);
#else
  return (value << shift) | (value >> ((-shift) & 31));
#endif
}

static inline uint32_t CRYPTO_rotr_u32(uint32_t value, int shift) {
#if defined(_MSC_VER)
  return _rotr(value, shift);
#else
  return (value >> shift) | (value << ((-shift) & 31));
#endif
}

static inline uint64_t CRYPTO_rotl_u64(uint64_t value, int shift) {
#if defined(_MSC_VER)
  return _rotl64(value, shift);
#else
  return (value << shift) | (value >> ((-shift) & 63));
#endif
}

static inline uint64_t CRYPTO_rotr_u64(uint64_t value, int shift) {
#if defined(_MSC_VER)
  return _rotr64(value, shift);
#else
  return (value >> shift) | (value << ((-shift) & 63));
#endif
}


// FIPS functions.

#if defined(BORINGSSL_FIPS)

// BORINGSSL_FIPS_abort is called when a FIPS power-on or continuous test
// fails. It prevents any further cryptographic operations by the current
// process.
void BORINGSSL_FIPS_abort(void) __attribute__((noreturn));

// boringssl_self_test_startup runs all startup self tests and returns one on
// success or zero on error. Startup self tests do not include lazy tests.
// Call |BORINGSSL_self_test| to run every self test.
int boringssl_self_test_startup(void);

// boringssl_ensure_rsa_self_test checks whether the RSA self-test has been run
// in this address space. If not, it runs it and crashes the address space if
// unsuccessful.
void boringssl_ensure_rsa_self_test(void);

// boringssl_ensure_ecc_self_test checks whether the ECDSA and ECDH self-test
// has been run in this address space. If not, it runs it and crashes the
// address space if unsuccessful.
void boringssl_ensure_ecc_self_test(void);

// boringssl_ensure_ffdh_self_test checks whether the FFDH self-test has been
// run in this address space. If not, it runs it and crashes the address space
// if unsuccessful.
void boringssl_ensure_ffdh_self_test(void);

#else

// Outside of FIPS mode, the lazy tests are no-ops.

OPENSSL_INLINE void boringssl_ensure_rsa_self_test(void) {}
OPENSSL_INLINE void boringssl_ensure_ecc_self_test(void) {}
OPENSSL_INLINE void boringssl_ensure_ffdh_self_test(void) {}

#endif  // FIPS

// boringssl_self_test_sha256 performs a SHA-256 KAT.
int boringssl_self_test_sha256(void);

// boringssl_self_test_sha512 performs a SHA-512 KAT.
int boringssl_self_test_sha512(void);

// boringssl_self_test_hmac_sha256 performs an HMAC-SHA-256 KAT.
int boringssl_self_test_hmac_sha256(void);

#if defined(BORINGSSL_FIPS_COUNTERS)
void boringssl_fips_inc_counter(enum fips_counter_t counter);
#else
OPENSSL_INLINE void boringssl_fips_inc_counter(enum fips_counter_t counter) {}
#endif

#if defined(BORINGSSL_FIPS_BREAK_TESTS)
OPENSSL_INLINE int boringssl_fips_break_test(const char *test) {
  const char *const value = getenv("BORINGSSL_FIPS_BREAK_TEST");
  return value != NULL && strcmp(value, test) == 0;
}
#else
OPENSSL_INLINE int boringssl_fips_break_test(const char *test) {
  return 0;
}
#endif  // BORINGSSL_FIPS_BREAK_TESTS


// Runtime CPU feature support

#if defined(OPENSSL_X86) || defined(OPENSSL_X86_64)
// OPENSSL_ia32cap_P contains the Intel CPUID bits when running on an x86 or
// x86-64 system.
//
//   Index 0:
//     EDX for CPUID where EAX = 1
//     Bit 20 is always zero
//     Bit 28 is adjusted to reflect whether the data cache is shared between
//       multiple logical cores
//     Bit 30 is used to indicate an Intel CPU
//   Index 1:
//     ECX for CPUID where EAX = 1
//     Bit 11 is used to indicate AMD XOP support, not SDBG
//   Index 2:
//     EBX for CPUID where EAX = 7
//   Index 3:
//     ECX for CPUID where EAX = 7
//
// Note: the CPUID bits are pre-adjusted for the OSXSAVE bit and the YMM and XMM
// bits in XCR0, so it is not necessary to check those.
extern uint32_t OPENSSL_ia32cap_P[4];

#if defined(BORINGSSL_FIPS) && !defined(BORINGSSL_SHARED_LIBRARY)
// The FIPS module, as a static library, requires an out-of-line version of
// |OPENSSL_ia32cap_get| so accesses can be rewritten by delocate. Mark the
// function const so multiple accesses can be optimized together.
const uint32_t *OPENSSL_ia32cap_get(void) __attribute__((const));
#else
OPENSSL_INLINE const uint32_t *OPENSSL_ia32cap_get(void) {
  return OPENSSL_ia32cap_P;
}
#endif

// See Intel manual, volume 2A, table 3-11.

OPENSSL_INLINE int CRYPTO_is_FXSR_capable(void) {
#if defined(__FXSR__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[0] & (1 << 24)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_intel_cpu(void) {
  // The reserved bit 30 is used to indicate an Intel CPU.
  return (OPENSSL_ia32cap_get()[0] & (1 << 30)) != 0;
}

// See Intel manual, volume 2A, table 3-10.

OPENSSL_INLINE int CRYPTO_is_PCLMUL_capable(void) {
#if defined(__PCLMUL__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[1] & (1 << 1)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_SSSE3_capable(void) {
#if defined(__SSSE3__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[1] & (1 << 9)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_SSE4_1_capable(void) {
#if defined(__SSE4_1__)
  return 1;
#else
  return (OPENSSL_ia32cap_P[1] & (1 << 19)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_MOVBE_capable(void) {
#if defined(__MOVBE__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[1] & (1 << 22)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_AESNI_capable(void) {
#if defined(__AES__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[1] & (1 << 25)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_AVX_capable(void) {
#if defined(__AVX__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[1] & (1 << 28)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_RDRAND_capable(void) {
  // The GCC/Clang feature name and preprocessor symbol for RDRAND are "rdrnd"
  // and |__RDRND__|, respectively.
#if defined(__RDRND__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[1] & (1u << 30)) != 0;
#endif
}

// See Intel manual, volume 2A, table 3-8.

OPENSSL_INLINE int CRYPTO_is_BMI1_capable(void) {
#if defined(__BMI1__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[2] & (1 << 3)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_AVX2_capable(void) {
#if defined(__AVX2__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[2] & (1 << 5)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_BMI2_capable(void) {
#if defined(__BMI2__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[2] & (1 << 8)) != 0;
#endif
}

OPENSSL_INLINE int CRYPTO_is_ADX_capable(void) {
#if defined(__ADX__)
  return 1;
#else
  return (OPENSSL_ia32cap_get()[2] & (1 << 19)) != 0;
#endif
}

#endif  // OPENSSL_X86 || OPENSSL_X86_64

#if defined(OPENSSL_ARM) || defined(OPENSSL_AARCH64)

#if defined(OPENSSL_APPLE) && defined(OPENSSL_ARM)
// We do not detect any features at runtime for Apple's 32-bit ARM platforms. On
// 64-bit ARM, we detect some post-ARMv8.0 features.
#define OPENSSL_STATIC_ARMCAP
#endif

// Normalize some older feature flags to their modern ACLE values.
// https://developer.arm.com/architectures/system-architectures/software-standards/acle
#if defined(__ARM_NEON__) && !defined(__ARM_NEON)
#define __ARM_NEON 1
#endif
#if defined(__ARM_FEATURE_CRYPTO)
#if !defined(__ARM_FEATURE_AES)
#define __ARM_FEATURE_AES 1
#endif
#if !defined(__ARM_FEATURE_SHA2)
#define __ARM_FEATURE_SHA2 1
#endif
#endif

#if !defined(OPENSSL_STATIC_ARMCAP)
// CRYPTO_is_NEON_capable_at_runtime returns true if the current CPU has a NEON
// unit. Note that |OPENSSL_armcap_P| also exists and contains the same
// information in a form that's easier for assembly to use.
OPENSSL_EXPORT int CRYPTO_is_NEON_capable_at_runtime(void);

// CRYPTO_is_ARMv8_AES_capable_at_runtime returns true if the current CPU
// supports the ARMv8 AES instruction.
int CRYPTO_is_ARMv8_AES_capable_at_runtime(void);

// CRYPTO_is_ARMv8_PMULL_capable_at_runtime returns true if the current CPU
// supports the ARMv8 PMULL instruction.
int CRYPTO_is_ARMv8_PMULL_capable_at_runtime(void);
#endif  // !OPENSSL_STATIC_ARMCAP

// CRYPTO_is_NEON_capable returns true if the current CPU has a NEON unit. If
// this is known statically, it is a constant inline function.
OPENSSL_INLINE int CRYPTO_is_NEON_capable(void) {
#if defined(OPENSSL_STATIC_ARMCAP_NEON) || defined(__ARM_NEON)
  return 1;
#elif defined(OPENSSL_STATIC_ARMCAP)
  return 0;
#else
  return CRYPTO_is_NEON_capable_at_runtime();
#endif
}

OPENSSL_INLINE int CRYPTO_is_ARMv8_AES_capable(void) {
#if defined(OPENSSL_STATIC_ARMCAP_AES) || defined(__ARM_FEATURE_AES)
  return 1;
#elif defined(OPENSSL_STATIC_ARMCAP)
  return 0;
#else
  return CRYPTO_is_ARMv8_AES_capable_at_runtime();
#endif
}

OPENSSL_INLINE int CRYPTO_is_ARMv8_PMULL_capable(void) {
#if defined(OPENSSL_STATIC_ARMCAP_PMULL) || defined(__ARM_FEATURE_AES)
  return 1;
#elif defined(OPENSSL_STATIC_ARMCAP)
  return 0;
#else
  return CRYPTO_is_ARMv8_PMULL_capable_at_runtime();
#endif
}

#endif  // OPENSSL_ARM || OPENSSL_AARCH64

#if defined(BORINGSSL_DISPATCH_TEST)
// Runtime CPU dispatch testing support

// BORINGSSL_function_hit is an array of flags. The following functions will
// set these flags if BORINGSSL_DISPATCH_TEST is defined.
//   0: aes_hw_ctr32_encrypt_blocks
//   1: aes_hw_encrypt
//   2: aesni_gcm_encrypt
//   3: aes_hw_set_encrypt_key
//   4: vpaes_encrypt
//   5: vpaes_set_encrypt_key
extern uint8_t BORINGSSL_function_hit[7];
#endif  // BORINGSSL_DISPATCH_TEST

// OPENSSL_vasprintf_internal is just like |vasprintf(3)|. If |system_malloc| is
// 0, memory will be allocated with |OPENSSL_malloc| and must be freed with
// |OPENSSL_free|. Otherwise the system |malloc| function is used and the memory
// must be freed with the system |free| function.
OPENSSL_EXPORT int OPENSSL_vasprintf_internal(char **str, const char *format,
                                              va_list args, int system_malloc)
    OPENSSL_PRINTF_FORMAT_FUNC(2, 0);

#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_INTERNAL_H
