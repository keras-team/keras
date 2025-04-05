// Copyright 2011 Google Inc. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Various stubs for the open-source version of Snappy.

#ifndef THIRD_PARTY_SNAPPY_OPENSOURCE_SNAPPY_STUBS_INTERNAL_H_
#define THIRD_PARTY_SNAPPY_OPENSOURCE_SNAPPY_STUBS_INTERNAL_H_

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdint.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#if HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif  // defined(_MSC_VER)

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define SNAPPY_ANNOTATE_MEMORY_IS_INITIALIZED(address, size) \
    __msan_unpoison((address), (size))
#else
#define SNAPPY_ANNOTATE_MEMORY_IS_INITIALIZED(address, size) /* empty */
#endif  // __has_feature(memory_sanitizer)

#include "snappy-stubs-public.h"

// Used to enable 64-bit optimized versions of some routines.
#if defined(__PPC64__) || defined(__powerpc64__)
#define ARCH_PPC 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ARCH_ARM 1
#endif

// Needed by OS X, among others.
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

// The size of an array, if known at compile-time.
// Will give unexpected results if used on a pointer.
// We undefine it first, since some compilers already have a definition.
#ifdef ARRAYSIZE
#undef ARRAYSIZE
#endif
#define ARRAYSIZE(a) int{sizeof(a) / sizeof(*(a))}

// Static prediction hints.
#if HAVE_BUILTIN_EXPECT
#define SNAPPY_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define SNAPPY_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define SNAPPY_PREDICT_FALSE(x) x
#define SNAPPY_PREDICT_TRUE(x) x
#endif  // HAVE_BUILTIN_EXPECT

// Inlining hints.
#if HAVE_ATTRIBUTE_ALWAYS_INLINE
#define SNAPPY_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#else
#define SNAPPY_ATTRIBUTE_ALWAYS_INLINE
#endif  // HAVE_ATTRIBUTE_ALWAYS_INLINE

#if HAVE_BUILTIN_PREFETCH
#define SNAPPY_PREFETCH(ptr) __builtin_prefetch(ptr, 0, 3)
#else
#define SNAPPY_PREFETCH(ptr) (void)(ptr)
#endif

// Stubbed version of ABSL_FLAG.
//
// In the open source version, flags can only be changed at compile time.
#define SNAPPY_FLAG(flag_type, flag_name, default_value, help) \
  flag_type FLAGS_ ## flag_name = default_value

namespace snappy {

// Stubbed version of absl::GetFlag().
template <typename T>
inline T GetFlag(T flag) { return flag; }

static const uint32_t kuint32max = std::numeric_limits<uint32_t>::max();
static const int64_t kint64max = std::numeric_limits<int64_t>::max();

// Potentially unaligned loads and stores.

inline uint16_t UNALIGNED_LOAD16(const void *p) {
  // Compiles to a single movzx/ldrh on clang/gcc/msvc.
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline uint32_t UNALIGNED_LOAD32(const void *p) {
  // Compiles to a single mov/ldr on clang/gcc/msvc.
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline uint64_t UNALIGNED_LOAD64(const void *p) {
  // Compiles to a single mov/ldr on clang/gcc/msvc.
  uint64_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline void UNALIGNED_STORE16(void *p, uint16_t v) {
  // Compiles to a single mov/strh on clang/gcc/msvc.
  std::memcpy(p, &v, sizeof(v));
}

inline void UNALIGNED_STORE32(void *p, uint32_t v) {
  // Compiles to a single mov/str on clang/gcc/msvc.
  std::memcpy(p, &v, sizeof(v));
}

inline void UNALIGNED_STORE64(void *p, uint64_t v) {
  // Compiles to a single mov/str on clang/gcc/msvc.
  std::memcpy(p, &v, sizeof(v));
}

// Convert to little-endian storage, opposite of network format.
// Convert x from host to little endian: x = LittleEndian.FromHost(x);
// convert x from little endian to host: x = LittleEndian.ToHost(x);
//
//  Store values into unaligned memory converting to little endian order:
//    LittleEndian.Store16(p, x);
//
//  Load unaligned values stored in little endian converting to host order:
//    x = LittleEndian.Load16(p);
class LittleEndian {
 public:
  // Functions to do unaligned loads and stores in little-endian order.
  static inline uint16_t Load16(const void *ptr) {
    // Compiles to a single mov/str on recent clang and gcc.
#if SNAPPY_IS_BIG_ENDIAN
    const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(ptr);
    return (static_cast<uint16_t>(buffer[0])) |
            (static_cast<uint16_t>(buffer[1]) << 8);
#else
    // memcpy() turns into a single instruction early in the optimization
    // pipeline (relatively to a series of byte accesses). So, using memcpy
    // instead of byte accesses may lead to better decisions in more stages of
    // the optimization pipeline.
    uint16_t value;
    std::memcpy(&value, ptr, 2);
    return value;
#endif
  }

  static inline uint32_t Load32(const void *ptr) {
    // Compiles to a single mov/str on recent clang and gcc.
#if SNAPPY_IS_BIG_ENDIAN
    const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(ptr);
    return (static_cast<uint32_t>(buffer[0])) |
            (static_cast<uint32_t>(buffer[1]) << 8) |
            (static_cast<uint32_t>(buffer[2]) << 16) |
            (static_cast<uint32_t>(buffer[3]) << 24);
#else
    // See Load16() for the rationale of using memcpy().
    uint32_t value;
    std::memcpy(&value, ptr, 4);
    return value;
#endif
  }

  static inline uint64_t Load64(const void *ptr) {
    // Compiles to a single mov/str on recent clang and gcc.
#if SNAPPY_IS_BIG_ENDIAN
    const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(ptr);
    return (static_cast<uint64_t>(buffer[0])) |
            (static_cast<uint64_t>(buffer[1]) << 8) |
            (static_cast<uint64_t>(buffer[2]) << 16) |
            (static_cast<uint64_t>(buffer[3]) << 24) |
            (static_cast<uint64_t>(buffer[4]) << 32) |
            (static_cast<uint64_t>(buffer[5]) << 40) |
            (static_cast<uint64_t>(buffer[6]) << 48) |
            (static_cast<uint64_t>(buffer[7]) << 56);
#else
    // See Load16() for the rationale of using memcpy().
    uint64_t value;
    std::memcpy(&value, ptr, 8);
    return value;
#endif
  }

  static inline void Store16(void *dst, uint16_t value) {
    // Compiles to a single mov/str on recent clang and gcc.
#if SNAPPY_IS_BIG_ENDIAN
    uint8_t* const buffer = reinterpret_cast<uint8_t*>(dst);
    buffer[0] = static_cast<uint8_t>(value);
    buffer[1] = static_cast<uint8_t>(value >> 8);
#else
    // See Load16() for the rationale of using memcpy().
    std::memcpy(dst, &value, 2);
#endif
  }

  static void Store32(void *dst, uint32_t value) {
    // Compiles to a single mov/str on recent clang and gcc.
#if SNAPPY_IS_BIG_ENDIAN
    uint8_t* const buffer = reinterpret_cast<uint8_t*>(dst);
    buffer[0] = static_cast<uint8_t>(value);
    buffer[1] = static_cast<uint8_t>(value >> 8);
    buffer[2] = static_cast<uint8_t>(value >> 16);
    buffer[3] = static_cast<uint8_t>(value >> 24);
#else
    // See Load16() for the rationale of using memcpy().
    std::memcpy(dst, &value, 4);
#endif
  }

  static void Store64(void* dst, uint64_t value) {
    // Compiles to a single mov/str on recent clang and gcc.
#if SNAPPY_IS_BIG_ENDIAN
    uint8_t* const buffer = reinterpret_cast<uint8_t*>(dst);
    buffer[0] = static_cast<uint8_t>(value);
    buffer[1] = static_cast<uint8_t>(value >> 8);
    buffer[2] = static_cast<uint8_t>(value >> 16);
    buffer[3] = static_cast<uint8_t>(value >> 24);
    buffer[4] = static_cast<uint8_t>(value >> 32);
    buffer[5] = static_cast<uint8_t>(value >> 40);
    buffer[6] = static_cast<uint8_t>(value >> 48);
    buffer[7] = static_cast<uint8_t>(value >> 56);
#else
    // See Load16() for the rationale of using memcpy().
    std::memcpy(dst, &value, 8);
#endif
  }

  static inline constexpr bool IsLittleEndian() {
#if SNAPPY_IS_BIG_ENDIAN
    return false;
#else
    return true;
#endif  // SNAPPY_IS_BIG_ENDIAN
  }
};

// Some bit-manipulation functions.
class Bits {
 public:
  // Return floor(log2(n)) for positive integer n.
  static int Log2FloorNonZero(uint32_t n);

  // Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
  static int Log2Floor(uint32_t n);

  // Return the first set least / most significant bit, 0-indexed.  Returns an
  // undefined value if n == 0.  FindLSBSetNonZero() is similar to ffs() except
  // that it's 0-indexed.
  static int FindLSBSetNonZero(uint32_t n);

  static int FindLSBSetNonZero64(uint64_t n);

 private:
  // No copying
  Bits(const Bits&);
  void operator=(const Bits&);
};

#if HAVE_BUILTIN_CTZ

inline int Bits::Log2FloorNonZero(uint32_t n) {
  assert(n != 0);
  // (31 ^ x) is equivalent to (31 - x) for x in [0, 31]. An easy proof
  // represents subtraction in base 2 and observes that there's no carry.
  //
  // GCC and Clang represent __builtin_clz on x86 as 31 ^ _bit_scan_reverse(x).
  // Using "31 ^" here instead of "31 -" allows the optimizer to strip the
  // function body down to _bit_scan_reverse(x).
  return 31 ^ __builtin_clz(n);
}

inline int Bits::Log2Floor(uint32_t n) {
  return (n == 0) ? -1 : Bits::Log2FloorNonZero(n);
}

inline int Bits::FindLSBSetNonZero(uint32_t n) {
  assert(n != 0);
  return __builtin_ctz(n);
}

#elif defined(_MSC_VER)

inline int Bits::Log2FloorNonZero(uint32_t n) {
  assert(n != 0);
  // NOLINTNEXTLINE(runtime/int): The MSVC intrinsic demands unsigned long.
  unsigned long where;
  _BitScanReverse(&where, n);
  return static_cast<int>(where);
}

inline int Bits::Log2Floor(uint32_t n) {
  // NOLINTNEXTLINE(runtime/int): The MSVC intrinsic demands unsigned long.
  unsigned long where;
  if (_BitScanReverse(&where, n))
    return static_cast<int>(where);
  return -1;
}

inline int Bits::FindLSBSetNonZero(uint32_t n) {
  assert(n != 0);
  // NOLINTNEXTLINE(runtime/int): The MSVC intrinsic demands unsigned long.
  unsigned long where;
  if (_BitScanForward(&where, n))
    return static_cast<int>(where);
  return 32;
}

#else  // Portable versions.

inline int Bits::Log2FloorNonZero(uint32_t n) {
  assert(n != 0);

  int log = 0;
  uint32_t value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

inline int Bits::Log2Floor(uint32_t n) {
  return (n == 0) ? -1 : Bits::Log2FloorNonZero(n);
}

inline int Bits::FindLSBSetNonZero(uint32_t n) {
  assert(n != 0);

  int rc = 31;
  for (int i = 4, shift = 1 << 4; i >= 0; --i) {
    const uint32_t x = n << shift;
    if (x != 0) {
      n = x;
      rc -= shift;
    }
    shift >>= 1;
  }
  return rc;
}

#endif  // End portable versions.

#if HAVE_BUILTIN_CTZ

inline int Bits::FindLSBSetNonZero64(uint64_t n) {
  assert(n != 0);
  return __builtin_ctzll(n);
}

#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
// _BitScanForward64() is only available on x64 and ARM64.

inline int Bits::FindLSBSetNonZero64(uint64_t n) {
  assert(n != 0);
  // NOLINTNEXTLINE(runtime/int): The MSVC intrinsic demands unsigned long.
  unsigned long where;
  if (_BitScanForward64(&where, n))
    return static_cast<int>(where);
  return 64;
}

#else  // Portable version.

// FindLSBSetNonZero64() is defined in terms of FindLSBSetNonZero().
inline int Bits::FindLSBSetNonZero64(uint64_t n) {
  assert(n != 0);

  const uint32_t bottombits = static_cast<uint32_t>(n);
  if (bottombits == 0) {
    // Bottom bits are zero, so scan the top bits.
    return 32 + FindLSBSetNonZero(static_cast<uint32_t>(n >> 32));
  } else {
    return FindLSBSetNonZero(bottombits);
  }
}

#endif  // HAVE_BUILTIN_CTZ

// Variable-length integer encoding.
class Varint {
 public:
  // Maximum lengths of varint encoding of uint32_t.
  static const int kMax32 = 5;

  // Attempts to parse a varint32 from a prefix of the bytes in [ptr,limit-1].
  // Never reads a character at or beyond limit.  If a valid/terminated varint32
  // was found in the range, stores it in *OUTPUT and returns a pointer just
  // past the last byte of the varint32. Else returns NULL.  On success,
  // "result <= limit".
  static const char* Parse32WithLimit(const char* ptr, const char* limit,
                                      uint32_t* OUTPUT);

  // REQUIRES   "ptr" points to a buffer of length sufficient to hold "v".
  // EFFECTS    Encodes "v" into "ptr" and returns a pointer to the
  //            byte just past the last encoded byte.
  static char* Encode32(char* ptr, uint32_t v);

  // EFFECTS    Appends the varint representation of "value" to "*s".
  static void Append32(std::string* s, uint32_t value);
};

inline const char* Varint::Parse32WithLimit(const char* p,
                                            const char* l,
                                            uint32_t* OUTPUT) {
  const unsigned char* ptr = reinterpret_cast<const unsigned char*>(p);
  const unsigned char* limit = reinterpret_cast<const unsigned char*>(l);
  uint32_t b, result;
  if (ptr >= limit) return NULL;
  b = *(ptr++); result = b & 127;          if (b < 128) goto done;
  if (ptr >= limit) return NULL;
  b = *(ptr++); result |= (b & 127) <<  7; if (b < 128) goto done;
  if (ptr >= limit) return NULL;
  b = *(ptr++); result |= (b & 127) << 14; if (b < 128) goto done;
  if (ptr >= limit) return NULL;
  b = *(ptr++); result |= (b & 127) << 21; if (b < 128) goto done;
  if (ptr >= limit) return NULL;
  b = *(ptr++); result |= (b & 127) << 28; if (b < 16) goto done;
  return NULL;       // Value is too long to be a varint32
 done:
  *OUTPUT = result;
  return reinterpret_cast<const char*>(ptr);
}

inline char* Varint::Encode32(char* sptr, uint32_t v) {
  // Operate on characters as unsigneds
  uint8_t* ptr = reinterpret_cast<uint8_t*>(sptr);
  static const uint8_t B = 128;
  if (v < (1 << 7)) {
    *(ptr++) = static_cast<uint8_t>(v);
  } else if (v < (1 << 14)) {
    *(ptr++) = static_cast<uint8_t>(v | B);
    *(ptr++) = static_cast<uint8_t>(v >> 7);
  } else if (v < (1 << 21)) {
    *(ptr++) = static_cast<uint8_t>(v | B);
    *(ptr++) = static_cast<uint8_t>((v >> 7) | B);
    *(ptr++) = static_cast<uint8_t>(v >> 14);
  } else if (v < (1 << 28)) {
    *(ptr++) = static_cast<uint8_t>(v | B);
    *(ptr++) = static_cast<uint8_t>((v >> 7) | B);
    *(ptr++) = static_cast<uint8_t>((v >> 14) | B);
    *(ptr++) = static_cast<uint8_t>(v >> 21);
  } else {
    *(ptr++) = static_cast<uint8_t>(v | B);
    *(ptr++) = static_cast<uint8_t>((v>>7) | B);
    *(ptr++) = static_cast<uint8_t>((v>>14) | B);
    *(ptr++) = static_cast<uint8_t>((v>>21) | B);
    *(ptr++) = static_cast<uint8_t>(v >> 28);
  }
  return reinterpret_cast<char*>(ptr);
}

// If you know the internal layout of the std::string in use, you can
// replace this function with one that resizes the string without
// filling the new space with zeros (if applicable) --
// it will be non-portable but faster.
inline void STLStringResizeUninitialized(std::string* s, size_t new_size) {
  s->resize(new_size);
}

// Return a mutable char* pointing to a string's internal buffer,
// which may not be null-terminated. Writing through this pointer will
// modify the string.
//
// string_as_array(&str)[i] is valid for 0 <= i < str.size() until the
// next call to a string method that invalidates iterators.
//
// As of 2006-04, there is no standard-blessed way of getting a
// mutable reference to a string's internal buffer. However, issue 530
// (http://www.open-std.org/JTC1/SC22/WG21/docs/lwg-defects.html#530)
// proposes this as the method. It will officially be part of the standard
// for C++0x. This should already work on all current implementations.
inline char* string_as_array(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

}  // namespace snappy

#endif  // THIRD_PARTY_SNAPPY_OPENSOURCE_SNAPPY_STUBS_INTERNAL_H_
