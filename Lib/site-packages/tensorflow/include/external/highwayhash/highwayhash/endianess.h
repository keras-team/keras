// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIGHWAYHASH_ENDIANESS_H_
#define HIGHWAYHASH_ENDIANESS_H_

// WARNING: this is a "restricted" header because it is included from
// translation units compiled with different flags. This header and its
// dependencies must not define any function unless it is static inline and/or
// within namespace HH_TARGET_NAME. See arch_specific.h for details.

#include <stdint.h>

#if defined(BYTE_ORDER) && defined(LITTLE_ENDIAN) && defined(BIG_ENDIAN)

  /* Someone has already included <endian.h> or equivalent. */

#elif defined(__LITTLE_ENDIAN__)

#  define HH_IS_LITTLE_ENDIAN  1
#  define HH_IS_BIG_ENDIAN     0
#  ifdef __BIG_ENDIAN__
#    error "Platform is both little and big endian?"
#  endif

#elif defined(__BIG_ENDIAN__)

#    define HH_IS_LITTLE_ENDIAN  0
#    define HH_IS_BIG_ENDIAN     1

#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
      defined(__ORDER_LITTLE_ENDIAN__)

#  define HH_IS_LITTLE_ENDIAN  (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#  define HH_IS_BIG_ENDIAN     (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)

#elif defined(__linux__) || defined(__CYGWIN__) || defined( __GNUC__ ) || \
      defined( __GNU_LIBRARY__ )

#  include <endian.h>

#elif defined(__OpenBSD__) || defined(__NetBSD__) || defined(__FreeBSD__) || \
      defined(__DragonFly__)

#  include <sys/endian.h>

#elif defined(_WIN32)

#define HH_IS_LITTLE_ENDIAN 1
#define HH_IS_BIG_ENDIAN 0

#else

#  error "Unsupported platform.  Cannot determine byte order."

#endif


#ifndef HH_IS_LITTLE_ENDIAN
#  define HH_IS_LITTLE_ENDIAN  (BYTE_ORDER == LITTLE_ENDIAN)
#  define HH_IS_BIG_ENDIAN     (BYTE_ORDER == BIG_ENDIAN)
#endif


namespace highwayhash {

#if HH_IS_LITTLE_ENDIAN

static inline uint32_t le32_from_host(uint32_t x) { return x; }
static inline uint32_t host_from_le32(uint32_t x) { return x; }
static inline uint64_t le64_from_host(uint64_t x) { return x; }
static inline uint64_t host_from_le64(uint64_t x) { return x; }

#elif !HH_IS_BIG_ENDIAN

#  error "Unsupported byte order."

#elif defined(_WIN16) || defined(_WIN32) || defined(_WIN64)

#include <intrin.h>
static inline uint32_t host_from_le32(uint32_t x) { return _byteswap_ulong(x); }
static inline uint32_t le32_from_host(uint32_t x) { return _byteswap_ulong(x); }
static inline uint64_t host_from_le64(uint64_t x) { return _byteswap_uint64(x);}
static inline uint64_t le64_from_host(uint64_t x) { return _byteswap_uint64(x);}

#else

static inline uint32_t host_from_le32(uint32_t x) {return __builtin_bswap32(x);}
static inline uint32_t le32_from_host(uint32_t x) {return __builtin_bswap32(x);}
static inline uint64_t host_from_le64(uint64_t x) {return __builtin_bswap64(x);}
static inline uint64_t le64_from_host(uint64_t x) {return __builtin_bswap64(x);}

#endif

}  // namespace highwayhash

#endif  // HIGHWAYHASH_ENDIANESS_H_
