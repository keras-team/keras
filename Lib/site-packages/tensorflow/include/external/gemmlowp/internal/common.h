// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

// common.h: contains stuff that's used throughout gemmlowp
// and should always be available.

#ifndef GEMMLOWP_INTERNAL_COMMON_H_
#define GEMMLOWP_INTERNAL_COMMON_H_

#include "../internal/platform.h"
#include "../profiling/pthread_everywhere.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include "../internal/detect_platform.h"
#include "../profiling/instrumentation.h"

namespace gemmlowp {

// Standard cache line size. Useful to optimize alignment and
// prefetches. Ideally we would query this at runtime, however
// 64 byte cache lines are the vast majority, and even if it's
// wrong on some device, it will be wrong by no more than a 2x factor,
// which should be acceptable.
const int kDefaultCacheLineSize = 64;

// Default L1 and L2 data cache sizes.
// The L1 cache size is assumed to be for each core.
// The L2 cache size is assumed to be shared among all cores. What
// we call 'L2' here is effectively top-level cache.
//
// On x86, we should ideally query this at
// runtime. On ARM, the instruction to query this is privileged and
// Android kernels do not expose it to userspace. Fortunately, the majority
// of ARM devices have roughly comparable values:
//   Nexus 5: L1 16k, L2 1M
//   Android One: L1 32k, L2 512k
// The following values are equal to or somewhat lower than that, and were
// found to perform well on both the Nexus 5 and Android One.
// Of course, these values are in principle too low for typical x86 CPUs
// where we should set the L2 value to (L3 cache size / number of cores) at
// least.
//
#if defined(GEMMLOWP_ARM) && defined(__APPLE__)
// iPhone/iPad
const int kDefaultL1CacheSize = 48 * 1024;
const int kDefaultL2CacheSize = 2 * 1024 * 1024;
#elif defined(GEMMLOWP_ARM) || defined(GEMMLOWP_ANDROID)
// Other ARM or ARM-like hardware (Android implies ARM-like) so here it's OK
// to tune for ARM, although on x86 Atom we might be able to query
// cache sizes at runtime, which would be better.
const int kDefaultL1CacheSize = 16 * 1024;
const int kDefaultL2CacheSize = 384 * 1024;
#elif defined(GEMMLOWP_X86_64)
// x86-64 and not Android. Therefore, likely desktop-class x86 hardware.
// Thus we assume larger cache sizes, though we really should query
// them at runtime.
const int kDefaultL1CacheSize = 32 * 1024;
const int kDefaultL2CacheSize = 4 * 1024 * 1024;
#elif defined(GEMMLOWP_X86_32)
// x86-32 and not Android. Same as x86-64 but less bullish.
const int kDefaultL1CacheSize = 32 * 1024;
const int kDefaultL2CacheSize = 2 * 1024 * 1024;
#elif defined(GEMMLOWP_MIPS)
// MIPS and not Android. TODO: MIPS and Android?
const int kDefaultL1CacheSize = 32 * 1024;
const int kDefaultL2CacheSize = 1024 * 1024;
#else
// Less common hardware. Maybe some unusual or older or embedded thing.
// Assume smaller caches, but don't depart too far from what we do
// on ARM/Android to avoid accidentally exposing unexpected behavior.
const int kDefaultL1CacheSize = 16 * 1024;
const int kDefaultL2CacheSize = 256 * 1024;
#endif

// The proportion of the cache that we intend to use for storing
// RHS blocks. This should be between 0 and 1, and typically closer to 1,
// as we typically want to use most of the L2 cache for storing a large
// RHS block.
#if defined(GEMMLOWP_X86)
// For IA, use the entire L2 cache for the RHS matrix. LHS matrix is not blocked
// for L2 cache.
const float kDefaultL2RhsFactor = 1.00f;
#else
const float kDefaultL2RhsFactor = 0.75f;
#endif

// The number of bytes in a SIMD register. This is used to determine
// the dimensions of PackingRegisterBlock so that such blocks can
// be efficiently loaded into registers, so that packing code can
// work within registers as much as possible.
// In the non-SIMD generic fallback code, this is just a generic array
// size, so any size would work there. Different platforms may set this
// to different values but must ensure that their own optimized packing paths
// are consistent with this value.

#ifdef GEMMLOWP_AVX2
const int kRegisterSize = 32;
#else
const int kRegisterSize = 16;
#endif

// Hints the CPU to prefetch the cache line containing ptr.
inline void Prefetch(const void* ptr) {
#if defined GEMMLOWP_ARM_64 && defined GEMMLOWP_ALLOW_INLINE_ASM
  // Aarch64 has very detailed prefetch instructions, that compilers
  // can't know how to map __builtin_prefetch to, and as a result, don't,
  // leaving __builtin_prefetch a no-op on this architecture.
  // For our purposes, "pldl1keep" is usually what we want, meaning:
  // "prefetch for load, into L1 cache, using each value multiple times".
  asm volatile("prfm pldl1keep, [%[ptr]]\n" ::[ptr] "r"(ptr) :);
#elif defined \
    __GNUC__  // Clang and GCC define __GNUC__ and have __builtin_prefetch.
  __builtin_prefetch(ptr);
#else
  (void)ptr;
#endif
}

// Returns the runtime argument rounded down to the nearest multiple of
// the fixed Modulus.
template <unsigned Modulus, typename Integer>
Integer RoundDown(Integer i) {
  return i - (i % Modulus);
}

// Returns the runtime argument rounded up to the nearest multiple of
// the fixed Modulus.
template <unsigned Modulus, typename Integer>
Integer RoundUp(Integer i) {
  return RoundDown<Modulus>(i + Modulus - 1);
}

// Returns the quotient a / b rounded up ('ceil') to the nearest integer.
template <typename Integer>
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}

// Returns the argument rounded up to the nearest power of two.
template <typename Integer>
Integer RoundUpToPowerOfTwo(Integer n) {
  Integer i = n - 1;
  i |= i >> 1;
  i |= i >> 2;
  i |= i >> 4;
  i |= i >> 8;
  i |= i >> 16;
  return i + 1;
}

template <int N>
struct IsPowerOfTwo {
  static constexpr bool value = !(N & (N - 1));
};

template <typename T>
void MarkMemoryAsInitialized(T* ptr, int size) {
#ifdef GEMMLOWP_MARK_MEMORY_AS_INITIALIZED
  GEMMLOWP_MARK_MEMORY_AS_INITIALIZED(static_cast<void*>(ptr),
                                      size * sizeof(T));
#else
  (void)ptr;
  (void)size;
#endif
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_COMMON_H_
