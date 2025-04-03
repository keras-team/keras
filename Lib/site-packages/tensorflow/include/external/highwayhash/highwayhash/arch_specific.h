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

#ifndef HIGHWAYHASH_ARCH_SPECIFIC_H_
#define HIGHWAYHASH_ARCH_SPECIFIC_H_

// WARNING: this is a "restricted" header because it is included from
// translation units compiled with different flags. This header and its
// dependencies must not define any function unless it is static inline and/or
// within namespace HH_TARGET_NAME.
//
// Background: older GCC/Clang require flags such as -mavx2 before AVX2 SIMD
// intrinsics can be used. These intrinsics are only used within blocks that
// first verify CPU capabilities. However, the flag also allows the compiler to
// generate AVX2 code in other places. This can violate the One Definition Rule,
// which requires multiple instances of a function with external linkage
// (e.g. extern inline in a header) to be "equivalent". To prevent the resulting
// crashes on non-AVX2 CPUs, any header (transitively) included from a
// translation unit compiled with different flags is "restricted". This means
// all function definitions must have internal linkage (e.g. static inline), or
// reside in namespace HH_TARGET_NAME, which expands to a name unique to the
// current compiler flags.
//
// Most C system headers are safe to include, but C++ headers should generally
// be avoided because they often do not specify static linkage and cannot
// reliably be wrapped in a namespace.

#include "highwayhash/compiler_specific.h"

#include <stdint.h>

#if HH_MSC_VERSION
#include <intrin.h>  // _byteswap_*
#endif

namespace highwayhash {

#if defined(__x86_64__) || defined(_M_X64)
#define HH_ARCH_X64 1
#else
#define HH_ARCH_X64 0
#endif

#if defined(__aarch64__) || defined(__arm64__)
#define HH_ARCH_AARCH64 1
#else
#define HH_ARCH_AARCH64 0
#endif

#ifdef __arm__
#define HH_ARCH_ARM 1
#else
#define HH_ARCH_ARM 0
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define HH_ARCH_NEON 1
#else
#define HH_ARCH_NEON 0
#endif

#if defined(__powerpc64__) || defined(_M_PPC)
#define HH_ARCH_PPC 1
#else
#define HH_ARCH_PPC 0
#endif

// Target := instruction set extension(s) such as SSE41. A translation unit can
// only provide a single target-specific implementation because they require
// different compiler flags.

// Either the build system specifies the target by defining HH_TARGET_NAME
// (which is necessary for Portable on X64, and SSE41 on MSVC), or we'll choose
// the most efficient one that can be compiled given the current flags:
#ifndef HH_TARGET_NAME

// To avoid excessive code size and dispatch overhead, we only support a few
// groups of extensions, e.g. FMA+BMI2+AVX+AVX2 =: "AVX2". These names must
// match the HH_TARGET_* suffixes below.
#ifdef __AVX2__
#define HH_TARGET_NAME AVX2
// MSVC does not set SSE4_1, but it does set AVX; checking for the latter means
// we at least get SSE4 on machines supporting AVX but not AVX2.
// https://stackoverflow.com/questions/18563978/detect-the-availability-of-sse-sse2-instruction-set-in-visual-studio
#elif defined(__SSE4_1__) || (HH_MSC_VERSION != 0 && defined(__AVX__))
#define HH_TARGET_NAME SSE41
#elif defined(__VSX__)
#define HH_TARGET_NAME VSX
#elif HH_ARCH_NEON
#define HH_TARGET_NAME NEON
#else
#define HH_TARGET_NAME Portable
#endif

#endif  // HH_TARGET_NAME

#define HH_CONCAT(first, second) first##second
// Required due to macro expansion rules.
#define HH_EXPAND_CONCAT(first, second) HH_CONCAT(first, second)
// Appends HH_TARGET_NAME to "identifier_prefix".
#define HH_ADD_TARGET_SUFFIX(identifier_prefix) \
  HH_EXPAND_CONCAT(identifier_prefix, HH_TARGET_NAME)

// HH_TARGET expands to an integer constant. Typical usage: HHStateT<HH_TARGET>.
// This ensures your code will work correctly when compiler flags are changed,
// and benefit from subsequently added targets/specializations.
#define HH_TARGET HH_ADD_TARGET_SUFFIX(HH_TARGET_)

// Deprecated former name of HH_TARGET; please use HH_TARGET instead.
#define HH_TARGET_PREFERRED HH_TARGET

// Associate targets with integer literals so the preprocessor can compare them
// with HH_TARGET. Do not instantiate templates with these values - use
// HH_TARGET instead. Must be unique powers of two, see TargetBits. Always
// defined even if unavailable on this HH_ARCH to allow calling TargetName.
// The suffixes must match the HH_TARGET_NAME identifiers.
#define HH_TARGET_Portable 1
#define HH_TARGET_SSE41 2
#define HH_TARGET_AVX2 4
#define HH_TARGET_VSX 8
#define HH_TARGET_NEON 16

// Bit array for one or more HH_TARGET_*. Used to indicate which target(s) are
// supported or were called by InstructionSets::RunAll.
using TargetBits = unsigned;

namespace HH_TARGET_NAME {

// Calls func(bit_value) for every nonzero bit in "bits".
template <class Func>
void ForeachTarget(TargetBits bits, const Func& func) {
  while (bits != 0) {
    const TargetBits lowest = bits & (~bits + 1);
    func(lowest);
    bits &= ~lowest;
  }
}

}  // namespace HH_TARGET_NAME

// Returns a brief human-readable string literal identifying one of the above
// bits, or nullptr if zero, multiple, or unknown bits are set.
const char* TargetName(const TargetBits target_bit);

// Returns the nominal (without Turbo Boost) CPU clock rate [Hertz]. Useful for
// (roughly) characterizing the CPU speed.
double NominalClockRate();

// Returns tsc_timer frequency, useful for converting ticks to seconds. This is
// unaffected by CPU throttling ("invariant"). Thread-safe. Returns timebase
// frequency on PPC and NominalClockRate on all other platforms.
double InvariantTicksPerSecond();

#if HH_ARCH_X64

// Calls CPUID instruction with eax=level and ecx=count and returns the result
// in abcd array where abcd = {eax, ebx, ecx, edx} (hence the name abcd).
void Cpuid(const uint32_t level, const uint32_t count,
           uint32_t* HH_RESTRICT abcd);

// Returns the APIC ID of the CPU on which we're currently running.
uint32_t ApicId();

#endif  // HH_ARCH_X64

}  // namespace highwayhash

#endif  // HIGHWAYHASH_ARCH_SPECIFIC_H_
