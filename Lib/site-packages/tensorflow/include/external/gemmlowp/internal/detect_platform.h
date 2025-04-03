// Copyright 2018 The Gemmlowp Authors. All Rights Reserved.
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

// detect_platform.h: Sets up macros that control architecture-specific
// features of gemmlowp's implementation.

#ifndef GEMMLOWP_INTERNAL_DETECT_PLATFORM_H_
#define GEMMLOWP_INTERNAL_DETECT_PLATFORM_H_

// Our inline assembly path assume GCC/Clang syntax.
// Native Client doesn't seem to support inline assembly(?).
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__native_client__)
#define GEMMLOWP_ALLOW_INLINE_ASM
#endif

// Define macro statement that avoids inlining for GCC.
// For non-GCC, define as empty macro.
#if defined(__GNUC__)
#define GEMMLOWP_NOINLINE __attribute__((noinline))
#else
#define GEMMLOWP_NOINLINE
#endif

// Detect ARM, 32-bit or 64-bit
#ifdef __arm__
#define GEMMLOWP_ARM_32
#endif

#ifdef __aarch64__
#define GEMMLOWP_ARM_64
#endif

#if defined(GEMMLOWP_ARM_32) || defined(GEMMLOWP_ARM_64)
#define GEMMLOWP_ARM
#endif

// Detect MIPS, 32-bit or 64-bit
#if defined(__mips) && !defined(__LP64__)
#define GEMMLOWP_MIPS_32
#endif

#if defined(__mips) && defined(__LP64__)
#define GEMMLOWP_MIPS_64
#endif

#if defined(GEMMLOWP_MIPS_32) || defined(GEMMLOWP_MIPS_64)
#define GEMMLOWP_MIPS
#endif

// Detect x86, 32-bit or 64-bit
#if defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__i386)
#define GEMMLOWP_X86_32
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
#define GEMMLOWP_X86_64
#endif

#if defined(GEMMLOWP_X86_32) || defined(GEMMLOWP_X86_64)
#define GEMMLOWP_X86
#endif

// Detect WebAssembly SIMD.
#if defined(__wasm_simd128__)
#define GEMMLOWP_WASMSIMD
#endif

// Some of our optimized paths use inline assembly and for
// now we don't bother enabling some other optimized paths using intrinddics
// where we can't use inline assembly paths.
#ifdef GEMMLOWP_ALLOW_INLINE_ASM

// Detect NEON. It's important to check for both tokens.
#if (defined __ARM_NEON) || (defined __ARM_NEON__)
#define GEMMLOWP_NEON
#endif

// Convenience NEON tokens for 32-bit or 64-bit
#if defined(GEMMLOWP_NEON) && defined(GEMMLOWP_ARM_32)
#define GEMMLOWP_NEON_32
#endif

#if defined(GEMMLOWP_NEON) && defined(GEMMLOWP_ARM_64)
#define GEMMLOWP_NEON_64
#endif

// Detect MIPS MSA.
// Limit MSA optimizations to little-endian CPUs for now.
// TODO: Perhaps, eventually support MSA optimizations on big-endian CPUs?
#if defined(GEMMLOWP_MIPS) && (__mips_isa_rev >= 5) && defined(__mips_msa) && \
    defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define GEMMLOWP_MSA
#endif

// Convenience MIPS MSA tokens for 32-bit or 64-bit.
#if defined(GEMMLOWP_MSA) && defined(GEMMLOWP_MIPS_32)
#define GEMMLOWP_MSA_32
#endif

#if defined(GEMMLOWP_MSA) && defined(GEMMLOWP_MIPS_64)
#define GEMMLOWP_MSA_64
#endif

// compiler define for AVX2 -D GEMMLOWP_ENABLE_AVX2
// Detect AVX2
#if defined(__AVX2__) && defined(GEMMLOWP_ENABLE_AVX2)
#define GEMMLOWP_AVX2
// Detect SSE4.
// MSVC does not have __SSE4_1__ macro, but will enable SSE4
// when AVX is turned on.
#elif defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__AVX__))
#define GEMMLOWP_SSE4
// Detect SSE3.
#elif defined(__SSE3__)
#define GEMMLOWP_SSE3
#endif

// Convenience SSE4 tokens for 32-bit or 64-bit
#if defined(GEMMLOWP_SSE4) && defined(GEMMLOWP_X86_32) && \
    !defined(GEMMLOWP_DISABLE_SSE4)
#define GEMMLOWP_SSE4_32
#endif

#if defined(GEMMLOWP_SSE3) && defined(GEMMLOWP_X86_32)
#define GEMMLOWP_SSE3_32
#endif

#if defined(GEMMLOWP_SSE4) && defined(GEMMLOWP_X86_64) && \
    !defined(GEMMLOWP_DISABLE_SSE4)
#define GEMMLOWP_SSE4_64
#endif

#if defined(GEMMLOWP_SSE3) && defined(GEMMLOWP_X86_64)
#define GEMMLOWP_SSE3_64
#endif

#if defined(GEMMLOWP_AVX2) && defined(GEMMLOWP_X86_64)
#define GEMMLOWP_AVX2_64
#endif

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define GEMMLOWP_MARK_MEMORY_AS_INITIALIZED __msan_unpoison
#elif __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define GEMMLOWP_MARK_MEMORY_AS_INITIALIZED __asan_unpoison_memory_region
#endif
#endif

#endif  // GEMMLOWP_ALLOW_INLINE_ASM

// Detect Android. Don't conflate with ARM - we care about tuning
// for non-ARM Android devices too. This can be used in conjunction
// with x86 to tune differently for mobile x86 CPUs (Atom) vs. desktop x86 CPUs.
#if defined(__ANDROID__) || defined(ANDROID)
#define GEMMLOWP_ANDROID
#endif

#endif  // GEMMLOWP_INTERNAL_DETECT_PLATFORM_H_
