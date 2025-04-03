// Copyright 2015 Google Inc. All Rights Reserved.
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

#ifndef HIGHWAYHASH_COMPILER_SPECIFIC_H_
#define HIGHWAYHASH_COMPILER_SPECIFIC_H_

// WARNING: this is a "restricted" header because it is included from
// translation units compiled with different flags. This header and its
// dependencies must not define any function unless it is static inline and/or
// within namespace HH_TARGET_NAME. See arch_specific.h for details.

// Compiler

// #if is shorter and safer than #ifdef. *_VERSION are zero if not detected,
// otherwise 100 * major + minor version. Note that other packages check for
// #ifdef COMPILER_MSVC, so we cannot use that same name.

#ifdef _MSC_VER
#define HH_MSC_VERSION _MSC_VER
#else
#define HH_MSC_VERSION 0
#endif

#ifdef __GNUC__
#define HH_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#define HH_GCC_VERSION 0
#endif

#ifdef __clang__
#define HH_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
#else
#define HH_CLANG_VERSION 0
#endif

//-----------------------------------------------------------------------------

#if HH_GCC_VERSION && HH_GCC_VERSION < 408
#define HH_ALIGNAS(multiple) __attribute__((aligned(multiple)))
#else
#define HH_ALIGNAS(multiple) alignas(multiple)  // C++11
#endif

#if HH_MSC_VERSION
#define HH_RESTRICT __restrict
#elif HH_GCC_VERSION
#define HH_RESTRICT __restrict__
#else
#define HH_RESTRICT
#endif

#if HH_MSC_VERSION
#define HH_INLINE __forceinline
#define HH_NOINLINE __declspec(noinline)
#else
#define HH_INLINE inline
#define HH_NOINLINE __attribute__((noinline))
#endif

#if HH_MSC_VERSION
// Unsupported, __assume is not the same.
#define HH_LIKELY(expr) expr
#define HH_UNLIKELY(expr) expr
#else
#define HH_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define HH_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#endif

#if HH_MSC_VERSION
#include <intrin.h>
#pragma intrinsic(_ReadWriteBarrier)
#define HH_COMPILER_FENCE _ReadWriteBarrier()
#elif HH_GCC_VERSION
#define HH_COMPILER_FENCE asm volatile("" : : : "memory")
#else
#define HH_COMPILER_FENCE
#endif

#endif  // HIGHWAYHASH_COMPILER_SPECIFIC_H_
