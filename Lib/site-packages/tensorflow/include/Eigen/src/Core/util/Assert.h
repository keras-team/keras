// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022, The Eigen authors.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CORE_UTIL_ASSERT_H
#define EIGEN_CORE_UTIL_ASSERT_H

// Eigen custom assert function.
//
// The combination of Eigen's relative includes and cassert's `assert` function
// (or any usage of the __FILE__ macro) can lead to ODR issues:
// a header included using different relative paths in two different TUs will
// have two different token-for-token definitions, since __FILE__ is expanded
// as an in-line string with different values.  Normally this would be
// harmless - the linker would just choose one definition. However, it breaks
// with C++20 modules when functions in different modules have different
// definitions.
//
// To get around this, we need to use __builtin_FILE() when available, which is
// considered a single token, and thus satisfies the ODR.

// Only define eigen_plain_assert if we are debugging, and either
//  - we are not compiling for GPU, or
//  - gpu debugging is enabled.
#if !defined(EIGEN_NO_DEBUG) && (!defined(EIGEN_GPU_COMPILE_PHASE) || !defined(EIGEN_NO_DEBUG_GPU))

#include <cassert>

#ifndef EIGEN_USE_CUSTOM_PLAIN_ASSERT
// Disable new custom asserts by default for now.
#define EIGEN_USE_CUSTOM_PLAIN_ASSERT 0
#endif

#if EIGEN_USE_CUSTOM_PLAIN_ASSERT

#ifndef EIGEN_HAS_BUILTIN_FILE
// Clang can check if __builtin_FILE() is supported.
// GCC > 5, MSVC 2019 14.26 (1926) all have __builtin_FILE().
//
// For NVCC, it's more complicated.  Through trial-and-error:
//   - nvcc+gcc supports __builtin_FILE() on host, and on device after CUDA 11.
//   - nvcc+msvc supports __builtin_FILE() only after CUDA 11.
#if (EIGEN_HAS_BUILTIN(__builtin_FILE) && (EIGEN_COMP_CLANG || !defined(EIGEN_CUDA_ARCH))) ||            \
    (EIGEN_GNUC_STRICT_AT_LEAST(5, 0, 0) && (EIGEN_COMP_NVCC >= 110000 || !defined(EIGEN_CUDA_ARCH))) || \
    (EIGEN_COMP_MSVC >= 1926 && (!EIGEN_COMP_NVCC || EIGEN_COMP_NVCC >= 110000))
#define EIGEN_HAS_BUILTIN_FILE 1
#else
#define EIGEN_HAS_BUILTIN_FILE 0
#endif
#endif  // EIGEN_HAS_BUILTIN_FILE

#if EIGEN_HAS_BUILTIN_FILE
#define EIGEN_BUILTIN_FILE __builtin_FILE()
#define EIGEN_BUILTIN_LINE __builtin_LINE()
#else
// Default (potentially unsafe) values.
#define EIGEN_BUILTIN_FILE __FILE__
#define EIGEN_BUILTIN_LINE __LINE__
#endif

// Use __PRETTY_FUNCTION__ when available, since it is more descriptive, as
// __builtin_FUNCTION() only returns the undecorated function name.
// This should still be okay ODR-wise since it is a compiler-specific fixed
// value.  Mixing compilers will likely lead to ODR violations anyways.
#if EIGEN_COMP_MSVC
#define EIGEN_BUILTIN_FUNCTION __FUNCSIG__
#elif EIGEN_COMP_GNUC
#define EIGEN_BUILTIN_FUNCTION __PRETTY_FUNCTION__
#else
#define EIGEN_BUILTIN_FUNCTION __func__
#endif

namespace Eigen {
namespace internal {

// Generic default assert handler.
template <typename EnableIf = void, typename... EmptyArgs>
struct assert_handler_impl {
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static inline void run(const char* expression, const char* file, unsigned line,
                                                             const char* function) {
#ifdef EIGEN_GPU_COMPILE_PHASE
    // GPU device code doesn't allow stderr or abort, so use printf and raise an
    // illegal instruction exception to trigger a kernel failure.
#ifndef EIGEN_NO_IO
    printf("Assertion failed at %s:%u in %s: %s\n", file == nullptr ? "<file>" : file, line,
           function == nullptr ? "<function>" : function, expression);
#endif
    __trap();

#else  // EIGEN_GPU_COMPILE_PHASE

    // Print to stderr and abort, as specified in <cassert>.
#ifndef EIGEN_NO_IO
    fprintf(stderr, "Assertion failed at %s:%u in %s: %s\n", file == nullptr ? "<file>" : file, line,
            function == nullptr ? "<function>" : function, expression);
#endif
    std::abort();

#endif  // EIGEN_GPU_COMPILE_PHASE
  }
};

// Use POSIX __assert_fail handler when available.
//
// This allows us to integrate with systems that have custom handlers.
//
// NOTE: this handler is not always available on all POSIX systems (otherwise
// we could simply test for __unix__ or similar).  The handler function name
// seems to depend on the specific toolchain implementation, and differs between
// compilers, platforms, OSes, etc.  Hence, we detect support via SFINAE.
template <typename... EmptyArgs>
struct assert_handler_impl<void_t<decltype(__assert_fail((const char*)nullptr,         // expression
                                                         (const char*)nullptr,         // file
                                                         0,                            // line
                                                         (const char*)nullptr,         // function
                                                         std::declval<EmptyArgs>()...  // Empty substitution required
                                                                                       // for SFINAE.
                                                         ))>,
                           EmptyArgs...> {
  EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE static inline void run(const char* expression, const char* file, unsigned line,
                                                             const char* function) {
    // GCC requires this call to be dependent on the template parameters.
    __assert_fail(expression, file, line, function, std::declval<EmptyArgs>()...);
  }
};

EIGEN_DEVICE_FUNC EIGEN_DONT_INLINE inline void __assert_handler(const char* expression, const char* file,
                                                                 unsigned line, const char* function) {
  assert_handler_impl<>::run(expression, file, line, function);
}

}  // namespace internal
}  // namespace Eigen

#define eigen_plain_assert(expression)                                                                                \
  (EIGEN_PREDICT_FALSE(!(expression)) ? Eigen::internal::__assert_handler(#expression, EIGEN_BUILTIN_FILE,            \
                                                                          EIGEN_BUILTIN_LINE, EIGEN_BUILTIN_FUNCTION) \
                                      : (void)0)

#else  // EIGEN_USE_CUSTOM_PLAIN_ASSERT

// Use regular assert.
#define eigen_plain_assert(condition) assert(condition)

#endif  // EIGEN_USE_CUSTOM_PLAIN_ASSERT

#else  // EIGEN_NO_DEBUG

#define eigen_plain_assert(condition) ((void)0)

#endif  // EIGEN_NO_DEBUG

#endif  // EIGEN_CORE_UTIL_ASSERT_H
