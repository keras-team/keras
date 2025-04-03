/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_IMPL_CODEGEN_PORT_PLATFORM_H
#define GRPC_IMPL_CODEGEN_PORT_PLATFORM_H

/*
 * Define GPR_BACKWARDS_COMPATIBILITY_MODE to try harder to be ABI
 * compatible with older platforms (currently only on Linux)
 * Causes:
 *  - some libc calls to be gotten via dlsym
 *  - some syscalls to be made directly
 */

/*
 * Defines GRPC_USE_ABSL to use Abseil Common Libraries (C++)
 */
#ifndef GRPC_USE_ABSL
#define GRPC_USE_ABSL 1
#endif

/* Get windows.h included everywhere (we need it) */
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define GRPC_WIN32_LEAN_AND_MEAN_WAS_NOT_DEFINED
#define WIN32_LEAN_AND_MEAN
#endif /* WIN32_LEAN_AND_MEAN */

#ifndef NOMINMAX
#define GRPC_NOMINMX_WAS_NOT_DEFINED
#define NOMINMAX
#endif /* NOMINMAX */

#ifndef _WIN32_WINNT
#error \
    "Please compile grpc with _WIN32_WINNT of at least 0x600 (aka Windows Vista)"
#else /* !defined(_WIN32_WINNT) */
#if (_WIN32_WINNT < 0x0600)
#error \
    "Please compile grpc with _WIN32_WINNT of at least 0x600 (aka Windows Vista)"
#endif /* _WIN32_WINNT < 0x0600 */
#endif /* defined(_WIN32_WINNT) */

#include <windows.h>

#ifdef GRPC_WIN32_LEAN_AND_MEAN_WAS_NOT_DEFINED
#undef GRPC_WIN32_LEAN_AND_MEAN_WAS_NOT_DEFINED
#undef WIN32_LEAN_AND_MEAN
#endif /* GRPC_WIN32_LEAN_AND_MEAN_WAS_NOT_DEFINED */

#ifdef GRPC_NOMINMAX_WAS_NOT_DEFINED
#undef GRPC_NOMINMAX_WAS_NOT_DEFINED
#undef NOMINMAX
#endif /* GRPC_WIN32_LEAN_AND_MEAN_WAS_NOT_DEFINED */
#endif /* defined(_WIN64) || defined(WIN64) || defined(_WIN32) || \
          defined(WIN32) */

/* Override this file with one for your platform if you need to redefine
   things.  */

#if !defined(GPR_NO_AUTODETECT_PLATFORM)
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#if defined(_WIN64) || defined(WIN64)
#define GPR_ARCH_64 1
#else
#define GPR_ARCH_32 1
#endif
#define GPR_PLATFORM_STRING "windows"
#define GPR_WINDOWS 1
#define GPR_WINDOWS_SUBPROCESS 1
#define GPR_WINDOWS_ENV
#ifdef __MSYS__
#define GPR_GETPID_IN_UNISTD_H 1
#define GPR_MSYS_TMPFILE
#define GPR_POSIX_LOG
#define GPR_POSIX_STRING
#define GPR_POSIX_TIME
#else
#define GPR_GETPID_IN_PROCESS_H 1
#define GPR_WINDOWS_TMPFILE
#define GPR_WINDOWS_LOG
#define GPR_WINDOWS_CRASH_HANDLER 1
#define GPR_WINDOWS_STRING
#define GPR_WINDOWS_TIME
#endif
#ifdef __GNUC__
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#else
#define GPR_WINDOWS_ATOMIC 1
#define GPR_MSVC_TLS 1
#endif
#elif defined(GPR_MANYLINUX1)
// TODO(atash): manylinux1 is just another __linux__ but with ancient
// libraries; it should be integrated with the `__linux__` definitions below.
#define GPR_PLATFORM_STRING "manylinux"
#define GPR_POSIX_CRASH_HANDLER 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#define GPR_LINUX 1
#define GPR_LINUX_LOG 1
#define GPR_SUPPORT_CHANNELS_FROM_FD 1
#define GPR_LINUX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#include <linux/version.h>
#elif defined(ANDROID) || defined(__ANDROID__)
#define GPR_PLATFORM_STRING "android"
#define GPR_ANDROID 1
// TODO(apolcyn): re-evaluate support for c-ares
// on android after upgrading our c-ares dependency.
// See https://github.com/grpc/grpc/issues/18038.
#define GRPC_ARES 0
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#define GPR_CPU_POSIX 1
#define GPR_GCC_SYNC 1
#define GPR_GCC_TLS 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_ANDROID_LOG 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#define GPR_SUPPORT_CHANNELS_FROM_FD 1
#elif defined(__linux__)
#define GPR_PLATFORM_STRING "linux"
#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <features.h>
#define GPR_CPU_LINUX 1
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#define GPR_LINUX 1
#define GPR_LINUX_LOG
#define GPR_SUPPORT_CHANNELS_FROM_FD 1
#define GPR_LINUX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#ifdef __GLIBC__
#define GPR_POSIX_CRASH_HANDLER 1
#define GPR_LINUX_PTHREAD_NAME 1
#include <linux/version.h>
#else /* musl libc */
#define GPR_MUSL_LIBC_COMPAT 1
#endif
#elif defined(__ASYLO__)
#define GPR_ARCH_64 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_TLS 1
#define GPR_PLATFORM_STRING "asylo"
#define GPR_GCC_SYNC 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_TIME 1
#define GPR_POSIX_ENV 1
#define GPR_ASYLO 1
#define GRPC_POSIX_SOCKET 1
#define GRPC_POSIX_SOCKETADDR
#define GRPC_POSIX_SOCKETUTILS 1
#define GRPC_TIMER_USE_GENERIC 1
#define GRPC_POSIX_NO_SPECIAL_WAKEUP_FD 1
#define GRPC_POSIX_WAKEUP_FD 1
#define GRPC_ARES 0
#define GPR_NO_AUTODETECT_PLATFORM 1
#elif defined(__APPLE__)
#include <Availability.h>
#include <TargetConditionals.h>
#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif
#if TARGET_OS_IPHONE
#define GPR_PLATFORM_STRING "ios"
#define GPR_CPU_IPHONE 1
#define GPR_PTHREAD_TLS 1
#define GRPC_CFSTREAM 1
/* the c-ares resolver isn't safe to enable on iOS */
#define GRPC_ARES 0
#else /* TARGET_OS_IPHONE */
#define GPR_PLATFORM_STRING "osx"
#ifdef __MAC_OS_X_VERSION_MIN_REQUIRED
#if __MAC_OS_X_VERSION_MIN_REQUIRED < __MAC_10_7
#define GPR_CPU_IPHONE 1
#define GPR_PTHREAD_TLS 1
#else /* __MAC_OS_X_VERSION_MIN_REQUIRED < __MAC_10_7 */
#define GPR_CPU_POSIX 1
/* TODO(vjpai): there is a reported issue in bazel build for Mac where __thread
   in a header is currently not working (bazelbuild/bazel#4341). Remove
   the following conditional and use GPR_GCC_TLS when that is fixed */
#ifndef GRPC_BAZEL_BUILD
#define GPR_GCC_TLS 1
#else /* GRPC_BAZEL_BUILD */
#define GPR_PTHREAD_TLS 1
#endif /* GRPC_BAZEL_BUILD */
#define GPR_APPLE_PTHREAD_NAME 1
#endif
#else /* __MAC_OS_X_VERSION_MIN_REQUIRED */
#define GPR_CPU_POSIX 1
/* TODO(vjpai): Remove the following conditional and use only GPR_GCC_TLS
   when bazelbuild/bazel#4341 is fixed */
#ifndef GRPC_BAZEL_BUILD
#define GPR_GCC_TLS 1
#else /* GRPC_BAZEL_BUILD */
#define GPR_PTHREAD_TLS 1
#endif /* GRPC_BAZEL_BUILD */
#endif
#define GPR_POSIX_CRASH_HANDLER 1
#endif
#define GPR_APPLE 1
#define GPR_GCC_ATOMIC 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#ifndef GRPC_CFSTREAM
#define GPR_SUPPORT_CHANNELS_FROM_FD 1
#endif
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#elif defined(__FreeBSD__)
#define GPR_PLATFORM_STRING "freebsd"
#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif
#define GPR_FREEBSD 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#define GPR_SUPPORT_CHANNELS_FROM_FD 1
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#elif defined(__OpenBSD__)
#define GPR_PLATFORM_STRING "openbsd"
#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif
#define GPR_OPENBSD 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#define GPR_SUPPORT_CHANNELS_FROM_FD 1
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#elif defined(__sun) && defined(__SVR4)
#define GPR_PLATFORM_STRING "solaris"
#define GPR_SOLARIS 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#elif defined(_AIX)
#define GPR_PLATFORM_STRING "aix"
#ifndef _ALL_SOURCE
#define _ALL_SOURCE
#endif
#define GPR_AIX 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#elif defined(__native_client__)
#define GPR_PLATFORM_STRING "nacl"
#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GPR_NACL 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_ATOMIC 1
#define GPR_GCC_TLS 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#ifdef _LP64
#define GPR_ARCH_64 1
#else /* _LP64 */
#define GPR_ARCH_32 1
#endif /* _LP64 */
#elif defined(__Fuchsia__)
#define GPR_FUCHSIA 1
#define GPR_ARCH_64 1
#define GPR_PLATFORM_STRING "fuchsia"
#include <features.h>
// Specifying musl libc affects wrap_memcpy.c. It causes memmove() to be
// invoked.
#define GPR_MUSL_LIBC_COMPAT 1
#define GPR_CPU_POSIX 1
#define GPR_GCC_ATOMIC 1
#define GPR_PTHREAD_TLS 1
#define GPR_POSIX_LOG 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_ENV 1
#define GPR_POSIX_TMPFILE 1
#define GPR_POSIX_SUBPROCESS 1
#define GPR_POSIX_SYNC 1
#define GPR_POSIX_STRING 1
#define GPR_POSIX_TIME 1
#define GPR_HAS_PTHREAD_H 1
#define GPR_GETPID_IN_UNISTD_H 1
#else
#error "Could not auto-detect platform"
#endif
#endif /* GPR_NO_AUTODETECT_PLATFORM */

#if defined(GPR_BACKWARDS_COMPATIBILITY_MODE)
/*
 * For backward compatibility mode, reset _FORTIFY_SOURCE to prevent
 * a library from having non-standard symbols such as __asprintf_chk.
 * This helps non-glibc systems such as alpine using musl to find symbols.
 */
#if defined(_FORTIFY_SOURCE) && _FORTIFY_SOURCE > 0
#undef _FORTIFY_SOURCE
#define _FORTIFY_SOURCE 0
#endif
#endif

/*
 *  There are platforms for which TLS should not be used even though the
 * compiler makes it seem like it's supported (Android NDK < r12b for example).
 * This is primarily because of linker problems and toolchain misconfiguration:
 * TLS isn't supported until NDK r12b per
 * https://developer.android.com/ndk/downloads/revision_history.html
 * TLS also does not work with Android NDK if GCC is being used as the compiler
 * instead of Clang.
 * Since NDK r16, `__NDK_MAJOR__` and `__NDK_MINOR__` are defined in
 * <android/ndk-version.h>. For NDK < r16, users should define these macros,
 * e.g. `-D__NDK_MAJOR__=11 -D__NKD_MINOR__=0` for NDK r11. */
#if defined(__ANDROID__) && defined(GPR_GCC_TLS)
#if __has_include(<android/ndk-version.h>)
#include <android/ndk-version.h>
#endif /* __has_include(<android/ndk-version.h>) */
#if (defined(__clang__) && defined(__NDK_MAJOR__) && defined(__NDK_MINOR__) && \
     ((__NDK_MAJOR__ < 12) ||                                                  \
      ((__NDK_MAJOR__ == 12) && (__NDK_MINOR__ < 1)))) ||                      \
    (defined(__GNUC__) && !defined(__clang__))
#undef GPR_GCC_TLS
#define GPR_PTHREAD_TLS 1
#endif
#endif /*defined(__ANDROID__) && defined(GPR_GCC_TLS) */

#if defined(__has_include)
#if __has_include(<atomic>)
#define GRPC_HAS_CXX11_ATOMIC
#endif /* __has_include(<atomic>) */
#endif /* defined(__has_include) */

#ifndef GPR_PLATFORM_STRING
#warning "GPR_PLATFORM_STRING not auto-detected"
#define GPR_PLATFORM_STRING "unknown"
#endif

#ifdef GPR_GCOV
#undef GPR_FORBID_UNREACHABLE_CODE
#define GPR_FORBID_UNREACHABLE_CODE 1
#endif

#ifdef _MSC_VER
#if _MSC_VER < 1700
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif /* _MSC_VER < 1700 */
#else
#include <stdint.h>
#endif /* _MSC_VER */

/* Type of cycle clock implementation */
#ifdef GPR_LINUX
/* Disable cycle clock by default.
   TODO(soheil): enable when we support fallback for unstable cycle clocks.
#if defined(__i386__)
#define GPR_CYCLE_COUNTER_RDTSC_32 1
#elif defined(__x86_64__) || defined(__amd64__)
#define GPR_CYCLE_COUNTER_RDTSC_64 1
#else
#define GPR_CYCLE_COUNTER_FALLBACK 1
#endif
*/
#define GPR_CYCLE_COUNTER_FALLBACK 1
#else
#define GPR_CYCLE_COUNTER_FALLBACK 1
#endif /* GPR_LINUX */

/* Cache line alignment */
#ifndef GPR_CACHELINE_SIZE_LOG
#if defined(__i386__) || defined(__x86_64__)
#define GPR_CACHELINE_SIZE_LOG 6
#endif
#ifndef GPR_CACHELINE_SIZE_LOG
/* A reasonable default guess. Note that overestimates tend to waste more
   space, while underestimates tend to waste more time. */
#define GPR_CACHELINE_SIZE_LOG 6
#endif /* GPR_CACHELINE_SIZE_LOG */
#endif /* GPR_CACHELINE_SIZE_LOG */

#define GPR_CACHELINE_SIZE (1 << GPR_CACHELINE_SIZE_LOG)

/* scrub GCC_ATOMIC if it's not available on this compiler */
#if defined(GPR_GCC_ATOMIC) && !defined(__ATOMIC_RELAXED)
#undef GPR_GCC_ATOMIC
#define GPR_GCC_SYNC 1
#endif

/* Validate platform combinations */
#if defined(GPR_GCC_ATOMIC) + defined(GPR_GCC_SYNC) + \
        defined(GPR_WINDOWS_ATOMIC) !=                \
    1
#error Must define exactly one of GPR_GCC_ATOMIC, GPR_GCC_SYNC, GPR_WINDOWS_ATOMIC
#endif

#if defined(GPR_ARCH_32) + defined(GPR_ARCH_64) != 1
#error Must define exactly one of GPR_ARCH_32, GPR_ARCH_64
#endif

#if defined(GPR_CPU_LINUX) + defined(GPR_CPU_POSIX) + defined(GPR_WINDOWS) + \
        defined(GPR_CPU_IPHONE) + defined(GPR_CPU_CUSTOM) !=                 \
    1
#error Must define exactly one of GPR_CPU_LINUX, GPR_CPU_POSIX, GPR_WINDOWS, GPR_CPU_IPHONE, GPR_CPU_CUSTOM
#endif

#if defined(GPR_MSVC_TLS) + defined(GPR_GCC_TLS) + defined(GPR_PTHREAD_TLS) + \
        defined(GPR_CUSTOM_TLS) !=                                            \
    1
#error Must define exactly one of GPR_MSVC_TLS, GPR_GCC_TLS, GPR_PTHREAD_TLS, GPR_CUSTOM_TLS
#endif

/* maximum alignment needed for any type on this platform, rounded up to a
   power of two */
#define GPR_MAX_ALIGNMENT 16

#ifndef GRPC_ARES
#define GRPC_ARES 1
#endif

#ifndef GRPC_IF_NAMETOINDEX
#define GRPC_IF_NAMETOINDEX 1
#endif

#ifndef GRPC_MUST_USE_RESULT
#if defined(__GNUC__) && !defined(__MINGW32__)
#define GRPC_MUST_USE_RESULT __attribute__((warn_unused_result))
#define GPR_ALIGN_STRUCT(n) __attribute__((aligned(n)))
#else
#define GRPC_MUST_USE_RESULT
#define GPR_ALIGN_STRUCT(n)
#endif
#endif

#ifndef GRPC_UNUSED
#if defined(__GNUC__) && !defined(__MINGW32__)
#define GRPC_UNUSED __attribute__((unused))
#else
#define GRPC_UNUSED
#endif
#endif

#ifndef GPR_PRINT_FORMAT_CHECK
#ifdef __GNUC__
#define GPR_PRINT_FORMAT_CHECK(FORMAT_STR, ARGS) \
  __attribute__((format(printf, FORMAT_STR, ARGS)))
#else
#define GPR_PRINT_FORMAT_CHECK(FORMAT_STR, ARGS)
#endif
#endif /* GPR_PRINT_FORMAT_CHECK */

#if GPR_FORBID_UNREACHABLE_CODE
#define GPR_UNREACHABLE_CODE(STATEMENT)
#else
#define GPR_UNREACHABLE_CODE(STATEMENT)             \
  do {                                              \
    gpr_log(GPR_ERROR, "Should never reach here."); \
    abort();                                        \
    STATEMENT;                                      \
  } while (0)
#endif /* GPR_FORBID_UNREACHABLE_CODE */

#ifndef GPRAPI
#define GPRAPI
#endif

#ifndef GRPCAPI
#define GRPCAPI GPRAPI
#endif

#ifndef CENSUSAPI
#define CENSUSAPI GRPCAPI
#endif

#ifndef GPR_HAS_ATTRIBUTE
#ifdef __has_attribute
#define GPR_HAS_ATTRIBUTE(a) __has_attribute(a)
#else
#define GPR_HAS_ATTRIBUTE(a) 0
#endif
#endif /* GPR_HAS_ATTRIBUTE */

#ifndef GPR_HAS_FEATURE
#ifdef __has_feature
#define GPR_HAS_FEATURE(a) __has_feature(a)
#else
#define GPR_HAS_FEATURE(a) 0
#endif
#endif /* GPR_HAS_FEATURE */

#ifndef GPR_ATTRIBUTE_NOINLINE
#if GPR_HAS_ATTRIBUTE(noinline) || (defined(__GNUC__) && !defined(__clang__))
#define GPR_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define GPR_HAS_ATTRIBUTE_NOINLINE 1
#else
#define GPR_ATTRIBUTE_NOINLINE
#endif
#endif /* GPR_ATTRIBUTE_NOINLINE */

#ifndef GPR_ATTRIBUTE_WEAK
/* Attribute weak is broken on LLVM/windows:
 * https://bugs.llvm.org/show_bug.cgi?id=37598 */
#if (GPR_HAS_ATTRIBUTE(weak) || (defined(__GNUC__) && !defined(__clang__))) && \
    !(defined(__llvm__) && defined(_WIN32))
#define GPR_ATTRIBUTE_WEAK __attribute__((weak))
#define GPR_HAS_ATTRIBUTE_WEAK 1
#else
#define GPR_ATTRIBUTE_WEAK
#endif
#endif /* GPR_ATTRIBUTE_WEAK */

#ifndef GPR_ATTRIBUTE_NO_TSAN /* (1) */
#if GPR_HAS_FEATURE(thread_sanitizer)
#define GPR_ATTRIBUTE_NO_TSAN __attribute__((no_sanitize("thread")))
#endif                        /* GPR_HAS_FEATURE */
#ifndef GPR_ATTRIBUTE_NO_TSAN /* (2) */
#define GPR_ATTRIBUTE_NO_TSAN
#endif /* GPR_ATTRIBUTE_NO_TSAN (2) */
#endif /* GPR_ATTRIBUTE_NO_TSAN (1) */

/* GRPC_TSAN_ENABLED will be defined, when compiled with thread sanitizer. */
#if defined(__SANITIZE_THREAD__)
#define GRPC_TSAN_ENABLED
#elif GPR_HAS_FEATURE(thread_sanitizer)
#define GRPC_TSAN_ENABLED
#endif

/* GRPC_ASAN_ENABLED will be defined, when compiled with address sanitizer. */
#if defined(__SANITIZE_ADDRESS__)
#define GRPC_ASAN_ENABLED
#elif GPR_HAS_FEATURE(address_sanitizer)
#define GRPC_ASAN_ENABLED
#endif

/* GRPC_ALLOW_EXCEPTIONS should be 0 or 1 if exceptions are allowed or not */
#ifndef GRPC_ALLOW_EXCEPTIONS
#ifdef GPR_WINDOWS
#if defined(_MSC_VER) && defined(_CPPUNWIND)
#define GRPC_ALLOW_EXCEPTIONS 1
#elif defined(__EXCEPTIONS)
#define GRPC_ALLOW_EXCEPTIONS 1
#else
#define GRPC_ALLOW_EXCEPTIONS 0
#endif
#else /* GPR_WINDOWS */
#ifdef __EXCEPTIONS
#define GRPC_ALLOW_EXCEPTIONS 1
#else /* __EXCEPTIONS */
#define GRPC_ALLOW_EXCEPTIONS 0
#endif /* __EXCEPTIONS */
#endif /* __GPR_WINDOWS */
#endif /* GRPC_ALLOW_EXCEPTIONS */

/* Use GPR_LIKELY only in cases where you are sure that a certain outcome is the
 * most likely. Ideally, also collect performance numbers to justify the claim.
 */
#ifdef __GNUC__
#define GPR_LIKELY(x) __builtin_expect((x), 1)
#define GPR_UNLIKELY(x) __builtin_expect((x), 0)
#else /* __GNUC__ */
#define GPR_LIKELY(x) (x)
#define GPR_UNLIKELY(x) (x)
#endif /* __GNUC__ */

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#endif /* GRPC_IMPL_CODEGEN_PORT_PLATFORM_H */
