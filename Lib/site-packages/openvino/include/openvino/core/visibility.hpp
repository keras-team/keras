// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// https://gcc.gnu.org/wiki/Visibility
// Generic helper definitions for shared library support

#ifndef OPENVINO_EXTERN_C
#    ifdef __cplusplus
#        define OPENVINO_EXTERN_C extern "C"
#    else
#        define OPENVINO_EXTERN_C
#    endif
#endif

#if defined _WIN32
#    define OPENVINO_CDECL   __cdecl
#    define OPENVINO_STDCALL __stdcall
#else
#    define OPENVINO_CDECL
#    define OPENVINO_STDCALL
#endif

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined(__INTEL_COMPILER) || defined(_MSC_VER) || defined(__GNUC__)
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2))
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#    define OPENVINO_CORE_IMPORTS __declspec(dllimport)
#    define OPENVINO_CORE_EXPORTS __declspec(dllexport)
#    define _OPENVINO_HIDDEN_METHOD
#elif defined(__GNUC__) && (__GNUC__ >= 4) || defined(__clang__)
#    define OPENVINO_CORE_IMPORTS   __attribute__((visibility("default")))
#    define OPENVINO_CORE_EXPORTS   __attribute__((visibility("default")))
#    define _OPENVINO_HIDDEN_METHOD __attribute__((visibility("hidden")))
#else
#    define OPENVINO_CORE_IMPORTS
#    define OPENVINO_CORE_EXPORTS
#    define _OPENVINO_HIDDEN_METHOD
#endif

// see https://sourceforge.net/p/predef/wiki/Architectures/
#if defined(__arm__) || defined(_M_ARM) || defined(__ARMEL__)
#    define OPENVINO_ARCH_ARM
#    define OPENVINO_ARCH_32_BIT
#elif defined(__aarch64__) || defined(_M_ARM64)
#    define OPENVINO_ARCH_ARM64
#    define OPENVINO_ARCH_64_BIT
#elif defined(i386) || defined(__i386) || defined(__i386__) || defined(__IA32__) || defined(_M_I86) || \
    defined(_M_IX86) || defined(__X86__) || defined(_X86_) || defined(__I86__) || defined(__386) ||    \
    defined(__ILP32__) || defined(_ILP32) || defined(__wasm32__) || defined(__wasm32)
#    define OPENVINO_ARCH_X86
#    define OPENVINO_ARCH_32_BIT
#elif defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || \
    defined(_M_AMD64)
#    define OPENVINO_ARCH_X86_64
#    define OPENVINO_ARCH_64_BIT
#elif defined(__riscv)
#    define OPENVINO_ARCH_RISCV64
#    define OPENVINO_ARCH_64_BIT
#endif

/**
 * @brief Define no dangling attribute.
 * Use it as C++ attribute e.g. OV_NO_DANGLING void my_func();
 */
#if defined(__GNUC__) && (__GNUC__ >= 14)
#    define OV_NO_DANGLING [[gnu::no_dangling]]
#else
#    define OV_NO_DANGLING
#endif

#if !(defined(_MSC_VER) && __cplusplus == 199711L)
#    if __cplusplus >= 201103L
#        define OPENVINO_CPP_VER_AT_LEAST_11
#        if __cplusplus >= 201402L
#            define OPENVINO_CPP_VER_AT_LEAST_14
#            if __cplusplus >= 201703L
#                define OPENVINO_CPP_VER_AT_LEAST_17
#                if __cplusplus >= 202002L
#                    define OPENVINO_CPP_VER_AT_LEAST_20
#                endif
#            endif
#        endif
#    endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
#    if _MSVC_LANG >= 201103L
#        define OPENVINO_CPP_VER_AT_LEAST_11
#        if _MSVC_LANG >= 201402L
#            define OPENVINO_CPP_VER_AT_LEAST_14
#            if _MSVC_LANG >= 201703L
#                define OPENVINO_CPP_VER_AT_LEAST_17
#                if _MSVC_LANG >= 202002L
#                    define OPENVINO_CPP_VER_AT_LEAST_20
#                endif
#            endif
#        endif
#    endif
#endif
