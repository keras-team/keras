/*
    Copyright (c) 2005-2021 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_detail__config_H
#define __TBB_detail__config_H

/** This header is supposed to contain macro definitions only.
    The macros defined here are intended to control such aspects of TBB build as
    - presence of compiler features
    - compilation modes
    - feature sets
    - known compiler/platform issues
**/

/* Check which standard library we use. */
#include <cstddef>

#if _MSC_VER
    #define __TBB_EXPORTED_FUNC   __cdecl
    #define __TBB_EXPORTED_METHOD __thiscall
#else
    #define __TBB_EXPORTED_FUNC
    #define __TBB_EXPORTED_METHOD
#endif

#if defined(_MSVC_LANG)
    #define __TBB_LANG _MSVC_LANG
#else
    #define __TBB_LANG __cplusplus
#endif // _MSVC_LANG

#define __TBB_CPP14_PRESENT (__TBB_LANG >= 201402L)
#define __TBB_CPP17_PRESENT (__TBB_LANG >= 201703L)
#define __TBB_CPP20_PRESENT (__TBB_LANG >= 201709L)

#if __INTEL_COMPILER || _MSC_VER
    #define __TBB_NOINLINE(decl) __declspec(noinline) decl
#elif __GNUC__
    #define __TBB_NOINLINE(decl) decl __attribute__ ((noinline))
#else
    #define __TBB_NOINLINE(decl) decl
#endif

#define __TBB_STRING_AUX(x) #x
#define __TBB_STRING(x) __TBB_STRING_AUX(x)

// Note that when ICC or Clang is in use, __TBB_GCC_VERSION might not fully match
// the actual GCC version on the system.
#define __TBB_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

/* Check which standard library we use. */

// Prior to GCC 7, GNU libstdc++ did not have a convenient version macro.
// Therefore we use different ways to detect its version.
#ifdef TBB_USE_GLIBCXX_VERSION
    // The version is explicitly specified in our public TBB_USE_GLIBCXX_VERSION macro.
    // Its format should match the __TBB_GCC_VERSION above, e.g. 70301 for libstdc++ coming with GCC 7.3.1.
    #define __TBB_GLIBCXX_VERSION TBB_USE_GLIBCXX_VERSION
#elif _GLIBCXX_RELEASE && _GLIBCXX_RELEASE != __GNUC__
    // Reported versions of GCC and libstdc++ do not match; trust the latter
    #define __TBB_GLIBCXX_VERSION (_GLIBCXX_RELEASE*10000)
#elif __GLIBCPP__ || __GLIBCXX__
    // The version macro is not defined or matches the GCC version; use __TBB_GCC_VERSION
    #define __TBB_GLIBCXX_VERSION __TBB_GCC_VERSION
#endif

#if __clang__
    // according to clang documentation, version can be vendor specific
    #define __TBB_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#endif

/** Macro helpers **/

#define __TBB_CONCAT_AUX(A,B) A##B
// The additional level of indirection is needed to expand macros A and B (not to get the AB macro).
// See [cpp.subst] and [cpp.concat] for more details.
#define __TBB_CONCAT(A,B) __TBB_CONCAT_AUX(A,B)
// The IGNORED argument and comma are needed to always have 2 arguments (even when A is empty).
#define __TBB_IS_MACRO_EMPTY(A,IGNORED) __TBB_CONCAT_AUX(__TBB_MACRO_EMPTY,A)
#define __TBB_MACRO_EMPTY 1

#if _M_X64
    #define __TBB_W(name) name##64
#else
    #define __TBB_W(name) name
#endif

/** User controlled TBB features & modes **/

#ifndef TBB_USE_DEBUG
    /*
    There are four cases that are supported:
    1. "_DEBUG is undefined" means "no debug";
    2. "_DEBUG defined to something that is evaluated to 0" (including "garbage", as per [cpp.cond]) means "no debug";
    3. "_DEBUG defined to something that is evaluated to a non-zero value" means "debug";
    4. "_DEBUG defined to nothing (empty)" means "debug".
    */
    #ifdef _DEBUG
        // Check if _DEBUG is empty.
        #define __TBB_IS__DEBUG_EMPTY (__TBB_IS_MACRO_EMPTY(_DEBUG,IGNORED)==__TBB_MACRO_EMPTY)
        #if __TBB_IS__DEBUG_EMPTY
            #define TBB_USE_DEBUG 1
        #else
            #define TBB_USE_DEBUG _DEBUG
        #endif // __TBB_IS__DEBUG_EMPTY
    #else
        #define TBB_USE_DEBUG 0
    #endif // _DEBUG
#endif // TBB_USE_DEBUG

#ifndef TBB_USE_ASSERT
    #define TBB_USE_ASSERT TBB_USE_DEBUG
#endif // TBB_USE_ASSERT

#ifndef TBB_USE_PROFILING_TOOLS
#if TBB_USE_DEBUG
    #define TBB_USE_PROFILING_TOOLS 2
#else // TBB_USE_DEBUG
    #define TBB_USE_PROFILING_TOOLS 0
#endif // TBB_USE_DEBUG
#endif // TBB_USE_PROFILING_TOOLS

// Exceptions support cases
#if !(__EXCEPTIONS || defined(_CPPUNWIND) || __SUNPRO_CC)
    #if TBB_USE_EXCEPTIONS
        #error Compilation settings do not support exception handling. Please do not set TBB_USE_EXCEPTIONS macro or set it to 0.
    #elif !defined(TBB_USE_EXCEPTIONS)
        #define TBB_USE_EXCEPTIONS 0
    #endif
#elif !defined(TBB_USE_EXCEPTIONS)
    #define TBB_USE_EXCEPTIONS 1
#endif

/** Preprocessor symbols to determine HW architecture **/

#if _WIN32 || _WIN64
    #if defined(_M_X64) || defined(__x86_64__)  // the latter for MinGW support
        #define __TBB_x86_64 1
    #elif defined(_M_IA64)
        #define __TBB_ipf 1
    #elif defined(_M_IX86) || defined(__i386__) // the latter for MinGW support
        #define __TBB_x86_32 1
    #else
        #define __TBB_generic_arch 1
    #endif
#else /* Assume generic Unix */
    #if __x86_64__
        #define __TBB_x86_64 1
    #elif __ia64__
        #define __TBB_ipf 1
    #elif __i386__||__i386  // __i386 is for Sun OS
        #define __TBB_x86_32 1
    #else
        #define __TBB_generic_arch 1
    #endif
#endif

/** Windows API or POSIX API **/

#if _WIN32 || _WIN64
    #define __TBB_USE_WINAPI 1
#else
    #define __TBB_USE_POSIX 1
#endif

/** Internal TBB features & modes **/

/** __TBB_DYNAMIC_LOAD_ENABLED describes the system possibility to load shared libraries at run time **/
#ifndef __TBB_DYNAMIC_LOAD_ENABLED
    #define __TBB_DYNAMIC_LOAD_ENABLED 1
#endif

/** __TBB_WIN8UI_SUPPORT enables support of Windows* Store Apps and limit a possibility to load
    shared libraries at run time only from application container **/
#if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_FAMILY_APP
    #define __TBB_WIN8UI_SUPPORT 1
#else
    #define __TBB_WIN8UI_SUPPORT 0
#endif

/** __TBB_WEAK_SYMBOLS_PRESENT denotes that the system supports the weak symbol mechanism **/
#ifndef __TBB_WEAK_SYMBOLS_PRESENT
    #define __TBB_WEAK_SYMBOLS_PRESENT ( !_WIN32 && !__APPLE__ && !__sun && (__TBB_GCC_VERSION >= 40000 || __INTEL_COMPILER ) )
#endif

/** Presence of compiler features **/

#if __clang__ && !__INTEL_COMPILER
    #define __TBB_USE_OPTIONAL_RTTI __has_feature(cxx_rtti)
#elif defined(_CPPRTTI)
    #define __TBB_USE_OPTIONAL_RTTI 1
#else
    #define __TBB_USE_OPTIONAL_RTTI (__GXX_RTTI || __RTTI || __INTEL_RTTI__)
#endif

/** Library features presence macros **/

#define __TBB_CPP14_INTEGER_SEQUENCE_PRESENT       (__TBB_LANG >= 201402L)
#define __TBB_CPP17_INVOKE_RESULT_PRESENT          (__TBB_LANG >= 201703L)

// TODO: Remove the condition(__INTEL_COMPILER > 2021) from the __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
// macro when this feature start working correctly on this compiler.
#if __INTEL_COMPILER && (!_MSC_VER || __INTEL_CXX11_MOVE__)
    #define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT (__TBB_LANG >= 201402L)
    #define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT   (__INTEL_COMPILER > 2021 && __TBB_LANG >= 201703L)
    #define __TBB_CPP20_CONCEPTS_PRESENT           0 // TODO: add a mechanism for future addition
#elif __clang__
    #define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT (__has_feature(cxx_variable_templates))
    #define __TBB_CPP20_CONCEPTS_PRESENT           0 // TODO: add a mechanism for future addition
    #ifdef __cpp_deduction_guides
        #define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT (__cpp_deduction_guides >= 201611L)
    #else
        #define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT 0
    #endif
#elif __GNUC__
    #define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT (__TBB_LANG >= 201402L && __TBB_GCC_VERSION >= 50000)
    #define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT   (__cpp_deduction_guides >= 201606L)
    #define __TBB_CPP20_CONCEPTS_PRESENT           (__TBB_LANG >= 201709L && __TBB_GCC_VERSION >= 100201)
#elif _MSC_VER
    #define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT (_MSC_FULL_VER >= 190023918 && (!__INTEL_COMPILER || __INTEL_COMPILER >= 1700))
    #define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT   (_MSC_VER >= 1914 && __TBB_LANG >= 201703L && (!__INTEL_COMPILER || __INTEL_COMPILER > 2021))
    #define __TBB_CPP20_CONCEPTS_PRESENT           (_MSC_VER >= 1923 && __TBB_LANG >= 202002L) // TODO: INTEL_COMPILER?
#else
    #define __TBB_CPP14_VARIABLE_TEMPLATES_PRESENT (__TBB_LANG >= 201402L)
    #define __TBB_CPP17_DEDUCTION_GUIDES_PRESENT   (__TBB_LANG >= 201703L)
    #define __TBB_CPP20_CONCEPTS_PRESENT           (__TBB_LANG >= 202002L)
#endif

// GCC4.8 on RHEL7 does not support std::get_new_handler
#define __TBB_CPP11_GET_NEW_HANDLER_PRESENT             (_MSC_VER >= 1900 || __TBB_GLIBCXX_VERSION >= 40900 && __GXX_EXPERIMENTAL_CXX0X__ || _LIBCPP_VERSION)
// GCC4.8 on RHEL7 does not support std::is_trivially_copyable
#define __TBB_CPP11_TYPE_PROPERTIES_PRESENT             (_LIBCPP_VERSION || _MSC_VER >= 1700 || (__TBB_GLIBCXX_VERSION >= 50000 && __GXX_EXPERIMENTAL_CXX0X__))

#define __TBB_CPP17_MEMORY_RESOURCE_PRESENT             (_MSC_VER >= 1913 && (__TBB_LANG > 201402L) || \
                                                        __TBB_GLIBCXX_VERSION >= 90000 && __TBB_LANG >= 201703L)
#define __TBB_CPP17_HW_INTERFERENCE_SIZE_PRESENT        (_MSC_VER >= 1911)
#define __TBB_CPP17_LOGICAL_OPERATIONS_PRESENT          (__TBB_LANG >= 201703L)
#define __TBB_CPP17_ALLOCATOR_IS_ALWAYS_EQUAL_PRESENT   (__TBB_LANG >= 201703L)
#define __TBB_CPP17_IS_SWAPPABLE_PRESENT                (__TBB_LANG >= 201703L)
#define __TBB_CPP20_COMPARISONS_PRESENT                 __TBB_CPP20_PRESENT

#define __TBB_RESUMABLE_TASKS                           (!__TBB_WIN8UI_SUPPORT && !__ANDROID__)

/* This macro marks incomplete code or comments describing ideas which are considered for the future.
 * See also for plain comment with TODO and FIXME marks for small improvement opportunities.
 */
#define __TBB_TODO 0

/* Check which standard library we use. */
/* __TBB_SYMBOL is defined only while processing exported symbols list where C++ is not allowed. */
#if !defined(__TBB_SYMBOL) && !__TBB_CONFIG_PREPROC_ONLY
    #include <cstddef>
#endif

/** Target OS is either iOS* or iOS* simulator **/
#if __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
    #define __TBB_IOS 1
#endif

#if __APPLE__
    #if __INTEL_COMPILER && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ > 1099 \
                         && __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101000
        // ICC does not correctly set the macro if -mmacosx-min-version is not specified
        #define __TBB_MACOS_TARGET_VERSION  (100000 + 10*(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ - 1000))
    #else
        #define __TBB_MACOS_TARGET_VERSION  __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
    #endif
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define __TBB_GCC_WARNING_IGNORED_ATTRIBUTES_PRESENT (__TBB_GCC_VERSION >= 60100)
#endif

#define __TBB_CPP17_FALLTHROUGH_PRESENT (__TBB_LANG >= 201703L)
#define __TBB_CPP17_NODISCARD_PRESENT   (__TBB_LANG >= 201703L)
#define __TBB_FALLTHROUGH_PRESENT       (__TBB_GCC_VERSION >= 70000 && !__INTEL_COMPILER)

#if __TBB_CPP17_FALLTHROUGH_PRESENT
    #define __TBB_fallthrough [[fallthrough]]
#elif __TBB_FALLTHROUGH_PRESENT
    #define __TBB_fallthrough __attribute__ ((fallthrough))
#else
    #define __TBB_fallthrough
#endif

#if __TBB_CPP17_NODISCARD_PRESENT
    #define __TBB_nodiscard [[nodiscard]]
#elif __clang__ || __GNUC__
    #define __TBB_nodiscard __attribute__((warn_unused_result))
#else
    #define __TBB_nodiscard
#endif

#define __TBB_CPP17_UNCAUGHT_EXCEPTIONS_PRESENT             (_MSC_VER >= 1900 || __GLIBCXX__ && __cpp_lib_uncaught_exceptions \
                                                            || _LIBCPP_VERSION >= 3700 && (!__TBB_MACOS_TARGET_VERSION || __TBB_MACOS_TARGET_VERSION >= 101200))


#define __TBB_TSX_INTRINSICS_PRESENT ((__RTM__ || _MSC_VER>=1700 || __INTEL_COMPILER) && !__ANDROID__)

#define __TBB_WAITPKG_INTRINSICS_PRESENT ((__INTEL_COMPILER >= 1900 || __TBB_GCC_VERSION >= 110000 || __TBB_CLANG_VERSION >= 120000) && !__ANDROID__)

/** Internal TBB features & modes **/

/** __TBB_SOURCE_DIRECTLY_INCLUDED is a mode used in whitebox testing when
    it's necessary to test internal functions not exported from TBB DLLs
**/
#if (_WIN32||_WIN64) && (__TBB_SOURCE_DIRECTLY_INCLUDED || TBB_USE_PREVIEW_BINARY)
    #define __TBB_NO_IMPLICIT_LINKAGE 1
    #define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
#endif

#if (__TBB_BUILD || __TBBMALLOC_BUILD || __TBBMALLOCPROXY_BUILD || __TBBBIND_BUILD) && !defined(__TBB_NO_IMPLICIT_LINKAGE)
    #define __TBB_NO_IMPLICIT_LINKAGE 1
#endif

#if _MSC_VER
    #if !__TBB_NO_IMPLICIT_LINKAGE
        #ifdef _DEBUG
            #pragma comment(lib, "tbb12_debug.lib")
        #else
            #pragma comment(lib, "tbb12.lib")
        #endif
    #endif
#endif

#ifndef __TBB_SCHEDULER_OBSERVER
    #define __TBB_SCHEDULER_OBSERVER 1
#endif /* __TBB_SCHEDULER_OBSERVER */

#ifndef __TBB_FP_CONTEXT
    #define __TBB_FP_CONTEXT 1
#endif /* __TBB_FP_CONTEXT */

#define __TBB_RECYCLE_TO_ENQUEUE __TBB_BUILD // keep non-official

#ifndef __TBB_ARENA_OBSERVER
    #define __TBB_ARENA_OBSERVER __TBB_SCHEDULER_OBSERVER
#endif /* __TBB_ARENA_OBSERVER */

#ifndef __TBB_ARENA_BINDING
    #define __TBB_ARENA_BINDING 1
#endif

#if TBB_PREVIEW_WAITING_FOR_WORKERS || __TBB_BUILD
    #define __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE 1
#endif

#if (TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION || __TBB_BUILD) && __TBB_ARENA_BINDING
    #define __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT 1
#endif

#ifndef __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    #define __TBB_ENQUEUE_ENFORCED_CONCURRENCY 1
#endif

#if !defined(__TBB_SURVIVE_THREAD_SWITCH) && \
          (_WIN32 || _WIN64 || __APPLE__ || (__linux__ && !__ANDROID__))
    #define __TBB_SURVIVE_THREAD_SWITCH 1
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

#ifndef TBB_PREVIEW_FLOW_GRAPH_FEATURES
    #define TBB_PREVIEW_FLOW_GRAPH_FEATURES __TBB_CPF_BUILD
#endif

#ifndef __TBB_DEFAULT_PARTITIONER
    #define __TBB_DEFAULT_PARTITIONER tbb::auto_partitioner
#endif

#ifndef __TBB_FLOW_TRACE_CODEPTR
    #define __TBB_FLOW_TRACE_CODEPTR __TBB_CPF_BUILD
#endif

// Intel(R) C++ Compiler starts analyzing usages of the deprecated content at the template
// instantiation site, which is too late for suppression of the corresponding messages for internal
// stuff.
#if !defined(__INTEL_COMPILER) && (!defined(TBB_SUPPRESS_DEPRECATED_MESSAGES) || (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0))
    #if (__TBB_LANG >= 201402L)
        #define __TBB_DEPRECATED [[deprecated]]
        #define __TBB_DEPRECATED_MSG(msg) [[deprecated(msg)]]
    #elif _MSC_VER
        #define __TBB_DEPRECATED __declspec(deprecated)
        #define __TBB_DEPRECATED_MSG(msg) __declspec(deprecated(msg))
    #elif (__GNUC__ && __TBB_GCC_VERSION >= 40805) || __clang__
        #define __TBB_DEPRECATED __attribute__((deprecated))
        #define __TBB_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
    #endif
#endif  // !defined(TBB_SUPPRESS_DEPRECATED_MESSAGES) || (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0)

#if !defined(__TBB_DEPRECATED)
    #define __TBB_DEPRECATED
    #define __TBB_DEPRECATED_MSG(msg)
#elif !defined(__TBB_SUPPRESS_INTERNAL_DEPRECATED_MESSAGES)
    // Suppress deprecated messages from self
    #define __TBB_SUPPRESS_INTERNAL_DEPRECATED_MESSAGES 1
#endif

#if defined(TBB_SUPPRESS_DEPRECATED_MESSAGES) && (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0)
    #define __TBB_DEPRECATED_VERBOSE __TBB_DEPRECATED
    #define __TBB_DEPRECATED_VERBOSE_MSG(msg) __TBB_DEPRECATED_MSG(msg)
#else
    #define __TBB_DEPRECATED_VERBOSE
    #define __TBB_DEPRECATED_VERBOSE_MSG(msg)
#endif // (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0)

#if (!defined(TBB_SUPPRESS_DEPRECATED_MESSAGES) || (TBB_SUPPRESS_DEPRECATED_MESSAGES == 0)) && !(__TBB_LANG >= 201103L || _MSC_VER >= 1900)
    #pragma message("TBB Warning: Support for C++98/03 is deprecated. Please use the compiler that supports C++11 features at least.")
#endif

#ifdef _VARIADIC_MAX
    #define __TBB_VARIADIC_MAX _VARIADIC_MAX
#else
    #if _MSC_VER == 1700
        #define __TBB_VARIADIC_MAX 5 // VS11 setting, issue resolved in VS12
    #elif _MSC_VER == 1600
        #define __TBB_VARIADIC_MAX 10 // VS10 setting
    #else
        #define __TBB_VARIADIC_MAX 15
    #endif
#endif

/** Macros of the form __TBB_XXX_BROKEN denote known issues that are caused by
    the bugs in compilers, standard or OS specific libraries. They should be
    removed as soon as the corresponding bugs are fixed or the buggy OS/compiler
    versions go out of the support list.
**/

// Some STL containers not support allocator traits in old GCC versions
#if __GXX_EXPERIMENTAL_CXX0X__ && __TBB_GLIBCXX_VERSION <= 50301
    #define TBB_ALLOCATOR_TRAITS_BROKEN 1
#endif

// GCC 4.8 C++ standard library implements std::this_thread::yield as no-op.
#if __TBB_GLIBCXX_VERSION >= 40800 && __TBB_GLIBCXX_VERSION < 40900
    #define __TBB_GLIBCXX_THIS_THREAD_YIELD_BROKEN 1
#endif

/** End of __TBB_XXX_BROKEN macro section **/

#if defined(_MSC_VER) && _MSC_VER>=1500 && !defined(__INTEL_COMPILER)
    // A macro to suppress erroneous or benign "unreachable code" MSVC warning (4702)
    #define __TBB_MSVC_UNREACHABLE_CODE_IGNORED 1
#endif

// Many OS versions (Android 4.0.[0-3] for example) need workaround for dlopen to avoid non-recursive loader lock hang
// Setting the workaround for all compile targets ($APP_PLATFORM) below Android 4.4 (android-19)
#if __ANDROID__
    #include <android/api-level.h>
#endif

#define __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING (TBB_PREVIEW_FLOW_GRAPH_FEATURES)

#ifndef __TBB_PREVIEW_CRITICAL_TASKS
#define __TBB_PREVIEW_CRITICAL_TASKS            1
#endif

#ifndef __TBB_PREVIEW_FLOW_GRAPH_NODE_SET
#define __TBB_PREVIEW_FLOW_GRAPH_NODE_SET       (TBB_PREVIEW_FLOW_GRAPH_FEATURES)
#endif

#endif // __TBB_detail__config_H
