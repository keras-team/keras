#ifndef EIGEN_WARNINGS_DISABLED
#define EIGEN_WARNINGS_DISABLED

#if defined(_MSC_VER)
// 4100 - unreferenced formal parameter (occurred e.g. in aligned_allocator::destroy(pointer p))
// 4101 - unreferenced local variable
// 4127 - conditional expression is constant
// 4181 - qualifier applied to reference type ignored
// 4211 - nonstandard extension used : redefined extern to static
// 4244 - 'argument' : conversion from 'type1' to 'type2', possible loss of data
// 4273 - QtAlignedMalloc, inconsistent DLL linkage
// 4324 - structure was padded due to declspec(align())
// 4503 - decorated name length exceeded, name was truncated
// 4512 - assignment operator could not be generated
// 4522 - 'class' : multiple assignment operators specified
// 4700 - uninitialized local variable 'xyz' used
// 4714 - function marked as __forceinline not inlined
// 4717 - 'function' : recursive on all control paths, function will cause runtime stack overflow
// 4800 - 'type' : forcing value to bool 'true' or 'false' (performance warning)
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma warning(push)
#endif
#pragma warning(disable : 4100 4101 4127 4181 4211 4244 4273 4324 4503 4512 4522 4700 4714 4717 4800)
// We currently rely on has_denorm in tests, and need it defined correctly for half/bfloat16.
#ifndef _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#define EIGEN_REENABLE_CXX23_DENORM_DEPRECATION_WARNING 1
#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#endif

#elif defined __INTEL_COMPILER
// 2196 - routine is both "inline" and "noinline" ("noinline" assumed)
//        ICC 12 generates this warning even without any inline keyword, when defining class methods 'inline' i.e.
//        inside of class body typedef that may be a reference type.
// 279  - controlling expression is constant
//        ICC 12 generates this warning on assert(constant_expression_depending_on_template_params) and frankly this is
//        a legitimate use case.
// 1684 - conversion from pointer to same-sized integral type (potential portability problem)
// 2259 - non-pointer conversion from "Eigen::Index={ptrdiff_t={long}}" to "int" may lose significant bits
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma warning push
#endif
#pragma warning disable 2196 279 1684 2259

#elif defined __clang__
#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#pragma clang diagnostic push
#endif
#if defined(__has_warning)
// -Wconstant-logical-operand - warning: use of logical && with constant operand; switch to bitwise & or remove constant
//     this is really a stupid warning as it warns on compile-time expressions involving enums
#if __has_warning("-Wconstant-logical-operand")
#pragma clang diagnostic ignored "-Wconstant-logical-operand"
#endif
#if __has_warning("-Wimplicit-int-float-conversion")
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#endif
#if (defined(__ALTIVEC__) || defined(__VSX__)) && (!defined(__STDC_VERSION__) || (__STDC_VERSION__ < 201112L))
// warning: generic selections are a C11-specific feature
// ignoring warnings thrown at vec_ctf in Altivec/PacketMath.h
#if __has_warning("-Wc11-extensions")
#pragma clang diagnostic ignored "-Wc11-extensions"
#endif
#endif
#endif

#elif defined __GNUC__ && !defined(__FUJITSU)

#if (!defined(EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS)) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic push
#endif
// g++ warns about local variables shadowing member functions, which is too strict
#pragma GCC diagnostic ignored "-Wshadow"
#if __GNUC__ == 4 && __GNUC_MINOR__ < 8
// Until g++-4.7 there are warnings when comparing unsigned int vs 0, even in templated functions:
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#if __GNUC__ == 7
// See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89325
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#endif

#if defined __NVCC__
// MSVC 14.16 (required by CUDA 9.*) does not support the _Pragma keyword, so
// we instead use Microsoft's __pragma extension.
#if defined _MSC_VER
#define EIGEN_MAKE_PRAGMA(X) __pragma(#X)
#else
#define EIGEN_MAKE_PRAGMA(X) _Pragma(#X)
#endif
#if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#define EIGEN_NV_DIAG_SUPPRESS(X) EIGEN_MAKE_PRAGMA(nv_diag_suppress X)
#else
#define EIGEN_NV_DIAG_SUPPRESS(X) EIGEN_MAKE_PRAGMA(diag_suppress X)
#endif

EIGEN_NV_DIAG_SUPPRESS(boolean_controlling_expr_is_constant)
// Disable the "statement is unreachable" message
EIGEN_NV_DIAG_SUPPRESS(code_is_unreachable)
// Disable the "dynamic initialization in unreachable code" message
EIGEN_NV_DIAG_SUPPRESS(initialization_not_reachable)
// Disable the "invalid error number" message that we get with older versions of nvcc
EIGEN_NV_DIAG_SUPPRESS(1222)
// Disable the "calling a __host__ function from a __host__ __device__ function is not allowed" messages (yes, there are
// many of them and they seem to change with every version of the compiler)
EIGEN_NV_DIAG_SUPPRESS(2527)
EIGEN_NV_DIAG_SUPPRESS(2529)
EIGEN_NV_DIAG_SUPPRESS(2651)
EIGEN_NV_DIAG_SUPPRESS(2653)
EIGEN_NV_DIAG_SUPPRESS(2668)
EIGEN_NV_DIAG_SUPPRESS(2669)
EIGEN_NV_DIAG_SUPPRESS(2670)
EIGEN_NV_DIAG_SUPPRESS(2671)
EIGEN_NV_DIAG_SUPPRESS(2735)
EIGEN_NV_DIAG_SUPPRESS(2737)
EIGEN_NV_DIAG_SUPPRESS(2739)
EIGEN_NV_DIAG_SUPPRESS(2885)
EIGEN_NV_DIAG_SUPPRESS(2888)
EIGEN_NV_DIAG_SUPPRESS(2976)
EIGEN_NV_DIAG_SUPPRESS(2979)
EIGEN_NV_DIAG_SUPPRESS(20011)
EIGEN_NV_DIAG_SUPPRESS(20014)
// Disable the "// __device__ annotation is ignored on a function(...) that is
//              explicitly defaulted on its first declaration" message.
// The __device__ annotation seems to actually be needed in some cases,
// otherwise resulting in kernel runtime errors.
EIGEN_NV_DIAG_SUPPRESS(2886)
EIGEN_NV_DIAG_SUPPRESS(2929)
EIGEN_NV_DIAG_SUPPRESS(2977)
EIGEN_NV_DIAG_SUPPRESS(20012)
#undef EIGEN_NV_DIAG_SUPPRESS
#undef EIGEN_MAKE_PRAGMA
#endif

#else
// warnings already disabled:
#ifndef EIGEN_WARNINGS_DISABLED_2
#define EIGEN_WARNINGS_DISABLED_2
#elif defined(EIGEN_INTERNAL_DEBUGGING)
#error "Do not include \"DisableStupidWarnings.h\" recursively more than twice!"
#endif

#endif  // not EIGEN_WARNINGS_DISABLED
