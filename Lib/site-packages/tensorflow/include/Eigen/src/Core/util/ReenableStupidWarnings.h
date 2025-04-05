#ifdef EIGEN_WARNINGS_DISABLED_2
// "DisableStupidWarnings.h" was included twice recursively: Do not re-enable warnings yet!
#undef EIGEN_WARNINGS_DISABLED_2

#elif defined(EIGEN_WARNINGS_DISABLED)
#undef EIGEN_WARNINGS_DISABLED

#ifndef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#ifdef _MSC_VER
#pragma warning(pop)
#ifdef EIGEN_REENABLE_CXX23_DENORM_DEPRECATION_WARNING
#undef EIGEN_REENABLE_CXX23_DENORM_DEPRECATION_WARNING
#undef _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#endif

#elif defined __INTEL_COMPILER
#pragma warning pop
#elif defined __clang__
#pragma clang diagnostic pop
#elif defined __GNUC__ && !defined(__FUJITSU) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic pop
#endif

#if defined __NVCC__
//    Don't re-enable the diagnostic messages, as it turns out these messages need
//    to be disabled at the point of the template instantiation (i.e the user code)
//    otherwise they'll be triggered by nvcc.
//    #define EIGEN_MAKE_PRAGMA(X) _Pragma(#X)
//    #if __NVCC_DIAG_PRAGMA_SUPPORT__
//      #define EIGEN_NV_DIAG_DEFAULT(X) EIGEN_MAKE_PRAGMA(nv_diag_default X)
//    #else
//      #define EIGEN_NV_DIAG_DEFAULT(X) EIGEN_MAKE_PRAGMA(diag_default X)
//    #endif
//    EIGEN_NV_DIAG_DEFAULT(code_is_unreachable)
//    EIGEN_NV_DIAG_DEFAULT(initialization_not_reachable)
//    EIGEN_NV_DIAG_DEFAULT(2651)
//    EIGEN_NV_DIAG_DEFAULT(2653)
//    #undef EIGEN_NV_DIAG_DEFAULT
//    #undef EIGEN_MAKE_PRAGMA
#endif

#endif

#endif  // EIGEN_WARNINGS_DISABLED
