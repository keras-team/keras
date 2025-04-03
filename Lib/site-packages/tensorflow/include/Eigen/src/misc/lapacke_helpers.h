// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Erik Schultheis <erik.schultheis@aalto.fi>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_LAPACKE_HELPERS_H
#define EIGEN_LAPACKE_HELPERS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#ifdef EIGEN_USE_MKL
#include "mkl_lapacke.h"
#else
#include "lapacke.h"
#endif

namespace Eigen {
namespace internal {
/**
 * \internal
 * \brief Implementation details and helper functions for the lapacke glue code.
 */
namespace lapacke_helpers {

// ---------------------------------------------------------------------------------------------------------------------
//                  Translation from Eigen to Lapacke for types and constants
// ---------------------------------------------------------------------------------------------------------------------

// For complex numbers, the types in Eigen and Lapacke are different, but layout compatible.
template <typename Scalar>
struct translate_type_imp;
template <>
struct translate_type_imp<float> {
  using type = float;
};
template <>
struct translate_type_imp<double> {
  using type = double;
};
template <>
struct translate_type_imp<std::complex<double>> {
  using type = lapack_complex_double;
};
template <>
struct translate_type_imp<std::complex<float>> {
  using type = lapack_complex_float;
};

/// Given an Eigen types, this is defined to be the corresponding, layout-compatible lapack type
template <typename Scalar>
using translated_type = typename translate_type_imp<Scalar>::type;

/// These functions convert their arguments from Eigen to Lapack types
/// This function performs conversion for any of the translations defined above.
template <typename Source, typename Target = translated_type<Source>>
EIGEN_ALWAYS_INLINE auto to_lapack(Source value) {
  return static_cast<Target>(value);
}

/// This function performs conversions for pointer types corresponding to the translations abovce.
/// This is valid because the translations are between layout-compatible types.
template <typename Source, typename Target = translated_type<Source>>
EIGEN_ALWAYS_INLINE auto to_lapack(Source *value) {
  return reinterpret_cast<Target *>(value);
}

/// This function converts the Eigen Index to a lapack index, with possible range checks
/// \sa internal::convert_index
EIGEN_ALWAYS_INLINE lapack_int to_lapack(Index index) { return convert_index<lapack_int>(index); }

/// translates storage order of the given Eigen object to the corresponding lapack constant
template <typename Derived>
EIGEN_ALWAYS_INLINE EIGEN_CONSTEXPR lapack_int lapack_storage_of(const EigenBase<Derived> &) {
  return Derived::IsRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
}

// ---------------------------------------------------------------------------------------------------------------------
//              Automatic generation of low-level wrappers
// ---------------------------------------------------------------------------------------------------------------------

/*!
 * \internal
 * \brief Helper type to facilitate the wrapping of raw LAPACKE functions for different types into a single, overloaded
 * C++ function. This is achieved in combination with \r EIGEN_MAKE_LAPACKE_WRAPPER \details This implementation works
 * by providing an overloaded call function that just forwards its arguments to the underlying lapack function. Each of
 * these overloads is enabled only if the call is actually well formed. Because these lapack functions take pointers to
 * the underlying scalar type as arguments, even though the actual Scalars would be implicitly convertible, the pointers
 * are not and therefore only a single overload can be valid at the same time. Thus, despite all functions taking fully
 * generic `Args&&... args` as arguments, there is never any ambiguity.
 */
template <typename DoubleFn, typename SingleFn, typename DoubleCpxFn, typename SingleCpxFn>
struct WrappingHelper {
  // The naming of double, single, double complex and single complex is purely for readability
  // and doesn't actually affect the workings of this class. In principle, the arguments can
  // be supplied in any permuted order.
  DoubleFn double_;
  SingleFn single_;
  DoubleCpxFn double_cpx_;
  SingleCpxFn single_cpx_;

  template <typename... Args>
  auto call(Args &&...args) -> decltype(double_(std::forward<Args>(args)...)) {
    return double_(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto call(Args &&...args) -> decltype(single_(std::forward<Args>(args)...)) {
    return single_(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto call(Args &&...args) -> decltype(double_cpx_(std::forward<Args>(args)...)) {
    return double_cpx_(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto call(Args &&...args) -> decltype(single_cpx_(std::forward<Args>(args)...)) {
    return single_cpx_(std::forward<Args>(args)...);
  }
};

/** \internal Helper function that generates a `WrappingHelper` object with the given function pointers and
 * invokes its `call` method, thus selecting one of the overloads.
 * \sa EIGEN_MAKE_LAPACKE_WRAPPER
 */
template <typename DoubleFn, typename SingleFn, typename DoubleCpxFn, typename SingleCpxFn, typename... Args>
EIGEN_ALWAYS_INLINE auto call_wrapper(DoubleFn df, SingleFn sf, DoubleCpxFn dcf, SingleCpxFn scf, Args &&...args) {
  WrappingHelper<DoubleFn, SingleFn, DoubleCpxFn, SingleCpxFn> helper{df, sf, dcf, scf};
  return helper.call(std::forward<Args>(args)...);
}

/**
 * \internal
 * Generates a new function `Function` that dispatches to the corresponding LAPACKE_? prefixed functions.
 * \sa WrappingHelper
 */
#define EIGEN_MAKE_LAPACKE_WRAPPER(FUNCTION)                                                                \
  template <typename... Args>                                                                               \
  EIGEN_ALWAYS_INLINE auto FUNCTION(Args &&...args) {                                                       \
    return call_wrapper(LAPACKE_d##FUNCTION, LAPACKE_s##FUNCTION, LAPACKE_z##FUNCTION, LAPACKE_c##FUNCTION, \
                        std::forward<Args>(args)...);                                                       \
  }

// Now with this macro and the helper wrappers, we can generate the dispatch for all the lapacke functions that are
// used in Eigen.
// We define these here instead of in the files where they are used because this allows us to #undef the macro again
// right here
EIGEN_MAKE_LAPACKE_WRAPPER(potrf)
EIGEN_MAKE_LAPACKE_WRAPPER(getrf)
EIGEN_MAKE_LAPACKE_WRAPPER(geqrf)
EIGEN_MAKE_LAPACKE_WRAPPER(gesdd)

#undef EIGEN_MAKE_LAPACKE_WRAPPER
}  // namespace lapacke_helpers
}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_LAPACKE_HELPERS_H
