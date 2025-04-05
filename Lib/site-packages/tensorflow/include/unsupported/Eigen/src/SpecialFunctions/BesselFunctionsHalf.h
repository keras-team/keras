// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BESSELFUNCTIONS_HALF_H
#define EIGEN_BESSELFUNCTIONS_HALF_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace numext {

#if EIGEN_HAS_C99_MATH
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_i0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_i0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_i0e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_i0e(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_i1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_i1(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_i1e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_i1e(static_cast<float>(x)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_j0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_j0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_j1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_j1(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_y0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_y0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_y1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_y1(static_cast<float>(x)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_k0(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_k0(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_k0e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_k0e(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_k1(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_k1(static_cast<float>(x)));
}
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half bessel_k1e(const Eigen::half& x) {
  return Eigen::half(Eigen::numext::bessel_k1e(static_cast<float>(x)));
}
#endif

}  // end namespace numext
}  // end namespace Eigen

#endif  // EIGEN_BESSELFUNCTIONS_HALF_H
