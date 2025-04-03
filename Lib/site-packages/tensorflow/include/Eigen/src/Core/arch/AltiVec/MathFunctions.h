// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2016 Konstantinos Margaritis <markos@freevec.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATH_FUNCTIONS_ALTIVEC_H
#define EIGEN_MATH_FUNCTIONS_ALTIVEC_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet4f)
#ifdef EIGEN_VECTORIZE_VSX
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet2d)
#endif

#ifdef EIGEN_VECTORIZE_VSX
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet4f psqrt<Packet4f>(const Packet4f& x) {
  return vec_sqrt(x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet2d psqrt<Packet2d>(const Packet2d& x) {
  return vec_sqrt(x);
}

#if !EIGEN_COMP_CLANG
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet4f prsqrt<Packet4f>(const Packet4f& x) {
  return pset1<Packet4f>(1.0f) / psqrt<Packet4f>(x);
  //  vec_rsqrt returns different results from the generic version
  //  return  vec_rsqrt(x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet2d prsqrt<Packet2d>(const Packet2d& x) {
  return pset1<Packet2d>(1.0) / psqrt<Packet2d>(x);
  //  vec_rsqrt returns different results from the generic version
  //  return  vec_rsqrt(x);
}

#endif

template <>
EIGEN_STRONG_INLINE Packet8bf psqrt<Packet8bf>(const Packet8bf& a) {
  BF16_TO_F32_UNARY_OP_WRAPPER(psqrt<Packet4f>, a);
}

#if !EIGEN_COMP_CLANG
template <>
EIGEN_STRONG_INLINE Packet8bf prsqrt<Packet8bf>(const Packet8bf& a) {
  BF16_TO_F32_UNARY_OP_WRAPPER(prsqrt<Packet4f>, a);
}
#endif
#else
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet4f psqrt<Packet4f>(const Packet4f& x) {
  Packet4f a;
  for (Index i = 0; i < packet_traits<float>::size; i++) {
    a[i] = numext::sqrt(x[i]);
  }
  return a;
}
#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_ALTIVEC_H
