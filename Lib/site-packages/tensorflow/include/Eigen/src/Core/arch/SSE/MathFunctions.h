// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Julien Pommier
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* The sin and cos and functions of this file come from
 * Julien Pommier's sse math library: http://gruntthepeon.free.fr/ssemath/
 */

#ifndef EIGEN_MATH_FUNCTIONS_SSE_H
#define EIGEN_MATH_FUNCTIONS_SSE_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet4f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet2d)

// Notice that for newer processors, it is counterproductive to use Newton
// iteration for square root. In particular, Skylake and Zen2 processors
// have approximately doubled throughput of the _mm_sqrt_ps instruction
// compared to their predecessors.
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet4f psqrt<Packet4f>(const Packet4f& x) {
  return _mm_sqrt_ps(x);
}
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet2d psqrt<Packet2d>(const Packet2d& x) {
  return _mm_sqrt_pd(x);
}
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet16b psqrt<Packet16b>(const Packet16b& x) {
  return x;
}

#if EIGEN_FAST_MATH
// Even on Skylake, using Newton iteration is a win for reciprocal square root.
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet4f prsqrt<Packet4f>(const Packet4f& x) {
  return generic_rsqrt_newton_step<Packet4f, /*Steps=*/1>::run(x, _mm_rsqrt_ps(x));
}

#ifdef EIGEN_VECTORIZE_FMA
// Trying to speed up reciprocal using Newton-Raphson is counterproductive
// unless FMA is available. Without FMA pdiv(pset1<Packet>(Scalar(1),a)) is
// 30% faster.
template <>
EIGEN_STRONG_INLINE Packet4f preciprocal<Packet4f>(const Packet4f& x) {
  return generic_reciprocal_newton_step<Packet4f, /*Steps=*/1>::run(x, _mm_rcp_ps(x));
}
#endif

#endif

}  // end namespace internal

namespace numext {

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float sqrt(const float& x) {
  return internal::pfirst(internal::Packet4f(_mm_sqrt_ss(_mm_set_ss(x))));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double sqrt(const double& x) {
#if EIGEN_COMP_GNUC_STRICT
  // This works around a GCC bug generating poor code for _mm_sqrt_pd
  // See https://gitlab.com/libeigen/eigen/commit/8dca9f97e38970
  return internal::pfirst(internal::Packet2d(__builtin_ia32_sqrtsd(_mm_set_sd(x))));
#else
  return internal::pfirst(internal::Packet2d(_mm_sqrt_pd(_mm_set_sd(x))));
#endif
}

}  // namespace numext

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_SSE_H
