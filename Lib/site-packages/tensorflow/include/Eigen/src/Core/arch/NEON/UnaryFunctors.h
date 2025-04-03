// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NEON_UNARY_FUNCTORS_H
#define EIGEN_NEON_UNARY_FUNCTORS_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

#if EIGEN_HAS_ARM64_FP16_VECTOR_ARITHMETIC
/** \internal
 * \brief Template specialization of the logistic function for Eigen::half.
 */
template <>
struct scalar_logistic_op<Eigen::half> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half operator()(const Eigen::half& x) const {
    // Convert to float and call scalar_logistic_op<float>.
    const scalar_logistic_op<float> float_op;
    return Eigen::half(float_op(float(x)));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half packetOp(const Eigen::half& x) const { return this->operator()(x); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4hf packetOp(const Packet4hf& x) const {
    const scalar_logistic_op<float> float_op;
    return vcvt_f16_f32(float_op.packetOp(vcvt_f32_f16(x)));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet8hf packetOp(const Packet8hf& x) const {
    const scalar_logistic_op<float> float_op;
    return vcombine_f16(vcvt_f16_f32(float_op.packetOp(vcvt_f32_f16(vget_low_f16(x)))),
                        vcvt_f16_f32(float_op.packetOp(vcvt_high_f32_f16(x))));
  }
};

template <>
struct functor_traits<scalar_logistic_op<Eigen::half>> {
  enum {
    Cost = functor_traits<scalar_logistic_op<float>>::Cost,
    PacketAccess = functor_traits<scalar_logistic_op<float>>::PacketAccess,
  };
};
#endif  // EIGEN_HAS_ARM64_FP16_VECTOR_ARITHMETIC

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_NEON_UNARY_FUNCTORS_H
