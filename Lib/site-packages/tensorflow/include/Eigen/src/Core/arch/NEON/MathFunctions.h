// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATH_FUNCTIONS_NEON_H
#define EIGEN_MATH_FUNCTIONS_NEON_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet2f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet4f)

#if EIGEN_HAS_ARM64_FP16_VECTOR_ARITHMETIC
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet4hf ptanh<Packet4hf>(const Packet4hf& x) {
  // Convert to float, call the float ptanh, and then convert back.
  return vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(x)));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet8hf ptanh<Packet8hf>(const Packet8hf& x) {
  // Convert each 4 halfs to float, call the float ptanh, and then convert back.
  return vcombine_f16(vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(vget_low_f16(x)))),
                      vcvt_f16_f32(ptanh<Packet4f>(vcvt_high_f32_f16(x))));
}
#endif  // EIGEN_HAS_ARM64_FP16_VECTOR_ARITHMETIC

BF16_PACKET_FUNCTION(Packet4f, Packet4bf, psin)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pcos)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, plog)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pexp)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, ptanh)

template <>
EIGEN_STRONG_INLINE Packet4bf pfrexp(const Packet4bf& a, Packet4bf& exponent) {
  Packet4f fexponent;
  const Packet4bf out = F32ToBf16(pfrexp<Packet4f>(Bf16ToF32(a), fexponent));
  exponent = F32ToBf16(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet4bf pldexp(const Packet4bf& a, const Packet4bf& exponent) {
  return F32ToBf16(pldexp<Packet4f>(Bf16ToF32(a), Bf16ToF32(exponent)));
}

//---------- double ----------

#if EIGEN_ARCH_ARM64 && !EIGEN_APPLE_DOUBLE_NEON_BUG

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet2d)

#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_NEON_H
