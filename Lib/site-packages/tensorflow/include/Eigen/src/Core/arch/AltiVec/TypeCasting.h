// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Rasmus Munk Larsen <rmlarsen@google.com>
// Copyright (C) 2023 Chip Kerchner (chip.kerchner@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_ALTIVEC_H
#define EIGEN_TYPE_CASTING_ALTIVEC_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <>
struct type_casting_traits<float, int> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<int, float> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<bfloat16, unsigned short int> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
struct type_casting_traits<unsigned short int, bfloat16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet4i pcast<Packet4f, Packet4i>(const Packet4f& a) {
  return vec_cts(a, 0);
}

template <>
EIGEN_STRONG_INLINE Packet4ui pcast<Packet4f, Packet4ui>(const Packet4f& a) {
  return vec_ctu(a, 0);
}

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet4i, Packet4f>(const Packet4i& a) {
  return vec_ctf(a, 0);
}

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet4ui, Packet4f>(const Packet4ui& a) {
  return vec_ctf(a, 0);
}

template <>
EIGEN_STRONG_INLINE Packet8us pcast<Packet8bf, Packet8us>(const Packet8bf& a) {
  Packet4f float_even = Bf16ToF32Even(a);
  Packet4f float_odd = Bf16ToF32Odd(a);
  Packet4ui int_even = pcast<Packet4f, Packet4ui>(float_even);
  Packet4ui int_odd = pcast<Packet4f, Packet4ui>(float_odd);
  const EIGEN_DECLARE_CONST_FAST_Packet4ui(low_mask, 0x0000FFFF);
  Packet4ui low_even = pand<Packet4ui>(int_even, p4ui_low_mask);
  Packet4ui low_odd = pand<Packet4ui>(int_odd, p4ui_low_mask);

  // Check values that are bigger than USHRT_MAX (0xFFFF)
  Packet4bi overflow_selector;
  if (vec_any_gt(int_even, p4ui_low_mask)) {
    overflow_selector = vec_cmpgt(int_even, p4ui_low_mask);
    low_even = vec_sel(low_even, p4ui_low_mask, overflow_selector);
  }
  if (vec_any_gt(int_odd, p4ui_low_mask)) {
    overflow_selector = vec_cmpgt(int_odd, p4ui_low_mask);
    low_odd = vec_sel(low_even, p4ui_low_mask, overflow_selector);
  }

  return pmerge(low_even, low_odd);
}

template <>
EIGEN_STRONG_INLINE Packet8bf pcast<Packet8us, Packet8bf>(const Packet8us& a) {
  // short -> int -> float -> bfloat16
  const EIGEN_DECLARE_CONST_FAST_Packet4ui(low_mask, 0x0000FFFF);
  Packet4ui int_cast = reinterpret_cast<Packet4ui>(a);
  Packet4ui int_even = pand<Packet4ui>(int_cast, p4ui_low_mask);
  Packet4ui int_odd = plogical_shift_right<16>(int_cast);
  Packet4f float_even = pcast<Packet4ui, Packet4f>(int_even);
  Packet4f float_odd = pcast<Packet4ui, Packet4f>(int_odd);
  return F32ToBf16(float_even, float_odd);
}

template <>
struct type_casting_traits<bfloat16, float> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 2 };
};

template <>
EIGEN_STRONG_INLINE Packet4f pcast<Packet8bf, Packet4f>(const Packet8bf& a) {
  Packet8us z = pset1<Packet8us>(0);
#ifdef _BIG_ENDIAN
  return reinterpret_cast<Packet4f>(vec_mergeh(a.m_val, z));
#else
  return reinterpret_cast<Packet4f>(vec_mergeh(z, a.m_val));
#endif
}

template <>
struct type_casting_traits<float, bfloat16> {
  enum { VectorizedCast = 1, SrcCoeffRatio = 2, TgtCoeffRatio = 1 };
};

template <>
EIGEN_STRONG_INLINE Packet8bf pcast<Packet4f, Packet8bf>(const Packet4f& a, const Packet4f& b) {
  return F32ToBf16Both(a, b);
}

template <>
EIGEN_STRONG_INLINE Packet4i preinterpret<Packet4i, Packet4f>(const Packet4f& a) {
  return reinterpret_cast<Packet4i>(a);
}

template <>
EIGEN_STRONG_INLINE Packet4f preinterpret<Packet4f, Packet4i>(const Packet4i& a) {
  return reinterpret_cast<Packet4f>(a);
}

#ifdef EIGEN_VECTORIZE_VSX
// VSX support varies between different compilers and even different
// versions of the same compiler.  For gcc version >= 4.9.3, we can use
// vec_cts to efficiently convert Packet2d to Packet2l.  Otherwise, use
// a slow version that works with older compilers.
// Update: apparently vec_cts/vec_ctf intrinsics for 64-bit doubles
// are buggy, https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70963
template <>
inline Packet2l pcast<Packet2d, Packet2l>(const Packet2d& x) {
#if EIGEN_GNUC_STRICT_AT_LEAST(7, 1, 0)
  return vec_cts(x, 0);  // TODO: check clang version.
#else
  double tmp[2];
  memcpy(tmp, &x, sizeof(tmp));
  Packet2l l = {static_cast<long long>(tmp[0]), static_cast<long long>(tmp[1])};
  return l;
#endif
}

template <>
inline Packet2d pcast<Packet2l, Packet2d>(const Packet2l& x) {
  unsigned long long tmp[2];
  memcpy(tmp, &x, sizeof(tmp));
  Packet2d d = {static_cast<double>(tmp[0]), static_cast<double>(tmp[1])};
  return d;
}
#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TYPE_CASTING_ALTIVEC_H
