// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TERNARY_FUNCTORS_H
#define EIGEN_TERNARY_FUNCTORS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

//---------- associative ternary functors ----------

template <typename ThenScalar, typename ElseScalar, typename ConditionScalar>
struct scalar_boolean_select_op {
  static constexpr bool ThenElseAreSame = is_same<ThenScalar, ElseScalar>::value;
  EIGEN_STATIC_ASSERT(ThenElseAreSame, THEN AND ELSE MUST BE SAME TYPE)
  using Scalar = ThenScalar;
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const ThenScalar& a, const ElseScalar& b,
                                                          const ConditionScalar& cond) const {
    return cond == ConditionScalar(0) ? b : a;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b, const Packet& cond) const {
    return pselect(pcmp_eq(cond, pzero(cond)), b, a);
  }
};

template <typename ThenScalar, typename ElseScalar, typename ConditionScalar>
struct functor_traits<scalar_boolean_select_op<ThenScalar, ElseScalar, ConditionScalar>> {
  using Scalar = ThenScalar;
  enum {
    Cost = 1,
    PacketAccess = is_same<ThenScalar, ElseScalar>::value && is_same<ConditionScalar, Scalar>::value &&
                   packet_traits<Scalar>::HasCmp
  };
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TERNARY_FUNCTORS_H
