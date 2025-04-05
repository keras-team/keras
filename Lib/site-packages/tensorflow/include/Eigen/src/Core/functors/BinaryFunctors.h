// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BINARY_FUNCTORS_H
#define EIGEN_BINARY_FUNCTORS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

//---------- associative binary functors ----------

template <typename Arg1, typename Arg2>
struct binary_op_base {
  typedef Arg1 first_argument_type;
  typedef Arg2 second_argument_type;
};

/** \internal
 * \brief Template functor to compute the sum of two scalars
 *
 * \sa class CwiseBinaryOp, MatrixBase::operator+, class VectorwiseOp, DenseBase::sum()
 */
template <typename LhsScalar, typename RhsScalar>
struct scalar_sum_op : binary_op_base<LhsScalar, RhsScalar> {
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_sum_op>::ReturnType result_type;
#ifdef EIGEN_SCALAR_BINARY_OP_PLUGIN
  scalar_sum_op(){EIGEN_SCALAR_BINARY_OP_PLUGIN}
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type
  operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a + b;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return internal::padd(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type predux(const Packet& a) const {
    return internal::predux(a);
  }
};
template <typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_sum_op<LhsScalar, RhsScalar>> {
  enum {
    Cost = (int(NumTraits<LhsScalar>::AddCost) + int(NumTraits<RhsScalar>::AddCost)) / 2,  // rough estimate!
    PacketAccess =
        is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasAdd && packet_traits<RhsScalar>::HasAdd
    // TODO vectorize mixed sum
  };
};

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool scalar_sum_op<bool, bool>::operator()(const bool& a, const bool& b) const {
  return a || b;
}

/** \internal
 * \brief Template functor to compute the product of two scalars
 *
 * \sa class CwiseBinaryOp, Cwise::operator*(), class VectorwiseOp, MatrixBase::redux()
 */
template <typename LhsScalar, typename RhsScalar>
struct scalar_product_op : binary_op_base<LhsScalar, RhsScalar> {
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_product_op>::ReturnType result_type;
#ifdef EIGEN_SCALAR_BINARY_OP_PLUGIN
  scalar_product_op(){EIGEN_SCALAR_BINARY_OP_PLUGIN}
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type
  operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a * b;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return internal::pmul(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type predux(const Packet& a) const {
    return internal::predux_mul(a);
  }
};
template <typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_product_op<LhsScalar, RhsScalar>> {
  enum {
    Cost = (int(NumTraits<LhsScalar>::MulCost) + int(NumTraits<RhsScalar>::MulCost)) / 2,  // rough estimate!
    PacketAccess =
        is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasMul && packet_traits<RhsScalar>::HasMul
    // TODO vectorize mixed product
  };
};

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool scalar_product_op<bool, bool>::operator()(const bool& a,
                                                                                     const bool& b) const {
  return a && b;
}

/** \internal
 * \brief Template functor to compute the conjugate product of two scalars
 *
 * This is a short cut for conj(x) * y which is needed for optimization purpose; in Eigen2 support mode, this becomes x
 * * conj(y)
 */
template <typename LhsScalar, typename RhsScalar>
struct scalar_conj_product_op : binary_op_base<LhsScalar, RhsScalar> {
  enum { Conj = NumTraits<LhsScalar>::IsComplex };

  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_conj_product_op>::ReturnType result_type;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return conj_helper<LhsScalar, RhsScalar, Conj, false>().pmul(a, b);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return conj_helper<Packet, Packet, Conj, false>().pmul(a, b);
  }
};
template <typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_conj_product_op<LhsScalar, RhsScalar>> {
  enum {
    Cost = NumTraits<LhsScalar>::MulCost,
    PacketAccess = internal::is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasMul
  };
};

/** \internal
 * \brief Template functor to compute the min of two scalars
 *
 * \sa class CwiseBinaryOp, MatrixBase::cwiseMin, class VectorwiseOp, MatrixBase::minCoeff()
 */
template <typename LhsScalar, typename RhsScalar, int NaNPropagation>
struct scalar_min_op : binary_op_base<LhsScalar, RhsScalar> {
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_min_op>::ReturnType result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return internal::pmin<NaNPropagation>(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return internal::pmin<NaNPropagation>(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type predux(const Packet& a) const {
    return internal::predux_min<NaNPropagation>(a);
  }
};

template <typename LhsScalar, typename RhsScalar, int NaNPropagation>
struct functor_traits<scalar_min_op<LhsScalar, RhsScalar, NaNPropagation>> {
  enum {
    Cost = (NumTraits<LhsScalar>::AddCost + NumTraits<RhsScalar>::AddCost) / 2,
    PacketAccess = internal::is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasMin
  };
};

/** \internal
 * \brief Template functor to compute the max of two scalars
 *
 * \sa class CwiseBinaryOp, MatrixBase::cwiseMax, class VectorwiseOp, MatrixBase::maxCoeff()
 */
template <typename LhsScalar, typename RhsScalar, int NaNPropagation>
struct scalar_max_op : binary_op_base<LhsScalar, RhsScalar> {
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_max_op>::ReturnType result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return internal::pmax<NaNPropagation>(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return internal::pmax<NaNPropagation>(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type predux(const Packet& a) const {
    return internal::predux_max<NaNPropagation>(a);
  }
};

template <typename LhsScalar, typename RhsScalar, int NaNPropagation>
struct functor_traits<scalar_max_op<LhsScalar, RhsScalar, NaNPropagation>> {
  enum {
    Cost = (NumTraits<LhsScalar>::AddCost + NumTraits<RhsScalar>::AddCost) / 2,
    PacketAccess = internal::is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasMax
  };
};

/** \internal
 * \brief Template functors for comparison of two scalars
 * \todo Implement packet-comparisons
 */
template <typename LhsScalar, typename RhsScalar, ComparisonName cmp, bool UseTypedComparators = false>
struct scalar_cmp_op;

template <typename LhsScalar, typename RhsScalar, ComparisonName cmp, bool UseTypedComparators>
struct functor_traits<scalar_cmp_op<LhsScalar, RhsScalar, cmp, UseTypedComparators>> {
  enum {
    Cost = (NumTraits<LhsScalar>::AddCost + NumTraits<RhsScalar>::AddCost) / 2,
    PacketAccess = (UseTypedComparators || is_same<LhsScalar, bool>::value) && is_same<LhsScalar, RhsScalar>::value &&
                   packet_traits<LhsScalar>::HasCmp
  };
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct typed_cmp_helper {
  static constexpr bool SameType = is_same<LhsScalar, RhsScalar>::value;
  static constexpr bool IsNumeric = is_arithmetic<typename NumTraits<LhsScalar>::Real>::value;
  static constexpr bool UseTyped = UseTypedComparators && SameType && IsNumeric;
  using type = typename conditional<UseTyped, LhsScalar, bool>::type;
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
using cmp_return_t = typename typed_cmp_helper<LhsScalar, RhsScalar, UseTypedComparators>::type;

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct scalar_cmp_op<LhsScalar, RhsScalar, cmp_EQ, UseTypedComparators> : binary_op_base<LhsScalar, RhsScalar> {
  using result_type = cmp_return_t<LhsScalar, RhsScalar, UseTypedComparators>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a == b ? result_type(1) : result_type(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(result_type(1));
    return pand(pcmp_eq(a, b), cst_one);
  }
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct scalar_cmp_op<LhsScalar, RhsScalar, cmp_LT, UseTypedComparators> : binary_op_base<LhsScalar, RhsScalar> {
  using result_type = cmp_return_t<LhsScalar, RhsScalar, UseTypedComparators>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a < b ? result_type(1) : result_type(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(result_type(1));
    return pand(pcmp_lt(a, b), cst_one);
  }
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct scalar_cmp_op<LhsScalar, RhsScalar, cmp_LE, UseTypedComparators> : binary_op_base<LhsScalar, RhsScalar> {
  using result_type = cmp_return_t<LhsScalar, RhsScalar, UseTypedComparators>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a <= b ? result_type(1) : result_type(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(result_type(1));
    return pand(cst_one, pcmp_le(a, b));
  }
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct scalar_cmp_op<LhsScalar, RhsScalar, cmp_GT, UseTypedComparators> : binary_op_base<LhsScalar, RhsScalar> {
  using result_type = cmp_return_t<LhsScalar, RhsScalar, UseTypedComparators>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a > b ? result_type(1) : result_type(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(result_type(1));
    return pand(cst_one, pcmp_lt(b, a));
  }
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct scalar_cmp_op<LhsScalar, RhsScalar, cmp_GE, UseTypedComparators> : binary_op_base<LhsScalar, RhsScalar> {
  using result_type = cmp_return_t<LhsScalar, RhsScalar, UseTypedComparators>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a >= b ? result_type(1) : result_type(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(result_type(1));
    return pand(cst_one, pcmp_le(b, a));
  }
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct scalar_cmp_op<LhsScalar, RhsScalar, cmp_UNORD, UseTypedComparators> : binary_op_base<LhsScalar, RhsScalar> {
  using result_type = cmp_return_t<LhsScalar, RhsScalar, UseTypedComparators>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return !(a <= b || b <= a) ? result_type(1) : result_type(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(result_type(1));
    return pandnot(cst_one, por(pcmp_le(a, b), pcmp_le(b, a)));
  }
};

template <typename LhsScalar, typename RhsScalar, bool UseTypedComparators>
struct scalar_cmp_op<LhsScalar, RhsScalar, cmp_NEQ, UseTypedComparators> : binary_op_base<LhsScalar, RhsScalar> {
  using result_type = cmp_return_t<LhsScalar, RhsScalar, UseTypedComparators>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a != b ? result_type(1) : result_type(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(result_type(1));
    return pandnot(cst_one, pcmp_eq(a, b));
  }
};

/** \internal
 * \brief Template functor to compute the hypot of two \b positive \b and \b real scalars
 *
 * \sa MatrixBase::stableNorm(), class Redux
 */
template <typename Scalar>
struct scalar_hypot_op<Scalar, Scalar> : binary_op_base<Scalar, Scalar> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x, const Scalar& y) const {
    // This functor is used by hypotNorm only for which it is faster to first apply abs
    // on all coefficients prior to reduction through hypot.
    // This way we avoid calling abs on positive and real entries, and this also permits
    // to seamlessly handle complexes. Otherwise we would have to handle both real and complexes
    // through the same functor...
    return internal::positive_real_hypot(x, y);
  }
};
template <typename Scalar>
struct functor_traits<scalar_hypot_op<Scalar, Scalar>> {
  enum {
    Cost = 3 * NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost + 2 * scalar_div_cost<Scalar, false>::value,
    PacketAccess = false
  };
};

/** \internal
 * \brief Template functor to compute the pow of two scalars
 * See the specification of pow in https://en.cppreference.com/w/cpp/numeric/math/pow
 */
template <typename Scalar, typename Exponent>
struct scalar_pow_op : binary_op_base<Scalar, Exponent> {
  typedef typename ScalarBinaryOpTraits<Scalar, Exponent, scalar_pow_op>::ReturnType result_type;
#ifdef EIGEN_SCALAR_BINARY_OP_PLUGIN
  scalar_pow_op() {
    typedef Scalar LhsScalar;
    typedef Exponent RhsScalar;
    EIGEN_SCALAR_BINARY_OP_PLUGIN
  }
#endif

  EIGEN_DEVICE_FUNC inline result_type operator()(const Scalar& a, const Exponent& b) const {
    return numext::pow(a, b);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const {
    return generic_pow(a, b);
  }
};

template <typename Scalar, typename Exponent>
struct functor_traits<scalar_pow_op<Scalar, Exponent>> {
  enum {
    Cost = 5 * NumTraits<Scalar>::MulCost,
    PacketAccess = (!NumTraits<Scalar>::IsComplex && !NumTraits<Scalar>::IsInteger && packet_traits<Scalar>::HasExp &&
                    packet_traits<Scalar>::HasLog && packet_traits<Scalar>::HasRound && packet_traits<Scalar>::HasCmp &&
                    // Temporarily disable packet access for half/bfloat16 until
                    // accuracy is improved.
                    !is_same<Scalar, half>::value && !is_same<Scalar, bfloat16>::value)
  };
};

//---------- non associative binary functors ----------

/** \internal
 * \brief Template functor to compute the difference of two scalars
 *
 * \sa class CwiseBinaryOp, MatrixBase::operator-
 */
template <typename LhsScalar, typename RhsScalar>
struct scalar_difference_op : binary_op_base<LhsScalar, RhsScalar> {
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_difference_op>::ReturnType result_type;
#ifdef EIGEN_SCALAR_BINARY_OP_PLUGIN
  scalar_difference_op(){EIGEN_SCALAR_BINARY_OP_PLUGIN}
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type
  operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a - b;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const {
    return internal::psub(a, b);
  }
};
template <typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_difference_op<LhsScalar, RhsScalar>> {
  enum {
    Cost = (int(NumTraits<LhsScalar>::AddCost) + int(NumTraits<RhsScalar>::AddCost)) / 2,
    PacketAccess =
        is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasSub && packet_traits<RhsScalar>::HasSub
  };
};

template <typename Packet, bool IsInteger = NumTraits<typename unpacket_traits<Packet>::type>::IsInteger>
struct maybe_raise_div_by_zero {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Packet x) { EIGEN_UNUSED_VARIABLE(x); }
};

#ifndef EIGEN_GPU_COMPILE_PHASE
template <typename Packet>
struct maybe_raise_div_by_zero<Packet, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void run(Packet x) {
    if (EIGEN_PREDICT_FALSE(predux_any(pcmp_eq(x, pzero(x))))) {
      // Use volatile variables to force a division by zero, which will
      // result in the default platform behaviour (usually SIGFPE).
      volatile typename unpacket_traits<Packet>::type zero = 0;
      volatile typename unpacket_traits<Packet>::type val = 1;
      val = val / zero;
    }
  }
};
#endif

/** \internal
 * \brief Template functor to compute the quotient of two scalars
 *
 * \sa class CwiseBinaryOp, Cwise::operator/()
 */
template <typename LhsScalar, typename RhsScalar>
struct scalar_quotient_op : binary_op_base<LhsScalar, RhsScalar> {
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_quotient_op>::ReturnType result_type;
#ifdef EIGEN_SCALAR_BINARY_OP_PLUGIN
  scalar_quotient_op(){EIGEN_SCALAR_BINARY_OP_PLUGIN}
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type
  operator()(const LhsScalar& a, const RhsScalar& b) const {
    return a / b;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const {
    maybe_raise_div_by_zero<Packet>::run(b);
    return internal::pdiv(a, b);
  }
};
template <typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_quotient_op<LhsScalar, RhsScalar>> {
  typedef typename scalar_quotient_op<LhsScalar, RhsScalar>::result_type result_type;
  enum {
    PacketAccess =
        is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasDiv && packet_traits<RhsScalar>::HasDiv,
    Cost = scalar_div_cost<result_type, PacketAccess>::value
  };
};

/** \internal
 * \brief Template functor to compute the and of two scalars as if they were booleans
 *
 * \sa class CwiseBinaryOp, ArrayBase::operator&&
 */
template <typename Scalar>
struct scalar_boolean_and_op {
  using result_type = Scalar;
  // `false` any value `a` that satisfies `a == Scalar(0)`
  // `true` is the complement of `false`
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
    return (a != Scalar(0)) && (b != Scalar(0)) ? Scalar(1) : Scalar(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(Scalar(1));
    // and(a,b) == !or(!a,!b)
    Packet not_a = pcmp_eq(a, pzero(a));
    Packet not_b = pcmp_eq(b, pzero(b));
    Packet a_nand_b = por(not_a, not_b);
    return pandnot(cst_one, a_nand_b);
  }
};
template <typename Scalar>
struct functor_traits<scalar_boolean_and_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasCmp };
};

/** \internal
 * \brief Template functor to compute the or of two scalars as if they were booleans
 *
 * \sa class CwiseBinaryOp, ArrayBase::operator||
 */
template <typename Scalar>
struct scalar_boolean_or_op {
  using result_type = Scalar;
  // `false` any value `a` that satisfies `a == Scalar(0)`
  // `true` is the complement of `false`
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
    return (a != Scalar(0)) || (b != Scalar(0)) ? Scalar(1) : Scalar(0);
  }
  template <typename Packet>
  EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(Scalar(1));
    // if or(a,b) == 0, then a == 0 and b == 0
    // or(a,b) == !nor(a,b)
    Packet a_nor_b = pcmp_eq(por(a, b), pzero(a));
    return pandnot(cst_one, a_nor_b);
  }
};
template <typename Scalar>
struct functor_traits<scalar_boolean_or_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasCmp };
};

/** \internal
 * \brief Template functor to compute the xor of two scalars as if they were booleans
 *
 * \sa class CwiseBinaryOp, ArrayBase::operator^
 */
template <typename Scalar>
struct scalar_boolean_xor_op {
  using result_type = Scalar;
  // `false` any value `a` that satisfies `a == Scalar(0)`
  // `true` is the complement of `false`
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
    return (a != Scalar(0)) != (b != Scalar(0)) ? Scalar(1) : Scalar(0);
  }
  template <typename Packet>
  EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    const Packet cst_one = pset1<Packet>(Scalar(1));
    // xor(a,b) == xor(!a,!b)
    Packet not_a = pcmp_eq(a, pzero(a));
    Packet not_b = pcmp_eq(b, pzero(b));
    Packet a_xor_b = pxor(not_a, not_b);
    return pand(cst_one, a_xor_b);
  }
};
template <typename Scalar>
struct functor_traits<scalar_boolean_xor_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasCmp };
};

template <typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct bitwise_binary_impl {
  static constexpr size_t Size = sizeof(Scalar);
  using uint_t = typename numext::get_integer_by_size<Size>::unsigned_type;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_and(const Scalar& a, const Scalar& b) {
    uint_t a_as_uint = numext::bit_cast<uint_t, Scalar>(a);
    uint_t b_as_uint = numext::bit_cast<uint_t, Scalar>(b);
    uint_t result = a_as_uint & b_as_uint;
    return numext::bit_cast<Scalar, uint_t>(result);
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_or(const Scalar& a, const Scalar& b) {
    uint_t a_as_uint = numext::bit_cast<uint_t, Scalar>(a);
    uint_t b_as_uint = numext::bit_cast<uint_t, Scalar>(b);
    uint_t result = a_as_uint | b_as_uint;
    return numext::bit_cast<Scalar, uint_t>(result);
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_xor(const Scalar& a, const Scalar& b) {
    uint_t a_as_uint = numext::bit_cast<uint_t, Scalar>(a);
    uint_t b_as_uint = numext::bit_cast<uint_t, Scalar>(b);
    uint_t result = a_as_uint ^ b_as_uint;
    return numext::bit_cast<Scalar, uint_t>(result);
  }
};

template <typename Scalar>
struct bitwise_binary_impl<Scalar, true> {
  using Real = typename NumTraits<Scalar>::Real;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_and(const Scalar& a, const Scalar& b) {
    Real real_result = bitwise_binary_impl<Real>::run_and(numext::real(a), numext::real(b));
    Real imag_result = bitwise_binary_impl<Real>::run_and(numext::imag(a), numext::imag(b));
    return Scalar(real_result, imag_result);
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_or(const Scalar& a, const Scalar& b) {
    Real real_result = bitwise_binary_impl<Real>::run_or(numext::real(a), numext::real(b));
    Real imag_result = bitwise_binary_impl<Real>::run_or(numext::imag(a), numext::imag(b));
    return Scalar(real_result, imag_result);
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_xor(const Scalar& a, const Scalar& b) {
    Real real_result = bitwise_binary_impl<Real>::run_xor(numext::real(a), numext::real(b));
    Real imag_result = bitwise_binary_impl<Real>::run_xor(numext::imag(a), numext::imag(b));
    return Scalar(real_result, imag_result);
  }
};

/** \internal
 * \brief Template functor to compute the bitwise and of two scalars
 *
 * \sa class CwiseBinaryOp, ArrayBase::operator&
 */
template <typename Scalar>
struct scalar_bitwise_and_op {
  EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::RequireInitialization,
                      BITWISE OPERATIONS MAY ONLY BE PERFORMED ON PLAIN DATA TYPES)
  EIGEN_STATIC_ASSERT((!internal::is_same<Scalar, bool>::value), DONT USE BITWISE OPS ON BOOLEAN TYPES)
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
    return bitwise_binary_impl<Scalar>::run_and(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return pand(a, b);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bitwise_and_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = true };
};

/** \internal
 * \brief Template functor to compute the bitwise or of two scalars
 *
 * \sa class CwiseBinaryOp, ArrayBase::operator|
 */
template <typename Scalar>
struct scalar_bitwise_or_op {
  EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::RequireInitialization,
                      BITWISE OPERATIONS MAY ONLY BE PERFORMED ON PLAIN DATA TYPES)
  EIGEN_STATIC_ASSERT((!internal::is_same<Scalar, bool>::value), DONT USE BITWISE OPS ON BOOLEAN TYPES)
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
    return bitwise_binary_impl<Scalar>::run_or(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return por(a, b);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bitwise_or_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = true };
};

/** \internal
 * \brief Template functor to compute the bitwise xor of two scalars
 *
 * \sa class CwiseBinaryOp, ArrayBase::operator^
 */
template <typename Scalar>
struct scalar_bitwise_xor_op {
  EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::RequireInitialization,
                      BITWISE OPERATIONS MAY ONLY BE PERFORMED ON PLAIN DATA TYPES)
  EIGEN_STATIC_ASSERT((!internal::is_same<Scalar, bool>::value), DONT USE BITWISE OPS ON BOOLEAN TYPES)
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
    return bitwise_binary_impl<Scalar>::run_xor(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a, const Packet& b) const {
    return pxor(a, b);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bitwise_xor_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = true };
};

/** \internal
 * \brief Template functor to compute the absolute difference of two scalars
 *
 * \sa class CwiseBinaryOp, MatrixBase::absolute_difference
 */
template <typename LhsScalar, typename RhsScalar>
struct scalar_absolute_difference_op : binary_op_base<LhsScalar, RhsScalar> {
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar, scalar_absolute_difference_op>::ReturnType result_type;
#ifdef EIGEN_SCALAR_BINARY_OP_PLUGIN
  scalar_absolute_difference_op(){EIGEN_SCALAR_BINARY_OP_PLUGIN}
#endif
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type
  operator()(const LhsScalar& a, const RhsScalar& b) const {
    return numext::absdiff(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const {
    return internal::pabsdiff(a, b);
  }
};
template <typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_absolute_difference_op<LhsScalar, RhsScalar>> {
  enum {
    Cost = (NumTraits<LhsScalar>::AddCost + NumTraits<RhsScalar>::AddCost) / 2,
    PacketAccess = is_same<LhsScalar, RhsScalar>::value && packet_traits<LhsScalar>::HasAbsDiff
  };
};

template <typename LhsScalar, typename RhsScalar>
struct scalar_atan2_op {
  using Scalar = LhsScalar;

  static constexpr bool Enable =
      is_same<LhsScalar, RhsScalar>::value && !NumTraits<Scalar>::IsInteger && !NumTraits<Scalar>::IsComplex;
  EIGEN_STATIC_ASSERT(Enable, "LhsScalar and RhsScalar must be the same non-integer, non-complex type")

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& y, const Scalar& x) const {
    return numext::atan2(y, x);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& y, const Packet& x) const {
    return internal::patan2(y, x);
  }
};

template <typename LhsScalar, typename RhsScalar>
struct functor_traits<scalar_atan2_op<LhsScalar, RhsScalar>> {
  using Scalar = LhsScalar;
  enum {
    PacketAccess = is_same<LhsScalar, RhsScalar>::value && packet_traits<Scalar>::HasATan &&
                   packet_traits<Scalar>::HasDiv && !NumTraits<Scalar>::IsInteger && !NumTraits<Scalar>::IsComplex,
    Cost = int(scalar_div_cost<Scalar, PacketAccess>::value) + int(functor_traits<scalar_atan_op<Scalar>>::Cost)
  };
};

//---------- binary functors bound to a constant, thus appearing as a unary functor ----------

// The following two classes permits to turn any binary functor into a unary one with one argument bound to a constant
// value. They are analogues to std::binder1st/binder2nd but with the following differences:
//  - they are compatible with packetOp
//  - they are portable across C++ versions (the std::binder* are deprecated in C++11)
template <typename BinaryOp>
struct bind1st_op : BinaryOp {
  typedef typename BinaryOp::first_argument_type first_argument_type;
  typedef typename BinaryOp::second_argument_type second_argument_type;
  typedef typename BinaryOp::result_type result_type;

  EIGEN_DEVICE_FUNC explicit bind1st_op(const first_argument_type& val) : m_value(val) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator()(const second_argument_type& b) const {
    return BinaryOp::operator()(m_value, b);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& b) const {
    return BinaryOp::packetOp(internal::pset1<Packet>(m_value), b);
  }

  first_argument_type m_value;
};
template <typename BinaryOp>
struct functor_traits<bind1st_op<BinaryOp>> : functor_traits<BinaryOp> {};

template <typename BinaryOp>
struct bind2nd_op : BinaryOp {
  typedef typename BinaryOp::first_argument_type first_argument_type;
  typedef typename BinaryOp::second_argument_type second_argument_type;
  typedef typename BinaryOp::result_type result_type;

  EIGEN_DEVICE_FUNC explicit bind2nd_op(const second_argument_type& val) : m_value(val) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator()(const first_argument_type& a) const {
    return BinaryOp::operator()(a, m_value);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return BinaryOp::packetOp(a, internal::pset1<Packet>(m_value));
  }

  second_argument_type m_value;
};
template <typename BinaryOp>
struct functor_traits<bind2nd_op<BinaryOp>> : functor_traits<BinaryOp> {};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_BINARY_FUNCTORS_H
