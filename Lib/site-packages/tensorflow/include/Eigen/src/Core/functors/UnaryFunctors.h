// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_UNARY_FUNCTORS_H
#define EIGEN_UNARY_FUNCTORS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal
 * \brief Template functor to compute the opposite of a scalar
 *
 * \sa class CwiseUnaryOp, MatrixBase::operator-
 */
template <typename Scalar>
struct scalar_opposite_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const { return numext::negate(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return internal::pnegate(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_opposite_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasNegate };
};

/** \internal
 * \brief Template functor to compute the absolute value of a scalar
 *
 * \sa class CwiseUnaryOp, Cwise::abs
 */
template <typename Scalar>
struct scalar_abs_op {
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator()(const Scalar& a) const { return numext::abs(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return internal::pabs(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_abs_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasAbs };
};

/** \internal
 * \brief Template functor to compute the score of a scalar, to chose a pivot
 *
 * \sa class CwiseUnaryOp
 */
template <typename Scalar>
struct scalar_score_coeff_op : scalar_abs_op<Scalar> {
  typedef void Score_is_abs;
};
template <typename Scalar>
struct functor_traits<scalar_score_coeff_op<Scalar>> : functor_traits<scalar_abs_op<Scalar>> {};

/* Avoid recomputing abs when we know the score and they are the same. Not a true Eigen functor.  */
template <typename Scalar, typename = void>
struct abs_knowing_score {
  typedef typename NumTraits<Scalar>::Real result_type;
  template <typename Score>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator()(const Scalar& a, const Score&) const {
    return numext::abs(a);
  }
};
template <typename Scalar>
struct abs_knowing_score<Scalar, typename scalar_score_coeff_op<Scalar>::Score_is_abs> {
  typedef typename NumTraits<Scalar>::Real result_type;
  template <typename Scal>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator()(const Scal&, const result_type& a) const {
    return a;
  }
};

/** \internal
 * \brief Template functor to compute the squared absolute value of a scalar
 *
 * \sa class CwiseUnaryOp, Cwise::abs2
 */
template <typename Scalar>
struct scalar_abs2_op {
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator()(const Scalar& a) const { return numext::abs2(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return internal::pmul(a, a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_abs2_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasAbs2 };
};

/** \internal
 * \brief Template functor to compute the conjugate of a complex value
 *
 * \sa class CwiseUnaryOp, MatrixBase::conjugate()
 */
template <typename Scalar>
struct scalar_conjugate_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const { return numext::conj(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return internal::pconj(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_conjugate_op<Scalar>> {
  enum {
    Cost = 0,
    // Yes the cost is zero even for complexes because in most cases for which
    // the cost is used, conjugation turns to be a no-op. Some examples:
    //   cost(a*conj(b)) == cost(a*b)
    //   cost(a+conj(b)) == cost(a+b)
    //   <etc.
    // If we don't set it to zero, then:
    //   A.conjugate().lazyProduct(B.conjugate())
    // will bake its operands. We definitely don't want that!
    PacketAccess = packet_traits<Scalar>::HasConj
  };
};

/** \internal
 * \brief Template functor to compute the phase angle of a complex
 *
 * \sa class CwiseUnaryOp, Cwise::arg
 */
template <typename Scalar>
struct scalar_arg_op {
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type operator()(const Scalar& a) const { return numext::arg(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return internal::parg(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_arg_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::IsComplex ? 5 * NumTraits<Scalar>::MulCost : NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasArg
  };
};

/** \internal
 * \brief Template functor to compute the complex argument, returned as a complex type
 *
 * \sa class CwiseUnaryOp, Cwise::carg
 */
template <typename Scalar>
struct scalar_carg_op {
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const {
    return Scalar(numext::arg(a));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return pcarg(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_carg_op<Scalar>> {
  using RealScalar = typename NumTraits<Scalar>::Real;
  enum { Cost = functor_traits<scalar_atan2_op<RealScalar>>::Cost, PacketAccess = packet_traits<RealScalar>::HasATan };
};

/** \internal
 * \brief Template functor to cast a scalar to another type
 *
 * \sa class CwiseUnaryOp, MatrixBase::cast()
 */
template <typename Scalar, typename NewType>
struct scalar_cast_op {
  typedef NewType result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const NewType operator()(const Scalar& a) const {
    return cast<Scalar, NewType>(a);
  }
};

template <typename Scalar, typename NewType>
struct functor_traits<scalar_cast_op<Scalar, NewType>> {
  enum { Cost = is_same<Scalar, NewType>::value ? 0 : NumTraits<NewType>::AddCost, PacketAccess = false };
};

/** \internal
 * `core_cast_op` serves to distinguish the vectorized implementation from that of the legacy `scalar_cast_op` for
 * backwards compatibility. The manner in which packet ops are handled is defined by the specialized unary_evaluator:
 * `unary_evaluator<CwiseUnaryOp<core_cast_op<SrcType, DstType>, ArgType>, IndexBased>` in CoreEvaluators.h
 * Otherwise, the non-vectorized behavior is identical to that of `scalar_cast_op`
 */
template <typename SrcType, typename DstType>
struct core_cast_op : scalar_cast_op<SrcType, DstType> {};

template <typename SrcType, typename DstType>
struct functor_traits<core_cast_op<SrcType, DstType>> {
  using CastingTraits = type_casting_traits<SrcType, DstType>;
  enum {
    Cost = is_same<SrcType, DstType>::value ? 0 : NumTraits<DstType>::AddCost,
    PacketAccess = CastingTraits::VectorizedCast && (CastingTraits::SrcCoeffRatio <= 8)
  };
};

/** \internal
 * \brief Template functor to arithmetically shift a scalar right by a number of bits
 *
 * \sa class CwiseUnaryOp, MatrixBase::shift_right()
 */
template <typename Scalar, int N>
struct scalar_shift_right_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const {
    return numext::arithmetic_shift_right(a);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return internal::parithmetic_shift_right<N>(a);
  }
};
template <typename Scalar, int N>
struct functor_traits<scalar_shift_right_op<Scalar, N>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasShift };
};

/** \internal
 * \brief Template functor to logically shift a scalar left by a number of bits
 *
 * \sa class CwiseUnaryOp, MatrixBase::shift_left()
 */
template <typename Scalar, int N>
struct scalar_shift_left_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const {
    return numext::logical_shift_left(a);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const {
    return internal::plogical_shift_left<N>(a);
  }
};
template <typename Scalar, int N>
struct functor_traits<scalar_shift_left_op<Scalar, N>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasShift };
};

/** \internal
 * \brief Template functor to extract the real part of a complex
 *
 * \sa class CwiseUnaryOp, MatrixBase::real()
 */
template <typename Scalar>
struct scalar_real_op {
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const Scalar& a) const { return numext::real(a); }
};
template <typename Scalar>
struct functor_traits<scalar_real_op<Scalar>> {
  enum { Cost = 0, PacketAccess = false };
};

/** \internal
 * \brief Template functor to extract the imaginary part of a complex
 *
 * \sa class CwiseUnaryOp, MatrixBase::imag()
 */
template <typename Scalar>
struct scalar_imag_op {
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const Scalar& a) const { return numext::imag(a); }
};
template <typename Scalar>
struct functor_traits<scalar_imag_op<Scalar>> {
  enum { Cost = 0, PacketAccess = false };
};

/** \internal
 * \brief Template functor to extract the real part of a complex as a reference
 *
 * \sa class CwiseUnaryOp, MatrixBase::real()
 */
template <typename Scalar>
struct scalar_real_ref_op {
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type& operator()(const Scalar& a) const {
    return numext::real_ref(a);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type& operator()(Scalar& a) const { return numext::real_ref(a); }
};
template <typename Scalar>
struct functor_traits<scalar_real_ref_op<Scalar>> {
  enum { Cost = 0, PacketAccess = false };
};

/** \internal
 * \brief Template functor to extract the imaginary part of a complex as a reference
 *
 * \sa class CwiseUnaryOp, MatrixBase::imag()
 */
template <typename Scalar>
struct scalar_imag_ref_op {
  typedef typename NumTraits<Scalar>::Real result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type& operator()(Scalar& a) const { return numext::imag_ref(a); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const result_type& operator()(const Scalar& a) const {
    return numext::imag_ref(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_imag_ref_op<Scalar>> {
  enum { Cost = 0, PacketAccess = false };
};

/** \internal
 *
 * \brief Template functor to compute the exponential of a scalar
 *
 * \sa class CwiseUnaryOp, Cwise::exp()
 */
template <typename Scalar>
struct scalar_exp_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return internal::pexp(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pexp(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_exp_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasExp,
  // The following numbers are based on the AVX implementation.
#ifdef EIGEN_VECTORIZE_FMA
    // Haswell can issue 2 add/mul/madd per cycle.
    Cost = (sizeof(Scalar) == 4
                // float: 8 pmadd, 4 pmul, 2 padd/psub, 6 other
                ? (8 * NumTraits<Scalar>::AddCost + 6 * NumTraits<Scalar>::MulCost)
                // double: 7 pmadd, 5 pmul, 3 padd/psub, 1 div,  13 other
                : (14 * NumTraits<Scalar>::AddCost + 6 * NumTraits<Scalar>::MulCost +
                   scalar_div_cost<Scalar, packet_traits<Scalar>::HasDiv>::value))
#else
    Cost = (sizeof(Scalar) == 4
                // float: 7 pmadd, 6 pmul, 4 padd/psub, 10 other
                ? (21 * NumTraits<Scalar>::AddCost + 13 * NumTraits<Scalar>::MulCost)
                // double: 7 pmadd, 5 pmul, 3 padd/psub, 1 div,  13 other
                : (23 * NumTraits<Scalar>::AddCost + 12 * NumTraits<Scalar>::MulCost +
                   scalar_div_cost<Scalar, packet_traits<Scalar>::HasDiv>::value))
#endif
  };
};

/** \internal
 *
 * \brief Template functor to compute the exponential of a scalar - 1.
 *
 * \sa class CwiseUnaryOp, ArrayBase::expm1()
 */
template <typename Scalar>
struct scalar_expm1_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::expm1(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pexpm1(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_expm1_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasExpm1,
    Cost = functor_traits<scalar_exp_op<Scalar>>::Cost  // TODO measure cost of expm1
  };
};

/** \internal
 *
 * \brief Template functor to compute the logarithm of a scalar
 *
 * \sa class CwiseUnaryOp, ArrayBase::log()
 */
template <typename Scalar>
struct scalar_log_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::log(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::plog(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_log_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasLog,
    Cost = (PacketAccess
  // The following numbers are based on the AVX implementation.
#ifdef EIGEN_VECTORIZE_FMA
                // 8 pmadd, 6 pmul, 8 padd/psub, 16 other, can issue 2 add/mul/madd per cycle.
                ? (20 * NumTraits<Scalar>::AddCost + 7 * NumTraits<Scalar>::MulCost)
#else
                // 8 pmadd, 6 pmul, 8 padd/psub, 20 other
                ? (36 * NumTraits<Scalar>::AddCost + 14 * NumTraits<Scalar>::MulCost)
#endif
                // Measured cost of std::log.
                : sizeof(Scalar) == 4 ? 40 : 85)
  };
};

/** \internal
 *
 * \brief Template functor to compute the logarithm of 1 plus a scalar value
 *
 * \sa class CwiseUnaryOp, ArrayBase::log1p()
 */
template <typename Scalar>
struct scalar_log1p_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::log1p(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::plog1p(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_log1p_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasLog1p,
    Cost = functor_traits<scalar_log_op<Scalar>>::Cost  // TODO measure cost of log1p
  };
};

/** \internal
 *
 * \brief Template functor to compute the base-10 logarithm of a scalar
 *
 * \sa class CwiseUnaryOp, Cwise::log10()
 */
template <typename Scalar>
struct scalar_log10_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { EIGEN_USING_STD(log10) return log10(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::plog10(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_log10_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasLog10 };
};

/** \internal
 *
 * \brief Template functor to compute the base-2 logarithm of a scalar
 *
 * \sa class CwiseUnaryOp, Cwise::log2()
 */
template <typename Scalar>
struct scalar_log2_op {
  using RealScalar = typename NumTraits<Scalar>::Real;
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const {
    return Scalar(RealScalar(EIGEN_LOG2E)) * numext::log(a);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::plog2(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_log2_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasLog };
};

/** \internal
 * \brief Template functor to compute the square root of a scalar
 * \sa class CwiseUnaryOp, Cwise::sqrt()
 */
template <typename Scalar>
struct scalar_sqrt_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::sqrt(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::psqrt(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_sqrt_op<Scalar>> {
  enum {
#if EIGEN_FAST_MATH
    // The following numbers are based on the AVX implementation.
    Cost = (sizeof(Scalar) == 8 ? 28
                                // 4 pmul, 1 pmadd, 3 other
                                : (3 * NumTraits<Scalar>::AddCost + 5 * NumTraits<Scalar>::MulCost)),
#else
    // The following numbers are based on min VSQRT throughput on Haswell.
    Cost = (sizeof(Scalar) == 8 ? 28 : 14),
#endif
    PacketAccess = packet_traits<Scalar>::HasSqrt
  };
};

// Boolean specialization to eliminate -Wimplicit-conversion-floating-point-to-bool warnings.
template <>
struct scalar_sqrt_op<bool> {
  EIGEN_DEPRECATED EIGEN_DEVICE_FUNC inline bool operator()(const bool& a) const { return a; }
  template <typename Packet>
  EIGEN_DEPRECATED EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return a;
  }
};
template <>
struct functor_traits<scalar_sqrt_op<bool>> {
  enum { Cost = 1, PacketAccess = packet_traits<bool>::Vectorizable };
};

/** \internal
 * \brief Template functor to compute the cube root of a scalar
 * \sa class CwiseUnaryOp, Cwise::sqrt()
 */
template <typename Scalar>
struct scalar_cbrt_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::cbrt(a); }
};

template <typename Scalar>
struct functor_traits<scalar_cbrt_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};

/** \internal
 * \brief Template functor to compute the reciprocal square root of a scalar
 * \sa class CwiseUnaryOp, Cwise::rsqrt()
 */
template <typename Scalar>
struct scalar_rsqrt_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::rsqrt(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::prsqrt(a);
  }
};

template <typename Scalar>
struct functor_traits<scalar_rsqrt_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasRsqrt };
};

/** \internal
 * \brief Template functor to compute the cosine of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::cos()
 */
template <typename Scalar>
struct scalar_cos_op {
  EIGEN_DEVICE_FUNC inline Scalar operator()(const Scalar& a) const { return numext::cos(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pcos(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_cos_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasCos };
};

/** \internal
 * \brief Template functor to compute the sine of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::sin()
 */
template <typename Scalar>
struct scalar_sin_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::sin(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::psin(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_sin_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasSin };
};

/** \internal
 * \brief Template functor to compute the tan of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::tan()
 */
template <typename Scalar>
struct scalar_tan_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::tan(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::ptan(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_tan_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasTan };
};

/** \internal
 * \brief Template functor to compute the arc cosine of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::acos()
 */
template <typename Scalar>
struct scalar_acos_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::acos(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pacos(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_acos_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasACos };
};

/** \internal
 * \brief Template functor to compute the arc sine of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::asin()
 */
template <typename Scalar>
struct scalar_asin_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::asin(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pasin(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_asin_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasASin };
};

/** \internal
 * \brief Template functor to compute the atan of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::atan()
 */
template <typename Scalar>
struct scalar_atan_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::atan(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::patan(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_atan_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasATan };
};

/** \internal
 * \brief Template functor to compute the tanh of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::tanh()
 */
template <typename Scalar>
struct scalar_tanh_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::tanh(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    return ptanh(x);
  }
};

template <typename Scalar>
struct functor_traits<scalar_tanh_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasTanh,
    Cost = ((EIGEN_FAST_MATH && is_same<Scalar, float>::value)
// The following numbers are based on the AVX implementation,
#ifdef EIGEN_VECTORIZE_FMA
                // Haswell can issue 2 add/mul/madd per cycle.
                // 9 pmadd, 2 pmul, 1 div, 2 other
                ? (2 * NumTraits<Scalar>::AddCost + 6 * NumTraits<Scalar>::MulCost +
                   scalar_div_cost<Scalar, packet_traits<Scalar>::HasDiv>::value)
#else
                ? (11 * NumTraits<Scalar>::AddCost + 11 * NumTraits<Scalar>::MulCost +
                   scalar_div_cost<Scalar, packet_traits<Scalar>::HasDiv>::value)
#endif
                // This number assumes a naive implementation of tanh
                : (6 * NumTraits<Scalar>::AddCost + 3 * NumTraits<Scalar>::MulCost +
                   2 * scalar_div_cost<Scalar, packet_traits<Scalar>::HasDiv>::value +
                   functor_traits<scalar_exp_op<Scalar>>::Cost))
  };
};

/** \internal
 * \brief Template functor to compute the atanh of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::atanh()
 */
template <typename Scalar>
struct scalar_atanh_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::atanh(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    return patanh(x);
  }
};

template <typename Scalar>
struct functor_traits<scalar_atanh_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasATanh };
};

/** \internal
 * \brief Template functor to compute the sinh of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::sinh()
 */
template <typename Scalar>
struct scalar_sinh_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::sinh(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::psinh(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_sinh_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasSinh };
};

/** \internal
 * \brief Template functor to compute the asinh of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::asinh()
 */
template <typename Scalar>
struct scalar_asinh_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::asinh(a); }
};

template <typename Scalar>
struct functor_traits<scalar_asinh_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};

/** \internal
 * \brief Template functor to compute the cosh of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::cosh()
 */
template <typename Scalar>
struct scalar_cosh_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::cosh(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pcosh(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_cosh_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasCosh };
};

/** \internal
 * \brief Template functor to compute the acosh of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::acosh()
 */
template <typename Scalar>
struct scalar_acosh_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::acosh(a); }
};

template <typename Scalar>
struct functor_traits<scalar_acosh_op<Scalar>> {
  enum { Cost = 5 * NumTraits<Scalar>::MulCost, PacketAccess = false };
};

/** \internal
 * \brief Template functor to compute the inverse of a scalar
 * \sa class CwiseUnaryOp, Cwise::inverse()
 */
template <typename Scalar>
struct scalar_inverse_op {
  EIGEN_DEVICE_FUNC inline Scalar operator()(const Scalar& a) const { return Scalar(1) / a; }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const {
    return internal::preciprocal(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_inverse_op<Scalar>> {
  enum {
    PacketAccess = packet_traits<Scalar>::HasDiv,
    // If packet_traits<Scalar>::HasReciprocal then the Estimated cost is that
    // of computing an approximation plus a single Newton-Raphson step, which
    // consists of 1 pmul + 1 pmadd.
    Cost = (packet_traits<Scalar>::HasReciprocal ? 4 * NumTraits<Scalar>::MulCost
                                                 : scalar_div_cost<Scalar, PacketAccess>::value)
  };
};

/** \internal
 * \brief Template functor to compute the square of a scalar
 * \sa class CwiseUnaryOp, Cwise::square()
 */
template <typename Scalar>
struct scalar_square_op {
  EIGEN_DEVICE_FUNC inline Scalar operator()(const Scalar& a) const { return a * a; }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const {
    return internal::pmul(a, a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_square_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasMul };
};

// Boolean specialization to avoid -Wint-in-bool-context warnings on GCC.
template <>
struct scalar_square_op<bool> {
  EIGEN_DEPRECATED EIGEN_DEVICE_FUNC inline bool operator()(const bool& a) const { return a; }
  template <typename Packet>
  EIGEN_DEPRECATED EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const {
    return a;
  }
};
template <>
struct functor_traits<scalar_square_op<bool>> {
  enum { Cost = 0, PacketAccess = packet_traits<bool>::Vectorizable };
};

/** \internal
 * \brief Template functor to compute the cube of a scalar
 * \sa class CwiseUnaryOp, Cwise::cube()
 */
template <typename Scalar>
struct scalar_cube_op {
  EIGEN_DEVICE_FUNC inline Scalar operator()(const Scalar& a) const { return a * a * a; }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const {
    return internal::pmul(a, pmul(a, a));
  }
};
template <typename Scalar>
struct functor_traits<scalar_cube_op<Scalar>> {
  enum { Cost = 2 * NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasMul };
};

// Boolean specialization to avoid -Wint-in-bool-context warnings on GCC.
template <>
struct scalar_cube_op<bool> {
  EIGEN_DEPRECATED EIGEN_DEVICE_FUNC inline bool operator()(const bool& a) const { return a; }
  template <typename Packet>
  EIGEN_DEPRECATED EIGEN_DEVICE_FUNC inline const Packet packetOp(const Packet& a) const {
    return a;
  }
};
template <>
struct functor_traits<scalar_cube_op<bool>> {
  enum { Cost = 0, PacketAccess = packet_traits<bool>::Vectorizable };
};

/** \internal
 * \brief Template functor to compute the rounded value of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::round()
 */
template <typename Scalar>
struct scalar_round_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const { return numext::round(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pround(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_round_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasRound || NumTraits<Scalar>::IsInteger
  };
};

/** \internal
 * \brief Template functor to compute the floor of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::floor()
 */
template <typename Scalar>
struct scalar_floor_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const { return numext::floor(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pfloor(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_floor_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasRound || NumTraits<Scalar>::IsInteger
  };
};

/** \internal
 * \brief Template functor to compute the rounded (with current rounding mode)  value of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::rint()
 */
template <typename Scalar>
struct scalar_rint_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const { return numext::rint(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::print(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_rint_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasRound || NumTraits<Scalar>::IsInteger
  };
};

/** \internal
 * \brief Template functor to compute the ceil of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::ceil()
 */
template <typename Scalar>
struct scalar_ceil_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const { return numext::ceil(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::pceil(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_ceil_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasRound || NumTraits<Scalar>::IsInteger
  };
};

/** \internal
 * \brief Template functor to compute the truncation of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::floor()
 */
template <typename Scalar>
struct scalar_trunc_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& a) const { return numext::trunc(a); }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::ptrunc(a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_trunc_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasRound || NumTraits<Scalar>::IsInteger
  };
};

/** \internal
 * \brief Template functor to compute whether a scalar is NaN
 * \sa class CwiseUnaryOp, ArrayBase::isnan()
 */
template <typename Scalar, bool UseTypedPredicate = false>
struct scalar_isnan_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& a) const {
#if defined(SYCL_DEVICE_ONLY)
    return numext::isnan(a);
#else
    return numext::isnan EIGEN_NOT_A_MACRO(a);
#endif
  }
};

template <typename Scalar>
struct scalar_isnan_op<Scalar, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
#if defined(SYCL_DEVICE_ONLY)
    return (numext::isnan(a) ? ptrue(a) : pzero(a));
#else
    return (numext::isnan EIGEN_NOT_A_MACRO(a) ? ptrue(a) : pzero(a));
#endif
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return pisnan(a);
  }
};

template <typename Scalar, bool UseTypedPredicate>
struct functor_traits<scalar_isnan_op<Scalar, UseTypedPredicate>> {
  enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasCmp && UseTypedPredicate };
};

/** \internal
 * \brief Template functor to check whether a scalar is +/-inf
 * \sa class CwiseUnaryOp, ArrayBase::isinf()
 */
template <typename Scalar, bool UseTypedPredicate = false>
struct scalar_isinf_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& a) const {
#if defined(SYCL_DEVICE_ONLY)
    return numext::isinf(a);
#else
    return (numext::isinf)(a);
#endif
  }
};

template <typename Scalar>
struct scalar_isinf_op<Scalar, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
#if defined(SYCL_DEVICE_ONLY)
    return (numext::isinf(a) ? ptrue(a) : pzero(a));
#else
    return (numext::isinf EIGEN_NOT_A_MACRO(a) ? ptrue(a) : pzero(a));
#endif
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return pisinf(a);
  }
};
template <typename Scalar, bool UseTypedPredicate>
struct functor_traits<scalar_isinf_op<Scalar, UseTypedPredicate>> {
  enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasCmp && UseTypedPredicate };
};

/** \internal
 * \brief Template functor to check whether a scalar has a finite value
 * \sa class CwiseUnaryOp, ArrayBase::isfinite()
 */
template <typename Scalar, bool UseTypedPredicate = false>
struct scalar_isfinite_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const Scalar& a) const {
#if defined(SYCL_DEVICE_ONLY)
    return numext::isfinite(a);
#else
    return (numext::isfinite)(a);
#endif
  }
};

template <typename Scalar>
struct scalar_isfinite_op<Scalar, true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
#if defined(SYCL_DEVICE_ONLY)
    return (numext::isfinite(a) ? ptrue(a) : pzero(a));
#else
    return (numext::isfinite EIGEN_NOT_A_MACRO(a) ? ptrue(a) : pzero(a));
#endif
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    constexpr Scalar inf = NumTraits<Scalar>::infinity();
    return pcmp_lt(pabs(a), pset1<Packet>(inf));
  }
};
template <typename Scalar, bool UseTypedPredicate>
struct functor_traits<scalar_isfinite_op<Scalar, UseTypedPredicate>> {
  enum { Cost = NumTraits<Scalar>::MulCost, PacketAccess = packet_traits<Scalar>::HasCmp && UseTypedPredicate };
};

/** \internal
 * \brief Template functor to compute the logical not of a scalar as if it were a boolean
 *
 * \sa class CwiseUnaryOp, ArrayBase::operator!
 */
template <typename Scalar>
struct scalar_boolean_not_op {
  using result_type = Scalar;
  // `false` any value `a` that satisfies `a == Scalar(0)`
  // `true` is the complement of `false`
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    return a == Scalar(0) ? Scalar(1) : Scalar(0);
  }
  template <typename Packet>
  EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    const Packet cst_one = pset1<Packet>(Scalar(1));
    Packet not_a = pcmp_eq(a, pzero(a));
    return pand(not_a, cst_one);
  }
};
template <typename Scalar>
struct functor_traits<scalar_boolean_not_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = packet_traits<Scalar>::HasCmp };
};

template <typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct bitwise_unary_impl {
  static constexpr size_t Size = sizeof(Scalar);
  using uint_t = typename numext::get_integer_by_size<Size>::unsigned_type;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_not(const Scalar& a) {
    uint_t a_as_uint = numext::bit_cast<uint_t, Scalar>(a);
    uint_t result = ~a_as_uint;
    return numext::bit_cast<Scalar, uint_t>(result);
  }
};

template <typename Scalar>
struct bitwise_unary_impl<Scalar, true> {
  using Real = typename NumTraits<Scalar>::Real;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run_not(const Scalar& a) {
    Real real_result = bitwise_unary_impl<Real>::run_not(numext::real(a));
    Real imag_result = bitwise_unary_impl<Real>::run_not(numext::imag(a));
    return Scalar(real_result, imag_result);
  }
};

/** \internal
 * \brief Template functor to compute the bitwise not of a scalar
 *
 * \sa class CwiseUnaryOp, ArrayBase::operator~
 */
template <typename Scalar>
struct scalar_bitwise_not_op {
  EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::RequireInitialization,
                      BITWISE OPERATIONS MAY ONLY BE PERFORMED ON PLAIN DATA TYPES)
  EIGEN_STATIC_ASSERT((!internal::is_same<Scalar, bool>::value), DONT USE BITWISE OPS ON BOOLEAN TYPES)
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    return bitwise_unary_impl<Scalar>::run_not(a);
  }
  template <typename Packet>
  EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return pandnot(ptrue(a), a);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bitwise_not_op<Scalar>> {
  enum { Cost = NumTraits<Scalar>::AddCost, PacketAccess = true };
};

/** \internal
 * \brief Template functor to compute the signum of a scalar
 * \sa class CwiseUnaryOp, Cwise::sign()
 */
template <typename Scalar>
struct scalar_sign_op {
  EIGEN_DEVICE_FUNC inline const Scalar operator()(const Scalar& a) const { return numext::sign(a); }

  template <typename Packet>
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const {
    return internal::psign(a);
  }
};

template <typename Scalar>
struct functor_traits<scalar_sign_op<Scalar>> {
  enum {
    Cost = NumTraits<Scalar>::IsComplex ? (8 * NumTraits<Scalar>::MulCost)  // roughly
                                        : (3 * NumTraits<Scalar>::AddCost),
    PacketAccess = packet_traits<Scalar>::HasSign && packet_traits<Scalar>::Vectorizable
  };
};

// Real-valued implementation.
template <typename T, typename EnableIf = void>
struct scalar_logistic_op_impl {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const { return packetOp(x); }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    const Packet one = pset1<Packet>(T(1));
    const Packet inf = pset1<Packet>(NumTraits<T>::infinity());
    const Packet e = pexp(x);
    const Packet inf_mask = pcmp_eq(e, inf);
    return pselect(inf_mask, one, pdiv(e, padd(one, e)));
  }
};

// Complex-valud implementation.
template <typename T>
struct scalar_logistic_op_impl<T, std::enable_if_t<NumTraits<T>::IsComplex>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    const T e = numext::exp(x);
    return (numext::isinf)(numext::real(e)) ? T(1) : e / (e + T(1));
  }
};

/** \internal
 * \brief Template functor to compute the logistic function of a scalar
 * \sa class CwiseUnaryOp, ArrayBase::logistic()
 */
template <typename T>
struct scalar_logistic_op : scalar_logistic_op_impl<T> {};

// TODO(rmlarsen): Enable the following on host when integer_packet is defined
// for the relevant packet types.
#ifndef EIGEN_GPUCC

/** \internal
 * \brief Template specialization of the logistic function for float.
 * Computes S(x) = exp(x) / (1 + exp(x)), where exp(x) is implemented
 * using an algorithm partly adopted from the implementation of
 * pexp_float. See the individual steps described in the code below.
 * Note that compared to pexp, we use an additional outer multiplicative
 * range reduction step using the identity exp(x) = exp(x/2)^2.
 * This prevert us from having to call ldexp on values that could produce
 * a denormal result, which allows us to call the faster implementation in
 * pldexp_fast_impl<Packet>::run(p, m).
 * The final squaring, however, doubles the error bound on the final
 * approximation. Exhaustive testing shows that we have a worst case error
 * of 4.5 ulps (compared to computing S(x) in double precision), which is
 * acceptable.
 */
template <>
struct scalar_logistic_op<float> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(const float& x) const {
    // Truncate at the first point where the interpolant is exactly one.
    const float cst_exp_hi = 16.6355324f;
    const float e = numext::exp(numext::mini(x, cst_exp_hi));
    return e / (1.0f + e);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& _x) const {
    const Packet cst_zero = pset1<Packet>(0.0f);
    const Packet cst_one = pset1<Packet>(1.0f);
    const Packet cst_half = pset1<Packet>(0.5f);
    // Truncate at the first point where the interpolant is exactly one.
    const Packet cst_exp_hi = pset1<Packet>(16.6355324f);
    const Packet cst_exp_lo = pset1<Packet>(-104.f);

    // Clamp x to the non-trivial range where S(x). Outside this
    // interval the correctly rounded value of S(x) is either zero
    // or one.
    Packet zero_mask = pcmp_lt(_x, cst_exp_lo);
    Packet x = pmin(_x, cst_exp_hi);

    // 1. Multiplicative range reduction:
    // Reduce the range of x by a factor of 2. This avoids having
    // to compute exp(x) accurately where the result is a denormalized
    // value.
    x = pmul(x, cst_half);

    // 2. Subtractive range reduction:
    // Express exp(x) as exp(m*ln(2) + r) = 2^m*exp(r), start by extracting
    // m = floor(x/ln(2) + 0.5), such that x = m*ln(2) + r.
    const Packet cst_cephes_LOG2EF = pset1<Packet>(1.44269504088896341f);
    Packet m = pfloor(pmadd(x, cst_cephes_LOG2EF, cst_half));
    // Get r = x - m*ln(2). We use a trick from Cephes where the term
    // m*ln(2) is subtracted out in two parts, m*C1+m*C2 = m*ln(2),
    // to avoid accumulating truncation errors.
    const Packet cst_cephes_exp_C1 = pset1<Packet>(-0.693359375f);
    const Packet cst_cephes_exp_C2 = pset1<Packet>(2.12194440e-4f);
    Packet r = pmadd(m, cst_cephes_exp_C1, x);
    r = pmadd(m, cst_cephes_exp_C2, r);

    // 3. Compute an approximation to exp(r) using a degree 5 minimax polynomial.
    // We compute even and odd terms separately to increase instruction level
    // parallelism.
    Packet r2 = pmul(r, r);
    const Packet cst_p2 = pset1<Packet>(0.49999141693115234375f);
    const Packet cst_p3 = pset1<Packet>(0.16666877269744873046875f);
    const Packet cst_p4 = pset1<Packet>(4.1898667812347412109375e-2f);
    const Packet cst_p5 = pset1<Packet>(8.33471305668354034423828125e-3f);

    const Packet p_even = pmadd(r2, cst_p4, cst_p2);
    const Packet p_odd = pmadd(r2, cst_p5, cst_p3);
    const Packet p_low = padd(r, cst_one);
    Packet p = pmadd(r, p_odd, p_even);
    p = pmadd(r2, p, p_low);

    // 4. Undo subtractive range reduction exp(m*ln(2) + r) = 2^m * exp(r).
    Packet e = pldexp_fast_impl<Packet>::run(p, m);

    // 5. Undo multiplicative range reduction by using exp(r) = exp(r/2)^2.
    e = pmul(e, e);

    // Return exp(x) / (1 + exp(x))
    return pselect(zero_mask, cst_zero, pdiv(e, padd(cst_one, e)));
  }
};
#endif  // #ifndef EIGEN_GPU_COMPILE_PHASE

template <typename T>
struct functor_traits<scalar_logistic_op<T>> {
  enum {
    // The cost estimate for float here here is for the common(?) case where
    // all arguments are greater than -9.
    Cost = scalar_div_cost<T, packet_traits<T>::HasDiv>::value +
           (internal::is_same<T, float>::value ? NumTraits<T>::AddCost * 15 + NumTraits<T>::MulCost * 11
                                               : NumTraits<T>::AddCost * 2 + functor_traits<scalar_exp_op<T>>::Cost),
    PacketAccess = !NumTraits<T>::IsComplex && packet_traits<T>::HasAdd && packet_traits<T>::HasDiv &&
                   (internal::is_same<T, float>::value
                        ? packet_traits<T>::HasMul && packet_traits<T>::HasMax && packet_traits<T>::HasMin
                        : packet_traits<T>::HasNegate && packet_traits<T>::HasExp)
  };
};

template <typename Scalar, typename ExponentScalar, bool IsBaseInteger = NumTraits<Scalar>::IsInteger,
          bool IsExponentInteger = NumTraits<ExponentScalar>::IsInteger,
          bool IsBaseComplex = NumTraits<Scalar>::IsComplex,
          bool IsExponentComplex = NumTraits<ExponentScalar>::IsComplex>
struct scalar_unary_pow_op {
  typedef typename internal::promote_scalar_arg<
      Scalar, ExponentScalar,
      internal::has_ReturnType<ScalarBinaryOpTraits<Scalar, ExponentScalar, scalar_unary_pow_op>>::value>::type
      PromotedExponent;
  typedef typename ScalarBinaryOpTraits<Scalar, PromotedExponent, scalar_unary_pow_op>::ReturnType result_type;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_unary_pow_op(const ExponentScalar& exponent) : m_exponent(exponent) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type operator()(const Scalar& a) const {
    EIGEN_USING_STD(pow);
    return static_cast<result_type>(pow(a, m_exponent));
  }

 private:
  const ExponentScalar m_exponent;
  scalar_unary_pow_op() {}
};

template <typename T>
constexpr int exponent_digits() {
  return CHAR_BIT * sizeof(T) - NumTraits<T>::digits() - NumTraits<T>::IsSigned;
}

template <typename From, typename To>
struct is_floating_exactly_representable {
  // TODO(rmlarsen): Add radix to NumTraits and enable this check.
  // (NumTraits<To>::radix == NumTraits<From>::radix) &&
  static constexpr bool value =
      (exponent_digits<To>() >= exponent_digits<From>() && NumTraits<To>::digits() >= NumTraits<From>::digits());
};

// Specialization for real, non-integer types, non-complex types.
template <typename Scalar, typename ExponentScalar>
struct scalar_unary_pow_op<Scalar, ExponentScalar, false, false, false, false> {
  template <bool IsExactlyRepresentable = is_floating_exactly_representable<ExponentScalar, Scalar>::value>
  std::enable_if_t<IsExactlyRepresentable, void> check_is_representable() const {}

  // Issue a deprecation warning if we do a narrowing conversion on the exponent.
  template <bool IsExactlyRepresentable = is_floating_exactly_representable<ExponentScalar, Scalar>::value>
  EIGEN_DEPRECATED std::enable_if_t<!IsExactlyRepresentable, void> check_is_representable() const {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_unary_pow_op(const ExponentScalar& exponent)
      : m_exponent(static_cast<Scalar>(exponent)) {
    check_is_representable();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    EIGEN_USING_STD(pow);
    return static_cast<Scalar>(pow(a, m_exponent));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return unary_pow_impl<Packet, Scalar>::run(a, m_exponent);
  }

 private:
  const Scalar m_exponent;
  scalar_unary_pow_op() {}
};

template <typename Scalar, typename ExponentScalar, bool BaseIsInteger>
struct scalar_unary_pow_op<Scalar, ExponentScalar, BaseIsInteger, true, false, false> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_unary_pow_op(const ExponentScalar& exponent) : m_exponent(exponent) {}
  // TODO: error handling logic for complex^real_integer
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    return unary_pow_impl<Scalar, ExponentScalar>::run(a, m_exponent);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return unary_pow_impl<Packet, ExponentScalar>::run(a, m_exponent);
  }

 private:
  const ExponentScalar m_exponent;
  scalar_unary_pow_op() {}
};

template <typename Scalar, typename ExponentScalar>
struct functor_traits<scalar_unary_pow_op<Scalar, ExponentScalar>> {
  enum {
    GenPacketAccess = functor_traits<scalar_pow_op<Scalar, ExponentScalar>>::PacketAccess,
    IntPacketAccess = !NumTraits<Scalar>::IsComplex && packet_traits<Scalar>::HasMul &&
                      (packet_traits<Scalar>::HasDiv || NumTraits<Scalar>::IsInteger) && packet_traits<Scalar>::HasCmp,
    PacketAccess = NumTraits<ExponentScalar>::IsInteger ? IntPacketAccess : (IntPacketAccess && GenPacketAccess),
    Cost = functor_traits<scalar_pow_op<Scalar, ExponentScalar>>::Cost
  };
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_FUNCTORS_H
