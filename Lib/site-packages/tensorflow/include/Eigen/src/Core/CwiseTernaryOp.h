// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CWISE_TERNARY_OP_H
#define EIGEN_CWISE_TERNARY_OP_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
struct traits<CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>> {
  // we must not inherit from traits<Arg1> since it has
  // the potential to cause problems with MSVC
  typedef remove_all_t<Arg1> Ancestor;
  typedef typename traits<Ancestor>::XprKind XprKind;
  enum {
    RowsAtCompileTime = traits<Ancestor>::RowsAtCompileTime,
    ColsAtCompileTime = traits<Ancestor>::ColsAtCompileTime,
    MaxRowsAtCompileTime = traits<Ancestor>::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = traits<Ancestor>::MaxColsAtCompileTime
  };

  // even though we require Arg1, Arg2, and Arg3 to have the same scalar type
  // (see CwiseTernaryOp constructor),
  // we still want to handle the case when the result type is different.
  typedef typename result_of<TernaryOp(const typename Arg1::Scalar&, const typename Arg2::Scalar&,
                                       const typename Arg3::Scalar&)>::type Scalar;

  typedef typename internal::traits<Arg1>::StorageKind StorageKind;
  typedef typename internal::traits<Arg1>::StorageIndex StorageIndex;

  typedef typename Arg1::Nested Arg1Nested;
  typedef typename Arg2::Nested Arg2Nested;
  typedef typename Arg3::Nested Arg3Nested;
  typedef std::remove_reference_t<Arg1Nested> Arg1Nested_;
  typedef std::remove_reference_t<Arg2Nested> Arg2Nested_;
  typedef std::remove_reference_t<Arg3Nested> Arg3Nested_;
  enum { Flags = Arg1Nested_::Flags & RowMajorBit };
};
}  // end namespace internal

template <typename TernaryOp, typename Arg1, typename Arg2, typename Arg3, typename StorageKind>
class CwiseTernaryOpImpl;

/** \class CwiseTernaryOp
 * \ingroup Core_Module
 *
 * \brief Generic expression where a coefficient-wise ternary operator is
 * applied to two expressions
 *
 * \tparam TernaryOp template functor implementing the operator
 * \tparam Arg1Type the type of the first argument
 * \tparam Arg2Type the type of the second argument
 * \tparam Arg3Type the type of the third argument
 *
 * This class represents an expression where a coefficient-wise ternary
 * operator is applied to three expressions.
 * It is the return type of ternary operators, by which we mean only those
 * ternary operators where
 * all three arguments are Eigen expressions.
 * For example, the return type of betainc(matrix1, matrix2, matrix3) is a
 * CwiseTernaryOp.
 *
 * Most of the time, this is the only way that it is used, so you typically
 * don't have to name
 * CwiseTernaryOp types explicitly.
 *
 * \sa MatrixBase::ternaryExpr(const MatrixBase<Argument2> &, const
 * MatrixBase<Argument3> &, const CustomTernaryOp &) const, class CwiseBinaryOp,
 * class CwiseUnaryOp, class CwiseNullaryOp
 */
template <typename TernaryOp, typename Arg1Type, typename Arg2Type, typename Arg3Type>
class CwiseTernaryOp : public CwiseTernaryOpImpl<TernaryOp, Arg1Type, Arg2Type, Arg3Type,
                                                 typename internal::traits<Arg1Type>::StorageKind>,
                       internal::no_assignment_operator {
 public:
  typedef internal::remove_all_t<Arg1Type> Arg1;
  typedef internal::remove_all_t<Arg2Type> Arg2;
  typedef internal::remove_all_t<Arg3Type> Arg3;

  // require the sizes to match
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Arg1, Arg2)
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Arg1, Arg3)

  // The index types should match
  EIGEN_STATIC_ASSERT((internal::is_same<typename internal::traits<Arg1Type>::StorageKind,
                                         typename internal::traits<Arg2Type>::StorageKind>::value),
                      STORAGE_KIND_MUST_MATCH)
  EIGEN_STATIC_ASSERT((internal::is_same<typename internal::traits<Arg1Type>::StorageKind,
                                         typename internal::traits<Arg3Type>::StorageKind>::value),
                      STORAGE_KIND_MUST_MATCH)

  typedef typename CwiseTernaryOpImpl<TernaryOp, Arg1Type, Arg2Type, Arg3Type,
                                      typename internal::traits<Arg1Type>::StorageKind>::Base Base;
  EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseTernaryOp)

  typedef typename internal::ref_selector<Arg1Type>::type Arg1Nested;
  typedef typename internal::ref_selector<Arg2Type>::type Arg2Nested;
  typedef typename internal::ref_selector<Arg3Type>::type Arg3Nested;
  typedef std::remove_reference_t<Arg1Nested> Arg1Nested_;
  typedef std::remove_reference_t<Arg2Nested> Arg2Nested_;
  typedef std::remove_reference_t<Arg3Nested> Arg3Nested_;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CwiseTernaryOp(const Arg1& a1, const Arg2& a2, const Arg3& a3,
                                                       const TernaryOp& func = TernaryOp())
      : m_arg1(a1), m_arg2(a2), m_arg3(a3), m_functor(func) {
    eigen_assert(a1.rows() == a2.rows() && a1.cols() == a2.cols() && a1.rows() == a3.rows() && a1.cols() == a3.cols());
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index rows() const {
    // return the fixed size type if available to enable compile time
    // optimizations
    if (internal::traits<internal::remove_all_t<Arg1Nested>>::RowsAtCompileTime == Dynamic &&
        internal::traits<internal::remove_all_t<Arg2Nested>>::RowsAtCompileTime == Dynamic)
      return m_arg3.rows();
    else if (internal::traits<internal::remove_all_t<Arg1Nested>>::RowsAtCompileTime == Dynamic &&
             internal::traits<internal::remove_all_t<Arg3Nested>>::RowsAtCompileTime == Dynamic)
      return m_arg2.rows();
    else
      return m_arg1.rows();
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index cols() const {
    // return the fixed size type if available to enable compile time
    // optimizations
    if (internal::traits<internal::remove_all_t<Arg1Nested>>::ColsAtCompileTime == Dynamic &&
        internal::traits<internal::remove_all_t<Arg2Nested>>::ColsAtCompileTime == Dynamic)
      return m_arg3.cols();
    else if (internal::traits<internal::remove_all_t<Arg1Nested>>::ColsAtCompileTime == Dynamic &&
             internal::traits<internal::remove_all_t<Arg3Nested>>::ColsAtCompileTime == Dynamic)
      return m_arg2.cols();
    else
      return m_arg1.cols();
  }

  /** \returns the first argument nested expression */
  EIGEN_DEVICE_FUNC const Arg1Nested_& arg1() const { return m_arg1; }
  /** \returns the first argument nested expression */
  EIGEN_DEVICE_FUNC const Arg2Nested_& arg2() const { return m_arg2; }
  /** \returns the third argument nested expression */
  EIGEN_DEVICE_FUNC const Arg3Nested_& arg3() const { return m_arg3; }
  /** \returns the functor representing the ternary operation */
  EIGEN_DEVICE_FUNC const TernaryOp& functor() const { return m_functor; }

 protected:
  Arg1Nested m_arg1;
  Arg2Nested m_arg2;
  Arg3Nested m_arg3;
  const TernaryOp m_functor;
};

// Generic API dispatcher
template <typename TernaryOp, typename Arg1, typename Arg2, typename Arg3, typename StorageKind>
class CwiseTernaryOpImpl : public internal::generic_xpr_base<CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>::type {
 public:
  typedef typename internal::generic_xpr_base<CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>::type Base;
};

}  // end namespace Eigen

#endif  // EIGEN_CWISE_TERNARY_OP_H
