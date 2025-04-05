// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INDEXED_VIEW_H
#define EIGEN_INDEXED_VIEW_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename XprType, typename RowIndices, typename ColIndices>
struct traits<IndexedView<XprType, RowIndices, ColIndices>> : traits<XprType> {
  enum {
    RowsAtCompileTime = int(IndexedViewHelper<RowIndices>::SizeAtCompileTime),
    ColsAtCompileTime = int(IndexedViewHelper<ColIndices>::SizeAtCompileTime),
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,

    XprTypeIsRowMajor = (int(traits<XprType>::Flags) & RowMajorBit) != 0,
    IsRowMajor = (MaxRowsAtCompileTime == 1 && MaxColsAtCompileTime != 1)   ? 1
                 : (MaxColsAtCompileTime == 1 && MaxRowsAtCompileTime != 1) ? 0
                                                                            : XprTypeIsRowMajor,

    RowIncr = int(IndexedViewHelper<RowIndices>::IncrAtCompileTime),
    ColIncr = int(IndexedViewHelper<ColIndices>::IncrAtCompileTime),
    InnerIncr = IsRowMajor ? ColIncr : RowIncr,
    OuterIncr = IsRowMajor ? RowIncr : ColIncr,

    HasSameStorageOrderAsXprType = (IsRowMajor == XprTypeIsRowMajor),
    XprInnerStride = HasSameStorageOrderAsXprType ? int(inner_stride_at_compile_time<XprType>::ret)
                                                  : int(outer_stride_at_compile_time<XprType>::ret),
    XprOuterstride = HasSameStorageOrderAsXprType ? int(outer_stride_at_compile_time<XprType>::ret)
                                                  : int(inner_stride_at_compile_time<XprType>::ret),

    InnerSize = XprTypeIsRowMajor ? ColsAtCompileTime : RowsAtCompileTime,
    IsBlockAlike = InnerIncr == 1 && OuterIncr == 1,
    IsInnerPannel = HasSameStorageOrderAsXprType &&
                    is_same<AllRange<InnerSize>, std::conditional_t<XprTypeIsRowMajor, ColIndices, RowIndices>>::value,

    InnerStrideAtCompileTime =
        InnerIncr < 0 || InnerIncr == DynamicIndex || XprInnerStride == Dynamic || InnerIncr == Undefined
            ? Dynamic
            : XprInnerStride * InnerIncr,
    OuterStrideAtCompileTime =
        OuterIncr < 0 || OuterIncr == DynamicIndex || XprOuterstride == Dynamic || OuterIncr == Undefined
            ? Dynamic
            : XprOuterstride * OuterIncr,

    ReturnAsScalar = is_single_range<RowIndices>::value && is_single_range<ColIndices>::value,
    ReturnAsBlock = (!ReturnAsScalar) && IsBlockAlike,
    ReturnAsIndexedView = (!ReturnAsScalar) && (!ReturnAsBlock),

    // FIXME we deal with compile-time strides if and only if we have DirectAccessBit flag,
    // but this is too strict regarding negative strides...
    DirectAccessMask = (int(InnerIncr) != Undefined && int(OuterIncr) != Undefined && InnerIncr >= 0 && OuterIncr >= 0)
                           ? DirectAccessBit
                           : 0,
    FlagsRowMajorBit = IsRowMajor ? RowMajorBit : 0,
    FlagsLvalueBit = is_lvalue<XprType>::value ? LvalueBit : 0,
    FlagsLinearAccessBit = (RowsAtCompileTime == 1 || ColsAtCompileTime == 1) ? LinearAccessBit : 0,
    Flags = (traits<XprType>::Flags & (HereditaryBits | DirectAccessMask)) | FlagsLvalueBit | FlagsRowMajorBit |
            FlagsLinearAccessBit
  };

  typedef Block<XprType, RowsAtCompileTime, ColsAtCompileTime, IsInnerPannel> BlockType;
};

template <typename XprType, typename RowIndices, typename ColIndices, typename StorageKind, bool DirectAccess>
class IndexedViewImpl;

}  // namespace internal

/** \class IndexedView
 * \ingroup Core_Module
 *
 * \brief Expression of a non-sequential sub-matrix defined by arbitrary sequences of row and column indices
 *
 * \tparam XprType the type of the expression in which we are taking the intersections of sub-rows and sub-columns
 * \tparam RowIndices the type of the object defining the sequence of row indices
 * \tparam ColIndices the type of the object defining the sequence of column indices
 *
 * This class represents an expression of a sub-matrix (or sub-vector) defined as the intersection
 * of sub-sets of rows and columns, that are themself defined by generic sequences of row indices \f$
 * \{r_0,r_1,..r_{m-1}\} \f$ and column indices \f$ \{c_0,c_1,..c_{n-1} \}\f$. Let \f$ A \f$  be the nested matrix, then
 * the resulting matrix \f$ B \f$ has \c m rows and \c n columns, and its entries are given by: \f$ B(i,j) = A(r_i,c_j)
 * \f$.
 *
 * The \c RowIndices and \c ColIndices types must be compatible with the following API:
 * \code
 * <integral type> operator[](Index) const;
 * Index size() const;
 * \endcode
 *
 * Typical supported types thus include:
 *  - std::vector<int>
 *  - std::valarray<int>
 *  - std::array<int>
 *  - Eigen::ArrayXi
 *  - decltype(ArrayXi::LinSpaced(...))
 *  - Any view/expressions of the previous types
 *  - Eigen::ArithmeticSequence
 *  - Eigen::internal::AllRange     (helper for Eigen::placeholders::all)
 *  - Eigen::internal::SingleRange  (helper for single index)
 *  - etc.
 *
 * In typical usages of %Eigen, this class should never be used directly. It is the return type of
 * DenseBase::operator()(const RowIndices&, const ColIndices&).
 *
 * \sa class Block
 */
template <typename XprType, typename RowIndices, typename ColIndices>
class IndexedView
    : public internal::IndexedViewImpl<XprType, RowIndices, ColIndices, typename internal::traits<XprType>::StorageKind,
                                       (internal::traits<IndexedView<XprType, RowIndices, ColIndices>>::Flags &
                                        DirectAccessBit) != 0> {
 public:
  typedef typename internal::IndexedViewImpl<
      XprType, RowIndices, ColIndices, typename internal::traits<XprType>::StorageKind,
      (internal::traits<IndexedView<XprType, RowIndices, ColIndices>>::Flags & DirectAccessBit) != 0>
      Base;
  EIGEN_GENERIC_PUBLIC_INTERFACE(IndexedView)
  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(IndexedView)

  template <typename T0, typename T1>
  IndexedView(XprType& xpr, const T0& rowIndices, const T1& colIndices) : Base(xpr, rowIndices, colIndices) {}
};

namespace internal {

// Generic API dispatcher
template <typename XprType, typename RowIndices, typename ColIndices, typename StorageKind, bool DirectAccess>
class IndexedViewImpl : public internal::generic_xpr_base<IndexedView<XprType, RowIndices, ColIndices>>::type {
 public:
  typedef typename internal::generic_xpr_base<IndexedView<XprType, RowIndices, ColIndices>>::type Base;
  typedef typename internal::ref_selector<XprType>::non_const_type MatrixTypeNested;
  typedef internal::remove_all_t<XprType> NestedExpression;
  typedef typename XprType::Scalar Scalar;

  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(IndexedViewImpl)

  template <typename T0, typename T1>
  IndexedViewImpl(XprType& xpr, const T0& rowIndices, const T1& colIndices)
      : m_xpr(xpr), m_rowIndices(rowIndices), m_colIndices(colIndices) {}

  /** \returns number of rows */
  Index rows() const { return IndexedViewHelper<RowIndices>::size(m_rowIndices); }

  /** \returns number of columns */
  Index cols() const { return IndexedViewHelper<ColIndices>::size(m_colIndices); }

  /** \returns the nested expression */
  const internal::remove_all_t<XprType>& nestedExpression() const { return m_xpr; }

  /** \returns the nested expression */
  std::remove_reference_t<XprType>& nestedExpression() { return m_xpr; }

  /** \returns a const reference to the object storing/generating the row indices */
  const RowIndices& rowIndices() const { return m_rowIndices; }

  /** \returns a const reference to the object storing/generating the column indices */
  const ColIndices& colIndices() const { return m_colIndices; }

  constexpr Scalar& coeffRef(Index rowId, Index colId) {
    return nestedExpression().coeffRef(m_rowIndices[rowId], m_colIndices[colId]);
  }

  constexpr const Scalar& coeffRef(Index rowId, Index colId) const {
    return nestedExpression().coeffRef(m_rowIndices[rowId], m_colIndices[colId]);
  }

 protected:
  MatrixTypeNested m_xpr;
  RowIndices m_rowIndices;
  ColIndices m_colIndices;
};

template <typename XprType, typename RowIndices, typename ColIndices, typename StorageKind>
class IndexedViewImpl<XprType, RowIndices, ColIndices, StorageKind, true>
    : public IndexedViewImpl<XprType, RowIndices, ColIndices, StorageKind, false> {
 public:
  using Base = internal::IndexedViewImpl<XprType, RowIndices, ColIndices,
                                         typename internal::traits<XprType>::StorageKind, false>;
  using Derived = IndexedView<XprType, RowIndices, ColIndices>;

  EIGEN_INHERIT_ASSIGNMENT_OPERATORS(IndexedViewImpl)

  template <typename T0, typename T1>
  IndexedViewImpl(XprType& xpr, const T0& rowIndices, const T1& colIndices) : Base(xpr, rowIndices, colIndices) {}

  Index rowIncrement() const {
    if (traits<Derived>::RowIncr != DynamicIndex && traits<Derived>::RowIncr != Undefined) {
      return traits<Derived>::RowIncr;
    }
    return IndexedViewHelper<RowIndices>::incr(this->rowIndices());
  }
  Index colIncrement() const {
    if (traits<Derived>::ColIncr != DynamicIndex && traits<Derived>::ColIncr != Undefined) {
      return traits<Derived>::ColIncr;
    }
    return IndexedViewHelper<ColIndices>::incr(this->colIndices());
  }

  Index innerIncrement() const { return traits<Derived>::IsRowMajor ? colIncrement() : rowIncrement(); }

  Index outerIncrement() const { return traits<Derived>::IsRowMajor ? rowIncrement() : colIncrement(); }

  std::decay_t<typename XprType::Scalar>* data() {
    Index row_offset = this->rowIndices()[0] * this->nestedExpression().rowStride();
    Index col_offset = this->colIndices()[0] * this->nestedExpression().colStride();
    return this->nestedExpression().data() + row_offset + col_offset;
  }

  const std::decay_t<typename XprType::Scalar>* data() const {
    Index row_offset = this->rowIndices()[0] * this->nestedExpression().rowStride();
    Index col_offset = this->colIndices()[0] * this->nestedExpression().colStride();
    return this->nestedExpression().data() + row_offset + col_offset;
  }

  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index innerStride() const EIGEN_NOEXCEPT {
    if (traits<Derived>::InnerStrideAtCompileTime != Dynamic) {
      return traits<Derived>::InnerStrideAtCompileTime;
    }
    return innerIncrement() * this->nestedExpression().innerStride();
  }

  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index outerStride() const EIGEN_NOEXCEPT {
    if (traits<Derived>::OuterStrideAtCompileTime != Dynamic) {
      return traits<Derived>::OuterStrideAtCompileTime;
    }
    return outerIncrement() * this->nestedExpression().outerStride();
  }
};

template <typename ArgType, typename RowIndices, typename ColIndices>
struct unary_evaluator<IndexedView<ArgType, RowIndices, ColIndices>, IndexBased>
    : evaluator_base<IndexedView<ArgType, RowIndices, ColIndices>> {
  typedef IndexedView<ArgType, RowIndices, ColIndices> XprType;

  enum {
    CoeffReadCost = evaluator<ArgType>::CoeffReadCost /* TODO + cost of row/col index */,

    FlagsLinearAccessBit =
        (traits<XprType>::RowsAtCompileTime == 1 || traits<XprType>::ColsAtCompileTime == 1) ? LinearAccessBit : 0,

    FlagsRowMajorBit = traits<XprType>::FlagsRowMajorBit,

    Flags = (evaluator<ArgType>::Flags & (HereditaryBits & ~RowMajorBit /*| LinearAccessBit | DirectAccessBit*/)) |
            FlagsLinearAccessBit | FlagsRowMajorBit,

    Alignment = 0
  };

  EIGEN_DEVICE_FUNC explicit unary_evaluator(const XprType& xpr) : m_argImpl(xpr.nestedExpression()), m_xpr(xpr) {
    EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
  }

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index row, Index col) const {
    eigen_assert(m_xpr.rowIndices()[row] >= 0 && m_xpr.rowIndices()[row] < m_xpr.nestedExpression().rows() &&
                 m_xpr.colIndices()[col] >= 0 && m_xpr.colIndices()[col] < m_xpr.nestedExpression().cols());
    return m_argImpl.coeff(m_xpr.rowIndices()[row], m_xpr.colIndices()[col]);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index row, Index col) {
    eigen_assert(m_xpr.rowIndices()[row] >= 0 && m_xpr.rowIndices()[row] < m_xpr.nestedExpression().rows() &&
                 m_xpr.colIndices()[col] >= 0 && m_xpr.colIndices()[col] < m_xpr.nestedExpression().cols());
    return m_argImpl.coeffRef(m_xpr.rowIndices()[row], m_xpr.colIndices()[col]);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) {
    EIGEN_STATIC_ASSERT_LVALUE(XprType)
    Index row = XprType::RowsAtCompileTime == 1 ? 0 : index;
    Index col = XprType::RowsAtCompileTime == 1 ? index : 0;
    eigen_assert(m_xpr.rowIndices()[row] >= 0 && m_xpr.rowIndices()[row] < m_xpr.nestedExpression().rows() &&
                 m_xpr.colIndices()[col] >= 0 && m_xpr.colIndices()[col] < m_xpr.nestedExpression().cols());
    return m_argImpl.coeffRef(m_xpr.rowIndices()[row], m_xpr.colIndices()[col]);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& coeffRef(Index index) const {
    Index row = XprType::RowsAtCompileTime == 1 ? 0 : index;
    Index col = XprType::RowsAtCompileTime == 1 ? index : 0;
    eigen_assert(m_xpr.rowIndices()[row] >= 0 && m_xpr.rowIndices()[row] < m_xpr.nestedExpression().rows() &&
                 m_xpr.colIndices()[col] >= 0 && m_xpr.colIndices()[col] < m_xpr.nestedExpression().cols());
    return m_argImpl.coeffRef(m_xpr.rowIndices()[row], m_xpr.colIndices()[col]);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const CoeffReturnType coeff(Index index) const {
    Index row = XprType::RowsAtCompileTime == 1 ? 0 : index;
    Index col = XprType::RowsAtCompileTime == 1 ? index : 0;
    eigen_assert(m_xpr.rowIndices()[row] >= 0 && m_xpr.rowIndices()[row] < m_xpr.nestedExpression().rows() &&
                 m_xpr.colIndices()[col] >= 0 && m_xpr.colIndices()[col] < m_xpr.nestedExpression().cols());
    return m_argImpl.coeff(m_xpr.rowIndices()[row], m_xpr.colIndices()[col]);
  }

 protected:
  evaluator<ArgType> m_argImpl;
  const XprType& m_xpr;
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_INDEXED_VIEW_H
