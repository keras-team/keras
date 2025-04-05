// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_PERMUTATION_H
#define EIGEN_SPARSE_PERMUTATION_H

// This file implements sparse * permutation products

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename ExpressionType, typename PlainObjectType,
          bool NeedEval = !is_same<ExpressionType, PlainObjectType>::value>
struct XprHelper {
  XprHelper(const ExpressionType& xpr) : m_xpr(xpr) {}
  inline const PlainObjectType& xpr() const { return m_xpr; }
  // this is a new PlainObjectType initialized by xpr
  const PlainObjectType m_xpr;
};
template <typename ExpressionType, typename PlainObjectType>
struct XprHelper<ExpressionType, PlainObjectType, false> {
  XprHelper(const ExpressionType& xpr) : m_xpr(xpr) {}
  inline const PlainObjectType& xpr() const { return m_xpr; }
  // this is a reference to xpr
  const PlainObjectType& m_xpr;
};

template <typename PermDerived, bool NeedInverseEval>
struct PermHelper {
  using IndicesType = typename PermDerived::IndicesType;
  using PermutationIndex = typename IndicesType::Scalar;
  using type = PermutationMatrix<IndicesType::SizeAtCompileTime, IndicesType::MaxSizeAtCompileTime, PermutationIndex>;
  PermHelper(const PermDerived& perm) : m_perm(perm.inverse()) {}
  inline const type& perm() const { return m_perm; }
  // this is a new PermutationMatrix initialized by perm.inverse()
  const type m_perm;
};
template <typename PermDerived>
struct PermHelper<PermDerived, false> {
  using type = PermDerived;
  PermHelper(const PermDerived& perm) : m_perm(perm) {}
  inline const type& perm() const { return m_perm; }
  // this is a reference to perm
  const type& m_perm;
};

template <typename ExpressionType, int Side, bool Transposed>
struct permutation_matrix_product<ExpressionType, Side, Transposed, SparseShape> {
  using MatrixType = typename nested_eval<ExpressionType, 1>::type;
  using MatrixTypeCleaned = remove_all_t<MatrixType>;

  using Scalar = typename MatrixTypeCleaned::Scalar;
  using StorageIndex = typename MatrixTypeCleaned::StorageIndex;

  // the actual "return type" is `Dest`. this is a temporary type
  using ReturnType = SparseMatrix<Scalar, MatrixTypeCleaned::IsRowMajor ? RowMajor : ColMajor, StorageIndex>;
  using TmpHelper = XprHelper<ExpressionType, ReturnType>;

  static constexpr bool NeedOuterPermutation = ExpressionType::IsRowMajor ? Side == OnTheLeft : Side == OnTheRight;
  static constexpr bool NeedInversePermutation = Transposed ? Side == OnTheLeft : Side == OnTheRight;

  template <typename Dest, typename PermutationType>
  static inline void permute_outer(Dest& dst, const PermutationType& perm, const ExpressionType& xpr) {
    // if ExpressionType is not ReturnType, evaluate `xpr` (allocation)
    // otherwise, just reference `xpr`
    // TODO: handle trivial expressions such as CwiseBinaryOp without temporary
    const TmpHelper tmpHelper(xpr);
    const ReturnType& tmp = tmpHelper.xpr();

    ReturnType result(tmp.rows(), tmp.cols());

    for (Index j = 0; j < tmp.outerSize(); j++) {
      Index jp = perm.indices().coeff(j);
      Index jsrc = NeedInversePermutation ? jp : j;
      Index jdst = NeedInversePermutation ? j : jp;
      Index begin = tmp.outerIndexPtr()[jsrc];
      Index end = tmp.isCompressed() ? tmp.outerIndexPtr()[jsrc + 1] : begin + tmp.innerNonZeroPtr()[jsrc];
      result.outerIndexPtr()[jdst + 1] += end - begin;
    }

    std::partial_sum(result.outerIndexPtr(), result.outerIndexPtr() + result.outerSize() + 1, result.outerIndexPtr());
    result.resizeNonZeros(result.nonZeros());

    for (Index j = 0; j < tmp.outerSize(); j++) {
      Index jp = perm.indices().coeff(j);
      Index jsrc = NeedInversePermutation ? jp : j;
      Index jdst = NeedInversePermutation ? j : jp;
      Index begin = tmp.outerIndexPtr()[jsrc];
      Index end = tmp.isCompressed() ? tmp.outerIndexPtr()[jsrc + 1] : begin + tmp.innerNonZeroPtr()[jsrc];
      Index target = result.outerIndexPtr()[jdst];
      smart_copy(tmp.innerIndexPtr() + begin, tmp.innerIndexPtr() + end, result.innerIndexPtr() + target);
      smart_copy(tmp.valuePtr() + begin, tmp.valuePtr() + end, result.valuePtr() + target);
    }
    dst = std::move(result);
  }

  template <typename Dest, typename PermutationType>
  static inline void permute_inner(Dest& dst, const PermutationType& perm, const ExpressionType& xpr) {
    using InnerPermHelper = PermHelper<PermutationType, NeedInversePermutation>;
    using InnerPermType = typename InnerPermHelper::type;

    // if ExpressionType is not ReturnType, evaluate `xpr` (allocation)
    // otherwise, just reference `xpr`
    // TODO: handle trivial expressions such as CwiseBinaryOp without temporary
    const TmpHelper tmpHelper(xpr);
    const ReturnType& tmp = tmpHelper.xpr();

    // if inverse permutation of inner indices is requested, calculate perm.inverse() (allocation)
    // otherwise, just reference `perm`
    const InnerPermHelper permHelper(perm);
    const InnerPermType& innerPerm = permHelper.perm();

    ReturnType result(tmp.rows(), tmp.cols());

    for (Index j = 0; j < tmp.outerSize(); j++) {
      Index begin = tmp.outerIndexPtr()[j];
      Index end = tmp.isCompressed() ? tmp.outerIndexPtr()[j + 1] : begin + tmp.innerNonZeroPtr()[j];
      result.outerIndexPtr()[j + 1] += end - begin;
    }

    std::partial_sum(result.outerIndexPtr(), result.outerIndexPtr() + result.outerSize() + 1, result.outerIndexPtr());
    result.resizeNonZeros(result.nonZeros());

    for (Index j = 0; j < tmp.outerSize(); j++) {
      Index begin = tmp.outerIndexPtr()[j];
      Index end = tmp.isCompressed() ? tmp.outerIndexPtr()[j + 1] : begin + tmp.innerNonZeroPtr()[j];
      Index target = result.outerIndexPtr()[j];
      std::transform(tmp.innerIndexPtr() + begin, tmp.innerIndexPtr() + end, result.innerIndexPtr() + target,
                     [&innerPerm](StorageIndex i) { return innerPerm.indices().coeff(i); });
      smart_copy(tmp.valuePtr() + begin, tmp.valuePtr() + end, result.valuePtr() + target);
    }
    // the inner indices were permuted, and must be sorted
    result.sortInnerIndices();
    dst = std::move(result);
  }

  template <typename Dest, typename PermutationType, bool DoOuter = NeedOuterPermutation,
            std::enable_if_t<DoOuter, int> = 0>
  static inline void run(Dest& dst, const PermutationType& perm, const ExpressionType& xpr) {
    permute_outer(dst, perm, xpr);
  }

  template <typename Dest, typename PermutationType, bool DoOuter = NeedOuterPermutation,
            std::enable_if_t<!DoOuter, int> = 0>
  static inline void run(Dest& dst, const PermutationType& perm, const ExpressionType& xpr) {
    permute_inner(dst, perm, xpr);
  }
};

}  // namespace internal

namespace internal {

template <int ProductTag>
struct product_promote_storage_type<Sparse, PermutationStorage, ProductTag> {
  typedef Sparse ret;
};
template <int ProductTag>
struct product_promote_storage_type<PermutationStorage, Sparse, ProductTag> {
  typedef Sparse ret;
};

// TODO, the following two overloads are only needed to define the right temporary type through
// typename traits<permutation_sparse_matrix_product<Rhs,Lhs,OnTheRight,false> >::ReturnType
// whereas it should be correctly handled by traits<Product<> >::PlainObject

template <typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, AliasFreeProduct>, ProductTag, PermutationShape, SparseShape>
    : public evaluator<typename permutation_matrix_product<Rhs, OnTheLeft, false, SparseShape>::ReturnType> {
  typedef Product<Lhs, Rhs, AliasFreeProduct> XprType;
  typedef typename permutation_matrix_product<Rhs, OnTheLeft, false, SparseShape>::ReturnType PlainObject;
  typedef evaluator<PlainObject> Base;

  enum { Flags = Base::Flags | EvalBeforeNestingBit };

  explicit product_evaluator(const XprType& xpr) : m_result(xpr.rows(), xpr.cols()) {
    internal::construct_at<Base>(this, m_result);
    generic_product_impl<Lhs, Rhs, PermutationShape, SparseShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }

 protected:
  PlainObject m_result;
};

template <typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, AliasFreeProduct>, ProductTag, SparseShape, PermutationShape>
    : public evaluator<typename permutation_matrix_product<Lhs, OnTheRight, false, SparseShape>::ReturnType> {
  typedef Product<Lhs, Rhs, AliasFreeProduct> XprType;
  typedef typename permutation_matrix_product<Lhs, OnTheRight, false, SparseShape>::ReturnType PlainObject;
  typedef evaluator<PlainObject> Base;

  enum { Flags = Base::Flags | EvalBeforeNestingBit };

  explicit product_evaluator(const XprType& xpr) : m_result(xpr.rows(), xpr.cols()) {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, SparseShape, PermutationShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }

 protected:
  PlainObject m_result;
};

}  // end namespace internal

/** \returns the matrix with the permutation applied to the columns
 */
template <typename SparseDerived, typename PermDerived>
inline const Product<SparseDerived, PermDerived, AliasFreeProduct> operator*(
    const SparseMatrixBase<SparseDerived>& matrix, const PermutationBase<PermDerived>& perm) {
  return Product<SparseDerived, PermDerived, AliasFreeProduct>(matrix.derived(), perm.derived());
}

/** \returns the matrix with the permutation applied to the rows
 */
template <typename SparseDerived, typename PermDerived>
inline const Product<PermDerived, SparseDerived, AliasFreeProduct> operator*(
    const PermutationBase<PermDerived>& perm, const SparseMatrixBase<SparseDerived>& matrix) {
  return Product<PermDerived, SparseDerived, AliasFreeProduct>(perm.derived(), matrix.derived());
}

/** \returns the matrix with the inverse permutation applied to the columns.
 */
template <typename SparseDerived, typename PermutationType>
inline const Product<SparseDerived, Inverse<PermutationType>, AliasFreeProduct> operator*(
    const SparseMatrixBase<SparseDerived>& matrix, const InverseImpl<PermutationType, PermutationStorage>& tperm) {
  return Product<SparseDerived, Inverse<PermutationType>, AliasFreeProduct>(matrix.derived(), tperm.derived());
}

/** \returns the matrix with the inverse permutation applied to the rows.
 */
template <typename SparseDerived, typename PermutationType>
inline const Product<Inverse<PermutationType>, SparseDerived, AliasFreeProduct> operator*(
    const InverseImpl<PermutationType, PermutationStorage>& tperm, const SparseMatrixBase<SparseDerived>& matrix) {
  return Product<Inverse<PermutationType>, SparseDerived, AliasFreeProduct>(tperm.derived(), matrix.derived());
}

}  // end namespace Eigen

#endif  // EIGEN_SPARSE_SELFADJOINTVIEW_H
