// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PRODUCT_H
#define EIGEN_PRODUCT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Lhs, typename Rhs, int Option, typename StorageKind>
class ProductImpl;

namespace internal {

template <typename Lhs, typename Rhs, int Option>
struct traits<Product<Lhs, Rhs, Option>> {
  typedef remove_all_t<Lhs> LhsCleaned;
  typedef remove_all_t<Rhs> RhsCleaned;
  typedef traits<LhsCleaned> LhsTraits;
  typedef traits<RhsCleaned> RhsTraits;

  typedef MatrixXpr XprKind;

  typedef typename ScalarBinaryOpTraits<typename traits<LhsCleaned>::Scalar,
                                        typename traits<RhsCleaned>::Scalar>::ReturnType Scalar;
  typedef typename product_promote_storage_type<typename LhsTraits::StorageKind, typename RhsTraits::StorageKind,
                                                internal::product_type<Lhs, Rhs>::ret>::ret StorageKind;
  typedef typename promote_index_type<typename LhsTraits::StorageIndex, typename RhsTraits::StorageIndex>::type
      StorageIndex;

  enum {
    RowsAtCompileTime = LhsTraits::RowsAtCompileTime,
    ColsAtCompileTime = RhsTraits::ColsAtCompileTime,
    MaxRowsAtCompileTime = LhsTraits::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = RhsTraits::MaxColsAtCompileTime,

    // FIXME: only needed by GeneralMatrixMatrixTriangular
    InnerSize = min_size_prefer_fixed(LhsTraits::ColsAtCompileTime, RhsTraits::RowsAtCompileTime),

    // The storage order is somewhat arbitrary here. The correct one will be determined through the evaluator.
    Flags = (MaxRowsAtCompileTime == 1 && MaxColsAtCompileTime != 1)   ? RowMajorBit
            : (MaxColsAtCompileTime == 1 && MaxRowsAtCompileTime != 1) ? 0
            : (((LhsTraits::Flags & NoPreferredStorageOrderBit) && (RhsTraits::Flags & RowMajorBit)) ||
               ((RhsTraits::Flags & NoPreferredStorageOrderBit) && (LhsTraits::Flags & RowMajorBit)))
                ? RowMajorBit
                : NoPreferredStorageOrderBit
  };
};

struct TransposeProductEnum {
  // convenience enumerations to specialize transposed products
  enum : int {
    Default = 0x00,
    Matrix = 0x01,
    Permutation = 0x02,
    MatrixMatrix = (Matrix << 8) | Matrix,
    MatrixPermutation = (Matrix << 8) | Permutation,
    PermutationMatrix = (Permutation << 8) | Matrix
  };
};
template <typename Xpr>
struct TransposeKind {
  static constexpr int Kind = is_matrix_base_xpr<Xpr>::value        ? TransposeProductEnum::Matrix
                              : is_permutation_base_xpr<Xpr>::value ? TransposeProductEnum::Permutation
                                                                    : TransposeProductEnum::Default;
};

template <typename Lhs, typename Rhs>
struct TransposeProductKind {
  static constexpr int Kind = (TransposeKind<Lhs>::Kind << 8) | TransposeKind<Rhs>::Kind;
};

template <typename Lhs, typename Rhs, int Option, int Kind = TransposeProductKind<Lhs, Rhs>::Kind>
struct product_transpose_helper {
  // by default, don't optimize the transposed product
  using Derived = Product<Lhs, Rhs, Option>;
  using Scalar = typename Derived::Scalar;
  using TransposeType = Transpose<const Derived>;
  using ConjugateTransposeType = CwiseUnaryOp<scalar_conjugate_op<Scalar>, TransposeType>;
  using AdjointType = std::conditional_t<NumTraits<Scalar>::IsComplex, ConjugateTransposeType, TransposeType>;

  // return (lhs * rhs)^T
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TransposeType run_transpose(const Derived& derived) {
    return TransposeType(derived);
  }
  // return (lhs * rhs)^H
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AdjointType run_adjoint(const Derived& derived) {
    return AdjointType(TransposeType(derived));
  }
};

template <typename Lhs, typename Rhs, int Option>
struct product_transpose_helper<Lhs, Rhs, Option, TransposeProductEnum::MatrixMatrix> {
  // expand the transposed matrix-matrix product
  using Derived = Product<Lhs, Rhs, Option>;

  using LhsScalar = typename traits<Lhs>::Scalar;
  using LhsTransposeType = typename DenseBase<Lhs>::ConstTransposeReturnType;
  using LhsConjugateTransposeType = CwiseUnaryOp<scalar_conjugate_op<LhsScalar>, LhsTransposeType>;
  using LhsAdjointType =
      std::conditional_t<NumTraits<LhsScalar>::IsComplex, LhsConjugateTransposeType, LhsTransposeType>;

  using RhsScalar = typename traits<Rhs>::Scalar;
  using RhsTransposeType = typename DenseBase<Rhs>::ConstTransposeReturnType;
  using RhsConjugateTransposeType = CwiseUnaryOp<scalar_conjugate_op<RhsScalar>, RhsTransposeType>;
  using RhsAdjointType =
      std::conditional_t<NumTraits<RhsScalar>::IsComplex, RhsConjugateTransposeType, RhsTransposeType>;

  using TransposeType = Product<RhsTransposeType, LhsTransposeType, Option>;
  using AdjointType = Product<RhsAdjointType, LhsAdjointType, Option>;

  // return rhs^T * lhs^T
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TransposeType run_transpose(const Derived& derived) {
    return TransposeType(RhsTransposeType(derived.rhs()), LhsTransposeType(derived.lhs()));
  }
  // return rhs^H * lhs^H
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AdjointType run_adjoint(const Derived& derived) {
    return AdjointType(RhsAdjointType(RhsTransposeType(derived.rhs())),
                       LhsAdjointType(LhsTransposeType(derived.lhs())));
  }
};
template <typename Lhs, typename Rhs, int Option>
struct product_transpose_helper<Lhs, Rhs, Option, TransposeProductEnum::PermutationMatrix> {
  // expand the transposed permutation-matrix product
  using Derived = Product<Lhs, Rhs, Option>;

  using LhsInverseType = typename PermutationBase<Lhs>::InverseReturnType;

  using RhsScalar = typename traits<Rhs>::Scalar;
  using RhsTransposeType = typename DenseBase<Rhs>::ConstTransposeReturnType;
  using RhsConjugateTransposeType = CwiseUnaryOp<scalar_conjugate_op<RhsScalar>, RhsTransposeType>;
  using RhsAdjointType =
      std::conditional_t<NumTraits<RhsScalar>::IsComplex, RhsConjugateTransposeType, RhsTransposeType>;

  using TransposeType = Product<RhsTransposeType, LhsInverseType, Option>;
  using AdjointType = Product<RhsAdjointType, LhsInverseType, Option>;

  // return rhs^T * lhs^-1
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TransposeType run_transpose(const Derived& derived) {
    return TransposeType(RhsTransposeType(derived.rhs()), LhsInverseType(derived.lhs()));
  }
  // return rhs^H * lhs^-1
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AdjointType run_adjoint(const Derived& derived) {
    return AdjointType(RhsAdjointType(RhsTransposeType(derived.rhs())), LhsInverseType(derived.lhs()));
  }
};
template <typename Lhs, typename Rhs, int Option>
struct product_transpose_helper<Lhs, Rhs, Option, TransposeProductEnum::MatrixPermutation> {
  // expand the transposed matrix-permutation product
  using Derived = Product<Lhs, Rhs, Option>;

  using LhsScalar = typename traits<Lhs>::Scalar;
  using LhsTransposeType = typename DenseBase<Lhs>::ConstTransposeReturnType;
  using LhsConjugateTransposeType = CwiseUnaryOp<scalar_conjugate_op<LhsScalar>, LhsTransposeType>;
  using LhsAdjointType =
      std::conditional_t<NumTraits<LhsScalar>::IsComplex, LhsConjugateTransposeType, LhsTransposeType>;

  using RhsInverseType = typename PermutationBase<Rhs>::InverseReturnType;

  using TransposeType = Product<RhsInverseType, LhsTransposeType, Option>;
  using AdjointType = Product<RhsInverseType, LhsAdjointType, Option>;

  // return rhs^-1 * lhs^T
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TransposeType run_transpose(const Derived& derived) {
    return TransposeType(RhsInverseType(derived.rhs()), LhsTransposeType(derived.lhs()));
  }
  // return rhs^-1 * lhs^H
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AdjointType run_adjoint(const Derived& derived) {
    return AdjointType(RhsInverseType(derived.rhs()), LhsAdjointType(LhsTransposeType(derived.lhs())));
  }
};

}  // end namespace internal

/** \class Product
 * \ingroup Core_Module
 *
 * \brief Expression of the product of two arbitrary matrices or vectors
 *
 * \tparam Lhs_ the type of the left-hand side expression
 * \tparam Rhs_ the type of the right-hand side expression
 *
 * This class represents an expression of the product of two arbitrary matrices.
 *
 * The other template parameters are:
 * \tparam Option     can be DefaultProduct, AliasFreeProduct, or LazyProduct
 *
 */
template <typename Lhs_, typename Rhs_, int Option>
class Product
    : public ProductImpl<Lhs_, Rhs_, Option,
                         typename internal::product_promote_storage_type<
                             typename internal::traits<Lhs_>::StorageKind, typename internal::traits<Rhs_>::StorageKind,
                             internal::product_type<Lhs_, Rhs_>::ret>::ret> {
 public:
  typedef Lhs_ Lhs;
  typedef Rhs_ Rhs;

  typedef
      typename ProductImpl<Lhs, Rhs, Option,
                           typename internal::product_promote_storage_type<
                               typename internal::traits<Lhs>::StorageKind, typename internal::traits<Rhs>::StorageKind,
                               internal::product_type<Lhs, Rhs>::ret>::ret>::Base Base;
  EIGEN_GENERIC_PUBLIC_INTERFACE(Product)

  typedef typename internal::ref_selector<Lhs>::type LhsNested;
  typedef typename internal::ref_selector<Rhs>::type RhsNested;
  typedef internal::remove_all_t<LhsNested> LhsNestedCleaned;
  typedef internal::remove_all_t<RhsNested> RhsNestedCleaned;

  using TransposeReturnType = typename internal::product_transpose_helper<Lhs, Rhs, Option>::TransposeType;
  using AdjointReturnType = typename internal::product_transpose_helper<Lhs, Rhs, Option>::AdjointType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Product(const Lhs& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs) {
    eigen_assert(lhs.cols() == rhs.rows() && "invalid matrix product" &&
                 "if you wanted a coeff-wise or a dot product use the respective explicit functions");
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT { return m_lhs.rows(); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT { return m_rhs.cols(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const LhsNestedCleaned& lhs() const { return m_lhs; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const RhsNestedCleaned& rhs() const { return m_rhs; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TransposeReturnType transpose() const {
    return internal::product_transpose_helper<Lhs, Rhs, Option>::run_transpose(*this);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AdjointReturnType adjoint() const {
    return internal::product_transpose_helper<Lhs, Rhs, Option>::run_adjoint(*this);
  }

 protected:
  LhsNested m_lhs;
  RhsNested m_rhs;
};

namespace internal {

template <typename Lhs, typename Rhs, int Option, int ProductTag = internal::product_type<Lhs, Rhs>::ret>
class dense_product_base : public internal::dense_xpr_base<Product<Lhs, Rhs, Option>>::type {};

/** Conversion to scalar for inner-products */
template <typename Lhs, typename Rhs, int Option>
class dense_product_base<Lhs, Rhs, Option, InnerProduct>
    : public internal::dense_xpr_base<Product<Lhs, Rhs, Option>>::type {
  typedef Product<Lhs, Rhs, Option> ProductXpr;
  typedef typename internal::dense_xpr_base<ProductXpr>::type Base;

 public:
  using Base::derived;
  typedef typename Base::Scalar Scalar;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE operator const Scalar() const {
    return internal::evaluator<ProductXpr>(derived()).coeff(0, 0);
  }
};

}  // namespace internal

// Generic API dispatcher
template <typename Lhs, typename Rhs, int Option, typename StorageKind>
class ProductImpl : public internal::generic_xpr_base<Product<Lhs, Rhs, Option>, MatrixXpr, StorageKind>::type {
 public:
  typedef typename internal::generic_xpr_base<Product<Lhs, Rhs, Option>, MatrixXpr, StorageKind>::type Base;
};

template <typename Lhs, typename Rhs, int Option>
class ProductImpl<Lhs, Rhs, Option, Dense> : public internal::dense_product_base<Lhs, Rhs, Option> {
  typedef Product<Lhs, Rhs, Option> Derived;

 public:
  typedef typename internal::dense_product_base<Lhs, Rhs, Option> Base;
  EIGEN_DENSE_PUBLIC_INTERFACE(Derived)
 protected:
  enum {
    IsOneByOne = (RowsAtCompileTime == 1 || RowsAtCompileTime == Dynamic) &&
                 (ColsAtCompileTime == 1 || ColsAtCompileTime == Dynamic),
    EnableCoeff = IsOneByOne || Option == LazyProduct
  };

 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(Index row, Index col) const {
    EIGEN_STATIC_ASSERT(EnableCoeff, THIS_METHOD_IS_ONLY_FOR_INNER_OR_LAZY_PRODUCTS);
    eigen_assert((Option == LazyProduct) || (this->rows() == 1 && this->cols() == 1));

    return internal::evaluator<Derived>(derived()).coeff(row, col);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(Index i) const {
    EIGEN_STATIC_ASSERT(EnableCoeff, THIS_METHOD_IS_ONLY_FOR_INNER_OR_LAZY_PRODUCTS);
    eigen_assert((Option == LazyProduct) || (this->rows() == 1 && this->cols() == 1));

    return internal::evaluator<Derived>(derived()).coeff(i);
  }
};

}  // end namespace Eigen

#endif  // EIGEN_PRODUCT_H
