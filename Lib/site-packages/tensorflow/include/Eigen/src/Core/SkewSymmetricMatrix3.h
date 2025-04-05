// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SKEWSYMMETRICMATRIX3_H
#define EIGEN_SKEWSYMMETRICMATRIX3_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \class SkewSymmetricBase
 * \ingroup Core_Module
 *
 * \brief Base class for skew symmetric matrices and expressions
 *
 * This is the base class that is inherited by SkewSymmetricMatrix3 and related expression
 * types, which internally use a three vector for storing the entries. SkewSymmetric
 * types always represent square three times three matrices.
 *
 * This implementations follows class DiagonalMatrix
 *
 * \tparam Derived is the derived type, a SkewSymmetricMatrix3 or SkewSymmetricWrapper.
 *
 * \sa class SkewSymmetricMatrix3, class SkewSymmetricWrapper
 */
template <typename Derived>
class SkewSymmetricBase : public EigenBase<Derived> {
 public:
  typedef typename internal::traits<Derived>::SkewSymmetricVectorType SkewSymmetricVectorType;
  typedef typename SkewSymmetricVectorType::Scalar Scalar;
  typedef typename SkewSymmetricVectorType::RealScalar RealScalar;
  typedef typename internal::traits<Derived>::StorageKind StorageKind;
  typedef typename internal::traits<Derived>::StorageIndex StorageIndex;

  enum {
    RowsAtCompileTime = SkewSymmetricVectorType::SizeAtCompileTime,
    ColsAtCompileTime = SkewSymmetricVectorType::SizeAtCompileTime,
    MaxRowsAtCompileTime = SkewSymmetricVectorType::MaxSizeAtCompileTime,
    MaxColsAtCompileTime = SkewSymmetricVectorType::MaxSizeAtCompileTime,
    IsVectorAtCompileTime = 0,
    Flags = NoPreferredStorageOrderBit
  };

  typedef Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, 0, MaxRowsAtCompileTime, MaxColsAtCompileTime>
      DenseMatrixType;
  typedef DenseMatrixType DenseType;
  typedef SkewSymmetricMatrix3<Scalar> PlainObject;

  /** \returns a reference to the derived object. */
  EIGEN_DEVICE_FUNC inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
  /** \returns a const reference to the derived object. */
  EIGEN_DEVICE_FUNC inline Derived& derived() { return *static_cast<Derived*>(this); }

  /**
   * Constructs a dense matrix from \c *this. Note, this directly returns a dense matrix type,
   * not an expression.
   * \returns A dense matrix, with its entries set from the the derived object. */
  EIGEN_DEVICE_FUNC DenseMatrixType toDenseMatrix() const { return derived(); }

  /** Determinant vanishes */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Scalar determinant() const { return 0; }

  /** A.transpose() = -A */
  EIGEN_DEVICE_FUNC PlainObject transpose() const { return (-vector()).asSkewSymmetric(); }

  /** \returns the exponential of this matrix using Rodrigues’ formula */
  EIGEN_DEVICE_FUNC DenseMatrixType exponential() const {
    DenseMatrixType retVal = DenseMatrixType::Identity();
    const SkewSymmetricVectorType& v = vector();
    if (v.isZero()) {
      return retVal;
    }
    const Scalar norm2 = v.squaredNorm();
    const Scalar norm = numext::sqrt(norm2);
    retVal += ((((1 - numext::cos(norm)) / norm2) * derived()) * derived()) +
              (numext::sin(norm) / norm) * derived().toDenseMatrix();
    return retVal;
  }

  /** \returns a reference to the derived object's vector of coefficients. */
  EIGEN_DEVICE_FUNC inline const SkewSymmetricVectorType& vector() const { return derived().vector(); }
  /** \returns a const reference to the derived object's vector of coefficients. */
  EIGEN_DEVICE_FUNC inline SkewSymmetricVectorType& vector() { return derived().vector(); }

  /** \returns the number of rows. */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index rows() const { return 3; }
  /** \returns the number of columns. */
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR inline Index cols() const { return 3; }

  /** \returns the matrix product of \c *this by the dense matrix, \a matrix */
  template <typename MatrixDerived>
  EIGEN_DEVICE_FUNC Product<Derived, MatrixDerived, LazyProduct> operator*(
      const MatrixBase<MatrixDerived>& matrix) const {
    return Product<Derived, MatrixDerived, LazyProduct>(derived(), matrix.derived());
  }

  /** \returns the matrix product of \c *this by the skew symmetric matrix, \a matrix */
  template <typename MatrixDerived>
  EIGEN_DEVICE_FUNC Product<Derived, MatrixDerived, LazyProduct> operator*(
      const SkewSymmetricBase<MatrixDerived>& matrix) const {
    return Product<Derived, MatrixDerived, LazyProduct>(derived(), matrix.derived());
  }

  template <typename OtherDerived>
  using SkewSymmetricProductReturnType = SkewSymmetricWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(
      SkewSymmetricVectorType, typename OtherDerived::SkewSymmetricVectorType, product)>;

  /** \returns the wedge product of \c *this by the skew symmetric matrix \a other
   *  A wedge B = AB - BA */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SkewSymmetricProductReturnType<OtherDerived> wedge(
      const SkewSymmetricBase<OtherDerived>& other) const {
    return vector().cross(other.vector()).asSkewSymmetric();
  }

  using SkewSymmetricScaleReturnType =
      SkewSymmetricWrapper<const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(SkewSymmetricVectorType, Scalar, product)>;

  /** \returns the product of \c *this by the scalar \a scalar */
  EIGEN_DEVICE_FUNC inline SkewSymmetricScaleReturnType operator*(const Scalar& scalar) const {
    return (vector() * scalar).asSkewSymmetric();
  }

  using ScaleSkewSymmetricReturnType =
      SkewSymmetricWrapper<const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(Scalar, SkewSymmetricVectorType, product)>;

  /** \returns the product of a scalar and the skew symmetric matrix \a other */
  EIGEN_DEVICE_FUNC friend inline ScaleSkewSymmetricReturnType operator*(const Scalar& scalar,
                                                                         const SkewSymmetricBase& other) {
    return (scalar * other.vector()).asSkewSymmetric();
  }

  template <typename OtherDerived>
  using SkewSymmetricSumReturnType = SkewSymmetricWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(
      SkewSymmetricVectorType, typename OtherDerived::SkewSymmetricVectorType, sum)>;

  /** \returns the sum of \c *this and the skew symmetric matrix \a other */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC inline SkewSymmetricSumReturnType<OtherDerived> operator+(
      const SkewSymmetricBase<OtherDerived>& other) const {
    return (vector() + other.vector()).asSkewSymmetric();
  }

  template <typename OtherDerived>
  using SkewSymmetricDifferenceReturnType = SkewSymmetricWrapper<const EIGEN_CWISE_BINARY_RETURN_TYPE(
      SkewSymmetricVectorType, typename OtherDerived::SkewSymmetricVectorType, difference)>;

  /** \returns the difference of \c *this and the skew symmetric matrix \a other */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC inline SkewSymmetricDifferenceReturnType<OtherDerived> operator-(
      const SkewSymmetricBase<OtherDerived>& other) const {
    return (vector() - other.vector()).asSkewSymmetric();
  }
};

/** \class SkewSymmetricMatrix3
 * \ingroup Core_Module
 *
 * \brief Represents a 3x3 skew symmetric matrix with its storage
 *
 * \tparam Scalar_ the type of coefficients
 *
 * \sa class SkewSymmetricBase, class SkewSymmetricWrapper
 */

namespace internal {
template <typename Scalar_>
struct traits<SkewSymmetricMatrix3<Scalar_>> : traits<Matrix<Scalar_, 3, 3, 0, 3, 3>> {
  typedef Matrix<Scalar_, 3, 1, 0, 3, 1> SkewSymmetricVectorType;
  typedef SkewSymmetricShape StorageKind;
  enum { Flags = LvalueBit | NoPreferredStorageOrderBit | NestByRefBit };
};
}  // namespace internal
template <typename Scalar_>
class SkewSymmetricMatrix3 : public SkewSymmetricBase<SkewSymmetricMatrix3<Scalar_>> {
 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  typedef typename internal::traits<SkewSymmetricMatrix3>::SkewSymmetricVectorType SkewSymmetricVectorType;
  typedef const SkewSymmetricMatrix3& Nested;
  typedef Scalar_ Scalar;
  typedef typename internal::traits<SkewSymmetricMatrix3>::StorageKind StorageKind;
  typedef typename internal::traits<SkewSymmetricMatrix3>::StorageIndex StorageIndex;
#endif

 protected:
  SkewSymmetricVectorType m_vector;

 public:
  /** const version of vector(). */
  EIGEN_DEVICE_FUNC inline const SkewSymmetricVectorType& vector() const { return m_vector; }
  /** \returns a reference to the stored vector of coefficients. */
  EIGEN_DEVICE_FUNC inline SkewSymmetricVectorType& vector() { return m_vector; }

  /** Default constructor without initialization */
  EIGEN_DEVICE_FUNC inline SkewSymmetricMatrix3() {}

  /** Constructor from three scalars */
  EIGEN_DEVICE_FUNC inline SkewSymmetricMatrix3(const Scalar& x, const Scalar& y, const Scalar& z)
      : m_vector(x, y, z) {}

  /** \brief Constructs a SkewSymmetricMatrix3 from an r-value vector type */
  EIGEN_DEVICE_FUNC explicit inline SkewSymmetricMatrix3(SkewSymmetricVectorType&& vec) : m_vector(std::move(vec)) {}

  /** generic constructor from expression of the coefficients */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC explicit inline SkewSymmetricMatrix3(const MatrixBase<OtherDerived>& other) : m_vector(other) {}

  /** Copy constructor. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC inline SkewSymmetricMatrix3(const SkewSymmetricBase<OtherDerived>& other)
      : m_vector(other.vector()) {}

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** copy constructor. prevent a default copy constructor from hiding the other templated constructor */
  inline SkewSymmetricMatrix3(const SkewSymmetricMatrix3& other) : m_vector(other.vector()) {}
#endif

  /** Copy operator. */
  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC SkewSymmetricMatrix3& operator=(const SkewSymmetricBase<OtherDerived>& other) {
    m_vector = other.vector();
    return *this;
  }

#ifndef EIGEN_PARSED_BY_DOXYGEN
  /** This is a special case of the templated operator=. Its purpose is to
   * prevent a default operator= from hiding the templated operator=.
   */
  EIGEN_DEVICE_FUNC SkewSymmetricMatrix3& operator=(const SkewSymmetricMatrix3& other) {
    m_vector = other.vector();
    return *this;
  }
#endif

  typedef SkewSymmetricWrapper<const CwiseNullaryOp<internal::scalar_constant_op<Scalar>, SkewSymmetricVectorType>>
      InitializeReturnType;

  /** Initializes a skew symmetric matrix with coefficients set to zero */
  EIGEN_DEVICE_FUNC static InitializeReturnType Zero() { return SkewSymmetricVectorType::Zero().asSkewSymmetric(); }

  /** Sets all coefficients to zero. */
  EIGEN_DEVICE_FUNC inline void setZero() { m_vector.setZero(); }
};

/** \class SkewSymmetricWrapper
 * \ingroup Core_Module
 *
 * \brief Expression of a skew symmetric matrix
 *
 * \tparam SkewSymmetricVectorType_ the type of the vector of coefficients
 *
 * This class is an expression of a skew symmetric matrix, but not storing its own vector of coefficients,
 * instead wrapping an existing vector expression. It is the return type of MatrixBase::asSkewSymmetric()
 * and most of the time this is the only way that it is used.
 *
 * \sa class SkewSymmetricMatrix3, class SkewSymmetricBase, MatrixBase::asSkewSymmetric()
 */

namespace internal {
template <typename SkewSymmetricVectorType_>
struct traits<SkewSymmetricWrapper<SkewSymmetricVectorType_>> {
  typedef SkewSymmetricVectorType_ SkewSymmetricVectorType;
  typedef typename SkewSymmetricVectorType::Scalar Scalar;
  typedef typename SkewSymmetricVectorType::StorageIndex StorageIndex;
  typedef SkewSymmetricShape StorageKind;
  typedef typename traits<SkewSymmetricVectorType>::XprKind XprKind;
  enum {
    RowsAtCompileTime = SkewSymmetricVectorType::SizeAtCompileTime,
    ColsAtCompileTime = SkewSymmetricVectorType::SizeAtCompileTime,
    MaxRowsAtCompileTime = SkewSymmetricVectorType::MaxSizeAtCompileTime,
    MaxColsAtCompileTime = SkewSymmetricVectorType::MaxSizeAtCompileTime,
    Flags = (traits<SkewSymmetricVectorType>::Flags & LvalueBit) | NoPreferredStorageOrderBit
  };
};
}  // namespace internal

template <typename SkewSymmetricVectorType_>
class SkewSymmetricWrapper : public SkewSymmetricBase<SkewSymmetricWrapper<SkewSymmetricVectorType_>>,
                             internal::no_assignment_operator {
 public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
  typedef SkewSymmetricVectorType_ SkewSymmetricVectorType;
  typedef SkewSymmetricWrapper Nested;
#endif

  /** Constructor from expression of coefficients to wrap. */
  EIGEN_DEVICE_FUNC explicit inline SkewSymmetricWrapper(SkewSymmetricVectorType& a_vector) : m_vector(a_vector) {}

  /** \returns a const reference to the wrapped expression of coefficients. */
  EIGEN_DEVICE_FUNC const SkewSymmetricVectorType& vector() const { return m_vector; }

 protected:
  typename SkewSymmetricVectorType::Nested m_vector;
};

/** \returns a pseudo-expression of a skew symmetric matrix with *this as vector of coefficients
 *
 * \only_for_vectors
 *
 * \sa class SkewSymmetricWrapper, class SkewSymmetricMatrix3, vector(), isSkewSymmetric()
 **/
template <typename Derived>
EIGEN_DEVICE_FUNC inline const SkewSymmetricWrapper<const Derived> MatrixBase<Derived>::asSkewSymmetric() const {
  return SkewSymmetricWrapper<const Derived>(derived());
}

/** \returns true if *this is approximately equal to a skew symmetric matrix,
 *          within the precision given by \a prec.
 */
template <typename Derived>
bool MatrixBase<Derived>::isSkewSymmetric(const RealScalar& prec) const {
  if (cols() != rows()) return false;
  return (this->transpose() + *this).isZero(prec);
}

/** \returns the matrix product of \c *this by the skew symmetric matrix \skew.
 */
template <typename Derived>
template <typename SkewDerived>
EIGEN_DEVICE_FUNC inline const Product<Derived, SkewDerived, LazyProduct> MatrixBase<Derived>::operator*(
    const SkewSymmetricBase<SkewDerived>& skew) const {
  return Product<Derived, SkewDerived, LazyProduct>(derived(), skew.derived());
}

namespace internal {

template <>
struct storage_kind_to_shape<SkewSymmetricShape> {
  typedef SkewSymmetricShape Shape;
};

struct SkewSymmetric2Dense {};

template <>
struct AssignmentKind<DenseShape, SkewSymmetricShape> {
  typedef SkewSymmetric2Dense Kind;
};

// SkewSymmetric matrix to Dense assignment
template <typename DstXprType, typename SrcXprType, typename Functor>
struct Assignment<DstXprType, SrcXprType, Functor, SkewSymmetric2Dense> {
  EIGEN_DEVICE_FUNC static void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>& /*func*/) {
    if ((dst.rows() != 3) || (dst.cols() != 3)) {
      dst.resize(3, 3);
    }
    dst.diagonal().setZero();
    const typename SrcXprType::SkewSymmetricVectorType v = src.vector();
    dst(0, 1) = -v(2);
    dst(1, 0) = v(2);
    dst(0, 2) = v(1);
    dst(2, 0) = -v(1);
    dst(1, 2) = -v(0);
    dst(2, 1) = v(0);
  }
  EIGEN_DEVICE_FUNC static void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::add_assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>& /*func*/) {
    dst.vector() += src.vector();
  }

  EIGEN_DEVICE_FUNC static void run(
      DstXprType& dst, const SrcXprType& src,
      const internal::sub_assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>& /*func*/) {
    dst.vector() -= src.vector();
  }
};

}  // namespace internal

}  // end namespace Eigen

#endif  // EIGEN_SKEWSYMMETRICMATRIX3_H
