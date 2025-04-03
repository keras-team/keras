// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LINALG_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_LINALG_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup linalg_ops_internal Linalg Ops Internal
/// @{

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class BandedTriangularSolve {
 public:
  /// Optional attribute setters for BandedTriangularSolve
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs Lower(bool x) {
      Attrs ret = *this;
      ret.lower_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Adjoint(bool x) {
      Attrs ret = *this;
      ret.adjoint_ = x;
      return ret;
    }

    bool lower_ = true;
    bool adjoint_ = false;
  };
  BandedTriangularSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      matrix, ::tensorflow::Input rhs);
  BandedTriangularSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      matrix, ::tensorflow::Input rhs, const
                      BandedTriangularSolve::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Lower(bool x) {
    return Attrs().Lower(x);
  }
  static Attrs Adjoint(bool x) {
    return Attrs().Adjoint(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the matrix logarithm of one or more square matrices:
///
///
/// \\(log(exp(A)) = A\\)
///
/// This op is only defined for complex matrices. If A is positive-definite and
/// real, then casting to a complex matrix, taking the logarithm and casting back
/// to a real matrix will give the correct result.
///
/// This function computes the matrix logarithm using the Schur-Parlett algorithm.
/// Details of the algorithm can be found in Section 11.6.2 of:
/// Nicholas J. Higham, Functions of Matrices: Theory and Computation, SIAM 2008.
/// ISBN 978-0-898716-46-7.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor of the same shape as the input
/// containing the exponential for all input submatrices `[..., :, :]`.
///
/// Args:
/// * scope: A Scope object
/// * input: Shape is `[..., M, M]`.
///
/// Returns:
/// * `Output`: Shape is `[..., M, M]`.
///
/// @compatibility(scipy)
/// Equivalent to scipy.linalg.logm
/// @end_compatibility
class MatrixLogarithm {
 public:
  MatrixLogarithm(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Calculate product with tridiagonal matrix.
///
/// Calculates product of two matrices, where left matrix is a tridiagonal matrix.
///
/// Args:
/// * scope: A Scope object
/// * superdiag: Tensor of shape `[..., 1, M]`, representing superdiagonals of
/// tri-diagonal matrices to the left of multiplication. Last element is ignored.
/// * maindiag: Tensor of shape `[..., 1, M]`, representing main diagonals of tri-diagonal
/// matrices to the left of multiplication.
/// * subdiag: Tensor of shape `[..., 1, M]`, representing subdiagonals of tri-diagonal
/// matrices to the left of multiplication. First element is ignored.
/// * rhs: Tensor of shape `[..., M, N]`, representing MxN matrices to the right of
/// multiplication.
///
/// Returns:
/// * `Output`: Tensor of shape `[..., M, N]` containing the product.
class TridiagonalMatMul {
 public:
  TridiagonalMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  superdiag, ::tensorflow::Input maindiag, ::tensorflow::Input
                  subdiag, ::tensorflow::Input rhs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Solves tridiagonal systems of equations.
///
///   Solves tridiagonal systems of equations.
///   Supports batch dimensions and multiple right-hand sides per each left-hand
///   side.
///   On CPU, solution is computed via Gaussian elimination with or without partial
///   pivoting, depending on `partial_pivoting` attribute. On GPU, Nvidia's cuSPARSE
///   library is used: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv
///   Partial pivoting is not yet supported by XLA backends.
///
/// Args:
/// * scope: A Scope object
/// * diagonals: Tensor of shape `[..., 3, M]` whose innermost 2 dimensions represent the
/// tridiagonal matrices with three rows being the superdiagonal, diagonals, and
/// subdiagonals, in order. The last element of the superdiagonal and the first
/// element of the subdiagonal is ignored.
/// * rhs: Tensor of shape `[..., M, K]`, representing K right-hand sides per each
/// left-hand side.
///
/// Optional attributes (see `Attrs`):
/// * partial_pivoting: Whether to apply partial pivoting. Partial pivoting makes the procedure more
/// stable, but slower.
///
/// Returns:
/// * `Output`: Tensor of shape `[..., M, K]` containing the solutions
class TridiagonalSolve {
 public:
  /// Optional attribute setters for TridiagonalSolve
  struct Attrs {
    /// Whether to apply partial pivoting. Partial pivoting makes the procedure more
    /// stable, but slower.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs PartialPivoting(bool x) {
      Attrs ret = *this;
      ret.partial_pivoting_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs PerturbSingular(bool x) {
      Attrs ret = *this;
      ret.perturb_singular_ = x;
      return ret;
    }

    bool partial_pivoting_ = true;
    bool perturb_singular_ = false;
  };
  TridiagonalSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 diagonals, ::tensorflow::Input rhs);
  TridiagonalSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input
                 diagonals, ::tensorflow::Input rhs, const
                 TridiagonalSolve::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs PartialPivoting(bool x) {
    return Attrs().PartialPivoting(x);
  }
  static Attrs PerturbSingular(bool x) {
    return Attrs().PerturbSingular(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LINALG_OPS_INTERNAL_H_
