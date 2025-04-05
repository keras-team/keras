// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LINALG_OPS_H_
#define TENSORFLOW_CC_OPS_LINALG_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup linalg_ops Linalg Ops
/// @{

/// Computes the Cholesky decomposition of one or more square matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices.
///
/// The input has to be symmetric and positive definite. Only the lower-triangular
/// part of the input will be used for this operation. The upper-triangular part
/// will not be read.
///
/// The output is a tensor of the same shape as the input
/// containing the Cholesky decompositions for all input submatrices `[..., :, :]`.
///
/// **Note**: The gradient computation on GPU is faster for large matrices but
/// not for large batch dimensions when the submatrices are small. In this
/// case it might be faster to use the CPU.
///
/// Args:
/// * scope: A Scope object
/// * input: Shape is `[..., M, M]`.
///
/// Returns:
/// * `Output`: Shape is `[..., M, M]`.
class Cholesky {
 public:
  Cholesky(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the reverse mode backpropagated gradient of the Cholesky algorithm.
///
/// For an explanation see "Differentiation of the Cholesky algorithm" by
/// Iain Murray http://arxiv.org/abs/1602.07527.
///
/// Args:
/// * scope: A Scope object
/// * l: Output of batch Cholesky algorithm l = cholesky(A). Shape is `[..., M, M]`.
/// Algorithm depends only on lower triangular part of the innermost matrices of
/// this tensor.
/// * grad: df/dl where f is some scalar function. Shape is `[..., M, M]`.
/// Algorithm depends only on lower triangular part of the innermost matrices of
/// this tensor.
///
/// Returns:
/// * `Output`: Symmetrized version of df/dA . Shape is `[..., M, M]`
class CholeskyGrad {
 public:
  CholeskyGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input l,
             ::tensorflow::Input grad);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the eigen decomposition of one or more square matrices.
///
/// Computes the eigenvalues and (optionally) right eigenvectors of each inner matrix in
/// `input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
/// are sorted in non-decreasing order.
///
/// ```python
/// # a is a tensor.
/// # e is a tensor of eigenvalues.
/// # v is a tensor of eigenvectors.
/// e, v = eig(a)
/// e = eig(a, compute_v=False)
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: `Tensor` input of shape `[N, N]`.
///
/// Optional attributes (see `Attrs`):
/// * compute_v: If `True` then eigenvectors will be computed and returned in `v`.
/// Otherwise, only the eigenvalues will be computed.
///
/// Returns:
/// * `Output` e: Eigenvalues. Shape is `[N]`.
/// * `Output` v: Eigenvectors. Shape is `[N, N]`.
class Eig {
 public:
  /// Optional attribute setters for Eig
  struct Attrs {
    /// If `True` then eigenvectors will be computed and returned in `v`.
    /// Otherwise, only the eigenvalues will be computed.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ComputeV(bool x) {
      Attrs ret = *this;
      ret.compute_v_ = x;
      return ret;
    }

    bool compute_v_ = true;
  };
  Eig(const ::tensorflow::Scope& scope, ::tensorflow::Input input, DataType Tout);
  Eig(const ::tensorflow::Scope& scope, ::tensorflow::Input input, DataType Tout,
    const Eig::Attrs& attrs);

  static Attrs ComputeV(bool x) {
    return Attrs().ComputeV(x);
  }

  Operation operation;
  ::tensorflow::Output e;
  ::tensorflow::Output v;
};

/// Tensor contraction according to Einstein summation convention.
///
/// Implements generalized Tensor contraction and reduction. Each input Tensor must
/// have a corresponding input subscript appearing in the comma-separated left-hand
/// side of the equation. The right-hand side of the equation consists of the
/// output subscript. The input subscripts and the output subscript should consist
/// of zero or more named axis labels and at most one ellipsis (`...`).
///
/// The named axis labels may be any single character other than those having
/// special meaning, namely `,.->`. The behavior of this Op is undefined if it
/// receives an ill-formatted equation; since the validation is done at
/// graph-building time, we omit format validation checks at runtime.
///
/// Note: This Op is *not* intended to be called by the user; instead users should
/// call `tf.einsum` directly. It is a hidden Op used by `tf.einsum`.
///
/// Operations are applied to the input(s) according to the following rules:
///
///  (a) Generalized Diagonals: For input dimensions corresponding to axis labels
///      appearing more than once in the same input subscript, we take the
///      generalized (`k`-dimensional) diagonal.
///      For example, in the equation `iii->i` with input shape `[3, 3, 3]`, the
///      generalized diagonal would consist of `3` elements at indices `(0, 0, 0)`,
///      `(1, 1, 1)` and `(2, 2, 2)` to create a Tensor of shape `[3]`.
///
///  (b) Reduction: Axes corresponding to labels appearing only in one input
///      subscript but not in the output subscript are summed over prior to Tensor
///      contraction.
///      For example, in the equation `ab,bc->b`, the axis labels `a` and `c` are
///      the reduction axis labels.
///
///  (c) Batch Dimensions: Axes corresponding to labels appearing in each of the
///      input subscripts and also in the output subscript make up the batch
///      dimensions in Tensor contraction. Unnamed axis labels corresponding to
///      ellipsis (`...`) also correspond to batch dimensions.
///      For example, for the equation denoting batch matrix multiplication,
///      `bij,bjk->bik`, the axis label `b` corresponds to a batch dimension.
///
///  (d) Contraction: In case of binary einsum, axes corresponding to labels
///      appearing in two different inputs (and not in the output) are contracted
///      against each other.
///      Considering the batch matrix multiplication equation again
///      (`bij,bjk->bik`), the contracted axis label is `j`.
///
///  (e) Expand Diagonal: If the output subscripts contain repeated (explicit) axis
///      labels, the opposite operation of (a) is applied. For example, in the
///      equation `i->iii`, and input shape `[3]`, the output of shape `[3, 3, 3]`
///      are all zeros, except for the (generalized) diagonal which is populated
///      with values from the input.
///      Note: This operation is not supported by `np.einsum` or `tf.einsum`; it is
///      provided to enable computing the symbolic gradient of `tf.einsum`.
///
/// The output subscripts must contain only labels appearing in at least one of the
/// input subscripts. Furthermore, all dimensions mapping to the same axis label
/// must be equal.
///
/// Any of the input and output subscripts may contain at most a single ellipsis
/// (`...`). These ellipsis are mapped against dimensions not corresponding to any
/// named axis label. If two inputs contain ellipsis, then they are broadcasted
/// according to standard NumPy broadcasting
/// [rules](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
/// The broadcasted dimensions are placed in the corresponding location of the
/// ellipsis in the output subscript. If the broadcasted dimensions are non-empty
/// and the output subscripts do not contain ellipsis, then an InvalidArgument error
/// is raised.
///
/// @compatibility(numpy)
/// Similar to [`numpy.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html).
///
/// Comparison with `numpy.einsum`:
///
///  * This Op only supports unary and binary forms of `numpy.einsum`.
///  * This Op does not support implicit form. (i.e. equations without `->`).
///  * This Op also supports repeated indices in the output subscript, which is not
///    supported by `numpy.einsum`.
/// @end_compatibility
///
///
/// Args:
/// * scope: A Scope object
/// * inputs: List of 1 or 2 Tensors.
/// * equation: String describing the Einstein Summation operation; in the format of np.einsum.
///
/// Returns:
/// * `Output`: Output Tensor with shape depending upon `equation`.
class Einsum {
 public:
  Einsum(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs,
       StringPiece equation);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the sign and the log of the absolute value of the determinant of
///
/// one or more square matrices.
///
/// The input is a tensor of shape `[N, M, M]` whose inner-most 2 dimensions
/// form square matrices. The outputs are two tensors containing the signs and
/// absolute values of the log determinants for all N input submatrices
/// `[..., :, :]` such that `determinant = sign*exp(log_abs_determinant)`.
/// The `log_abs_determinant` is computed as `det(P)*sum(log(diag(LU)))` where `LU`
/// is the `LU` decomposition of the input and `P` is the corresponding
/// permutation matrix.
///
/// Args:
/// * scope: A Scope object
/// * input: Shape is `[N, M, M]`.
///
/// Returns:
/// * `Output` sign: The signs of the log determinants of the inputs. Shape is `[N]`.
/// * `Output` log_abs_determinant: The logs of the absolute values of the determinants
/// of the N input matrices.  Shape is `[N]`.
class LogMatrixDeterminant {
 public:
  LogMatrixDeterminant(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input);

  Operation operation;
  ::tensorflow::Output sign;
  ::tensorflow::Output log_abs_determinant;
};

/// Computes the LU decomposition of one or more square matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices.
///
/// The input has to be invertible.
///
/// The output consists of two tensors LU and P containing the LU decomposition
/// of all input submatrices `[..., :, :]`. LU encodes the lower triangular and
/// upper triangular factors.
///
/// For each input submatrix of shape `[M, M]`, L is a lower triangular matrix of
/// shape `[M, M]` with unit diagonal whose entries correspond to the strictly lower
/// triangular part of LU. U is a upper triangular matrix of shape `[M, M]` whose
/// entries correspond to the upper triangular part, including the diagonal, of LU.
///
/// P represents a permutation matrix encoded as a list of indices each between `0`
/// and `M-1`, inclusive. If P_mat denotes the permutation matrix corresponding to
/// P, then the L, U and P satisfies P_mat * input = L * U.
///
/// Args:
/// * scope: A Scope object
/// * input: A tensor of shape `[..., M, M]` whose inner-most 2 dimensions form matrices of
/// size `[M, M]`.
///
/// Returns:
/// * `Output` lu: A tensor of shape `[..., M, M]` whose strictly lower triangular part denotes the
/// lower triangular factor `L` with unit diagonal, and whose upper triangular part
/// denotes the upper triangular factor `U`.
/// * `Output` p: Permutation of the rows encoded as a list of indices in `0..M-1`. Shape is
/// `[..., M]`.
/// @compatibility(scipy)
/// Similar to `scipy.linalg.lu`, except the triangular factors `L` and `U` are
/// packed into a single tensor, the permutation is applied to `input` instead of
/// the right hand side and the permutation `P` is returned as a list of indices
/// instead of a permutation matrix.
/// @end_compatibility
class Lu {
 public:
  /// Optional attribute setters for Lu
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutputIdxType(DataType x) {
      Attrs ret = *this;
      ret.output_idx_type_ = x;
      return ret;
    }

    DataType output_idx_type_ = DT_INT32;
  };
  Lu(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Lu(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
   Lu::Attrs& attrs);

  static Attrs OutputIdxType(DataType x) {
    return Attrs().OutputIdxType(x);
  }

  Operation operation;
  ::tensorflow::Output lu;
  ::tensorflow::Output p;
};

/// Computes the determinant of one or more square matrices.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor containing the determinants
/// for all input submatrices `[..., :, :]`.
///
/// Args:
/// * scope: A Scope object
/// * input: Shape is `[..., M, M]`.
///
/// Returns:
/// * `Output`: Shape is `[...]`.
class MatrixDeterminant {
 public:
  MatrixDeterminant(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the inverse of one or more square invertible matrices or their adjoints (conjugate transposes).
///
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor of the same shape as the input
/// containing the inverse for all input submatrices `[..., :, :]`.
///
/// The op uses LU decomposition with partial pivoting to compute the inverses.
///
/// If a matrix is not invertible there is no guarantee what the op does. It
/// may detect the condition and raise an exception or it may simply return a
/// garbage result.
///
/// Args:
/// * scope: A Scope object
/// * input: Shape is `[..., M, M]`.
///
/// Returns:
/// * `Output`: Shape is `[..., M, M]`.
///
/// @compatibility(numpy)
/// Equivalent to np.linalg.inv
/// @end_compatibility
class MatrixInverse {
 public:
  /// Optional attribute setters for MatrixInverse
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Adjoint(bool x) {
      Attrs ret = *this;
      ret.adjoint_ = x;
      return ret;
    }

    bool adjoint_ = false;
  };
  MatrixInverse(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  MatrixInverse(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              const MatrixInverse::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Adjoint(bool x) {
    return Attrs().Adjoint(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Solves systems of linear equations.
///
/// `Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
/// a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
/// satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
/// If `adjoint` is `True` then each output matrix satisfies
/// `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.
///
/// Args:
/// * scope: A Scope object
/// * matrix: Shape is `[..., M, M]`.
/// * rhs: Shape is `[..., M, K]`.
///
/// Optional attributes (see `Attrs`):
/// * adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
/// adjoint.
///
/// Returns:
/// * `Output`: Shape is `[..., M, K]`.
class MatrixSolve {
 public:
  /// Optional attribute setters for MatrixSolve
  struct Attrs {
    /// Boolean indicating whether to solve with `matrix` or its (block-wise)
    /// adjoint.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Adjoint(bool x) {
      Attrs ret = *this;
      ret.adjoint_ = x;
      return ret;
    }

    bool adjoint_ = false;
  };
  MatrixSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input matrix,
            ::tensorflow::Input rhs);
  MatrixSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input matrix,
            ::tensorflow::Input rhs, const MatrixSolve::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Adjoint(bool x) {
    return Attrs().Adjoint(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Solves one or more linear least-squares problems.
///
/// `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
/// form real or complex matrices of size `[M, N]`. `Rhs` is a tensor of the same
/// type as `matrix` and shape `[..., M, K]`.
/// The output is a tensor shape `[..., N, K]` where each output matrix solves
/// each of the equations
/// `matrix[..., :, :]` * `output[..., :, :]` = `rhs[..., :, :]`
/// in the least squares sense.
///
/// We use the following notation for (complex) matrix and right-hand sides
/// in the batch:
///
/// `matrix`=\\(A \in \mathbb{C}^{m \times n}\\),
/// `rhs`=\\(B  \in \mathbb{C}^{m \times k}\\),
/// `output`=\\(X  \in \mathbb{C}^{n \times k}\\),
/// `l2_regularizer`=\\(\lambda \in \mathbb{R}\\).
///
/// If `fast` is `True`, then the solution is computed by solving the normal
/// equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
/// \\(X = (A^H A + \lambda I)^{-1} A^H B\\), which solves the least-squares
/// problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k} } ||A Z - B||_F^2 + \lambda ||Z||_F^2\\).
/// If \\(m \lt n\\) then `output` is computed as
/// \\(X = A^H (A A^H + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
/// minimum-norm solution to the under-determined linear system, i.e.
/// \\(X = \mathrm{argmin}_{Z \in \mathbb{C}^{n \times k} } ||Z||_F^2 \\),
/// subject to \\(A Z = B\\). Notice that the fast path is only numerically stable
/// when \\(A\\) is numerically full rank and has a condition number
/// \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach} } }\\) or \\(\lambda\\) is
/// sufficiently large.
///
/// If `fast` is `False` an algorithm based on the numerically robust complete
/// orthogonal decomposition is used. This computes the minimum-norm
/// least-squares solution, even when \\(A\\) is rank deficient. This path is
/// typically 6-7 times slower than the fast path. If `fast` is `False` then
/// `l2_regularizer` is ignored.
///
/// Args:
/// * scope: A Scope object
/// * matrix: Shape is `[..., M, N]`.
/// * rhs: Shape is `[..., M, K]`.
/// * l2_regularizer: Scalar tensor.
///
/// @compatibility(numpy)
/// Equivalent to np.linalg.lstsq
/// @end_compatibility
///
/// Returns:
/// * `Output`: Shape is `[..., N, K]`.
class MatrixSolveLs {
 public:
  /// Optional attribute setters for MatrixSolveLs
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs Fast(bool x) {
      Attrs ret = *this;
      ret.fast_ = x;
      return ret;
    }

    bool fast_ = true;
  };
  MatrixSolveLs(const ::tensorflow::Scope& scope, ::tensorflow::Input matrix,
              ::tensorflow::Input rhs, ::tensorflow::Input l2_regularizer);
  MatrixSolveLs(const ::tensorflow::Scope& scope, ::tensorflow::Input matrix,
              ::tensorflow::Input rhs, ::tensorflow::Input l2_regularizer,
              const MatrixSolveLs::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Fast(bool x) {
    return Attrs().Fast(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the matrix square root of one or more square matrices:
///
/// matmul(sqrtm(A), sqrtm(A)) = A
///
/// The input matrix should be invertible. If the input matrix is real, it should
/// have no eigenvalues which are real and negative (pairs of complex conjugate
/// eigenvalues are allowed).
///
/// The matrix square root is computed by first reducing the matrix to
/// quasi-triangular form with the real Schur decomposition. The square root
/// of the quasi-triangular matrix is then computed directly. Details of
/// the algorithm can be found in: Nicholas J. Higham, "Computing real
/// square roots of a real matrix", Linear Algebra Appl., 1987.
///
/// The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
/// form square matrices. The output is a tensor of the same shape as the input
/// containing the matrix square root for all input submatrices `[..., :, :]`.
///
/// Args:
/// * scope: A Scope object
/// * input: Shape is `[..., M, M]`.
///
/// Returns:
/// * `Output`: Shape is `[..., M, M]`.
///
/// @compatibility(scipy)
/// Equivalent to scipy.linalg.sqrtm
/// @end_compatibility
class MatrixSquareRoot {
 public:
  MatrixSquareRoot(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Solves systems of linear equations with upper or lower triangular matrices by backsubstitution.
///
///
/// `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
/// square matrices. If `lower` is `True` then the strictly upper triangular part
/// of each inner-most matrix is assumed to be zero and not accessed.
/// If `lower` is False then the strictly lower triangular part of each inner-most
/// matrix is assumed to be zero and not accessed.
/// `rhs` is a tensor of shape `[..., M, N]`.
///
/// The output is a tensor of shape `[..., M, N]`. If `adjoint` is
/// `True` then the innermost matrices in `output` satisfy matrix equations
/// `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
/// If `adjoint` is `False` then the strictly then the  innermost matrices in
/// `output` satisfy matrix equations
/// `adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.
///
/// Note, the batch shapes for the inputs only need to broadcast.
///
/// Example:
/// ```python
///
/// a = tf.constant([[3,  0,  0,  0],
///                  [2,  1,  0,  0],
///                  [1,  0,  1,  0],
///                  [1,  1,  1,  1]], dtype=tf.float32)
///
/// b = tf.constant([[4],
///                  [2],
///                  [4],
///                  [2]], dtype=tf.float32)
///
/// x = tf.linalg.triangular_solve(a, b, lower=True)
/// x
/// # <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
/// # array([[ 1.3333334 ],
/// #        [-0.66666675],
/// #        [ 2.6666665 ],
/// #        [-1.3333331 ]], dtype=float32)>
///
/// # in python3 one can use `a@x`
/// tf.matmul(a, x)
/// # <tf.Tensor: shape=(4, 1), dtype=float32, numpy=
/// # array([[4.       ],
/// #        [2.       ],
/// #        [4.       ],
/// #        [1.9999999]], dtype=float32)>
/// ```
///
/// Args:
/// * scope: A Scope object
/// * matrix: Shape is `[..., M, M]`.
/// * rhs: Shape is `[..., M, K]`.
///
/// Optional attributes (see `Attrs`):
/// * lower: Boolean indicating whether the innermost matrices in `matrix` are
/// lower or upper triangular.
/// * adjoint: Boolean indicating whether to solve with `matrix` or its (block-wise)
///          adjoint.
///
/// @compatibility(numpy)
/// Equivalent to scipy.linalg.solve_triangular
/// @end_compatibility
///
/// Returns:
/// * `Output`: Shape is `[..., M, K]`.
class MatrixTriangularSolve {
 public:
  /// Optional attribute setters for MatrixTriangularSolve
  struct Attrs {
    /// Boolean indicating whether the innermost matrices in `matrix` are
    /// lower or upper triangular.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs Lower(bool x) {
      Attrs ret = *this;
      ret.lower_ = x;
      return ret;
    }

    /// Boolean indicating whether to solve with `matrix` or its (block-wise)
    ///          adjoint.
    ///
    /// @compatibility(numpy)
    /// Equivalent to scipy.linalg.solve_triangular
    /// @end_compatibility
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Adjoint(bool x) {
      Attrs ret = *this;
      ret.adjoint_ = x;
      return ret;
    }

    bool lower_ = true;
    bool adjoint_ = false;
  };
  MatrixTriangularSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      matrix, ::tensorflow::Input rhs);
  MatrixTriangularSolve(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      matrix, ::tensorflow::Input rhs, const
                      MatrixTriangularSolve::Attrs& attrs);
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

/// Computes the QR decompositions of one or more matrices.
///
/// Computes the QR decomposition of each inner matrix in `tensor` such that
/// `tensor[..., :, :] = q[..., :, :] * r[..., :,:])`
///
/// Currently, the gradient for the QR decomposition is well-defined only when
/// the first `P` columns of the inner matrix are linearly independent, where
/// `P` is the minimum of `M` and `N`, the 2 inner-most dimmensions of `tensor`.
///
/// ```python
/// # a is a tensor.
/// # q is a tensor of orthonormal matrices.
/// # r is a tensor of upper triangular matrices.
/// q, r = qr(a)
/// q_full, r_full = qr(a, full_matrices=True)
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
/// form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
///
/// Optional attributes (see `Attrs`):
/// * full_matrices: If true, compute full-sized `q` and `r`. If false
/// (the default), compute only the leading `P` columns of `q`.
///
/// Returns:
/// * `Output` q: Orthonormal basis for range of `a`. If `full_matrices` is `False` then
/// shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
/// `[..., M, M]`.
/// * `Output` r: Triangular factor. If `full_matrices` is `False` then shape is
/// `[..., P, N]`. If `full_matrices` is `True` then shape is `[..., M, N]`.
class Qr {
 public:
  /// Optional attribute setters for Qr
  struct Attrs {
    /// If true, compute full-sized `q` and `r`. If false
    /// (the default), compute only the leading `P` columns of `q`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs FullMatrices(bool x) {
      Attrs ret = *this;
      ret.full_matrices_ = x;
      return ret;
    }

    bool full_matrices_ = false;
  };
  Qr(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Qr(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
   Qr::Attrs& attrs);

  static Attrs FullMatrices(bool x) {
    return Attrs().FullMatrices(x);
  }

  Operation operation;
  ::tensorflow::Output q;
  ::tensorflow::Output r;
};

/// Computes the eigen decomposition of one or more square self-adjoint matrices.
///
/// Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
/// `input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`. The eigenvalues
/// are sorted in non-decreasing order.
///
/// ```python
/// # a is a tensor.
/// # e is a tensor of eigenvalues.
/// # v is a tensor of eigenvectors.
/// e, v = self_adjoint_eig(a)
/// e = self_adjoint_eig(a, compute_v=False)
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: `Tensor` input of shape `[N, N]`.
///
/// Optional attributes (see `Attrs`):
/// * compute_v: If `True` then eigenvectors will be computed and returned in `v`.
/// Otherwise, only the eigenvalues will be computed.
///
/// Returns:
/// * `Output` e: Eigenvalues. Shape is `[N]`.
/// * `Output` v: Eigenvectors. Shape is `[N, N]`.
class SelfAdjointEig {
 public:
  /// Optional attribute setters for SelfAdjointEig
  struct Attrs {
    /// If `True` then eigenvectors will be computed and returned in `v`.
    /// Otherwise, only the eigenvalues will be computed.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ComputeV(bool x) {
      Attrs ret = *this;
      ret.compute_v_ = x;
      return ret;
    }

    bool compute_v_ = true;
  };
  SelfAdjointEig(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  SelfAdjointEig(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
               const SelfAdjointEig::Attrs& attrs);

  static Attrs ComputeV(bool x) {
    return Attrs().ComputeV(x);
  }

  Operation operation;
  ::tensorflow::Output e;
  ::tensorflow::Output v;
};

/// Computes the singular value decompositions of one or more matrices.
///
/// Computes the SVD of each inner matrix in `input` such that
/// `input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`
///
/// ```python
/// # a is a tensor containing a batch of matrices.
/// # s is a tensor of singular values for each matrix.
/// # u is the tensor containing the left singular vectors for each matrix.
/// # v is the tensor containing the right singular vectors for each matrix.
/// s, u, v = svd(a)
/// s, _, _ = svd(a, compute_uv=False)
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
/// form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
///
/// Optional attributes (see `Attrs`):
/// * compute_uv: If true, left and right singular vectors will be
/// computed and returned in `u` and `v`, respectively.
/// If false, `u` and `v` are not set and should never referenced.
/// * full_matrices: If true, compute full-sized `u` and `v`. If false
/// (the default), compute only the leading `P` singular vectors.
/// Ignored if `compute_uv` is `False`.
///
/// Returns:
/// * `Output` s: Singular values. Shape is `[..., P]`.
/// * `Output` u: Left singular vectors. If `full_matrices` is `False` then shape is
/// `[..., M, P]`; if `full_matrices` is `True` then shape is
/// `[..., M, M]`. Undefined if `compute_uv` is `False`.
/// * `Output` v: Left singular vectors. If `full_matrices` is `False` then shape is
/// `[..., N, P]`. If `full_matrices` is `True` then shape is `[..., N, N]`.
/// Undefined if `compute_uv` is false.
class Svd {
 public:
  /// Optional attribute setters for Svd
  struct Attrs {
    /// If true, left and right singular vectors will be
    /// computed and returned in `u` and `v`, respectively.
    /// If false, `u` and `v` are not set and should never referenced.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ComputeUv(bool x) {
      Attrs ret = *this;
      ret.compute_uv_ = x;
      return ret;
    }

    /// If true, compute full-sized `u` and `v`. If false
    /// (the default), compute only the leading `P` singular vectors.
    /// Ignored if `compute_uv` is `False`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs FullMatrices(bool x) {
      Attrs ret = *this;
      ret.full_matrices_ = x;
      return ret;
    }

    bool compute_uv_ = true;
    bool full_matrices_ = false;
  };
  Svd(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Svd(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
    Svd::Attrs& attrs);

  static Attrs ComputeUv(bool x) {
    return Attrs().ComputeUv(x);
  }
  static Attrs FullMatrices(bool x) {
    return Attrs().FullMatrices(x);
  }

  Operation operation;
  ::tensorflow::Output s;
  ::tensorflow::Output u;
  ::tensorflow::Output v;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LINALG_OPS_H_
