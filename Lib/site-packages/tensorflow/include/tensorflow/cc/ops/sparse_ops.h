// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_SPARSE_OPS_H_
#define TENSORFLOW_CC_OPS_SPARSE_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup sparse_ops Sparse Ops
/// @{

/// Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.
///
/// A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
/// `sparse_values`, and `sparse_shape`, where
///
/// ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```
///
/// An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
/// having a first `sparse_indices` column taking values between `[0, N)`, where
/// the minibatch size `N == sparse_shape[0]`.
///
/// The input `SparseTensor` must have rank `R` greater than 1, and the first
/// dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
/// must be sorted in increasing order of this first dimension.  The stored
/// `SparseTensor` objects pointed to by each row of the output `sparse_handles`
/// will have rank `R-1`.
///
/// The `SparseTensor` values can then be read out as part of a minibatch by passing
/// the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
/// the correct `SparseTensorsMap` is accessed, ensure that the same
/// `container` and `shared_name` are passed to that Op.  If no `shared_name`
/// is provided here, instead use the *name* of the Operation created by calling
/// `AddManySparseToTensorsMap` as the `shared_name` passed to
/// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
///
/// Args:
/// * scope: A Scope object
/// * sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
/// `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
/// * sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
/// * sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
/// The minibatch size `N == sparse_shape[0]`.
///
/// Optional attributes (see `Attrs`):
/// * container: The container name for the `SparseTensorsMap` created by this op.
/// * shared_name: The shared name for the `SparseTensorsMap` created by this op.
/// If blank, the new Operation's unique name is used.
///
/// Returns:
/// * `Output`: 1-D.  The handles of the `SparseTensor` now stored in the
/// `SparseTensorsMap`.  Shape: `[N]`.
class AddManySparseToTensorsMap {
 public:
  /// Optional attribute setters for AddManySparseToTensorsMap
  struct Attrs {
    /// The container name for the `SparseTensorsMap` created by this op.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// The shared name for the `SparseTensorsMap` created by this op.
    /// If blank, the new Operation's unique name is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  AddManySparseToTensorsMap(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          sparse_indices, ::tensorflow::Input sparse_values,
                          ::tensorflow::Input sparse_shape);
  AddManySparseToTensorsMap(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          sparse_indices, ::tensorflow::Input sparse_values,
                          ::tensorflow::Input sparse_shape, const
                          AddManySparseToTensorsMap::Attrs& attrs);
  operator ::tensorflow::Output() const { return sparse_handles; }
  operator ::tensorflow::Input() const { return sparse_handles; }
  ::tensorflow::Node* node() const { return sparse_handles.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output sparse_handles;
};

/// Add a `SparseTensor` to a `SparseTensorsMap` return its handle.
///
/// A `SparseTensor` is represented by three tensors: `sparse_indices`,
/// `sparse_values`, and `sparse_shape`.
///
/// This operator takes the given `SparseTensor` and adds it to a container
/// object (a `SparseTensorsMap`).  A unique key within this container is generated
/// in the form of an `int64`, and this is the value that is returned.
///
/// The `SparseTensor` can then be read out as part of a minibatch by passing
/// the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
/// the correct `SparseTensorsMap` is accessed, ensure that the same
/// `container` and `shared_name` are passed to that Op.  If no `shared_name`
/// is provided here, instead use the *name* of the Operation created by calling
/// `AddSparseToTensorsMap` as the `shared_name` passed to
/// `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.
///
/// Args:
/// * scope: A Scope object
/// * sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
/// * sparse_values: 1-D.  The `values` of the `SparseTensor`.
/// * sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
///
/// Optional attributes (see `Attrs`):
/// * container: The container name for the `SparseTensorsMap` created by this op.
/// * shared_name: The shared name for the `SparseTensorsMap` created by this op.
/// If blank, the new Operation's unique name is used.
///
/// Returns:
/// * `Output`: 0-D.  The handle of the `SparseTensor` now stored in the
/// `SparseTensorsMap`.
class AddSparseToTensorsMap {
 public:
  /// Optional attribute setters for AddSparseToTensorsMap
  struct Attrs {
    /// The container name for the `SparseTensorsMap` created by this op.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// The shared name for the `SparseTensorsMap` created by this op.
    /// If blank, the new Operation's unique name is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  AddSparseToTensorsMap(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      sparse_indices, ::tensorflow::Input sparse_values,
                      ::tensorflow::Input sparse_shape);
  AddSparseToTensorsMap(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      sparse_indices, ::tensorflow::Input sparse_values,
                      ::tensorflow::Input sparse_shape, const
                      AddSparseToTensorsMap::Attrs& attrs);
  operator ::tensorflow::Output() const { return sparse_handle; }
  operator ::tensorflow::Input() const { return sparse_handle; }
  ::tensorflow::Node* node() const { return sparse_handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output sparse_handle;
};

/// Deserialize and concatenate `SparseTensors` from a serialized minibatch.
///
/// The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
/// `N` is the minibatch size and the rows correspond to packed outputs of
/// `SerializeSparse`.  The ranks of the original `SparseTensor` objects
/// must all match.  When the final `SparseTensor` is created, it has rank one
/// higher than the ranks of the incoming `SparseTensor` objects
/// (they have been concatenated along a new row dimension).
///
/// The output `SparseTensor` object's shape values for all dimensions but the
/// first are the max across the input `SparseTensor` objects' shape values
/// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
/// size.
///
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
///
/// For example, if the serialized input is a `[2 x 3]` matrix representing two
/// original `SparseTensor` objects:
///
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
///
/// and
///
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
///
/// then the final deserialized `SparseTensor` will be:
///
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
///
/// Args:
/// * scope: A Scope object
/// * serialized_sparse: 2-D, The `N` serialized `SparseTensor` objects.
/// Must have 3 columns.
/// * dtype: The `dtype` of the serialized `SparseTensor` objects.
///
/// Returns:
/// * `Output` sparse_indices
/// * `Output` sparse_values
/// * `Output` sparse_shape
class DeserializeManySparse {
 public:
  DeserializeManySparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      serialized_sparse, DataType dtype);

  Operation operation;
  ::tensorflow::Output sparse_indices;
  ::tensorflow::Output sparse_values;
  ::tensorflow::Output sparse_shape;
};

/// Deserialize `SparseTensor` objects.
///
/// The input `serialized_sparse` must have the shape `[?, ?, ..., ?, 3]` where
/// the last dimension stores serialized `SparseTensor` objects and the other N
/// dimensions (N >= 0) correspond to a batch. The ranks of the original
/// `SparseTensor` objects must all match. When the final `SparseTensor` is
/// created, its rank is the rank of the incoming `SparseTensor` objects plus N;
/// the sparse tensors have been concatenated along new dimensions, one for each
/// batch.
///
/// The output `SparseTensor` object's shape values for the original dimensions
/// are the max across the input `SparseTensor` objects' shape values for the
/// corresponding dimensions. The new dimensions match the size of the batch.
///
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
///
/// For example, if the serialized input is a `[2 x 3]` matrix representing two
/// original `SparseTensor` objects:
///
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
///
/// and
///
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
///
/// then the final deserialized `SparseTensor` will be:
///
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
///
/// Args:
/// * scope: A Scope object
/// * serialized_sparse: The serialized `SparseTensor` objects. The last dimension
/// must have 3 columns.
/// * dtype: The `dtype` of the serialized `SparseTensor` objects.
///
/// Returns:
/// * `Output` sparse_indices
/// * `Output` sparse_values
/// * `Output` sparse_shape
class DeserializeSparse {
 public:
  DeserializeSparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  serialized_sparse, DataType dtype);

  Operation operation;
  ::tensorflow::Output sparse_indices;
  ::tensorflow::Output sparse_values;
  ::tensorflow::Output sparse_shape;
};

/// Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor` object.
///
/// The `SparseTensor` must have rank `R` greater than 1, and the first dimension
/// is treated as the minibatch dimension.  Elements of the `SparseTensor`
/// must be sorted in increasing order of this first dimension.  The serialized
/// `SparseTensor` objects going into each row of `serialized_sparse` will have
/// rank `R-1`.
///
/// The minibatch size `N` is extracted from `sparse_shape[0]`.
///
/// Args:
/// * scope: A Scope object
/// * sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
/// * sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
/// * sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The `dtype` to use for serialization; the supported types are `string`
/// (default) and `variant`.
///
/// Returns:
/// * `Output`: The serialized_sparse tensor.
class SerializeManySparse {
 public:
  /// Optional attribute setters for SerializeManySparse
  struct Attrs {
    /// The `dtype` to use for serialization; the supported types are `string`
    /// (default) and `variant`.
    ///
    /// Defaults to DT_STRING
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_STRING;
  };
  SerializeManySparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    sparse_indices, ::tensorflow::Input sparse_values,
                    ::tensorflow::Input sparse_shape);
  SerializeManySparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    sparse_indices, ::tensorflow::Input sparse_values,
                    ::tensorflow::Input sparse_shape, const
                    SerializeManySparse::Attrs& attrs);
  operator ::tensorflow::Output() const { return serialized_sparse; }
  operator ::tensorflow::Input() const { return serialized_sparse; }
  ::tensorflow::Node* node() const { return serialized_sparse.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output serialized_sparse;
};

/// Serialize a `SparseTensor` into a `[3]` `Tensor` object.
///
/// Args:
/// * scope: A Scope object
/// * sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
/// * sparse_values: 1-D.  The `values` of the `SparseTensor`.
/// * sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The `dtype` to use for serialization; the supported types are `string`
/// (default) and `variant`.
///
/// Returns:
/// * `Output`: The serialized_sparse tensor.
class SerializeSparse {
 public:
  /// Optional attribute setters for SerializeSparse
  struct Attrs {
    /// The `dtype` to use for serialization; the supported types are `string`
    /// (default) and `variant`.
    ///
    /// Defaults to DT_STRING
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_STRING;
  };
  SerializeSparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                sparse_indices, ::tensorflow::Input sparse_values,
                ::tensorflow::Input sparse_shape);
  SerializeSparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                sparse_indices, ::tensorflow::Input sparse_values,
                ::tensorflow::Input sparse_shape, const SerializeSparse::Attrs&
                attrs);
  operator ::tensorflow::Output() const { return serialized_sparse; }
  operator ::tensorflow::Input() const { return serialized_sparse; }
  ::tensorflow::Node* node() const { return serialized_sparse.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output serialized_sparse;
};

/// Adds two `SparseTensor` objects to produce another `SparseTensor`.
///
/// The input `SparseTensor` objects' indices are assumed ordered in standard
/// lexicographic order.  If this is not the case, before this step run
/// `SparseReorder` to restore index ordering.
///
/// By default, if two values sum to zero at some index, the output `SparseTensor`
/// would still include that particular location in its index, storing a zero in the
/// corresponding value slot.  To override this, callers can specify `thresh`,
/// indicating that if the sum has a magnitude strictly smaller than `thresh`, its
/// corresponding value and index would then not be included.  In particular,
/// `thresh == 0` (default) means everything is kept and actual thresholding happens
/// only for a positive value.
///
/// In the following shapes, `nnz` is the count after taking `thresh` into account.
///
/// Args:
/// * scope: A Scope object
/// * a_indices: 2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
/// * a_values: 1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.
/// * a_shape: 1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.
/// * b_indices: 2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.
/// * b_values: 1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.
/// * b_shape: 1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
/// * thresh: 0-D.  The magnitude threshold that determines if an output value/index
/// pair takes space.
///
/// Returns:
/// * `Output` sum_indices
/// * `Output` sum_values
/// * `Output` sum_shape
class SparseAdd {
 public:
  SparseAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input a_indices,
          ::tensorflow::Input a_values, ::tensorflow::Input a_shape,
          ::tensorflow::Input b_indices, ::tensorflow::Input b_values,
          ::tensorflow::Input b_shape, ::tensorflow::Input thresh);

  Operation operation;
  ::tensorflow::Output sum_indices;
  ::tensorflow::Output sum_values;
  ::tensorflow::Output sum_shape;
};

/// The gradient operator for the SparseAdd op.
///
/// The SparseAdd op calculates A + B, where A, B, and the sum are all represented
/// as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
/// non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
/// values of A and B.
///
/// Args:
/// * scope: A Scope object
/// * backprop_val_grad: 1-D with shape `[nnz(sum)]`.  The gradient with respect to
/// the non-empty values of the sum.
/// * a_indices: 2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
/// * b_indices: 2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
/// * sum_indices: 2-D.  The `indices` of the sum `SparseTensor`, size
/// `[nnz(sum), ndims]`.
///
/// Returns:
/// * `Output` a_val_grad: 1-D with shape `[nnz(A)]`. The gradient with respect to the
/// non-empty values of A.
/// * `Output` b_val_grad: 1-D with shape `[nnz(B)]`. The gradient with respect to the
/// non-empty values of B.
class SparseAddGrad {
 public:
  SparseAddGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
              backprop_val_grad, ::tensorflow::Input a_indices,
              ::tensorflow::Input b_indices, ::tensorflow::Input sum_indices);

  Operation operation;
  ::tensorflow::Output a_val_grad;
  ::tensorflow::Output b_val_grad;
};

/// Concatenates a list of `SparseTensor` along the specified dimension.
///
/// Concatenation is with respect to the dense versions of these sparse tensors.
/// It is assumed that each input is a `SparseTensor` whose elements are ordered
/// along increasing dimension number.
///
/// All inputs' shapes must match, except for the concat dimension.  The
/// `indices`, `values`, and `shapes` lists must have the same length.
///
/// The output shape is identical to the inputs', except along the concat
/// dimension, where it is the sum of the inputs' sizes along that dimension.
///
/// The output elements will be resorted to preserve the sort order along
/// increasing dimension number.
///
/// This op runs in `O(M log M)` time, where `M` is the total number of non-empty
/// values across all inputs. This is due to the need for an internal sort in
/// order to concatenate efficiently across an arbitrary dimension.
///
/// For example, if `concat_dim = 1` and the inputs are
///
///     sp_inputs[0]: shape = [2, 3]
///     [0, 2]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
///     sp_inputs[1]: shape = [2, 4]
///     [0, 1]: "d"
///     [0, 2]: "e"
///
/// then the output will be
///
///     shape = [2, 7]
///     [0, 2]: "a"
///     [0, 4]: "d"
///     [0, 5]: "e"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
/// Graphically this is equivalent to doing
///
///     [    a] concat [  d e  ] = [    a   d e  ]
///     [b c  ]        [       ]   [b c          ]
///
/// Args:
/// * scope: A Scope object
/// * indices: 2-D.  Indices of each input `SparseTensor`.
/// * values: 1-D.  Non-empty values of each `SparseTensor`.
/// * shapes: 1-D.  Shapes of each `SparseTensor`.
/// * concat_dim: Dimension to concatenate along. Must be in range [-rank, rank),
/// where rank is the number of dimensions in each input `SparseTensor`.
///
/// Returns:
/// * `Output` output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
/// * `Output` output_values: 1-D.  Non-empty values of the concatenated `SparseTensor`.
/// * `Output` output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
class SparseConcat {
 public:
  SparseConcat(const ::tensorflow::Scope& scope, ::tensorflow::InputList indices,
             ::tensorflow::InputList values, ::tensorflow::InputList shapes,
             int64 concat_dim);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output output_shape;
};

/// Generates sparse cross from a list of sparse and dense tensors.
///
/// The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
/// representing features of one feature column. It outputs a 2D `SparseTensor` with
/// the batchwise crosses of these features.
///
/// For example, if the inputs are
///
///     inputs[0]: SparseTensor with shape = [2, 2]
///     [0, 0]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
///     inputs[1]: SparseTensor with shape = [2, 1]
///     [0, 0]: "d"
///     [1, 0]: "e"
///
///     inputs[2]: Tensor [["f"], ["g"]]
///
/// then the output will be
///
///     shape = [2, 2]
///     [0, 0]: "a_X_d_X_f"
///     [1, 0]: "b_X_e_X_g"
///     [1, 1]: "c_X_e_X_g"
///
/// if hashed_output=true then the output will be
///
///     shape = [2, 2]
///     [0, 0]: FingerprintCat64(
///                 Fingerprint64("f"), FingerprintCat64(
///                     Fingerprint64("d"), Fingerprint64("a")))
///     [1, 0]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("b")))
///     [1, 1]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("c")))
///
/// Args:
/// * scope: A Scope object
/// * indices: 2-D.  Indices of each input `SparseTensor`.
/// * values: 1-D.   values of each `SparseTensor`.
/// * shapes: 1-D.   Shapes of each `SparseTensor`.
/// * dense_inputs: 2-D.    Columns represented by dense `Tensor`.
/// * hashed_output: If true, returns the hash of the cross instead of the string.
/// This will allow us avoiding string manipulations.
/// * num_buckets: It is used if hashed_output is true.
/// output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
/// * hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
/// function to combine the crosses fingerprints.
///
/// Returns:
/// * `Output` output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
/// * `Output` output_values: 1-D.  Non-empty values of the concatenated or hashed
/// `SparseTensor`.
/// * `Output` output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
class SparseCross {
 public:
  SparseCross(const ::tensorflow::Scope& scope, ::tensorflow::InputList indices,
            ::tensorflow::InputList values, ::tensorflow::InputList shapes,
            ::tensorflow::InputList dense_inputs, bool hashed_output, int64
            num_buckets, int64 hash_key, DataType out_type, DataType
            internal_type);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output output_shape;
};

/// Generates sparse cross from a list of sparse and dense tensors.
///
/// The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
/// representing features of one feature column. It outputs a 2D `SparseTensor` with
/// the batchwise crosses of these features.
///
/// For example, if the inputs are
///
///     inputs[0]: SparseTensor with shape = [2, 2]
///     [0, 0]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
///     inputs[1]: SparseTensor with shape = [2, 1]
///     [0, 0]: "d"
///     [1, 0]: "e"
///
///     inputs[2]: Tensor [["f"], ["g"]]
///
/// then the output will be
///
///     shape = [2, 2]
///     [0, 0]: "a_X_d_X_f"
///     [1, 0]: "b_X_e_X_g"
///     [1, 1]: "c_X_e_X_g"
///
/// if hashed_output=true then the output will be
///
///     shape = [2, 2]
///     [0, 0]: FingerprintCat64(
///                 Fingerprint64("f"), FingerprintCat64(
///                     Fingerprint64("d"), Fingerprint64("a")))
///     [1, 0]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("b")))
///     [1, 1]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("c")))
///
/// Args:
/// * scope: A Scope object
/// * indices: 2-D.  Indices of each input `SparseTensor`.
/// * values: 1-D.   values of each `SparseTensor`.
/// * shapes: 1-D.   Shapes of each `SparseTensor`.
/// * dense_inputs: 2-D.    Columns represented by dense `Tensor`.
/// * num_buckets: It is used if hashed_output is true.
/// output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
/// * strong_hash: boolean, if true, siphash with salt will be used instead of farmhash.
/// * salt: Specify the salt that will be used by the siphash function.
///
/// Returns:
/// * `Output` output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
/// * `Output` output_values: 1-D.  Non-empty values of the concatenated or hashed
/// `SparseTensor`.
/// * `Output` output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
class SparseCrossHashed {
 public:
  SparseCrossHashed(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                  indices, ::tensorflow::InputList values,
                  ::tensorflow::InputList shapes, ::tensorflow::InputList
                  dense_inputs, ::tensorflow::Input num_buckets,
                  ::tensorflow::Input strong_hash, ::tensorflow::Input salt);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output output_shape;
};

/// Generates sparse cross from a list of sparse and dense tensors.
///
/// The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
/// representing features of one feature column. It outputs a 2D `SparseTensor` with
/// the batchwise crosses of these features.
///
/// For example, if the inputs are
///
///     inputs[0]: SparseTensor with shape = [2, 2]
///     [0, 0]: "a"
///     [1, 0]: "b"
///     [1, 1]: "c"
///
///     inputs[1]: SparseTensor with shape = [2, 1]
///     [0, 0]: "d"
///     [1, 0]: "e"
///
///     inputs[2]: Tensor [["f"], ["g"]]
///
/// then the output will be
///
///     shape = [2, 2]
///     [0, 0]: "a_X_d_X_f"
///     [1, 0]: "b_X_e_X_g"
///     [1, 1]: "c_X_e_X_g"
///
/// if hashed_output=true then the output will be
///
///     shape = [2, 2]
///     [0, 0]: FingerprintCat64(
///                 Fingerprint64("f"), FingerprintCat64(
///                     Fingerprint64("d"), Fingerprint64("a")))
///     [1, 0]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("b")))
///     [1, 1]: FingerprintCat64(
///                 Fingerprint64("g"), FingerprintCat64(
///                     Fingerprint64("e"), Fingerprint64("c")))
///
/// Args:
/// * scope: A Scope object
/// * indices: 2-D.  Indices of each input `SparseTensor`.
/// * values: 1-D.   values of each `SparseTensor`.
/// * shapes: 1-D.   Shapes of each `SparseTensor`.
/// * dense_inputs: 2-D.    Columns represented by dense `Tensor`.
/// * sep: string used when joining a list of string inputs, can be used as separator later.
///
/// Returns:
/// * `Output` output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
/// * `Output` output_values: 1-D.  Non-empty values of the concatenated or hashed
/// `SparseTensor`.
/// * `Output` output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
class SparseCrossV2 {
 public:
  SparseCrossV2(const ::tensorflow::Scope& scope, ::tensorflow::InputList
              indices, ::tensorflow::InputList values, ::tensorflow::InputList
              shapes, ::tensorflow::InputList dense_inputs, ::tensorflow::Input
              sep);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output output_shape;
};

/// Adds up a SparseTensor and a dense Tensor, using these special rules:
///
/// (1) Broadcasts the dense side to have the same shape as the sparse side, if
///     eligible;
/// (2) Then, only the dense values pointed to by the indices of the SparseTensor
///     participate in the cwise addition.
///
/// By these rules, the result is a logical SparseTensor with exactly the same
/// indices and shape, but possibly with different non-zero values.  The output of
/// this Op is the resultant non-zero values.
///
/// Args:
/// * scope: A Scope object
/// * sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
/// * sp_shape: 1-D.  Shape of the input SparseTensor.
/// * dense: `R`-D.  The dense Tensor operand.
///
/// Returns:
/// * `Output`: 1-D.  The `N` values that are operated on.
class SparseDenseCwiseAdd {
 public:
  SparseDenseCwiseAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    sp_indices, ::tensorflow::Input sp_values,
                    ::tensorflow::Input sp_shape, ::tensorflow::Input dense);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Component-wise divides a SparseTensor by a dense Tensor.
///
/// *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
/// the other direction.
///
/// Args:
/// * scope: A Scope object
/// * sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
/// * sp_shape: 1-D.  Shape of the input SparseTensor.
/// * dense: `R`-D.  The dense Tensor operand.
///
/// Returns:
/// * `Output`: 1-D.  The `N` values that are operated on.
class SparseDenseCwiseDiv {
 public:
  SparseDenseCwiseDiv(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    sp_indices, ::tensorflow::Input sp_values,
                    ::tensorflow::Input sp_shape, ::tensorflow::Input dense);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Component-wise multiplies a SparseTensor by a dense Tensor.
///
/// The output locations corresponding to the implicitly zero elements in the sparse
/// tensor will be zero (i.e., will not take up storage space), regardless of the
/// contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).
///
/// *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
/// the other direction.
///
/// Args:
/// * scope: A Scope object
/// * sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
/// * sp_shape: 1-D.  Shape of the input SparseTensor.
/// * dense: `R`-D.  The dense Tensor operand.
///
/// Returns:
/// * `Output`: 1-D.  The `N` values that are operated on.
class SparseDenseCwiseMul {
 public:
  SparseDenseCwiseMul(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    sp_indices, ::tensorflow::Input sp_values,
                    ::tensorflow::Input sp_shape, ::tensorflow::Input dense);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Fills empty rows in the input 2-D `SparseTensor` with a default value.
///
/// The input `SparseTensor` is represented via the tuple of inputs
/// (`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
/// same `dense_shape` but with indices `output_indices` and values
/// `output_values`.
///
/// This op inserts a single entry for every row that doesn't have any values.
/// The index is created as `[row, 0, ..., 0]` and the inserted value
/// is `default_value`.
///
/// For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:
///
///     [0, 1]: a
///     [0, 3]: b
///     [2, 0]: c
///     [3, 1]: d
///
/// Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:
///
///     [0, 1]: a
///     [0, 3]: b
///     [1, 0]: default_value
///     [2, 0]: c
///     [3, 1]: d
///     [4, 0]: default_value
///
/// The output `SparseTensor` will be in row-major order and will have the
/// same shape as the input.
///
/// This op also returns an indicator vector shaped `[dense_shape[0]]` such that
///
///     empty_row_indicator[i] = True iff row i was an empty row.
///
/// And a reverse index map vector shaped `[indices.shape[0]]` that is used during
/// backpropagation,
///
///     reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]
///
/// Args:
/// * scope: A Scope object
/// * indices: 2-D. the indices of the sparse tensor.
/// * values: 1-D. the values of the sparse tensor.
/// * dense_shape: 1-D. the shape of the sparse tensor.
/// * default_value: 0-D. default value to insert into location `[row, 0, ..., 0]`
///   for rows missing from the input sparse tensor.
/// output indices: 2-D. the indices of the filled sparse tensor.
///
/// Returns:
/// * `Output` output_indices
/// * `Output` output_values: 1-D. the values of the filled sparse tensor.
/// * `Output` empty_row_indicator: 1-D. whether the dense row was missing in the
/// input sparse tensor.
/// * `Output` reverse_index_map: 1-D. a map from the input indices to the output indices.
class SparseFillEmptyRows {
 public:
  SparseFillEmptyRows(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    indices, ::tensorflow::Input values, ::tensorflow::Input
                    dense_shape, ::tensorflow::Input default_value);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output empty_row_indicator;
  ::tensorflow::Output reverse_index_map;
};

/// The gradient of SparseFillEmptyRows.
///
/// Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
/// shaped `[N_full]`, where `N_full >= N` and copies data into either
/// `d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
/// `d_default_value` is a scalar.
///
///   d_values[j] = grad_values[reverse_index_map[j]]
///   d_default_value = sum_{k : 0 .. N_full - 1} (
///      grad_values[k] * 1{k not in reverse_index_map})
///
/// Args:
/// * scope: A Scope object
/// * reverse_index_map: 1-D.  The reverse index map from SparseFillEmptyRows.
/// * grad_values: 1-D.  The gradients from backprop.
///
/// Returns:
/// * `Output` d_values: 1-D.  The backprop into values.
/// * `Output` d_default_value: 0-D.  The backprop into default_value.
class SparseFillEmptyRowsGrad {
 public:
  SparseFillEmptyRowsGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        reverse_index_map, ::tensorflow::Input grad_values);

  Operation operation;
  ::tensorflow::Output d_values;
  ::tensorflow::Output d_default_value;
};

/// Computes the max of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
/// instead of a sparse one.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// Args:
/// * scope: A Scope object
/// * input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// * input_shape: 1-D.  Shape of the input SparseTensor.
/// * reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: `R-K`-D.  The reduced Tensor.
class SparseReduceMax {
 public:
  /// Optional attribute setters for SparseReduceMax
  struct Attrs {
    /// If true, retain reduced dimensions with length 1.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs KeepDims(bool x) {
      Attrs ret = *this;
      ret.keep_dims_ = x;
      return ret;
    }

    bool keep_dims_ = false;
  };
  SparseReduceMax(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_indices, ::tensorflow::Input input_values,
                ::tensorflow::Input input_shape, ::tensorflow::Input
                reduction_axes);
  SparseReduceMax(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_indices, ::tensorflow::Input input_values,
                ::tensorflow::Input input_shape, ::tensorflow::Input
                reduction_axes, const SparseReduceMax::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the max of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_max()`.  In contrast to SparseReduceMax, this Op returns a
/// SparseTensor.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// Args:
/// * scope: A Scope object
/// * input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// * input_shape: 1-D.  Shape of the input SparseTensor.
/// * reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output` output_indices
/// * `Output` output_values
/// * `Output` output_shape
class SparseReduceMaxSparse {
 public:
  /// Optional attribute setters for SparseReduceMaxSparse
  struct Attrs {
    /// If true, retain reduced dimensions with length 1.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs KeepDims(bool x) {
      Attrs ret = *this;
      ret.keep_dims_ = x;
      return ret;
    }

    bool keep_dims_ = false;
  };
  SparseReduceMaxSparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_indices, ::tensorflow::Input input_values,
                      ::tensorflow::Input input_shape, ::tensorflow::Input
                      reduction_axes);
  SparseReduceMaxSparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_indices, ::tensorflow::Input input_values,
                      ::tensorflow::Input input_shape, ::tensorflow::Input
                      reduction_axes, const SparseReduceMaxSparse::Attrs&
                      attrs);

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output output_shape;
};

/// Computes the sum of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
/// instead of a sparse one.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// Args:
/// * scope: A Scope object
/// * input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// * input_shape: 1-D.  Shape of the input SparseTensor.
/// * reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: `R-K`-D.  The reduced Tensor.
class SparseReduceSum {
 public:
  /// Optional attribute setters for SparseReduceSum
  struct Attrs {
    /// If true, retain reduced dimensions with length 1.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs KeepDims(bool x) {
      Attrs ret = *this;
      ret.keep_dims_ = x;
      return ret;
    }

    bool keep_dims_ = false;
  };
  SparseReduceSum(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_indices, ::tensorflow::Input input_values,
                ::tensorflow::Input input_shape, ::tensorflow::Input
                reduction_axes);
  SparseReduceSum(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_indices, ::tensorflow::Input input_values,
                ::tensorflow::Input input_shape, ::tensorflow::Input
                reduction_axes, const SparseReduceSum::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the sum of elements across dimensions of a SparseTensor.
///
/// This Op takes a SparseTensor and is the sparse counterpart to
/// `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
/// SparseTensor.
///
/// Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
/// with length 1.
///
/// If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
/// with a single element is returned.  Additionally, the axes can be negative,
/// which are interpreted according to the indexing rules in Python.
///
/// Args:
/// * scope: A Scope object
/// * input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// * input_shape: 1-D.  Shape of the input SparseTensor.
/// * reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output` output_indices
/// * `Output` output_values
/// * `Output` output_shape
class SparseReduceSumSparse {
 public:
  /// Optional attribute setters for SparseReduceSumSparse
  struct Attrs {
    /// If true, retain reduced dimensions with length 1.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs KeepDims(bool x) {
      Attrs ret = *this;
      ret.keep_dims_ = x;
      return ret;
    }

    bool keep_dims_ = false;
  };
  SparseReduceSumSparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_indices, ::tensorflow::Input input_values,
                      ::tensorflow::Input input_shape, ::tensorflow::Input
                      reduction_axes);
  SparseReduceSumSparse(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input_indices, ::tensorflow::Input input_values,
                      ::tensorflow::Input input_shape, ::tensorflow::Input
                      reduction_axes, const SparseReduceSumSparse::Attrs&
                      attrs);

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output output_shape;
};

/// Reorders a SparseTensor into the canonical, row-major ordering.
///
/// Note that by convention, all sparse ops preserve the canonical ordering along
/// increasing dimension number. The only time ordering can be violated is during
/// manual manipulation of the indices and values vectors to add entries.
///
/// Reordering does not affect the shape of the SparseTensor.
///
/// If the tensor has rank `R` and `N` non-empty values, `input_indices` has
/// shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.
///
/// Args:
/// * scope: A Scope object
/// * input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, possibly not in canonical ordering.
/// * input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
/// * input_shape: 1-D.  Shape of the input SparseTensor.
///
/// Returns:
/// * `Output` output_indices: 2-D.  `N x R` matrix with the same indices as input_indices, but
/// in canonical row-major ordering.
/// * `Output` output_values: 1-D.  `N` non-empty values corresponding to `output_indices`.
class SparseReorder {
 public:
  SparseReorder(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_indices, ::tensorflow::Input input_values,
              ::tensorflow::Input input_shape);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
};

/// Reshapes a SparseTensor to represent values in a new dense shape.
///
/// This operation has the same semantics as reshape on the represented dense
/// tensor.  The `input_indices` are recomputed based on the requested `new_shape`.
///
/// If one component of `new_shape` is the special value -1, the size of that
/// dimension is computed so that the total dense size remains constant.  At
/// most one component of `new_shape` can be -1.  The number of dense elements
/// implied by `new_shape` must be the same as the number of dense elements
/// originally implied by `input_shape`.
///
/// Reshaping does not affect the order of values in the SparseTensor.
///
/// If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
/// has length `R_out`, then `input_indices` has shape `[N, R_in]`,
/// `input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
/// `output_shape` has length `R_out`.
///
/// Args:
/// * scope: A Scope object
/// * input_indices: 2-D.  `N x R_in` matrix with the indices of non-empty values in a
/// SparseTensor.
/// * input_shape: 1-D.  `R_in` vector with the input SparseTensor's dense shape.
/// * new_shape: 1-D.  `R_out` vector with the requested new dense shape.
///
/// Returns:
/// * `Output` output_indices: 2-D.  `N x R_out` matrix with the updated indices of non-empty
/// values in the output SparseTensor.
/// * `Output` output_shape: 1-D.  `R_out` vector with the full dense shape of the output
/// SparseTensor.  This is the same as `new_shape` but with any -1 dimensions
/// filled in.
class SparseReshape {
 public:
  SparseReshape(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_indices, ::tensorflow::Input input_shape,
              ::tensorflow::Input new_shape);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_shape;
};

/// Slice a `SparseTensor` based on the `start` and `size`.
///
/// For example, if the input is
///
///     input_tensor = shape = [2, 7]
///     [    a   d e  ]
///     [b c          ]
///
/// Graphically the output tensors are:
///
///     sparse_slice([0, 0], [2, 4]) = shape = [2, 4]
///     [    a  ]
///     [b c    ]
///
///     sparse_slice([0, 4], [2, 3]) = shape = [2, 3]
///     [ d e  ]
///     [      ]
///
/// Args:
/// * scope: A Scope object
/// * indices: 2-D tensor represents the indices of the sparse tensor.
/// * values: 1-D tensor represents the values of the sparse tensor.
/// * shape: 1-D. tensor represents the shape of the sparse tensor.
/// * start: 1-D. tensor represents the start of the slice.
/// * size: 1-D. tensor represents the size of the slice.
/// output indices: A list of 1-D tensors represents the indices of the output
/// sparse tensors.
///
/// Returns:
/// * `Output` output_indices
/// * `Output` output_values: A list of 1-D tensors represents the values of the output sparse
/// tensors.
/// * `Output` output_shape: A list of 1-D tensors represents the shape of the output sparse
/// tensors.
class SparseSlice {
 public:
  SparseSlice(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
            ::tensorflow::Input values, ::tensorflow::Input shape,
            ::tensorflow::Input start, ::tensorflow::Input size);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
  ::tensorflow::Output output_shape;
};

/// The gradient operator for the SparseSlice op.
///
/// This op takes in the upstream gradient w.r.t. non-empty values of
/// the sliced `SparseTensor`, and outputs the gradients w.r.t.
/// the non-empty values of input `SparseTensor`.
///
/// Args:
/// * scope: A Scope object
/// * backprop_val_grad: 1-D. The gradient with respect to
/// the non-empty values of the sliced `SparseTensor`.
/// * input_indices: 2-D.  The `indices` of the input `SparseTensor`.
/// * input_start: 1-D. tensor represents the start of the slice.
/// * output_indices: 2-D.  The `indices` of the sliced `SparseTensor`.
///
/// Returns:
/// * `Output`: 1-D. The gradient with respect to the non-empty values of input `SparseTensor`.
class SparseSliceGrad {
 public:
  SparseSliceGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                backprop_val_grad, ::tensorflow::Input input_indices,
                ::tensorflow::Input input_start, ::tensorflow::Input
                output_indices);
  operator ::tensorflow::Output() const { return val_grad; }
  operator ::tensorflow::Input() const { return val_grad; }
  ::tensorflow::Node* node() const { return val_grad.node(); }

  Operation operation;
  ::tensorflow::Output val_grad;
};

/// Applies softmax to a batched N-D `SparseTensor`.
///
/// The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
/// (where `N >= 2`), and with indices sorted in the canonical lexicographic order.
///
/// This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
/// logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
/// zero elements do not participate*.  Specifically, the algorithm is equivalent
/// to the following:
///
///   (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
///       with shape `[B, C]`, along the size-C dimension;
///   (2) Masks out the original implicitly-zero locations;
///   (3) Renormalizes the remaining elements.
///
/// Hence, the `SparseTensor` result has exactly the same non-zero indices and
/// shape.
///
/// Args:
/// * scope: A Scope object
/// * sp_indices: 2-D.  `NNZ x R` matrix with the indices of non-empty values in a
/// SparseTensor, in canonical ordering.
/// * sp_values: 1-D.  `NNZ` non-empty values corresponding to `sp_indices`.
/// * sp_shape: 1-D.  Shape of the input SparseTensor.
///
/// Returns:
/// * `Output`: 1-D.  The `NNZ` values for the result `SparseTensor`.
class SparseSoftmax {
 public:
  SparseSoftmax(const ::tensorflow::Scope& scope, ::tensorflow::Input sp_indices,
              ::tensorflow::Input sp_values, ::tensorflow::Input sp_shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the element-wise max of two SparseTensors.
///
/// Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
///
/// Args:
/// * scope: A Scope object
/// * a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, in the canonical lexicographic ordering.
/// * a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
/// * a_shape: 1-D.  Shape of the input SparseTensor.
/// * b_indices: counterpart to `a_indices` for the other operand.
/// * b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
/// * b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
///
/// Returns:
/// * `Output` output_indices: 2-D.  The indices of the output SparseTensor.
/// * `Output` output_values: 1-D.  The values of the output SparseTensor.
class SparseSparseMaximum {
 public:
  SparseSparseMaximum(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    a_indices, ::tensorflow::Input a_values,
                    ::tensorflow::Input a_shape, ::tensorflow::Input b_indices,
                    ::tensorflow::Input b_values, ::tensorflow::Input b_shape);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
};

/// Returns the element-wise min of two SparseTensors.
///
/// Assumes the two SparseTensors have the same shape, i.e., no broadcasting.
///
/// Args:
/// * scope: A Scope object
/// * a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
/// SparseTensor, in the canonical lexicographic ordering.
/// * a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
/// * a_shape: 1-D.  Shape of the input SparseTensor.
/// * b_indices: counterpart to `a_indices` for the other operand.
/// * b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
/// * b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.
///
/// Returns:
/// * `Output` output_indices: 2-D.  The indices of the output SparseTensor.
/// * `Output` output_values: 1-D.  The values of the output SparseTensor.
class SparseSparseMinimum {
 public:
  SparseSparseMinimum(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    a_indices, ::tensorflow::Input a_values,
                    ::tensorflow::Input a_shape, ::tensorflow::Input b_indices,
                    ::tensorflow::Input b_values, ::tensorflow::Input b_shape);

  Operation operation;
  ::tensorflow::Output output_indices;
  ::tensorflow::Output output_values;
};

/// Split a `SparseTensor` into `num_split` tensors along one dimension.
///
/// If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
/// `[0 : shape[split_dim] % num_split]` gets one extra dimension.
/// For example, if `split_dim = 1` and `num_split = 2` and the input is
///
///     input_tensor = shape = [2, 7]
///     [    a   d e  ]
///     [b c          ]
///
/// Graphically the output tensors are:
///
///     output_tensor[0] = shape = [2, 4]
///     [    a  ]
///     [b c    ]
///
///     output_tensor[1] = shape = [2, 3]
///     [ d e  ]
///     [      ]
///
/// Args:
/// * scope: A Scope object
/// * split_dim: 0-D.  The dimension along which to split.  Must be in the range
/// `[0, rank(shape))`.
/// * indices: 2-D tensor represents the indices of the sparse tensor.
/// * values: 1-D tensor represents the values of the sparse tensor.
/// * shape: 1-D. tensor represents the shape of the sparse tensor.
/// output indices: A list of 1-D tensors represents the indices of the output
/// sparse tensors.
/// * num_split: The number of ways to split.
///
/// Returns:
/// * `OutputList` output_indices
/// * `OutputList` output_values: A list of 1-D tensors represents the values of the output sparse
/// tensors.
/// * `OutputList` output_shape: A list of 1-D tensors represents the shape of the output sparse
/// tensors.
class SparseSplit {
 public:
  SparseSplit(const ::tensorflow::Scope& scope, ::tensorflow::Input split_dim,
            ::tensorflow::Input indices, ::tensorflow::Input values,
            ::tensorflow::Input shape, int64 num_split);

  Operation operation;
  ::tensorflow::OutputList output_indices;
  ::tensorflow::OutputList output_values;
  ::tensorflow::OutputList output_shape;
};

/// Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.
///
/// This Op does not require `a_indices` be sorted in standard lexicographic order.
///
/// Args:
/// * scope: A Scope object
/// * a_indices: 2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
/// * a_values: 1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
/// * a_shape: 1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
/// * b: `ndims`-D Tensor.  With shape `a_shape`.
///
/// Returns:
/// * `Output`: The output tensor.
class SparseTensorDenseAdd {
 public:
  SparseTensorDenseAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     a_indices, ::tensorflow::Input a_values,
                     ::tensorflow::Input a_shape, ::tensorflow::Input b);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Multiply SparseTensor (of rank 2) "A" by dense matrix "B".
///
/// No validity checking is performed on the indices of A.  However, the following
/// input format is recommended for optimal behavior:
///
/// if adjoint_a == false:
///   A should be sorted in lexicographically increasing order.  Use SparseReorder
///   if you're not sure.
/// if adjoint_a == true:
///   A should be sorted in order of increasing dimension 1 (i.e., "column major"
///   order instead of "row major" order).
///
/// Args:
/// * scope: A Scope object
/// * a_indices: 2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
/// * a_values: 1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
/// * a_shape: 1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
/// * b: 2-D.  A dense Matrix.
///
/// Optional attributes (see `Attrs`):
/// * adjoint_a: Use the adjoint of A in the matrix multiply.  If A is complex, this
/// is transpose(conj(A)).  Otherwise it's transpose(A).
/// * adjoint_b: Use the adjoint of B in the matrix multiply.  If B is complex, this
/// is transpose(conj(B)).  Otherwise it's transpose(B).
///
/// Returns:
/// * `Output`: The product tensor.
class SparseTensorDenseMatMul {
 public:
  /// Optional attribute setters for SparseTensorDenseMatMul
  struct Attrs {
    /// Use the adjoint of A in the matrix multiply.  If A is complex, this
    /// is transpose(conj(A)).  Otherwise it's transpose(A).
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjointA(bool x) {
      Attrs ret = *this;
      ret.adjoint_a_ = x;
      return ret;
    }

    /// Use the adjoint of B in the matrix multiply.  If B is complex, this
    /// is transpose(conj(B)).  Otherwise it's transpose(B).
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjointB(bool x) {
      Attrs ret = *this;
      ret.adjoint_b_ = x;
      return ret;
    }

    bool adjoint_a_ = false;
    bool adjoint_b_ = false;
  };
  SparseTensorDenseMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        a_indices, ::tensorflow::Input a_values,
                        ::tensorflow::Input a_shape, ::tensorflow::Input b);
  SparseTensorDenseMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        a_indices, ::tensorflow::Input a_values,
                        ::tensorflow::Input a_shape, ::tensorflow::Input b,
                        const SparseTensorDenseMatMul::Attrs& attrs);
  operator ::tensorflow::Output() const { return product; }
  operator ::tensorflow::Input() const { return product; }
  ::tensorflow::Node* node() const { return product.node(); }

  static Attrs AdjointA(bool x) {
    return Attrs().AdjointA(x);
  }
  static Attrs AdjointB(bool x) {
    return Attrs().AdjointB(x);
  }

  Operation operation;
  ::tensorflow::Output product;
};

/// Converts a sparse representation into a dense tensor.
///
/// Builds an array `dense` with shape `output_shape` such that
///
/// ```
/// # If sparse_indices is scalar
/// dense[i] = (i == sparse_indices ? sparse_values : default_value)
///
/// # If sparse_indices is a vector, then for each i
/// dense[sparse_indices[i]] = sparse_values[i]
///
/// # If sparse_indices is an n by d matrix, then for each i in [0, n)
/// dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
/// ```
///
/// All other values in `dense` are set to `default_value`.  If `sparse_values` is a
/// scalar, all sparse indices are set to this single value.
///
/// Indices should be sorted in lexicographic order, and indices must not
/// contain any repeats. If `validate_indices` is true, these properties
/// are checked during execution.
///
/// Args:
/// * scope: A Scope object
/// * sparse_indices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
/// index where `sparse_values[i]` will be placed.
/// * output_shape: 1-D.  Shape of the dense output tensor.
/// * sparse_values: 1-D.  Values corresponding to each row of `sparse_indices`,
/// or a scalar value to be used for all sparse indices.
/// * default_value: Scalar value to set for indices not specified in
/// `sparse_indices`.
///
/// Optional attributes (see `Attrs`):
/// * validate_indices: If true, indices are checked to make sure they are sorted in
/// lexicographic order and that there are no repeats.
///
/// Returns:
/// * `Output`: Dense output tensor of shape `output_shape`.
class SparseToDense {
 public:
  /// Optional attribute setters for SparseToDense
  struct Attrs {
    /// If true, indices are checked to make sure they are sorted in
    /// lexicographic order and that there are no repeats.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ValidateIndices(bool x) {
      Attrs ret = *this;
      ret.validate_indices_ = x;
      return ret;
    }

    bool validate_indices_ = true;
  };
  SparseToDense(const ::tensorflow::Scope& scope, ::tensorflow::Input
              sparse_indices, ::tensorflow::Input output_shape,
              ::tensorflow::Input sparse_values, ::tensorflow::Input
              default_value);
  SparseToDense(const ::tensorflow::Scope& scope, ::tensorflow::Input
              sparse_indices, ::tensorflow::Input output_shape,
              ::tensorflow::Input sparse_values, ::tensorflow::Input
              default_value, const SparseToDense::Attrs& attrs);
  operator ::tensorflow::Output() const { return dense; }
  operator ::tensorflow::Input() const { return dense; }
  ::tensorflow::Node* node() const { return dense.node(); }

  static Attrs ValidateIndices(bool x) {
    return Attrs().ValidateIndices(x);
  }

  Operation operation;
  ::tensorflow::Output dense;
};

/// Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.
///
/// The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
/// `N` is the minibatch size and the rows correspond to the output handles of
/// `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
/// original `SparseTensor` objects that went into the given input ops must all
/// match.  When the final `SparseTensor` is created, it has rank one
/// higher than the ranks of the incoming `SparseTensor` objects
/// (they have been concatenated along a new row dimension on the left).
///
/// The output `SparseTensor` object's shape values for all dimensions but the
/// first are the max across the input `SparseTensor` objects' shape values
/// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
/// size.
///
/// The input `SparseTensor` objects' indices are assumed ordered in
/// standard lexicographic order.  If this is not the case, after this
/// step run `SparseReorder` to restore index ordering.
///
/// For example, if the handles represent an input, which is a `[2, 3]` matrix
/// representing two original `SparseTensor` objects:
///
/// ```
///     index = [ 0]
///             [10]
///             [20]
///     values = [1, 2, 3]
///     shape = [50]
/// ```
///
/// and
///
/// ```
///     index = [ 2]
///             [10]
///     values = [4, 5]
///     shape = [30]
/// ```
///
/// then the final `SparseTensor` will be:
///
/// ```
///     index = [0  0]
///             [0 10]
///             [0 20]
///             [1  2]
///             [1 10]
///     values = [1, 2, 3, 4, 5]
///     shape = [2 50]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * sparse_handles: 1-D, The `N` serialized `SparseTensor` objects.
/// Shape: `[N]`.
/// * dtype: The `dtype` of the `SparseTensor` objects stored in the
/// `SparseTensorsMap`.
///
/// Optional attributes (see `Attrs`):
/// * container: The container name for the `SparseTensorsMap` read by this op.
/// * shared_name: The shared name for the `SparseTensorsMap` read by this op.
/// It should not be blank; rather the `shared_name` or unique Operation name
/// of the Op that created the original `SparseTensorsMap` should be used.
///
/// Returns:
/// * `Output` sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
/// * `Output` sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
/// * `Output` sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
class TakeManySparseFromTensorsMap {
 public:
  /// Optional attribute setters for TakeManySparseFromTensorsMap
  struct Attrs {
    /// The container name for the `SparseTensorsMap` read by this op.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// The shared name for the `SparseTensorsMap` read by this op.
    /// It should not be blank; rather the `shared_name` or unique Operation name
    /// of the Op that created the original `SparseTensorsMap` should be used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  TakeManySparseFromTensorsMap(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input sparse_handles, DataType
                             dtype);
  TakeManySparseFromTensorsMap(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input sparse_handles, DataType
                             dtype, const TakeManySparseFromTensorsMap::Attrs&
                             attrs);

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output sparse_indices;
  ::tensorflow::Output sparse_values;
  ::tensorflow::Output sparse_shape;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_SPARSE_OPS_H_
