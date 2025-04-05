// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_STATE_OPS_H_
#define TENSORFLOW_CC_OPS_STATE_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup state_ops State Ops
/// @{

/// Update 'ref' by assigning 'value' to it.
///
/// This operation outputs "ref" after the assignment is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node. May be uninitialized.
/// * value: The value to be assigned to the variable.
///
/// Optional attributes (see `Attrs`):
/// * validate_shape: If true, the operation will validate that the shape
/// of 'value' matches the shape of the Tensor being assigned to.  If false,
/// 'ref' will take on the shape of 'value'.
/// * use_locking: If True, the assignment will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as "ref".  Returned as a convenience for operations that want
/// to use the new value after the variable has been reset.
class Assign {
 public:
  /// Optional attribute setters for Assign
  struct Attrs {
    /// If true, the operation will validate that the shape
    /// of 'value' matches the shape of the Tensor being assigned to.  If false,
    /// 'ref' will take on the shape of 'value'.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ValidateShape(bool x) {
      Attrs ret = *this;
      ret.validate_shape_ = x;
      return ret;
    }

    /// If True, the assignment will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool validate_shape_ = true;
    bool use_locking_ = true;
  };
  Assign(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
       ::tensorflow::Input value);
  Assign(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
       ::tensorflow::Input value, const Assign::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs ValidateShape(bool x) {
    return Attrs().ValidateShape(x);
  }
  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Update 'ref' by adding 'value' to it.
///
/// This operation outputs "ref" after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * value: The value to be added to the variable.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the addition will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as "ref".  Returned as a convenience for operations that want
/// to use the new value after the variable has been updated.
class AssignAdd {
 public:
  /// Optional attribute setters for AssignAdd
  struct Attrs {
    /// If True, the addition will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  AssignAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
          ::tensorflow::Input value);
  AssignAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
          ::tensorflow::Input value, const AssignAdd::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Update 'ref' by subtracting 'value' from it.
///
/// This operation outputs "ref" after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * value: The value to be subtracted to the variable.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as "ref".  Returned as a convenience for operations that want
/// to use the new value after the variable has been updated.
class AssignSub {
 public:
  /// Optional attribute setters for AssignSub
  struct Attrs {
    /// If True, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  AssignSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
          ::tensorflow::Input value);
  AssignSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
          ::tensorflow::Input value, const AssignSub::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Increments 'ref' until it reaches 'limit'.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a scalar `Variable` node.
/// * limit: If incrementing ref would bring it above limit, instead generates an
/// 'OutOfRange' error.
///
/// Returns:
/// * `Output`: A copy of the input before increment. If nothing else modifies the
/// input, the values produced will all be distinct.
class CountUpTo {
 public:
  CountUpTo(const ::tensorflow::Scope& scope, ::tensorflow::Input ref, int64
          limit);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Destroys the temporary variable and returns its final value.
///
/// Sets output to the value of the Tensor pointed to by 'ref', then destroys
/// the temporary variable called 'var_name'.
/// All other uses of 'ref' *must* have executed before this op.
/// This is typically achieved by chaining the ref through each assign op, or by
/// using control dependencies.
///
/// Outputs the final value of the tensor pointed to by 'ref'.
///
/// Args:
/// * scope: A Scope object
/// * ref: A reference to the temporary variable tensor.
/// * var_name: Name of the temporary variable, usually the name of the matching
/// 'TemporaryVariable' op.
///
/// Returns:
/// * `Output`: The value tensor.
class DestroyTemporaryVariable {
 public:
  DestroyTemporaryVariable(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         ref, StringPiece var_name);
  operator ::tensorflow::Output() const { return value; }
  operator ::tensorflow::Input() const { return value; }
  ::tensorflow::Node* node() const { return value.node(); }

  Operation operation;
  ::tensorflow::Output value;
};

/// Checks whether a tensor has been initialized.
///
/// Outputs boolean scalar indicating whether the tensor has been initialized.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node. May be uninitialized.
///
/// Returns:
/// * `Output`: The is_initialized tensor.
class IsVariableInitialized {
 public:
  IsVariableInitialized(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      ref);
  operator ::tensorflow::Output() const { return is_initialized; }
  operator ::tensorflow::Input() const { return is_initialized; }
  ::tensorflow::Node* node() const { return is_initialized.node(); }

  Operation operation;
  ::tensorflow::Output is_initialized;
};

/// Increments variable pointed to by 'resource' until it reaches 'limit'.
///
/// Args:
/// * scope: A Scope object
/// * resource: Should be from a scalar `Variable` node.
/// * limit: If incrementing ref would bring it above limit, instead generates an
/// 'OutOfRange' error.
///
/// Returns:
/// * `Output`: A copy of the input before increment. If nothing else modifies the
/// input, the values produced will all be distinct.
class ResourceCountUpTo {
 public:
  ResourceCountUpTo(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  resource, int64 limit, DataType T);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Applies sparse addition to individual values or slices in a Variable.
///
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
/// ```
///
/// For example, say we want to add 4 scattered elements to a rank-1 tensor to
/// 8 elements. In Python, that addition would look like this:
///
/// ```python
/// ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8], use_resource=True)
/// indices = tf.constant([[4], [3], [1], [7]])
/// updates = tf.constant([9, 10, 11, 12])
/// add = tf.scatter_nd_add(ref, indices, updates)
/// with tf.Session() as sess:
///   print sess.run(add)
/// ```
///
/// The resulting update to ref would look like this:
///
///     [1, 13, 3, 14, 14, 6, 7, 20]
///
/// See `tf.scatter_nd` for more details about how to make updates to
/// slices.
///
/// Args:
/// * scope: A Scope object
/// * ref: A resource handle. Must be from a VarHandleOp.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of
/// values to add to ref.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceScatterNdAdd {
 public:
  /// Optional attribute setters for ResourceScatterNdAdd
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = true;
    StringPiece bad_indices_policy_ = "";
  };
  ResourceScatterNdAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates);
  ResourceScatterNdAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates,
                     const ResourceScatterNdAdd::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
/// * ref: A resource handle. Must be from a VarHandleOp.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of
/// values whose element wise max is taken with ref
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceScatterNdMax {
 public:
  /// Optional attribute setters for ResourceScatterNdMax
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = true;
    StringPiece bad_indices_policy_ = "";
  };
  ResourceScatterNdMax(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates);
  ResourceScatterNdMax(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates,
                     const ResourceScatterNdMax::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
/// * ref: A resource handle. Must be from a VarHandleOp.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of
/// values whose element wise min is taken with ref.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceScatterNdMin {
 public:
  /// Optional attribute setters for ResourceScatterNdMin
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = true;
    StringPiece bad_indices_policy_ = "";
  };
  ResourceScatterNdMin(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates);
  ResourceScatterNdMin(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates,
                     const ResourceScatterNdMin::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
};

/// Applies sparse subtraction to individual values or slices in a Variable.
///
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
/// ```
///
/// For example, say we want to subtract 4 scattered elements from a rank-1 tensor
/// with 8 elements. In Python, that subtraction would look like this:
///
/// ```python
/// ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8], use_resource=True)
/// indices = tf.constant([[4], [3], [1], [7]])
/// updates = tf.constant([9, 10, 11, 12])
/// sub = tf.scatter_nd_sub(ref, indices, updates)
/// with tf.Session() as sess:
///   print sess.run(sub)
/// ```
///
/// The resulting update to ref would look like this:
///
///     [1, -9, 3, -6, -4, 6, 7, -4]
///
/// See `tf.scatter_nd` for more details about how to make updates to
/// slices.
///
/// Args:
/// * scope: A Scope object
/// * ref: A resource handle. Must be from a VarHandleOp.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of
/// values to add to ref.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceScatterNdSub {
 public:
  /// Optional attribute setters for ResourceScatterNdSub
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = true;
    StringPiece bad_indices_policy_ = "";
  };
  ResourceScatterNdSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates);
  ResourceScatterNdSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                     ::tensorflow::Input indices, ::tensorflow::Input updates,
                     const ResourceScatterNdSub::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
};

/// Applies sparse `updates` to individual values or slices within a given
///
/// variable according to `indices`.
///
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
/// ```
///
/// For example, say we want to update 4 scattered elements to a rank-1 tensor to
/// 8 elements. In Python, that update would look like this:
///
/// ```python
///     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1] ,[7]])
///     updates = tf.constant([9, 10, 11, 12])
///     update = tf.scatter_nd_update(ref, indices, updates)
///     with tf.Session() as sess:
///       print sess.run(update)
/// ```
///
/// The resulting update to ref would look like this:
///
///     [1, 11, 3, 10, 9, 6, 7, 12]
///
/// See `tf.scatter_nd` for more details about how to make updates to
/// slices.
///
/// Args:
/// * scope: A Scope object
/// * ref: A resource handle. Must be from a VarHandleOp.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of updated
/// values to add to ref.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceScatterNdUpdate {
 public:
  /// Optional attribute setters for ResourceScatterNdUpdate
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = true;
    StringPiece bad_indices_policy_ = "";
  };
  ResourceScatterNdUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        ref, ::tensorflow::Input indices, ::tensorflow::Input
                        updates);
  ResourceScatterNdUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        ref, ::tensorflow::Input indices, ::tensorflow::Input
                        updates, const ResourceScatterNdUpdate::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
};

/// Adds sparse updates to a variable reference.
///
/// This operation computes
///
///     # Scalar indices
///     ref[indices, ...] += updates[...]
///
///     # Vector indices (for each i)
///     ref[indices[i], ...] += updates[i, ...]
///
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
///
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions add.
///
/// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
/// </div>
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * indices: A tensor of indices into the first dimension of `ref`.
/// * updates: A tensor of updated values to add to `ref`.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the addition will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterAdd {
 public:
  /// Optional attribute setters for ScatterAdd
  struct Attrs {
    /// If True, the addition will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ScatterAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates, const
           ScatterAdd::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Divides a variable reference by sparse updates.
///
/// This operation computes
///
/// ```python
///     # Scalar indices
///     ref[indices, ...] /= updates[...]
///
///     # Vector indices (for each i)
///     ref[indices[i], ...] /= updates[i, ...]
///
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]
/// ```
///
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions divide.
///
/// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * indices: A tensor of indices into the first dimension of `ref`.
/// * updates: A tensor of values that `ref` is divided by.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the operation will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterDiv {
 public:
  /// Optional attribute setters for ScatterDiv
  struct Attrs {
    /// If True, the operation will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ScatterDiv(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterDiv(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates, const
           ScatterDiv::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Reduces sparse updates into a variable reference using the `max` operation.
///
/// This operation computes
///
///     # Scalar indices
///     ref[indices, ...] = max(ref[indices, ...], updates[...])
///
///     # Vector indices (for each i)
///     ref[indices[i], ...] = max(ref[indices[i], ...], updates[i, ...])
///
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] = max(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
///
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions combine.
///
/// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
/// </div>
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * indices: A tensor of indices into the first dimension of `ref`.
/// * updates: A tensor of updated values to reduce into `ref`.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the update will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterMax {
 public:
  /// Optional attribute setters for ScatterMax
  struct Attrs {
    /// If True, the update will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ScatterMax(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterMax(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates, const
           ScatterMax::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Reduces sparse updates into a variable reference using the `min` operation.
///
/// This operation computes
///
///     # Scalar indices
///     ref[indices, ...] = min(ref[indices, ...], updates[...])
///
///     # Vector indices (for each i)
///     ref[indices[i], ...] = min(ref[indices[i], ...], updates[i, ...])
///
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] = min(ref[indices[i, ..., j], ...], updates[i, ..., j, ...])
///
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions combine.
///
/// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
/// </div>
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * indices: A tensor of indices into the first dimension of `ref`.
/// * updates: A tensor of updated values to reduce into `ref`.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the update will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterMin {
 public:
  /// Optional attribute setters for ScatterMin
  struct Attrs {
    /// If True, the update will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ScatterMin(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterMin(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates, const
           ScatterMin::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Multiplies sparse updates into a variable reference.
///
/// This operation computes
///
/// ```python
///     # Scalar indices
///     ref[indices, ...] *= updates[...]
///
///     # Vector indices (for each i)
///     ref[indices[i], ...] *= updates[i, ...]
///
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] *= updates[i, ..., j, ...]
/// ```
///
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their contributions multiply.
///
/// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * indices: A tensor of indices into the first dimension of `ref`.
/// * updates: A tensor of updated values to multiply to `ref`.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the operation will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterMul {
 public:
  /// Optional attribute setters for ScatterMul
  struct Attrs {
    /// If True, the operation will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ScatterMul(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterMul(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates, const
           ScatterMul::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Applies sparse addition to individual values or slices in a Variable.
///
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
/// ```
///
/// For example, say we want to add 4 scattered elements to a rank-1 tensor to
/// 8 elements. In Python, that addition would look like this:
///
/// ```python
/// ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
/// indices = tf.constant([[4], [3], [1], [7]])
/// updates = tf.constant([9, 10, 11, 12])
/// add = tf.scatter_nd_add(ref, indices, updates)
/// with tf.Session() as sess:
///   print sess.run(add)
/// ```
///
/// The resulting update to ref would look like this:
///
///     [1, 13, 3, 14, 14, 6, 7, 20]
///
/// See `tf.scatter_nd` for more details about how to make updates to
/// slices.
///
/// Args:
/// * scope: A Scope object
/// * ref: A mutable Tensor. Should be from a Variable node.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of updated values
/// to add to ref.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as ref. Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterNdAdd {
 public:
  /// Optional attribute setters for ScatterNdAdd
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = false;
    StringPiece bad_indices_policy_ = "";
  };
  ScatterNdAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterNdAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates, const
             ScatterNdAdd::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Applies sparse subtraction to individual values or slices in a Variable.
///
/// within a given variable according to `indices`.
///
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// ```
/// [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]
/// ```
///
/// For example, say we want to subtract 4 scattered elements from a rank-1 tensor
/// with 8 elements. In Python, that subtraction would look like this:
///
/// ```python
/// ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
/// indices = tf.constant([[4], [3], [1], [7]])
/// updates = tf.constant([9, 10, 11, 12])
/// sub = tf.scatter_nd_sub(ref, indices, updates)
/// with tf.Session() as sess:
///   print sess.run(sub)
/// ```
///
/// The resulting update to ref would look like this:
///
///     [1, -9, 3, -6, -4, 6, 7, -4]
///
/// See `tf.scatter_nd` for more details about how to make updates to
/// slices.
///
/// Args:
/// * scope: A Scope object
/// * ref: A mutable Tensor. Should be from a Variable node.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of updated values
/// to subtract from ref.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as ref. Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterNdSub {
 public:
  /// Optional attribute setters for ScatterNdSub
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = false;
    StringPiece bad_indices_policy_ = "";
  };
  ScatterNdSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterNdSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates, const
             ScatterNdSub::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Applies sparse `updates` to individual values or slices within a given
///
/// variable according to `indices`.
///
/// `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `ref`.
/// It must be shape \\([d_0, ..., d_{Q-2}, K]\\) where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
/// dimension of `ref`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// $$[d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].$$
///
/// For example, say we want to update 4 scattered elements to a rank-1 tensor to
/// 8 elements. In Python, that update would look like this:
///
/// ```python
///     ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1] ,[7]])
///     updates = tf.constant([9, 10, 11, 12])
///     update = tf.scatter_nd_update(ref, indices, updates)
///     with tf.Session() as sess:
///       print sess.run(update)
/// ```
///
/// The resulting update to ref would look like this:
///
///     [1, 11, 3, 10, 9, 6, 7, 12]
///
/// See `tf.scatter_nd` for more details about how to make updates to
/// slices.
///
/// See also `tf.scatter_update` and `tf.batch_scatter_update`.
///
/// Args:
/// * scope: A Scope object
/// * ref: A mutable Tensor. Should be from a Variable node.
/// * indices: A Tensor. Must be one of the following types: int32, int64.
/// A tensor of indices into ref.
/// * updates: A Tensor. Must have the same type as ref. A tensor of updated
/// values to add to ref.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: An optional bool. Defaults to True. If True, the assignment will
/// be protected by a lock; otherwise the behavior is undefined,
/// but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as ref. Returned as a convenience for operations that want to
/// use the updated values after the update is done.
class ScatterNdUpdate {
 public:
  /// Optional attribute setters for ScatterNdUpdate
  struct Attrs {
    /// An optional bool. Defaults to True. If True, the assignment will
    /// be protected by a lock; otherwise the behavior is undefined,
    /// but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    bool use_locking_ = true;
    StringPiece bad_indices_policy_ = "";
  };
  ScatterNdUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterNdUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                ::tensorflow::Input indices, ::tensorflow::Input updates, const
                ScatterNdUpdate::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Subtracts sparse updates to a variable reference.
///
/// ```python
///     # Scalar indices
///     ref[indices, ...] -= updates[...]
///
///     # Vector indices (for each i)
///     ref[indices[i], ...] -= updates[i, ...]
///
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
/// ```
///
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// Duplicate entries are handled correctly: if multiple `indices` reference
/// the same location, their (negated) contributions add.
///
/// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterSub.png" alt>
/// </div>
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * indices: A tensor of indices into the first dimension of `ref`.
/// * updates: A tensor of updated values to subtract from `ref`.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterSub {
 public:
  /// Optional attribute setters for ScatterSub
  struct Attrs {
    /// If True, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ScatterSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterSub(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
           ::tensorflow::Input indices, ::tensorflow::Input updates, const
           ScatterSub::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Applies sparse updates to a variable reference.
///
/// This operation computes
///
/// ```python
///     # Scalar indices
///     ref[indices, ...] = updates[...]
///
///     # Vector indices (for each i)
///     ref[indices[i], ...] = updates[i, ...]
///
///     # High rank indices (for each i, ..., j)
///     ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
/// ```
///
/// This operation outputs `ref` after the update is done.
/// This makes it easier to chain operations that need to use the reset value.
///
/// If values in `ref` is to be updated more than once, because there are
/// duplicate entries in `indices`, the order at which the updates happen
/// for each value is undefined.
///
/// Requires `updates.shape = indices.shape + ref.shape[1:]` or `updates.shape = []`.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterUpdate.png" alt>
/// </div>
///
/// See also `tf.batch_scatter_update` and `tf.scatter_nd_update`.
///
/// Args:
/// * scope: A Scope object
/// * ref: Should be from a `Variable` node.
/// * indices: A tensor of indices into the first dimension of `ref`.
/// * updates: A tensor of updated values to store in `ref`.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the assignment will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: = Same as `ref`.  Returned as a convenience for operations that want
/// to use the updated values after the update is done.
class ScatterUpdate {
 public:
  /// Optional attribute setters for ScatterUpdate
  struct Attrs {
    /// If True, the assignment will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = true;
  };
  ScatterUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
              ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
              ::tensorflow::Input indices, ::tensorflow::Input updates, const
              ScatterUpdate::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Returns a tensor that may be mutated, but only persists within a single step.
///
/// This is an experimental op for internal use only and it is possible to use this
/// op in unsafe ways.  DO NOT USE unless you fully understand the risks.
///
/// It is the caller's responsibility to ensure that 'ref' is eventually passed to a
/// matching 'DestroyTemporaryVariable' op after all other uses have completed.
///
/// Outputs a ref to the tensor state so it may be read or modified.
///
///   E.g.
///       var = state_ops._temporary_variable([1, 2], types.float_)
///       var_name = var.op.name
///       var = state_ops.assign(var, [[4.0, 5.0]])
///       var = state_ops.assign_add(var, [[6.0, 7.0]])
///       final = state_ops._destroy_temporary_variable(var, var_name=var_name)
///
/// Args:
/// * scope: A Scope object
/// * shape: The shape of the variable tensor.
/// * dtype: The type of elements in the variable tensor.
///
/// Optional attributes (see `Attrs`):
/// * var_name: Overrides the name used for the temporary variable resource. Default
/// value is the name of the 'TemporaryVariable' op (which is guaranteed unique).
///
/// Returns:
/// * `Output`: A reference to the variable tensor.
class TemporaryVariable {
 public:
  /// Optional attribute setters for TemporaryVariable
  struct Attrs {
    /// Overrides the name used for the temporary variable resource. Default
    /// value is the name of the 'TemporaryVariable' op (which is guaranteed unique).
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs VarName(StringPiece x) {
      Attrs ret = *this;
      ret.var_name_ = x;
      return ret;
    }

    StringPiece var_name_ = "";
  };
  TemporaryVariable(const ::tensorflow::Scope& scope, PartialTensorShape shape,
                  DataType dtype);
  TemporaryVariable(const ::tensorflow::Scope& scope, PartialTensorShape shape,
                  DataType dtype, const TemporaryVariable::Attrs& attrs);
  operator ::tensorflow::Output() const { return ref; }
  operator ::tensorflow::Input() const { return ref; }
  ::tensorflow::Node* node() const { return ref.node(); }

  static Attrs VarName(StringPiece x) {
    return Attrs().VarName(x);
  }

  Operation operation;
  ::tensorflow::Output ref;
};

/// Holds state in the form of a tensor that persists across steps.
///
/// Outputs a ref to the tensor state so it may be read or modified.
/// TODO(zhifengc/mrry): Adds a pointer to a more detail document
/// about sharing states in tensorflow.
///
/// Args:
/// * scope: A Scope object
/// * shape: The shape of the variable tensor.
/// * dtype: The type of elements in the variable tensor.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this variable is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this variable is named in the given bucket
/// with this shared_name. Otherwise, the node name is used instead.
///
/// Returns:
/// * `Output`: A reference to the variable tensor.
class Variable {
 public:
  /// Optional attribute setters for Variable
  struct Attrs {
    /// If non-empty, this variable is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this variable is named in the given bucket
    /// with this shared_name. Otherwise, the node name is used instead.
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
  Variable(const ::tensorflow::Scope& scope, PartialTensorShape shape, DataType
         dtype);
  Variable(const ::tensorflow::Scope& scope, PartialTensorShape shape, DataType
         dtype, const Variable::Attrs& attrs);
  operator ::tensorflow::Output() const { return ref; }
  operator ::tensorflow::Input() const { return ref; }
  ::tensorflow::Node* node() const { return ref.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output ref;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_STATE_OPS_H_
