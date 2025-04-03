// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_STATE_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_STATE_OPS_INTERNAL_H_

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

/// @defgroup state_ops_internal State Ops Internal
/// @{

/// Computes element-wise maximum.
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
class ScatterNdMax {
 public:
  /// Optional attribute setters for ScatterNdMax
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
  ScatterNdMax(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterNdMax(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates, const
             ScatterNdMax::Attrs& attrs);
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

/// Computes element-wise minimum.
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
class ScatterNdMin {
 public:
  /// Optional attribute setters for ScatterNdMin
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
  ScatterNdMin(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates);
  ScatterNdMin(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
             ::tensorflow::Input indices, ::tensorflow::Input updates, const
             ScatterNdMin::Attrs& attrs);
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

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_STATE_OPS_INTERNAL_H_
