// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_ARRAY_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_ARRAY_OPS_INTERNAL_H_

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

/// @defgroup array_ops_internal Array Ops Internal
/// @{

/// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
///
/// This is typically used by gradient computations for a broadcasting operation.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` r0
/// * `Output` r1
class BroadcastGradientArgs {
 public:
  BroadcastGradientArgs(const ::tensorflow::Scope& scope, ::tensorflow::Input s0,
                      ::tensorflow::Input s1);

  Operation operation;
  ::tensorflow::Output r0;
  ::tensorflow::Output r1;
};

/// Checks a tensor for NaN, -Inf and +Inf values.
///
/// When run, reports an `InvalidArgument` error if `tensor` has any values
/// that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
/// tensor. Unlike CheckNumerics (V1), CheckNumericsV2 distinguishes -Inf and +Inf
/// in the errors it throws.
///
/// Args:
/// * scope: A Scope object
/// * message: Prefix of the error message.
///
/// Returns:
/// * `Output`: The output tensor.
class CheckNumericsV2 {
 public:
  CheckNumericsV2(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                StringPiece message);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Applies lower_bound(sorted_search_values, values) along each row.
///
/// Each set of rows with the same index in (sorted_inputs, values) is treated
/// independently.  The resulting row is the equivalent of calling
/// `np.searchsorted(sorted_inputs, values, side='left')`.
///
/// The result is not a global index to the entire
/// `Tensor`, but rather just the index in the last dimension.
///
/// A 2-D example:
///   sorted_sequence = [[0, 3, 9, 9, 10],
///                      [1, 2, 3, 4, 5]]
///   values = [[2, 4, 9],
///             [0, 2, 6]]
///
///   result = LowerBound(sorted_sequence, values)
///
///   result == [[1, 2, 2],
///              [0, 1, 5]]
///
/// Args:
/// * scope: A Scope object
/// * sorted_inputs: 2-D Tensor where each row is ordered.
/// * values: 2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
/// the values that will be searched for in `sorted_search_values`.
///
/// Returns:
/// * `Output`: A `Tensor` with the same shape as `values`.  It contains the first scalar index
/// into the last dimension where values can be inserted without changing the
/// ordered property.
class LowerBound {
 public:
  /// Optional attribute setters for LowerBound
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_INT32;
  };
  LowerBound(const ::tensorflow::Scope& scope, ::tensorflow::Input sorted_inputs,
           ::tensorflow::Input values);
  LowerBound(const ::tensorflow::Scope& scope, ::tensorflow::Input sorted_inputs,
           ::tensorflow::Input values, const LowerBound::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.
///
/// This operation folds the padded areas of `input` by `MirrorPad` according to the
/// `paddings` you specify. `paddings` must be the same as `paddings` argument
/// given to the corresponding `MirrorPad` op.
///
/// The folded size of each dimension D of the output is:
///
/// `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
/// # 'paddings' is [[0, 1]], [0, 1]].
/// # 'mode' is SYMMETRIC.
/// # rank of 't' is 2.
/// pad(t, paddings) ==> [[ 1,  5]
///                       [11, 28]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: The input tensor to be folded.
/// * paddings: A two-column matrix specifying the padding sizes. The number of
/// rows must be the same as the rank of `input`.
/// * mode: The mode used in the `MirrorPad` op.
///
/// Returns:
/// * `Output`: The folded tensor.
class MirrorPadGrad {
 public:
  MirrorPadGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input paddings, StringPiece mode);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Return the same ref tensor as the input ref tensor.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class RefIdentity {
 public:
  RefIdentity(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Applies upper_bound(sorted_search_values, values) along each row.
///
/// Each set of rows with the same index in (sorted_inputs, values) is treated
/// independently.  The resulting row is the equivalent of calling
/// `np.searchsorted(sorted_inputs, values, side='right')`.
///
/// The result is not a global index to the entire
/// `Tensor`, but rather just the index in the last dimension.
///
/// A 2-D example:
///   sorted_sequence = [[0, 3, 9, 9, 10],
///                      [1, 2, 3, 4, 5]]
///   values = [[2, 4, 9],
///             [0, 2, 6]]
///
///   result = UpperBound(sorted_sequence, values)
///
///   result == [[1, 2, 4],
///              [0, 2, 5]]
///
/// Args:
/// * scope: A Scope object
/// * sorted_inputs: 2-D Tensor where each row is ordered.
/// * values: 2-D Tensor with the same numbers of rows as `sorted_search_values`. Contains
/// the values that will be searched for in `sorted_search_values`.
///
/// Returns:
/// * `Output`: A `Tensor` with the same shape as `values`.  It contains the last scalar index
/// into the last dimension where values can be inserted without changing the
/// ordered property.
class UpperBound {
 public:
  /// Optional attribute setters for UpperBound
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_INT32;
  };
  UpperBound(const ::tensorflow::Scope& scope, ::tensorflow::Input sorted_inputs,
           ::tensorflow::Input values);
  UpperBound(const ::tensorflow::Scope& scope, ::tensorflow::Input sorted_inputs,
           ::tensorflow::Input values, const UpperBound::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_ARRAY_OPS_INTERNAL_H_
