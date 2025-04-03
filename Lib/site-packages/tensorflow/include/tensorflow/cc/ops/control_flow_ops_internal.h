// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_INTERNAL_H_

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

/// @defgroup control_flow_ops_internal Control Flow Ops Internal
/// @{

/// Creates or finds a child frame, and makes `data` available to the child frame.
///
/// This op is used together with `Exit` to create loops in the graph.
/// The unique `frame_name` is used by the `Executor` to identify frames. If
/// `is_constant` is true, `output` is a constant in the child frame; otherwise
/// it may be changed in the child frame. At most `parallel_iterations` iterations
/// are run in parallel in the child frame.
///
/// Args:
/// * scope: A Scope object
/// * data: The tensor to be made available to the child frame.
/// * frame_name: The name of the child frame.
///
/// Optional attributes (see `Attrs`):
/// * is_constant: If true, the output is constant within the child frame.
/// * parallel_iterations: The number of iterations allowed to run in parallel.
///
/// Returns:
/// * `Output`: The same tensor as `data`.
class Enter {
 public:
  /// Optional attribute setters for Enter
  struct Attrs {
    /// If true, the output is constant within the child frame.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs IsConstant(bool x) {
      Attrs ret = *this;
      ret.is_constant_ = x;
      return ret;
    }

    /// The number of iterations allowed to run in parallel.
    ///
    /// Defaults to 10
    TF_MUST_USE_RESULT Attrs ParallelIterations(int64 x) {
      Attrs ret = *this;
      ret.parallel_iterations_ = x;
      return ret;
    }

    bool is_constant_ = false;
    int64 parallel_iterations_ = 10;
  };
  Enter(const ::tensorflow::Scope& scope, ::tensorflow::Input data, StringPiece
      frame_name);
  Enter(const ::tensorflow::Scope& scope, ::tensorflow::Input data, StringPiece
      frame_name, const Enter::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs IsConstant(bool x) {
    return Attrs().IsConstant(x);
  }
  static Attrs ParallelIterations(int64 x) {
    return Attrs().ParallelIterations(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Exits the current frame to its parent frame.
///
/// Exit makes its input `data` available to the parent frame.
///
/// Args:
/// * scope: A Scope object
/// * data: The tensor to be made available to the parent frame.
///
/// Returns:
/// * `Output`: The same tensor as `data`.
class Exit {
 public:
  Exit(const ::tensorflow::Scope& scope, ::tensorflow::Input data);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Creates or finds a child frame, and makes `data` available to the child frame.
///
/// The unique `frame_name` is used by the `Executor` to identify frames. If
/// `is_constant` is true, `output` is a constant in the child frame; otherwise
/// it may be changed in the child frame. At most `parallel_iterations` iterations
/// are run in parallel in the child frame.
///
/// Args:
/// * scope: A Scope object
/// * data: The tensor to be made available to the child frame.
/// * frame_name: The name of the child frame.
///
/// Optional attributes (see `Attrs`):
/// * is_constant: If true, the output is constant within the child frame.
/// * parallel_iterations: The number of iterations allowed to run in parallel.
///
/// Returns:
/// * `Output`: The same tensor as `data`.
class RefEnter {
 public:
  /// Optional attribute setters for RefEnter
  struct Attrs {
    /// If true, the output is constant within the child frame.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs IsConstant(bool x) {
      Attrs ret = *this;
      ret.is_constant_ = x;
      return ret;
    }

    /// The number of iterations allowed to run in parallel.
    ///
    /// Defaults to 10
    TF_MUST_USE_RESULT Attrs ParallelIterations(int64 x) {
      Attrs ret = *this;
      ret.parallel_iterations_ = x;
      return ret;
    }

    bool is_constant_ = false;
    int64 parallel_iterations_ = 10;
  };
  RefEnter(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
         StringPiece frame_name);
  RefEnter(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
         StringPiece frame_name, const RefEnter::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs IsConstant(bool x) {
    return Attrs().IsConstant(x);
  }
  static Attrs ParallelIterations(int64 x) {
    return Attrs().ParallelIterations(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Exits the current frame to its parent frame.
///
/// Exit makes its input `data` available to the parent frame.
///
/// Args:
/// * scope: A Scope object
/// * data: The tensor to be made available to the parent frame.
///
/// Returns:
/// * `Output`: The same tensor as `data`.
class RefExit {
 public:
  RefExit(const ::tensorflow::Scope& scope, ::tensorflow::Input data);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Forwards the value of an available tensor from `inputs` to `output`.
///
/// `Merge` waits for at least one of the tensors in `inputs` to become available.
/// It is usually combined with `Switch` to implement branching.
///
/// `Merge` forwards the first tensor for become available to `output`, and sets
/// `value_index` to its index in `inputs`.
///
/// Args:
/// * scope: A Scope object
/// * inputs: The input tensors, exactly one of which will become available.
///
/// Returns:
/// * `Output` output: Will be set to the available input tensor.
/// * `Output` value_index: The index of the chosen input tensor in `inputs`.
class RefMerge {
 public:
  RefMerge(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output value_index;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_INTERNAL_H_
