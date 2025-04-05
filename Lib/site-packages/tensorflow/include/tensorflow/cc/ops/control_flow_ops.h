// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_H_
#define TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup control_flow_ops Control Flow Ops
/// @{

/// Raise a exception to abort the process when called.
///
/// If exit_without_error is true, the process will exit normally,
/// otherwise it will exit with a SIGABORT signal.
///
/// Returns nothing but an exception.
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * error_msg: A string which is the message associated with the exception.
///
/// Returns:
/// * the created `Operation`
class Abort {
 public:
  /// Optional attribute setters for Abort
  struct Attrs {
    /// A string which is the message associated with the exception.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs ErrorMsg(StringPiece x) {
      Attrs ret = *this;
      ret.error_msg_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ExitWithoutError(bool x) {
      Attrs ret = *this;
      ret.exit_without_error_ = x;
      return ret;
    }

    StringPiece error_msg_ = "";
    bool exit_without_error_ = false;
  };
  Abort(const ::tensorflow::Scope& scope);
  Abort(const ::tensorflow::Scope& scope, const Abort::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs ErrorMsg(StringPiece x) {
    return Attrs().ErrorMsg(x);
  }
  static Attrs ExitWithoutError(bool x) {
    return Attrs().ExitWithoutError(x);
  }

  Operation operation;
};

/// Does nothing. Serves as a control trigger for scheduling.
///
/// Only useful as a placeholder for control edges.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class ControlTrigger {
 public:
  ControlTrigger(const ::tensorflow::Scope& scope);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Forwards the input to the output.
///
/// This operator represents the loop termination condition used by the
/// "pivot" switches of a loop.
///
/// Args:
/// * scope: A Scope object
/// * input: A boolean scalar, representing the branch predicate of the Switch op.
///
/// Returns:
/// * `Output`: The same tensor as `input`.
class LoopCond {
 public:
  LoopCond(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
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
/// `Merge` forwards the first tensor to become available to `output`, and sets
/// `value_index` to its index in `inputs`.
///
/// Args:
/// * scope: A Scope object
/// * inputs: The input tensors, exactly one of which will become available.
///
/// Returns:
/// * `Output` output: Will be set to the available input tensor.
/// * `Output` value_index: The index of the chosen input tensor in `inputs`.
class Merge {
 public:
  Merge(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output value_index;
};

/// Makes its input available to the next iteration.
///
/// Args:
/// * scope: A Scope object
/// * data: The tensor to be made available to the next iteration.
///
/// Returns:
/// * `Output`: The same tensor as `data`.
class NextIteration {
 public:
  NextIteration(const ::tensorflow::Scope& scope, ::tensorflow::Input data);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Makes its input available to the next iteration.
///
/// Args:
/// * scope: A Scope object
/// * data: The tensor to be made available to the next iteration.
///
/// Returns:
/// * `Output`: The same tensor as `data`.
class RefNextIteration {
 public:
  RefNextIteration(const ::tensorflow::Scope& scope, ::tensorflow::Input data);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Forwards the `index`th element of `inputs` to `output`.
///
/// Args:
/// * scope: A Scope object
/// * index: A scalar that determines the input that gets selected.
/// * inputs: A list of ref tensors, one of which will be forwarded to `output`.
///
/// Returns:
/// * `Output`: The forwarded tensor.
class RefSelect {
 public:
  RefSelect(const ::tensorflow::Scope& scope, ::tensorflow::Input index,
          ::tensorflow::InputList inputs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Forwards the ref tensor `data` to the output port determined by `pred`.
///
/// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
/// the data goes to `output_false`.
///
/// See also `Switch` and `Merge`.
///
/// Args:
/// * scope: A Scope object
/// * data: The ref tensor to be forwarded to the appropriate output.
/// * pred: A scalar that specifies which output port will receive data.
///
/// Returns:
/// * `Output` output_false: If `pred` is false, data will be forwarded to this output.
/// * `Output` output_true: If `pred` is true, data will be forwarded to this output.
class RefSwitch {
 public:
  RefSwitch(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
          ::tensorflow::Input pred);

  Operation operation;
  ::tensorflow::Output output_false;
  ::tensorflow::Output output_true;
};

/// Forwards `data` to the output port determined by `pred`.
///
/// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
/// the data goes to `output_false`.
///
/// See also `RefSwitch` and `Merge`.
///
/// Args:
/// * scope: A Scope object
/// * data: The tensor to be forwarded to the appropriate output.
/// * pred: A scalar that specifies which output port will receive data.
///
/// Returns:
/// * `Output` output_false: If `pred` is false, data will be forwarded to this output.
/// * `Output` output_true: If `pred` is true, data will be forwarded to this output.
class Switch {
 public:
  Switch(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
       ::tensorflow::Input pred);

  Operation operation;
  ::tensorflow::Output output_false;
  ::tensorflow::Output output_true;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_H_
