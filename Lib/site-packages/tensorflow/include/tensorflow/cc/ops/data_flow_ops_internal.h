// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_DATA_FLOW_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_DATA_FLOW_OPS_INTERNAL_H_

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

/// @defgroup data_flow_ops_internal Data Flow Ops Internal
/// @{

/// Applies a gradient to a given accumulator.
///
/// Does not add if local_step is lesser than the accumulator's global_step.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a accumulator.
/// * local_step: The local_step value at which the gradient was computed.
/// * gradient: A tensor of the gradient to be accumulated.
///
/// Returns:
/// * the created `Operation`
class ResourceAccumulatorApplyGradient {
 public:
  ResourceAccumulatorApplyGradient(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input handle,
                                 ::tensorflow::Input local_step,
                                 ::tensorflow::Input gradient);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Returns the number of gradients aggregated in the given accumulators.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to an accumulator.
///
/// Returns:
/// * `Output`: The number of gradients aggregated in the given accumulator.
class ResourceAccumulatorNumAccumulated {
 public:
  ResourceAccumulatorNumAccumulated(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input handle);
  operator ::tensorflow::Output() const { return num_accumulated; }
  operator ::tensorflow::Input() const { return num_accumulated; }
  ::tensorflow::Node* node() const { return num_accumulated.node(); }

  Operation operation;
  ::tensorflow::Output num_accumulated;
};

/// Updates the accumulator with a new value for global_step.
///
/// Logs warning if the accumulator's value is already higher than
/// new_global_step.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to an accumulator.
/// * new_global_step: The new global_step value to set.
///
/// Returns:
/// * the created `Operation`
class ResourceAccumulatorSetGlobalStep {
 public:
  ResourceAccumulatorSetGlobalStep(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input handle,
                                 ::tensorflow::Input new_global_step);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Extracts the average gradient in the given ConditionalAccumulator.
///
/// The op blocks until sufficient (i.e., more than num_required)
/// gradients have been accumulated.  If the accumulator has already
/// aggregated more than num_required gradients, it returns the average of
/// the accumulated gradients.  Also automatically increments the recorded
/// global_step in the accumulator by 1, and resets the aggregate to 0.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to an accumulator.
/// * num_required: Number of gradients required before we return an aggregate.
/// * dtype: The data type of accumulated gradients. Needs to correspond to the type
/// of the accumulator.
///
/// Returns:
/// * `Output`: The average of the accumulated gradients.
class ResourceAccumulatorTakeGradient {
 public:
  ResourceAccumulatorTakeGradient(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input handle, ::tensorflow::Input
                                num_required, DataType dtype);
  operator ::tensorflow::Output() const { return average; }
  operator ::tensorflow::Input() const { return average; }
  ::tensorflow::Node* node() const { return average.node(); }

  Operation operation;
  ::tensorflow::Output average;
};

/// A conditional accumulator for aggregating gradients.
///
/// The accumulator accepts gradients marked with local_step greater or
/// equal to the most recent global_step known to the accumulator. The
/// average can be extracted from the accumulator, provided sufficient
/// gradients have been accumulated. Extracting the average automatically
/// resets the aggregate to 0, and increments the global_step recorded by
/// the accumulator.
/// This is a resource version of ConditionalAccumulator that will work in TF2.0
/// with tf.cond version 2.
///
/// Args:
/// * scope: A Scope object
/// * dtype: The type of the value being accumulated.
/// * shape: The shape of the values, can be [], in which case shape is unknown.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this accumulator is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this accumulator will be shared under the
/// given name across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the accumulator.
class ResourceConditionalAccumulator {
 public:
  /// Optional attribute setters for ResourceConditionalAccumulator
  struct Attrs {
    /// If non-empty, this accumulator is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this accumulator will be shared under the
    /// given name across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// Defaults to "MEAN"
    TF_MUST_USE_RESULT Attrs ReductionType(StringPiece x) {
      Attrs ret = *this;
      ret.reduction_type_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    StringPiece reduction_type_ = "MEAN";
  };
  ResourceConditionalAccumulator(const ::tensorflow::Scope& scope, DataType
                               dtype, PartialTensorShape shape);
  ResourceConditionalAccumulator(const ::tensorflow::Scope& scope, DataType
                               dtype, PartialTensorShape shape, const
                               ResourceConditionalAccumulator::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs ReductionType(StringPiece x) {
    return Attrs().ReductionType(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_DATA_FLOW_OPS_INTERNAL_H_
