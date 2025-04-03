// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_EXPERIMENTAL_DATASET_OPS_H_
#define TENSORFLOW_CC_OPS_EXPERIMENTAL_DATASET_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup experimental_dataset_ops Experimental Dataset Ops
/// @{

/// Checks whether a tensor is located in host memory pinned for GPU.
///
/// When run:
/// - Reports an `InvalidArgument` error if `tensor` is not in pinned memory.
/// - Reports a `FailedPrecondition` error if not built with CUDA.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class CheckPinned {
 public:
  CheckPinned(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_EXPERIMENTAL_DATASET_OPS_H_
