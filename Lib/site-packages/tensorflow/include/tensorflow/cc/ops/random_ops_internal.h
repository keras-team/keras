// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_RANDOM_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_RANDOM_OPS_INTERNAL_H_

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

/// @defgroup random_ops_internal Random Ops Internal
/// @{

/// Computes the derivative of a Gamma random sample w.r.t. `alpha`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class RandomGammaGrad {
 public:
  RandomGammaGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input alpha,
                ::tensorflow::Input sample);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_RANDOM_OPS_INTERNAL_H_
