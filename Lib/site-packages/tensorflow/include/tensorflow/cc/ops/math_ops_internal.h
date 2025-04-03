// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_MATH_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_MATH_OPS_INTERNAL_H_

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

/// @defgroup math_ops_internal Math Ops Internal
/// @{

/// Compute the cumulative product of the tensor `x` along `axis`.
///
/// By default, this op performs an inclusive cumulative log-sum-exp,
/// which means that the first
/// element of the input is identical to the first element of the output:
/// ```python
/// tf.math.cumulative_logsumexp([a, b, c])  # => [a, log(exp(a) + exp(b)), log(exp(a) + exp(b) + exp(c))]
/// ```
///
/// By setting the `exclusive` kwarg to `True`, an exclusive cumulative log-sum-exp is
/// performed instead:
/// ```python
/// tf.cumulative_logsumexp([a, b, c], exclusive=True)  # => [-inf, a, log(exp(a) * exp(b))]
/// ```
/// Note that the neutral element of the log-sum-exp operation is `-inf`,
/// however, for performance reasons, the minimal value representable by the
/// floating point type is used instead.
///
/// By setting the `reverse` kwarg to `True`, the cumulative log-sum-exp is performed in the
/// opposite direction.
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
/// * axis: A `Tensor` of type `int32` (default: 0). Must be in the range
/// `[-rank(x), rank(x))`.
///
/// Optional attributes (see `Attrs`):
/// * exclusive: If `True`, perform exclusive cumulative log-sum-exp.
/// * reverse: A `bool` (default: False).
///
/// Returns:
/// * `Output`: The out tensor.
class CumulativeLogsumexp {
 public:
  /// Optional attribute setters for CumulativeLogsumexp
  struct Attrs {
    /// If `True`, perform exclusive cumulative log-sum-exp.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Exclusive(bool x) {
      Attrs ret = *this;
      ret.exclusive_ = x;
      return ret;
    }

    /// A `bool` (default: False).
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Reverse(bool x) {
      Attrs ret = *this;
      ret.reverse_ = x;
      return ret;
    }

    bool exclusive_ = false;
    bool reverse_ = false;
  };
  CumulativeLogsumexp(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                    ::tensorflow::Input axis);
  CumulativeLogsumexp(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                    ::tensorflow::Input axis, const CumulativeLogsumexp::Attrs&
                    attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs Exclusive(bool x) {
    return Attrs().Exclusive(x);
  }
  static Attrs Reverse(bool x) {
    return Attrs().Reverse(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Computes the gradient of `igamma(a, x)` wrt `a`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class IgammaGradA {
 public:
  IgammaGradA(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
            ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the gradient for the inverse of `x` wrt its input.
///
/// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
/// is the corresponding input gradient.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class InvGrad {
 public:
  InvGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input y,
        ::tensorflow::Input dy);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Generates values in an interval.
///
/// A sequence of `num` evenly-spaced values are generated beginning at `start`.
/// If `num > 1`, the values in the sequence increase by
/// `(stop - start) / (num - 1)`, so that the last one is exactly `stop`.
///
/// For example:
///
/// ```
/// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * start: 0-D tensor. First entry in the range.
/// * stop: 0-D tensor. Last entry in the range.
/// * num: 0-D tensor. Number of values to generate.
///
/// Returns:
/// * `Output`: 1-D. The generated values.
class LinSpace {
 public:
  LinSpace(const ::tensorflow::Scope& scope, ::tensorflow::Input start,
         ::tensorflow::Input stop, ::tensorflow::Input num);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the gradient for the inverse of `x` wrt its input.
///
/// Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
/// is the corresponding input gradient.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class ReciprocalGrad {
 public:
  ReciprocalGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input y,
               ::tensorflow::Input dy);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes requantization range per channel.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input tensor.
/// * input_min: The minimum value of the input tensor
/// * input_max: The maximum value of the input tensor.
/// * clip_value_max: The maximum value of the output that needs to be clipped.
/// Example: set this to 6 for Relu6.
///
/// Returns:
/// * `Output` output_min: The minimum value of the final output tensor
/// * `Output` output_max: The maximum value of the final output tensor.
class RequantizationRangePerChannel {
 public:
  RequantizationRangePerChannel(const ::tensorflow::Scope& scope,
                              ::tensorflow::Input input, ::tensorflow::Input
                              input_min, ::tensorflow::Input input_max, float
                              clip_value_max);

  Operation operation;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Requantizes input with min and max values known per channel.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input tensor.
/// * input_min: The minimum value of the input tensor
/// * input_max: The maximum value of the input tensor.
/// * requested_output_min: The minimum value of the output tensor requested.
/// * requested_output_max: The maximum value of the output tensor requested.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The quantized type of output tensor that needs to be converted.
///
/// Returns:
/// * `Output` output: Output tensor.
/// * `Output` output_min: The minimum value of the final output tensor
/// * `Output` output_max: The maximum value of the final output tensor.
class RequantizePerChannel {
 public:
  /// Optional attribute setters for RequantizePerChannel
  struct Attrs {
    /// The quantized type of output tensor that needs to be converted.
    ///
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_QUINT8;
  };
  RequantizePerChannel(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input, ::tensorflow::Input input_min, ::tensorflow::Input
                     input_max, ::tensorflow::Input requested_output_min,
                     ::tensorflow::Input requested_output_max);
  RequantizePerChannel(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input, ::tensorflow::Input input_min, ::tensorflow::Input
                     input_max, ::tensorflow::Input requested_output_min,
                     ::tensorflow::Input requested_output_max, const
                     RequantizePerChannel::Attrs& attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Computes the gradient for the rsqrt of `x` wrt its input.
///
/// Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
/// is the corresponding input gradient.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class RsqrtGrad {
 public:
  RsqrtGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input y,
          ::tensorflow::Input dy);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the gradient of the sigmoid of `x` wrt its input.
///
/// Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
/// `dy` is the corresponding input gradient.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class SigmoidGrad {
 public:
  SigmoidGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input y,
            ::tensorflow::Input dy);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Generates points from the Sobol sequence.
///
/// Creates a Sobol sequence with `num_results` samples. Each sample has dimension
/// `dim`. Skips the first `skip` samples.
///
/// Args:
/// * scope: A Scope object
/// * dim: Positive scalar `Tensor` representing each sample's dimension.
/// * num_results: Positive scalar `Tensor` of dtype int32. The number of Sobol points to return
/// in the output.
/// * skip: Positive scalar `Tensor` of dtype int32. The number of initial points of the
/// Sobol sequence to skip.
///
/// Optional attributes (see `Attrs`):
/// * dtype: The type of the sample. One of: `float32` or `float64`.
///
/// Returns:
/// * `Output`: `Tensor` of samples from Sobol sequence with `shape` [num_results, dim].
class SobolSample {
 public:
  /// Optional attribute setters for SobolSample
  struct Attrs {
    /// The type of the sample. One of: `float32` or `float64`.
    ///
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs Dtype(DataType x) {
      Attrs ret = *this;
      ret.dtype_ = x;
      return ret;
    }

    DataType dtype_ = DT_FLOAT;
  };
  SobolSample(const ::tensorflow::Scope& scope, ::tensorflow::Input dim,
            ::tensorflow::Input num_results, ::tensorflow::Input skip);
  SobolSample(const ::tensorflow::Scope& scope, ::tensorflow::Input dim,
            ::tensorflow::Input num_results, ::tensorflow::Input skip, const
            SobolSample::Attrs& attrs);
  operator ::tensorflow::Output() const { return samples; }
  operator ::tensorflow::Input() const { return samples; }
  ::tensorflow::Node* node() const { return samples.node(); }

  static Attrs Dtype(DataType x) {
    return Attrs().Dtype(x);
  }

  Operation operation;
  ::tensorflow::Output samples;
};

/// Computes the gradient for the sqrt of `x` wrt its input.
///
/// Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
/// is the corresponding input gradient.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class SqrtGrad {
 public:
  SqrtGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input y,
         ::tensorflow::Input dy);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the gradient for the tanh of `x` wrt its input.
///
/// Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
/// is the corresponding input gradient.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class TanhGrad {
 public:
  TanhGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input y,
         ::tensorflow::Input dy);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_MATH_OPS_INTERNAL_H_
