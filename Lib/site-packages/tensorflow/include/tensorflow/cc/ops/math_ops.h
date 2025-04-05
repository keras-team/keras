// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_MATH_OPS_H_
#define TENSORFLOW_CC_OPS_MATH_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup math_ops Math Ops
/// @{

/// Computes the absolute value of a tensor.
///
/// Given a tensor `x`, this operation returns a tensor containing the absolute
/// value of each element in `x`. For example, if x is an input element and y is
/// an output element, this operation computes \\(y = |x|\\).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Abs {
 public:
  Abs(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns the element-wise sum of a list of tensors.
///
/// `tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
/// wait for all of its inputs to be ready before beginning to sum. This can
/// save memory if inputs are ready at different times, since minimum temporary
/// storage is proportional to the output size rather than the inputs size.
///
/// Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.
///
/// Returns a `Tensor` of same shape and type as the elements of `inputs`.
///
/// Args:
/// * scope: A Scope object
/// * inputs: A list of `Tensor` objects, each with same shape and type.
/// * shape: Shape of elements of `inputs`.
///
/// Returns:
/// * `Output`: The sum tensor.
class AccumulateNV2 {
 public:
  AccumulateNV2(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs,
              PartialTensorShape shape);
  operator ::tensorflow::Output() const { return sum; }
  operator ::tensorflow::Input() const { return sum; }
  ::tensorflow::Node* node() const { return sum.node(); }

  Operation operation;
  ::tensorflow::Output sum;
};

/// Computes acos of x element-wise.
///
///
///   Provided an input tensor, the `tf.math.acos` operation returns the inverse cosine of each element of the tensor. If `y = tf.math.cos(x)` then, `x = tf.math.acos(y)`.
///
///   Input range is `[-1, 1]` and the output has a range of `[0, pi]`.
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Acos {
 public:
  Acos(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes inverse hyperbolic cosine of x element-wise.
///
/// Given an input tensor, the function computes inverse hyperbolic cosine of every element.
/// Input range is `[1, inf]`. It returns `nan` if the input lies outside the range.
///
/// ```python
/// x = tf.constant([-2, -0.5, 1, 1.2, 200, 10000, float("inf")])
/// tf.math.acosh(x) ==> [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Acosh {
 public:
  Acosh(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns x + y element-wise.
///
/// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Given two input tensors, the `tf.add` operation computes the sum for every element in the tensor.
///
/// Both input and output have a range `(-inf, inf)`.
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Add {
 public:
  Add(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
    ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Add all input tensors element wise.
///
///   Inputs must be of same size and shape.
///
///   ```python
///   x = [9, 7, 10]
///   tf.math.add_n(x) ==> 26
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The sum tensor.
class AddN {
 public:
  AddN(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs);
  operator ::tensorflow::Output() const { return sum; }
  operator ::tensorflow::Input() const { return sum; }
  ::tensorflow::Node* node() const { return sum.node(); }

  Operation operation;
  ::tensorflow::Output sum;
};

/// Returns x + y element-wise.
///
/// *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class AddV2 {
 public:
  AddV2(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
      ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the "logical and" of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
///
/// Aliases:
/// * ReduceAll
class All {
 public:
  /// Optional attribute setters for All
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
  All(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis);
  All(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis, const All::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};
typedef All ReduceAll;

/// Returns the argument of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the argument of each element in `input`. All elements in
/// `input` must be complex numbers of the form \\(a + bj\\), where *a*
/// is the real part and *b* is the imaginary part.
///
/// The argument returned by this operation is of the form \\(atan2(b, a)\\).
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.math.angle(input) ==> [2.0132, 1.056]
/// ```
///
/// @compatibility(numpy)
/// Equivalent to np.angle.
/// @end_compatibility
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Angle {
 public:
  /// Optional attribute setters for Angle
  struct Attrs {
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs Tout(DataType x) {
      Attrs ret = *this;
      ret.Tout_ = x;
      return ret;
    }

    DataType Tout_ = DT_FLOAT;
  };
  Angle(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Angle(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
      Angle::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Tout(DataType x) {
    return Attrs().Tout(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the "logical or" of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
///
/// Aliases:
/// * ReduceAny
class Any {
 public:
  /// Optional attribute setters for Any
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
  Any(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis);
  Any(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis, const Any::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};
typedef Any ReduceAny;

/// Returns the truth value of abs(x-y) < tolerance element-wise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class ApproximateEqual {
 public:
  /// Optional attribute setters for ApproximateEqual
  struct Attrs {
    /// Defaults to 1e-05
    TF_MUST_USE_RESULT Attrs Tolerance(float x) {
      Attrs ret = *this;
      ret.tolerance_ = x;
      return ret;
    }

    float tolerance_ = 1e-05f;
  };
  ApproximateEqual(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                 ::tensorflow::Input y);
  ApproximateEqual(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                 ::tensorflow::Input y, const ApproximateEqual::Attrs& attrs);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  static Attrs Tolerance(float x) {
    return Attrs().Tolerance(x);
  }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns the index with the largest value across dimensions of a tensor.
///
/// Note that in case of ties the identity of the return value is not guaranteed.
///
/// Usage:
///   ```python
///   import tensorflow as tf
///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
///   b = tf.math.argmax(input = a)
///   c = tf.keras.backend.eval(b)
///   # c = 4
///   # here a[4] = 166.32 which is the largest element of a across axis 0
///   ```
///
/// Args:
/// * scope: A Scope object
/// * dimension: int16, int32 or int64, must be in the range `[-rank(input), rank(input))`.
/// Describes which dimension of the input Tensor to reduce across. For vectors,
/// use dimension = 0.
///
/// Returns:
/// * `Output`: The output tensor.
class ArgMax {
 public:
  /// Optional attribute setters for ArgMax
  struct Attrs {
    /// Defaults to DT_INT64
    TF_MUST_USE_RESULT Attrs OutputType(DataType x) {
      Attrs ret = *this;
      ret.output_type_ = x;
      return ret;
    }

    DataType output_type_ = DT_INT64;
  };
  ArgMax(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
       ::tensorflow::Input dimension);
  ArgMax(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
       ::tensorflow::Input dimension, const ArgMax::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutputType(DataType x) {
    return Attrs().OutputType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the index with the smallest value across dimensions of a tensor.
///
/// Note that in case of ties the identity of the return value is not guaranteed.
///
/// Usage:
///   ```python
///   import tensorflow as tf
///   a = [1, 10, 26.9, 2.8, 166.32, 62.3]
///   b = tf.math.argmin(input = a)
///   c = tf.keras.backend.eval(b)
///   # c = 0
///   # here a[0] = 1 which is the smallest element of a across axis 0
///   ```
///
/// Args:
/// * scope: A Scope object
/// * dimension: int32 or int64, must be in the range `[-rank(input), rank(input))`.
/// Describes which dimension of the input Tensor to reduce across. For vectors,
/// use dimension = 0.
///
/// Returns:
/// * `Output`: The output tensor.
class ArgMin {
 public:
  /// Optional attribute setters for ArgMin
  struct Attrs {
    /// Defaults to DT_INT64
    TF_MUST_USE_RESULT Attrs OutputType(DataType x) {
      Attrs ret = *this;
      ret.output_type_ = x;
      return ret;
    }

    DataType output_type_ = DT_INT64;
  };
  ArgMin(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
       ::tensorflow::Input dimension);
  ArgMin(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
       ::tensorflow::Input dimension, const ArgMin::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutputType(DataType x) {
    return Attrs().OutputType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the trignometric inverse sine of x element-wise.
///
/// The `tf.math.asin` operation returns the inverse of `tf.math.sin`, such that
/// if `y = tf.math.sin(x)` then, `x = tf.math.asin(y)`.
///
/// **Note**: The output of `tf.math.asin` will lie within the invertible range
/// of sine, i.e [-pi/2, pi/2].
///
/// For example:
///
/// ```python
/// # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
/// x = tf.constant([1.047, 0.785])
/// y = tf.math.sin(x) # [0.8659266, 0.7068252]
///
/// tf.math.asin(y) # [1.047, 0.785] = x
/// ```
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Asin {
 public:
  Asin(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes inverse hyperbolic sine of x element-wise.
///
///   Given an input tensor, this function computes inverse hyperbolic sine
///   for every element in the tensor. Both input and output has a range of
///   `[-inf, inf]`.
///
///   ```python
///   x = tf.constant([-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")])
///   tf.math.asinh(x) ==> [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Asinh {
 public:
  Asinh(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes the trignometric inverse tangent of x element-wise.
///
/// The `tf.math.atan` operation returns the inverse of `tf.math.tan`, such that
/// if `y = tf.math.tan(x)` then, `x = tf.math.atan(y)`.
///
/// **Note**: The output of `tf.math.atan` will lie within the invertible range
/// of tan, i.e (-pi/2, pi/2).
///
/// For example:
///
/// ```python
/// # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
/// x = tf.constant([1.047, 0.785])
/// y = tf.math.tan(x) # [1.731261, 0.99920404]
///
/// tf.math.atan(y) # [1.047, 0.785] = x
/// ```
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Atan {
 public:
  Atan(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
///
/// This is the angle \\( \theta \in [-\pi, \pi] \\) such that
/// \\[ x = r \cos(\theta) \\]
/// and
/// \\[ y = r \sin(\theta) \\]
/// where \\(r = \sqrt{x^2 + y^2} \\).
///
/// For example:
///
/// >>> x = [1., 1.]
/// >>> y = [1., -1.]
/// >>> print((tf.math.atan2(y,x) * (180 / np.pi)).numpy())
/// [ 45. -45.]
///
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Atan2 {
 public:
  Atan2(const ::tensorflow::Scope& scope, ::tensorflow::Input y,
      ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes inverse hyperbolic tangent of x element-wise.
///
///   Given an input tensor, this function computes inverse hyperbolic tangent
///   for every element in the tensor. Input range is `[-1,1]` and output range is
///   `[-inf, inf]`. If input is `-1`, output will be `-inf` and if the
///   input is `1`, output will be `inf`. Values outside the range will have
///   `nan` as output.
///
///   ```python
///   x = tf.constant([-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")])
///   tf.math.atanh(x) ==> [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Atanh {
 public:
  Atanh(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Multiplies slices of two tensors in batches.
///
/// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
/// viewed as an element of a batch), and arranges the individual results
/// in a single output tensor of the same batch size. Each of the
/// individual slices can optionally be adjointed (to adjoint a matrix
/// means to transpose and conjugate it) before multiplication by setting
/// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
///
/// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
/// and `[..., r_y, c_y]`.
///
/// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
///
///     r_o = c_x if adj_x else r_x
///     c_o = r_y if adj_y else c_y
///
/// It is computed as:
///
///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
///
/// Args:
/// * scope: A Scope object
/// * x: 2-D or higher with shape `[..., r_x, c_x]`.
/// * y: 2-D or higher with shape `[..., r_y, c_y]`.
///
/// Optional attributes (see `Attrs`):
/// * adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
/// * adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
///
/// Returns:
/// * `Output`: 3-D or higher with shape `[..., r_o, c_o]`
class BatchMatMul {
 public:
  /// Optional attribute setters for BatchMatMul
  struct Attrs {
    /// If `True`, adjoint the slices of `x`. Defaults to `False`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjX(bool x) {
      Attrs ret = *this;
      ret.adj_x_ = x;
      return ret;
    }

    /// If `True`, adjoint the slices of `y`. Defaults to `False`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjY(bool x) {
      Attrs ret = *this;
      ret.adj_y_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradX(bool x) {
      Attrs ret = *this;
      ret.grad_x_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradY(bool x) {
      Attrs ret = *this;
      ret.grad_y_ = x;
      return ret;
    }

    bool adj_x_ = false;
    bool adj_y_ = false;
    bool grad_x_ = false;
    bool grad_y_ = false;
  };
  BatchMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
            ::tensorflow::Input y);
  BatchMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
            ::tensorflow::Input y, const BatchMatMul::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs AdjX(bool x) {
    return Attrs().AdjX(x);
  }
  static Attrs AdjY(bool x) {
    return Attrs().AdjY(x);
  }
  static Attrs GradX(bool x) {
    return Attrs().GradX(x);
  }
  static Attrs GradY(bool x) {
    return Attrs().GradY(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Multiplies slices of two tensors in batches.
///
/// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
/// viewed as an element of a batch), and arranges the individual results
/// in a single output tensor of the same batch size. Each of the
/// individual slices can optionally be adjointed (to adjoint a matrix
/// means to transpose and conjugate it) before multiplication by setting
/// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
///
/// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
/// and `[..., r_y, c_y]`.
///
/// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
///
///     r_o = c_x if adj_x else r_x
///     c_o = r_y if adj_y else c_y
///
/// It is computed as:
///
///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
///
/// *NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
/// about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
///
/// Args:
/// * scope: A Scope object
/// * x: 2-D or higher with shape `[..., r_x, c_x]`.
/// * y: 2-D or higher with shape `[..., r_y, c_y]`.
///
/// Optional attributes (see `Attrs`):
/// * adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
/// * adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
///
/// Returns:
/// * `Output`: 3-D or higher with shape `[..., r_o, c_o]`
class BatchMatMulV2 {
 public:
  /// Optional attribute setters for BatchMatMulV2
  struct Attrs {
    /// If `True`, adjoint the slices of `x`. Defaults to `False`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjX(bool x) {
      Attrs ret = *this;
      ret.adj_x_ = x;
      return ret;
    }

    /// If `True`, adjoint the slices of `y`. Defaults to `False`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjY(bool x) {
      Attrs ret = *this;
      ret.adj_y_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradX(bool x) {
      Attrs ret = *this;
      ret.grad_x_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradY(bool x) {
      Attrs ret = *this;
      ret.grad_y_ = x;
      return ret;
    }

    bool adj_x_ = false;
    bool adj_y_ = false;
    bool grad_x_ = false;
    bool grad_y_ = false;
  };
  BatchMatMulV2(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
              ::tensorflow::Input y);
  BatchMatMulV2(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
              ::tensorflow::Input y, const BatchMatMulV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs AdjX(bool x) {
    return Attrs().AdjX(x);
  }
  static Attrs AdjY(bool x) {
    return Attrs().AdjY(x);
  }
  static Attrs GradX(bool x) {
    return Attrs().GradX(x);
  }
  static Attrs GradY(bool x) {
    return Attrs().GradY(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Multiplies slices of two tensors in batches.
///
/// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
/// viewed as an element of a batch), and arranges the individual results
/// in a single output tensor of the same batch size. Each of the
/// individual slices can optionally be adjointed (to adjoint a matrix
/// means to transpose and conjugate it) before multiplication by setting
/// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
///
/// The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
/// and `[..., r_y, c_y]`.
///
/// The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:
///
///     r_o = c_x if adj_x else r_x
///     c_o = r_y if adj_y else c_y
///
/// It is computed as:
///
///     output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
///
/// *NOTE*: `BatchMatMulV3` supports broadcasting in the batch dimensions. More
/// about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
///
///
/// Args:
/// * scope: A Scope object
/// * x: 2-D or higher with shape `[..., r_x, c_x]`.
/// * y: 2-D or higher with shape `[..., r_y, c_y]`.
/// * Tout: If not spcified, Tout is the same type to input type.
///
/// Optional attributes (see `Attrs`):
/// * adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
/// * adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
///
/// Returns:
/// * `Output`: 3-D or higher with shape `[..., r_o, c_o]`
class BatchMatMulV3 {
 public:
  /// Optional attribute setters for BatchMatMulV3
  struct Attrs {
    /// If `True`, adjoint the slices of `x`. Defaults to `False`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjX(bool x) {
      Attrs ret = *this;
      ret.adj_x_ = x;
      return ret;
    }

    /// If `True`, adjoint the slices of `y`. Defaults to `False`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AdjY(bool x) {
      Attrs ret = *this;
      ret.adj_y_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradX(bool x) {
      Attrs ret = *this;
      ret.grad_x_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradY(bool x) {
      Attrs ret = *this;
      ret.grad_y_ = x;
      return ret;
    }

    bool adj_x_ = false;
    bool adj_y_ = false;
    bool grad_x_ = false;
    bool grad_y_ = false;
  };
  BatchMatMulV3(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
              ::tensorflow::Input y, DataType Tout);
  BatchMatMulV3(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
              ::tensorflow::Input y, DataType Tout, const BatchMatMulV3::Attrs&
              attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs AdjX(bool x) {
    return Attrs().AdjX(x);
  }
  static Attrs AdjY(bool x) {
    return Attrs().AdjY(x);
  }
  static Attrs GradX(bool x) {
    return Attrs().GradX(x);
  }
  static Attrs GradY(bool x) {
    return Attrs().GradY(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Compute the regularized incomplete beta integral \\(I_x(a, b)\\).
///
/// The regularized incomplete beta integral is defined as:
///
///
/// \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)
///
/// where
///
///
/// \\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)
///
///
/// is the incomplete beta function and \\(B(a, b)\\) is the *complete*
/// beta function.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Betainc {
 public:
  Betainc(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
        ::tensorflow::Input b, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Counts the number of occurrences of each value in an integer array.
///
/// Outputs a vector with length `size` and the same dtype as `weights`. If
/// `weights` are empty, then index `i` stores the number of times the value `i` is
/// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
/// the value in `weights` at each index where the corresponding value in `arr` is
/// `i`.
///
/// Values in `arr` outside of the range [0, size) are ignored.
///
/// Args:
/// * scope: A Scope object
/// * arr: int32 `Tensor`.
/// * size: non-negative int32 scalar `Tensor`.
/// * weights: is an int32, int64, float32, or float64 `Tensor` with the same
/// shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
/// equal to 1.
///
/// Returns:
/// * `Output`: 1D `Tensor` with length equal to `size`. The counts or summed weights for
/// each value in the range [0, size).
class Bincount {
 public:
  Bincount(const ::tensorflow::Scope& scope, ::tensorflow::Input arr,
         ::tensorflow::Input size, ::tensorflow::Input weights);
  operator ::tensorflow::Output() const { return bins; }
  operator ::tensorflow::Input() const { return bins; }
  ::tensorflow::Node* node() const { return bins.node(); }

  Operation operation;
  ::tensorflow::Output bins;
};

/// Bucketizes 'input' based on 'boundaries'.
///
/// For example, if the inputs are
///     boundaries = [0, 10, 100]
///     input = [[-5, 10000]
///              [150,   10]
///              [5,    100]]
///
/// then the output will be
///     output = [[0, 3]
///               [3, 2]
///               [1, 3]]
///
/// Args:
/// * scope: A Scope object
/// * input: Any shape of Tensor contains with int or float type.
/// * boundaries: A sorted list of floats gives the boundary of the buckets.
///
/// Returns:
/// * `Output`: Same shape with 'input', each value of input replaced with bucket index.
///
/// @compatibility(numpy)
/// Equivalent to np.digitize.
/// @end_compatibility
class Bucketize {
 public:
  Bucketize(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
          gtl::ArraySlice<float>& boundaries);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Cast x of type SrcT to y of DstT.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Cast {
 public:
  /// Optional attribute setters for Cast
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Truncate(bool x) {
      Attrs ret = *this;
      ret.Truncate_ = x;
      return ret;
    }

    bool Truncate_ = false;
  };
  Cast(const ::tensorflow::Scope& scope, ::tensorflow::Input x, DataType DstT);
  Cast(const ::tensorflow::Scope& scope, ::tensorflow::Input x, DataType DstT,
     const Cast::Attrs& attrs);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  static Attrs Truncate(bool x) {
    return Attrs().Truncate(x);
  }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns element-wise smallest integer not less than x.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Ceil {
 public:
  Ceil(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Clips tensor values to a specified min and max.
///
/// Given a tensor `t`, this operation returns a tensor of the same type and
/// shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
/// Any values less than `clip_value_min` are set to `clip_value_min`. Any values
/// greater than `clip_value_max` are set to `clip_value_max`.
///
/// Args:
/// * scope: A Scope object
/// * t: A `Tensor`.
/// * clip_value_min: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
/// as `t`. The minimum value to clip by.
/// * clip_value_max: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
/// as `t`. The maximum value to clip by.
///
/// Returns:
/// * `Output`: A clipped `Tensor` with the same shape as input 't'.
class ClipByValue {
 public:
  ClipByValue(const ::tensorflow::Scope& scope, ::tensorflow::Input t,
            ::tensorflow::Input clip_value_min, ::tensorflow::Input
            clip_value_max);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Converts two real numbers to a complex number.
///
/// Given a tensor `real` representing the real part of a complex number, and a
/// tensor `imag` representing the imaginary part of a complex number, this
/// operation returns complex numbers elementwise of the form \\(a + bj\\), where
/// *a* represents the `real` part and *b* represents the `imag` part.
///
/// The input tensors `real` and `imag` must have the same shape.
///
/// For example:
///
/// ```
/// # tensor 'real' is [2.25, 3.25]
/// # tensor `imag` is [4.75, 5.75]
/// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The out tensor.
class Complex {
 public:
  /// Optional attribute setters for Complex
  struct Attrs {
    /// Defaults to DT_COMPLEX64
    TF_MUST_USE_RESULT Attrs Tout(DataType x) {
      Attrs ret = *this;
      ret.Tout_ = x;
      return ret;
    }

    DataType Tout_ = DT_COMPLEX64;
  };
  Complex(const ::tensorflow::Scope& scope, ::tensorflow::Input real,
        ::tensorflow::Input imag);
  Complex(const ::tensorflow::Scope& scope, ::tensorflow::Input real,
        ::tensorflow::Input imag, const Complex::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs Tout(DataType x) {
    return Attrs().Tout(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Computes the complex absolute value of a tensor.
///
/// Given a tensor `x` of complex numbers, this operation returns a tensor of type
/// `float` or `double` that is the absolute value of each element in `x`. All
/// elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
/// value is computed as \\( \sqrt{a^2 + b^2}\\).
///
/// For example:
///
/// >>> x = tf.complex(3.0, 4.0)
/// >>> print((tf.raw_ops.ComplexAbs(x=x, Tout=tf.dtypes.float32, name=None)).numpy())
/// 5.0
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class ComplexAbs {
 public:
  /// Optional attribute setters for ComplexAbs
  struct Attrs {
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs Tout(DataType x) {
      Attrs ret = *this;
      ret.Tout_ = x;
      return ret;
    }

    DataType Tout_ = DT_FLOAT;
  };
  ComplexAbs(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  ComplexAbs(const ::tensorflow::Scope& scope, ::tensorflow::Input x, const
           ComplexAbs::Attrs& attrs);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  static Attrs Tout(DataType x) {
    return Attrs().Tout(x);
  }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns the complex conjugate of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// complex numbers that are the complex conjugate of each element in `input`. The
/// complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
/// real part and *b* is the imaginary part.
///
/// The complex conjugate returned by this operation is of the form \\(a - bj\\).
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Conj {
 public:
  Conj(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes cos of x element-wise.
///
///   Given an input tensor, this function computes cosine of every
///   element in the tensor. Input range is `(-inf, inf)` and
///   output range is `[-1,1]`. If input lies outside the boundary, `nan`
///   is returned.
///
///   ```python
///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
///   tf.math.cos(x) ==> [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Cos {
 public:
  Cos(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes hyperbolic cosine of x element-wise.
///
///   Given an input tensor, this function computes hyperbolic cosine of every
///   element in the tensor. Input range is `[-inf, inf]` and output range
///   is `[1, inf]`.
///
///   ```python
///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
///   tf.math.cosh(x) ==> [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Cosh {
 public:
  Cosh(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Compute the pairwise cross product.
///
/// `a` and `b` must be the same shape; they can either be simple 3-element vectors,
/// or any shape where the innermost dimension is 3. In the latter case, each pair
/// of corresponding 3-element vectors is cross-multiplied independently.
///
/// Args:
/// * scope: A Scope object
/// * a: A tensor containing 3-element vectors.
/// * b: Another tensor, of same type and shape as `a`.
///
/// Returns:
/// * `Output`: Pairwise cross product of the vectors in `a` and `b`.
class Cross {
 public:
  Cross(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
      ::tensorflow::Input b);
  operator ::tensorflow::Output() const { return product; }
  operator ::tensorflow::Input() const { return product; }
  ::tensorflow::Node* node() const { return product.node(); }

  Operation operation;
  ::tensorflow::Output product;
};

/// Compute the cumulative product of the tensor `x` along `axis`.
///
/// By default, this op performs an inclusive cumprod, which means that the first
/// element of the input is identical to the first element of the output:
///
/// ```python
/// tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
/// ```
///
/// By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
/// performed instead:
///
/// ```python
/// tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
/// ```
///
/// By setting the `reverse` kwarg to `True`, the cumprod is performed in the
/// opposite direction:
///
/// ```python
/// tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
/// ```
///
/// This is more efficient than using separate `tf.reverse` ops.
///
/// The `reverse` and `exclusive` kwargs can also be combined:
///
/// ```python
/// tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
/// `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
/// `complex128`, `qint8`, `quint8`, `qint32`, `half`.
/// * axis: A `Tensor` of type `int32` (default: 0). Must be in the range
/// `[-rank(x), rank(x))`.
///
/// Optional attributes (see `Attrs`):
/// * exclusive: If `True`, perform exclusive cumprod.
/// * reverse: A `bool` (default: False).
///
/// Returns:
/// * `Output`: The out tensor.
class Cumprod {
 public:
  /// Optional attribute setters for Cumprod
  struct Attrs {
    /// If `True`, perform exclusive cumprod.
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
  Cumprod(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
        ::tensorflow::Input axis);
  Cumprod(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
        ::tensorflow::Input axis, const Cumprod::Attrs& attrs);
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

/// Compute the cumulative sum of the tensor `x` along `axis`.
///
/// By default, this op performs an inclusive cumsum, which means that the first
/// element of the input is identical to the first element of the output:
///
/// ```python
/// tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
/// ```
///
/// By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
/// performed instead:
///
/// ```python
/// tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
/// ```
///
/// By setting the `reverse` kwarg to `True`, the cumsum is performed in the
/// opposite direction:
///
/// ```python
/// tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
/// ```
///
/// This is more efficient than using separate `tf.reverse` ops.
///
/// The `reverse` and `exclusive` kwargs can also be combined:
///
/// ```python
/// tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
/// `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
/// `complex128`, `qint8`, `quint8`, `qint32`, `half`.
/// * axis: A `Tensor` of type `int32` (default: 0). Must be in the range
/// `[-rank(x), rank(x))`.
///
/// Optional attributes (see `Attrs`):
/// * exclusive: If `True`, perform exclusive cumsum.
/// * reverse: A `bool` (default: False).
///
/// Returns:
/// * `Output`: The out tensor.
class Cumsum {
 public:
  /// Optional attribute setters for Cumsum
  struct Attrs {
    /// If `True`, perform exclusive cumsum.
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
  Cumsum(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
       ::tensorflow::Input axis);
  Cumsum(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
       ::tensorflow::Input axis, const Cumsum::Attrs& attrs);
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

/// Counts the number of occurrences of each value in an integer array.
///
/// Outputs a vector with length `size` and the same dtype as `weights`. If
/// `weights` are empty, then index `i` stores the number of times the value `i` is
/// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
/// the value in `weights` at each index where the corresponding value in `arr` is
/// `i`.
///
/// Values in `arr` outside of the range [0, size) are ignored.
///
/// Args:
/// * scope: A Scope object
/// * input: 1D or 2D int `Tensor`.
/// * size: non-negative int scalar `Tensor`.
/// * weights: is an int32, int64, float32, or float64 `Tensor` with the same
/// shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
/// equal to 1.
///
/// Optional attributes (see `Attrs`):
/// * binary_output: bool; Whether the kernel should count the appearance or number of occurrences.
///
/// Returns:
/// * `Output`: 1D `Tensor` with length equal to `size` or 2D `Tensor` with [batch_size, `size`].
/// The counts or summed weights for each value in the range [0, size).
class DenseBincount {
 public:
  /// Optional attribute setters for DenseBincount
  struct Attrs {
    /// bool; Whether the kernel should count the appearance or number of occurrences.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs BinaryOutput(bool x) {
      Attrs ret = *this;
      ret.binary_output_ = x;
      return ret;
    }

    bool binary_output_ = false;
  };
  DenseBincount(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input size, ::tensorflow::Input weights);
  DenseBincount(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input size, ::tensorflow::Input weights, const
              DenseBincount::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BinaryOutput(bool x) {
    return Attrs().BinaryOutput(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes Psi, the derivative of Lgamma (the log of the absolute value of
///
/// `Gamma(x)`), element-wise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Digamma {
 public:
  Digamma(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns x / y element-wise.
///
/// *NOTE*: `Div` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Div {
 public:
  Div(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
    ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns 0 if the denominator is zero.
///
///
/// *NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class DivNoNan {
 public:
  DivNoNan(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns the truth value of (x == y) element-wise.
///
/// *NOTE*: `Equal` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// ```python
/// x = tf.constant([2, 4])
/// y = tf.constant(2)
/// tf.math.equal(x, y) ==> array([True, False])
///
/// x = tf.constant([2, 4])
/// y = tf.constant([2, 4])
/// tf.math.equal(x, y) ==> array([True,  True])
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Equal {
 public:
  /// Optional attribute setters for Equal
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs IncompatibleShapeError(bool x) {
      Attrs ret = *this;
      ret.incompatible_shape_error_ = x;
      return ret;
    }

    bool incompatible_shape_error_ = true;
  };
  Equal(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
      ::tensorflow::Input y);
  Equal(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
      ::tensorflow::Input y, const Equal::Attrs& attrs);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  static Attrs IncompatibleShapeError(bool x) {
    return Attrs().IncompatibleShapeError(x);
  }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the [Gauss error function](https://en.wikipedia.org/wiki/Error_function) of `x` element-wise. In statistics, for non-negative values of $x$, the error function has the following interpretation: for a random variable $Y$ that is normally distributed with mean 0 and variance $1/\sqrt{2}$, $erf(x)$ is the probability that $Y$ falls in the range $[−x, x]$.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Erf {
 public:
  Erf(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes the complementary error function of `x` element-wise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Erfc {
 public:
  Erfc(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Erfinv {
 public:
  Erfinv(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes the euclidean norm of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
class EuclideanNorm {
 public:
  /// Optional attribute setters for EuclideanNorm
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
  EuclideanNorm(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input axis);
  EuclideanNorm(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input axis, const EuclideanNorm::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes exponential of x element-wise.  \\(y = e^x\\).
///
///   This function computes the exponential of every element in the input tensor.
///   i.e. `exp(x)` or `e^(x)`, where `x` is the input tensor.
///   `e` denotes Euler's number and is approximately equal to 2.718281.
///   Output is positive for any real input.
///
///   ```python
///   x = tf.constant(2.0)
///   tf.math.exp(x) ==> 7.389056
///
///   x = tf.constant([2.0, 8.0])
///   tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)
///   ```
///
///   For complex numbers, the exponential value is calculated as follows:
///
///   ```
///   e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
///   ```
///
///   Let's consider complex number 1+1j as an example.
///   e^1 * (cos 1 + i sin 1) = 2.7182818284590 * (0.54030230586+0.8414709848j)
///
///   ```python
///   x = tf.constant(1 + 1j)
///   tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Exp {
 public:
  Exp(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes `exp(x) - 1` element-wise.
///
///   i.e. `exp(x) - 1` or `e^(x) - 1`, where `x` is the input tensor.
///   `e` denotes Euler's number and is approximately equal to 2.718281.
///
///   ```python
///   x = tf.constant(2.0)
///   tf.math.expm1(x) ==> 6.389056
///
///   x = tf.constant([2.0, 8.0])
///   tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)
///
///   x = tf.constant(1 + 1j)
///   tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Expm1 {
 public:
  Expm1(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns element-wise largest integer not greater than x.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Floor {
 public:
  Floor(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns x // y element-wise.
///
/// *NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class FloorDiv {
 public:
  FloorDiv(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns element-wise remainder of division.
///
/// This follows Python semantics in that the
/// result here is consistent with a flooring divide. E.g.
/// `floor(x / y) * y + floormod(x, y) = x`, regardless of the signs of x and y.
///
/// *NOTE*: `FloorMod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class FloorMod {
 public:
  FloorMod(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns the truth value of (x > y) element-wise.
///
/// *NOTE*: `Greater` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Example:
///
/// ```python
/// x = tf.constant([5, 4, 6])
/// y = tf.constant([5, 2, 5])
/// tf.math.greater(x, y) ==> [False, True, True]
///
/// x = tf.constant([5, 4, 6])
/// y = tf.constant([5])
/// tf.math.greater(x, y) ==> [False, False, True]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Greater {
 public:
  Greater(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
        ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns the truth value of (x >= y) element-wise.
///
/// *NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Example:
///
/// ```python
/// x = tf.constant([5, 4, 6, 7])
/// y = tf.constant([5, 2, 5, 10])
/// tf.math.greater_equal(x, y) ==> [True, True, True, False]
///
/// x = tf.constant([5, 4, 6, 7])
/// y = tf.constant([5])
/// tf.math.greater_equal(x, y) ==> [True, False, True, True]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class GreaterEqual {
 public:
  GreaterEqual(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
             ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Return histogram of values.
///
/// Given the tensor `values`, this operation returns a rank 1 histogram counting
/// the number of entries in `values` that fall into every bin.  The bins are
/// equal width and determined by the arguments `value_range` and `nbins`.
///
/// ```python
/// # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
/// nbins = 5
/// value_range = [0.0, 5.0]
/// new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]
///
/// with tf.get_default_session() as sess:
///   hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
///   variables.global_variables_initializer().run()
///   sess.run(hist) => [2, 1, 1, 0, 2]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * values: Numeric `Tensor`.
/// * value_range: Shape [2] `Tensor` of same `dtype` as `values`.
/// values <= value_range[0] will be mapped to hist[0],
/// values >= value_range[1] will be mapped to hist[-1].
/// * nbins: Scalar `int32 Tensor`.  Number of histogram bins.
///
/// Returns:
/// * `Output`: A 1-D `Tensor` holding histogram of values.
class HistogramFixedWidth {
 public:
  /// Optional attribute setters for HistogramFixedWidth
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs Dtype(DataType x) {
      Attrs ret = *this;
      ret.dtype_ = x;
      return ret;
    }

    DataType dtype_ = DT_INT32;
  };
  HistogramFixedWidth(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    values, ::tensorflow::Input value_range,
                    ::tensorflow::Input nbins);
  HistogramFixedWidth(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    values, ::tensorflow::Input value_range,
                    ::tensorflow::Input nbins, const
                    HistogramFixedWidth::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs Dtype(DataType x) {
    return Attrs().Dtype(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Compute the lower regularized incomplete Gamma function `P(a, x)`.
///
/// The lower regularized incomplete Gamma function is defined as:
///
///
/// \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)
///
/// where
///
/// \\(gamma(a, x) = \\int_{0}^{x} t^{a-1} exp(-t) dt\\)
///
/// is the lower incomplete Gamma function.
///
/// Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
/// Gamma function.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Igamma {
 public:
  Igamma(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
       ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Compute the upper regularized incomplete Gamma function `Q(a, x)`.
///
/// The upper regularized incomplete Gamma function is defined as:
///
/// \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)
///
/// where
///
/// \\(Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt\\)
///
/// is the upper incomplete Gamma function.
///
/// Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
/// Gamma function.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Igammac {
 public:
  Igammac(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
        ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns the imaginary part of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the imaginary part of each element in `input`. All
/// elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
/// is the real part and *b* is the imaginary part returned by this operation.
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.imag(input) ==> [4.75, 5.75]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Imag {
 public:
  /// Optional attribute setters for Imag
  struct Attrs {
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs Tout(DataType x) {
      Attrs ret = *this;
      ret.Tout_ = x;
      return ret;
    }

    DataType Tout_ = DT_FLOAT;
  };
  Imag(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Imag(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
     Imag::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Tout(DataType x) {
    return Attrs().Tout(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the reciprocal of x element-wise.
///
/// I.e., \\(y = 1 / x\\).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Inv {
 public:
  Inv(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns which elements of x are finite.
///
/// @compatibility(numpy)
/// Equivalent to np.isfinite
/// @end_compatibility
///
/// Example:
///
/// ```python
/// x = tf.constant([5.0, 4.8, 6.8, np.inf, np.nan])
/// tf.math.is_finite(x) ==> [True, True, True, False, False]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class IsFinite {
 public:
  IsFinite(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns which elements of x are Inf.
///
/// @compatibility(numpy)
/// Equivalent to np.isinf
/// @end_compatibility
///
/// Example:
///
/// ```python
/// x = tf.constant([5.0, np.inf, 6.8, np.inf])
/// tf.math.is_inf(x) ==> [False, True, False, True]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class IsInf {
 public:
  IsInf(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns which elements of x are NaN.
///
/// @compatibility(numpy)
/// Equivalent to np.isnan
/// @end_compatibility
///
/// Example:
///
/// ```python
/// x = tf.constant([5.0, np.nan, 6.8, np.nan, np.inf])
/// tf.math.is_nan(x) ==> [False, True, False, True, False]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class IsNan {
 public:
  IsNan(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns the truth value of (x < y) element-wise.
///
/// *NOTE*: `Less` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Example:
///
/// ```python
/// x = tf.constant([5, 4, 6])
/// y = tf.constant([5])
/// tf.math.less(x, y) ==> [False, True, False]
///
/// x = tf.constant([5, 4, 6])
/// y = tf.constant([5, 6, 7])
/// tf.math.less(x, y) ==> [False, True, True]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Less {
 public:
  Less(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
     ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns the truth value of (x <= y) element-wise.
///
/// *NOTE*: `LessEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Example:
///
/// ```python
/// x = tf.constant([5, 4, 6])
/// y = tf.constant([5])
/// tf.math.less_equal(x, y) ==> [True, True, False]
///
/// x = tf.constant([5, 4, 6])
/// y = tf.constant([5, 6, 6])
/// tf.math.less_equal(x, y) ==> [True, True, True]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class LessEqual {
 public:
  LessEqual(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
          ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the log of the absolute value of `Gamma(x)` element-wise.
///
///   For positive numbers, this function computes log((input - 1)!) for every element in the tensor.
///   `lgamma(5) = log((5-1)!) = log(4!) = log(24) = 3.1780539`
///
/// Example:
///
/// ```python
/// x = tf.constant([0, 0.5, 1, 4.5, -4, -5.6])
/// tf.math.lgamma(x) ==> [inf, 0.5723649, 0., 2.4537368, inf, -4.6477685]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Lgamma {
 public:
  Lgamma(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes natural logarithm of x element-wise.
///
/// I.e., \\(y = \log_e x\\).
///
/// Example:
///
/// ```python
/// x = tf.constant([0, 0.5, 1, 5])
/// tf.math.log(x) ==> [-inf, -0.6931472,  0. ,  1.609438]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Log {
 public:
  Log(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes natural logarithm of (1 + x) element-wise.
///
/// I.e., \\(y = \log_e (1 + x)\\).
///
/// Example:
///
/// ```python
/// x = tf.constant([0, 0.5, 1, 5])
/// tf.math.log1p(x) ==> [0., 0.4054651, 0.6931472, 1.7917595]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Log1p {
 public:
  Log1p(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns the truth value of x AND y element-wise.
///
/// *NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class LogicalAnd {
 public:
  LogicalAnd(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
           ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns the truth value of `NOT x` element-wise.
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor` of type `bool`.
///
/// Returns:
/// * `Output`: A `Tensor` of type `bool` with the same shape as `x`. The logical negation of `x`.
class LogicalNot {
 public:
  LogicalNot(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns the truth value of x OR y element-wise.
///
/// *NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class LogicalOr {
 public:
  LogicalOr(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
          ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Multiply the matrix "a" by the matrix "b".
///
/// The inputs must be two-dimensional matrices and the inner dimension of
/// "a" (after being transposed if transpose_a is true) must match the
/// outer dimension of "b" (after being transposed if transposed_b is
/// true).
///
/// *Note*: The default kernel implementation for MatMul on GPUs uses
/// cublas.
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * transpose_a: If true, "a" is transposed before multiplication.
/// * transpose_b: If true, "b" is transposed before multiplication.
///
/// Returns:
/// * `Output`: The product tensor.
class MatMul {
 public:
  /// Optional attribute setters for MatMul
  struct Attrs {
    /// If true, "a" is transposed before multiplication.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs TransposeA(bool x) {
      Attrs ret = *this;
      ret.transpose_a_ = x;
      return ret;
    }

    /// If true, "b" is transposed before multiplication.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs TransposeB(bool x) {
      Attrs ret = *this;
      ret.transpose_b_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradA(bool x) {
      Attrs ret = *this;
      ret.grad_a_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs GradB(bool x) {
      Attrs ret = *this;
      ret.grad_b_ = x;
      return ret;
    }

    bool transpose_a_ = false;
    bool transpose_b_ = false;
    bool grad_a_ = false;
    bool grad_b_ = false;
  };
  MatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
       ::tensorflow::Input b);
  MatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
       ::tensorflow::Input b, const MatMul::Attrs& attrs);
  operator ::tensorflow::Output() const { return product; }
  operator ::tensorflow::Input() const { return product; }
  ::tensorflow::Node* node() const { return product.node(); }

  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs GradA(bool x) {
    return Attrs().GradA(x);
  }
  static Attrs GradB(bool x) {
    return Attrs().GradB(x);
  }

  Operation operation;
  ::tensorflow::Output product;
};

/// Computes the maximum of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
///
/// Aliases:
/// * ReduceMax
class Max {
 public:
  /// Optional attribute setters for Max
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
  Max(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis);
  Max(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis, const Max::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};
typedef Max ReduceMax;

/// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
/// *NOTE*: `Maximum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Maximum {
 public:
  Maximum(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
        ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the mean of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
///
/// Aliases:
/// * ReduceMean
class Mean {
 public:
  /// Optional attribute setters for Mean
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
  Mean(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
     ::tensorflow::Input axis);
  Mean(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
     ::tensorflow::Input axis, const Mean::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};
typedef Mean ReduceMean;

/// Computes the minimum of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
///
/// Aliases:
/// * ReduceMin
class Min {
 public:
  /// Optional attribute setters for Min
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
  Min(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis);
  Min(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis, const Min::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};
typedef Min ReduceMin;

/// Returns the min of x and y (i.e. x < y ? x : y) element-wise.
///
/// *NOTE*: `Minimum` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Minimum {
 public:
  Minimum(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
        ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns element-wise remainder of division. This emulates C semantics in that
///
/// the result here is consistent with a truncating divide. E.g.
/// `tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.
///
/// *NOTE*: `Mod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Mod {
 public:
  Mod(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
    ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns x * y element-wise.
///
/// *NOTE*: `Multiply` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
///
/// Aliases:
/// * Mul
class Multiply {
 public:
  Multiply(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};
typedef Multiply Mul;

/// Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.
///
/// *NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class MulNoNan {
 public:
  MulNoNan(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Ndtri {
 public:
  Ndtri(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes numerical negative value element-wise.
///
/// I.e., \\(y = -x\\).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
///
/// Aliases:
/// * Neg
class Negate {
 public:
  Negate(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};
typedef Negate Neg;

/// Returns the next representable value of `x1` in the direction of `x2`, element-wise.
///
/// This operation returns the same result as the C++ std::nextafter function.
///
/// It can also return a subnormal number.
///
/// @compatibility(cpp)
/// Equivalent to C++ std::nextafter function.
/// @end_compatibility
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class NextAfter {
 public:
  NextAfter(const ::tensorflow::Scope& scope, ::tensorflow::Input x1,
          ::tensorflow::Input x2);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the truth value of (x != y) element-wise.
///
/// *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class NotEqual {
 public:
  /// Optional attribute setters for NotEqual
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs IncompatibleShapeError(bool x) {
      Attrs ret = *this;
      ret.incompatible_shape_error_ = x;
      return ret;
    }

    bool incompatible_shape_error_ = true;
  };
  NotEqual(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y);
  NotEqual(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y, const NotEqual::Attrs& attrs);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  static Attrs IncompatibleShapeError(bool x) {
    return Attrs().IncompatibleShapeError(x);
  }

  Operation operation;
  ::tensorflow::Output z;
};

/// Compute the polygamma function \\(\psi^{(n)}(x)\\).
///
/// The polygamma function is defined as:
///
///
/// \\(\psi^{(a)}(x) = \frac{d^a}{dx^a} \psi(x)\\)
///
/// where \\(\psi(x)\\) is the digamma function.
/// The polygamma function is defined only for non-negative integer orders \\a\\.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Polygamma {
 public:
  Polygamma(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
          ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the power of one value to another.
///
/// Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
/// corresponding elements in `x` and `y`. For example:
///
/// ```
/// # tensor 'x' is [[2, 2]], [3, 3]]
/// # tensor 'y' is [[8, 16], [2, 3]]
/// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Pow {
 public:
  Pow(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
    ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the product of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
///
/// Aliases:
/// * ReduceProd
class Prod {
 public:
  /// Optional attribute setters for Prod
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
  Prod(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
     ::tensorflow::Input axis);
  Prod(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
     ::tensorflow::Input axis, const Prod::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};
typedef Prod ReduceProd;

/// Convert the quantized 'input' tensor into a lower-precision 'output', using the
///
/// actual distribution of the values to maximize the usage of the lower bit depth
/// and adjusting the output min and max ranges accordingly.
///
/// [input_min, input_max] are scalar floats that specify the range for the float
/// interpretation of the 'input' data. For example, if input_min is -1.0f and
/// input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
/// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
///
/// This operator tries to squeeze as much precision as possible into an output with
/// a lower bit depth by calculating the actual min and max values found in the
/// data. For example, maybe that quint16 input has no values lower than 16,384 and
/// none higher than 49,152. That means only half the range is actually needed, all
/// the float interpretations are between -0.5f and 0.5f, so if we want to compress
/// the data into a quint8 output, we can use that range rather than the theoretical
/// -1.0f to 1.0f that is suggested by the input min and max.
///
/// In practice, this is most useful for taking output from operations like
/// QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
/// may have large potential output ranges, but in practice have a distribution of
/// input values that only uses a small fraction of the possible range. By feeding
/// that output into this operator, we can reduce it from 32 bits down to 8 with
/// minimal loss of accuracy.
///
/// Args:
/// * scope: A Scope object
/// * input_min: The float value that the minimum quantized input value represents.
/// * input_max: The float value that the maximum quantized input value represents.
/// * out_type: The type of the output. Should be a lower bit depth than Tinput.
///
/// Returns:
/// * `Output` output
/// * `Output` output_min: The float value that the minimum quantized output value represents.
/// * `Output` output_max: The float value that the maximum quantized output value represents.
class QuantizeDownAndShrinkRange {
 public:
  QuantizeDownAndShrinkRange(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input input, ::tensorflow::Input
                           input_min, ::tensorflow::Input input_max, DataType
                           out_type);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Returns x + y element-wise, working on quantized buffers.
///
/// Args:
/// * scope: A Scope object
/// * min_x: The float value that the lowest quantized `x` value represents.
/// * max_x: The float value that the highest quantized `x` value represents.
/// * min_y: The float value that the lowest quantized `y` value represents.
/// * max_y: The float value that the highest quantized `y` value represents.
///
/// Returns:
/// * `Output` z
/// * `Output` min_z: The float value that the lowest quantized output value represents.
/// * `Output` max_z: The float value that the highest quantized output value represents.
///
/// *NOTE*: `QuantizedAdd` supports limited forms of broadcasting. More about
/// broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
class QuantizedAdd {
 public:
  /// Optional attribute setters for QuantizedAdd
  struct Attrs {
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs Toutput(DataType x) {
      Attrs ret = *this;
      ret.Toutput_ = x;
      return ret;
    }

    DataType Toutput_ = DT_QINT32;
  };
  QuantizedAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
             ::tensorflow::Input y, ::tensorflow::Input min_x,
             ::tensorflow::Input max_x, ::tensorflow::Input min_y,
             ::tensorflow::Input max_y);
  QuantizedAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
             ::tensorflow::Input y, ::tensorflow::Input min_x,
             ::tensorflow::Input max_x, ::tensorflow::Input min_y,
             ::tensorflow::Input max_y, const QuantizedAdd::Attrs& attrs);

  static Attrs Toutput(DataType x) {
    return Attrs().Toutput(x);
  }

  Operation operation;
  ::tensorflow::Output z;
  ::tensorflow::Output min_z;
  ::tensorflow::Output max_z;
};

/// Perform a quantized matrix multiplication of  `a` by the matrix `b`.
///
/// The inputs must be two-dimensional matrices and the inner dimension of
/// `a` (after being transposed if `transpose_a` is non-zero) must match the
/// outer dimension of `b` (after being transposed if `transposed_b` is
/// non-zero).
///
/// Args:
/// * scope: A Scope object
/// * a: Must be a two-dimensional tensor.
/// * b: Must be a two-dimensional tensor.
/// * min_a: The float value that the lowest quantized `a` value represents.
/// * max_a: The float value that the highest quantized `a` value represents.
/// * min_b: The float value that the lowest quantized `b` value represents.
/// * max_b: The float value that the highest quantized `b` value represents.
///
/// Optional attributes (see `Attrs`):
/// * transpose_a: If true, `a` is transposed before multiplication.
/// * transpose_b: If true, `b` is transposed before multiplication.
/// * Tactivation: The type of output produced by activation function
/// following this operation.
///
/// Returns:
/// * `Output` out
/// * `Output` min_out: The float value that the lowest quantized output value represents.
/// * `Output` max_out: The float value that the highest quantized output value represents.
class QuantizedMatMul {
 public:
  /// Optional attribute setters for QuantizedMatMul
  struct Attrs {
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs Toutput(DataType x) {
      Attrs ret = *this;
      ret.Toutput_ = x;
      return ret;
    }

    /// If true, `a` is transposed before multiplication.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs TransposeA(bool x) {
      Attrs ret = *this;
      ret.transpose_a_ = x;
      return ret;
    }

    /// If true, `b` is transposed before multiplication.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs TransposeB(bool x) {
      Attrs ret = *this;
      ret.transpose_b_ = x;
      return ret;
    }

    /// The type of output produced by activation function
    /// following this operation.
    ///
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs Tactivation(DataType x) {
      Attrs ret = *this;
      ret.Tactivation_ = x;
      return ret;
    }

    DataType Toutput_ = DT_QINT32;
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    DataType Tactivation_ = DT_QUINT8;
  };
  QuantizedMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
                ::tensorflow::Input b, ::tensorflow::Input min_a,
                ::tensorflow::Input max_a, ::tensorflow::Input min_b,
                ::tensorflow::Input max_b);
  QuantizedMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
                ::tensorflow::Input b, ::tensorflow::Input min_a,
                ::tensorflow::Input max_a, ::tensorflow::Input min_b,
                ::tensorflow::Input max_b, const QuantizedMatMul::Attrs& attrs);

  static Attrs Toutput(DataType x) {
    return Attrs().Toutput(x);
  }
  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs Tactivation(DataType x) {
    return Attrs().Tactivation(x);
  }

  Operation operation;
  ::tensorflow::Output out;
  ::tensorflow::Output min_out;
  ::tensorflow::Output max_out;
};

/// Returns x * y element-wise, working on quantized buffers.
///
/// Args:
/// * scope: A Scope object
/// * min_x: The float value that the lowest quantized `x` value represents.
/// * max_x: The float value that the highest quantized `x` value represents.
/// * min_y: The float value that the lowest quantized `y` value represents.
/// * max_y: The float value that the highest quantized `y` value represents.
///
/// Returns:
/// * `Output` z
/// * `Output` min_z: The float value that the lowest quantized output value represents.
/// * `Output` max_z: The float value that the highest quantized output value represents.
///
/// *NOTE*: `QuantizedMul` supports limited forms of broadcasting. More about
/// broadcasting [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
class QuantizedMul {
 public:
  /// Optional attribute setters for QuantizedMul
  struct Attrs {
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs Toutput(DataType x) {
      Attrs ret = *this;
      ret.Toutput_ = x;
      return ret;
    }

    DataType Toutput_ = DT_QINT32;
  };
  QuantizedMul(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
             ::tensorflow::Input y, ::tensorflow::Input min_x,
             ::tensorflow::Input max_x, ::tensorflow::Input min_y,
             ::tensorflow::Input max_y);
  QuantizedMul(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
             ::tensorflow::Input y, ::tensorflow::Input min_x,
             ::tensorflow::Input max_x, ::tensorflow::Input min_y,
             ::tensorflow::Input max_y, const QuantizedMul::Attrs& attrs);

  static Attrs Toutput(DataType x) {
    return Attrs().Toutput(x);
  }

  Operation operation;
  ::tensorflow::Output z;
  ::tensorflow::Output min_z;
  ::tensorflow::Output max_z;
};

/// Counts the number of occurrences of each value in an integer array.
///
/// Outputs a vector with length `size` and the same dtype as `weights`. If
/// `weights` are empty, then index `i` stores the number of times the value `i` is
/// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
/// the value in `weights` at each index where the corresponding value in `arr` is
/// `i`.
///
/// Values in `arr` outside of the range [0, size) are ignored.
///
/// Args:
/// * scope: A Scope object
/// * splits: 1D int64 `Tensor`.
/// * values: 2D int `Tensor`.
/// * size: non-negative int scalar `Tensor`.
/// * weights: is an int32, int64, float32, or float64 `Tensor` with the same
/// shape as `input`, or a length-0 `Tensor`, in which case it acts as all weights
/// equal to 1.
///
/// Optional attributes (see `Attrs`):
/// * binary_output: bool; Whether the kernel should count the appearance or number of occurrences.
///
/// Returns:
/// * `Output`: 1D `Tensor` with length equal to `size` or 2D `Tensor` with [batch_size, `size`].
/// The counts or summed weights for each value in the range [0, size).
class RaggedBincount {
 public:
  /// Optional attribute setters for RaggedBincount
  struct Attrs {
    /// bool; Whether the kernel should count the appearance or number of occurrences.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs BinaryOutput(bool x) {
      Attrs ret = *this;
      ret.binary_output_ = x;
      return ret;
    }

    bool binary_output_ = false;
  };
  RaggedBincount(const ::tensorflow::Scope& scope, ::tensorflow::Input splits,
               ::tensorflow::Input values, ::tensorflow::Input size,
               ::tensorflow::Input weights);
  RaggedBincount(const ::tensorflow::Scope& scope, ::tensorflow::Input splits,
               ::tensorflow::Input values, ::tensorflow::Input size,
               ::tensorflow::Input weights, const RaggedBincount::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BinaryOutput(bool x) {
    return Attrs().BinaryOutput(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Creates a sequence of numbers.
///
/// This operation creates a sequence of numbers that begins at `start` and
/// extends by increments of `delta` up to but not including `limit`.
///
/// For example:
///
/// ```
/// # 'start' is 3
/// # 'limit' is 18
/// # 'delta' is 3
/// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * start: 0-D (scalar). First entry in the sequence.
/// * limit: 0-D (scalar). Upper limit of sequence, exclusive.
/// * delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
///
/// Returns:
/// * `Output`: 1-D.
class Range {
 public:
  Range(const ::tensorflow::Scope& scope, ::tensorflow::Input start,
      ::tensorflow::Input limit, ::tensorflow::Input delta);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the real part of a complex number.
///
/// Given a tensor `input` of complex numbers, this operation returns a tensor of
/// type `float` that is the real part of each element in `input`. All elements in
/// `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
///  part returned by this operation and *b* is the imaginary part.
///
/// For example:
///
/// ```
/// # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
/// tf.real(input) ==> [-2.25, 3.25]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Real {
 public:
  /// Optional attribute setters for Real
  struct Attrs {
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs Tout(DataType x) {
      Attrs ret = *this;
      ret.Tout_ = x;
      return ret;
    }

    DataType Tout_ = DT_FLOAT;
  };
  Real(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Real(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
     Real::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Tout(DataType x) {
    return Attrs().Tout(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns x / y element-wise for real types.
///
/// If `x` and `y` are reals, this will return the floating-point division.
///
/// *NOTE*: `Div` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class RealDiv {
 public:
  RealDiv(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
        ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the reciprocal of x element-wise.
///
/// I.e., \\(y = 1 / x\\).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Reciprocal {
 public:
  Reciprocal(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes a range that covers the actual values present in a quantized tensor.
///
/// Given a quantized tensor described by `(input, input_min, input_max)`, outputs a
/// range that covers the actual values present in that tensor. This op is typically
/// used to produce the `requested_output_min` and `requested_output_max` for
/// `Requantize`.
///
/// Args:
/// * scope: A Scope object
/// * input_min: The float value that the minimum quantized input value represents.
/// * input_max: The float value that the maximum quantized input value represents.
///
/// Returns:
/// * `Output` output_min: The computed min output.
/// * `Output` output_max: the computed max output.
class RequantizationRange {
 public:
  RequantizationRange(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    input, ::tensorflow::Input input_min, ::tensorflow::Input
                    input_max);

  Operation operation;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Converts the quantized `input` tensor into a lower-precision `output`.
///
/// Converts the quantized `input` tensor into a lower-precision `output`, using the
/// output range specified with `requested_output_min` and `requested_output_max`.
///
/// `[input_min, input_max]` are scalar floats that specify the range for the float
/// interpretation of the `input` data. For example, if `input_min` is -1.0f and
/// `input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
/// value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.
///
/// Args:
/// * scope: A Scope object
/// * input_min: The float value that the minimum quantized input value represents.
/// * input_max: The float value that the maximum quantized input value represents.
/// * requested_output_min: The float value that the minimum quantized output value represents.
/// * requested_output_max: The float value that the maximum quantized output value represents.
/// * out_type: The type of the output. Should be a lower bit depth than Tinput.
///
/// Returns:
/// * `Output` output
/// * `Output` output_min: The requested_output_min value is copied into this output.
/// * `Output` output_max: The requested_output_max value is copied into this output.
class Requantize {
 public:
  Requantize(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
           ::tensorflow::Input input_min, ::tensorflow::Input input_max,
           ::tensorflow::Input requested_output_min, ::tensorflow::Input
           requested_output_max, DataType out_type);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Returns element-wise integer closest to x.
///
/// If the result is midway between two representable values,
/// the even representable is chosen.
/// For example:
///
/// ```
/// rint(-1.5) ==> -2.0
/// rint(0.5000001) ==> 1.0
/// rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Rint {
 public:
  Rint(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Rounds the values of a tensor to the nearest integer, element-wise.
///
/// Rounds half to even.  Also known as bankers rounding. If you want to round
/// according to the current system rounding mode use std::cint.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Round {
 public:
  Round(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes reciprocal of square root of x element-wise.
///
/// I.e., \\(y = 1 / \sqrt{x}\\).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Rsqrt {
 public:
  Rsqrt(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes the maximum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the max is empty for a given segment ID `i`, `output[i] = 0`.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
/// and an error is thrown for indices that are not increasing. On GPU, this
/// does not throw an error for unsorted indices. On GPU, out-of-order indices
/// result in safe but unspecified behavior, which may include treating
/// out-of-order indices as the same as a smaller following index.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMax.png" alt>
/// </div>
///
/// For example:
///
/// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> tf.math.segment_max(c, tf.constant([0, 0, 1])).numpy()
/// array([[4, 3, 3, 4],
///        [5, 6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SegmentMax {
 public:
  SegmentMax(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
           ::tensorflow::Input segment_ids);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the maximum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the maximum is empty for a given segment ID `i`, it outputs the smallest
/// possible value for the specific numeric type,
/// `output[i] = numeric_limits<T>::lowest()`.
///
/// Note: That this op is currently only supported with jit_compile=True.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
/// and an error is thrown for indices that are not increasing. On GPU, this
/// does not throw an error for unsorted indices. On GPU, out-of-order indices
/// result in safe but unspecified behavior, which may include treating
/// out-of-order indices as the same as a smaller following index.
///
/// The only difference with SegmentMax is the additional input  `num_segments`.
/// This helps in evaluating the output shape in compile time.
/// `num_segments` should be consistent with segment_ids.
/// e.g. Max(segment_ids) should be equal to `num_segments` - 1 for a 1-d segment_ids
/// With inconsistent num_segments, the op still runs. only difference is,
/// the output takes the size of num_segments irrespective of size of segment_ids and data.
/// for num_segments less than expected output size, the last elements are ignored
/// for num_segments more than the expected output size, last elements are assigned
/// smallest possible value for the specific numeric type.
///
/// For example:
///
/// >>> @tf.function(jit_compile=True)
/// ... def test(c):
/// ...   return tf.raw_ops.SegmentMaxV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
/// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> test(c).numpy()
/// array([[4, 3, 3, 4],
///        [5, 6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimensionw which has size
/// `num_segments`.
///
class SegmentMaxV2 {
 public:
  SegmentMaxV2(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
             ::tensorflow::Input segment_ids, ::tensorflow::Input num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the mean along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
/// over `j` such that `segment_ids[j] == i` and `N` is the total number of
/// values summed.
///
/// If the mean is empty for a given segment ID `i`, `output[i] = 0`.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
/// and an error is thrown for indices that are not increasing. On GPU, this
/// does not throw an error for unsorted indices. On GPU, out-of-order indices
/// result in safe but unspecified behavior, which may include treating
/// out-of-order indices as a smaller following index when computing the numerator
/// of the mean.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMean.png" alt>
/// </div>
///
/// For example:
///
/// >>> c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> tf.math.segment_mean(c, tf.constant([0, 0, 1])).numpy()
/// array([[2.5, 2.5, 2.5, 2.5],
///        [5., 6., 7., 8.]], dtype=float32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SegmentMean {
 public:
  SegmentMean(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
            ::tensorflow::Input segment_ids);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the minimum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the min is empty for a given segment ID `i`, `output[i] = 0`.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
/// and an error is thrown for indices that are not increasing. On GPU, this
/// does not throw an error for unsorted indices. On GPU, out-of-order indices
/// result in safe but unspecified behavior, which may include treating
/// out-of-order indices as the same as a smaller following index.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentMin.png" alt>
/// </div>
///
/// For example:
///
/// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> tf.math.segment_min(c, tf.constant([0, 0, 1])).numpy()
/// array([[1, 2, 2, 1],
///        [5, 6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SegmentMin {
 public:
  SegmentMin(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
           ::tensorflow::Input segment_ids);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the minimum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the minimum is empty for a given segment ID `i`, it outputs the largest
/// possible value for the specific numeric type,
/// `output[i] = numeric_limits<T>::max()`.
///
/// Note: That this op is currently only supported with jit_compile=True.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
/// and an error is thrown for indices that are not increasing. On GPU, this
/// does not throw an error for unsorted indices. On GPU, out-of-order indices
/// result in safe but unspecified behavior, which may include treating
/// out-of-order indices as the same as a smaller following index.
///
/// The only difference with SegmentMin is the additional input  `num_segments`.
/// This helps in evaluating the output shape in compile time.
/// `num_segments` should be consistent with segment_ids.
/// e.g. Max(segment_ids) should be equal to `num_segments` - 1 for a 1-d segment_ids
/// With inconsistent num_segments, the op still runs. only difference is,
/// the output takes the size of num_segments irrespective of size of segment_ids and data.
/// for num_segments less than expected output size, the last elements are ignored
/// for num_segments more than the expected output size, last elements are assigned
/// the largest possible value for the specific numeric type.
///
/// For example:
///
/// >>> @tf.function(jit_compile=True)
/// ... def test(c):
/// ...   return tf.raw_ops.SegmentMinV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
/// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> test(c).numpy()
/// array([[1, 2, 2, 1],
///        [5, 6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimensionw which has size
/// `num_segments`.
class SegmentMinV2 {
 public:
  SegmentMinV2(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
             ::tensorflow::Input segment_ids, ::tensorflow::Input num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the product along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \prod_j data_j\\) where the product is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the product is empty for a given segment ID `i`, `output[i] = 1`.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
/// and an error is thrown for indices that are not increasing. On GPU, this
/// does not throw an error for unsorted indices. On GPU, out-of-order indices
/// result in safe but unspecified behavior, which may include treating
/// out-of-order indices as the same as a smaller following index.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentProd.png" alt>
/// </div>
///
/// For example:
///
/// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> tf.math.segment_prod(c, tf.constant([0, 0, 1])).numpy()
/// array([[4, 6, 6, 4],
///        [5, 6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SegmentProd {
 public:
  SegmentProd(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
            ::tensorflow::Input segment_ids);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the product along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \prod_j data_j\\) where the product is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the product is empty for a given segment ID `i`, `output[i] = 1`.
///
/// Note: That this op is currently only supported with jit_compile=True.
///
/// The only difference with SegmentProd is the additional input  `num_segments`.
/// This helps in evaluating the output shape in compile time.
/// `num_segments` should be consistent with segment_ids.
/// e.g. Max(segment_ids) - 1 should be equal to `num_segments` for a 1-d segment_ids
/// With inconsistent num_segments, the op still runs. only difference is,
/// the output takes the size of num_segments irrespective of size of segment_ids and data.
/// for num_segments less than expected output size, the last elements are ignored
/// for num_segments more than the expected output size, last elements are assigned 1.
///
/// For example:
///
/// >>> @tf.function(jit_compile=True)
/// ... def test(c):
/// ...   return tf.raw_ops.SegmentProdV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
/// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> test(c).numpy()
/// array([[4, 6, 6, 4],
///        [5, 6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimensionw which has size
/// `num_segments`.
class SegmentProdV2 {
 public:
  SegmentProdV2(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
              ::tensorflow::Input segment_ids, ::tensorflow::Input
              num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the sum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \sum_j data_j\\) where sum is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be sorted,
/// and an error is thrown for indices that are not increasing. On GPU, this
/// does not throw an error for unsorted indices. On GPU, out-of-order indices
/// result in safe but unspecified behavior, which may include treating
/// out-of-order indices as the same as a smaller following index.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/SegmentSum.png" alt>
/// </div>
///
/// For example:
///
/// >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
/// >>> tf.math.segment_sum(c, tf.constant([0, 0, 1])).numpy()
/// array([[5, 5, 5, 5],
///        [5, 6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SegmentSum {
 public:
  SegmentSum(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
           ::tensorflow::Input segment_ids);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the sum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output_i = \sum_j data_j\\) where sum is over `j` such
/// that `segment_ids[j] == i`.
///
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
///
/// Note that this op is currently only supported with jit_compile=True.
/// </div>
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A 1-D tensor whose size is equal to the size of `data`'s
/// first dimension.  Values should be sorted and can be repeated.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be sorted on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimension which has size
/// `num_segments`.
class SegmentSumV2 {
 public:
  SegmentSumV2(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
             ::tensorflow::Input segment_ids, ::tensorflow::Input num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Selects elements from `x` or `y`, depending on `condition`.
///
/// The `x`, and `y` tensors must all have the same shape, and the
/// output will also have that shape.
///
/// The `condition` tensor must be a scalar if `x` and `y` are scalars.
/// If `x` and `y` are vectors or higher rank, then `condition` must be either a
/// scalar, a vector with size matching the first dimension of `x`, or must have
/// the same shape as `x`.
///
/// The `condition` tensor acts as a mask that chooses, based on the value at each
/// element, whether the corresponding element / row in the output should be
/// taken from `x` (if true) or `y` (if false).
///
/// If `condition` is a vector and `x` and `y` are higher rank matrices, then
/// it chooses which row (outer dimension) to copy from `x` and `y`.
/// If `condition` has the same shape as `x` and `y`, then it chooses which
/// element to copy from `x` and `y`.
///
/// For example:
///
/// ```python
/// # 'condition' tensor is [[True,  False]
/// #                        [False, True]]
/// # 't' is [[1, 2],
/// #         [3, 4]]
/// # 'e' is [[5, 6],
/// #         [7, 8]]
/// select(condition, t, e)  # => [[1, 6], [7, 4]]
///
///
/// # 'condition' tensor is [True, False]
/// # 't' is [[1, 2],
/// #         [3, 4]]
/// # 'e' is [[5, 6],
/// #         [7, 8]]
/// select(condition, t, e) ==> [[1, 2],
///                              [7, 8]]
///
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: = A `Tensor` which may have the same shape as `condition`.
/// If `condition` is rank 1, `x` may have higher rank,
/// but its first dimension must match the size of `condition`.
/// * y: = A `Tensor` with the same type and shape as `x`.
///
/// Returns:
/// * `Output`: = A `Tensor` with the same type and shape as `x` and `y`.
class Where3 {
 public:
  Where3(const ::tensorflow::Scope& scope, ::tensorflow::Input condition,
       ::tensorflow::Input x, ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class SelectV2 {
 public:
  SelectV2(const ::tensorflow::Scope& scope, ::tensorflow::Input condition,
         ::tensorflow::Input t, ::tensorflow::Input e);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes sigmoid of `x` element-wise.
///
/// Specifically, `y = 1 / (1 + exp(-x))`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Sigmoid {
 public:
  Sigmoid(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns an element-wise indication of the sign of a number.
///
/// `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.
///
/// For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
///
/// Example usage:
/// >>> tf.math.sign([0., 2., -3.])
/// <tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 0.,  1., -1.], dtype=float32)>
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Sign {
 public:
  Sign(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes sine of x element-wise.
///
///   Given an input tensor, this function computes sine of every
///   element in the tensor. Input range is `(-inf, inf)` and
///   output range is `[-1,1]`.
///
///   ```python
///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")])
///   tf.math.sin(x) ==> [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Sin {
 public:
  Sin(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes hyperbolic sine of x element-wise.
///
///   Given an input tensor, this function computes hyperbolic sine of every
///   element in the tensor. Input range is `[-inf,inf]` and output range
///   is `[-inf,inf]`.
///
///   ```python
///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")])
///   tf.math.sinh(x) ==> [-inf -4.0515420e+03 -5.2109528e-01 1.1752012e+00 1.5094614e+00 3.6268604e+00 1.1013232e+04 inf]
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Sinh {
 public:
  Sinh(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Counts the number of occurrences of each value in an integer array.
///
/// Outputs a vector with length `size` and the same dtype as `weights`. If
/// `weights` are empty, then index `i` stores the number of times the value `i` is
/// counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
/// the value in `weights` at each index where the corresponding value in `arr` is
/// `i`.
///
/// Values in `arr` outside of the range [0, size) are ignored.
///
/// Args:
/// * scope: A Scope object
/// * indices: 2D int64 `Tensor`.
/// * values: 1D int `Tensor`.
/// * dense_shape: 1D int64 `Tensor`.
/// * size: non-negative int scalar `Tensor`.
/// * weights: is an int32, int64, float32, or float64 `Tensor` with the same
/// shape as `input`, or a length-0 `Tensor`, in which case it acts as all weights
/// equal to 1.
///
/// Optional attributes (see `Attrs`):
/// * binary_output: bool; Whether the kernel should count the appearance or number of occurrences.
///
/// Returns:
/// * `Output`: 1D `Tensor` with length equal to `size` or 2D `Tensor` with [batch_size, `size`].
/// The counts or summed weights for each value in the range [0, size).
class SparseBincount {
 public:
  /// Optional attribute setters for SparseBincount
  struct Attrs {
    /// bool; Whether the kernel should count the appearance or number of occurrences.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs BinaryOutput(bool x) {
      Attrs ret = *this;
      ret.binary_output_ = x;
      return ret;
    }

    bool binary_output_ = false;
  };
  SparseBincount(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
               ::tensorflow::Input values, ::tensorflow::Input dense_shape,
               ::tensorflow::Input size, ::tensorflow::Input weights);
  SparseBincount(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
               ::tensorflow::Input values, ::tensorflow::Input dense_shape,
               ::tensorflow::Input size, ::tensorflow::Input weights, const
               SparseBincount::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BinaryOutput(bool x) {
    return Attrs().BinaryOutput(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Multiply matrix "a" by matrix "b".
///
/// The inputs must be two-dimensional matrices and the inner dimension of "a" must
/// match the outer dimension of "b". Both "a" and "b" must be `Tensor`s not
/// `SparseTensor`s.  This op is optimized for the case where at least one of "a" or
/// "b" is sparse, in the sense that they have a large proportion of zero values.
/// The breakeven for using this versus a dense matrix multiply on one platform was
/// 30% zero values in the sparse matrix.
///
/// The gradient computation of this operation will only take advantage of sparsity
/// in the input gradient when that gradient comes from a Relu.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The product tensor.
class SparseMatMul {
 public:
  /// Optional attribute setters for SparseMatMul
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs TransposeA(bool x) {
      Attrs ret = *this;
      ret.transpose_a_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs TransposeB(bool x) {
      Attrs ret = *this;
      ret.transpose_b_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AIsSparse(bool x) {
      Attrs ret = *this;
      ret.a_is_sparse_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs BIsSparse(bool x) {
      Attrs ret = *this;
      ret.b_is_sparse_ = x;
      return ret;
    }

    bool transpose_a_ = false;
    bool transpose_b_ = false;
    bool a_is_sparse_ = false;
    bool b_is_sparse_ = false;
  };
  SparseMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
             ::tensorflow::Input b);
  SparseMatMul(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
             ::tensorflow::Input b, const SparseMatMul::Attrs& attrs);
  operator ::tensorflow::Output() const { return product; }
  operator ::tensorflow::Input() const { return product; }
  ::tensorflow::Node* node() const { return product.node(); }

  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs AIsSparse(bool x) {
    return Attrs().AIsSparse(x);
  }
  static Attrs BIsSparse(bool x) {
    return Attrs().BIsSparse(x);
  }

  Operation operation;
  ::tensorflow::Output product;
};

/// Computes the mean along sparse segments of a tensor.
///
/// See `tf.sparse.segment_sum` for usage examples.
///
/// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
/// dimension, selecting a subset of dimension 0, specified by `indices`.
///
/// Args:
/// * scope: A Scope object
/// * indices: A 1-D tensor. Has same rank as `segment_ids`.
/// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SparseSegmentMean {
 public:
  /// Optional attribute setters for SparseSegmentMean
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs SparseGradient(bool x) {
      Attrs ret = *this;
      ret.sparse_gradient_ = x;
      return ret;
    }

    bool sparse_gradient_ = false;
  };
  SparseSegmentMean(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                  ::tensorflow::Input indices, ::tensorflow::Input segment_ids);
  SparseSegmentMean(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                  ::tensorflow::Input indices, ::tensorflow::Input segment_ids,
                  const SparseSegmentMean::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SparseGradient(bool x) {
    return Attrs().SparseGradient(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients for SparseSegmentMean.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is output_dim0.
///
/// Args:
/// * scope: A Scope object
/// * grad: gradient propagated to the SparseSegmentMean op.
/// * indices: indices passed to the corresponding SparseSegmentMean op.
/// * segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
/// * output_dim0: dimension 0 of "data" passed to SparseSegmentMean op.
///
/// Returns:
/// * `Output`: The output tensor.
class SparseSegmentMeanGrad {
 public:
  SparseSegmentMeanGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      grad, ::tensorflow::Input indices, ::tensorflow::Input
                      segment_ids, ::tensorflow::Input output_dim0);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients for SparseSegmentMean.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is the number of unique indexes in "indices". Also returns vector
/// "sorted_unique_indices" containing the corresponding indexes from "indices".
///
/// Args:
/// * scope: A Scope object
/// * grad: gradient propagated to the SparseSegmentMean op.
/// * indices: indices passed to the corresponding SparseSegmentMean op.
/// * segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
/// * dense_output_dim0: dimension 0 of "data" passed to SparseSegmentMean op.
///
/// Returns:
/// * `Output` output
/// * `Output` sorted_unique_indices
class SparseSegmentMeanGradV2 {
 public:
  SparseSegmentMeanGradV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        grad, ::tensorflow::Input indices, ::tensorflow::Input
                        segment_ids, ::tensorflow::Input dense_output_dim0);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output sorted_unique_indices;
};

/// Computes the mean along sparse segments of a tensor.
///
/// Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
/// missing, the `output` tensor at that position will be zeroed.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Args:
/// * scope: A Scope object
/// * indices: A 1-D tensor. Has same rank as `segment_ids`.
/// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
/// * num_segments: Should equal the number of distinct segment IDs.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which has size
/// `num_segments`.
class SparseSegmentMeanWithNumSegments {
 public:
  /// Optional attribute setters for SparseSegmentMeanWithNumSegments
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs SparseGradient(bool x) {
      Attrs ret = *this;
      ret.sparse_gradient_ = x;
      return ret;
    }

    bool sparse_gradient_ = false;
  };
  SparseSegmentMeanWithNumSegments(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input data, ::tensorflow::Input
                                 indices, ::tensorflow::Input segment_ids,
                                 ::tensorflow::Input num_segments);
  SparseSegmentMeanWithNumSegments(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input data, ::tensorflow::Input
                                 indices, ::tensorflow::Input segment_ids,
                                 ::tensorflow::Input num_segments, const
                                 SparseSegmentMeanWithNumSegments::Attrs&
                                 attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SparseGradient(bool x) {
    return Attrs().SparseGradient(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
///
/// N is the size of the segment being reduced.
///
/// See `tf.sparse.segment_sum` for usage examples.
///
///
/// Args:
/// * scope: A Scope object
/// * indices: A 1-D tensor. Has same rank as `segment_ids`.
/// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SparseSegmentSqrtN {
 public:
  /// Optional attribute setters for SparseSegmentSqrtN
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs SparseGradient(bool x) {
      Attrs ret = *this;
      ret.sparse_gradient_ = x;
      return ret;
    }

    bool sparse_gradient_ = false;
  };
  SparseSegmentSqrtN(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                   ::tensorflow::Input indices, ::tensorflow::Input
                   segment_ids);
  SparseSegmentSqrtN(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                   ::tensorflow::Input indices, ::tensorflow::Input
                   segment_ids, const SparseSegmentSqrtN::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SparseGradient(bool x) {
    return Attrs().SparseGradient(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients for SparseSegmentSqrtN.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is output_dim0.
///
/// Args:
/// * scope: A Scope object
/// * grad: gradient propagated to the SparseSegmentSqrtN op.
/// * indices: indices passed to the corresponding SparseSegmentSqrtN op.
/// * segment_ids: segment_ids passed to the corresponding SparseSegmentSqrtN op.
/// * output_dim0: dimension 0 of "data" passed to SparseSegmentSqrtN op.
///
/// Returns:
/// * `Output`: The output tensor.
class SparseSegmentSqrtNGrad {
 public:
  SparseSegmentSqrtNGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       grad, ::tensorflow::Input indices, ::tensorflow::Input
                       segment_ids, ::tensorflow::Input output_dim0);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients for SparseSegmentSqrtN.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is the number of unique indexes in "indices". Also returns vector
/// "sorted_unique_indices" containing the corresponding indexes from "indices".
///
/// Args:
/// * scope: A Scope object
/// * grad: gradient propagated to the SparseSegmentSqrtN op.
/// * indices: indices passed to the corresponding SparseSegmentSqrtN op.
/// * segment_ids: segment_ids passed to the corresponding SparseSegmentSqrtN op.
/// * dense_output_dim0: dimension 0 of "data" passed to SparseSegmentSqrtN op.
///
/// Returns:
/// * `Output` output
/// * `Output` sorted_unique_indices
class SparseSegmentSqrtNGradV2 {
 public:
  SparseSegmentSqrtNGradV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         grad, ::tensorflow::Input indices, ::tensorflow::Input
                         segment_ids, ::tensorflow::Input dense_output_dim0);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output sorted_unique_indices;
};

/// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
///
/// N is the size of the segment being reduced.
///
/// Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
/// missing, the `output` tensor at that position will be zeroed.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Args:
/// * scope: A Scope object
/// * indices: A 1-D tensor. Has same rank as `segment_ids`.
/// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
/// * num_segments: Should equal the number of distinct segment IDs.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SparseSegmentSqrtNWithNumSegments {
 public:
  /// Optional attribute setters for SparseSegmentSqrtNWithNumSegments
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs SparseGradient(bool x) {
      Attrs ret = *this;
      ret.sparse_gradient_ = x;
      return ret;
    }

    bool sparse_gradient_ = false;
  };
  SparseSegmentSqrtNWithNumSegments(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input data, ::tensorflow::Input
                                  indices, ::tensorflow::Input segment_ids,
                                  ::tensorflow::Input num_segments);
  SparseSegmentSqrtNWithNumSegments(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input data, ::tensorflow::Input
                                  indices, ::tensorflow::Input segment_ids,
                                  ::tensorflow::Input num_segments, const
                                  SparseSegmentSqrtNWithNumSegments::Attrs&
                                  attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SparseGradient(bool x) {
    return Attrs().SparseGradient(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the sum along sparse segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
/// dimension, selecting a subset of dimension 0, specified by `indices`.
///
/// For example:
///
/// ```python
/// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
///
/// # Select two rows, one segment.
/// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
/// # => [[0 0 0 0]]
///
/// # Select two rows, two segment.
/// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
/// # => [[ 1  2  3  4]
/// #     [-1 -2 -3 -4]]
///
/// # Select all rows, two segments.
/// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
/// # => [[0 0 0 0]
/// #     [5 6 7 8]]
///
/// # Which is equivalent to:
/// tf.segment_sum(c, tf.constant([0, 0, 1]))
/// ```
///
/// Args:
/// * scope: A Scope object
/// * indices: A 1-D tensor. Has same rank as `segment_ids`.
/// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `k`, the number of segments.
class SparseSegmentSum {
 public:
  /// Optional attribute setters for SparseSegmentSum
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs SparseGradient(bool x) {
      Attrs ret = *this;
      ret.sparse_gradient_ = x;
      return ret;
    }

    bool sparse_gradient_ = false;
  };
  SparseSegmentSum(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                 ::tensorflow::Input indices, ::tensorflow::Input segment_ids);
  SparseSegmentSum(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                 ::tensorflow::Input indices, ::tensorflow::Input segment_ids,
                 const SparseSegmentSum::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SparseGradient(bool x) {
    return Attrs().SparseGradient(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients for SparseSegmentSum.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is output_dim0.
///
/// Args:
/// * scope: A Scope object
/// * grad: gradient propagated to the SparseSegmentSum op.
/// * indices: indices passed to the corresponding SparseSegmentSum op.
/// * segment_ids: segment_ids passed to the corresponding SparseSegmentSum op.
/// * output_dim0: dimension 0 of "data" passed to SparseSegmentSum op.
///
/// Returns:
/// * `Output`: The output tensor.
class SparseSegmentSumGrad {
 public:
  SparseSegmentSumGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     grad, ::tensorflow::Input indices, ::tensorflow::Input
                     segment_ids, ::tensorflow::Input output_dim0);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients for SparseSegmentSum.
///
/// Returns tensor "output" with same shape as grad, except for dimension 0 whose
/// value is the number of unique indexes in "indices". Also returns vector
/// "sorted_unique_indices" containing the corresponding indexes from "indices".
///
/// Args:
/// * scope: A Scope object
/// * grad: gradient propagated to the SparseSegmentSum op.
/// * indices: indices passed to the corresponding SparseSegmentSum op.
/// * segment_ids: segment_ids passed to the corresponding SparseSegmentSum op.
/// * dense_output_dim0: dimension 0 of "data" passed to SparseSegmentSum op.
///
/// Returns:
/// * `Output` output
/// * `Output` sorted_unique_indices
class SparseSegmentSumGradV2 {
 public:
  SparseSegmentSumGradV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       grad, ::tensorflow::Input indices, ::tensorflow::Input
                       segment_ids, ::tensorflow::Input dense_output_dim0);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output sorted_unique_indices;
};

/// Computes the sum along sparse segments of a tensor.
///
/// Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
/// missing, the `output` tensor at that position will be zeroed.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/sparse#Segmentation)
/// for an explanation of segments.
///
/// For example:
///
/// ```python
/// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
///
/// tf.sparse_segment_sum_with_num_segments(
///     c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
/// # => [[0 0 0 0]
/// #     [0 0 0 0]
/// #     [0 0 0 0]]
///
/// tf.sparse_segment_sum_with_num_segments(c,
///                                         tf.constant([0, 1]),
///                                         tf.constant([0, 2],
///                                         num_segments=4))
/// # => [[ 1  2  3  4]
/// #     [ 0  0  0  0]
/// #     [-1 -2 -3 -4]
/// #     [ 0  0  0  0]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * indices: A 1-D tensor. Has same rank as `segment_ids`.
/// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
/// * num_segments: Should equal the number of distinct segment IDs.
///
/// Returns:
/// * `Output`: Has same shape as data, except for dimension 0 which
/// has size `num_segments`.
class SparseSegmentSumWithNumSegments {
 public:
  /// Optional attribute setters for SparseSegmentSumWithNumSegments
  struct Attrs {
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs SparseGradient(bool x) {
      Attrs ret = *this;
      ret.sparse_gradient_ = x;
      return ret;
    }

    bool sparse_gradient_ = false;
  };
  SparseSegmentSumWithNumSegments(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input data, ::tensorflow::Input
                                indices, ::tensorflow::Input segment_ids,
                                ::tensorflow::Input num_segments);
  SparseSegmentSumWithNumSegments(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input data, ::tensorflow::Input
                                indices, ::tensorflow::Input segment_ids,
                                ::tensorflow::Input num_segments, const
                                SparseSegmentSumWithNumSegments::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SparseGradient(bool x) {
    return Attrs().SparseGradient(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes square root of x element-wise.
///
/// I.e., \\(y = \sqrt{x} = x^{1/2}\\).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Sqrt {
 public:
  Sqrt(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes square of x element-wise.
///
/// I.e., \\(y = x * x = x^2\\).
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Square {
 public:
  Square(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns conj(x - y)(x - y) element-wise.
///
/// *NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class SquaredDifference {
 public:
  SquaredDifference(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                  ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns x - y element-wise.
///
/// *NOTE*: `Subtract` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
///
/// Aliases:
/// * Sub
class Subtract {
 public:
  Subtract(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};
typedef Subtract Sub;

/// Computes the sum of elements across dimensions of a tensor.
///
/// Reduces `input` along the dimensions given in `axis`. Unless
/// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
/// `axis`. If `keep_dims` is true, the reduced dimensions are
/// retained with length 1.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor to reduce.
/// * axis: The dimensions to reduce. Must be in the range
/// `[-rank(input), rank(input))`.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If true, retain reduced dimensions with length 1.
///
/// Returns:
/// * `Output`: The reduced tensor.
///
/// Aliases:
/// * ReduceSum
class Sum {
 public:
  /// Optional attribute setters for Sum
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
  Sum(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis);
  Sum(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input axis, const Sum::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};
typedef Sum ReduceSum;

/// Computes tan of x element-wise.
///
///   Given an input tensor, this function computes tangent of every
///   element in the tensor. Input range is `(-inf, inf)` and
///   output range is `(-inf, inf)`. If input lies outside the boundary, `nan`
///   is returned.
///
///   ```python
///   x = tf.constant([-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")])
///   tf.math.tan(x) ==> [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
///   ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Tan {
 public:
  Tan(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes hyperbolic tangent of `x` element-wise.
///
///   Given an input tensor, this function computes hyperbolic tangent of every
///   element in the tensor. Input range is `[-inf, inf]` and
///   output range is `[-1,1]`.
///
///   >>> x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
///   >>> tf.math.tanh(x)
///   <tf.Tensor: shape=(8,), dtype=float32, numpy=
///   array([-1.0, -0.99990916, -0.46211717,  0.7615942 ,  0.8336547 ,
///           0.9640276 ,  0.9950547 ,  1.0], dtype=float32)>
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Tanh {
 public:
  Tanh(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Returns x / y element-wise, rounded towards zero.
///
/// Truncation designates that negative numbers will round fractional quantities
/// toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
/// than Python semantics. See `FloorDiv` for a division function that matches
/// Python Semantics.
///
/// *NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class TruncateDiv {
 public:
  TruncateDiv(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
            ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns element-wise remainder of division. This emulates C semantics in that
///
/// the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
/// y + truncate_mod(x, y) = x`.
///
/// *NOTE*: `TruncateMod` supports broadcasting. More about broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class TruncateMod {
 public:
  TruncateMod(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
            ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Computes the maximum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// This operator is similar to `tf.math.unsorted_segment_sum`,
/// Instead of computing the sum over segments, it computes the maximum such that:
///
/// \\(output_i = \max_{j...} data[j...]\\) where max is over tuples `j...` such
/// that `segment_ids[j...] == i`.
///
/// If the maximum is empty for a given segment ID `i`, it outputs the smallest
/// possible value for the specific numeric type,
/// `output[i] = numeric_limits<T>::lowest()`.
///
/// If the given segment ID `i` is negative, then the corresponding value is
/// dropped, and will not be included in the result.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be less than
/// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
/// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
/// result in safe but unspecified behavior, which may include ignoring
/// out-of-bound indices or outputting a tensor with a 0 stored in the first
/// dimension of its shape if `num_segments` is 0.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentMax.png" alt>
/// </div>
///
/// For example:
///
/// >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
/// >>> tf.math.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
/// array([[4, 3, 3, 4],
///        [5,  6, 7, 8]], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A tensor whose shape is a prefix of `data.shape`.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be in range on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimension which has size
/// `num_segments`.
class UnsortedSegmentMax {
 public:
  UnsortedSegmentMax(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                   ::tensorflow::Input segment_ids, ::tensorflow::Input
                   num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the minimum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// This operator is similar to `tf.math.unsorted_segment_sum`,
/// Instead of computing the sum over segments, it computes the minimum such that:
///
/// \\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
/// that `segment_ids[j...] == i`.
///
/// If the minimum is empty for a given segment ID `i`, it outputs the largest
/// possible value for the specific numeric type,
/// `output[i] = numeric_limits<T>::max()`.
///
/// For example:
///
/// >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
/// >>> tf.math.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
/// array([[1, 2, 2, 1],
///        [5, 6, 7, 8]], dtype=int32)
///
/// If the given segment ID `i` is negative, then the corresponding value is
/// dropped, and will not be included in the result.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be less than
/// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
/// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
/// result in safe but unspecified behavior, which may include ignoring
/// out-of-bound indices or outputting a tensor with a 0 stored in the first
/// dimension of its shape if `num_segments` is 0.
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A tensor whose shape is a prefix of `data.shape`.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be in range on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimension which has size
/// `num_segments`.
class UnsortedSegmentMin {
 public:
  UnsortedSegmentMin(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                   ::tensorflow::Input segment_ids, ::tensorflow::Input
                   num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the product along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// This operator is similar to `tf.math.unsorted_segment_sum`,
/// Instead of computing the sum over segments, it computes the product of all
/// entries belonging to a segment such that:
///
/// \\(output_i = \prod_{j...} data[j...]\\) where the product is over tuples
/// `j...` such that `segment_ids[j...] == i`.
///
/// For example:
///
/// >>> c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
/// >>> tf.math.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2).numpy()
/// array([[4, 6, 6, 4],
///        [5, 6, 7, 8]], dtype=int32)
///
/// If there is no entry for a given segment ID `i`, it outputs 1.
///
/// If the given segment ID `i` is negative, then the corresponding value is
/// dropped, and will not be included in the result.
/// Caution: On CPU, values in `segment_ids` are always validated to be less than
/// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
/// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
/// result in safe but unspecified behavior, which may include ignoring
/// out-of-bound indices or outputting a tensor with a 0 stored in the first
/// dimension of its shape if `num_segments` is 0.
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A tensor whose shape is a prefix of `data.shape`.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be in range on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimension which has size
/// `num_segments`.
class UnsortedSegmentProd {
 public:
  UnsortedSegmentProd(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                    ::tensorflow::Input segment_ids, ::tensorflow::Input
                    num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the sum along segments of a tensor.
///
/// Read
/// [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
/// for an explanation of segments.
///
/// Computes a tensor such that
/// \\(output[i] = \sum_{j...} data[j...]\\) where the sum is over tuples `j...` such
/// that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
/// need not be sorted and need not cover all values in the full
/// range of valid values.
///
/// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
/// If the given segment ID `i` is negative, the value is dropped and will not be
/// added to the sum of the segment.
///
/// `num_segments` should equal the number of distinct segment IDs.
///
/// Caution: On CPU, values in `segment_ids` are always validated to be less than
/// `num_segments`, and an error is thrown for out-of-bound indices. On GPU, this
/// does not throw an error for out-of-bound indices. On Gpu, out-of-bound indices
/// result in safe but unspecified behavior, which may include ignoring
/// out-of-bound indices or outputting a tensor with a 0 stored in the first
/// dimension of its shape if `num_segments` is 0.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/UnsortedSegmentSum.png" alt>
/// </div>
///
/// >>> c = [[1,2,3,4], [5,6,7,8], [4,3,2,1]]
/// >>> tf.math.unsorted_segment_sum(c, [0, 1, 0], num_segments=2).numpy()
/// array([[5, 5, 5, 5],
///        [5, 6, 7, 8]], dtype=int32)
///
///
///
/// Args:
/// * scope: A Scope object
/// * segment_ids: A tensor whose shape is a prefix of `data.shape`.
/// The values must be less than `num_segments`.
///
/// Caution: The values are always validated to be in range on CPU, never validated
/// on GPU.
///
/// Returns:
/// * `Output`: Has same shape as data, except for the first `segment_ids.rank`
/// dimensions, which are replaced with a single dimension which has size
/// `num_segments`.
class UnsortedSegmentSum {
 public:
  UnsortedSegmentSum(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                   ::tensorflow::Input segment_ids, ::tensorflow::Input
                   num_segments);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns 0 if x == 0, and x / y otherwise, elementwise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Xdivy {
 public:
  Xdivy(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
      ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns 0 if x == 0, and x * log1p(y) otherwise, elementwise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Xlog1py {
 public:
  Xlog1py(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
        ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Returns 0 if x == 0, and x * log(y) otherwise, elementwise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Xlogy {
 public:
  Xlogy(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
      ::tensorflow::Input y);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// Compute the Hurwitz zeta function \\(\zeta(x, q)\\).
///
/// The Hurwitz zeta function is defined as:
///
///
/// \\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The z tensor.
class Zeta {
 public:
  Zeta(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
     ::tensorflow::Input q);
  operator ::tensorflow::Output() const { return z; }
  operator ::tensorflow::Input() const { return z; }
  ::tensorflow::Node* node() const { return z.node(); }

  Operation operation;
  ::tensorflow::Output z;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_MATH_OPS_H_
