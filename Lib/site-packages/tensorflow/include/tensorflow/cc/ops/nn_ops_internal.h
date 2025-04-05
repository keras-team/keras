// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_NN_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_NN_OPS_INTERNAL_H_

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

/// @defgroup nn_ops_internal Nn Ops Internal
/// @{

/// Computes gradients of the average pooling function.
///
/// Args:
/// * scope: A Scope object
/// * orig_input_shape: 1-D.  Shape of the original input to `avg_pool`.
/// * grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
/// the output of `avg_pool`.
/// * ksize: The size of the sliding window for each dimension of the input.
/// * strides: The stride of the sliding window for each dimension of the input.
/// * padding: The type of padding algorithm to use.
///
/// Optional attributes (see `Attrs`):
/// * data_format: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
///
/// Returns:
/// * `Output`: 4-D.  Gradients w.r.t. the input of `avg_pool`.
class AvgPoolGrad {
 public:
  /// Optional attribute setters for AvgPoolGrad
  struct Attrs {
    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    ///
    /// Defaults to "NHWC"
    TF_MUST_USE_RESULT Attrs DataFormat(StringPiece x) {
      Attrs ret = *this;
      ret.data_format_ = x;
      return ret;
    }

    StringPiece data_format_ = "NHWC";
  };
  AvgPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
            orig_input_shape, ::tensorflow::Input grad, const
            gtl::ArraySlice<int>& ksize, const gtl::ArraySlice<int>& strides,
            StringPiece padding);
  AvgPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
            orig_input_shape, ::tensorflow::Input grad, const
            gtl::ArraySlice<int>& ksize, const gtl::ArraySlice<int>& strides,
            StringPiece padding, const AvgPoolGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs DataFormat(StringPiece x) {
    return Attrs().DataFormat(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients for the exponential linear (Elu) operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: The backpropagated gradients to the corresponding Elu operation.
/// * outputs: The outputs of the corresponding Elu operation.
///
/// Returns:
/// * `Output`: The gradients: `gradients * (outputs + 1)` if outputs < 0,
/// `gradients` otherwise.
class EluGrad {
 public:
  EluGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
        ::tensorflow::Input outputs);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  Operation operation;
  ::tensorflow::Output backprops;
};

/// Computes gradient of the FractionalAvgPool function.
///
/// Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
/// FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
/// out_backprop to those indices that form the same pooling cell. Therefore, we
/// just need to know the shape of original input tensor, instead of the whole
/// tensor.
///
/// Args:
/// * scope: A Scope object
/// * orig_input_tensor_shape: Original input tensor shape for `fractional_avg_pool`
/// * out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
/// w.r.t. the output of `fractional_avg_pool`.
/// * row_pooling_sequence: row pooling sequence, form pooling region with
/// col_pooling_sequence.
/// * col_pooling_sequence: column pooling sequence, form pooling region with
/// row_pooling sequence.
///
/// Optional attributes (see `Attrs`):
/// * overlapping: When set to True, it means when pooling, the values at the boundary
/// of adjacent pooling cells are used by both cells. For example:
///
/// `index  0  1  2  3  4`
///
/// `value  20 5  16 3  7`
///
/// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
/// The result would be [41/3, 26/3] for fractional avg pooling.
///
/// Returns:
/// * `Output`: 4-D.  Gradients w.r.t. the input of `fractional_avg_pool`.
class FractionalAvgPoolGrad {
 public:
  /// Optional attribute setters for FractionalAvgPoolGrad
  struct Attrs {
    /// When set to True, it means when pooling, the values at the boundary
    /// of adjacent pooling cells are used by both cells. For example:
    ///
    /// `index  0  1  2  3  4`
    ///
    /// `value  20 5  16 3  7`
    ///
    /// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
    /// The result would be [41/3, 26/3] for fractional avg pooling.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Overlapping(bool x) {
      Attrs ret = *this;
      ret.overlapping_ = x;
      return ret;
    }

    bool overlapping_ = false;
  };
  FractionalAvgPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      orig_input_tensor_shape, ::tensorflow::Input
                      out_backprop, ::tensorflow::Input row_pooling_sequence,
                      ::tensorflow::Input col_pooling_sequence);
  FractionalAvgPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      orig_input_tensor_shape, ::tensorflow::Input
                      out_backprop, ::tensorflow::Input row_pooling_sequence,
                      ::tensorflow::Input col_pooling_sequence, const
                      FractionalAvgPoolGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Overlapping(bool x) {
    return Attrs().Overlapping(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradient of the FractionalMaxPool function.
///
/// Args:
/// * scope: A Scope object
/// * orig_input: Original input for `fractional_max_pool`
/// * orig_output: Original output for `fractional_max_pool`
/// * out_backprop: 4-D with shape `[batch, height, width, channels]`.  Gradients
/// w.r.t. the output of `fractional_max_pool`.
/// * row_pooling_sequence: row pooling sequence, form pooling region with
/// col_pooling_sequence.
/// * col_pooling_sequence: column pooling sequence, form pooling region with
/// row_pooling sequence.
///
/// Optional attributes (see `Attrs`):
/// * overlapping: When set to True, it means when pooling, the values at the boundary
/// of adjacent pooling cells are used by both cells. For example:
///
/// `index  0  1  2  3  4`
///
/// `value  20 5  16 3  7`
///
/// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
/// The result would be [20, 16] for fractional max pooling.
///
/// Returns:
/// * `Output`: 4-D.  Gradients w.r.t. the input of `fractional_max_pool`.
class FractionalMaxPoolGrad {
 public:
  /// Optional attribute setters for FractionalMaxPoolGrad
  struct Attrs {
    /// When set to True, it means when pooling, the values at the boundary
    /// of adjacent pooling cells are used by both cells. For example:
    ///
    /// `index  0  1  2  3  4`
    ///
    /// `value  20 5  16 3  7`
    ///
    /// If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
    /// The result would be [20, 16] for fractional max pooling.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Overlapping(bool x) {
      Attrs ret = *this;
      ret.overlapping_ = x;
      return ret;
    }

    bool overlapping_ = false;
  };
  FractionalMaxPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      orig_input, ::tensorflow::Input orig_output,
                      ::tensorflow::Input out_backprop, ::tensorflow::Input
                      row_pooling_sequence, ::tensorflow::Input
                      col_pooling_sequence);
  FractionalMaxPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      orig_input, ::tensorflow::Input orig_output,
                      ::tensorflow::Input out_backprop, ::tensorflow::Input
                      row_pooling_sequence, ::tensorflow::Input
                      col_pooling_sequence, const FractionalMaxPoolGrad::Attrs&
                      attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Overlapping(bool x) {
    return Attrs().Overlapping(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Solves a batch of isotonic regression problems.
///
/// Args:
/// * scope: A Scope object
/// * input: A (batch_size, dim)-tensor holding a batch of inputs.
///
/// Optional attributes (see `Attrs`):
/// * output_dtype: Dtype of output.
///
/// Returns:
/// * `Output` output: A (batch_size, dim)-tensor holding the per-batch element solutions.
/// * `Output` segments: An int32 (batch_size, dim)-tensor with the segments.
class IsotonicRegression {
 public:
  /// Optional attribute setters for IsotonicRegression
  struct Attrs {
    /// Dtype of output.
    ///
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs OutputDtype(DataType x) {
      Attrs ret = *this;
      ret.output_dtype_ = x;
      return ret;
    }

    DataType output_dtype_ = DT_FLOAT;
  };
  IsotonicRegression(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  IsotonicRegression(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                   const IsotonicRegression::Attrs& attrs);

  static Attrs OutputDtype(DataType x) {
    return Attrs().OutputDtype(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output segments;
};

/// Gradients for Local Response Normalization.
///
/// Args:
/// * scope: A Scope object
/// * input_grads: 4-D with shape `[batch, height, width, channels]`.
/// * input_image: 4-D with shape `[batch, height, width, channels]`.
/// * output_image: 4-D with shape `[batch, height, width, channels]`.
///
/// Optional attributes (see `Attrs`):
/// * depth_radius: A depth radius.
/// * bias: An offset (usually > 0 to avoid dividing by 0).
/// * alpha: A scale factor, usually positive.
/// * beta: An exponent.
///
/// Returns:
/// * `Output`: The gradients for LRN.
class LRNGrad {
 public:
  /// Optional attribute setters for LRNGrad
  struct Attrs {
    /// A depth radius.
    ///
    /// Defaults to 5
    TF_MUST_USE_RESULT Attrs DepthRadius(int64 x) {
      Attrs ret = *this;
      ret.depth_radius_ = x;
      return ret;
    }

    /// An offset (usually > 0 to avoid dividing by 0).
    ///
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs Bias(float x) {
      Attrs ret = *this;
      ret.bias_ = x;
      return ret;
    }

    /// A scale factor, usually positive.
    ///
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs Alpha(float x) {
      Attrs ret = *this;
      ret.alpha_ = x;
      return ret;
    }

    /// An exponent.
    ///
    /// Defaults to 0.5
    TF_MUST_USE_RESULT Attrs Beta(float x) {
      Attrs ret = *this;
      ret.beta_ = x;
      return ret;
    }

    int64 depth_radius_ = 5;
    float bias_ = 1.0f;
    float alpha_ = 1.0f;
    float beta_ = 0.5f;
  };
  LRNGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input input_grads,
        ::tensorflow::Input input_image, ::tensorflow::Input output_image);
  LRNGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input input_grads,
        ::tensorflow::Input input_image, ::tensorflow::Input output_image,
        const LRNGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs DepthRadius(int64 x) {
    return Attrs().DepthRadius(x);
  }
  static Attrs Bias(float x) {
    return Attrs().Bias(x);
  }
  static Attrs Alpha(float x) {
    return Attrs().Alpha(x);
  }
  static Attrs Beta(float x) {
    return Attrs().Beta(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes rectified linear: `max(features, features * alpha)`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The activations tensor.
class LeakyRelu {
 public:
  /// Optional attribute setters for LeakyRelu
  struct Attrs {
    /// Defaults to 0.2
    TF_MUST_USE_RESULT Attrs Alpha(float x) {
      Attrs ret = *this;
      ret.alpha_ = x;
      return ret;
    }

    float alpha_ = 0.2f;
  };
  LeakyRelu(const ::tensorflow::Scope& scope, ::tensorflow::Input features);
  LeakyRelu(const ::tensorflow::Scope& scope, ::tensorflow::Input features, const
          LeakyRelu::Attrs& attrs);
  operator ::tensorflow::Output() const { return activations; }
  operator ::tensorflow::Input() const { return activations; }
  ::tensorflow::Node* node() const { return activations.node(); }

  static Attrs Alpha(float x) {
    return Attrs().Alpha(x);
  }

  Operation operation;
  ::tensorflow::Output activations;
};

/// Computes rectified linear gradients for a LeakyRelu operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: The backpropagated gradients to the corresponding LeakyRelu operation.
/// * features: The features passed as input to the corresponding LeakyRelu operation,
/// OR the outputs of that operation (both work equivalently).
///
/// Returns:
/// * `Output`: `gradients * (features > 0) + alpha * gradients * (features <= 0)`.
class LeakyReluGrad {
 public:
  /// Optional attribute setters for LeakyReluGrad
  struct Attrs {
    /// Defaults to 0.2
    TF_MUST_USE_RESULT Attrs Alpha(float x) {
      Attrs ret = *this;
      ret.alpha_ = x;
      return ret;
    }

    float alpha_ = 0.2f;
  };
  LeakyReluGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
              ::tensorflow::Input features);
  LeakyReluGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
              ::tensorflow::Input features, const LeakyReluGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  static Attrs Alpha(float x) {
    return Attrs().Alpha(x);
  }

  Operation operation;
  ::tensorflow::Output backprops;
};

/// Computes gradients of the maxpooling function.
///
/// Args:
/// * scope: A Scope object
/// * orig_input: The original input tensor.
/// * orig_output: The original output tensor.
/// * grad: 4-D.  Gradients w.r.t. the output of `max_pool`.
/// * ksize: The size of the window for each dimension of the input tensor.
/// * strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// * padding: The type of padding algorithm to use.
///
/// Optional attributes (see `Attrs`):
/// * data_format: Specify the data format of the input and output data. With the
/// default format "NHWC", the data is stored in the order of:
///     [batch, in_height, in_width, in_channels].
/// Alternatively, the format could be "NCHW", the data storage order of:
///     [batch, in_channels, in_height, in_width].
///
/// Returns:
/// * `Output`: Gradients w.r.t. the input to `max_pool`.
class MaxPoolGrad {
 public:
  /// Optional attribute setters for MaxPoolGrad
  struct Attrs {
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ExplicitPaddings(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.explicit_paddings_ = x;
      return ret;
    }

    /// Specify the data format of the input and output data. With the
    /// default format "NHWC", the data is stored in the order of:
    ///     [batch, in_height, in_width, in_channels].
    /// Alternatively, the format could be "NCHW", the data storage order of:
    ///     [batch, in_channels, in_height, in_width].
    ///
    /// Defaults to "NHWC"
    TF_MUST_USE_RESULT Attrs DataFormat(StringPiece x) {
      Attrs ret = *this;
      ret.data_format_ = x;
      return ret;
    }

    gtl::ArraySlice<int> explicit_paddings_ = {};
    StringPiece data_format_ = "NHWC";
  };
  MaxPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input orig_input,
            ::tensorflow::Input orig_output, ::tensorflow::Input grad, const
            gtl::ArraySlice<int>& ksize, const gtl::ArraySlice<int>& strides,
            StringPiece padding);
  MaxPoolGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input orig_input,
            ::tensorflow::Input orig_output, ::tensorflow::Input grad, const
            gtl::ArraySlice<int>& ksize, const gtl::ArraySlice<int>& strides,
            StringPiece padding, const MaxPoolGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs ExplicitPaddings(const gtl::ArraySlice<int>& x) {
    return Attrs().ExplicitPaddings(x);
  }
  static Attrs DataFormat(StringPiece x) {
    return Attrs().DataFormat(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes gradients of the maxpooling function.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input.
/// * grad: 4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
/// output of `max_pool`.
/// * argmax: The indices of the maximum values chosen for each output of `max_pool`.
/// * ksize: The size of the window for each dimension of the input tensor.
/// * strides: The stride of the sliding window for each dimension of the
/// input tensor.
/// * padding: The type of padding algorithm to use.
///
/// Optional attributes (see `Attrs`):
/// * include_batch_in_index: Whether to include batch dimension in flattened index of `argmax`.
///
/// Returns:
/// * `Output`: Gradients w.r.t. the input of `max_pool`.
class MaxPoolGradWithArgmax {
 public:
  /// Optional attribute setters for MaxPoolGradWithArgmax
  struct Attrs {
    /// Whether to include batch dimension in flattened index of `argmax`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs IncludeBatchInIndex(bool x) {
      Attrs ret = *this;
      ret.include_batch_in_index_ = x;
      return ret;
    }

    bool include_batch_in_index_ = false;
  };
  MaxPoolGradWithArgmax(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input, ::tensorflow::Input grad, ::tensorflow::Input
                      argmax, const gtl::ArraySlice<int>& ksize, const
                      gtl::ArraySlice<int>& strides, StringPiece padding);
  MaxPoolGradWithArgmax(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input, ::tensorflow::Input grad, ::tensorflow::Input
                      argmax, const gtl::ArraySlice<int>& ksize, const
                      gtl::ArraySlice<int>& strides, StringPiece padding, const
                      MaxPoolGradWithArgmax::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs IncludeBatchInIndex(bool x) {
    return Attrs().IncludeBatchInIndex(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DAndRelu {
 public:
  /// Optional attribute setters for QuantizedConv2DAndRelu
  struct Attrs {
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DAndRelu(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       input, ::tensorflow::Input filter, ::tensorflow::Input
                       min_input, ::tensorflow::Input max_input,
                       ::tensorflow::Input min_filter, ::tensorflow::Input
                       max_filter, const gtl::ArraySlice<int>& strides,
                       StringPiece padding);
  QuantizedConv2DAndRelu(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       input, ::tensorflow::Input filter, ::tensorflow::Input
                       min_input, ::tensorflow::Input max_input,
                       ::tensorflow::Input min_filter, ::tensorflow::Input
                       max_filter, const gtl::ArraySlice<int>& strides,
                       StringPiece padding, const
                       QuantizedConv2DAndRelu::Attrs& attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DAndReluAndRequantize {
 public:
  /// Optional attribute setters for QuantizedConv2DAndReluAndRequantize
  struct Attrs {
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QUINT8;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DAndReluAndRequantize(const ::tensorflow::Scope& scope,
                                    ::tensorflow::Input input,
                                    ::tensorflow::Input filter,
                                    ::tensorflow::Input min_input,
                                    ::tensorflow::Input max_input,
                                    ::tensorflow::Input min_filter,
                                    ::tensorflow::Input max_filter,
                                    ::tensorflow::Input min_freezed_output,
                                    ::tensorflow::Input max_freezed_output,
                                    const gtl::ArraySlice<int>& strides,
                                    StringPiece padding);
  QuantizedConv2DAndReluAndRequantize(const ::tensorflow::Scope& scope,
                                    ::tensorflow::Input input,
                                    ::tensorflow::Input filter,
                                    ::tensorflow::Input min_input,
                                    ::tensorflow::Input max_input,
                                    ::tensorflow::Input min_filter,
                                    ::tensorflow::Input max_filter,
                                    ::tensorflow::Input min_freezed_output,
                                    ::tensorflow::Input max_freezed_output,
                                    const gtl::ArraySlice<int>& strides,
                                    StringPiece padding, const
                                    QuantizedConv2DAndReluAndRequantize::Attrs&
                                    attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DAndRequantize {
 public:
  /// Optional attribute setters for QuantizedConv2DAndRequantize
  struct Attrs {
    /// Defaults to DT_QINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT8;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DAndRequantize(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input, ::tensorflow::Input
                             filter, ::tensorflow::Input min_input,
                             ::tensorflow::Input max_input, ::tensorflow::Input
                             min_filter, ::tensorflow::Input max_filter,
                             ::tensorflow::Input min_freezed_output,
                             ::tensorflow::Input max_freezed_output, const
                             gtl::ArraySlice<int>& strides, StringPiece
                             padding);
  QuantizedConv2DAndRequantize(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input, ::tensorflow::Input
                             filter, ::tensorflow::Input min_input,
                             ::tensorflow::Input max_input, ::tensorflow::Input
                             min_filter, ::tensorflow::Input max_filter,
                             ::tensorflow::Input min_freezed_output,
                             ::tensorflow::Input max_freezed_output, const
                             gtl::ArraySlice<int>& strides, StringPiece
                             padding, const
                             QuantizedConv2DAndRequantize::Attrs& attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// Computes QuantizedConv2D per channel.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input tensor.
/// * filter: The original filter tensor.
/// * min_input: The minimum value of the input tensor
/// * max_input: The maximum value of the input tensor.
/// * min_filter: The minimum value of the filter tensor.
/// * max_filter: The maximum value of the filter tensor.
/// * strides: list of stride values.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The quantized type of output tensor that needs to be converted.
/// * dilations: list of dilation values.
///
/// Returns:
/// * `Output` output: The output tensor.
/// * `Output` min_output: The minimum value of the final output tensor.
/// * `Output` max_output: The maximum value of the final output tensor.
class QuantizedConv2DPerChannel {
 public:
  /// Optional attribute setters for QuantizedConv2DPerChannel
  struct Attrs {
    /// The quantized type of output tensor that needs to be converted.
    ///
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// list of dilation values.
    ///
    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DPerChannel(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input, ::tensorflow::Input filter,
                          ::tensorflow::Input min_input, ::tensorflow::Input
                          max_input, ::tensorflow::Input min_filter,
                          ::tensorflow::Input max_filter, const
                          gtl::ArraySlice<int>& strides, StringPiece padding);
  QuantizedConv2DPerChannel(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          input, ::tensorflow::Input filter,
                          ::tensorflow::Input min_input, ::tensorflow::Input
                          max_input, ::tensorflow::Input min_filter,
                          ::tensorflow::Input max_filter, const
                          gtl::ArraySlice<int>& strides, StringPiece padding,
                          const QuantizedConv2DPerChannel::Attrs& attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DWithBias {
 public:
  /// Optional attribute setters for QuantizedConv2DWithBias
  struct Attrs {
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DWithBias(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input filter, ::tensorflow::Input
                        bias, ::tensorflow::Input min_input,
                        ::tensorflow::Input max_input, ::tensorflow::Input
                        min_filter, ::tensorflow::Input max_filter, const
                        gtl::ArraySlice<int>& strides, StringPiece padding);
  QuantizedConv2DWithBias(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input filter, ::tensorflow::Input
                        bias, ::tensorflow::Input min_input,
                        ::tensorflow::Input max_input, ::tensorflow::Input
                        min_filter, ::tensorflow::Input max_filter, const
                        gtl::ArraySlice<int>& strides, StringPiece padding,
                        const QuantizedConv2DWithBias::Attrs& attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DWithBiasAndRelu {
 public:
  /// Optional attribute setters for QuantizedConv2DWithBiasAndRelu
  struct Attrs {
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DWithBiasAndRelu(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input, ::tensorflow::Input
                               filter, ::tensorflow::Input bias,
                               ::tensorflow::Input min_input,
                               ::tensorflow::Input max_input,
                               ::tensorflow::Input min_filter,
                               ::tensorflow::Input max_filter, const
                               gtl::ArraySlice<int>& strides, StringPiece
                               padding);
  QuantizedConv2DWithBiasAndRelu(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input, ::tensorflow::Input
                               filter, ::tensorflow::Input bias,
                               ::tensorflow::Input min_input,
                               ::tensorflow::Input max_input,
                               ::tensorflow::Input min_filter,
                               ::tensorflow::Input max_filter, const
                               gtl::ArraySlice<int>& strides, StringPiece
                               padding, const
                               QuantizedConv2DWithBiasAndRelu::Attrs& attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DWithBiasAndReluAndRequantize {
 public:
  /// Optional attribute setters for QuantizedConv2DWithBiasAndReluAndRequantize
  struct Attrs {
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QUINT8;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DWithBiasAndReluAndRequantize(const ::tensorflow::Scope& scope,
                                            ::tensorflow::Input input,
                                            ::tensorflow::Input filter,
                                            ::tensorflow::Input bias,
                                            ::tensorflow::Input min_input,
                                            ::tensorflow::Input max_input,
                                            ::tensorflow::Input min_filter,
                                            ::tensorflow::Input max_filter,
                                            ::tensorflow::Input
                                            min_freezed_output,
                                            ::tensorflow::Input
                                            max_freezed_output, const
                                            gtl::ArraySlice<int>& strides,
                                            StringPiece padding);
  QuantizedConv2DWithBiasAndReluAndRequantize(const ::tensorflow::Scope& scope,
                                            ::tensorflow::Input input,
                                            ::tensorflow::Input filter,
                                            ::tensorflow::Input bias,
                                            ::tensorflow::Input min_input,
                                            ::tensorflow::Input max_input,
                                            ::tensorflow::Input min_filter,
                                            ::tensorflow::Input max_filter,
                                            ::tensorflow::Input
                                            min_freezed_output,
                                            ::tensorflow::Input
                                            max_freezed_output, const
                                            gtl::ArraySlice<int>& strides,
                                            StringPiece padding, const
                                            QuantizedConv2DWithBiasAndReluAndRequantize::Attrs&
                                            attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DWithBiasAndRequantize {
 public:
  /// Optional attribute setters for QuantizedConv2DWithBiasAndRequantize
  struct Attrs {
    /// Defaults to DT_QINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT8;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DWithBiasAndRequantize(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input input,
                                     ::tensorflow::Input filter,
                                     ::tensorflow::Input bias,
                                     ::tensorflow::Input min_input,
                                     ::tensorflow::Input max_input,
                                     ::tensorflow::Input min_filter,
                                     ::tensorflow::Input max_filter,
                                     ::tensorflow::Input min_freezed_output,
                                     ::tensorflow::Input max_freezed_output,
                                     const gtl::ArraySlice<int>& strides,
                                     StringPiece padding);
  QuantizedConv2DWithBiasAndRequantize(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input input,
                                     ::tensorflow::Input filter,
                                     ::tensorflow::Input bias,
                                     ::tensorflow::Input min_input,
                                     ::tensorflow::Input max_input,
                                     ::tensorflow::Input min_filter,
                                     ::tensorflow::Input max_filter,
                                     ::tensorflow::Input min_freezed_output,
                                     ::tensorflow::Input max_freezed_output,
                                     const gtl::ArraySlice<int>& strides,
                                     StringPiece padding, const
                                     QuantizedConv2DWithBiasAndRequantize::Attrs&
                                     attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DWithBiasSignedSumAndReluAndRequantize {
 public:
  /// Optional attribute setters for QuantizedConv2DWithBiasSignedSumAndReluAndRequantize
  struct Attrs {
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QUINT8;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DWithBiasSignedSumAndReluAndRequantize(const ::tensorflow::Scope&
                                                     scope, ::tensorflow::Input
                                                     input, ::tensorflow::Input
                                                     filter,
                                                     ::tensorflow::Input bias,
                                                     ::tensorflow::Input
                                                     min_input,
                                                     ::tensorflow::Input
                                                     max_input,
                                                     ::tensorflow::Input
                                                     min_filter,
                                                     ::tensorflow::Input
                                                     max_filter,
                                                     ::tensorflow::Input
                                                     min_freezed_output,
                                                     ::tensorflow::Input
                                                     max_freezed_output,
                                                     ::tensorflow::Input
                                                     summand,
                                                     ::tensorflow::Input
                                                     min_summand,
                                                     ::tensorflow::Input
                                                     max_summand, const
                                                     gtl::ArraySlice<int>&
                                                     strides, StringPiece
                                                     padding);
  QuantizedConv2DWithBiasSignedSumAndReluAndRequantize(const ::tensorflow::Scope&
                                                     scope, ::tensorflow::Input
                                                     input, ::tensorflow::Input
                                                     filter,
                                                     ::tensorflow::Input bias,
                                                     ::tensorflow::Input
                                                     min_input,
                                                     ::tensorflow::Input
                                                     max_input,
                                                     ::tensorflow::Input
                                                     min_filter,
                                                     ::tensorflow::Input
                                                     max_filter,
                                                     ::tensorflow::Input
                                                     min_freezed_output,
                                                     ::tensorflow::Input
                                                     max_freezed_output,
                                                     ::tensorflow::Input
                                                     summand,
                                                     ::tensorflow::Input
                                                     min_summand,
                                                     ::tensorflow::Input
                                                     max_summand, const
                                                     gtl::ArraySlice<int>&
                                                     strides, StringPiece
                                                     padding, const
                                                     QuantizedConv2DWithBiasSignedSumAndReluAndRequantize::Attrs&
                                                     attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DWithBiasSumAndRelu {
 public:
  /// Optional attribute setters for QuantizedConv2DWithBiasSumAndRelu
  struct Attrs {
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DWithBiasSumAndRelu(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input input,
                                  ::tensorflow::Input filter,
                                  ::tensorflow::Input bias, ::tensorflow::Input
                                  min_input, ::tensorflow::Input max_input,
                                  ::tensorflow::Input min_filter,
                                  ::tensorflow::Input max_filter,
                                  ::tensorflow::Input summand, const
                                  gtl::ArraySlice<int>& strides, StringPiece
                                  padding);
  QuantizedConv2DWithBiasSumAndRelu(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input input,
                                  ::tensorflow::Input filter,
                                  ::tensorflow::Input bias, ::tensorflow::Input
                                  min_input, ::tensorflow::Input max_input,
                                  ::tensorflow::Input min_filter,
                                  ::tensorflow::Input max_filter,
                                  ::tensorflow::Input summand, const
                                  gtl::ArraySlice<int>& strides, StringPiece
                                  padding, const
                                  QuantizedConv2DWithBiasSumAndRelu::Attrs&
                                  attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` output
/// * `Output` min_output
/// * `Output` max_output
class QuantizedConv2DWithBiasSumAndReluAndRequantize {
 public:
  /// Optional attribute setters for QuantizedConv2DWithBiasSumAndReluAndRequantize
  struct Attrs {
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QUINT8;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedConv2DWithBiasSumAndReluAndRequantize(const ::tensorflow::Scope&
                                               scope, ::tensorflow::Input
                                               input, ::tensorflow::Input
                                               filter, ::tensorflow::Input
                                               bias, ::tensorflow::Input
                                               min_input, ::tensorflow::Input
                                               max_input, ::tensorflow::Input
                                               min_filter, ::tensorflow::Input
                                               max_filter, ::tensorflow::Input
                                               min_freezed_output,
                                               ::tensorflow::Input
                                               max_freezed_output,
                                               ::tensorflow::Input summand,
                                               ::tensorflow::Input min_summand,
                                               ::tensorflow::Input max_summand,
                                               const gtl::ArraySlice<int>&
                                               strides, StringPiece padding);
  QuantizedConv2DWithBiasSumAndReluAndRequantize(const ::tensorflow::Scope&
                                               scope, ::tensorflow::Input
                                               input, ::tensorflow::Input
                                               filter, ::tensorflow::Input
                                               bias, ::tensorflow::Input
                                               min_input, ::tensorflow::Input
                                               max_input, ::tensorflow::Input
                                               min_filter, ::tensorflow::Input
                                               max_filter, ::tensorflow::Input
                                               min_freezed_output,
                                               ::tensorflow::Input
                                               max_freezed_output,
                                               ::tensorflow::Input summand,
                                               ::tensorflow::Input min_summand,
                                               ::tensorflow::Input max_summand,
                                               const gtl::ArraySlice<int>&
                                               strides, StringPiece padding,
                                               const
                                               QuantizedConv2DWithBiasSumAndReluAndRequantize::Attrs&
                                               attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// Computes quantized depthwise Conv2D.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input tensor.
/// * filter: The original filter tensor.
/// * min_input: The float value that the minimum quantized input value represents.
/// * max_input: The float value that the maximum quantized input value represents.
/// * min_filter: The float value that the minimum quantized filter value represents.
/// * max_filter: The float value that the maximum quantized filter value represents.
/// * strides: List of stride values.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The type of the output.
/// * dilations: List of dilation values.
///
/// Returns:
/// * `Output` output: The output tensor.
/// * `Output` min_output: The float value that the minimum quantized output value represents.
/// * `Output` max_output: The float value that the maximum quantized output value represents.
class QuantizedDepthwiseConv2D {
 public:
  /// Optional attribute setters for QuantizedDepthwiseConv2D
  struct Attrs {
    /// The type of the output.
    ///
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// List of dilation values.
    ///
    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedDepthwiseConv2D(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input, ::tensorflow::Input filter, ::tensorflow::Input
                         min_input, ::tensorflow::Input max_input,
                         ::tensorflow::Input min_filter, ::tensorflow::Input
                         max_filter, const gtl::ArraySlice<int>& strides,
                         StringPiece padding);
  QuantizedDepthwiseConv2D(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input, ::tensorflow::Input filter, ::tensorflow::Input
                         min_input, ::tensorflow::Input max_input,
                         ::tensorflow::Input min_filter, ::tensorflow::Input
                         max_filter, const gtl::ArraySlice<int>& strides,
                         StringPiece padding, const
                         QuantizedDepthwiseConv2D::Attrs& attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// Computes quantized depthwise Conv2D with Bias.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input tensor.
/// * filter: The original filter tensor.
/// * bias: The original bias tensor.
/// * min_input: The float value that the minimum quantized input value represents.
/// * max_input: The float value that the maximum quantized input value represents.
/// * min_filter: The float value that the minimum quantized filter value represents.
/// * max_filter: The float value that the maximum quantized filter value represents.
/// * strides: List of stride values.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The type of the output.
/// * dilations: List of dilation values.
///
/// Returns:
/// * `Output` output: The output tensor.
/// * `Output` min_output: The float value that the minimum quantized output value represents.
/// * `Output` max_output: The float value that the maximum quantized output value represents.
class QuantizedDepthwiseConv2DWithBias {
 public:
  /// Optional attribute setters for QuantizedDepthwiseConv2DWithBias
  struct Attrs {
    /// The type of the output.
    ///
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// List of dilation values.
    ///
    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedDepthwiseConv2DWithBias(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input input, ::tensorflow::Input
                                 filter, ::tensorflow::Input bias,
                                 ::tensorflow::Input min_input,
                                 ::tensorflow::Input max_input,
                                 ::tensorflow::Input min_filter,
                                 ::tensorflow::Input max_filter, const
                                 gtl::ArraySlice<int>& strides, StringPiece
                                 padding);
  QuantizedDepthwiseConv2DWithBias(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input input, ::tensorflow::Input
                                 filter, ::tensorflow::Input bias,
                                 ::tensorflow::Input min_input,
                                 ::tensorflow::Input max_input,
                                 ::tensorflow::Input min_filter,
                                 ::tensorflow::Input max_filter, const
                                 gtl::ArraySlice<int>& strides, StringPiece
                                 padding, const
                                 QuantizedDepthwiseConv2DWithBias::Attrs&
                                 attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// Computes quantized depthwise Conv2D with Bias and Relu.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input tensor.
/// * filter: The original filter tensor.
/// * bias: The original bias tensor.
/// * min_input: The float value that the minimum quantized input value represents.
/// * max_input: The float value that the maximum quantized input value represents.
/// * min_filter: The float value that the minimum quantized filter value represents.
/// * max_filter: The float value that the maximum quantized filter value represents.
/// * strides: List of stride values.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The type of the output.
/// * dilations: List of dilation values.
///
/// Returns:
/// * `Output` output: The output tensor.
/// * `Output` min_output: The float value that the minimum quantized output value represents.
/// * `Output` max_output: The float value that the maximum quantized output value represents.
class QuantizedDepthwiseConv2DWithBiasAndRelu {
 public:
  /// Optional attribute setters for QuantizedDepthwiseConv2DWithBiasAndRelu
  struct Attrs {
    /// The type of the output.
    ///
    /// Defaults to DT_QINT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// List of dilation values.
    ///
    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QINT32;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedDepthwiseConv2DWithBiasAndRelu(const ::tensorflow::Scope& scope,
                                        ::tensorflow::Input input,
                                        ::tensorflow::Input filter,
                                        ::tensorflow::Input bias,
                                        ::tensorflow::Input min_input,
                                        ::tensorflow::Input max_input,
                                        ::tensorflow::Input min_filter,
                                        ::tensorflow::Input max_filter, const
                                        gtl::ArraySlice<int>& strides,
                                        StringPiece padding);
  QuantizedDepthwiseConv2DWithBiasAndRelu(const ::tensorflow::Scope& scope,
                                        ::tensorflow::Input input,
                                        ::tensorflow::Input filter,
                                        ::tensorflow::Input bias,
                                        ::tensorflow::Input min_input,
                                        ::tensorflow::Input max_input,
                                        ::tensorflow::Input min_filter,
                                        ::tensorflow::Input max_filter, const
                                        gtl::ArraySlice<int>& strides,
                                        StringPiece padding, const
                                        QuantizedDepthwiseConv2DWithBiasAndRelu::Attrs&
                                        attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// Computes quantized depthwise Conv2D with Bias, Relu and Requantize.
///
/// Args:
/// * scope: A Scope object
/// * input: The original input tensor.
/// * filter: The original filter tensor.
/// * bias: The original bias tensor.
/// * min_input: The float value that the minimum quantized input value represents.
/// * max_input: The float value that the maximum quantized input value represents.
/// * min_filter: The float value that the minimum quantized filter value represents.
/// * max_filter: The float value that the maximum quantized filter value represents.
/// * min_freezed_output: The minimum float value of the output tensor.
/// * max_freezed_output: The maximum float value of the output tensor.
/// * strides: List of stride values.
///
/// Optional attributes (see `Attrs`):
/// * out_type: The type of the output.
/// * dilations: List of dilation values.
///
/// Returns:
/// * `Output` output: The output tensor.
/// * `Output` min_output: The float value that the minimum quantized output value represents.
/// * `Output` max_output: The float value that the maximum quantized output value represents.
class QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize {
 public:
  /// Optional attribute setters for QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize
  struct Attrs {
    /// The type of the output.
    ///
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    /// List of dilation values.
    ///
    /// Defaults to [1, 1, 1, 1]
    TF_MUST_USE_RESULT Attrs Dilations(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.dilations_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs PaddingList(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.padding_list_ = x;
      return ret;
    }

    DataType out_type_ = DT_QUINT8;
    gtl::ArraySlice<int> dilations_ = Default_dilations();
    gtl::ArraySlice<int> padding_list_ = {};
  private:
    static gtl::ArraySlice<int> Default_dilations() {
      static const int kStorage[] = {1, 1, 1, 1};
      return gtl::ArraySlice<int>(kStorage);
    }
  };
  QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize(const ::tensorflow::Scope&
                                                     scope, ::tensorflow::Input
                                                     input, ::tensorflow::Input
                                                     filter,
                                                     ::tensorflow::Input bias,
                                                     ::tensorflow::Input
                                                     min_input,
                                                     ::tensorflow::Input
                                                     max_input,
                                                     ::tensorflow::Input
                                                     min_filter,
                                                     ::tensorflow::Input
                                                     max_filter,
                                                     ::tensorflow::Input
                                                     min_freezed_output,
                                                     ::tensorflow::Input
                                                     max_freezed_output, const
                                                     gtl::ArraySlice<int>&
                                                     strides, StringPiece
                                                     padding);
  QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize(const ::tensorflow::Scope&
                                                     scope, ::tensorflow::Input
                                                     input, ::tensorflow::Input
                                                     filter,
                                                     ::tensorflow::Input bias,
                                                     ::tensorflow::Input
                                                     min_input,
                                                     ::tensorflow::Input
                                                     max_input,
                                                     ::tensorflow::Input
                                                     min_filter,
                                                     ::tensorflow::Input
                                                     max_filter,
                                                     ::tensorflow::Input
                                                     min_freezed_output,
                                                     ::tensorflow::Input
                                                     max_freezed_output, const
                                                     gtl::ArraySlice<int>&
                                                     strides, StringPiece
                                                     padding, const
                                                     QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize::Attrs&
                                                     attrs);

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }
  static Attrs Dilations(const gtl::ArraySlice<int>& x) {
    return Attrs().Dilations(x);
  }
  static Attrs PaddingList(const gtl::ArraySlice<int>& x) {
    return Attrs().PaddingList(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output min_output;
  ::tensorflow::Output max_output;
};

/// Performs a quantized matrix multiplication of `a` by the matrix `b` with bias
/// add.
///
/// The inputs must be two-dimensional matrices and 1D bias vector. And the inner
/// dimension of `a` (after being transposed if `transpose_a` is non-zero) must
/// match the outer dimension of `b` (after being transposed if `transposed_b` is
/// non-zero). Then do broadcast add operation with bias values on the matrix
/// multiplication result. The bias size must match inner dimension of `b`.
///
/// Args:
/// * scope: A Scope object
/// * a: A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
/// * b: A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
/// * bias: A 1D bias tensor with size matching inner dimension of `b` (after being
/// transposed if `transposed_b` is non-zero).
/// * min_a: The float value that the lowest quantized `a` value represents.
/// * max_a: The float value that the highest quantized `a` value represents.
/// * min_b: The float value that the lowest quantized `b` value represents.
/// * max_b: The float value that the highest quantized `b` value represents.
///
/// Optional attributes (see `Attrs`):
/// * transpose_a: If true, `a` is transposed before multiplication.
/// * transpose_b: If true, `b` is transposed before multiplication.
/// * input_quant_mode: Input data quantization mode. Either MIN_FIRST(default) or SCALED.
///
/// Returns:
/// * `Output` out
/// * `Output` min_out: The float value that the lowest quantized output value represents.
/// * `Output` max_out: The float value that the highest quantized output value represents.
class QuantizedMatMulWithBias {
 public:
  /// Optional attribute setters for QuantizedMatMulWithBias
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

    /// Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    ///
    /// Defaults to "MIN_FIRST"
    TF_MUST_USE_RESULT Attrs InputQuantMode(StringPiece x) {
      Attrs ret = *this;
      ret.input_quant_mode_ = x;
      return ret;
    }

    DataType Toutput_ = DT_QINT32;
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    StringPiece input_quant_mode_ = "MIN_FIRST";
  };
  QuantizedMatMulWithBias(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        a, ::tensorflow::Input b, ::tensorflow::Input bias,
                        ::tensorflow::Input min_a, ::tensorflow::Input max_a,
                        ::tensorflow::Input min_b, ::tensorflow::Input max_b);
  QuantizedMatMulWithBias(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        a, ::tensorflow::Input b, ::tensorflow::Input bias,
                        ::tensorflow::Input min_a, ::tensorflow::Input max_a,
                        ::tensorflow::Input min_b, ::tensorflow::Input max_b,
                        const QuantizedMatMulWithBias::Attrs& attrs);

  static Attrs Toutput(DataType x) {
    return Attrs().Toutput(x);
  }
  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs InputQuantMode(StringPiece x) {
    return Attrs().InputQuantMode(x);
  }

  Operation operation;
  ::tensorflow::Output out;
  ::tensorflow::Output min_out;
  ::tensorflow::Output max_out;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The out tensor.
class QuantizedMatMulWithBiasAndDequantize {
 public:
  /// Optional attribute setters for QuantizedMatMulWithBiasAndDequantize
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

    /// Defaults to "MIN_FIRST"
    TF_MUST_USE_RESULT Attrs InputQuantMode(StringPiece x) {
      Attrs ret = *this;
      ret.input_quant_mode_ = x;
      return ret;
    }

    bool transpose_a_ = false;
    bool transpose_b_ = false;
    StringPiece input_quant_mode_ = "MIN_FIRST";
  };
  QuantizedMatMulWithBiasAndDequantize(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input a, ::tensorflow::Input
                                     b, ::tensorflow::Input bias,
                                     ::tensorflow::Input min_a,
                                     ::tensorflow::Input max_a,
                                     ::tensorflow::Input min_b,
                                     ::tensorflow::Input max_b,
                                     ::tensorflow::Input min_freezed_output,
                                     ::tensorflow::Input max_freezed_output,
                                     DataType Toutput);
  QuantizedMatMulWithBiasAndDequantize(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input a, ::tensorflow::Input
                                     b, ::tensorflow::Input bias,
                                     ::tensorflow::Input min_a,
                                     ::tensorflow::Input max_a,
                                     ::tensorflow::Input min_b,
                                     ::tensorflow::Input max_b,
                                     ::tensorflow::Input min_freezed_output,
                                     ::tensorflow::Input max_freezed_output,
                                     DataType Toutput, const
                                     QuantizedMatMulWithBiasAndDequantize::Attrs&
                                     attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs InputQuantMode(StringPiece x) {
    return Attrs().InputQuantMode(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
/// add and relu fusion.
///
/// The inputs must be two-dimensional matrices and 1D bias vector. And the inner
/// dimension of `a` (after being transposed if `transpose_a` is non-zero) must
/// match the outer dimension of `b` (after being transposed if `transposed_b` is
/// non-zero). Then do broadcast add operation with bias values on the matrix
/// multiplication result. The bias size must match inner dimension of `b`. Then do
/// relu activation to get non-negative result.
///
/// Args:
/// * scope: A Scope object
/// * a: A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
/// * b: A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
/// * bias: A 1D bias tensor with size matching with inner dimension of `b` (after being
/// transposed if `transposed_b` is non-zero).
/// * min_a: The float value that the lowest quantized `a` value represents.
/// * max_a: The float value that the highest quantized `a` value represents.
/// * min_b: The float value that the lowest quantized `b` value represents.
/// * max_b: The float value that the highest quantized `b` value represents.
///
/// Optional attributes (see `Attrs`):
/// * transpose_a: If true, `a` is transposed before multiplication.
/// * transpose_b: If true, `b` is transposed before multiplication.
/// * input_quant_mode: Input data quantization mode. Either MIN_FIRST(default) or SCALED.
///
/// Returns:
/// * `Output` out
/// * `Output` min_out: The float value that the lowest quantized output value represents.
/// * `Output` max_out: The float value that the highest quantized output value represents.
class QuantizedMatMulWithBiasAndRelu {
 public:
  /// Optional attribute setters for QuantizedMatMulWithBiasAndRelu
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

    /// Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    ///
    /// Defaults to "MIN_FIRST"
    TF_MUST_USE_RESULT Attrs InputQuantMode(StringPiece x) {
      Attrs ret = *this;
      ret.input_quant_mode_ = x;
      return ret;
    }

    DataType Toutput_ = DT_QINT32;
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    StringPiece input_quant_mode_ = "MIN_FIRST";
  };
  QuantizedMatMulWithBiasAndRelu(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input a, ::tensorflow::Input b,
                               ::tensorflow::Input bias, ::tensorflow::Input
                               min_a, ::tensorflow::Input max_a,
                               ::tensorflow::Input min_b, ::tensorflow::Input
                               max_b);
  QuantizedMatMulWithBiasAndRelu(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input a, ::tensorflow::Input b,
                               ::tensorflow::Input bias, ::tensorflow::Input
                               min_a, ::tensorflow::Input max_a,
                               ::tensorflow::Input min_b, ::tensorflow::Input
                               max_b, const
                               QuantizedMatMulWithBiasAndRelu::Attrs& attrs);

  static Attrs Toutput(DataType x) {
    return Attrs().Toutput(x);
  }
  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs InputQuantMode(StringPiece x) {
    return Attrs().InputQuantMode(x);
  }

  Operation operation;
  ::tensorflow::Output out;
  ::tensorflow::Output min_out;
  ::tensorflow::Output max_out;
};

/// Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
/// add and relu and requantize fusion.
///
/// The inputs must be two-dimensional matrices and 1D bias vector. And the inner
/// dimension of `a` (after being transposed if `transpose_a` is non-zero) must
/// match the outer dimension of `b` (after being transposed if `transposed_b` is
/// non-zero). Then do broadcast add operation with bias values on the matrix
/// multiplication result. The bias size must match inner dimension of `b`.  Then do
/// relu activation to get non-negative result. Then do requantize operation to get
/// final uint8 result.
///
/// Args:
/// * scope: A Scope object
/// * a: A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
/// * b: A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
/// * bias: A 1D bias tensor with size matching with inner dimension of `b` (after being
/// transposed if `transposed_b` is non-zero).
/// * min_a: The float value that the lowest quantized `a` value represents.
/// * max_a: The float value that the highest quantized `a` value represents.
/// * min_b: The float value that the lowest quantized `b` value represents.
/// * max_b: The float value that the highest quantized `b` value represents.
/// * min_freezed_output: The float value that the highest quantized output value after requantize.
///
/// Optional attributes (see `Attrs`):
/// * transpose_a: If true, `a` is transposed before multiplication.
/// * transpose_b: If true, `b` is transposed before multiplication.
/// * input_quant_mode: Input data quantization mode. Either MIN_FIRST(default) or SCALED.
///
/// Returns:
/// * `Output` out
/// * `Output` min_out: The float value that the lowest quantized output value represents.
/// * `Output` max_out: The float value that the highest quantized output value represents.
class QuantizedMatMulWithBiasAndReluAndRequantize {
 public:
  /// Optional attribute setters for QuantizedMatMulWithBiasAndReluAndRequantize
  struct Attrs {
    /// Defaults to DT_QUINT8
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

    /// Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    ///
    /// Defaults to "MIN_FIRST"
    TF_MUST_USE_RESULT Attrs InputQuantMode(StringPiece x) {
      Attrs ret = *this;
      ret.input_quant_mode_ = x;
      return ret;
    }

    DataType Toutput_ = DT_QUINT8;
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    StringPiece input_quant_mode_ = "MIN_FIRST";
  };
  QuantizedMatMulWithBiasAndReluAndRequantize(const ::tensorflow::Scope& scope,
                                            ::tensorflow::Input a,
                                            ::tensorflow::Input b,
                                            ::tensorflow::Input bias,
                                            ::tensorflow::Input min_a,
                                            ::tensorflow::Input max_a,
                                            ::tensorflow::Input min_b,
                                            ::tensorflow::Input max_b,
                                            ::tensorflow::Input
                                            min_freezed_output,
                                            ::tensorflow::Input
                                            max_freezed_output);
  QuantizedMatMulWithBiasAndReluAndRequantize(const ::tensorflow::Scope& scope,
                                            ::tensorflow::Input a,
                                            ::tensorflow::Input b,
                                            ::tensorflow::Input bias,
                                            ::tensorflow::Input min_a,
                                            ::tensorflow::Input max_a,
                                            ::tensorflow::Input min_b,
                                            ::tensorflow::Input max_b,
                                            ::tensorflow::Input
                                            min_freezed_output,
                                            ::tensorflow::Input
                                            max_freezed_output, const
                                            QuantizedMatMulWithBiasAndReluAndRequantize::Attrs&
                                            attrs);

  static Attrs Toutput(DataType x) {
    return Attrs().Toutput(x);
  }
  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs InputQuantMode(StringPiece x) {
    return Attrs().InputQuantMode(x);
  }

  Operation operation;
  ::tensorflow::Output out;
  ::tensorflow::Output min_out;
  ::tensorflow::Output max_out;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` out
/// * `Output` min_out
/// * `Output` max_out
class QuantizedMatMulWithBiasAndRequantize {
 public:
  /// Optional attribute setters for QuantizedMatMulWithBiasAndRequantize
  struct Attrs {
    /// Defaults to DT_QUINT8
    TF_MUST_USE_RESULT Attrs Toutput(DataType x) {
      Attrs ret = *this;
      ret.Toutput_ = x;
      return ret;
    }

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

    /// Defaults to "MIN_FIRST"
    TF_MUST_USE_RESULT Attrs InputQuantMode(StringPiece x) {
      Attrs ret = *this;
      ret.input_quant_mode_ = x;
      return ret;
    }

    DataType Toutput_ = DT_QUINT8;
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    StringPiece input_quant_mode_ = "MIN_FIRST";
  };
  QuantizedMatMulWithBiasAndRequantize(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input a, ::tensorflow::Input
                                     b, ::tensorflow::Input bias,
                                     ::tensorflow::Input min_a,
                                     ::tensorflow::Input max_a,
                                     ::tensorflow::Input min_b,
                                     ::tensorflow::Input max_b,
                                     ::tensorflow::Input min_freezed_output,
                                     ::tensorflow::Input max_freezed_output);
  QuantizedMatMulWithBiasAndRequantize(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input a, ::tensorflow::Input
                                     b, ::tensorflow::Input bias,
                                     ::tensorflow::Input min_a,
                                     ::tensorflow::Input max_a,
                                     ::tensorflow::Input min_b,
                                     ::tensorflow::Input max_b,
                                     ::tensorflow::Input min_freezed_output,
                                     ::tensorflow::Input max_freezed_output,
                                     const
                                     QuantizedMatMulWithBiasAndRequantize::Attrs&
                                     attrs);

  static Attrs Toutput(DataType x) {
    return Attrs().Toutput(x);
  }
  static Attrs TransposeA(bool x) {
    return Attrs().TransposeA(x);
  }
  static Attrs TransposeB(bool x) {
    return Attrs().TransposeB(x);
  }
  static Attrs InputQuantMode(StringPiece x) {
    return Attrs().InputQuantMode(x);
  }

  Operation operation;
  ::tensorflow::Output out;
  ::tensorflow::Output min_out;
  ::tensorflow::Output max_out;
};

/// Computes rectified linear 6 gradients for a Relu6 operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: The backpropagated gradients to the corresponding Relu6 operation.
/// * features: The features passed as input to the corresponding Relu6 operation, or
/// its output; using either one produces the same result.
///
/// Returns:
/// * `Output`: The gradients:
/// `gradients * (features > 0) * (features < 6)`.
class Relu6Grad {
 public:
  Relu6Grad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
          ::tensorflow::Input features);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  Operation operation;
  ::tensorflow::Output backprops;
};

/// Computes rectified linear gradients for a Relu operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: The backpropagated gradients to the corresponding Relu operation.
/// * features: The features passed as input to the corresponding Relu operation, OR
/// the outputs of that operation (both work equivalently).
///
/// Returns:
/// * `Output`: `gradients * (features > 0)`.
class ReluGrad {
 public:
  ReluGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
         ::tensorflow::Input features);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  Operation operation;
  ::tensorflow::Output backprops;
};

/// Computes gradients for the scaled exponential linear (Selu) operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: The backpropagated gradients to the corresponding Selu operation.
/// * outputs: The outputs of the corresponding Selu operation.
///
/// Returns:
/// * `Output`: The gradients: `gradients * (outputs + scale * alpha)`
/// if outputs < 0, `scale * gradients` otherwise.
class SeluGrad {
 public:
  SeluGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
         ::tensorflow::Input outputs);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  Operation operation;
  ::tensorflow::Output backprops;
};

/// Computes softplus gradients for a softplus operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: The backpropagated gradients to the corresponding softplus operation.
/// * features: The features passed as input to the corresponding softplus operation.
///
/// Returns:
/// * `Output`: The gradients: `gradients / (1 + exp(-features))`.
class SoftplusGrad {
 public:
  SoftplusGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
             ::tensorflow::Input features);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  Operation operation;
  ::tensorflow::Output backprops;
};

/// Computes softsign gradients for a softsign operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: The backpropagated gradients to the corresponding softsign operation.
/// * features: The features passed as input to the corresponding softsign operation.
///
/// Returns:
/// * `Output`: The gradients: `gradients / (1 + abs(features)) ** 2`.
class SoftsignGrad {
 public:
  SoftsignGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input gradients,
             ::tensorflow::Input features);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  Operation operation;
  ::tensorflow::Output backprops;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_NN_OPS_INTERNAL_H_
