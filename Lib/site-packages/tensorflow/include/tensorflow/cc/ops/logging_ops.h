// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LOGGING_OPS_H_
#define TENSORFLOW_CC_OPS_LOGGING_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup logging_ops Logging Ops
/// @{

/// Asserts that the given condition is true.
///
/// If `condition` evaluates to false, print the list of tensors in `data`.
/// `summarize` determines how many entries of the tensors to print.
///
/// Args:
/// * scope: A Scope object
/// * condition: The condition to evaluate.
/// * data: The tensors to print out when condition is false.
///
/// Optional attributes (see `Attrs`):
/// * summarize: Print this many entries of each tensor.
///
/// Returns:
/// * the created `Operation`
class Assert {
 public:
  /// Optional attribute setters for Assert
  struct Attrs {
    /// Print this many entries of each tensor.
    ///
    /// Defaults to 3
    TF_MUST_USE_RESULT Attrs Summarize(int64 x) {
      Attrs ret = *this;
      ret.summarize_ = x;
      return ret;
    }

    int64 summarize_ = 3;
  };
  Assert(const ::tensorflow::Scope& scope, ::tensorflow::Input condition,
       ::tensorflow::InputList data);
  Assert(const ::tensorflow::Scope& scope, ::tensorflow::Input condition,
       ::tensorflow::InputList data, const Assert::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Summarize(int64 x) {
    return Attrs().Summarize(x);
  }

  Operation operation;
};

/// Outputs a `Summary` protocol buffer with audio.
///
/// The summary has up to `max_outputs` summary values containing audio. The
/// audio is built from `tensor` which must be 3-D with shape `[batch_size,
/// frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
/// assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.
///
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
///
/// *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
/// *  If `max_outputs` is greater than 1, the summary value tags are
///    generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.
///
/// Args:
/// * scope: A Scope object
/// * tag: Scalar. Used to build the `tag` attribute of the summary values.
/// * tensor: 2-D of shape `[batch_size, frames]`.
/// * sample_rate: The sample rate of the signal in hertz.
///
/// Optional attributes (see `Attrs`):
/// * max_outputs: Max number of batch elements to generate audio for.
///
/// Returns:
/// * `Output`: Scalar. Serialized `Summary` protocol buffer.
class AudioSummary {
 public:
  /// Optional attribute setters for AudioSummary
  struct Attrs {
    /// Max number of batch elements to generate audio for.
    ///
    /// Defaults to 3
    TF_MUST_USE_RESULT Attrs MaxOutputs(int64 x) {
      Attrs ret = *this;
      ret.max_outputs_ = x;
      return ret;
    }

    int64 max_outputs_ = 3;
  };
  AudioSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tag,
             ::tensorflow::Input tensor, ::tensorflow::Input sample_rate);
  AudioSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tag,
             ::tensorflow::Input tensor, ::tensorflow::Input sample_rate, const
             AudioSummary::Attrs& attrs);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  static Attrs MaxOutputs(int64 x) {
    return Attrs().MaxOutputs(x);
  }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Outputs a `Summary` protocol buffer with a histogram.
///
/// The generated
/// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
/// has one summary value containing a histogram for `values`.
///
/// This op reports an `InvalidArgument` error if any value is not finite.
///
/// Args:
/// * scope: A Scope object
/// * tag: Scalar.  Tag to use for the `Summary.Value`.
/// * values: Any shape. Values to use to build the histogram.
///
/// Returns:
/// * `Output`: Scalar. Serialized `Summary` protocol buffer.
class HistogramSummary {
 public:
  HistogramSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tag,
                 ::tensorflow::Input values);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Outputs a `Summary` protocol buffer with images.
///
/// The summary has up to `max_images` summary values containing images. The
/// images are built from `tensor` which must be 4-D with shape `[batch_size,
/// height, width, channels]` and where `channels` can be:
///
/// *  1: `tensor` is interpreted as Grayscale.
/// *  3: `tensor` is interpreted as RGB.
/// *  4: `tensor` is interpreted as RGBA.
///
/// The images have the same number of channels as the input tensor. For float
/// input, the values are normalized one image at a time to fit in the range
/// `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
/// normalization algorithms:
///
/// *  If the input values are all positive, they are rescaled so the largest one
///    is 255.
///
/// *  If any input value is negative, the values are shifted so input value 0.0
///    is at 127.  They are then rescaled so that either the smallest value is 0,
///    or the largest one is 255.
///
/// The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
/// build the `tag` of the summary values:
///
/// *  If `max_images` is 1, the summary value tag is '*tag*/image'.
/// *  If `max_images` is greater than 1, the summary value tags are
///    generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.
///
/// The `bad_color` argument is the color to use in the generated images for
/// non-finite input values.  It is a `uint8` 1-D tensor of length `channels`.
/// Each element must be in the range `[0, 255]` (It represents the value of a
/// pixel in the output image).  Non-finite values in the input tensor are
/// replaced by this tensor in the output image.  The default value is the color
/// red.
///
/// Args:
/// * scope: A Scope object
/// * tag: Scalar. Used to build the `tag` attribute of the summary values.
/// * tensor: 4-D of shape `[batch_size, height, width, channels]` where
/// `channels` is 1, 3, or 4.
///
/// Optional attributes (see `Attrs`):
/// * max_images: Max number of batch elements to generate images for.
/// * bad_color: Color to use for pixels with non-finite values.
///
/// Returns:
/// * `Output`: Scalar. Serialized `Summary` protocol buffer.
class ImageSummary {
 public:
  /// Optional attribute setters for ImageSummary
  struct Attrs {
    /// Max number of batch elements to generate images for.
    ///
    /// Defaults to 3
    TF_MUST_USE_RESULT Attrs MaxImages(int64 x) {
      Attrs ret = *this;
      ret.max_images_ = x;
      return ret;
    }

    /// Color to use for pixels with non-finite values.
    ///
    /// Defaults to Tensor<type: uint8 shape: [4] values: 255 0 0...>
    TF_MUST_USE_RESULT Attrs BadColor(const TensorProto& x) {
      Attrs ret = *this;
      ret.bad_color_ = x;
      return ret;
    }

    int64 max_images_ = 3;
    TensorProto bad_color_ = Input::Initializer({255, 0, 0, 255}, {4}).AsTensorProto();
  };
  ImageSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tag,
             ::tensorflow::Input tensor);
  ImageSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tag,
             ::tensorflow::Input tensor, const ImageSummary::Attrs& attrs);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  static Attrs MaxImages(int64 x) {
    return Attrs().MaxImages(x);
  }
  static Attrs BadColor(const TensorProto& x) {
    return Attrs().BadColor(x);
  }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Merges summaries.
///
/// This op creates a
/// [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
/// protocol buffer that contains the union of all the values in the input
/// summaries.
///
/// When the Op is run, it reports an `InvalidArgument` error if multiple values
/// in the summaries to merge use the same tag.
///
/// Args:
/// * scope: A Scope object
/// * inputs: Can be of any shape.  Each must contain serialized `Summary` protocol
/// buffers.
///
/// Returns:
/// * `Output`: Scalar. Serialized `Summary` protocol buffer.
class MergeSummary {
 public:
  MergeSummary(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Prints a list of tensors.
///
/// Passes `input` through to `output` and prints `data` when evaluating.
///
/// Args:
/// * scope: A Scope object
/// * input: The tensor passed to `output`
/// * data: A list of tensors to print out when op is evaluated.
///
/// Optional attributes (see `Attrs`):
/// * message: A string, prefix of the error message.
/// * first_n: Only log `first_n` number of times. -1 disables logging.
/// * summarize: Only print this many entries of each tensor.
///
/// Returns:
/// * `Output`: The unmodified `input` tensor
class Print {
 public:
  /// Optional attribute setters for Print
  struct Attrs {
    /// A string, prefix of the error message.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Message(StringPiece x) {
      Attrs ret = *this;
      ret.message_ = x;
      return ret;
    }

    /// Only log `first_n` number of times. -1 disables logging.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs FirstN(int64 x) {
      Attrs ret = *this;
      ret.first_n_ = x;
      return ret;
    }

    /// Only print this many entries of each tensor.
    ///
    /// Defaults to 3
    TF_MUST_USE_RESULT Attrs Summarize(int64 x) {
      Attrs ret = *this;
      ret.summarize_ = x;
      return ret;
    }

    StringPiece message_ = "";
    int64 first_n_ = -1;
    int64 summarize_ = 3;
  };
  Print(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
      ::tensorflow::InputList data);
  Print(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
      ::tensorflow::InputList data, const Print::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Message(StringPiece x) {
    return Attrs().Message(x);
  }
  static Attrs FirstN(int64 x) {
    return Attrs().FirstN(x);
  }
  static Attrs Summarize(int64 x) {
    return Attrs().Summarize(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Prints a string scalar.
///
/// Prints a string scalar to the desired output_stream.
///
/// Args:
/// * scope: A Scope object
/// * input: The string scalar to print.
///
/// Optional attributes (see `Attrs`):
/// * output_stream: A string specifying the output stream or logging level to print to.
///
/// Returns:
/// * the created `Operation`
class PrintV2 {
 public:
  /// Optional attribute setters for PrintV2
  struct Attrs {
    /// A string specifying the output stream or logging level to print to.
    ///
    /// Defaults to "stderr"
    TF_MUST_USE_RESULT Attrs OutputStream(StringPiece x) {
      Attrs ret = *this;
      ret.output_stream_ = x;
      return ret;
    }

    /// Defaults to "\n"
    TF_MUST_USE_RESULT Attrs End(StringPiece x) {
      Attrs ret = *this;
      ret.end_ = x;
      return ret;
    }

    StringPiece output_stream_ = "stderr";
    StringPiece end_ = "\n";
  };
  PrintV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  PrintV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
        PrintV2::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs OutputStream(StringPiece x) {
    return Attrs().OutputStream(x);
  }
  static Attrs End(StringPiece x) {
    return Attrs().End(x);
  }

  Operation operation;
};

/// Outputs a `Summary` protocol buffer with scalar values.
///
/// The input `tags` and `values` must have the same shape.  The generated summary
/// has a summary value for each tag-value pair in `tags` and `values`.
///
/// Args:
/// * scope: A Scope object
/// * tags: Tags for the summary.
/// * values: Same shape as `tags.  Values for the summary.
///
/// Returns:
/// * `Output`: Scalar.  Serialized `Summary` protocol buffer.
class ScalarSummary {
 public:
  ScalarSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tags,
              ::tensorflow::Input values);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Outputs a `Summary` protocol buffer with a tensor.
///
/// This op is being phased out in favor of TensorSummaryV2, which lets callers pass
/// a tag as well as a serialized SummaryMetadata proto string that contains
/// plugin-specific data. We will keep this op to maintain backwards compatibility.
///
/// Args:
/// * scope: A Scope object
/// * tensor: A tensor to serialize.
///
/// Optional attributes (see `Attrs`):
/// * description: A json-encoded SummaryDescription proto.
/// * labels: An unused list of strings.
/// * display_name: An unused string.
///
/// Returns:
/// * `Output`: The summary tensor.
class TensorSummary {
 public:
  /// Optional attribute setters for TensorSummary
  struct Attrs {
    /// A json-encoded SummaryDescription proto.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Description(StringPiece x) {
      Attrs ret = *this;
      ret.description_ = x;
      return ret;
    }

    /// An unused list of strings.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs Labels(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.labels_ = x;
      return ret;
    }

    /// An unused string.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs DisplayName(StringPiece x) {
      Attrs ret = *this;
      ret.display_name_ = x;
      return ret;
    }

    StringPiece description_ = "";
    gtl::ArraySlice<::tensorflow::tstring> labels_ = {};
    StringPiece display_name_ = "";
  };
  TensorSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor);
  TensorSummary(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
              const TensorSummary::Attrs& attrs);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  static Attrs Description(StringPiece x) {
    return Attrs().Description(x);
  }
  static Attrs Labels(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().Labels(x);
  }
  static Attrs DisplayName(StringPiece x) {
    return Attrs().DisplayName(x);
  }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Outputs a `Summary` protocol buffer with a tensor and per-plugin data.
///
/// Args:
/// * scope: A Scope object
/// * tag: A string attached to this summary. Used for organization in TensorBoard.
/// * tensor: A tensor to serialize.
/// * serialized_summary_metadata: A serialized SummaryMetadata proto. Contains plugin
/// data.
///
/// Returns:
/// * `Output`: The summary tensor.
class TensorSummaryV2 {
 public:
  TensorSummaryV2(const ::tensorflow::Scope& scope, ::tensorflow::Input tag,
                ::tensorflow::Input tensor, ::tensorflow::Input
                serialized_summary_metadata);
  operator ::tensorflow::Output() const { return summary; }
  operator ::tensorflow::Input() const { return summary; }
  ::tensorflow::Node* node() const { return summary.node(); }

  Operation operation;
  ::tensorflow::Output summary;
};

/// Provides the time since epoch in seconds.
///
/// Returns the timestamp as a `float64` for seconds since the Unix epoch.
///
/// Common usages include:
/// * Logging
/// * Providing a random number seed
/// * Debugging graph execution
/// * Generating timing information, mainly through comparison of timestamps
///
/// Note: In graph mode, the timestamp is computed when the op is executed,
/// not when it is added to the graph.  In eager mode, the timestamp is computed
/// when the op is eagerly executed.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The ts tensor.
class Timestamp {
 public:
  Timestamp(const ::tensorflow::Scope& scope);
  operator ::tensorflow::Output() const { return ts; }
  operator ::tensorflow::Input() const { return ts; }
  ::tensorflow::Node* node() const { return ts.node(); }

  Operation operation;
  ::tensorflow::Output ts;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LOGGING_OPS_H_
