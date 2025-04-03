// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_STRING_OPS_INTERNAL_H_
#define TENSORFLOW_CC_OPS_STRING_OPS_INTERNAL_H_

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

/// @defgroup string_ops_internal String Ops Internal
/// @{

/// Check if the input matches the regex pattern.
///
/// The input is a string tensor of any shape. The pattern is the
/// regular expression to be matched with every element of the input tensor.
/// The boolean values (True or False) of the output tensor indicate
/// if the input matches the regex pattern provided.
///
/// The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// Args:
/// * scope: A Scope object
/// * input: A string tensor of the text to be processed.
/// * pattern: The regular expression to match the input.
///
/// Returns:
/// * `Output`: A bool tensor with the same shape as `input`.
class StaticRegexFullMatch {
 public:
  StaticRegexFullMatch(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input, StringPiece pattern);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Replaces the match of pattern in input with rewrite.
///
/// It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// Args:
/// * scope: A Scope object
/// * input: The text to be processed.
/// * pattern: The regular expression to match the input.
/// * rewrite: The rewrite to be applied to the matched expression.
///
/// Optional attributes (see `Attrs`):
/// * replace_global: If True, the replacement is global, otherwise the replacement
/// is done only on the first match.
///
/// Returns:
/// * `Output`: The text after applying pattern and rewrite.
class StaticRegexReplace {
 public:
  /// Optional attribute setters for StaticRegexReplace
  struct Attrs {
    /// If True, the replacement is global, otherwise the replacement
    /// is done only on the first match.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ReplaceGlobal(bool x) {
      Attrs ret = *this;
      ret.replace_global_ = x;
      return ret;
    }

    bool replace_global_ = true;
  };
  StaticRegexReplace(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                   StringPiece pattern, StringPiece rewrite);
  StaticRegexReplace(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                   StringPiece pattern, StringPiece rewrite, const
                   StaticRegexReplace::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs ReplaceGlobal(bool x) {
    return Attrs().ReplaceGlobal(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Decodes each string in `input` into a sequence of Unicode code points.
///
/// The character codepoints for all strings are returned using a single vector
/// `char_values`, with strings expanded to characters in row-major order.
///
/// The `row_splits` tensor indicates where the codepoints for
/// each input string begin and end within the `char_values` tensor.
/// In particular, the values for the `i`th
/// string (in row-major order) are stored in the slice
/// `[row_splits[i]:row_splits[i+1]]`. Thus:
///
/// * `char_values[row_splits[i]+j]` is the Unicode codepoint for the `j`th
///   character in the `i`th string (in row-major order).
/// * `row_splits[i+1] - row_splits[i]` is the number of characters in the `i`th
///   string (in row-major order).
///
/// Args:
/// * scope: A Scope object
/// * input: The text to be decoded. Can have any shape. Note that the output is flattened
/// to a vector of char values.
/// * input_encoding: Text encoding of the input strings. This is any of the encodings supported
/// by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
///
/// Optional attributes (see `Attrs`):
/// * errors: Error handling policy when there is invalid formatting found in the input.
/// The value of 'strict' will cause the operation to produce a InvalidArgument
/// error on any invalid input formatting. A value of 'replace' (the default) will
/// cause the operation to replace any invalid formatting in the input with the
/// `replacement_char` codepoint. A value of 'ignore' will cause the operation to
/// skip any invalid formatting in the input and produce no corresponding output
/// character.
/// * replacement_char: The replacement character codepoint to be used in place of any invalid
/// formatting in the input when `errors='replace'`. Any valid unicode codepoint may
/// be used. The default value is the default unicode replacement character is
/// 0xFFFD or U+65533.)
/// * replace_control_characters: Whether to replace the C0 control characters (00-1F) with the
/// `replacement_char`. Default is false.
///
/// Returns:
/// * `Output` row_splits: A 1D int32 tensor containing the row splits.
/// * `Output` char_values: A 1D int32 Tensor containing the decoded codepoints.
class UnicodeDecode {
 public:
  /// Optional attribute setters for UnicodeDecode
  struct Attrs {
    /// Error handling policy when there is invalid formatting found in the input.
    /// The value of 'strict' will cause the operation to produce a InvalidArgument
    /// error on any invalid input formatting. A value of 'replace' (the default) will
    /// cause the operation to replace any invalid formatting in the input with the
    /// `replacement_char` codepoint. A value of 'ignore' will cause the operation to
    /// skip any invalid formatting in the input and produce no corresponding output
    /// character.
    ///
    /// Defaults to "replace"
    TF_MUST_USE_RESULT Attrs Errors(StringPiece x) {
      Attrs ret = *this;
      ret.errors_ = x;
      return ret;
    }

    /// The replacement character codepoint to be used in place of any invalid
    /// formatting in the input when `errors='replace'`. Any valid unicode codepoint may
    /// be used. The default value is the default unicode replacement character is
    /// 0xFFFD or U+65533.)
    ///
    /// Defaults to 65533
    TF_MUST_USE_RESULT Attrs ReplacementChar(int64 x) {
      Attrs ret = *this;
      ret.replacement_char_ = x;
      return ret;
    }

    /// Whether to replace the C0 control characters (00-1F) with the
    /// `replacement_char`. Default is false.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ReplaceControlCharacters(bool x) {
      Attrs ret = *this;
      ret.replace_control_characters_ = x;
      return ret;
    }

    /// Defaults to DT_INT64
    TF_MUST_USE_RESULT Attrs Tsplits(DataType x) {
      Attrs ret = *this;
      ret.Tsplits_ = x;
      return ret;
    }

    StringPiece errors_ = "replace";
    int64 replacement_char_ = 65533;
    bool replace_control_characters_ = false;
    DataType Tsplits_ = DT_INT64;
  };
  UnicodeDecode(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              StringPiece input_encoding);
  UnicodeDecode(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              StringPiece input_encoding, const UnicodeDecode::Attrs& attrs);

  static Attrs Errors(StringPiece x) {
    return Attrs().Errors(x);
  }
  static Attrs ReplacementChar(int64 x) {
    return Attrs().ReplacementChar(x);
  }
  static Attrs ReplaceControlCharacters(bool x) {
    return Attrs().ReplaceControlCharacters(x);
  }
  static Attrs Tsplits(DataType x) {
    return Attrs().Tsplits(x);
  }

  Operation operation;
  ::tensorflow::Output row_splits;
  ::tensorflow::Output char_values;
};

/// Decodes each string in `input` into a sequence of Unicode code points.
///
/// The character codepoints for all strings are returned using a single vector
/// `char_values`, with strings expanded to characters in row-major order.
/// Similarly, the character start byte offsets are returned using a single vector
/// `char_to_byte_starts`, with strings expanded in row-major order.
///
/// The `row_splits` tensor indicates where the codepoints and start offsets for
/// each input string begin and end within the `char_values` and
/// `char_to_byte_starts` tensors.  In particular, the values for the `i`th
/// string (in row-major order) are stored in the slice
/// `[row_splits[i]:row_splits[i+1]]`. Thus:
///
/// * `char_values[row_splits[i]+j]` is the Unicode codepoint for the `j`th
///   character in the `i`th string (in row-major order).
/// * `char_to_bytes_starts[row_splits[i]+j]` is the start byte offset for the `j`th
///   character in the `i`th string (in row-major order).
/// * `row_splits[i+1] - row_splits[i]` is the number of characters in the `i`th
///   string (in row-major order).
///
/// Args:
/// * scope: A Scope object
/// * input: The text to be decoded. Can have any shape. Note that the output is flattened
/// to a vector of char values.
/// * input_encoding: Text encoding of the input strings. This is any of the encodings supported
/// by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
///
/// Optional attributes (see `Attrs`):
/// * errors: Error handling policy when there is invalid formatting found in the input.
/// The value of 'strict' will cause the operation to produce a InvalidArgument
/// error on any invalid input formatting. A value of 'replace' (the default) will
/// cause the operation to replace any invalid formatting in the input with the
/// `replacement_char` codepoint. A value of 'ignore' will cause the operation to
/// skip any invalid formatting in the input and produce no corresponding output
/// character.
/// * replacement_char: The replacement character codepoint to be used in place of any invalid
/// formatting in the input when `errors='replace'`. Any valid unicode codepoint may
/// be used. The default value is the default unicode replacement character is
/// 0xFFFD or U+65533.)
/// * replace_control_characters: Whether to replace the C0 control characters (00-1F) with the
/// `replacement_char`. Default is false.
///
/// Returns:
/// * `Output` row_splits: A 1D int32 tensor containing the row splits.
/// * `Output` char_values: A 1D int32 Tensor containing the decoded codepoints.
/// * `Output` char_to_byte_starts: A 1D int32 Tensor containing the byte index in the input string where each
/// character in `char_values` starts.
class UnicodeDecodeWithOffsets {
 public:
  /// Optional attribute setters for UnicodeDecodeWithOffsets
  struct Attrs {
    /// Error handling policy when there is invalid formatting found in the input.
    /// The value of 'strict' will cause the operation to produce a InvalidArgument
    /// error on any invalid input formatting. A value of 'replace' (the default) will
    /// cause the operation to replace any invalid formatting in the input with the
    /// `replacement_char` codepoint. A value of 'ignore' will cause the operation to
    /// skip any invalid formatting in the input and produce no corresponding output
    /// character.
    ///
    /// Defaults to "replace"
    TF_MUST_USE_RESULT Attrs Errors(StringPiece x) {
      Attrs ret = *this;
      ret.errors_ = x;
      return ret;
    }

    /// The replacement character codepoint to be used in place of any invalid
    /// formatting in the input when `errors='replace'`. Any valid unicode codepoint may
    /// be used. The default value is the default unicode replacement character is
    /// 0xFFFD or U+65533.)
    ///
    /// Defaults to 65533
    TF_MUST_USE_RESULT Attrs ReplacementChar(int64 x) {
      Attrs ret = *this;
      ret.replacement_char_ = x;
      return ret;
    }

    /// Whether to replace the C0 control characters (00-1F) with the
    /// `replacement_char`. Default is false.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs ReplaceControlCharacters(bool x) {
      Attrs ret = *this;
      ret.replace_control_characters_ = x;
      return ret;
    }

    /// Defaults to DT_INT64
    TF_MUST_USE_RESULT Attrs Tsplits(DataType x) {
      Attrs ret = *this;
      ret.Tsplits_ = x;
      return ret;
    }

    StringPiece errors_ = "replace";
    int64 replacement_char_ = 65533;
    bool replace_control_characters_ = false;
    DataType Tsplits_ = DT_INT64;
  };
  UnicodeDecodeWithOffsets(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input, StringPiece input_encoding);
  UnicodeDecodeWithOffsets(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input, StringPiece input_encoding, const
                         UnicodeDecodeWithOffsets::Attrs& attrs);

  static Attrs Errors(StringPiece x) {
    return Attrs().Errors(x);
  }
  static Attrs ReplacementChar(int64 x) {
    return Attrs().ReplacementChar(x);
  }
  static Attrs ReplaceControlCharacters(bool x) {
    return Attrs().ReplaceControlCharacters(x);
  }
  static Attrs Tsplits(DataType x) {
    return Attrs().Tsplits(x);
  }

  Operation operation;
  ::tensorflow::Output row_splits;
  ::tensorflow::Output char_values;
  ::tensorflow::Output char_to_byte_starts;
};

/// Encode a tensor of ints into unicode strings.
///
/// Returns a vector of strings, where `output[i]` is constructed by encoding the
/// Unicode codepoints in `input_values[input_splits[i]:input_splits[i+1]]`
/// using `output_encoding`.
///
/// ---
///
/// Example:
///
/// ```
/// input_values = [72, 101, 108, 108, 111, 87, 111, 114, 108, 100]
/// input_splits = [0, 5, 10]
/// output_encoding = 'UTF-8'
///
/// output = ['Hello', 'World']
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input_values: A 1D tensor containing the unicode codepoints that should be encoded.
/// * input_splits: A 1D tensor specifying how the unicode codepoints should be split into strings.
/// In particular, `output[i]` is constructed by encoding the codepoints in the
/// slice `input_values[input_splits[i]:input_splits[i+1]]`.
/// * output_encoding: Unicode encoding of the output strings. Valid encodings are: `"UTF-8",
/// "UTF-16-BE", and "UTF-32-BE"`.
///
/// Optional attributes (see `Attrs`):
/// * errors: Error handling policy when there is invalid formatting found in the input.
/// The value of 'strict' will cause the operation to produce a InvalidArgument
/// error on any invalid input formatting. A value of 'replace' (the default) will
/// cause the operation to replace any invalid formatting in the input with the
/// `replacement_char` codepoint. A value of 'ignore' will cause the operation to
/// skip any invalid formatting in the input and produce no corresponding output
/// character.
/// * replacement_char: The replacement character codepoint to be used in place of any invalid
/// formatting in the input when `errors='replace'`. Any valid unicode codepoint may
/// be used. The default value is the default unicode replacement character is
/// 0xFFFD (U+65533).
///
/// Returns:
/// * `Output`: The 1-D Tensor of strings encoded from the provided unicode codepoints.
class UnicodeEncode {
 public:
  /// Optional attribute setters for UnicodeEncode
  struct Attrs {
    /// Error handling policy when there is invalid formatting found in the input.
    /// The value of 'strict' will cause the operation to produce a InvalidArgument
    /// error on any invalid input formatting. A value of 'replace' (the default) will
    /// cause the operation to replace any invalid formatting in the input with the
    /// `replacement_char` codepoint. A value of 'ignore' will cause the operation to
    /// skip any invalid formatting in the input and produce no corresponding output
    /// character.
    ///
    /// Defaults to "replace"
    TF_MUST_USE_RESULT Attrs Errors(StringPiece x) {
      Attrs ret = *this;
      ret.errors_ = x;
      return ret;
    }

    /// The replacement character codepoint to be used in place of any invalid
    /// formatting in the input when `errors='replace'`. Any valid unicode codepoint may
    /// be used. The default value is the default unicode replacement character is
    /// 0xFFFD (U+65533).
    ///
    /// Defaults to 65533
    TF_MUST_USE_RESULT Attrs ReplacementChar(int64 x) {
      Attrs ret = *this;
      ret.replacement_char_ = x;
      return ret;
    }

    StringPiece errors_ = "replace";
    int64 replacement_char_ = 65533;
  };
  UnicodeEncode(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_values, ::tensorflow::Input input_splits, StringPiece
              output_encoding);
  UnicodeEncode(const ::tensorflow::Scope& scope, ::tensorflow::Input
              input_values, ::tensorflow::Input input_splits, StringPiece
              output_encoding, const UnicodeEncode::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Errors(StringPiece x) {
    return Attrs().Errors(x);
  }
  static Attrs ReplacementChar(int64 x) {
    return Attrs().ReplacementChar(x);
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
/// * `Output`: The output tensor.
class UnsortedSegmentJoin {
 public:
  /// Optional attribute setters for UnsortedSegmentJoin
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Separator(StringPiece x) {
      Attrs ret = *this;
      ret.separator_ = x;
      return ret;
    }

    StringPiece separator_ = "";
  };
  UnsortedSegmentJoin(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    inputs, ::tensorflow::Input segment_ids,
                    ::tensorflow::Input num_segments);
  UnsortedSegmentJoin(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    inputs, ::tensorflow::Input segment_ids,
                    ::tensorflow::Input num_segments, const
                    UnsortedSegmentJoin::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Separator(StringPiece x) {
    return Attrs().Separator(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_STRING_OPS_INTERNAL_H_
