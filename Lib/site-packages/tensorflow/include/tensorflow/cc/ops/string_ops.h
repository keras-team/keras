// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_STRING_OPS_H_
#define TENSORFLOW_CC_OPS_STRING_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup string_ops String Ops
/// @{

/// Converts each entry in the given tensor to strings.
///
/// Supports many numeric types and boolean.
///
/// For Unicode, see the
/// [https://www.tensorflow.org/tutorials/representation/unicode](Working with Unicode text)
/// tutorial.
///
/// Examples:
///
/// >>> tf.strings.as_string([3, 2])
/// <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'3', b'2'], dtype=object)>
/// >>> tf.strings.as_string([3.1415926, 2.71828], precision=2).numpy()
/// array([b'3.14', b'2.72'], dtype=object)
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * precision: The post-decimal precision to use for floating point numbers.
/// Only used if precision > -1.
/// * scientific: Use scientific notation for floating point numbers.
/// * shortest: Use shortest representation (either scientific or standard) for
/// floating point numbers.
/// * width: Pad pre-decimal numbers to this width.
/// Applies to both floating point and integer numbers.
/// Only used if width > -1.
/// * fill: The value to pad if width > -1.  If empty, pads with spaces.
/// Another typical value is '0'.  String cannot be longer than 1 character.
///
/// Returns:
/// * `Output`: The output tensor.
class AsString {
 public:
  /// Optional attribute setters for AsString
  struct Attrs {
    /// The post-decimal precision to use for floating point numbers.
    /// Only used if precision > -1.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Precision(int64 x) {
      Attrs ret = *this;
      ret.precision_ = x;
      return ret;
    }

    /// Use scientific notation for floating point numbers.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Scientific(bool x) {
      Attrs ret = *this;
      ret.scientific_ = x;
      return ret;
    }

    /// Use shortest representation (either scientific or standard) for
    /// floating point numbers.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Shortest(bool x) {
      Attrs ret = *this;
      ret.shortest_ = x;
      return ret;
    }

    /// Pad pre-decimal numbers to this width.
    /// Applies to both floating point and integer numbers.
    /// Only used if width > -1.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Width(int64 x) {
      Attrs ret = *this;
      ret.width_ = x;
      return ret;
    }

    /// The value to pad if width > -1.  If empty, pads with spaces.
    /// Another typical value is '0'.  String cannot be longer than 1 character.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Fill(StringPiece x) {
      Attrs ret = *this;
      ret.fill_ = x;
      return ret;
    }

    int64 precision_ = -1;
    bool scientific_ = false;
    bool shortest_ = false;
    int64 width_ = -1;
    StringPiece fill_ = "";
  };
  AsString(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  AsString(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
         AsString::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Precision(int64 x) {
    return Attrs().Precision(x);
  }
  static Attrs Scientific(bool x) {
    return Attrs().Scientific(x);
  }
  static Attrs Shortest(bool x) {
    return Attrs().Shortest(x);
  }
  static Attrs Width(int64 x) {
    return Attrs().Width(x);
  }
  static Attrs Fill(StringPiece x) {
    return Attrs().Fill(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Decode web-safe base64-encoded strings.
///
/// Input may or may not have padding at the end. See
/// [EncodeBase64](https://www.tensorflow.org/api_docs/python/tf/io/encode_base64)
/// for padding. Web-safe means that input must use - and _ instead of + and /.
///
/// Args:
/// * scope: A Scope object
/// * input: Base64 strings to decode.
///
/// Returns:
/// * `Output`: Decoded strings.
class DecodeBase64 {
 public:
  DecodeBase64(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Encode strings into web-safe base64 format.
///
/// Refer to [this article](https://en.wikipedia.org/wiki/Base64) for more information on
/// base64 format. Base64 strings may have padding with '=' at the
/// end so that the encoded has length multiple of 4. See Padding section of the
/// link above.
///
/// Web-safe means that the encoder uses - and _ instead of + and /.
///
/// Args:
/// * scope: A Scope object
/// * input: Strings to be encoded.
///
/// Optional attributes (see `Attrs`):
/// * pad: Bool whether padding is applied at the ends.
///
/// Returns:
/// * `Output`: Input strings encoded in base64.
class EncodeBase64 {
 public:
  /// Optional attribute setters for EncodeBase64
  struct Attrs {
    /// Bool whether padding is applied at the ends.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Pad(bool x) {
      Attrs ret = *this;
      ret.pad_ = x;
      return ret;
    }

    bool pad_ = false;
  };
  EncodeBase64(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  EncodeBase64(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
             EncodeBase64::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Pad(bool x) {
    return Attrs().Pad(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Joins a string Tensor across the given dimensions.
///
/// Computes the string join across dimensions in the given string Tensor of shape
/// `[\\(d_0, d_1, ..., d_{n-1}\\)]`.  Returns a new Tensor created by joining the input
/// strings with the given separator (default: empty string).  Negative indices are
/// counted backwards from the end, with `-1` being equivalent to `n - 1`.  If
/// indices are not specified, joins across all dimensions beginning from `n - 1`
/// through `0`.
///
/// For example:
///
/// ```python
/// # tensor `a` is [["a", "b"], ["c", "d"]]
/// tf.reduce_join(a, 0) ==> ["ac", "bd"]
/// tf.reduce_join(a, 1) ==> ["ab", "cd"]
/// tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
/// tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
/// tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
/// tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
/// tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
/// tf.reduce_join(a, [0, 1]) ==> "acbd"
/// tf.reduce_join(a, [1, 0]) ==> "abcd"
/// tf.reduce_join(a, []) ==> [["a", "b"], ["c", "d"]]
/// tf.reduce_join(a) = tf.reduce_join(a, [1, 0]) ==> "abcd"
/// ```
///
/// Args:
/// * scope: A Scope object
/// * inputs: The input to be joined.  All reduced indices must have non-zero size.
/// * reduction_indices: The dimensions to reduce over.  Dimensions are reduced in the
/// order specified.  Omitting `reduction_indices` is equivalent to passing
/// `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
///
/// Optional attributes (see `Attrs`):
/// * keep_dims: If `True`, retain reduced dimensions with length `1`.
/// * separator: The separator to use when joining.
///
/// Returns:
/// * `Output`: Has shape equal to that of the input with reduced dimensions removed or
/// set to `1` depending on `keep_dims`.
class ReduceJoin {
 public:
  /// Optional attribute setters for ReduceJoin
  struct Attrs {
    /// If `True`, retain reduced dimensions with length `1`.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs KeepDims(bool x) {
      Attrs ret = *this;
      ret.keep_dims_ = x;
      return ret;
    }

    /// The separator to use when joining.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Separator(StringPiece x) {
      Attrs ret = *this;
      ret.separator_ = x;
      return ret;
    }

    bool keep_dims_ = false;
    StringPiece separator_ = "";
  };
  ReduceJoin(const ::tensorflow::Scope& scope, ::tensorflow::Input inputs,
           ::tensorflow::Input reduction_indices);
  ReduceJoin(const ::tensorflow::Scope& scope, ::tensorflow::Input inputs,
           ::tensorflow::Input reduction_indices, const ReduceJoin::Attrs&
           attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs KeepDims(bool x) {
    return Attrs().KeepDims(x);
  }
  static Attrs Separator(StringPiece x) {
    return Attrs().Separator(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Check if the input matches the regex pattern.
///
/// The input is a string tensor of any shape. The pattern is a scalar
/// string tensor which is applied to every element of the input tensor.
/// The boolean values (True or False) of the output tensor indicate
/// if the input matches the regex pattern provided.
///
/// The pattern follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// Examples:
///
/// >>> tf.strings.regex_full_match(["TF lib", "lib TF"], ".*lib$")
/// <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>
/// >>> tf.strings.regex_full_match(["TF lib", "lib TF"], ".*TF$")
/// <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  True])>
///
/// Args:
/// * scope: A Scope object
/// * input: A string tensor of the text to be processed.
/// * pattern: A scalar string tensor containing the regular expression to match the input.
///
/// Returns:
/// * `Output`: A bool tensor with the same shape as `input`.
class RegexFullMatch {
 public:
  RegexFullMatch(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
               ::tensorflow::Input pattern);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Replaces matches of the `pattern` regular expression in `input` with the
/// replacement string provided in `rewrite`.
///
/// It follows the re2 syntax (https://github.com/google/re2/wiki/Syntax)
///
/// Args:
/// * scope: A Scope object
/// * input: The text to be processed.
/// * pattern: The regular expression to be matched in the `input` strings.
/// * rewrite: The rewrite string to be substituted for the `pattern` expression where it is
/// matched in the `input` strings.
///
/// Optional attributes (see `Attrs`):
/// * replace_global: If True, the replacement is global (that is, all matches of the `pattern` regular
/// expression in each input string are rewritten), otherwise the `rewrite`
/// substitution is only made for the first `pattern` match.
///
/// Returns:
/// * `Output`: The text after applying pattern match and rewrite substitution.
class RegexReplace {
 public:
  /// Optional attribute setters for RegexReplace
  struct Attrs {
    /// If True, the replacement is global (that is, all matches of the `pattern` regular
    /// expression in each input string are rewritten), otherwise the `rewrite`
    /// substitution is only made for the first `pattern` match.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ReplaceGlobal(bool x) {
      Attrs ret = *this;
      ret.replace_global_ = x;
      return ret;
    }

    bool replace_global_ = true;
  };
  RegexReplace(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
             ::tensorflow::Input pattern, ::tensorflow::Input rewrite);
  RegexReplace(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
             ::tensorflow::Input pattern, ::tensorflow::Input rewrite, const
             RegexReplace::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs ReplaceGlobal(bool x) {
    return Attrs().ReplaceGlobal(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Formats a string template using a list of tensors.
///
/// Formats a string template using a list of tensors, pretty-printing tensor summaries.
///
/// Args:
/// * scope: A Scope object
/// * inputs: The list of tensors to format into the placeholder string.
///
/// Optional attributes (see `Attrs`):
/// * template_: A string, the template to format tensor summaries into.
/// * placeholder: A string, at each placeholder in the template a subsequent tensor summary will be inserted.
/// * summarize: When formatting the tensor summaries print the first and last summarize entries of each tensor dimension.
///
/// Returns:
/// * `Output`: = The resulting string scalar.
class StringFormat {
 public:
  /// Optional attribute setters for StringFormat
  struct Attrs {
    /// A string, the template to format tensor summaries into.
    ///
    /// Defaults to "%s"
    TF_MUST_USE_RESULT Attrs Template(StringPiece x) {
      Attrs ret = *this;
      ret.template_ = x;
      return ret;
    }

    /// A string, at each placeholder in the template a subsequent tensor summary will be inserted.
    ///
    /// Defaults to "%s"
    TF_MUST_USE_RESULT Attrs Placeholder(StringPiece x) {
      Attrs ret = *this;
      ret.placeholder_ = x;
      return ret;
    }

    /// When formatting the tensor summaries print the first and last summarize entries of each tensor dimension.
    ///
    /// Defaults to 3
    TF_MUST_USE_RESULT Attrs Summarize(int64 x) {
      Attrs ret = *this;
      ret.summarize_ = x;
      return ret;
    }

    StringPiece template_ = "%s";
    StringPiece placeholder_ = "%s";
    int64 summarize_ = 3;
  };
  StringFormat(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs);
  StringFormat(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs,
             const StringFormat::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Template(StringPiece x) {
    return Attrs().Template(x);
  }
  static Attrs Placeholder(StringPiece x) {
    return Attrs().Placeholder(x);
  }
  static Attrs Summarize(int64 x) {
    return Attrs().Summarize(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Joins the strings in the given list of string tensors into one tensor;
///
/// with the given separator (default is an empty separator).
///
/// Examples:
///
/// >>> s = ["hello", "world", "tensorflow"]
/// >>> tf.strings.join(s, " ")
/// <tf.Tensor: shape=(), dtype=string, numpy=b'hello world tensorflow'>
///
/// Args:
/// * scope: A Scope object
/// * inputs: A list of string tensors.  The tensors must all have the same shape,
/// or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
/// of non-scalar inputs.
///
/// Optional attributes (see `Attrs`):
/// * separator: string, an optional join separator.
///
/// Returns:
/// * `Output`: The output tensor.
class StringJoin {
 public:
  /// Optional attribute setters for StringJoin
  struct Attrs {
    /// string, an optional join separator.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Separator(StringPiece x) {
      Attrs ret = *this;
      ret.separator_ = x;
      return ret;
    }

    StringPiece separator_ = "";
  };
  StringJoin(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs);
  StringJoin(const ::tensorflow::Scope& scope, ::tensorflow::InputList inputs,
           const StringJoin::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Separator(StringPiece x) {
    return Attrs().Separator(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// String lengths of `input`.
///
/// Computes the length of each string given in the input tensor.
///
/// >>> strings = tf.constant(['Hello','TensorFlow', '\U0001F642'])
/// >>> tf.strings.length(strings).numpy() # default counts bytes
/// array([ 5, 10, 4], dtype=int32)
/// >>> tf.strings.length(strings, unit="UTF8_CHAR").numpy()
/// array([ 5, 10, 1], dtype=int32)
///
///
/// Args:
/// * scope: A Scope object
/// * input: The strings for which to compute the length for each element.
///
/// Optional attributes (see `Attrs`):
/// * unit: The unit that is counted to compute string length.  One of: `"BYTE"` (for
/// the number of bytes in each string) or `"UTF8_CHAR"` (for the number of UTF-8
/// encoded Unicode code points in each string).  Results are undefined
/// if `unit=UTF8_CHAR` and the `input` strings do not contain structurally
/// valid UTF-8.
///
/// Returns:
/// * `Output`: Integer tensor that has the same shape as `input`. The output contains the
/// element-wise string lengths of `input`.
class StringLength {
 public:
  /// Optional attribute setters for StringLength
  struct Attrs {
    /// The unit that is counted to compute string length.  One of: `"BYTE"` (for
    /// the number of bytes in each string) or `"UTF8_CHAR"` (for the number of UTF-8
    /// encoded Unicode code points in each string).  Results are undefined
    /// if `unit=UTF8_CHAR` and the `input` strings do not contain structurally
    /// valid UTF-8.
    ///
    /// Defaults to "BYTE"
    TF_MUST_USE_RESULT Attrs Unit(StringPiece x) {
      Attrs ret = *this;
      ret.unit_ = x;
      return ret;
    }

    StringPiece unit_ = "BYTE";
  };
  StringLength(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  StringLength(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
             StringLength::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Unit(StringPiece x) {
    return Attrs().Unit(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Converts all uppercase characters into their respective lowercase replacements.
///
/// Example:
///
/// >>> tf.strings.lower("CamelCase string and ALL CAPS")
/// <tf.Tensor: shape=(), dtype=string, numpy=b'camelcase string and all caps'>
///
///
/// Args:
/// * scope: A Scope object
/// * input: The input to be lower-cased.
///
/// Optional attributes (see `Attrs`):
/// * encoding: Character encoding of `input`. Allowed values are '' and 'utf-8'.
/// Value '' is interpreted as ASCII.
///
/// Returns:
/// * `Output`: The output tensor.
class StringLower {
 public:
  /// Optional attribute setters for StringLower
  struct Attrs {
    /// Character encoding of `input`. Allowed values are '' and 'utf-8'.
    /// Value '' is interpreted as ASCII.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Encoding(StringPiece x) {
      Attrs ret = *this;
      ret.encoding_ = x;
      return ret;
    }

    StringPiece encoding_ = "";
  };
  StringLower(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  StringLower(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
            StringLower::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Encoding(StringPiece x) {
    return Attrs().Encoding(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Creates ngrams from ragged string data.
///
/// This op accepts a ragged tensor with 1 ragged dimension containing only
/// strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
/// of that string, joined along the innermost axis.
///
/// Args:
/// * scope: A Scope object
/// * data: The values tensor of the ragged string tensor to make ngrams out of. Must be a
/// 1D string tensor.
/// * data_splits: The splits tensor of the ragged string tensor to make ngrams out of.
/// * separator: The string to append between elements of the token. Use "" for no separator.
/// * ngram_widths: The sizes of the ngrams to create.
/// * left_pad: The string to use to pad the left side of the ngram sequence. Only used if
/// pad_width != 0.
/// * right_pad: The string to use to pad the right side of the ngram sequence. Only used if
/// pad_width != 0.
/// * pad_width: The number of padding elements to add to each side of each
/// sequence. Note that padding will never be greater than 'ngram_widths'-1
/// regardless of this value. If `pad_width=-1`, then add `max(ngram_widths)-1`
/// elements.
///
/// Returns:
/// * `Output` ngrams: The values tensor of the output ngrams ragged tensor.
/// * `Output` ngrams_splits: The splits tensor of the output ngrams ragged tensor.
class StringNGrams {
 public:
  StringNGrams(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
             ::tensorflow::Input data_splits, StringPiece separator, const
             gtl::ArraySlice<int>& ngram_widths, StringPiece left_pad,
             StringPiece right_pad, int64 pad_width, bool
             preserve_short_sequences);

  Operation operation;
  ::tensorflow::Output ngrams;
  ::tensorflow::Output ngrams_splits;
};

/// Split elements of `input` based on `delimiter` into a `SparseTensor`.
///
/// Let N be the size of source (typically N will be the batch size). Split each
/// element of `input` based on `delimiter` and return a `SparseTensor`
/// containing the splitted tokens. Empty tokens are ignored.
///
/// `delimiter` can be empty, or a string of split characters. If `delimiter` is an
///  empty string, each element of `input` is split into individual single-byte
///  character strings, including splitting of UTF-8 multibyte sequences. Otherwise
///  every character of `delimiter` is a potential split point.
///
/// For example:
///   N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
///   will be
///
///   indices = [0, 0;
///              0, 1;
///              1, 0;
///              1, 1;
///              1, 2]
///   shape = [2, 3]
///   values = ['hello', 'world', 'a', 'b', 'c']
///
/// Args:
/// * scope: A Scope object
/// * input: 1-D. Strings to split.
/// * delimiter: 0-D. Delimiter characters (bytes), or empty string.
///
/// Optional attributes (see `Attrs`):
/// * skip_empty: A `bool`. If `True`, skip the empty strings from the result.
///
/// Returns:
/// * `Output` indices: A dense matrix of int64 representing the indices of the sparse tensor.
/// * `Output` values: A vector of strings corresponding to the splited values.
/// * `Output` shape: a length-2 vector of int64 representing the shape of the sparse
/// tensor, where the first value is N and the second value is the maximum number
/// of tokens in a single input entry.
class StringSplit {
 public:
  /// Optional attribute setters for StringSplit
  struct Attrs {
    /// A `bool`. If `True`, skip the empty strings from the result.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs SkipEmpty(bool x) {
      Attrs ret = *this;
      ret.skip_empty_ = x;
      return ret;
    }

    bool skip_empty_ = true;
  };
  StringSplit(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
            ::tensorflow::Input delimiter);
  StringSplit(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
            ::tensorflow::Input delimiter, const StringSplit::Attrs& attrs);

  static Attrs SkipEmpty(bool x) {
    return Attrs().SkipEmpty(x);
  }

  Operation operation;
  ::tensorflow::Output indices;
  ::tensorflow::Output values;
  ::tensorflow::Output shape;
};

/// Split elements of `source` based on `sep` into a `SparseTensor`.
///
/// Let N be the size of source (typically N will be the batch size). Split each
/// element of `source` based on `sep` and return a `SparseTensor`
/// containing the split tokens. Empty tokens are ignored.
///
/// For example, N = 2, source[0] is 'hello world' and source[1] is 'a b c',
/// then the output will be
/// ```
/// st.indices = [0, 0;
///               0, 1;
///               1, 0;
///               1, 1;
///               1, 2]
/// st.shape = [2, 3]
/// st.values = ['hello', 'world', 'a', 'b', 'c']
/// ```
///
/// If `sep` is given, consecutive delimiters are not grouped together and are
/// deemed to delimit empty strings. For example, source of `"1<>2<><>3"` and
/// sep of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty
/// string, consecutive whitespace are regarded as a single separator, and the
/// result will contain no empty strings at the startor end if the string has
/// leading or trailing whitespace.
///
/// Note that the above mentioned behavior matches python's str.split.
///
/// Args:
/// * scope: A Scope object
/// * input: `1-D` string `Tensor`, the strings to split.
/// * sep: `0-D` string `Tensor`, the delimiter character.
///
/// Optional attributes (see `Attrs`):
/// * maxsplit: An `int`. If `maxsplit > 0`, limit of the split of the result.
///
/// Returns:
/// * `Output` indices
/// * `Output` values
/// * `Output` shape
class StringSplitV2 {
 public:
  /// Optional attribute setters for StringSplitV2
  struct Attrs {
    /// An `int`. If `maxsplit > 0`, limit of the split of the result.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Maxsplit(int64 x) {
      Attrs ret = *this;
      ret.maxsplit_ = x;
      return ret;
    }

    int64 maxsplit_ = -1;
  };
  StringSplitV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input sep);
  StringSplitV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input sep, const StringSplitV2::Attrs& attrs);

  static Attrs Maxsplit(int64 x) {
    return Attrs().Maxsplit(x);
  }

  Operation operation;
  ::tensorflow::Output indices;
  ::tensorflow::Output values;
  ::tensorflow::Output shape;
};

/// Strip leading and trailing whitespaces from the Tensor.
///
/// Examples:
///
/// >>> tf.strings.strip(["\nTensorFlow", "     The python library    "]).numpy()
/// array([b'TensorFlow', b'The python library'], dtype=object)
///
/// Args:
/// * scope: A Scope object
/// * input: A string `Tensor` of any shape.
///
/// Returns:
/// * `Output`: A string `Tensor` of the same shape as the input.
class StringStrip {
 public:
  StringStrip(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Converts each string in the input Tensor to its hash mod by a number of buckets.
///
/// The hash function is deterministic on the content of the string within the
/// process.
///
/// Note that the hash function may change from time to time.
/// This functionality will be deprecated and it's recommended to use
/// `tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.
///
/// Args:
/// * scope: A Scope object
/// * num_buckets: The number of buckets.
///
/// Returns:
/// * `Output`: A Tensor of the same shape as the input `string_tensor`.
class StringToHashBucket {
 public:
  StringToHashBucket(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   string_tensor, int64 num_buckets);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Converts each string in the input Tensor to its hash mod by a number of buckets.
///
/// The hash function is deterministic on the content of the string within the
/// process and will never change. However, it is not suitable for cryptography.
/// This function may be used when CPU time is scarce and inputs are trusted or
/// unimportant. There is a risk of adversaries constructing inputs that all hash
/// to the same bucket. To prevent this problem, use a strong hash function with
/// `tf.string_to_hash_bucket_strong`.
///
/// Examples:
///
/// >>> tf.strings.to_hash_bucket_fast(["Hello", "TensorFlow", "2.x"], 3).numpy()
/// array([0, 2, 2])
///
/// Args:
/// * scope: A Scope object
/// * input: The strings to assign a hash bucket.
/// * num_buckets: The number of buckets.
///
/// Returns:
/// * `Output`: A Tensor of the same shape as the input `string_tensor`.
class StringToHashBucketFast {
 public:
  StringToHashBucketFast(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       input, int64 num_buckets);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Converts each string in the input Tensor to its hash mod by a number of buckets.
///
/// The hash function is deterministic on the content of the string within the
/// process. The hash function is a keyed hash function, where attribute `key`
/// defines the key of the hash function. `key` is an array of 2 elements.
///
/// A strong hash is important when inputs may be malicious, e.g. URLs with
/// additional components. Adversaries could try to make their inputs hash to the
/// same bucket for a denial-of-service attack or to skew the results. A strong
/// hash can be used to make it difficult to find inputs with a skewed hash value
/// distribution over buckets. This requires that the hash function is
/// seeded by a high-entropy (random) "key" unknown to the adversary.
///
/// The additional robustness comes at a cost of roughly 4x higher compute
/// time than `tf.string_to_hash_bucket_fast`.
///
/// Examples:
///
/// >>> tf.strings.to_hash_bucket_strong(["Hello", "TF"], 3, [1, 2]).numpy()
/// array([2, 0])
///
/// Args:
/// * scope: A Scope object
/// * input: The strings to assign a hash bucket.
/// * num_buckets: The number of buckets.
/// * key: The key used to seed the hash function, passed as a list of two uint64
/// elements.
///
/// Returns:
/// * `Output`: A Tensor of the same shape as the input `string_tensor`.
class StringToHashBucketStrong {
 public:
  StringToHashBucketStrong(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input, int64 num_buckets, const gtl::ArraySlice<int>&
                         key);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Converts all lowercase characters into their respective uppercase replacements.
///
/// Example:
///
/// >>> tf.strings.upper("CamelCase string and ALL CAPS")
/// <tf.Tensor: shape=(), dtype=string, numpy=b'CAMELCASE STRING AND ALL CAPS'>
///
///
/// Args:
/// * scope: A Scope object
/// * input: The input to be upper-cased.
///
/// Optional attributes (see `Attrs`):
/// * encoding: Character encoding of `input`. Allowed values are '' and 'utf-8'.
/// Value '' is interpreted as ASCII.
///
/// Returns:
/// * `Output`: The output tensor.
class StringUpper {
 public:
  /// Optional attribute setters for StringUpper
  struct Attrs {
    /// Character encoding of `input`. Allowed values are '' and 'utf-8'.
    /// Value '' is interpreted as ASCII.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Encoding(StringPiece x) {
      Attrs ret = *this;
      ret.encoding_ = x;
      return ret;
    }

    StringPiece encoding_ = "";
  };
  StringUpper(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  StringUpper(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
            StringUpper::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Encoding(StringPiece x) {
    return Attrs().Encoding(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Return substrings from `Tensor` of strings.
///
/// For each string in the input `Tensor`, creates a substring starting at index
/// `pos` with a total length of `len`.
///
/// If `len` defines a substring that would extend beyond the length of the input
/// string, or if `len` is negative, then as many characters as possible are used.
///
/// A negative `pos` indicates distance within the string backwards from the end.
///
/// If `pos` specifies an index which is out of range for any of the input strings,
/// then an `InvalidArgumentError` is thrown.
///
/// `pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
/// Op creation.
///
/// *NOTE*: `Substr` supports broadcasting up to two dimensions. More about
/// broadcasting
/// [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///
/// ---
///
/// Examples
///
/// Using scalar `pos` and `len`:
///
/// ```python
/// input = [b'Hello', b'World']
/// position = 1
/// length = 3
///
/// output = [b'ell', b'orl']
/// ```
///
/// Using `pos` and `len` with same shape as `input`:
///
/// ```python
/// input = [[b'ten', b'eleven', b'twelve'],
///          [b'thirteen', b'fourteen', b'fifteen'],
///          [b'sixteen', b'seventeen', b'eighteen']]
/// position = [[1, 2, 3],
///             [1, 2, 3],
///             [1, 2, 3]]
/// length =   [[2, 3, 4],
///             [4, 3, 2],
///             [5, 5, 5]]
///
/// output = [[b'en', b'eve', b'lve'],
///           [b'hirt', b'urt', b'te'],
///           [b'ixtee', b'vente', b'hteen']]
/// ```
///
/// Broadcasting `pos` and `len` onto `input`:
///
/// ```
/// input = [[b'ten', b'eleven', b'twelve'],
///          [b'thirteen', b'fourteen', b'fifteen'],
///          [b'sixteen', b'seventeen', b'eighteen'],
///          [b'nineteen', b'twenty', b'twentyone']]
/// position = [1, 2, 3]
/// length =   [1, 2, 3]
///
/// output = [[b'e', b'ev', b'lve'],
///           [b'h', b'ur', b'tee'],
///           [b'i', b've', b'hte'],
///           [b'i', b'en', b'nty']]
/// ```
///
/// Broadcasting `input` onto `pos` and `len`:
///
/// ```
/// input = b'thirteen'
/// position = [1, 5, 7]
/// length =   [3, 2, 1]
///
/// output = [b'hir', b'ee', b'n']
/// ```
///
/// Raises:
///
///   * `ValueError`: If the first argument cannot be converted to a
///      Tensor of `dtype string`.
///   * `InvalidArgumentError`: If indices are out of range.
///   * `ValueError`: If `pos` and `len` are not the same shape.
///
///
/// Args:
/// * scope: A Scope object
/// * input: Tensor of strings
/// * pos: Scalar defining the position of first character in each substring
/// * len: Scalar defining the number of characters to include in each substring
///
/// Optional attributes (see `Attrs`):
/// * unit: The unit that is used to create the substring.  One of: `"BYTE"` (for
/// defining position and length by bytes) or `"UTF8_CHAR"` (for the UTF-8
/// encoded Unicode code points).  The default is `"BYTE"`. Results are undefined if
/// `unit=UTF8_CHAR` and the `input` strings do not contain structurally valid
/// UTF-8.
///
/// Returns:
/// * `Output`: Tensor of substrings
class Substr {
 public:
  /// Optional attribute setters for Substr
  struct Attrs {
    /// The unit that is used to create the substring.  One of: `"BYTE"` (for
    /// defining position and length by bytes) or `"UTF8_CHAR"` (for the UTF-8
    /// encoded Unicode code points).  The default is `"BYTE"`. Results are undefined if
    /// `unit=UTF8_CHAR` and the `input` strings do not contain structurally valid
    /// UTF-8.
    ///
    /// Defaults to "BYTE"
    TF_MUST_USE_RESULT Attrs Unit(StringPiece x) {
      Attrs ret = *this;
      ret.unit_ = x;
      return ret;
    }

    StringPiece unit_ = "BYTE";
  };
  Substr(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
       ::tensorflow::Input pos, ::tensorflow::Input len);
  Substr(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
       ::tensorflow::Input pos, ::tensorflow::Input len, const Substr::Attrs&
       attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Unit(StringPiece x) {
    return Attrs().Unit(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Determine the script codes of a given tensor of Unicode integer code points.
///
/// This operation converts Unicode code points to script codes corresponding to
/// each code point. Script codes correspond to International Components for
/// Unicode (ICU) UScriptCode values.
///
/// See
/// [ICU project docs](http://icu-project.org/apiref/icu4c/uscript_8h.html)
/// for more details on script codes.
///
/// For an example, see the unicode strings guide on [unicode scripts]
/// (https://www.tensorflow.org/tutorials/load_data/unicode#representing_unicode).
///
/// Returns -1 (USCRIPT_INVALID_CODE) for invalid codepoints. Output shape will
/// match input shape.
///
/// Examples:
///
/// >>> tf.strings.unicode_script([1, 31, 38])
/// <tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 0, 0], dtype=int32)>
///
/// Args:
/// * scope: A Scope object
/// * input: A Tensor of int32 Unicode code points.
///
/// Returns:
/// * `Output`: A Tensor of int32 script codes corresponding to each input code point.
class UnicodeScript {
 public:
  UnicodeScript(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Transcode the input text from a source encoding to a destination encoding.
///
/// The input is a string tensor of any shape. The output is a string tensor of
/// the same shape containing the transcoded strings. Output strings are always
/// valid unicode. If the input contains invalid encoding positions, the
/// `errors` attribute sets the policy for how to deal with them. If the default
/// error-handling policy is used, invalid formatting will be substituted in the
/// output by the `replacement_char`. If the errors policy is to `ignore`, any
/// invalid encoding positions in the input are skipped and not included in the
/// output. If it set to `strict` then any invalid formatting will result in an
/// InvalidArgument error.
///
/// This operation can be used with `output_encoding = input_encoding` to enforce
/// correct formatting for inputs even if they are already in the desired encoding.
///
/// If the input is prefixed by a Byte Order Mark needed to determine encoding
/// (e.g. if the encoding is UTF-16 and the BOM indicates big-endian), then that
/// BOM will be consumed and not emitted into the output. If the input encoding
/// is marked with an explicit endianness (e.g. UTF-16-BE), then the BOM is
/// interpreted as a non-breaking-space and is preserved in the output (including
/// always for UTF-8).
///
/// The end result is that if the input is marked as an explicit endianness the
/// transcoding is faithful to all codepoints in the source. If it is not marked
/// with an explicit endianness, the BOM is not considered part of the string itself
/// but as metadata, and so is not preserved in the output.
///
/// Examples:
///
/// >>> tf.strings.unicode_transcode(["Hello", "TensorFlow", "2.x"], "UTF-8", "UTF-16-BE")
/// <tf.Tensor: shape=(3,), dtype=string, numpy=
/// array([b'\x00H\x00e\x00l\x00l\x00o',
///        b'\x00T\x00e\x00n\x00s\x00o\x00r\x00F\x00l\x00o\x00w',
///        b'\x002\x00.\x00x'], dtype=object)>
/// >>> tf.strings.unicode_transcode(["A", "B", "C"], "US ASCII", "UTF-8").numpy()
/// array([b'A', b'B', b'C'], dtype=object)
///
/// Args:
/// * scope: A Scope object
/// * input: The text to be processed. Can have any shape.
/// * input_encoding: Text encoding of the input strings. This is any of the encodings supported
/// by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
/// * output_encoding: The unicode encoding to use in the output. Must be one of
/// `"UTF-8", "UTF-16-BE", "UTF-32-BE"`. Multi-byte encodings will be big-endian.
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
///
/// Note that for UTF-8, passing a replacement character expressible in 1 byte, such
/// as ' ', will preserve string alignment to the source since invalid bytes will be
/// replaced with a 1-byte replacement. For UTF-16-BE and UTF-16-LE, any 1 or 2 byte
/// replacement character will preserve byte alignment to the source.
/// * replace_control_characters: Whether to replace the C0 control characters (00-1F) with the
/// `replacement_char`. Default is false.
///
/// Returns:
/// * `Output`: A string tensor containing unicode text encoded using `output_encoding`.
class UnicodeTranscode {
 public:
  /// Optional attribute setters for UnicodeTranscode
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
    /// Note that for UTF-8, passing a replacement character expressible in 1 byte, such
    /// as ' ', will preserve string alignment to the source since invalid bytes will be
    /// replaced with a 1-byte replacement. For UTF-16-BE and UTF-16-LE, any 1 or 2 byte
    /// replacement character will preserve byte alignment to the source.
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

    StringPiece errors_ = "replace";
    int64 replacement_char_ = 65533;
    bool replace_control_characters_ = false;
  };
  UnicodeTranscode(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 StringPiece input_encoding, StringPiece output_encoding);
  UnicodeTranscode(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 StringPiece input_encoding, StringPiece output_encoding, const
                 UnicodeTranscode::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Errors(StringPiece x) {
    return Attrs().Errors(x);
  }
  static Attrs ReplacementChar(int64 x) {
    return Attrs().ReplacementChar(x);
  }
  static Attrs ReplaceControlCharacters(bool x) {
    return Attrs().ReplaceControlCharacters(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_STRING_OPS_H_
