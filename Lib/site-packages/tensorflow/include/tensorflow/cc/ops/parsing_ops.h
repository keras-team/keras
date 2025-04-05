// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_PARSING_OPS_H_
#define TENSORFLOW_CC_OPS_PARSING_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup parsing_ops Parsing Ops
/// @{

/// Convert CSV records to tensors. Each column maps to one tensor.
///
/// RFC 4180 format is expected for the CSV records.
/// (https://tools.ietf.org/html/rfc4180)
/// Note that we allow leading and trailing spaces with int or float field.
///
/// Args:
/// * scope: A Scope object
/// * records: Each string is a record/row in the csv and all records should have
/// the same format.
/// * record_defaults: One tensor per column of the input record, with either a
/// scalar default value for that column or an empty vector if the column is
/// required.
///
/// Optional attributes (see `Attrs`):
/// * field_delim: char delimiter to separate fields in a record.
/// * use_quote_delim: If false, treats double quotation marks as regular
/// characters inside of the string fields (ignoring RFC 4180, Section 2,
/// Bullet 5).
/// * na_value: Additional string to recognize as NA/NaN.
///
/// Returns:
/// * `OutputList`: Each tensor will have the same shape as records.
class DecodeCSV {
 public:
  /// Optional attribute setters for DecodeCSV
  struct Attrs {
    /// char delimiter to separate fields in a record.
    ///
    /// Defaults to ","
    TF_MUST_USE_RESULT Attrs FieldDelim(StringPiece x) {
      Attrs ret = *this;
      ret.field_delim_ = x;
      return ret;
    }

    /// If false, treats double quotation marks as regular
    /// characters inside of the string fields (ignoring RFC 4180, Section 2,
    /// Bullet 5).
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UseQuoteDelim(bool x) {
      Attrs ret = *this;
      ret.use_quote_delim_ = x;
      return ret;
    }

    /// Additional string to recognize as NA/NaN.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs NaValue(StringPiece x) {
      Attrs ret = *this;
      ret.na_value_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs SelectCols(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.select_cols_ = x;
      return ret;
    }

    StringPiece field_delim_ = ",";
    bool use_quote_delim_ = true;
    StringPiece na_value_ = "";
    gtl::ArraySlice<int> select_cols_ = {};
  };
  DecodeCSV(const ::tensorflow::Scope& scope, ::tensorflow::Input records,
          ::tensorflow::InputList record_defaults);
  DecodeCSV(const ::tensorflow::Scope& scope, ::tensorflow::Input records,
          ::tensorflow::InputList record_defaults, const DecodeCSV::Attrs&
          attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs FieldDelim(StringPiece x) {
    return Attrs().FieldDelim(x);
  }
  static Attrs UseQuoteDelim(bool x) {
    return Attrs().UseQuoteDelim(x);
  }
  static Attrs NaValue(StringPiece x) {
    return Attrs().NaValue(x);
  }
  static Attrs SelectCols(const gtl::ArraySlice<int>& x) {
    return Attrs().SelectCols(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// Decompress strings.
///
/// This op decompresses each element of the `bytes` input `Tensor`, which
/// is assumed to be compressed using the given `compression_type`.
///
/// The `output` is a string `Tensor` of the same shape as `bytes`,
/// each element containing the decompressed data from the corresponding
/// element in `bytes`.
///
/// Args:
/// * scope: A Scope object
/// * bytes: A Tensor of string which is compressed.
///
/// Optional attributes (see `Attrs`):
/// * compression_type: A scalar containing either (i) the empty string (no
/// compression), (ii) "ZLIB", or (iii) "GZIP".
///
/// Returns:
/// * `Output`: A Tensor with the same shape as input `bytes`, uncompressed
/// from bytes.
class DecodeCompressed {
 public:
  /// Optional attribute setters for DecodeCompressed
  struct Attrs {
    /// A scalar containing either (i) the empty string (no
    /// compression), (ii) "ZLIB", or (iii) "GZIP".
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CompressionType(StringPiece x) {
      Attrs ret = *this;
      ret.compression_type_ = x;
      return ret;
    }

    StringPiece compression_type_ = "";
  };
  DecodeCompressed(const ::tensorflow::Scope& scope, ::tensorflow::Input bytes);
  DecodeCompressed(const ::tensorflow::Scope& scope, ::tensorflow::Input bytes,
                 const DecodeCompressed::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs CompressionType(StringPiece x) {
    return Attrs().CompressionType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Convert JSON-encoded Example records to binary protocol buffer strings.
///
///
/// Note: This is **not** a general purpose JSON parsing op.
///
/// This op converts JSON-serialized
/// `tf.train.Example` (created with `json_format.MessageToJson`, following the
/// [standard JSON mapping](https://developers.google.com/protocol-buffers/docs/proto3#json))
/// to a binary-serialized `tf.train.Example` (equivalent to
/// `Example.SerializeToString()`) suitable for conversion to tensors with
/// `tf.io.parse_example`.
///
/// Args:
/// * scope: A Scope object
/// * json_examples: Each string is a JSON object serialized according to the JSON
/// mapping of the Example proto.
///
/// Returns:
/// * `Output`: Each string is a binary Example protocol buffer corresponding
/// to the respective element of `json_examples`.
class DecodeJSONExample {
 public:
  DecodeJSONExample(const ::tensorflow::Scope& scope, ::tensorflow::Input
                  json_examples);
  operator ::tensorflow::Output() const { return binary_examples; }
  operator ::tensorflow::Input() const { return binary_examples; }
  ::tensorflow::Node* node() const { return binary_examples.node(); }

  Operation operation;
  ::tensorflow::Output binary_examples;
};

/// Reinterpret the bytes of a string as a vector of numbers.
///
/// Args:
/// * scope: A Scope object
/// * input_bytes: Tensor of string to be decoded.
/// * fixed_length: Length in bytes for each element of the decoded output. Must be a multiple
/// of the size of the output type.
///
/// Optional attributes (see `Attrs`):
/// * little_endian: Whether the input `input_bytes` is in little-endian order. Ignored for
/// `out_type` values that are stored in a single byte, like `uint8`
///
/// Returns:
/// * `Output`: A Tensor with one more dimension than the input `bytes`. The added dimension
/// will have size equal to the length of the elements of `bytes` divided by the
/// number of bytes to represent `out_type`.
class DecodePaddedRaw {
 public:
  /// Optional attribute setters for DecodePaddedRaw
  struct Attrs {
    /// Whether the input `input_bytes` is in little-endian order. Ignored for
    /// `out_type` values that are stored in a single byte, like `uint8`
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs LittleEndian(bool x) {
      Attrs ret = *this;
      ret.little_endian_ = x;
      return ret;
    }

    bool little_endian_ = true;
  };
  DecodePaddedRaw(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_bytes, ::tensorflow::Input fixed_length, DataType
                out_type);
  DecodePaddedRaw(const ::tensorflow::Scope& scope, ::tensorflow::Input
                input_bytes, ::tensorflow::Input fixed_length, DataType
                out_type, const DecodePaddedRaw::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs LittleEndian(bool x) {
    return Attrs().LittleEndian(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Reinterpret the bytes of a string as a vector of numbers.
///
/// Args:
/// * scope: A Scope object
/// * bytes: All the elements must have the same length.
///
/// Optional attributes (see `Attrs`):
/// * little_endian: Whether the input `bytes` are in little-endian order.
/// Ignored for `out_type` values that are stored in a single byte like
/// `uint8`.
///
/// Returns:
/// * `Output`: A Tensor with one more dimension than the input `bytes`.  The
/// added dimension will have size equal to the length of the elements
/// of `bytes` divided by the number of bytes to represent `out_type`.
class DecodeRaw {
 public:
  /// Optional attribute setters for DecodeRaw
  struct Attrs {
    /// Whether the input `bytes` are in little-endian order.
    /// Ignored for `out_type` values that are stored in a single byte like
    /// `uint8`.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs LittleEndian(bool x) {
      Attrs ret = *this;
      ret.little_endian_ = x;
      return ret;
    }

    bool little_endian_ = true;
  };
  DecodeRaw(const ::tensorflow::Scope& scope, ::tensorflow::Input bytes, DataType
          out_type);
  DecodeRaw(const ::tensorflow::Scope& scope, ::tensorflow::Input bytes, DataType
          out_type, const DecodeRaw::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs LittleEndian(bool x) {
    return Attrs().LittleEndian(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Transforms a vector of brain.Example protos (as strings) into typed tensors.
///
/// Args:
/// * scope: A Scope object
/// * serialized: A vector containing a batch of binary serialized Example protos.
/// * names: A vector containing the names of the serialized protos.
/// May contain, for example, table key (descriptive) names for the
/// corresponding serialized protos.  These are purely useful for debugging
/// purposes, and the presence of values here has no effect on the output.
/// May also be an empty vector if no names are available.
/// If non-empty, this vector must be the same length as "serialized".
/// * sparse_keys: A list of Nsparse string Tensors (scalars).
/// The keys expected in the Examples' features associated with sparse values.
/// * dense_keys: A list of Ndense string Tensors (scalars).
/// The keys expected in the Examples' features associated with dense values.
/// * dense_defaults: A list of Ndense Tensors (some may be empty).
/// dense_defaults[j] provides default values
/// when the example's feature_map lacks dense_key[j].  If an empty Tensor is
/// provided for dense_defaults[j], then the Feature dense_keys[j] is required.
/// The input type is inferred from dense_defaults[j], even when it's empty.
/// If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
/// then the shape of dense_defaults[j] must match that of dense_shapes[j].
/// If dense_shapes[j] has an undefined major dimension (variable strides dense
/// feature), dense_defaults[j] must contain a single element:
/// the padding element.
/// * sparse_types: A list of Nsparse types; the data types of data in each Feature
/// given in sparse_keys.
/// Currently the ParseExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * dense_shapes: A list of Ndense shapes; the shapes of data in each Feature
/// given in dense_keys.
/// The number of elements in the Feature corresponding to dense_key[j]
/// must always equal dense_shapes[j].NumEntries().
/// If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
/// Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
/// The dense outputs are just the inputs row-stacked by batch.
/// This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
/// the shape of the output Tensor dense_values[j] will be
/// (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
/// of elements of length D1 * .... * DN, across all minibatch entries
/// in the input.  Any minibatch entry with less than M blocks of elements of
/// length D1 * ... * DN will be padded with the corresponding default_value
/// scalar element along the second dimension.
///
/// Returns:
/// * `OutputList` sparse_indices
/// * `OutputList` sparse_values
/// * `OutputList` sparse_shapes
/// * `OutputList` dense_values
class ParseExample {
 public:
  ParseExample(const ::tensorflow::Scope& scope, ::tensorflow::Input serialized,
             ::tensorflow::Input names, ::tensorflow::InputList sparse_keys,
             ::tensorflow::InputList dense_keys, ::tensorflow::InputList
             dense_defaults, const DataTypeSlice& sparse_types, const
             gtl::ArraySlice<PartialTensorShape>& dense_shapes);

  Operation operation;
  ::tensorflow::OutputList sparse_indices;
  ::tensorflow::OutputList sparse_values;
  ::tensorflow::OutputList sparse_shapes;
  ::tensorflow::OutputList dense_values;
};

/// Transforms a vector of tf.Example protos (as strings) into typed tensors.
///
/// Args:
/// * scope: A Scope object
/// * serialized: A scalar or vector containing binary serialized Example protos.
/// * names: A tensor containing the names of the serialized protos.
/// Corresponds 1:1 with the `serialized` tensor.
/// May contain, for example, table key (descriptive) names for the
/// corresponding serialized protos.  These are purely useful for debugging
/// purposes, and the presence of values here has no effect on the output.
/// May also be an empty vector if no names are available.
/// If non-empty, this tensor must have the same shape as "serialized".
/// * sparse_keys: Vector of strings.
/// The keys expected in the Examples' features associated with sparse values.
/// * dense_keys: Vector of strings.
/// The keys expected in the Examples' features associated with dense values.
/// * ragged_keys: Vector of strings.
/// The keys expected in the Examples' features associated with ragged values.
/// * dense_defaults: A list of Tensors (some may be empty).  Corresponds 1:1 with `dense_keys`.
/// dense_defaults[j] provides default values
/// when the example's feature_map lacks dense_key[j].  If an empty Tensor is
/// provided for dense_defaults[j], then the Feature dense_keys[j] is required.
/// The input type is inferred from dense_defaults[j], even when it's empty.
/// If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
/// then the shape of dense_defaults[j] must match that of dense_shapes[j].
/// If dense_shapes[j] has an undefined major dimension (variable strides dense
/// feature), dense_defaults[j] must contain a single element:
/// the padding element.
/// * num_sparse: The number of sparse keys.
/// * sparse_types: A list of `num_sparse` types; the data types of data in each Feature
/// given in sparse_keys.
/// Currently the ParseExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * ragged_value_types: A list of `num_ragged` types; the data types of data in each Feature
/// given in ragged_keys (where `num_ragged = sparse_keys.size()`).
/// Currently the ParseExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * ragged_split_types: A list of `num_ragged` types; the data types of row_splits in each Feature
/// given in ragged_keys (where `num_ragged = sparse_keys.size()`).
/// May be DT_INT32 or DT_INT64.
/// * dense_shapes: A list of `num_dense` shapes; the shapes of data in each Feature
/// given in dense_keys (where `num_dense = dense_keys.size()`).
/// The number of elements in the Feature corresponding to dense_key[j]
/// must always equal dense_shapes[j].NumEntries().
/// If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
/// Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
/// The dense outputs are just the inputs row-stacked by batch.
/// This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
/// the shape of the output Tensor dense_values[j] will be
/// (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
/// of elements of length D1 * .... * DN, across all minibatch entries
/// in the input.  Any minibatch entry with less than M blocks of elements of
/// length D1 * ... * DN will be padded with the corresponding default_value
/// scalar element along the second dimension.
///
/// Returns:
/// * `OutputList` sparse_indices
/// * `OutputList` sparse_values
/// * `OutputList` sparse_shapes
/// * `OutputList` dense_values
/// * `OutputList` ragged_values
/// * `OutputList` ragged_row_splits
class ParseExampleV2 {
 public:
  ParseExampleV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
               serialized, ::tensorflow::Input names, ::tensorflow::Input
               sparse_keys, ::tensorflow::Input dense_keys, ::tensorflow::Input
               ragged_keys, ::tensorflow::InputList dense_defaults, int64
               num_sparse, const DataTypeSlice& sparse_types, const
               DataTypeSlice& ragged_value_types, const DataTypeSlice&
               ragged_split_types, const gtl::ArraySlice<PartialTensorShape>&
               dense_shapes);

  Operation operation;
  ::tensorflow::OutputList sparse_indices;
  ::tensorflow::OutputList sparse_values;
  ::tensorflow::OutputList sparse_shapes;
  ::tensorflow::OutputList dense_values;
  ::tensorflow::OutputList ragged_values;
  ::tensorflow::OutputList ragged_row_splits;
};

/// Transforms a vector of brain.SequenceExample protos (as strings) into typed tensors.
///
/// Args:
/// * scope: A Scope object
/// * serialized: A vector containing binary serialized SequenceExample protos.
/// * debug_name: A vector containing the names of the serialized protos.
/// May contain, for example, table key (descriptive) name for the
/// corresponding serialized proto.  This is purely useful for debugging
/// purposes, and the presence of values here has no effect on the output.
/// May also be an empty vector if no name is available.
/// * context_dense_defaults: A list of Ncontext_dense Tensors (some may be empty).
/// context_dense_defaults[j] provides default values
/// when the SequenceExample's context map lacks context_dense_key[j].
/// If an empty Tensor is provided for context_dense_defaults[j],
/// then the Feature context_dense_keys[j] is required.
/// The input type is inferred from context_dense_defaults[j], even when it's
/// empty.  If context_dense_defaults[j] is not empty, its shape must match
/// context_dense_shapes[j].
/// * feature_list_dense_missing_assumed_empty: A vector listing the
/// FeatureList keys which may be missing from the SequenceExamples.  If the
/// associated FeatureList is missing, it is treated as empty.  By default,
/// any FeatureList not listed in this vector must exist in the SequenceExamples.
/// * context_sparse_keys: A list of Ncontext_sparse string Tensors (scalars).
/// The keys expected in the Examples' features associated with context_sparse
/// values.
/// * context_dense_keys: A list of Ncontext_dense string Tensors (scalars).
/// The keys expected in the SequenceExamples' context features associated with
/// dense values.
/// * feature_list_sparse_keys: A list of Nfeature_list_sparse string Tensors
/// (scalars).  The keys expected in the FeatureLists associated with sparse
/// values.
/// * feature_list_dense_keys: A list of Nfeature_list_dense string Tensors (scalars).
/// The keys expected in the SequenceExamples' feature_lists associated
/// with lists of dense values.
///
/// Optional attributes (see `Attrs`):
/// * context_sparse_types: A list of Ncontext_sparse types; the data types of data in
/// each context Feature given in context_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * context_dense_shapes: A list of Ncontext_dense shapes; the shapes of data in
/// each context Feature given in context_dense_keys.
/// The number of elements in the Feature corresponding to context_dense_key[j]
/// must always equal context_dense_shapes[j].NumEntries().
/// The shape of context_dense_values[j] will match context_dense_shapes[j].
/// * feature_list_sparse_types: A list of Nfeature_list_sparse types; the data types
/// of data in each FeatureList given in feature_list_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * feature_list_dense_shapes: A list of Nfeature_list_dense shapes; the shapes of
/// data in each FeatureList given in feature_list_dense_keys.
/// The shape of each Feature in the FeatureList corresponding to
/// feature_list_dense_key[j] must always equal
/// feature_list_dense_shapes[j].NumEntries().
///
/// Returns:
/// * `OutputList` context_sparse_indices
/// * `OutputList` context_sparse_values
/// * `OutputList` context_sparse_shapes
/// * `OutputList` context_dense_values
/// * `OutputList` feature_list_sparse_indices
/// * `OutputList` feature_list_sparse_values
/// * `OutputList` feature_list_sparse_shapes
/// * `OutputList` feature_list_dense_values
/// * `OutputList` feature_list_dense_lengths
class ParseSequenceExample {
 public:
  /// Optional attribute setters for ParseSequenceExample
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NcontextSparse(int64 x) {
      Attrs ret = *this;
      ret.Ncontext_sparse_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NcontextDense(int64 x) {
      Attrs ret = *this;
      ret.Ncontext_dense_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NfeatureListSparse(int64 x) {
      Attrs ret = *this;
      ret.Nfeature_list_sparse_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NfeatureListDense(int64 x) {
      Attrs ret = *this;
      ret.Nfeature_list_dense_ = x;
      return ret;
    }

    /// A list of Ncontext_sparse types; the data types of data in
    /// each context Feature given in context_sparse_keys.
    /// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
    /// DT_INT64 (Int64List), and DT_STRING (BytesList).
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextSparseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.context_sparse_types_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListDenseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_dense_types_ = x;
      return ret;
    }

    /// A list of Ncontext_dense shapes; the shapes of data in
    /// each context Feature given in context_dense_keys.
    /// The number of elements in the Feature corresponding to context_dense_key[j]
    /// must always equal context_dense_shapes[j].NumEntries().
    /// The shape of context_dense_values[j] will match context_dense_shapes[j].
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.context_dense_shapes_ = x;
      return ret;
    }

    /// A list of Nfeature_list_sparse types; the data types
    /// of data in each FeatureList given in feature_list_sparse_keys.
    /// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
    /// DT_INT64 (Int64List), and DT_STRING (BytesList).
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListSparseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_sparse_types_ = x;
      return ret;
    }

    /// A list of Nfeature_list_dense shapes; the shapes of
    /// data in each FeatureList given in feature_list_dense_keys.
    /// The shape of each Feature in the FeatureList corresponding to
    /// feature_list_dense_key[j] must always equal
    /// feature_list_dense_shapes[j].NumEntries().
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.feature_list_dense_shapes_ = x;
      return ret;
    }

    int64 Ncontext_sparse_ = 0;
    int64 Ncontext_dense_ = 0;
    int64 Nfeature_list_sparse_ = 0;
    int64 Nfeature_list_dense_ = 0;
    DataTypeSlice context_sparse_types_ = {};
    DataTypeSlice feature_list_dense_types_ = {};
    gtl::ArraySlice<PartialTensorShape> context_dense_shapes_ = {};
    DataTypeSlice feature_list_sparse_types_ = {};
    gtl::ArraySlice<PartialTensorShape> feature_list_dense_shapes_ = {};
  };
  ParseSequenceExample(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     serialized, ::tensorflow::Input debug_name,
                     ::tensorflow::InputList context_dense_defaults, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     feature_list_dense_missing_assumed_empty, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     context_sparse_keys, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     context_dense_keys, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     feature_list_sparse_keys, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     feature_list_dense_keys);
  ParseSequenceExample(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     serialized, ::tensorflow::Input debug_name,
                     ::tensorflow::InputList context_dense_defaults, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     feature_list_dense_missing_assumed_empty, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     context_sparse_keys, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     context_dense_keys, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     feature_list_sparse_keys, const
                     gtl::ArraySlice<::tensorflow::tstring>&
                     feature_list_dense_keys, const
                     ParseSequenceExample::Attrs& attrs);

  static Attrs NcontextSparse(int64 x) {
    return Attrs().NcontextSparse(x);
  }
  static Attrs NcontextDense(int64 x) {
    return Attrs().NcontextDense(x);
  }
  static Attrs NfeatureListSparse(int64 x) {
    return Attrs().NfeatureListSparse(x);
  }
  static Attrs NfeatureListDense(int64 x) {
    return Attrs().NfeatureListDense(x);
  }
  static Attrs ContextSparseTypes(const DataTypeSlice& x) {
    return Attrs().ContextSparseTypes(x);
  }
  static Attrs FeatureListDenseTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListDenseTypes(x);
  }
  static Attrs ContextDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().ContextDenseShapes(x);
  }
  static Attrs FeatureListSparseTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListSparseTypes(x);
  }
  static Attrs FeatureListDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().FeatureListDenseShapes(x);
  }

  Operation operation;
  ::tensorflow::OutputList context_sparse_indices;
  ::tensorflow::OutputList context_sparse_values;
  ::tensorflow::OutputList context_sparse_shapes;
  ::tensorflow::OutputList context_dense_values;
  ::tensorflow::OutputList feature_list_sparse_indices;
  ::tensorflow::OutputList feature_list_sparse_values;
  ::tensorflow::OutputList feature_list_sparse_shapes;
  ::tensorflow::OutputList feature_list_dense_values;
  ::tensorflow::OutputList feature_list_dense_lengths;
};

/// Transforms a vector of tf.io.SequenceExample protos (as strings) into
/// typed tensors.
///
/// Args:
/// * scope: A Scope object
/// * serialized: A scalar or vector containing binary serialized SequenceExample protos.
/// * debug_name: A scalar or vector containing the names of the serialized protos.
/// May contain, for example, table key (descriptive) name for the
/// corresponding serialized proto.  This is purely useful for debugging
/// purposes, and the presence of values here has no effect on the output.
/// May also be an empty vector if no name is available.
/// * context_sparse_keys: The keys expected in the Examples' features associated with context_sparse
/// values.
/// * context_dense_keys: The keys expected in the SequenceExamples' context features associated with
/// dense values.
/// * context_ragged_keys: The keys expected in the Examples' features associated with context_ragged
/// values.
/// * feature_list_sparse_keys: The keys expected in the FeatureLists associated with sparse values.
/// * feature_list_dense_keys: The keys expected in the SequenceExamples' feature_lists associated
/// with lists of dense values.
/// * feature_list_ragged_keys: The keys expected in the FeatureLists associated with ragged values.
/// * feature_list_dense_missing_assumed_empty: A vector corresponding 1:1 with feature_list_dense_keys, indicating which
/// features may be missing from the SequenceExamples.  If the associated
/// FeatureList is missing, it is treated as empty.
/// * context_dense_defaults: A list of Ncontext_dense Tensors (some may be empty).
/// context_dense_defaults[j] provides default values
/// when the SequenceExample's context map lacks context_dense_key[j].
/// If an empty Tensor is provided for context_dense_defaults[j],
/// then the Feature context_dense_keys[j] is required.
/// The input type is inferred from context_dense_defaults[j], even when it's
/// empty.  If context_dense_defaults[j] is not empty, its shape must match
/// context_dense_shapes[j].
///
/// Optional attributes (see `Attrs`):
/// * context_sparse_types: A list of Ncontext_sparse types; the data types of data in
/// each context Feature given in context_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * context_ragged_value_types: RaggedTensor.value dtypes for the ragged context features.
/// * context_ragged_split_types: RaggedTensor.row_split dtypes for the ragged context features.
/// * context_dense_shapes: A list of Ncontext_dense shapes; the shapes of data in
/// each context Feature given in context_dense_keys.
/// The number of elements in the Feature corresponding to context_dense_key[j]
/// must always equal context_dense_shapes[j].NumEntries().
/// The shape of context_dense_values[j] will match context_dense_shapes[j].
/// * feature_list_sparse_types: A list of Nfeature_list_sparse types; the data types
/// of data in each FeatureList given in feature_list_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * feature_list_ragged_value_types: RaggedTensor.value dtypes for the ragged FeatureList features.
/// * feature_list_ragged_split_types: RaggedTensor.row_split dtypes for the ragged FeatureList features.
/// * feature_list_dense_shapes: A list of Nfeature_list_dense shapes; the shapes of
/// data in each FeatureList given in feature_list_dense_keys.
/// The shape of each Feature in the FeatureList corresponding to
/// feature_list_dense_key[j] must always equal
/// feature_list_dense_shapes[j].NumEntries().
///
/// Returns:
/// * `OutputList` context_sparse_indices
/// * `OutputList` context_sparse_values
/// * `OutputList` context_sparse_shapes
/// * `OutputList` context_dense_values
/// * `OutputList` context_ragged_values
/// * `OutputList` context_ragged_row_splits
/// * `OutputList` feature_list_sparse_indices
/// * `OutputList` feature_list_sparse_values
/// * `OutputList` feature_list_sparse_shapes
/// * `OutputList` feature_list_dense_values
/// * `OutputList` feature_list_dense_lengths
/// * `OutputList` feature_list_ragged_values
/// * `OutputList` feature_list_ragged_outer_splits
/// * `OutputList` feature_list_ragged_inner_splits
class ParseSequenceExampleV2 {
 public:
  /// Optional attribute setters for ParseSequenceExampleV2
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NcontextSparse(int64 x) {
      Attrs ret = *this;
      ret.Ncontext_sparse_ = x;
      return ret;
    }

    /// A list of Ncontext_sparse types; the data types of data in
    /// each context Feature given in context_sparse_keys.
    /// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
    /// DT_INT64 (Int64List), and DT_STRING (BytesList).
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextSparseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.context_sparse_types_ = x;
      return ret;
    }

    /// RaggedTensor.value dtypes for the ragged context features.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextRaggedValueTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.context_ragged_value_types_ = x;
      return ret;
    }

    /// RaggedTensor.row_split dtypes for the ragged context features.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextRaggedSplitTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.context_ragged_split_types_ = x;
      return ret;
    }

    /// A list of Ncontext_dense shapes; the shapes of data in
    /// each context Feature given in context_dense_keys.
    /// The number of elements in the Feature corresponding to context_dense_key[j]
    /// must always equal context_dense_shapes[j].NumEntries().
    /// The shape of context_dense_values[j] will match context_dense_shapes[j].
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.context_dense_shapes_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NfeatureListSparse(int64 x) {
      Attrs ret = *this;
      ret.Nfeature_list_sparse_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NfeatureListDense(int64 x) {
      Attrs ret = *this;
      ret.Nfeature_list_dense_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListDenseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_dense_types_ = x;
      return ret;
    }

    /// A list of Nfeature_list_sparse types; the data types
    /// of data in each FeatureList given in feature_list_sparse_keys.
    /// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
    /// DT_INT64 (Int64List), and DT_STRING (BytesList).
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListSparseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_sparse_types_ = x;
      return ret;
    }

    /// RaggedTensor.value dtypes for the ragged FeatureList features.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListRaggedValueTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_ragged_value_types_ = x;
      return ret;
    }

    /// RaggedTensor.row_split dtypes for the ragged FeatureList features.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListRaggedSplitTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_ragged_split_types_ = x;
      return ret;
    }

    /// A list of Nfeature_list_dense shapes; the shapes of
    /// data in each FeatureList given in feature_list_dense_keys.
    /// The shape of each Feature in the FeatureList corresponding to
    /// feature_list_dense_key[j] must always equal
    /// feature_list_dense_shapes[j].NumEntries().
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.feature_list_dense_shapes_ = x;
      return ret;
    }

    int64 Ncontext_sparse_ = 0;
    DataTypeSlice context_sparse_types_ = {};
    DataTypeSlice context_ragged_value_types_ = {};
    DataTypeSlice context_ragged_split_types_ = {};
    gtl::ArraySlice<PartialTensorShape> context_dense_shapes_ = {};
    int64 Nfeature_list_sparse_ = 0;
    int64 Nfeature_list_dense_ = 0;
    DataTypeSlice feature_list_dense_types_ = {};
    DataTypeSlice feature_list_sparse_types_ = {};
    DataTypeSlice feature_list_ragged_value_types_ = {};
    DataTypeSlice feature_list_ragged_split_types_ = {};
    gtl::ArraySlice<PartialTensorShape> feature_list_dense_shapes_ = {};
  };
  ParseSequenceExampleV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       serialized, ::tensorflow::Input debug_name,
                       ::tensorflow::Input context_sparse_keys,
                       ::tensorflow::Input context_dense_keys,
                       ::tensorflow::Input context_ragged_keys,
                       ::tensorflow::Input feature_list_sparse_keys,
                       ::tensorflow::Input feature_list_dense_keys,
                       ::tensorflow::Input feature_list_ragged_keys,
                       ::tensorflow::Input
                       feature_list_dense_missing_assumed_empty,
                       ::tensorflow::InputList context_dense_defaults);
  ParseSequenceExampleV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       serialized, ::tensorflow::Input debug_name,
                       ::tensorflow::Input context_sparse_keys,
                       ::tensorflow::Input context_dense_keys,
                       ::tensorflow::Input context_ragged_keys,
                       ::tensorflow::Input feature_list_sparse_keys,
                       ::tensorflow::Input feature_list_dense_keys,
                       ::tensorflow::Input feature_list_ragged_keys,
                       ::tensorflow::Input
                       feature_list_dense_missing_assumed_empty,
                       ::tensorflow::InputList context_dense_defaults, const
                       ParseSequenceExampleV2::Attrs& attrs);

  static Attrs NcontextSparse(int64 x) {
    return Attrs().NcontextSparse(x);
  }
  static Attrs ContextSparseTypes(const DataTypeSlice& x) {
    return Attrs().ContextSparseTypes(x);
  }
  static Attrs ContextRaggedValueTypes(const DataTypeSlice& x) {
    return Attrs().ContextRaggedValueTypes(x);
  }
  static Attrs ContextRaggedSplitTypes(const DataTypeSlice& x) {
    return Attrs().ContextRaggedSplitTypes(x);
  }
  static Attrs ContextDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().ContextDenseShapes(x);
  }
  static Attrs NfeatureListSparse(int64 x) {
    return Attrs().NfeatureListSparse(x);
  }
  static Attrs NfeatureListDense(int64 x) {
    return Attrs().NfeatureListDense(x);
  }
  static Attrs FeatureListDenseTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListDenseTypes(x);
  }
  static Attrs FeatureListSparseTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListSparseTypes(x);
  }
  static Attrs FeatureListRaggedValueTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListRaggedValueTypes(x);
  }
  static Attrs FeatureListRaggedSplitTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListRaggedSplitTypes(x);
  }
  static Attrs FeatureListDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().FeatureListDenseShapes(x);
  }

  Operation operation;
  ::tensorflow::OutputList context_sparse_indices;
  ::tensorflow::OutputList context_sparse_values;
  ::tensorflow::OutputList context_sparse_shapes;
  ::tensorflow::OutputList context_dense_values;
  ::tensorflow::OutputList context_ragged_values;
  ::tensorflow::OutputList context_ragged_row_splits;
  ::tensorflow::OutputList feature_list_sparse_indices;
  ::tensorflow::OutputList feature_list_sparse_values;
  ::tensorflow::OutputList feature_list_sparse_shapes;
  ::tensorflow::OutputList feature_list_dense_values;
  ::tensorflow::OutputList feature_list_dense_lengths;
  ::tensorflow::OutputList feature_list_ragged_values;
  ::tensorflow::OutputList feature_list_ragged_outer_splits;
  ::tensorflow::OutputList feature_list_ragged_inner_splits;
};

/// Transforms a tf.Example proto (as a string) into typed tensors.
///
/// Args:
/// * scope: A Scope object
/// * serialized: A vector containing a batch of binary serialized Example protos.
/// * dense_defaults: A list of Tensors (some may be empty), whose length matches
/// the length of `dense_keys`. dense_defaults[j] provides default values
/// when the example's feature_map lacks dense_key[j].  If an empty Tensor is
/// provided for dense_defaults[j], then the Feature dense_keys[j] is required.
/// The input type is inferred from dense_defaults[j], even when it's empty.
/// If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
/// then the shape of dense_defaults[j] must match that of dense_shapes[j].
/// If dense_shapes[j] has an undefined major dimension (variable strides dense
/// feature), dense_defaults[j] must contain a single element:
/// the padding element.
/// * num_sparse: The number of sparse features to be parsed from the example. This
/// must match the lengths of `sparse_keys` and `sparse_types`.
/// * sparse_keys: A list of `num_sparse` strings.
/// The keys expected in the Examples' features associated with sparse values.
/// * dense_keys: The keys expected in the Examples' features associated with dense
/// values.
/// * sparse_types: A list of `num_sparse` types; the data types of data in each
/// Feature given in sparse_keys.
/// Currently the ParseSingleExample op supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * dense_shapes: The shapes of data in each Feature given in dense_keys.
/// The length of this list must match the length of `dense_keys`.  The
/// number of elements in the Feature corresponding to dense_key[j] must
/// always equal dense_shapes[j].NumEntries().  If dense_shapes[j] ==
/// (D0, D1, ..., DN) then the shape of output Tensor dense_values[j]
/// will be (D0, D1, ..., DN): In the case dense_shapes[j] = (-1, D1,
/// ..., DN), the shape of the output Tensor dense_values[j] will be (M,
/// D1, .., DN), where M is the number of blocks of elements of length
/// D1 * .... * DN, in the input.
///
/// Returns:
/// * `OutputList` sparse_indices
/// * `OutputList` sparse_values
/// * `OutputList` sparse_shapes
/// * `OutputList` dense_values
class ParseSingleExample {
 public:
  ParseSingleExample(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   serialized, ::tensorflow::InputList dense_defaults, int64
                   num_sparse, const gtl::ArraySlice<::tensorflow::tstring>&
                   sparse_keys, const gtl::ArraySlice<::tensorflow::tstring>&
                   dense_keys, const DataTypeSlice& sparse_types, const
                   gtl::ArraySlice<PartialTensorShape>& dense_shapes);

  Operation operation;
  ::tensorflow::OutputList sparse_indices;
  ::tensorflow::OutputList sparse_values;
  ::tensorflow::OutputList sparse_shapes;
  ::tensorflow::OutputList dense_values;
};

/// Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.
///
/// Args:
/// * scope: A Scope object
/// * serialized: A scalar containing a binary serialized SequenceExample proto.
/// * feature_list_dense_missing_assumed_empty: A vector listing the
/// FeatureList keys which may be missing from the SequenceExample.  If the
/// associated FeatureList is missing, it is treated as empty.  By default,
/// any FeatureList not listed in this vector must exist in the SequenceExample.
/// * context_sparse_keys: A list of Ncontext_sparse string Tensors (scalars).
/// The keys expected in the Examples' features associated with context_sparse
/// values.
/// * context_dense_keys: A list of Ncontext_dense string Tensors (scalars).
/// The keys expected in the SequenceExamples' context features associated with
/// dense values.
/// * feature_list_sparse_keys: A list of Nfeature_list_sparse string Tensors
/// (scalars).  The keys expected in the FeatureLists associated with sparse
/// values.
/// * feature_list_dense_keys: A list of Nfeature_list_dense string Tensors (scalars).
/// The keys expected in the SequenceExamples' feature_lists associated
/// with lists of dense values.
/// * context_dense_defaults: A list of Ncontext_dense Tensors (some may be empty).
/// context_dense_defaults[j] provides default values
/// when the SequenceExample's context map lacks context_dense_key[j].
/// If an empty Tensor is provided for context_dense_defaults[j],
/// then the Feature context_dense_keys[j] is required.
/// The input type is inferred from context_dense_defaults[j], even when it's
/// empty.  If context_dense_defaults[j] is not empty, its shape must match
/// context_dense_shapes[j].
/// * debug_name: A scalar containing the name of the serialized proto.
/// May contain, for example, table key (descriptive) name for the
/// corresponding serialized proto.  This is purely useful for debugging
/// purposes, and the presence of values here has no effect on the output.
/// May also be an empty scalar if no name is available.
///
/// Optional attributes (see `Attrs`):
/// * context_sparse_types: A list of Ncontext_sparse types; the data types of data in
/// each context Feature given in context_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * context_dense_shapes: A list of Ncontext_dense shapes; the shapes of data in
/// each context Feature given in context_dense_keys.
/// The number of elements in the Feature corresponding to context_dense_key[j]
/// must always equal context_dense_shapes[j].NumEntries().
/// The shape of context_dense_values[j] will match context_dense_shapes[j].
/// * feature_list_sparse_types: A list of Nfeature_list_sparse types; the data types
/// of data in each FeatureList given in feature_list_sparse_keys.
/// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
/// DT_INT64 (Int64List), and DT_STRING (BytesList).
/// * feature_list_dense_shapes: A list of Nfeature_list_dense shapes; the shapes of
/// data in each FeatureList given in feature_list_dense_keys.
/// The shape of each Feature in the FeatureList corresponding to
/// feature_list_dense_key[j] must always equal
/// feature_list_dense_shapes[j].NumEntries().
///
/// Returns:
/// * `OutputList` context_sparse_indices
/// * `OutputList` context_sparse_values
/// * `OutputList` context_sparse_shapes
/// * `OutputList` context_dense_values
/// * `OutputList` feature_list_sparse_indices
/// * `OutputList` feature_list_sparse_values
/// * `OutputList` feature_list_sparse_shapes
/// * `OutputList` feature_list_dense_values
class ParseSingleSequenceExample {
 public:
  /// Optional attribute setters for ParseSingleSequenceExample
  struct Attrs {
    /// A list of Ncontext_sparse types; the data types of data in
    /// each context Feature given in context_sparse_keys.
    /// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
    /// DT_INT64 (Int64List), and DT_STRING (BytesList).
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextSparseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.context_sparse_types_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListDenseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_dense_types_ = x;
      return ret;
    }

    /// A list of Ncontext_dense shapes; the shapes of data in
    /// each context Feature given in context_dense_keys.
    /// The number of elements in the Feature corresponding to context_dense_key[j]
    /// must always equal context_dense_shapes[j].NumEntries().
    /// The shape of context_dense_values[j] will match context_dense_shapes[j].
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ContextDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.context_dense_shapes_ = x;
      return ret;
    }

    /// A list of Nfeature_list_sparse types; the data types
    /// of data in each FeatureList given in feature_list_sparse_keys.
    /// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
    /// DT_INT64 (Int64List), and DT_STRING (BytesList).
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListSparseTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.feature_list_sparse_types_ = x;
      return ret;
    }

    /// A list of Nfeature_list_dense shapes; the shapes of
    /// data in each FeatureList given in feature_list_dense_keys.
    /// The shape of each Feature in the FeatureList corresponding to
    /// feature_list_dense_key[j] must always equal
    /// feature_list_dense_shapes[j].NumEntries().
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FeatureListDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.feature_list_dense_shapes_ = x;
      return ret;
    }

    DataTypeSlice context_sparse_types_ = {};
    DataTypeSlice feature_list_dense_types_ = {};
    gtl::ArraySlice<PartialTensorShape> context_dense_shapes_ = {};
    DataTypeSlice feature_list_sparse_types_ = {};
    gtl::ArraySlice<PartialTensorShape> feature_list_dense_shapes_ = {};
  };
  ParseSingleSequenceExample(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input serialized, ::tensorflow::Input
                           feature_list_dense_missing_assumed_empty,
                           ::tensorflow::InputList context_sparse_keys,
                           ::tensorflow::InputList context_dense_keys,
                           ::tensorflow::InputList feature_list_sparse_keys,
                           ::tensorflow::InputList feature_list_dense_keys,
                           ::tensorflow::InputList context_dense_defaults,
                           ::tensorflow::Input debug_name);
  ParseSingleSequenceExample(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input serialized, ::tensorflow::Input
                           feature_list_dense_missing_assumed_empty,
                           ::tensorflow::InputList context_sparse_keys,
                           ::tensorflow::InputList context_dense_keys,
                           ::tensorflow::InputList feature_list_sparse_keys,
                           ::tensorflow::InputList feature_list_dense_keys,
                           ::tensorflow::InputList context_dense_defaults,
                           ::tensorflow::Input debug_name, const
                           ParseSingleSequenceExample::Attrs& attrs);

  static Attrs ContextSparseTypes(const DataTypeSlice& x) {
    return Attrs().ContextSparseTypes(x);
  }
  static Attrs FeatureListDenseTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListDenseTypes(x);
  }
  static Attrs ContextDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().ContextDenseShapes(x);
  }
  static Attrs FeatureListSparseTypes(const DataTypeSlice& x) {
    return Attrs().FeatureListSparseTypes(x);
  }
  static Attrs FeatureListDenseShapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().FeatureListDenseShapes(x);
  }

  Operation operation;
  ::tensorflow::OutputList context_sparse_indices;
  ::tensorflow::OutputList context_sparse_values;
  ::tensorflow::OutputList context_sparse_shapes;
  ::tensorflow::OutputList context_dense_values;
  ::tensorflow::OutputList feature_list_sparse_indices;
  ::tensorflow::OutputList feature_list_sparse_values;
  ::tensorflow::OutputList feature_list_sparse_shapes;
  ::tensorflow::OutputList feature_list_dense_values;
};

/// Transforms a serialized tensorflow.TensorProto proto into a Tensor.
///
/// Args:
/// * scope: A Scope object
/// * serialized: A scalar string containing a serialized TensorProto proto.
/// * out_type: The type of the serialized tensor.  The provided type must match the
/// type of the serialized tensor and no implicit conversion will take place.
///
/// Returns:
/// * `Output`: A Tensor of type `out_type`.
class ParseTensor {
 public:
  ParseTensor(const ::tensorflow::Scope& scope, ::tensorflow::Input serialized,
            DataType out_type);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Transforms a Tensor into a serialized TensorProto proto.
///
/// Args:
/// * scope: A Scope object
/// * tensor: A Tensor of type `T`.
///
/// Returns:
/// * `Output`: A serialized TensorProto proto of the input tensor.
class SerializeTensor {
 public:
  SerializeTensor(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor);
  operator ::tensorflow::Output() const { return serialized; }
  operator ::tensorflow::Input() const { return serialized; }
  ::tensorflow::Node* node() const { return serialized.node(); }

  Operation operation;
  ::tensorflow::Output serialized;
};

/// Converts each string in the input Tensor to the specified numeric type.
///
/// (Note that int32 overflow results in an error while float overflow
/// results in a rounded value.)
///
/// Example:
///
/// >>> strings = ["5.0", "3.0", "7.0"]
/// >>> tf.strings.to_number(strings)
/// <tf.Tensor: shape=(3,), dtype=float32, numpy=array([5., 3., 7.], dtype=float32)>
///
///
/// Args:
/// * scope: A Scope object
///
/// Optional attributes (see `Attrs`):
/// * out_type: The numeric type to interpret each string in `string_tensor` as.
///
/// Returns:
/// * `Output`: A Tensor of the same shape as the input `string_tensor`.
class StringToNumber {
 public:
  /// Optional attribute setters for StringToNumber
  struct Attrs {
    /// The numeric type to interpret each string in `string_tensor` as.
    ///
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_FLOAT;
  };
  StringToNumber(const ::tensorflow::Scope& scope, ::tensorflow::Input
               string_tensor);
  StringToNumber(const ::tensorflow::Scope& scope, ::tensorflow::Input
               string_tensor, const StringToNumber::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_PARSING_OPS_H_
