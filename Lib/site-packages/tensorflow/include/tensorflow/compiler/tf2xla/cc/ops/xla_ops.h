// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_COMPILER_TF_XLA_CC_OPS_XLA_OPS_H_
#define TENSORFLOW_COMPILER_TF_XLA_CC_OPS_XLA_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup xla_ops Xla Ops
/// @{

/// Wraps the XLA AllReduce operator
///
///   documented at https://www.tensorflow.org/xla/operation_semantics#allreduce.
///
/// Args:
/// * scope: A Scope object
/// * input: Array or a non-empty tuple of arrays to reduce across replicas.
/// * group_assignment: Groups between which the reductions are performed.
/// * reduce_op: Reduction computation.
/// * mode: group mode.
/// CrossReplica: group_assignment contains replica_id. Each group contains the
///   replicas for the current partition.
/// CrossReplicaAndPartition: group_assignment contains replica_id. Each group
///   contains the replicas for all partitions.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaAllReduce {
 public:
  XlaAllReduce(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
             ::tensorflow::Input group_assignment, StringPiece reduce_op,
             StringPiece mode);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Helper operator for performing XLA-style broadcasts
///
/// Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
/// whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
/// for binary operators.
///
/// Args:
/// * scope: A Scope object
/// * lhs: the LHS input tensor
/// * rhs: the RHS input tensor
/// * broadcast_dims: an XLA-style broadcast dimension specification
///
/// Returns:
/// * `Output` lhs_output: the broadcasted LHS tensor
/// * `Output` rhs_output: the broadcasted RHS tensor
class XlaBroadcastHelper {
 public:
  XlaBroadcastHelper(const ::tensorflow::Scope& scope, ::tensorflow::Input lhs,
                   ::tensorflow::Input rhs, ::tensorflow::Input broadcast_dims);

  Operation operation;
  ::tensorflow::Output lhs_output;
  ::tensorflow::Output rhs_output;
};

/// Invokes a StableHLO module.
///
/// This op is used with JAX native serialization in a TensorFlow context with
/// stability guarantees.
///
/// Args:
/// * scope: A Scope object
/// * args: A list of `Tensor` with possibly different types to be passed as arguments
/// to the `module`. These are the actual arguments and do not include the
/// platform argument (see `platforms`) nor the dimension arguments (see
/// `dim_args_spec`).
/// * version: Tracks changes the semantics of the op, to support backwards
/// compatibility. Minimum supported version is 2. From
/// version 2, the op carries a StableHLO text or bytecode `module`. From
/// version 3, the op also supports the `platforms` attribute. From version 4,
/// the op carries a StableHLO module with compatibility guarantees. From version
/// 5, XLACallModule can include `stablehlo.custom_call` op to execute tf
/// functions. From version 6 the op supports the `disabled_checks` attribute.
/// See more versioning details at https://github.com/search?q=repo%3Atensorflow%2Ftensorflow+path%3Axla_call_module+%22int+kVersionMaximumSupported%22&type=code.
/// * module_: A serialized computation, a text or bytecode representation of
/// an mlir.Module. The return type must be a tuple if and only if the `Sout` is
/// a list with 0 or more than 1 elements. The length of `Tout` and
/// `Sout` must match. This op always returns a tuple of results, even if the
/// module returns a single result.
/// * Sout: List of output tensor shapes.
/// * Tout: List of output tensor data types.
///
/// Optional attributes (see `Attrs`):
/// * dim_args_spec: this attribute is not supported anymore.
/// * platforms: the list of platforms supported by `module`. The list can contain
/// the strings "CPU", "CUDA", "ROCM", or "TPU". It is an error to compile
/// this op for a platform that does not appear in the list. This check can be
/// disabled using `disabled_checks`. If the list contains more than
/// one platform, then the `module` takes one additional 0-dimensional
/// integer-tensor parameter in the first position, encoding the index in
/// `platforms` of the current compilation platform. This parameter has value 0
/// if the plaform is not among `platforms` and the check has been disabled.
/// The list can be empty in old versions (earlier than 6) to denote that no
/// platform checking must be performed at loading time.
/// * function_list: This list contains the TensorFlow FunctionDefs that are used by
/// the XLACallModule. If the XLACallModule contains `stablehlo.custom_call`
/// operations, they can call TensorFlow graph functions outside of the
/// XLACallModule. This `function_list` attribute registers the dependency of the
/// XLACallModule on those functions. This attribute was added in version 5.
/// * has_token_input_output: If true, the embedded StableHLO module's main function
/// must take a `!stablehlo.token` as its first argument and returns a token as
/// its first result. This can be used in conjunction with the TF2XLA's side
/// effect mechanism in order to model side effects. This is used only in versions
/// prior to version 9. After that, the number and position of tokens among
/// the arguments and results are obtained from the main function type. This
/// allows us to support more than one token and not necessarily at the start.
/// * disabled_checks: A list of strings describing the safety checks that were
/// disabled at serialization time. This attribute was added in version 6.
/// For more details see
/// https://github.com/search?q=repo%3Agoogle%2Fjax+path%3Ajax_export+%22class+DisabledSafetyCheck%22&type=code.
/// This list, supplemented with a comma-separate list of directives specified
/// using the flag --tf_xla_call_module_disabled_checks,
/// is used at module loading time to skip the corresponding checks.
///
/// Returns:
/// * `OutputList`: The output tensor.
class XlaCallModule {
 public:
  /// Optional attribute setters for XlaCallModule
  struct Attrs {
    /// this attribute is not supported anymore.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs DimArgsSpec(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.dim_args_spec_ = x;
      return ret;
    }

    /// the list of platforms supported by `module`. The list can contain
    /// the strings "CPU", "CUDA", "ROCM", or "TPU". It is an error to compile
    /// this op for a platform that does not appear in the list. This check can be
    /// disabled using `disabled_checks`. If the list contains more than
    /// one platform, then the `module` takes one additional 0-dimensional
    /// integer-tensor parameter in the first position, encoding the index in
    /// `platforms` of the current compilation platform. This parameter has value 0
    /// if the plaform is not among `platforms` and the check has been disabled.
    /// The list can be empty in old versions (earlier than 6) to denote that no
    /// platform checking must be performed at loading time.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs Platforms(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.platforms_ = x;
      return ret;
    }

    /// This list contains the TensorFlow FunctionDefs that are used by
    /// the XLACallModule. If the XLACallModule contains `stablehlo.custom_call`
    /// operations, they can call TensorFlow graph functions outside of the
    /// XLACallModule. This `function_list` attribute registers the dependency of the
    /// XLACallModule on those functions. This attribute was added in version 5.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs FunctionList(const gtl::ArraySlice<NameAttrList>& x) {
      Attrs ret = *this;
      ret.function_list_ = x;
      return ret;
    }

    /// If true, the embedded StableHLO module's main function
    /// must take a `!stablehlo.token` as its first argument and returns a token as
    /// its first result. This can be used in conjunction with the TF2XLA's side
    /// effect mechanism in order to model side effects. This is used only in versions
    /// prior to version 9. After that, the number and position of tokens among
    /// the arguments and results are obtained from the main function type. This
    /// allows us to support more than one token and not necessarily at the start.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs HasTokenInputOutput(bool x) {
      Attrs ret = *this;
      ret.has_token_input_output_ = x;
      return ret;
    }

    /// A list of strings describing the safety checks that were
    /// disabled at serialization time. This attribute was added in version 6.
    /// For more details see
    /// https://github.com/search?q=repo%3Agoogle%2Fjax+path%3Ajax_export+%22class+DisabledSafetyCheck%22&type=code.
    /// This list, supplemented with a comma-separate list of directives specified
    /// using the flag --tf_xla_call_module_disabled_checks,
    /// is used at module loading time to skip the corresponding checks.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs DisabledChecks(const gtl::ArraySlice<::tensorflow::tstring>& x) {
      Attrs ret = *this;
      ret.disabled_checks_ = x;
      return ret;
    }

    gtl::ArraySlice<::tensorflow::tstring> dim_args_spec_ = {};
    gtl::ArraySlice<::tensorflow::tstring> platforms_ = {};
    gtl::ArraySlice<NameAttrList> function_list_ = {};
    bool has_token_input_output_ = false;
    gtl::ArraySlice<::tensorflow::tstring> disabled_checks_ = {};
  };
  XlaCallModule(const ::tensorflow::Scope& scope, ::tensorflow::InputList args,
              int64 version, StringPiece module_, const
              gtl::ArraySlice<PartialTensorShape>& Sout, const DataTypeSlice&
              Tout);
  XlaCallModule(const ::tensorflow::Scope& scope, ::tensorflow::InputList args,
              int64 version, StringPiece module_, const
              gtl::ArraySlice<PartialTensorShape>& Sout, const DataTypeSlice&
              Tout, const XlaCallModule::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs DimArgsSpec(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().DimArgsSpec(x);
  }
  static Attrs Platforms(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().Platforms(x);
  }
  static Attrs FunctionList(const gtl::ArraySlice<NameAttrList>& x) {
    return Attrs().FunctionList(x);
  }
  static Attrs HasTokenInputOutput(bool x) {
    return Attrs().HasTokenInputOutput(x);
  }
  static Attrs DisabledChecks(const gtl::ArraySlice<::tensorflow::tstring>& x) {
    return Attrs().DisabledChecks(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// Wraps the XLA ConvGeneralDilated operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
/// .
///
/// Args:
/// * scope: A Scope object
/// * lhs: the input tensor
/// * rhs: the kernel tensor
/// * window_strides: the inter-window strides
/// * padding: the padding to apply at the start and end of each input dimensions
/// * lhs_dilation: dilation to apply between input elements
/// * rhs_dilation: dilation to apply between kernel elements
/// * feature_group_count: number of feature groups for grouped convolution.
/// * dimension_numbers: a serialized xla::ConvolutionDimensionNumbers proto.
/// * precision_config: a serialized xla::PrecisionConfig proto.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaConv {
 public:
  XlaConv(const ::tensorflow::Scope& scope, ::tensorflow::Input lhs,
        ::tensorflow::Input rhs, ::tensorflow::Input window_strides,
        ::tensorflow::Input padding, ::tensorflow::Input lhs_dilation,
        ::tensorflow::Input rhs_dilation, ::tensorflow::Input
        feature_group_count, StringPiece dimension_numbers, StringPiece
        precision_config);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA ConvGeneralDilated operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
/// .
///
/// Args:
/// * scope: A Scope object
/// * lhs: input tensor
/// * rhs: kernel tensor
/// * window_strides: inter-window strides
/// * padding: padding to apply at the start and end of each input dimensions
/// * lhs_dilation: dilation to apply between input elements
/// * rhs_dilation: dilation to apply between kernel elements
/// * feature_group_count: number of feature groups for grouped convolution.
/// * dimension_numbers: serialized xla::ConvolutionDimensionNumbers proto.
/// * precision_config: serialized xla::PrecisionConfig proto.
/// * preferred_element_type: type of the tensor.
///
/// Optional attributes (see `Attrs`):
/// * batch_group_count: number of batch groups or grouped filters.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaConvV2 {
 public:
  /// Optional attribute setters for XlaConvV2
  struct Attrs {
    /// number of batch groups or grouped filters.
    ///
    /// Defaults to 1
    TF_MUST_USE_RESULT Attrs BatchGroupCount(int64 x) {
      Attrs ret = *this;
      ret.batch_group_count_ = x;
      return ret;
    }

    int64 batch_group_count_ = 1;
  };
  XlaConvV2(const ::tensorflow::Scope& scope, ::tensorflow::Input lhs,
          ::tensorflow::Input rhs, ::tensorflow::Input window_strides,
          ::tensorflow::Input padding, ::tensorflow::Input lhs_dilation,
          ::tensorflow::Input rhs_dilation, ::tensorflow::Input
          feature_group_count, StringPiece dimension_numbers, StringPiece
          precision_config, DataType preferred_element_type);
  XlaConvV2(const ::tensorflow::Scope& scope, ::tensorflow::Input lhs,
          ::tensorflow::Input rhs, ::tensorflow::Input window_strides,
          ::tensorflow::Input padding, ::tensorflow::Input lhs_dilation,
          ::tensorflow::Input rhs_dilation, ::tensorflow::Input
          feature_group_count, StringPiece dimension_numbers, StringPiece
          precision_config, DataType preferred_element_type, const
          XlaConvV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BatchGroupCount(int64 x) {
    return Attrs().BatchGroupCount(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA CustomCall operator
///
///   documented at https://www.tensorflow.org/xla/operation_semantics#customcall.
///
/// Args:
/// * scope: A Scope object
/// * args: A list of `Tensor` with possibly different types.
/// * target_name: Name of the function. A call instruction will be emitted which
/// targets this symbol name.
/// * backend_config: String, used to encode serialized metadata to the backend.
/// * dtype: Output tensor data type.
/// * shape: Output tensor shape.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaCustomCall {
 public:
  XlaCustomCall(const ::tensorflow::Scope& scope, ::tensorflow::InputList args,
              StringPiece target_name, StringPiece backend_config, DataType
              dtype, PartialTensorShape shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Emits an HLO `CustomCall` operation with multiple outputs.
///
/// As opposed to `XlaCustomCall`, this operation supports multiple outputs.
///
/// See `CustomCall` specification at
///   https://tensorflow.org/xla/operation_semantics#customcall,
/// and `mhlo.custom_call` specification at
///   https://tensorflow.org/mlir/hlo_ops#mhlocustom_call_mlirmhlocustomcallop.
///
/// Args:
/// * scope: A Scope object
/// * operands: A sequence of tensors with possibly different types.
/// * call_target_name: Name of the user function. The function signature must conform
/// to version 3 of the API, see `API_VERSION_STATUS_RETURNING_UNIFIED`. All
/// operands and results assumed to be in the default layout.
/// * backend_config: A string that encodes a metadata for the backend.
/// * has_side_effect: Indicates whether the custom call has side effects.
/// * result_dtypes: Types of all results.
/// * result_shapes: Shapes of all results.
///
/// Returns:
/// * `OutputList`: The results tensor.
class XlaCustomCallV2 {
 public:
  XlaCustomCallV2(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                operands, StringPiece call_target_name, StringPiece
                backend_config, bool has_side_effect, const DataTypeSlice&
                result_dtypes, const gtl::ArraySlice<PartialTensorShape>&
                result_shapes);
  ::tensorflow::Output operator[](size_t index) const { return results[index]; }


  Operation operation;
  ::tensorflow::OutputList results;
};

/// Takes the packed uint32 input and unpacks the input to uint8 to do
///
/// Dequantization on device.
///
/// Args:
/// * scope: A Scope object
/// * input: Input tensors whose types is uint32, shape is [d0, ..., dn].
/// * min_range: The minimum scalar value possibly produced for the input.
/// * max_range: The maximum scalar value possibly produced for the input.
/// * mode: String to determine the dequantize mode in {"MIN_COMBINED", "MIN_FIRST", "SCALED"}.
/// * transpose_output: Boolean to determine if output is transposed. transpose_output
/// is faster when input is large and rank of input is higher than 1.
///
/// Returns:
/// * `Output`: Output tensors whose types is bfloat16. If transpose_output is true,
/// output shape is [dn * 4, dn-1, ..., d1, d0]. If transpose_output
/// is false, output shape is [d0,..., dn * 4].
class XlaDequantize {
 public:
  XlaDequantize(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              float min_range, float max_range, StringPiece mode, bool
              transpose_output);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA DotGeneral operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
/// .
///
/// Args:
/// * scope: A Scope object
/// * lhs: the LHS tensor
/// * rhs: the RHS tensor
/// * dimension_numbers: a serialized xla::DotDimensionNumbers proto.
/// * precision_config: a serialized xla::PrecisionConfig proto.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaDot {
 public:
  XlaDot(const ::tensorflow::Scope& scope, ::tensorflow::Input lhs,
       ::tensorflow::Input rhs, StringPiece dimension_numbers, StringPiece
       precision_config);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA DotGeneral operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
/// .
///
/// Args:
/// * scope: A Scope object
/// * lhs: the LHS tensor
/// * rhs: the RHS tensor
/// * dimension_numbers: a serialized xla::DotDimensionNumbers proto.
/// * precision_config: a serialized xla::PrecisionConfig proto.
/// * preferred_element_type: The type of the tensor.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaDotV2 {
 public:
  XlaDotV2(const ::tensorflow::Scope& scope, ::tensorflow::Input lhs,
         ::tensorflow::Input rhs, StringPiece dimension_numbers, StringPiece
         precision_config, DataType preferred_element_type);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA DynamicSlice operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#dynamicslice
/// .
///
/// DynamicSlice extracts a sub-array from the input array at dynamic
/// start_indices. The size of the slice in each dimension is passed in
/// size_indices, which specify the end point of exclusive slice intervals in each
/// dimension -- [start, start + size). The shape of start_indices must have rank 1,
/// with dimension size equal to the rank of operand.
///
/// Args:
/// * scope: A Scope object
/// * input: A `Tensor` of type T.
/// * start_indices: List of N integers containing the slice size for each
/// dimension. Each value must be strictly greater than zero, and start + size
/// must be less than or equal to the size of the dimension to avoid
/// implementation defined behavior.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaDynamicSlice {
 public:
  XlaDynamicSlice(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                ::tensorflow::Input start_indices, ::tensorflow::Input
                size_indices);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA DynamicUpdateSlice operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
/// .
///
/// XlaDynamicUpdateSlice generates a result which is the value of the `input`
/// operand, with a slice update overwritten at `indices`. The shape of `update`
/// determines the shape of the sub-array of the result which is updated. The shape
/// of indices must be rank == 1, with dimension size equal to the rank of `input`.
///
/// Handling of out-of-bounds slice indices is implementation-defined.
///
/// Args:
/// * scope: A Scope object
/// * input: A `Tensor` of type T.
/// * update: A `Tensor` of type T. Same rank as `input`.
/// * indices: A vector of indices into `input`. Must have length equal to the rank of
/// `input`.
///
/// Returns:
/// * `Output`: A `Tensor` of type T.
class XlaDynamicUpdateSlice {
 public:
  XlaDynamicUpdateSlice(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input, ::tensorflow::Input update, ::tensorflow::Input
                      indices);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// An op which supports basic einsum op with 2 inputs and 1 output.
///
/// This op has better TPU performance since it doesn't have explicitly reshape and
/// transpose operations as tf.einsum does.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The product tensor.
class XlaEinsum {
 public:
  XlaEinsum(const ::tensorflow::Scope& scope, ::tensorflow::Input a,
          ::tensorflow::Input b, StringPiece equation);
  operator ::tensorflow::Output() const { return product; }
  operator ::tensorflow::Input() const { return product; }
  ::tensorflow::Node* node() const { return product.node(); }

  Operation operation;
  ::tensorflow::Output product;
};

/// Wraps the XLA Gather operator documented at
///
///   https://www.tensorflow.org/xla/operation_semantics#gather
///
/// Args:
/// * scope: A Scope object
/// * operand: The array we're gathering from.
/// * start_indices: Array containing the starting indices of the slices we gather.
/// * slice_sizes: slice_sizes[i] is the bounds for the slice on dimension i.
/// * dimension_numbers: A serialized xla::GatherDimensionNumbers proto.
/// * indices_are_sorted: Boolean indicating if the indices are sorted.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaGather {
 public:
  XlaGather(const ::tensorflow::Scope& scope, ::tensorflow::Input operand,
          ::tensorflow::Input start_indices, ::tensorflow::Input slice_sizes,
          StringPiece dimension_numbers, bool indices_are_sorted);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// output = cond ? then_branch(inputs) : else_branch(inputs).
///
/// Args:
/// * scope: A Scope object
/// * cond: A boolean scalar.
/// * inputs: A list of input tensors.
/// * then_branch: A function takes 'inputs' and returns a list of tensors,
/// whose types are the same as what else_branch returns.
/// * else_branch: A function takes 'inputs' and returns a list of tensors.
/// whose types are the same as what then_branch returns.
///
/// Returns:
/// * `OutputList`: A list of tensors returned by either then_branch(inputs) or
/// else_branch(inputs). The input shapes of the then_branch and
/// else_branch must match.
class XlaIf {
 public:
  XlaIf(const ::tensorflow::Scope& scope, ::tensorflow::Input cond,
      ::tensorflow::InputList inputs, const NameAttrList& then_branch, const
      NameAttrList& else_branch, const DataTypeSlice& Tout);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// Wraps the XLA Sort operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#sort
/// .
///
/// Sorts a tensor. Currently only sorts in ascending order are supported.
///
/// Args:
/// * scope: A Scope object
/// * keys: A `Tensor` of type K.
/// * values: A `Tensor` of type V.
///
/// Returns:
/// * `Output` sorted_keys: A `Tensor` of type K.
/// * `Output` sorted_values: A `Tensor` of type V.
class XlaKeyValueSort {
 public:
  XlaKeyValueSort(const ::tensorflow::Scope& scope, ::tensorflow::Input keys,
                ::tensorflow::Input values);

  Operation operation;
  ::tensorflow::Output sorted_keys;
  ::tensorflow::Output sorted_values;
};

/// Wraps the XLA OptimizationBarrier operator.
///
/// Documented at https://www.tensorflow.org/xla/operation_semantics#optimizationbarrier.
///
/// Args:
/// * scope: A Scope object
/// * input: A Tuple of Arrays of any type.
///
/// Returns:
/// * `OutputList`: The output tensor.
class XlaOptimizationBarrier {
 public:
  XlaOptimizationBarrier(const ::tensorflow::Scope& scope,
                       ::tensorflow::InputList input);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// Wraps the XLA Pad operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#pad
/// .
///
/// Args:
/// * scope: A Scope object
/// * input: A `Tensor` of type T.
/// * padding_value: A scalar `Tensor` of type T.
/// * padding_low: the padding to apply at the start of each input dimensions. Must
/// be a compile-time constant 1D tensor of length equal to rank of input.
/// * padding_high: the padding to apply at the end of each input dimension. Must
/// be a compile-time constant 1D tensor of length equal to rank of input.
/// * padding_interior: the padding to apply between each input element. Must
/// be a compile-time constant 1D tensor of length equal to rank of input,
/// containing only non-negative values.
///
/// Returns:
/// * `Output`: A `Tensor` of type T.
class XlaPad {
 public:
  XlaPad(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
       ::tensorflow::Input padding_value, ::tensorflow::Input padding_low,
       ::tensorflow::Input padding_high, ::tensorflow::Input padding_interior);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Receives the named tensor from another XLA computation. Wraps the XLA Recv
///
/// operator documented at
///  https://www.tensorflow.org/performance/xla/operation_semantics#recv .
///
/// Args:
/// * scope: A Scope object
/// * dtype: The type of the tensor.
/// * tensor_name: A string key that identifies the channel.
/// * shape: The shape of the tensor.
///
/// Returns:
/// * `Output`: The tensor to receive.
class XlaRecv {
 public:
  XlaRecv(const ::tensorflow::Scope& scope, DataType dtype, StringPiece
        tensor_name, PartialTensorShape shape);
  operator ::tensorflow::Output() const { return tensor; }
  operator ::tensorflow::Input() const { return tensor; }
  ::tensorflow::Node* node() const { return tensor.node(); }

  Operation operation;
  ::tensorflow::Output tensor;
};

/// Wraps the XLA Reduce operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#reduce .
///
/// Args:
/// * scope: A Scope object
/// * input: the input tensor
/// * init_value: a scalar representing the initial value for the reduction
/// * dimensions_to_reduce: dimension numbers over which to reduce
/// * reducer: a reducer function to apply
///
/// Returns:
/// * `Output`: The output tensor.
class XlaReduce {
 public:
  XlaReduce(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
          ::tensorflow::Input init_value, const gtl::ArraySlice<int>&
          dimensions_to_reduce, const NameAttrList& reducer);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA ReducePrecision operator
///
///   documented at https://www.tensorflow.org/xla/operation_semantics#reduceprecision.
///
/// Args:
/// * scope: A Scope object
/// * operand: array of floating-point type.
/// * exponent_bits: number of exponent bits in lower-precision format
/// * mantissa_bits: number of mantissa bits in lower-precision format
///
/// Returns:
/// * `Output`: The output tensor.
class XlaReducePrecision {
 public:
  XlaReducePrecision(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   operand, int64 exponent_bits, int64 mantissa_bits);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA ReduceScatter operator
///
///   documented at https://www.tensorflow.org/xla/operation_semantics#reducescatter.
///
/// Args:
/// * scope: A Scope object
/// * input: Array or a non-empty tuple of arrays to reduce across replicas.
/// * group_assignment: Groups between which the reductions are performed.
/// * scatter_dimension: Dimension to scatter.
/// * reduce_op: Reduction computation.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaReduceScatter {
 public:
  XlaReduceScatter(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 ::tensorflow::Input group_assignment, ::tensorflow::Input
                 scatter_dimension, StringPiece reduce_op);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA ReduceWindow operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .
///
/// Args:
/// * scope: A Scope object
/// * input: the input tensor
/// * init_value: a scalar representing the initial value for the reduction
/// * window_dimensions: the shape of the window
/// * window_strides: the inter-window strides
/// * padding: the padding to apply at the start and end of each input dimensions
/// * computation: a reducer function to apply
///
/// Returns:
/// * `Output`: The output tensor.
class XlaReduceWindow {
 public:
  XlaReduceWindow(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                ::tensorflow::Input init_value, ::tensorflow::Input
                window_dimensions, ::tensorflow::Input window_strides,
                ::tensorflow::Input base_dilations, ::tensorflow::Input
                window_dilations, ::tensorflow::Input padding, const
                NameAttrList& computation);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Inverse of XlaSetDynamicDimensionSize.
///
/// Make an xla bounded dynamic dimension into a static dimension. The bound of the
/// size of dimension `dim_index` becomes the static dimension size.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class XlaRemoveDynamicDimensionSize {
 public:
  XlaRemoveDynamicDimensionSize(const ::tensorflow::Scope& scope,
                              ::tensorflow::Input input, ::tensorflow::Input
                              dim_index);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Replica ID.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The id tensor.
class XlaReplicaId {
 public:
  XlaReplicaId(const ::tensorflow::Scope& scope);
  operator ::tensorflow::Output() const { return id; }
  operator ::tensorflow::Input() const { return id; }
  ::tensorflow::Node* node() const { return id.node(); }

  Operation operation;
  ::tensorflow::Output id;
};

/// Stateless PRNG bit generator.
///
/// Wraps the XLA RngBitGenerator operator, documented at
///  https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.
///
/// Args:
/// * scope: A Scope object
/// * algorithm: The PRNG algorithm to use, one of
/// tf.random.Algorithm.{PHILOX, THREEFRY, AUTO_SELECT}.
/// * initial_state: Initial state for the PRNG algorithm. For THREEFRY, it should be
/// a u64[2] and for PHILOX a u64[3].
/// * shape: The output shape of the generated data.
///
/// Optional attributes (see `Attrs`):
/// * dtype: The type of the tensor.
///
/// Returns:
/// * `Output` output_key
/// * `Output` output
class XlaRngBitGenerator {
 public:
  /// Optional attribute setters for XlaRngBitGenerator
  struct Attrs {
    /// The type of the tensor.
    ///
    /// Defaults to DT_UINT64
    TF_MUST_USE_RESULT Attrs Dtype(DataType x) {
      Attrs ret = *this;
      ret.dtype_ = x;
      return ret;
    }

    DataType dtype_ = DT_UINT64;
  };
  XlaRngBitGenerator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   algorithm, ::tensorflow::Input initial_state,
                   ::tensorflow::Input shape);
  XlaRngBitGenerator(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   algorithm, ::tensorflow::Input initial_state,
                   ::tensorflow::Input shape, const XlaRngBitGenerator::Attrs&
                   attrs);

  static Attrs Dtype(DataType x) {
    return Attrs().Dtype(x);
  }

  Operation operation;
  ::tensorflow::Output output_key;
  ::tensorflow::Output output;
};

/// Wraps the XLA Scatter operator documented at
///
///   https://www.tensorflow.org/xla/operation_semantics#scatter.
///
/// Args:
/// * scope: A Scope object
/// * operand: Array to be scattered into.
/// * scatter_indices: Array containing the starting indices of the slices that must
/// be scattered to.
/// * updates: Array containing the values that must be used for scattering.
/// * update_computation: Computation to be used for combining the existing values in
/// the input array and the updates during scatter.
/// * dimension_numbers: A serialized xla::ScatterDimensionNumbers proto.
/// * indices_are_sorted: Boolean indicating if the indices are sorted.
///
/// Returns:
/// * `Output`: The output tensor.
class XlaScatter {
 public:
  XlaScatter(const ::tensorflow::Scope& scope, ::tensorflow::Input operand,
           ::tensorflow::Input scatter_indices, ::tensorflow::Input updates,
           const NameAttrList& update_computation, StringPiece
           dimension_numbers, bool indices_are_sorted);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA SelectAndScatter operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
/// .
///
/// Args:
/// * scope: A Scope object
/// * operand: the input tensor
/// * window_dimensions: the shape of the window
/// * window_strides: the inter-window strides
/// * padding: the padding to apply at the start and end of each input dimensions
/// * source: a tensor of values to scatter
/// * init_value: a scalar representing the initial value for the output tensor
/// * select: a selection function to apply
/// * scatter: a scatter function to apply
///
/// Returns:
/// * `Output`: The output tensor.
class XlaSelectAndScatter {
 public:
  XlaSelectAndScatter(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    operand, ::tensorflow::Input window_dimensions,
                    ::tensorflow::Input window_strides, ::tensorflow::Input
                    padding, ::tensorflow::Input source, ::tensorflow::Input
                    init_value, const NameAttrList& select, const NameAttrList&
                    scatter);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the eigen decomposition of a batch of self-adjoint matrices
///
/// (Note: Only real inputs are supported).
///
/// Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
/// tensor such that tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i], for
/// i=0...N-1.
///
/// Args:
/// * scope: A Scope object
/// * a: the input tensor.
/// * lower: a boolean specifies whether the calculation is done with the lower
/// triangular part or the upper triangular part.
/// * max_iter: maximum number of sweep update, i.e., the whole lower triangular
/// part or upper triangular part based on parameter lower. Heuristically, it has
/// been argued that approximately logN sweeps are needed in practice (Ref: Golub &
/// van Loan "Matrix Computation").
/// * epsilon: the tolerance ratio.
///
/// Returns:
/// * `Output` w: The eigenvalues in ascending order, each repeated according to its
/// multiplicity.
/// * `Output` v: The column v[..., :, i] is the normalized eigenvector corresponding to the
/// eigenvalue w[..., i].
class XlaSelfAdjointEig {
 public:
  XlaSelfAdjointEig(const ::tensorflow::Scope& scope, ::tensorflow::Input a, bool
                  lower, int64 max_iter, float epsilon);

  Operation operation;
  ::tensorflow::Output w;
  ::tensorflow::Output v;
};

/// Sends the named tensor to another XLA computation. Wraps the XLA Send operator
///
/// documented at
///  https://www.tensorflow.org/performance/xla/operation_semantics#send .
///
/// Args:
/// * scope: A Scope object
/// * tensor: The tensor to send.
/// * tensor_name: A string key that identifies the channel.
///
/// Returns:
/// * the created `Operation`
class XlaSend {
 public:
  XlaSend(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
        StringPiece tensor_name);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Set a bound for the given input value as a hint to Xla compiler,
///
///         returns the same value.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class XlaSetBound {
 public:
  XlaSetBound(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
            ::tensorflow::Input bound);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Make a static dimension into a xla bounded dynamic dimension.
///
///         The current static dimension size will become the bound and the second
///         operand becomes the dynamic size of the dimension.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class XlaSetDynamicDimensionSize {
 public:
  XlaSetDynamicDimensionSize(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input input, ::tensorflow::Input
                           dim_index, ::tensorflow::Input size);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// An op which shards the input based on the given sharding attribute. It can
///
/// selectively annotate a subset of tensor dimensions by skipping unspecified_dims,
/// and the sharding annotation should be replicated in those dims.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class XlaSharding {
 public:
  /// Optional attribute setters for XlaSharding
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Sharding(StringPiece x) {
      Attrs ret = *this;
      ret.sharding_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs UnspecifiedDims(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.unspecified_dims_ = x;
      return ret;
    }

    StringPiece sharding_ = "";
    gtl::ArraySlice<int> unspecified_dims_ = {};
  };
  XlaSharding(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  XlaSharding(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
            XlaSharding::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Sharding(StringPiece x) {
    return Attrs().Sharding(x);
  }
  static Attrs UnspecifiedDims(const gtl::ArraySlice<int>& x) {
    return Attrs().UnspecifiedDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Wraps the XLA Sort operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#sort
/// .
///
/// Sorts a tensor. Currently only sorts in ascending order are supported.
///
/// Args:
/// * scope: A Scope object
/// * input: A `Tensor` of type T.
///
/// Returns:
/// * `Output`: A `Tensor` of type T.
class XlaSort {
 public:
  XlaSort(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// An op used by XLA SPMD partitioner to switch from automatic partitioning to
///
/// manual partitioning. It annotates the input (full-shape, to be automatically
/// partitioned) with the same sharding used by manual partitioning, and outputs a
/// shard-shaped tensor to be consumed by later manually-partitioned ops. If the
/// shape is not evenly partitionable, the padding region will be masked with 0s.
/// The conversion can happen partially in subgroups, by specifying the dim
/// attribute, where only that dim will be converted.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class XlaSpmdFullToShardShape {
 public:
  /// Optional attribute setters for XlaSpmdFullToShardShape
  struct Attrs {
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Dim(int64 x) {
      Attrs ret = *this;
      ret.dim_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs UnspecifiedDims(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.unspecified_dims_ = x;
      return ret;
    }

    int64 dim_ = -1;
    gtl::ArraySlice<int> unspecified_dims_ = {};
  };
  XlaSpmdFullToShardShape(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, StringPiece manual_sharding);
  XlaSpmdFullToShardShape(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, StringPiece manual_sharding, const
                        XlaSpmdFullToShardShape::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Dim(int64 x) {
    return Attrs().Dim(x);
  }
  static Attrs UnspecifiedDims(const gtl::ArraySlice<int>& x) {
    return Attrs().UnspecifiedDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// An op used by XLA SPMD partitioner to switch from manual partitioning to
///
/// automatic partitioning. It converts the shard-shaped, manually partitioned input
/// into full-shaped tensor to be partitioned automatically with the same sharding
/// used by manual partitioning. The conversion can happen partially in subgroups,
/// by specifying the dim attribute, where only that dim will be converted.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class XlaSpmdShardToFullShape {
 public:
  /// Optional attribute setters for XlaSpmdShardToFullShape
  struct Attrs {
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Dim(int64 x) {
      Attrs ret = *this;
      ret.dim_ = x;
      return ret;
    }

    /// Defaults to []
    TF_MUST_USE_RESULT Attrs UnspecifiedDims(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.unspecified_dims_ = x;
      return ret;
    }

    int64 dim_ = -1;
    gtl::ArraySlice<int> unspecified_dims_ = {};
  };
  XlaSpmdShardToFullShape(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, StringPiece manual_sharding, PartialTensorShape
                        full_shape);
  XlaSpmdShardToFullShape(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, StringPiece manual_sharding, PartialTensorShape
                        full_shape, const XlaSpmdShardToFullShape::Attrs&
                        attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Dim(int64 x) {
    return Attrs().Dim(x);
  }
  static Attrs UnspecifiedDims(const gtl::ArraySlice<int>& x) {
    return Attrs().UnspecifiedDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Computes the eigen decomposition of a batch of self-adjoint matrices
///
/// (Note: Only real inputs are supported).
///
/// Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
/// tensor such that tensor[...,:,:] = u[..., :, :] * Diag(s[..., :]) * Transpose(v[...,:,:]).
///
/// Args:
/// * scope: A Scope object
/// * a: the input tensor.
/// * max_iter: maximum number of sweep update, i.e., the whole lower triangular
/// part or upper triangular part based on parameter lower. Heuristically, it has
/// been argued that approximately log(min (M, N)) sweeps are needed in practice
/// (Ref: Golub & van Loan "Matrix Computation").
/// * epsilon: the tolerance ratio.
/// * precision_config: a serialized xla::PrecisionConfig proto.
///
/// Returns:
/// * `Output` s: Singular values. The values are sorted in reverse order of magnitude, so
/// s[..., 0] is the largest value, s[..., 1] is the second largest, etc.
/// * `Output` u: Left singular vectors.
/// * `Output` v: Right singular vectors.
class XlaSvd {
 public:
  XlaSvd(const ::tensorflow::Scope& scope, ::tensorflow::Input a, int64 max_iter,
       float epsilon, StringPiece precision_config);

  Operation operation;
  ::tensorflow::Output s;
  ::tensorflow::Output u;
  ::tensorflow::Output v;
};

/// Wraps the variadic XLA Reduce operator.
///
/// Semantics are documented at
///  https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.
///
/// This version is limited to operands of the same dtype.
/// XlaVariadicReduceV2 is a version that supports heterogeneous operands.
///
/// Args:
/// * scope: A Scope object
/// * input: the input tensor(s)
/// * init_value: scalar initial value(s) for the reduction
/// * dimensions_to_reduce: dimension numbers over which to reduce
/// * reducer: a reducer function to apply
///
/// Returns:
/// * `OutputList`: The output tensor.
class XlaVariadicReduce {
 public:
  XlaVariadicReduce(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                  input, ::tensorflow::InputList init_value, const
                  gtl::ArraySlice<int>& dimensions_to_reduce, const
                  NameAttrList& reducer);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// Wraps the variadic XLA Reduce operator.
///
/// Semantics are documented at
///  https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.
///
/// This is an expanded version of XlaVariadicReduce, with support for
/// operands of different dtypes, and improved shape inference.
///
/// Args:
/// * scope: A Scope object
/// * inputs: the input tensor(s)
/// * init_values: scalar initial value(s) for the reduction
/// * dimensions_to_reduce: dimension numbers over which to reduce
/// * reducer: a reducer function to apply
///
/// Returns:
/// * `OutputList`: The outputs tensor.
class XlaVariadicReduceV2 {
 public:
  XlaVariadicReduceV2(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                    inputs, ::tensorflow::InputList init_values, const
                    gtl::ArraySlice<int>& dimensions_to_reduce, const
                    NameAttrList& reducer);
  ::tensorflow::Output operator[](size_t index) const { return outputs[index]; }


  Operation operation;
  ::tensorflow::OutputList outputs;
};

/// Wraps the XLA Sort operator, documented at
///
///  https://www.tensorflow.org/performance/xla/operation_semantics#sort
/// .
///
/// Sorts one or more tensors, with support for custom comparator, dimension, and
/// is_stable attributes.
///
/// Args:
/// * scope: A Scope object
/// * inputs: A list of `Tensor` of identical shape but possibly different types.
/// * dimension: The dimension along which to sort. Must be a compile-time constant.
/// * comparator: A comparator function to apply to 2*N scalars and returning a
/// boolean. N is the number of sort inputs. If you want to sort in ascending
/// order then the comparator should perform a less-than comparison.
/// * is_stable: Whether to use stable sort.
///
/// Returns:
/// * `OutputList`: A list of `Tensor` of same shape and types as the `input`.
class XlaVariadicSort {
 public:
  XlaVariadicSort(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                inputs, ::tensorflow::Input dimension, const NameAttrList&
                comparator, bool is_stable);
  ::tensorflow::Output operator[](size_t index) const { return outputs[index]; }


  Operation operation;
  ::tensorflow::OutputList outputs;
};

/// output = input; While (Cond(output)) { output = Body(output) }
///
/// Args:
/// * scope: A Scope object
/// * input: A list of input tensors whose types are T.
/// * cond: A function takes 'input' and returns a tensor.  If the tensor is
/// a scalar of non-boolean, the scalar is converted to a boolean
/// according to the following rule: if the scalar is a numerical
/// value, non-zero means True and zero means False; if the scalar is
/// a string, non-empty means True and empty means False. If the
/// tensor is not a scalar, non-emptiness means True and False
/// otherwise.
/// * body: A function that takes a list of tensors and returns another
/// list of tensors. Both lists have the same types as specified by T.
///
/// Returns:
/// * `OutputList`: A list of output tensors whose types are T.
class XlaWhile {
 public:
  XlaWhile(const ::tensorflow::Scope& scope, ::tensorflow::InputList input, const
         NameAttrList& cond, const NameAttrList& body);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF_XLA_CC_OPS_XLA_OPS_H_
