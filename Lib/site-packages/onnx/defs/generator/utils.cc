/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/generator/utils.h"

#include <algorithm>
#include <cmath>

namespace ONNX_NAMESPACE {

void ConstantOpInference(InferenceContext& ctx) {
  auto* value = ctx.getAttribute("value");
  auto* sparse_value = ctx.getAttribute("sparse_value");
  auto* value_int = ctx.getAttribute("value_int");
  auto* value_ints = ctx.getAttribute("value_ints");
  auto* value_float = ctx.getAttribute("value_float");
  auto* value_floats = ctx.getAttribute("value_floats");
  auto* value_string = ctx.getAttribute("value_string");
  auto* value_strings = ctx.getAttribute("value_strings");

  std::vector<bool> non_null_attr = {
      (nullptr != value),
      (nullptr != sparse_value),
      (nullptr != value_int),
      (nullptr != value_ints),
      (nullptr != value_float),
      (nullptr != value_floats),
      (nullptr != value_string),
      (nullptr != value_strings)};

  if (std::count(non_null_attr.begin(), non_null_attr.end(), true) != 1) {
    fail_shape_inference(
        "One and only one of the attributes 'value', 'value_*' or 'sparse_value' must be specified for a Constant node.");
  }

  if (nullptr != value) {
    // OpSchema::Verify check ensures that the attribute value has_t():
    const TensorProto& tensor_proto = value->t();
    updateOutputElemType(ctx, 0, tensor_proto.data_type());
    updateOutputShape(ctx, 0, tensor_proto);
    return;
  }

  if (nullptr != value_int) {
    // OpSchema::Verify check ensures that the attribute value has_i():
    if (!value_int->has_i()) {
      fail_shape_inference("Attribute 'value_int' expect an integer.")
    }
    updateOutputElemType(ctx, 0, TensorProto::INT64);
    updateOutputShape(ctx, 0, TensorShapeProto());
    return;
  }

  if (nullptr != value_ints) {
    updateOutputElemType(ctx, 0, TensorProto::INT64);
    appendDim(getOutputShape(ctx, 0), value_ints->ints_size());
    return;
  }

  if (nullptr != value_float) {
    // OpSchema::Verify check ensures that the attribute value has_i():
    if (!value_float->has_f()) {
      fail_shape_inference("Attribute 'value_float' expect a float.");
    }
    updateOutputElemType(ctx, 0, TensorProto::FLOAT);
    updateOutputShape(ctx, 0, TensorShapeProto());
    return;
  }

  if (nullptr != value_floats) {
    updateOutputElemType(ctx, 0, TensorProto::FLOAT);
    appendDim(getOutputShape(ctx, 0), value_floats->floats_size());
    return;
  }

  if (nullptr != value_string) {
    // OpSchema::Verify check ensures that the attribute value has_i():
    if (!value_string->has_s()) {
      fail_shape_inference("Attribute 'value_string' expect a string.");
    }
    updateOutputElemType(ctx, 0, TensorProto::STRING);
    updateOutputShape(ctx, 0, TensorShapeProto());
    return;
  }

  if (nullptr != value_strings) {
    updateOutputElemType(ctx, 0, TensorProto::STRING);
    appendDim(getOutputShape(ctx, 0), value_strings->strings_size());
    return;
  }

  if (nullptr != sparse_value) {
    // OpSchema::Verify check ensures that the attribute value
    // has_sparse_tensor():
    const SparseTensorProto& sparse = sparse_value->sparse_tensor();
    // checker.cc::check_sparse_tensor checks that the sparse-value is
    // well-formed
    updateOutputElemType(ctx, 0, sparse.values().data_type());
    auto* output_shape = getOutputShape(ctx, 0);
    for (int i = 0; i < sparse.dims_size(); ++i)
      appendDim(output_shape, sparse.dims(i));
    return;
  }

  fail_shape_inference(
      "TypeAndShapeInferenceFunction implementation incomplete: "
      "this line should never be reached.");
}

} // namespace ONNX_NAMESPACE
