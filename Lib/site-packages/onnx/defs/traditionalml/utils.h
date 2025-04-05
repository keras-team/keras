#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

void AssertAttributeProtoTypeAndLength(
    const AttributeProto* attr_proto,
    int expected_length,
    TensorProto_DataType expected_type,
    bool required) {
  if (nullptr == attr_proto) {
    if (required) {
      fail_shape_inference("Unspecified required attribute.");
    }
    return;
  }
  const auto& [type, length] = getAttributeProtoElemTypeAndLength(attr_proto);
  if (type != expected_type) {
    fail_shape_inference(
        "Attribute '", attr_proto->name(), "' must have type ", TensorProto_DataType_Name(expected_type), ".");
  }
  if (length != expected_length) {
    fail_shape_inference("Attribute '", attr_proto->name(), "' must have ", expected_length, " elements.");
  }
}

} // namespace ONNX_NAMESPACE
