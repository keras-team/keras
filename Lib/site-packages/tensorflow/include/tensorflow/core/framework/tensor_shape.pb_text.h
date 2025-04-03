// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_tensor_shape_proto_H_
#define tensorflow_core_framework_tensor_shape_proto_H_

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Message-text conversion for tensorflow.TensorShapeProto.Dim
string ProtoDebugString(
    const ::tensorflow::TensorShapeProto_Dim& msg);
string ProtoShortDebugString(
    const ::tensorflow::TensorShapeProto_Dim& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::TensorShapeProto_Dim* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.TensorShapeProto
string ProtoDebugString(
    const ::tensorflow::TensorShapeProto& msg);
string ProtoShortDebugString(
    const ::tensorflow::TensorShapeProto& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::TensorShapeProto* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_framework_tensor_shape_proto_H_
