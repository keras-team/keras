// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_tensor_proto_H_
#define tensorflow_core_framework_tensor_proto_H_

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Message-text conversion for tensorflow.TensorProto
string ProtoDebugString(
    const ::tensorflow::TensorProto& msg);
string ProtoShortDebugString(
    const ::tensorflow::TensorProto& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::TensorProto* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.VariantTensorDataProto
string ProtoDebugString(
    const ::tensorflow::VariantTensorDataProto& msg);
string ProtoShortDebugString(
    const ::tensorflow::VariantTensorDataProto& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::VariantTensorDataProto* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_framework_tensor_proto_H_
