// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_types_proto_H_
#define tensorflow_core_framework_types_proto_H_

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Enum text output for tensorflow.DataType
const char* EnumName_DataType(
    ::tensorflow::DataType value);

// Message-text conversion for tensorflow.SerializedDType
string ProtoDebugString(
    const ::tensorflow::SerializedDType& msg);
string ProtoShortDebugString(
    const ::tensorflow::SerializedDType& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::SerializedDType* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_framework_types_proto_H_
