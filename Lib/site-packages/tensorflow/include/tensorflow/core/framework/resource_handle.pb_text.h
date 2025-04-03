// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_resource_handle_proto_H_
#define tensorflow_core_framework_resource_handle_proto_H_

#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Message-text conversion for tensorflow.ResourceHandleProto.DtypeAndShape
string ProtoDebugString(
    const ::tensorflow::ResourceHandleProto_DtypeAndShape& msg);
string ProtoShortDebugString(
    const ::tensorflow::ResourceHandleProto_DtypeAndShape& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::ResourceHandleProto_DtypeAndShape* msg)
        TF_MUST_USE_RESULT;

// Message-text conversion for tensorflow.ResourceHandleProto
string ProtoDebugString(
    const ::tensorflow::ResourceHandleProto& msg);
string ProtoShortDebugString(
    const ::tensorflow::ResourceHandleProto& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::ResourceHandleProto* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_framework_resource_handle_proto_H_
